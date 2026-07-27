"""Server-driven rate-limit backoff — what to do after the broker says
"you are going too fast".

Why this lives in gatewaykit rather than in one broker package
--------------------------------------------------------------
Every broker this project talks to can refuse a request for going too fast,
and each announces it in its own dialect: uSMART answers `code=429` inside
its JSON envelope (observed live on SG UAT, 2026-07-26: "The system is busy,
please try again later!"), an HTTP gateway may answer with status 429 and a
`Retry-After` header, futu's OpenD answers `RET_ERROR` with a frequency
message. The *detection* is therefore broker-specific and stays in the broker
package. What follows detection — how long to wait, how fast to escalate, how
to keep two clients from coming back in the same second, when it is safe to
try again, and how to stay visible while waiting — has nothing broker-specific
in it. That policy is this module, and it is the part that is easy to get
subtly wrong (unbounded growth, no jitter, no ceiling, silent suppression),
so it is worth having in one tested place instead of once per gateway.

The deliberate design choices
-----------------------------
* **Not a retry loop.** This never sleeps and never calls anything. It is a
  clock-driven gate: the caller asks "may I send?" and reports back "I was
  refused" / "that worked". Sleeping inside a request would block whichever
  thread the caller happens to poll on, and a poller already has its own
  cadence — it just needs to be told to skip a few rounds. This is also the
  distinction from a *transport* retry: a TLS hiccup is worth retrying in
  half a second on the spot, a 429 is worth not retrying at all for a while.
* **Bounded.** The delay doubles up to `cap` and then stops growing, so a
  server that stays busy for an hour cannot push the next attempt past the
  cap. "Back off forever" and "stop querying" are the same thing to a caller
  that needs the answer — and for an end-of-day reconciliation, never
  querying again is the failure mode this whole module exists to prevent.
* **Jittered.** Two gateways (or two markets on one gateway) that hit the wall
  in the same second would otherwise return in the same second and re-arm the
  server's window together. The delay is spread by ±`jitter`.
* **Loud on a schedule.** `due_report` gives the caller a throttle so a
  60-second wait produces a handful of log lines instead of one per poll
  round — visible without being the wall of repeated text that made the
  original incident hard to read.
* **Honours a server hint.** `Retry-After` wins when it is longer than our own
  escalation, clamped by `hint_cap` so a hostile or mistaken header cannot
  park a gateway indefinitely.

The counterpart to this is a *client-side courtesy limiter* (e.g. uSMART's
documented 120/min and 20/min buckets), which paces requests before they are
sent. The two are complementary: the limiter is what we promise, this is what
we do when the server tells us we broke the promise anyway.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_BASE_SEC = 1.0
DEFAULT_FACTOR = 2.0
DEFAULT_CAP_SEC = 60.0
DEFAULT_JITTER = 0.25
DEFAULT_REPORT_EVERY_SEC = 10.0
DEFAULT_HINT_CAP_SEC = 300.0


@dataclass(frozen=True)
class BackoffPolicy:
    """Tunables + injected clock/RNG. Frozen so one policy can be shared by
    every gate in a registry without any of them being able to mutate it.

    base:         first wait, in seconds.
    factor:       multiplier per consecutive refusal.
    cap:          ceiling on the escalation *before* jitter is applied, so an
                  armed delay can reach `cap * (1 + jitter)`. Jitter has to sit
                  outside the cap or every gate parked at the ceiling would
                  come back in the same instant, which is the pile-up jitter
                  exists to break up. `hint_cap` is the hard bound.
    jitter:       fraction of the delay spread as ±jitter (0 disables).
    report_every: minimum seconds between `due_report` returning True.
    hint_cap:     absolute ceiling on any armed delay, jitter and a server
                  `Retry-After` hint included.
    clock:        monotonic seconds source (injected for tests).
    rand:         uniform [0, 1) source (injected for tests).
    """

    base: float = DEFAULT_BASE_SEC
    factor: float = DEFAULT_FACTOR
    cap: float = DEFAULT_CAP_SEC
    jitter: float = DEFAULT_JITTER
    report_every: float = DEFAULT_REPORT_EVERY_SEC
    hint_cap: float = DEFAULT_HINT_CAP_SEC
    clock: Callable[[], float] = time.monotonic
    rand: Callable[[], float] = random.random

    def __post_init__(self) -> None:
        # A misconfigured policy would silently disable the very protection it
        # is here to provide, so every bound is checked at construction.
        if not self.base > 0.0:
            raise ValueError(f"base 必须为正数(秒): {self.base}")
        if self.factor < 1.0:
            raise ValueError(f"factor 必须 >=1,否则退避会越来越短: {self.factor}")
        if self.cap < self.base:
            raise ValueError(f"cap({self.cap}) 不得小于 base({self.base})")
        if not 0.0 <= self.jitter < 1.0:
            raise ValueError(f"jitter 必须在 [0,1) 内: {self.jitter}")
        if self.report_every < 0.0:
            raise ValueError(f"report_every 不得为负: {self.report_every}")
        if self.hint_cap < self.cap:
            raise ValueError(f"hint_cap({self.hint_cap}) 不得小于 cap({self.cap})")


DEFAULT_POLICY = BackoffPolicy()


class RateLimitBackoff:
    """One endpoint's (or one connection's) backoff state.

    Usage from a client's request path::

        if gate.blocked():
            raise SomethingSaying(gate.remaining(), report=gate.due_report())
        response = send()
        if refused_for_rate(response):
            delay = gate.penalize(retry_after=hint)
            raise SomethingSaying(delay)
        gate.succeed()

    Thread-safe: a gateway polls several markets from one worker thread while
    the GUI thread can place an order down the same client.
    """

    def __init__(self, policy: BackoffPolicy | None = None) -> None:
        self._policy = policy if policy is not None else DEFAULT_POLICY
        self._lock = threading.Lock()
        self._strikes = 0
        self._step = 0.0
        self._until = -math.inf
        self._last_report = -math.inf

    @property
    def policy(self) -> BackoffPolicy:
        return self._policy

    @property
    def strikes(self) -> int:
        """Consecutive refusals since the last success."""
        with self._lock:
            return self._strikes

    def penalize(self, retry_after: float | None = None) -> float:
        """Record a refusal and arm the next wait. Returns the armed delay.

        `retry_after` is the server's own instruction in seconds, when it sent
        one. It raises the wait but never lowers it: the server said "at least
        this long", and our escalation encodes what we already know about how
        many times in a row it has refused us.
        """
        policy = self._policy
        with self._lock:
            self._strikes += 1
            # Stepwise rather than base * factor**strikes: an exponent that
            # keeps climbing for a server that stays busy all afternoon
            # overflows to inf (and then raises) long before anyone notices.
            self._step = (
                policy.base if self._strikes == 1
                else min(self._step * policy.factor, policy.cap)
            )
            step = self._step
            if retry_after is not None and retry_after > 0.0:
                step = min(max(step, retry_after), policy.hint_cap)
            delay = step * (1.0 + policy.jitter * (2.0 * policy.rand() - 1.0))
            delay = min(max(delay, 0.0), policy.hint_cap)
            now = policy.clock()
            self._until = now + delay
            # Arming is itself a report; the caller logs the line that carries
            # this delay, so the throttle should not immediately allow another.
            self._last_report = now
            return delay

    def succeed(self) -> bool:
        """Clear the penalty. Returns True only on the transition out of a
        backoff, so the caller can announce the recovery exactly once."""
        with self._lock:
            recovered = self._strikes > 0
            self._strikes = 0
            self._step = 0.0
            self._until = -math.inf
            self._last_report = -math.inf
            return recovered

    def remaining(self) -> float:
        """Seconds until the next attempt is allowed; 0.0 when it is allowed."""
        with self._lock:
            return max(0.0, self._until - self._policy.clock())

    def blocked(self) -> bool:
        return self.remaining() > 0.0

    def due_report(self) -> bool:
        """True at most once per `report_every` seconds. Consumes the slot —
        call it exactly once per line you are about to emit."""
        with self._lock:
            now = self._policy.clock()
            if now - self._last_report >= self._policy.report_every:
                self._last_report = now
                return True
            return False


class BackoffRegistry:
    """Lazily-created gates keyed by whatever the caller scopes a limit to —
    an endpoint path, a (path, market) pair, a whole connection.

    Per-endpoint is usually right: brokers publish per-interface quotas
    (uSMART documents 120/min and 20/min buckets), and one throttled endpoint
    should not stop an unrelated one from being queried at all.
    """

    def __init__(self, policy: BackoffPolicy | None = None) -> None:
        self._policy = policy if policy is not None else DEFAULT_POLICY
        self._gates: dict[str, RateLimitBackoff] = {}
        self._lock = threading.Lock()

    @property
    def policy(self) -> BackoffPolicy:
        return self._policy

    def get(self, key: str) -> RateLimitBackoff:
        with self._lock:
            gate = self._gates.get(key)
            if gate is None:
                gate = RateLimitBackoff(self._policy)
                self._gates[key] = gate
            return gate

    def blocked_keys(self) -> list[str]:
        """Keys currently in a backoff — for a status readout."""
        with self._lock:
            gates = list(self._gates.items())
        return [key for key, gate in gates if gate.blocked()]

    def clear(self) -> None:
        """Forget every penalty (e.g. after a reconnect with a fresh token)."""
        with self._lock:
            self._gates.clear()
