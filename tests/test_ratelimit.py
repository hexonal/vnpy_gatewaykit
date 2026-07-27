"""RateLimitBackoff / BackoffRegistry — the timing state machine every
gateway shares when a broker answers "you are going too fast".

Every test runs on an injected clock and an injected RNG, so the escalation
schedule is asserted exactly rather than slept through.
"""

from __future__ import annotations

import threading

import pytest

from vnpy_gatewaykit.ratelimit import (
    BackoffPolicy,
    BackoffRegistry,
    RateLimitBackoff,
)


class FakeClock:
    """Monotonic clock the test drives by hand."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def policy(**kwargs: object) -> BackoffPolicy:
    """A deterministic policy: rand()=0.5 means the ±jitter term is exactly 0."""
    defaults: dict[str, object] = {
        "base": 1.0,
        "factor": 2.0,
        "cap": 16.0,
        "jitter": 0.25,
        "report_every": 5.0,
        "rand": lambda: 0.5,
    }
    defaults.update(kwargs)
    return BackoffPolicy(**defaults)  # type: ignore[arg-type]


def test_delays_double_until_they_hit_the_cap() -> None:
    clock = FakeClock()
    gate = RateLimitBackoff(policy(clock=clock))

    armed = [gate.penalize() for _ in range(8)]

    assert armed[:5] == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert armed[5:] == [16.0, 16.0, 16.0], "cap 之上必须停止增长,否则会退避到永不查询"
    assert gate.strikes == 8


def test_jitter_spreads_the_delay_but_never_past_the_cap() -> None:
    """Two clients that hit the wall in the same second must not come back in
    the same second — that is what keeps the limit window alive."""
    clock = FakeClock()
    lo = RateLimitBackoff(policy(clock=clock, rand=lambda: 0.0))
    hi = RateLimitBackoff(policy(clock=clock, rand=lambda: 1.0))

    # jitter=0.25 → rand()=0 is -25%, rand()=1 is +25%.
    assert lo.penalize() == pytest.approx(0.75)
    assert hi.penalize() == pytest.approx(1.25)

    for _ in range(10):
        armed = hi.penalize()
        assert armed <= policy().cap * (1.0 + policy().jitter)


def test_blocked_until_the_delay_elapses_then_lets_one_through() -> None:
    clock = FakeClock()
    gate = RateLimitBackoff(policy(clock=clock))

    gate.penalize()                      # arms 1.0s
    assert gate.blocked() is True
    assert gate.remaining() == pytest.approx(1.0)

    clock.advance(0.5)
    assert gate.blocked() is True
    assert gate.remaining() == pytest.approx(0.5)

    clock.advance(0.5)
    assert gate.blocked() is False
    assert gate.remaining() == 0.0


def test_success_clears_the_penalty_and_reports_the_recovery_once() -> None:
    clock = FakeClock()
    gate = RateLimitBackoff(policy(clock=clock))

    gate.penalize()
    gate.penalize()
    assert gate.strikes == 2

    assert gate.succeed() is True, "从退避状态恢复必须可被观察到(供日志留痕)"
    assert gate.strikes == 0
    assert gate.blocked() is False
    assert gate.succeed() is False, "已经正常的连接不应重复播报恢复"


def test_report_throttle_keeps_a_60s_window_from_printing_60_lines() -> None:
    clock = FakeClock()
    gate = RateLimitBackoff(policy(clock=clock, report_every=5.0))

    gate.penalize()                       # arming counts as the first report
    assert gate.due_report() is False     # …so the next second stays quiet

    clock.advance(5.0)
    assert gate.due_report() is True
    assert gate.due_report() is False

    clock.advance(4.99)
    assert gate.due_report() is False
    clock.advance(0.01)
    assert gate.due_report() is True


def test_server_retry_after_hint_wins_when_it_is_longer() -> None:
    clock = FakeClock()
    gate = RateLimitBackoff(policy(clock=clock, jitter=0.0))

    assert gate.penalize(retry_after=30.0) == pytest.approx(30.0)
    # A hint shorter than our own escalation does not shorten the backoff:
    # the server said "at least this long", not "at most".
    assert gate.penalize(retry_after=0.1) == pytest.approx(2.0)


def test_retry_after_hint_is_clamped_so_it_can_never_park_us_forever() -> None:
    clock = FakeClock()
    gate = RateLimitBackoff(policy(clock=clock, jitter=0.0, hint_cap=120.0))

    assert gate.penalize(retry_after=86_400.0) == pytest.approx(120.0)


def test_policy_rejects_settings_that_would_disable_the_backoff() -> None:
    for bad in (
        {"base": 0.0},
        {"base": -1.0},
        {"factor": 0.9},
        {"cap": 0.5},          # cap below base
        {"jitter": 1.0},
        {"jitter": -0.1},
        {"report_every": -1.0},
        {"hint_cap": 1.0},     # hint_cap below cap
    ):
        with pytest.raises(ValueError):
            policy(**bad)


def test_registry_keeps_one_gate_per_endpoint() -> None:
    clock = FakeClock()
    registry = BackoffRegistry(policy(clock=clock))

    registry.get("/today-entrust").penalize()

    assert registry.get("/today-entrust").blocked() is True
    assert registry.get("/stock-record").blocked() is False, (
        "一个接口被限流不应让无关接口停查"
    )
    assert registry.get("/today-entrust") is registry.get("/today-entrust")


def test_registry_is_thread_safe_under_concurrent_first_touch() -> None:
    registry = BackoffRegistry(policy())
    seen: list[RateLimitBackoff] = []
    barrier = threading.Barrier(8)

    def touch() -> None:
        barrier.wait()
        seen.append(registry.get("/same"))

    threads = [threading.Thread(target=touch) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({id(gate) for gate in seen}) == 1


def test_penalize_is_serialised_so_strikes_cannot_be_lost() -> None:
    gate = RateLimitBackoff(policy(clock=FakeClock()))
    barrier = threading.Barrier(16)

    def hit() -> None:
        barrier.wait()
        gate.penalize()

    threads = [threading.Thread(target=hit) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert gate.strikes == 16
