"""Tick hygiene for HK / US cash equities.

Port of the community "TickFilter" idea (vnpy forum thread 30601) to markets
that have no CTP-style `InstrumentStatus`节次状态机. The original filter keyed
everything off the exchange's own status field: keep ticks only while the
instrument is in continuous-auction or call-auction state, drop ticks already
folded into a bar, and disambiguate repeated same-second timestamps. Two of
those three ideas survive the port unchanged; the first one has to be rebuilt,
because "what phase is this market in" is answered differently here:

  * futu DOES ship a per-quote status field — `sec_status` (futu.SecurityStatus)
    plus `dark_status` and the `suspension` bool, all three written by
    futu/quote/quote_query.py:parse_pb_BasicQot (lines 1241-1246), which is the
    parser behind BOTH get_stock_quote() and the QUOTE push. So HK/US quotes
    carry SUSPENDED / DARK_TRADING / RECOVERABLE_CIRCUIT_BREAKER / CALLED /
    DELISTED / ... That is the real analogue of InstrumentStatus, and neither
    vnpy_futu nor vnpy_usmart reads it today.
  * uSMART's realtime payload has no status field at all (usmart_mapping.
    parse_realtime_tick reads latestTime/latestPrice/open/high/low/preClose/
    volume/turnOver/upLimit/downLimit/bid*/ask* and nothing else). For uSMART
    the only available phase signal is the clock.

So this module takes phase from two independent sources — broker status when
the feed supplies it, the session calendar (vnpy_gatewaykit.sessions) when it
does not — and keeps them separate in the output so a caller can see which one
fired.

WHY IT LIVES IN gatewaykit
--------------------------
Not in vnpy core: this fork tracks upstream, and a patch inside
vnpy/trader/engine.py is a permanent merge conflict. Not per-gateway: every
rule except the broker-status one is identical for futu and uSMART, and two
copies is how two gateways end up disagreeing. And by the time a tick reaches
the engine its provenance is gone — a TickData no longer knows it came from a
futu QUOTE push (the channel with the frozen-snapshot pathology below) rather
than a uSMART `rt` push. gatewaykit is the one layer that is shared, is outside
the vnpy fork, and still sits at the boundary where provenance exists. It also
already owns sessions.py/market_clock.py, which this module consumes.

THE 误杀 PROBLEM
----------------
A wrongly dropped tick can be the whole bar: on a 2x ETP an 8% single-tick move
is ordinary, and on a thin HK line one lot at the top IS the high. So:

  * The default mode is OBSERVE — nothing is ever dropped, every rule is only
    counted. Enforcement is opt-in *per rule* (`enforce={...}`), so a rule is
    turned on only after its counters have been read on real data.
  * Reference state (last kept timestamp, fingerprint, per-minute extremes)
    advances ONLY on ticks the rules would keep, in both modes. That makes the
    OBSERVE counters exactly equal to what ENFORCE would have done, instead of
    an approximation — see test_observe_and_enforce_agree.
  * Every would-be drop is audited for what it would have cost:
    `suppressed_extreme` counts drops that carried a price outside the extremes
    kept so far in that minute, `suppressed_volume` sums the positive
    cumulative-volume delta a drop would have hidden, and `blank_minutes`
    counts minute buckets where something was dropped and nothing was kept.
    A rule is safe to enforce when, over a real session, all three read zero
    for it (blank_minutes in a CLOSED bucket is expected — see its docstring).

The audit is relative to what was kept, so it cannot judge the very first tick
of a symbol or of a minute; `blank_minutes` is what covers that blind spot.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite

from vnpy.trader.constant import Exchange
from vnpy.trader.object import TickData

from .sessions import (
    ALL_KINDS,
    DEFAULT_CALENDAR,
    SessionKind,
    TradingCalendar,
    active_session,
    open_seconds_between,
)


class Phase(Enum):
    """What kind of trading produced this tick.

    A superset of SessionKind: the three clock-derived kinds, plus the states
    only a broker status field can report (HALTED / DARK) and the two "the
    calendar cannot answer" outcomes (CLOSED / UNKNOWN).
    """

    AUCTION = "auction"      # HK 09:00-09:30 POS, HK 16:00-16:10 CAS
    REGULAR = "regular"      # continuous session
    EXTENDED = "extended"    # US pre-market 04:00-09:30 / after-hours 16:00-20:00 ET
    CLOSED = "closed"        # a trading day, but no window open (HK lunch, overnight)
    DARK = "dark"            # HK 暗盘 / grey market — broker-run, off-exchange
    HALTED = "halted"        # suspension, circuit breaker, called, delisted
    UNKNOWN = "unknown"      # exchange has no session map (SSE/SZSE here)


class Rule(Enum):
    """One filter rule. Order of declaration is order of evaluation."""

    NAIVE_DATETIME = "naive_datetime"
    BAD_PRICE = "bad_price"
    HALTED = "halted"
    DARK_MARKET = "dark_market"
    PHASE_NOT_ALLOWED = "phase_not_allowed"
    STALE = "stale"
    REGRESSION = "regression"
    DUPLICATE = "duplicate"


class FilterMode(Enum):
    """OBSERVE = dry-run: count everything, drop nothing. The default."""

    OBSERVE = "observe"
    ENFORCE = "enforce"


# --- futu.SecurityStatus string values -------------------------------------
# Mirrored as plain strings rather than imported: gatewaykit depends on vnpy
# only, and must not pull in a broker SDK. Values copied verbatim from
# futu/common/constant.py:2788 (futu-api 10.09.6908) — including futu's own
# spellings 'BEFORE_DARK_TRADE_OPEING' and 'CHANGED_CODE_TRAD_END', which are
# typos in the SDK and must be matched exactly.
#
# HALT: the instrument is not matching orders on the exchange, and any price on
# the tick is a leftover, not a trade.
HALT_STATUSES: frozenset[str] = frozenset(
    {
        "SUSPENDED",
        "CALLED",                          # HK CBBC knocked out — quotes stop meaning anything
        "DELISTED",
        "EXPIRED",
        "EXPIRED_LAST_TRADING_DATE",
        "RECOVERABLE_CIRCUIT_BREAKER",
        "UNRECOVERABLE_CIRCUIT_BREAKER",
        "CHANGE_TO_TEMPORARY_CODE",
        "TEMPORARY_CODE_TRADE_END",
        "CHANGED_PLATE_TRADE_END",
        "CHANGED_CODE_TRAD_END",
    }
)

# DARK: HK 暗盘 — the grey market brokers run between listing eve 16:15-18:30
# and the IPO's first exchange session. Real trades, real fills, but matched
# inside the broker, NOT on SEHK: they never appear on the exchange tape and
# the next morning's opening auction ignores them entirely. Mixing them into
# the same bar series as SEHK prints produces a candle no exchange ever saw.
DARK_STATUSES: frozenset[str] = frozenset(
    {"BEFORE_DARK_TRADE_OPEING", "DARK_TRADING", "DARK_TRAD_END"}
)

# futu.DarkStatus (constant.py:1147) is the separate, narrower field.
DARK_STATUS_ACTIVE: frozenset[str] = frozenset({"TRADING", "END"})

# NOT-YET-TRADING: reported before an instrument's first session. Deliberately
# NOT in HALT_STATUSES. It is unverified whether futu reports TO_BE_OPEN during
# HK's 09:00-09:30 pre-opening session; if it does, halting on it would throw
# away the opening auction — which on HK.00700 2026-07-24 was 545,400 shares,
# i.e. the single largest print of the morning. Counted, never dropped.
PRE_TRADE_STATUSES: frozenset[str] = frozenset(
    {"LISTING", "PURCHASING", "SUBSCRIBING", "TO_BE_OPEN"}
)

_KIND_TO_PHASE: dict[SessionKind, Phase] = {
    SessionKind.AUCTION: Phase.AUCTION,
    SessionKind.REGULAR: Phase.REGULAR,
    SessionKind.EXTENDED: Phase.EXTENDED,
}

# Rules whose dropped ticks are provably information-free: a non-positive or
# non-finite price cannot be a trade, a naive timestamp cannot be placed on any
# clock, and a byte-identical repeat (timestamp + trade fields + full book)
# carries nothing the previous tick did not already carry. Safe first step.
SAFE_RULES: frozenset[Rule] = frozenset(
    {Rule.NAIVE_DATETIME, Rule.BAD_PRICE, Rule.DUPLICATE}
)

# Everything except PHASE_NOT_ALLOWED, which needs an explicit allowed_phases
# policy and is therefore never implicitly on.
STRICT_RULES: frozenset[Rule] = SAFE_RULES | {
    Rule.HALTED,
    Rule.DARK_MARKET,
    Rule.STALE,
    Rule.REGRESSION,
}

_BOOK_FIELDS: tuple[str, ...] = tuple(
    f"{side}_{kind}_{i}"
    for side in ("bid", "ask")
    for kind in ("price", "volume")
    for i in range(1, 6)
)

# The full content of a tick, for duplicate detection. The book is included on
# purpose: futu and uSMART both deliver the order book on a channel SEPARATE
# from the quote, so a book update arrives as a tick whose trade fields are
# byte-identical to the previous one. Fingerprinting on (datetime, price,
# volume) alone would drop every one of those — and vnpy's client-side stop
# prices from bid_price_5/ask_price_5 (vnpy_ctastrategy.check_stop_order), so
# starving it of book updates would price stops off a stale ladder. This is the
# single most dangerous 误杀 in the whole module; test_book_only_update_survives
# guards it.
_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "last_price",
    "last_volume",
    "volume",
    "turnover",
    "open_price",
    "high_price",
    "low_price",
    "pre_close",
    "open_interest",
    *_BOOK_FIELDS,
)


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the rules decided about one tick.

    `keep` is the rule outcome, not necessarily what happened to the tick: in
    OBSERVE mode a tick with keep=False is still forwarded. `delivered` is what
    actually happened.
    """

    keep: bool
    delivered: bool
    phase: Phase
    rule: Rule | None = None
    detail: str = ""
    lost_extreme: bool = False
    lost_volume: float = 0.0


@dataclass
class FilterStats:
    """Counters. Read these before ever turning a rule on.

    would_drop        — per rule, ticks the rule matched (both modes).
    dropped           — per rule, ticks actually withheld (ENFORCE only).
    suppressed_extreme— per rule, matched ticks whose last_price fell outside
                        the high/low kept so far in that tick's own minute.
                        NON-ZERO MEANS THE RULE EATS BAR EXTREMES. Do not
                        enforce it.
    suppressed_volume — per rule, summed positive cumulative-volume delta the
                        matched ticks would have hidden. Non-zero means real
                        trades happened inside the dropped ticks.
    blank_minutes     — per phase, minute buckets in which at least one tick
                        matched a rule and none were kept. In CLOSED/EXTENDED
                        this is normal (that is exactly the frozen out-of-hours
                        snapshot being suppressed). In REGULAR or AUCTION it
                        means a bar lost every one of its ticks — an alarm.
                        Only completed buckets are counted; the in-progress one
                        is not, so this figure lags by one minute per symbol.
    same_second       — ticks sharing a timestamp with the previous kept tick
                        but differing in content. Never dropped. This is the
                        port of thread 30601's 同秒重复 handling: futu stamps HK
                        quotes to the second ('16:07:57', no milliseconds —
                        futu_mapping._FUTU_DATETIME_FORMATS), so collisions are
                        guaranteed on any active HK line. They are real, distinct
                        trades; anything storing ticks keyed on
                        (symbol, datetime) will silently lose them.
    """

    seen: int = 0
    kept: int = 0
    would_drop: dict[Rule, int] = field(default_factory=dict)
    dropped: dict[Rule, int] = field(default_factory=dict)
    phase_counts: dict[Phase, int] = field(default_factory=dict)
    suppressed_extreme: dict[Rule, int] = field(default_factory=dict)
    suppressed_volume: dict[Rule, float] = field(default_factory=dict)
    blank_minutes: dict[Phase, int] = field(default_factory=dict)
    same_second: int = 0
    pre_trade_status: int = 0
    examples: dict[Rule, str] = field(default_factory=dict)

    def report(self) -> str:
        """Human-readable dry-run summary."""
        lines: list[str] = [
            f"ticks seen={self.seen} kept={self.kept} same_second={self.same_second}"
        ]
        if self.pre_trade_status:
            lines.append(f"  (pre-trade status seen {self.pre_trade_status}x, never dropped)")

        phases = ", ".join(
            f"{p.value}={n}" for p, n in sorted(
                self.phase_counts.items(), key=lambda kv: kv[0].value
            )
        )
        lines.append(f"phase: {phases or 'none'}")

        if not self.would_drop:
            lines.append("no rule matched any tick")
        else:
            lines.append("rule                 would_drop  dropped  extremes  volume_lost")
            for rule in Rule:
                n = self.would_drop.get(rule, 0)
                if not n:
                    continue
                lines.append(
                    f"  {rule.value:<18} {n:>10} {self.dropped.get(rule, 0):>8}"
                    f" {self.suppressed_extreme.get(rule, 0):>9}"
                    f" {self.suppressed_volume.get(rule, 0.0):>12.0f}"
                )
                example = self.examples.get(rule)
                if example:
                    lines.append(f"      first: {example}")

        for phase, n in sorted(self.blank_minutes.items(), key=lambda kv: kv[0].value):
            marker = "ALARM" if phase in (Phase.REGULAR, Phase.AUCTION) else "expected"
            lines.append(f"blank minutes in {phase.value}: {n} ({marker})")

        unsafe = sorted(
            rule.value
            for rule in Rule
            if self.suppressed_extreme.get(rule, 0) or self.suppressed_volume.get(rule, 0.0)
        )
        lines.append(
            "NOT safe to enforce: " + ", ".join(unsafe)
            if unsafe
            else "every matched rule dropped only information-free ticks"
        )
        return "\n".join(lines)


@dataclass
class _SymbolState:
    """Per-symbol reference point. Advances only on ticks the rules keep."""

    last_dt: datetime | None = None
    fingerprint: tuple[object, ...] | None = None
    last_total_volume: float | None = None
    bucket: tuple[object, ...] | None = None
    bucket_phase: Phase = Phase.UNKNOWN
    bucket_kept: int = 0
    bucket_dropped: int = 0
    bucket_high: float = 0.0
    bucket_low: float = 0.0


def _utc_now() -> datetime:
    # timezone.utc, not datetime.UTC: this package declares requires-python
    # >=3.10 and the UTC alias only exists from 3.11.
    return datetime.now(timezone.utc)  # noqa: UP017


class TickFilter:
    """Stateful, per-symbol tick hygiene. Not thread-safe by itself.

    Both gateways push from a single feed thread per connection
    (futu's handler callbacks, uSMART's ws reader), so one filter instance per
    gateway is safe. Sharing one instance across gateways is not.

    Cost is ~30 attribute reads plus one `active_session` call per tick, sized
    for an equities watchlist (a few hundred ticks/second across all symbols),
    not for a full-market L2 firehose.
    """

    def __init__(
        self,
        *,
        mode: FilterMode = FilterMode.OBSERVE,
        enforce: Iterable[Rule] = (),
        allowed_phases: Iterable[Phase] | None = None,
        max_stale_seconds: float = 300.0,
        calendar: TradingCalendar = DEFAULT_CALENDAR,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """
        mode             OBSERVE (default) forwards every tick and only counts.
        enforce          Which rules may actually drop, when mode is ENFORCE.
                         Empty means ENFORCE behaves like OBSERVE — enforcement
                         is opt-in per rule so it can be rolled out one
                         validated rule at a time.
        allowed_phases   None (default) disables Rule.PHASE_NOT_ALLOWED
                         entirely. Pass e.g. {Phase.REGULAR} to build an
                         RTH-only tape.
        max_stale_seconds
                         Staleness budget measured in MARKET-OPEN seconds (via
                         sessions.open_seconds_between), not wall clock, so an
                         overnight or a weekend does not by itself make the
                         first tick of the next session look stale.
        clock            Injected for tests; must return an aware datetime.
        """
        self.mode = mode
        self.enforce = frozenset(enforce)
        self.allowed_phases = None if allowed_phases is None else frozenset(allowed_phases)
        self.max_stale_seconds = max_stale_seconds
        self.calendar = calendar
        self.clock = clock
        self.stats = FilterStats()
        self._states: dict[str, _SymbolState] = {}

    # -- public API ---------------------------------------------------------

    def check(
        self,
        tick: TickData,
        *,
        sec_status: str | None = None,
        dark_status: str | None = None,
        suspended: bool | None = None,
    ) -> Verdict:
        """Classify one tick and update state. Returns the verdict.

        The status arguments are the broker's, passed straight through from the
        feed row: futu's `sec_status` / `dark_status` / `suspension` columns
        (parse_pb_BasicQot). uSMART has no equivalent and passes none, so its
        phase comes from the clock alone.
        """
        self.stats.seen += 1
        state = self._states.setdefault(tick.vt_symbol, _SymbolState())

        rule, phase, detail = self._classify(tick, sec_status, dark_status, suspended)

        if sec_status in PRE_TRADE_STATUSES:
            self.stats.pre_trade_status += 1

        self._roll_bucket(state, tick, phase)
        self.stats.phase_counts[phase] = self.stats.phase_counts.get(phase, 0) + 1

        if rule is None:
            self._accept(state, tick)
            self.stats.kept += 1
            return Verdict(keep=True, delivered=True, phase=phase)

        lost_extreme, lost_volume = self._audit(state, tick)
        self._record_drop(rule, tick, detail, lost_extreme, lost_volume)
        state.bucket_dropped += 1

        # State is NOT advanced here, in either mode: that is what makes the
        # OBSERVE counters identical to ENFORCE's, rather than a drifting
        # approximation of them.
        delivered = not (self.mode is FilterMode.ENFORCE and rule in self.enforce)
        if not delivered:
            self.stats.dropped[rule] = self.stats.dropped.get(rule, 0) + 1

        return Verdict(
            keep=False,
            delivered=delivered,
            phase=phase,
            rule=rule,
            detail=detail,
            lost_extreme=lost_extreme,
            lost_volume=lost_volume,
        )

    def reset_symbol(self, vt_symbol: str) -> None:
        """Forget one symbol's reference point.

        Call on resubscribe/reconnect: both feeds replay a snapshot on
        subscribe, and that replay is legitimately a repeat of the last tick —
        without a reset it would be attributed to DUPLICATE or REGRESSION and
        pollute the counters the enforcement decision is based on.
        """
        self._states.pop(vt_symbol, None)

    def reset_all(self) -> None:
        self._states.clear()

    def reset_stats(self) -> None:
        """Zero the counters, keep the per-symbol reference points."""
        self.stats = FilterStats()

    # -- rules --------------------------------------------------------------

    def _classify(
        self,
        tick: TickData,
        sec_status: str | None,
        dark_status: str | None,
        suspended: bool | None,
    ) -> tuple[Rule | None, Phase, str]:
        state = self._states[tick.vt_symbol]

        # 1. A naive datetime has no instant. vnpy's storage layer reads it as
        #    machine-local, so on this box (US Pacific) an HK tick would land
        #    ~15h off. Both mappings localize at the boundary, so a naive tick
        #    means a gateway regression — surfaced, never guessed at.
        if tick.datetime.tzinfo is None:
            return Rule.NAIVE_DATETIME, Phase.UNKNOWN, f"naive dt {tick.datetime}"

        phase = self._phase_of(tick, sec_status, dark_status, suspended)

        # 2. A non-positive or non-finite last_price is not a trade. futu and
        #    uSMART both default missing numerics to 0 (futu_mapping's
        #    `or 0`, usmart_mapping's `data.get(..., 0) or 0`), so a partial
        #    payload arrives as price 0. vnpy's BarGenerator already ignores
        #    these, but a tick recorder and a strategy's on_tick do not.
        if not isfinite(tick.last_price) or tick.last_price <= 0:
            return Rule.BAD_PRICE, phase, f"last_price={tick.last_price!r}"

        # 3. Broker says the instrument is not matching on the exchange.
        if phase is Phase.HALTED:
            return Rule.HALTED, phase, f"sec_status={sec_status!r} suspension={suspended!r}"

        # 4. HK 暗盘: real fills, but broker-internal and off the SEHK tape.
        if phase is Phase.DARK:
            return Rule.DARK_MARKET, phase, f"dark_status={dark_status!r} sec_status={sec_status!r}"

        # 5. Caller-declared phase policy (e.g. RTH-only tape).
        if self.allowed_phases is not None and phase not in self.allowed_phases:
            return Rule.PHASE_NOT_ALLOWED, phase, f"phase={phase.value}"

        # 6. Stale snapshot. The concrete case this exists for: futu's QUOTE
        #    channel is RTH-only. Outside 09:30-16:00 ET it keeps re-pushing the
        #    regular-session close — data_time frozen at '16:00:00.324',
        #    last_price frozen at the RTH close, volume frozen at the RTH total.
        #    (Structural proof from the feed itself: after_change_val is defined
        #    as after_price - last_price, so last_price cannot track after-hours
        #    prints; and RTH low 321.62 > pre_low 320.96 on AAPL 2026-07-24
        #    shows the RTH fields exclude extended-hours extremes.) Pre-market
        #    ticks are therefore a 5.5-hour replay of a tick already delivered
        #    the previous afternoon — no new information by construction.
        age = self._open_age(tick)
        if age > self.max_stale_seconds:
            return Rule.STALE, phase, f"{age:.0f}s of open time behind clock"

        # 7. Backwards timestamp. BarGenerator buckets on (hour, minute)
        #    (vnpy/trader/utility.py:214), so a tick that steps back into a
        #    previous minute closes the live bar early and opens a duplicate
        #    one — bars land out of order and the real bar is truncated.
        if state.last_dt is not None and tick.datetime < state.last_dt:
            return Rule.REGRESSION, phase, f"{tick.datetime} < {state.last_dt}"

        # 8. Byte-identical repeat: same timestamp, same trade fields, same
        #    full book. This is thread 30601's "已参与K线合成的重复tick".
        fingerprint = self._fingerprint(tick)
        if state.fingerprint is not None and fingerprint == state.fingerprint:
            return Rule.DUPLICATE, phase, f"repeat of {tick.datetime}"

        if state.last_dt is not None and tick.datetime == state.last_dt:
            # Same second, different content — a genuinely distinct trade that
            # futu's second-resolution HK timestamps cannot separate. Kept.
            self.stats.same_second += 1

        return None, phase, ""

    def _phase_of(
        self,
        tick: TickData,
        sec_status: str | None,
        dark_status: str | None,
        suspended: bool | None,
    ) -> Phase:
        if suspended:
            return Phase.HALTED
        if sec_status:
            if sec_status in HALT_STATUSES:
                return Phase.HALTED
            if sec_status in DARK_STATUSES:
                return Phase.DARK
        if dark_status and dark_status in DARK_STATUS_ACTIVE:
            return Phase.DARK
        return self._clock_phase(tick.datetime, tick.exchange)

    def _clock_phase(self, moment: datetime, exchange: Exchange) -> Phase:
        try:
            session = active_session(
                exchange, moment, kinds=ALL_KINDS, calendar=self.calendar
            )
        except KeyError:
            # sessions.py maps SEHK and SMART only. futu also serves SH/SZ
            # contracts; this fork does not trade them, and guessing a phase
            # for an unmapped market would be worse than admitting ignorance.
            # UNKNOWN never triggers a drop.
            return Phase.UNKNOWN
        if session is None:
            return Phase.CLOSED
        return _KIND_TO_PHASE[session.kind]

    def _open_age(self, tick: TickData) -> float:
        """Market-open seconds between the tick's timestamp and now.

        Wall clock is the wrong unit: an HK tick stamped 16:00 read at 09:30
        the next morning is 17.5 hours old by wall clock but only minutes old
        in trading time. Returns 0.0 for a future-dated tick (clock skew
        between the broker and this machine must not read as staleness) and
        +inf when the span is too large for sessions.open_seconds_between,
        which is a bogus timestamp rather than a stale one.
        """
        now = self.clock()
        if tick.datetime >= now:
            return 0.0
        try:
            return open_seconds_between(
                tick.exchange, tick.datetime, now, kinds=ALL_KINDS, calendar=self.calendar
            )
        except KeyError:
            return 0.0  # unmapped market: no session model, so no staleness claim
        except ValueError:
            return float("inf")  # beyond max_span_days — a garbage timestamp

    @staticmethod
    def _fingerprint(tick: TickData) -> tuple[object, ...]:
        return (tick.datetime, *(getattr(tick, name) for name in _FINGERPRINT_FIELDS))

    # -- bookkeeping --------------------------------------------------------

    def _roll_bucket(self, state: _SymbolState, tick: TickData, phase: Phase) -> None:
        """Advance the per-minute bucket, banking a blank-minute if one closed.

        Bucketing follows BarGenerator: the tick's OWN (date, hour, minute),
        not the wall clock. A stale tick therefore lands in its own stale
        bucket rather than corrupting the live one's extremes.
        """
        if tick.datetime.tzinfo is None:
            bucket: tuple[object, ...] = ("naive",)
        else:
            bucket = (tick.datetime.date(), tick.datetime.hour, tick.datetime.minute)
        if state.bucket == bucket:
            return
        if state.bucket is not None and state.bucket_kept == 0 and state.bucket_dropped > 0:
            key = state.bucket_phase
            self.stats.blank_minutes[key] = self.stats.blank_minutes.get(key, 0) + 1
        state.bucket = bucket
        state.bucket_phase = phase
        state.bucket_kept = 0
        state.bucket_dropped = 0
        state.bucket_high = 0.0
        state.bucket_low = 0.0

    def _accept(self, state: _SymbolState, tick: TickData) -> None:
        state.last_dt = tick.datetime
        state.fingerprint = self._fingerprint(tick)
        state.last_total_volume = tick.volume
        if state.bucket_kept == 0:
            state.bucket_high = tick.last_price
            state.bucket_low = tick.last_price
        else:
            state.bucket_high = max(state.bucket_high, tick.last_price)
            state.bucket_low = min(state.bucket_low, tick.last_price)
        state.bucket_kept += 1

    def _audit(self, state: _SymbolState, tick: TickData) -> tuple[bool, float]:
        """What dropping this tick would cost, measured against what was kept."""
        lost_extreme = False
        if (
            state.bucket_kept > 0
            and isfinite(tick.last_price)
            and tick.last_price > 0
            and (tick.last_price > state.bucket_high or tick.last_price < state.bucket_low)
        ):
            lost_extreme = True

        lost_volume = 0.0
        if state.last_total_volume is not None and isfinite(tick.volume):
            delta = tick.volume - state.last_total_volume
            if delta > 0:
                lost_volume = delta
        return lost_extreme, lost_volume

    def _record_drop(
        self, rule: Rule, tick: TickData, detail: str, lost_extreme: bool, lost_volume: float
    ) -> None:
        self.stats.would_drop[rule] = self.stats.would_drop.get(rule, 0) + 1
        if lost_extreme:
            self.stats.suppressed_extreme[rule] = self.stats.suppressed_extreme.get(rule, 0) + 1
        if lost_volume > 0:
            self.stats.suppressed_volume[rule] = (
                self.stats.suppressed_volume.get(rule, 0.0) + lost_volume
            )
        self.stats.examples.setdefault(
            rule, f"{tick.vt_symbol} @{tick.datetime} px={tick.last_price} {detail}"
        )


class TickFilterMixin:
    """Gateway-side wiring. Mix in BEFORE BaseGateway, like the other kit mixins:

        class FutuGateway(TickFilterMixin, NonBlockingConnectMixin, ..., BaseGateway):
            ...

    Then replace every `self.on_tick(tick)` call site with `self.push_tick(...)`.
    on_tick is deliberately NOT overridden: futu can supply broker status
    (sec_status/dark_status/suspension) and uSMART cannot, and an override
    cannot carry those extra arguments. An explicit call site also keeps the
    filtering visible where the tick is produced instead of hiding it in the
    MRO — and makes double-filtering impossible.

    With no filter installed, push_tick is a straight passthrough, so a gateway
    can adopt the call site before adopting the policy.
    """

    tick_filter: TickFilter | None = None

    def install_tick_filter(self, tick_filter: TickFilter | None) -> None:
        self.tick_filter = tick_filter

    def push_tick(
        self,
        tick: TickData,
        *,
        sec_status: str | None = None,
        dark_status: str | None = None,
        suspended: bool | None = None,
    ) -> Verdict | None:
        """Filter, then forward to BaseGateway.on_tick if it survives."""
        if self.tick_filter is None:
            self.on_tick(tick)  # type: ignore[attr-defined]
            return None

        verdict = self.tick_filter.check(
            tick, sec_status=sec_status, dark_status=dark_status, suspended=suspended
        )
        if verdict.delivered:
            self.on_tick(tick)  # type: ignore[attr-defined]
        return verdict

    def tick_filter_report(self) -> str:
        if self.tick_filter is None:
            return "tick filter not installed"
        return self.tick_filter.stats.report()
