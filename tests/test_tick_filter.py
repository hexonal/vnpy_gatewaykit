"""Tests for vnpy_gatewaykit.tick_filter.

The 误杀 tests are the point of this file, not an afterthought:
test_book_only_update_survives, test_suppressed_extreme_is_reported and
test_observe_and_enforce_agree are the three that decide whether the filter is
allowed anywhere near a live tape.

All timestamps are real HK/US trading days: 2026-07-24 is a Friday,
2026-07-27 a Monday, 2026-01-15 a Thursday (US winter, EST) and 2026-07-15 a
Wednesday (US summer, EDT) — the last two pin the DST behaviour.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from math import inf, nan
from zoneinfo import ZoneInfo

import pytest
from vnpy.trader.constant import Exchange
from vnpy.trader.object import TickData
from vnpy_gatewaykit.tick_filter import (
    SAFE_RULES,
    STRICT_RULES,
    FilterMode,
    Phase,
    Rule,
    TickFilter,
    TickFilterMixin,
)

HKT = ZoneInfo("Asia/Hong_Kong")
ET = ZoneInfo("America/New_York")

GATEWAY = "TEST"


def hk(hour: int, minute: int, second: int = 0, *, day: int = 24) -> datetime:
    return datetime(2026, 7, day, hour, minute, second, tzinfo=HKT)


def us(hour: int, minute: int, second: int = 0, *, day: int = 24, month: int = 7) -> datetime:
    return datetime(2026, month, day, hour, minute, second, tzinfo=ET)


def tick(
    dt: datetime,
    price: float = 100.0,
    *,
    symbol: str = "700",
    exchange: Exchange = Exchange.SEHK,
    volume: float = 1_000.0,
    turnover: float = 100_000.0,
    bid1: float = 99.9,
    ask1: float = 100.1,
    bid1_vol: float = 500.0,
) -> TickData:
    return TickData(
        symbol=symbol,
        exchange=exchange,
        datetime=dt,
        last_price=price,
        volume=volume,
        turnover=turnover,
        bid_price_1=bid1,
        ask_price_1=ask1,
        bid_volume_1=bid1_vol,
        gateway_name=GATEWAY,
    )


def fixed_clock(moment: datetime):
    return lambda: moment


# ---------------------------------------------------------------------------
# Weekday sanity — every other test's session expectations depend on these.
# ---------------------------------------------------------------------------


def test_reference_dates_are_the_weekdays_the_tests_assume() -> None:
    assert datetime(2026, 7, 24).weekday() == 4   # Friday
    assert datetime(2026, 7, 27).weekday() == 0   # Monday
    assert datetime(2026, 1, 15).weekday() == 3   # Thursday
    assert datetime(2026, 7, 15).weekday() == 2   # Wednesday


# ---------------------------------------------------------------------------
# Dry-run guarantee
# ---------------------------------------------------------------------------


def test_default_mode_never_drops_anything() -> None:
    """OBSERVE is the default and must be incapable of removing a tick."""
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    stream = [
        tick(hk(10, 0, 0), 100.0),
        tick(hk(10, 0, 0), 100.0),            # exact duplicate
        tick(hk(9, 59, 0), 100.0),            # regression
        tick(hk(10, 0, 5), 0.0),              # bad price
        tick(hk(10, 0, 6), 101.0),
    ]
    verdicts = [f.check(t) for t in stream]

    assert all(v.delivered for v in verdicts)
    assert f.stats.dropped == {}
    # ...but it still saw everything
    assert f.stats.would_drop[Rule.DUPLICATE] == 1
    assert f.stats.would_drop[Rule.REGRESSION] == 1
    assert f.stats.would_drop[Rule.BAD_PRICE] == 1


def test_enforce_only_drops_rules_explicitly_listed() -> None:
    f = TickFilter(
        mode=FilterMode.ENFORCE,
        enforce={Rule.BAD_PRICE},
        clock=fixed_clock(hk(10, 0, 30)),
    )
    kept = f.check(tick(hk(10, 0, 0), 100.0))
    bad = f.check(tick(hk(10, 0, 1), 0.0))
    dup = f.check(tick(hk(10, 0, 0), 100.0))  # duplicate AND regression-adjacent

    assert kept.delivered
    assert not bad.delivered and bad.rule is Rule.BAD_PRICE
    assert dup.delivered, "DUPLICATE was not in enforce=, so it must still pass through"
    assert f.stats.dropped == {Rule.BAD_PRICE: 1}


def test_enforce_with_empty_rule_set_is_a_passthrough() -> None:
    f = TickFilter(mode=FilterMode.ENFORCE, clock=fixed_clock(hk(10, 0, 30)))
    f.check(tick(hk(10, 0, 0)))
    v = f.check(tick(hk(10, 0, 0)))
    assert v.rule is Rule.DUPLICATE
    assert v.delivered
    assert f.stats.dropped == {}


def test_observe_and_enforce_agree() -> None:
    """The dry-run count must equal what enforcement would really have done.

    This only holds because reference state advances on kept ticks alone, in
    both modes. If OBSERVE let a would-be-dropped tick become the reference,
    the next tick would be judged against a poisoned baseline and the two runs
    would diverge — making the dry-run numbers useless for the go/no-go call.
    """
    stream = [
        tick(hk(10, 0, 0), 100.0),
        tick(hk(10, 0, 0), 100.0),          # duplicate
        tick(hk(10, 0, 0), 100.0),          # duplicate again
        tick(hk(9, 59, 0), 90.0),           # regression
        # Still a regression: the baseline is 10:00:00, the last KEPT tick —
        # not 09:59:00, which was rejected. If a rejected tick were allowed to
        # become the baseline, this one would sail through and the two modes
        # would report different numbers.
        tick(hk(9, 59, 30), 91.0),
        tick(hk(10, 0, 1), 100.5),
        tick(hk(9, 58, 0), 80.0),           # regression against 10:00:01
        tick(hk(10, 0, 2), 101.0),
    ]

    observe = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    enforce = TickFilter(
        mode=FilterMode.ENFORCE, enforce=STRICT_RULES, clock=fixed_clock(hk(10, 0, 30))
    )
    for t in stream:
        observe.check(t)
        enforce.check(t)

    assert observe.stats.would_drop == enforce.stats.would_drop
    assert observe.stats.kept == enforce.stats.kept
    assert enforce.stats.dropped == {Rule.DUPLICATE: 2, Rule.REGRESSION: 3}


# ---------------------------------------------------------------------------
# 误杀 guards
# ---------------------------------------------------------------------------


def test_book_only_update_survives() -> None:
    """A book push repeats the trade fields verbatim. It must NOT be a duplicate.

    Both gateways deliver the order book on a channel separate from the quote,
    so this shape is the normal case, not an edge one. vnpy's client-side stop
    prices from bid_price_5/ask_price_5, so swallowing book updates would leave
    it pricing off a stale ladder.
    """
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    base = tick(hk(10, 0, 0), 100.0, bid1=99.9, ask1=100.1)
    moved = tick(hk(10, 0, 0), 100.0, bid1=99.95, ask1=100.05)
    thinner = tick(hk(10, 0, 0), 100.0, bid1=99.95, ask1=100.05, bid1_vol=100.0)

    assert f.check(base).keep
    assert f.check(moved).keep, "bid/ask moved — that is new information"
    assert f.check(thinner).keep, "book size changed — also new information"
    assert Rule.DUPLICATE not in f.stats.would_drop


def test_same_second_different_content_is_kept_and_counted() -> None:
    """futu stamps HK quotes to the second, so collisions are guaranteed.

    Two real trades inside 10:00:00 are indistinguishable by timestamp. They
    are distinct trades and must survive; the counter exists so anyone keying
    storage on (symbol, datetime) can see how many rows they are about to lose.
    """
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    assert f.check(tick(hk(10, 0, 0), 100.0, volume=1_000)).keep
    assert f.check(tick(hk(10, 0, 0), 100.5, volume=1_200)).keep
    assert f.check(tick(hk(10, 0, 0), 101.0, volume=1_500)).keep

    assert f.stats.same_second == 2
    assert f.stats.would_drop == {}


def test_suppressed_extreme_is_reported() -> None:
    """A rule that would eat a bar extreme has to say so, loudly."""
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    f.check(tick(hk(10, 0, 1), 100.0, volume=1_000))
    f.check(tick(hk(10, 0, 3), 101.0, volume=1_100))
    late_high = f.check(tick(hk(10, 0, 2), 105.0, volume=1_400))  # out of order, new high

    assert late_high.rule is Rule.REGRESSION
    assert late_high.lost_extreme is True
    assert late_high.lost_volume == pytest.approx(300.0)
    assert f.stats.suppressed_extreme[Rule.REGRESSION] == 1
    assert f.stats.suppressed_volume[Rule.REGRESSION] == pytest.approx(300.0)
    assert "regression" in f.stats.report()
    assert "NOT safe to enforce" in f.stats.report()


def test_information_free_drop_reports_no_loss() -> None:
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    f.check(tick(hk(10, 0, 0), 100.0, volume=1_000))
    repeat = f.check(tick(hk(10, 0, 0), 100.0, volume=1_000))

    assert repeat.rule is Rule.DUPLICATE
    assert repeat.lost_extreme is False
    assert repeat.lost_volume == 0.0
    assert f.stats.suppressed_extreme == {}
    assert f.stats.suppressed_volume == {}
    assert "every matched rule dropped only information-free ticks" in f.stats.report()


def test_blank_minute_in_regular_session_is_flagged_separately_from_closed() -> None:
    """Losing every tick of a minute is the failure the audit exists to catch.

    A blank minute outside the continuous session is expected (it is the
    out-of-hours frozen snapshot being suppressed); a blank minute inside it
    means a bar lost all of its ticks.
    """
    f = TickFilter(clock=fixed_clock(hk(10, 2, 0)))
    f.check(tick(hk(10, 0, 0), 100.0))            # kept -> minute 10:00 not blank
    f.check(tick(hk(10, 1, 0), 0.0))              # bad price, only tick of 10:01
    f.check(tick(hk(10, 1, 30), -5.0))            # bad price, same minute
    f.check(tick(hk(10, 2, 0), 100.5))            # rolls 10:01 shut

    assert f.stats.blank_minutes.get(Phase.REGULAR) == 1
    assert "ALARM" in f.stats.report()


def test_blank_minute_outside_regular_hours_is_marked_expected() -> None:
    f = TickFilter(clock=fixed_clock(us(5, 0)))
    f.check(tick(us(16, 0, 0, day=23), 100.0, exchange=Exchange.SMART, symbol="AAPL"))
    f.check(tick(us(16, 0, 0, day=23), 100.0, exchange=Exchange.SMART, symbol="AAPL"))
    f.check(tick(us(5, 0, 0), 100.0, exchange=Exchange.SMART, symbol="AAPL"))

    assert f.stats.blank_minutes.get(Phase.REGULAR, 0) == 0
    assert f.stats.blank_minutes.get(Phase.EXTENDED) == 1
    assert "expected" in f.stats.report()


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("price", [0.0, -1.0, nan, inf, -inf])
def test_bad_price_variants(price: float) -> None:
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    v = f.check(tick(hk(10, 0, 0), price))
    assert v.rule is Rule.BAD_PRICE
    assert not v.keep


def test_naive_datetime_is_rejected_not_guessed() -> None:
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    naive = TickData(
        symbol="700",
        exchange=Exchange.SEHK,
        datetime=datetime(2026, 7, 24, 10, 0, 0),
        last_price=100.0,
        gateway_name=GATEWAY,
    )
    v = f.check(naive)
    assert v.rule is Rule.NAIVE_DATETIME
    assert v.phase is Phase.UNKNOWN


def test_regression_matches_only_a_strictly_earlier_timestamp() -> None:
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    f.check(tick(hk(10, 0, 5), 100.0))
    assert f.check(tick(hk(10, 0, 5), 100.5)).keep          # equal is fine
    assert f.check(tick(hk(10, 0, 4), 100.6)).rule is Rule.REGRESSION


def test_exact_duplicate_matches() -> None:
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    f.check(tick(hk(10, 0, 0), 100.0))
    assert f.check(tick(hk(10, 0, 0), 100.0)).rule is Rule.DUPLICATE


def test_duplicate_compares_against_last_kept_not_last_seen() -> None:
    """A rejected tick must not become the baseline."""
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    f.check(tick(hk(10, 0, 0), 100.0))
    f.check(tick(hk(10, 0, 1), 0.0))                        # rejected: bad price
    assert f.check(tick(hk(10, 0, 0), 100.0)).rule is Rule.DUPLICATE


# ---------------------------------------------------------------------------
# Staleness — the futu QUOTE out-of-hours pathology
# ---------------------------------------------------------------------------


def test_futu_premarket_frozen_snapshot_is_stale_and_costs_nothing() -> None:
    """futu's QUOTE channel is RTH-only; out of hours it replays the close.

    Pre-market at 05:00 ET the push still carries data_time 16:00:00.324,
    last_price = the RTH close and volume = the RTH total. That is a verbatim
    replay of a tick already delivered the previous afternoon, so the audit
    must show it costs nothing to drop.
    """
    f = TickFilter(clock=fixed_clock(us(5, 0, 0, day=27)))
    frozen = tick(
        us(16, 0, 0, day=24),
        333.02,
        symbol="AAPL",
        exchange=Exchange.SMART,
        volume=47_489_415.0,
    )
    v = f.check(frozen)

    assert v.rule is Rule.STALE
    assert v.phase is Phase.EXTENDED       # 16:00-20:00 ET is the after-hours window
    assert v.lost_extreme is False
    assert v.lost_volume == 0.0


def test_staleness_is_measured_in_open_seconds_not_wall_clock() -> None:
    """HK lunch is an hour of wall clock and zero seconds of trading."""
    f = TickFilter(max_stale_seconds=300.0, clock=fixed_clock(hk(12, 59, 0)))
    v = f.check(tick(hk(11, 59, 30), 100.0))
    assert v.keep, "only 30 open seconds elapsed — the lunch hour does not count"

    g = TickFilter(max_stale_seconds=300.0, clock=fixed_clock(hk(13, 10, 0)))
    assert g.check(tick(hk(11, 50, 0), 100.0)).rule is Rule.STALE  # 10min + 10min open


def test_weekend_does_not_by_itself_make_the_open_stale() -> None:
    f = TickFilter(max_stale_seconds=300.0, clock=fixed_clock(hk(9, 30, 10, day=27)))
    assert f.check(tick(hk(9, 30, 0, day=27), 100.0)).keep


def test_future_dated_tick_is_not_stale() -> None:
    """Broker/machine clock skew must not read as staleness."""
    f = TickFilter(max_stale_seconds=1.0, clock=fixed_clock(hk(10, 0, 0)))
    assert f.check(tick(hk(10, 0, 30), 100.0)).keep


def test_absurd_timestamp_is_stale_rather_than_crashing() -> None:
    f = TickFilter(clock=fixed_clock(hk(10, 0, 0)))
    ancient = tick(datetime(1970, 1, 2, 10, 0, tzinfo=HKT), 100.0)
    assert f.check(ancient).rule is Rule.STALE


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (hk(9, 15), Phase.AUCTION),    # pre-opening session
        (hk(9, 45), Phase.REGULAR),
        (hk(12, 30), Phase.CLOSED),    # lunch
        (hk(14, 0), Phase.REGULAR),
        (hk(16, 5), Phase.AUCTION),    # closing auction
        (hk(17, 0), Phase.CLOSED),
    ],
)
def test_hk_phase_classification(moment: datetime, expected: Phase) -> None:
    f = TickFilter(clock=fixed_clock(moment + timedelta(seconds=1)))
    assert f.check(tick(moment, 100.0)).phase is expected


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (us(5, 0), Phase.EXTENDED),
        (us(10, 0), Phase.REGULAR),
        (us(18, 0), Phase.EXTENDED),
        (us(21, 0), Phase.CLOSED),
        (us(5, 0, month=1, day=15), Phase.EXTENDED),   # EST
        (us(10, 0, month=1, day=15), Phase.REGULAR),   # EST
        (us(10, 0, month=7, day=15), Phase.REGULAR),   # EDT
    ],
)
def test_us_phase_classification_across_dst(moment: datetime, expected: Phase) -> None:
    f = TickFilter(clock=fixed_clock(moment + timedelta(seconds=1)))
    v = f.check(tick(moment, 100.0, symbol="AAPL", exchange=Exchange.SMART))
    assert v.phase is expected


def test_unmapped_exchange_is_unknown_and_never_dropped() -> None:
    """sessions.py maps SEHK/SMART only; futu also serves SH/SZ contracts."""
    f = TickFilter(
        mode=FilterMode.ENFORCE, enforce=STRICT_RULES, clock=fixed_clock(hk(10, 0, 30))
    )
    v = f.check(
        tick(datetime(2026, 7, 24, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
             100.0, symbol="600000", exchange=Exchange.SSE)
    )
    assert v.phase is Phase.UNKNOWN
    assert v.delivered


def test_phase_policy_is_opt_in() -> None:
    auction = tick(hk(9, 15), 100.0)

    permissive = TickFilter(clock=fixed_clock(hk(9, 15, 1)))
    assert permissive.check(auction).keep

    rth_only = TickFilter(
        allowed_phases={Phase.REGULAR}, clock=fixed_clock(hk(9, 15, 1))
    )
    v = rth_only.check(auction)
    assert v.rule is Rule.PHASE_NOT_ALLOWED
    assert v.phase is Phase.AUCTION


def test_auction_ticks_are_kept_by_default() -> None:
    """The HK opening auction was the largest print of the morning on 00700.

    futu's own 1m series carries it as a standalone 545,400-share bar. Dropping
    the auction by default would delete the open.
    """
    f = TickFilter(
        mode=FilterMode.ENFORCE, enforce=STRICT_RULES, clock=fixed_clock(hk(9, 20, 5))
    )
    v = f.check(tick(hk(9, 20, 0), 438.2, volume=545_400))
    assert v.delivered
    assert v.phase is Phase.AUCTION


# ---------------------------------------------------------------------------
# Broker status — the actual port of CTP's InstrumentStatus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    ["SUSPENDED", "CALLED", "DELISTED", "RECOVERABLE_CIRCUIT_BREAKER", "CHANGED_CODE_TRAD_END"],
)
def test_halt_statuses(status: str) -> None:
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    v = f.check(tick(hk(10, 0, 0), 100.0), sec_status=status)
    assert v.rule is Rule.HALTED
    assert v.phase is Phase.HALTED


def test_suspension_boolean_alone_halts() -> None:
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    v = f.check(tick(hk(10, 0, 0), 100.0), sec_status="NORMAL", suspended=True)
    assert v.rule is Rule.HALTED


@pytest.mark.parametrize(
    ("sec_status", "dark_status"),
    [("DARK_TRADING", None), ("BEFORE_DARK_TRADE_OPEING", None), (None, "TRADING"), (None, "END")],
)
def test_dark_market_is_classified_separately(sec_status, dark_status) -> None:
    """HK 暗盘 fills are real but broker-internal — never on the SEHK tape."""
    f = TickFilter(clock=fixed_clock(hk(17, 0, 0)))
    v = f.check(tick(hk(16, 30, 0), 100.0), sec_status=sec_status, dark_status=dark_status)
    assert v.rule is Rule.DARK_MARKET
    assert v.phase is Phase.DARK


def test_normal_status_does_not_interfere() -> None:
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    v = f.check(tick(hk(10, 0, 0), 100.0), sec_status="NORMAL", dark_status="N/A", suspended=False)
    assert v.keep
    assert v.phase is Phase.REGULAR


def test_pre_trade_status_is_counted_but_never_dropped() -> None:
    """TO_BE_OPEN during HK's pre-opening session is unverified.

    If futu does report it there, halting on it would delete the opening
    auction — so it is counted for later measurement and nothing else.
    """
    f = TickFilter(
        mode=FilterMode.ENFORCE, enforce=STRICT_RULES, clock=fixed_clock(hk(9, 20, 5))
    )
    v = f.check(tick(hk(9, 20, 0), 438.2), sec_status="TO_BE_OPEN")
    assert v.delivered
    assert f.stats.pre_trade_status == 1


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def test_reset_symbol_clears_the_resubscribe_replay() -> None:
    """Both feeds replay a snapshot on subscribe; that is not a real duplicate."""
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    snap = tick(hk(10, 0, 0), 100.0)
    assert f.check(snap).keep
    f.reset_symbol(snap.vt_symbol)
    assert f.check(tick(hk(10, 0, 0), 100.0)).keep
    assert Rule.DUPLICATE not in f.stats.would_drop


def test_symbols_do_not_share_state() -> None:
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    assert f.check(tick(hk(10, 0, 0), 100.0, symbol="700")).keep
    assert f.check(tick(hk(9, 59, 0), 50.0, symbol="9988")).keep, "different symbol"
    assert f.check(tick(hk(9, 59, 0), 100.0, symbol="700")).rule is Rule.REGRESSION


def test_reset_stats_keeps_reference_points() -> None:
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    f.check(tick(hk(10, 0, 0), 100.0))
    f.reset_stats()
    assert f.stats.seen == 0
    assert f.check(tick(hk(10, 0, 0), 100.0)).rule is Rule.DUPLICATE


def test_report_is_readable_when_nothing_matched() -> None:
    f = TickFilter(clock=fixed_clock(hk(10, 0, 30)))
    f.check(tick(hk(10, 0, 0), 100.0))
    text = f.stats.report()
    assert "no rule matched any tick" in text
    assert "regular=1" in text


# ---------------------------------------------------------------------------
# Gateway mixin
# ---------------------------------------------------------------------------


class _FakeGateway(TickFilterMixin):
    """Minimal stand-in: TickFilterMixin only ever calls self.on_tick."""

    def __init__(self) -> None:
        self.pushed: list[TickData] = []

    def on_tick(self, tick_: TickData) -> None:
        self.pushed.append(tick_)


def test_mixin_without_filter_is_a_straight_passthrough() -> None:
    gw = _FakeGateway()
    t = tick(hk(10, 0, 0), 100.0)
    assert gw.push_tick(t) is None
    assert gw.pushed == [t]
    assert gw.tick_filter_report() == "tick filter not installed"


def test_mixin_forwards_kept_ticks_and_withholds_enforced_drops() -> None:
    gw = _FakeGateway()
    gw.install_tick_filter(
        TickFilter(
            mode=FilterMode.ENFORCE, enforce=SAFE_RULES, clock=fixed_clock(hk(10, 0, 30))
        )
    )
    good = tick(hk(10, 0, 0), 100.0)
    zero = tick(hk(10, 0, 1), 0.0)

    assert gw.push_tick(good).delivered
    assert gw.push_tick(zero).delivered is False
    assert gw.pushed == [good]


def test_mixin_passes_broker_status_through() -> None:
    gw = _FakeGateway()
    gw.install_tick_filter(
        TickFilter(
            mode=FilterMode.ENFORCE, enforce=STRICT_RULES, clock=fixed_clock(hk(10, 0, 30))
        )
    )
    verdict = gw.push_tick(tick(hk(10, 0, 0), 100.0), sec_status="SUSPENDED")
    assert verdict is not None and verdict.rule is Rule.HALTED
    assert gw.pushed == []
    assert "halted" in gw.tick_filter_report()


def test_mixin_observe_mode_forwards_everything() -> None:
    gw = _FakeGateway()
    gw.install_tick_filter(TickFilter(clock=fixed_clock(hk(10, 0, 30))))
    gw.push_tick(tick(hk(10, 0, 0), 100.0))
    gw.push_tick(tick(hk(10, 0, 0), 100.0))
    gw.push_tick(tick(hk(10, 0, 1), 0.0), sec_status="SUSPENDED")
    assert len(gw.pushed) == 3
