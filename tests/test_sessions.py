"""Trading-session windows — the time-of-day half of the market clock.

The cases that matter are the ones a hand-written `if 9 <= hour < 16` gets
wrong: HK's lunch break, US DST (the same 09:30 ET is a different UTC instant
in January and July), abutting windows that must not double-claim an instant,
weekends/holidays, and naive datetimes sneaking in from a machine whose
timezone is not the market's.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from vnpy.trader.constant import Exchange

from vnpy_gatewaykit.sessions import (
    MAX_SCAN_DAYS,
    Session,
    SessionKind,
    StaticHolidayCalendar,
    WeekdayCalendar,
    active_session,
    day_close,
    is_open,
    next_window_change,
    open_seconds_between,
    previous_close,
    sessions_for,
    supported_exchanges,
    windows,
)

HKT = ZoneInfo("Asia/Hong_Kong")
ET = ZoneInfo("America/New_York")

# 2026-07-23 is a Thursday; 2026-07-25/26 the weekend after it.
THU = date(2026, 7, 23)
SAT = date(2026, 7, 25)
SUN = date(2026, 7, 26)
MON = date(2026, 7, 27)


def hk(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=HKT)


def et(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ET)


# ---------------------------------------------------------------------------
# Table sanity
# ---------------------------------------------------------------------------
def test_only_hk_and_us_are_mapped() -> None:
    """This fork trades HK + US cash equities only — no CN, no futures."""
    assert set(supported_exchanges()) == {Exchange.SEHK, Exchange.SMART}


def test_unmapped_exchange_raises_rather_than_defaulting() -> None:
    with pytest.raises(KeyError):
        sessions_for(Exchange.SSE)


@pytest.mark.parametrize("exchange", [Exchange.SEHK, Exchange.SMART])
def test_sessions_are_ordered_non_overlapping_and_same_day(exchange: Exchange) -> None:
    previous_end = time(0, 0)
    for session in sessions_for(exchange):
        assert session.start < session.end, "no window may cross local midnight"
        assert session.start >= previous_end, "windows must be ordered and disjoint"
        previous_end = session.end


def test_empty_session_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        Session("坏窗口", time(10, 0), time(10, 0), SessionKind.REGULAR)


# ---------------------------------------------------------------------------
# HK: lunch break is a real hole, auctions glue onto the continuous session
# ---------------------------------------------------------------------------
def test_hk_lunch_break_is_closed() -> None:
    assert is_open(Exchange.SEHK, hk(2026, 7, 23, 11, 59))
    assert not is_open(Exchange.SEHK, hk(2026, 7, 23, 12, 0))
    assert not is_open(Exchange.SEHK, hk(2026, 7, 23, 12, 59))
    assert is_open(Exchange.SEHK, hk(2026, 7, 23, 13, 0))


def test_hk_windows_merge_auctions_but_keep_lunch() -> None:
    got = windows(Exchange.SEHK, THU)
    assert got == (
        (hk(2026, 7, 23, 9, 0), hk(2026, 7, 23, 12, 0)),
        (hk(2026, 7, 23, 13, 0), hk(2026, 7, 23, 16, 10)),
    )


def test_hk_regular_only_excludes_auctions() -> None:
    got = windows(Exchange.SEHK, THU, kinds=[SessionKind.REGULAR])
    assert got == (
        (hk(2026, 7, 23, 9, 30), hk(2026, 7, 23, 12, 0)),
        (hk(2026, 7, 23, 13, 0), hk(2026, 7, 23, 16, 0)),
    )
    assert not is_open(
        Exchange.SEHK, hk(2026, 7, 23, 9, 15), kinds=[SessionKind.REGULAR]
    )
    assert is_open(Exchange.SEHK, hk(2026, 7, 23, 9, 15))


def test_hk_close_is_1610_with_auction_and_1600_without() -> None:
    assert day_close(Exchange.SEHK, THU) == hk(2026, 7, 23, 16, 10)
    assert day_close(Exchange.SEHK, THU, kinds=[SessionKind.REGULAR]) == hk(
        2026, 7, 23, 16, 0
    )


def test_session_boundaries_are_half_open() -> None:
    """09:30 belongs to the continuous session, not to the auction that ended."""
    at_open = active_session(Exchange.SEHK, hk(2026, 7, 23, 9, 30))
    assert at_open is not None and at_open.kind is SessionKind.REGULAR

    just_before = active_session(Exchange.SEHK, hk(2026, 7, 23, 9, 29))
    assert just_before is not None and just_before.kind is SessionKind.AUCTION


# ---------------------------------------------------------------------------
# US: extended hours + DST
# ---------------------------------------------------------------------------
def test_us_extended_hours_are_open_but_not_regular() -> None:
    premarket = et(2026, 7, 23, 5, 0)
    assert is_open(Exchange.SMART, premarket)
    assert not is_open(Exchange.SMART, premarket, kinds=[SessionKind.REGULAR])

    afterhours = et(2026, 7, 23, 18, 0)
    assert is_open(Exchange.SMART, afterhours)
    assert not is_open(Exchange.SMART, afterhours, kinds=[SessionKind.REGULAR])

    assert not is_open(Exchange.SMART, et(2026, 7, 23, 20, 0))
    assert not is_open(Exchange.SMART, et(2026, 7, 23, 3, 59))


def test_us_open_is_a_different_utc_instant_in_summer_and_winter() -> None:
    """The DST bug a fixed UTC offset would bake in."""
    summer = windows(Exchange.SMART, date(2026, 7, 23), kinds=[SessionKind.REGULAR])[0][0]
    winter = windows(Exchange.SMART, date(2026, 1, 22), kinds=[SessionKind.REGULAR])[0][0]

    assert summer.astimezone(timezone.utc).hour == 13  # EDT = UTC-4
    assert winter.astimezone(timezone.utc).hour == 14  # EST = UTC-5
    assert summer.timetz().hour == winter.timetz().hour == 9


def test_us_regular_open_judged_from_a_utc_instant() -> None:
    """Callers pass UTC; the market's own wall clock still decides."""
    utc_1330_july = datetime(2026, 7, 23, 13, 30, tzinfo=timezone.utc)
    assert is_open(Exchange.SMART, utc_1330_july, kinds=[SessionKind.REGULAR])

    utc_1330_january = datetime(2026, 1, 22, 13, 30, tzinfo=timezone.utc)
    assert not is_open(Exchange.SMART, utc_1330_january, kinds=[SessionKind.REGULAR])


def test_us_session_spans_the_hk_night_without_crossing_local_midnight() -> None:
    """21:30 HKT (US open in summer) is still the same US local day."""
    us_open_in_hk_time = hk(2026, 7, 23, 21, 30)
    session = active_session(Exchange.SMART, us_open_in_hk_time)
    assert session is not None and session.kind is SessionKind.REGULAR
    assert us_open_in_hk_time.astimezone(ET).date() == date(2026, 7, 23)


# ---------------------------------------------------------------------------
# Naive datetimes are refused
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "call",
    [
        lambda dt: is_open(Exchange.SEHK, dt),
        lambda dt: active_session(Exchange.SEHK, dt),
        lambda dt: next_window_change(Exchange.SEHK, dt),
        lambda dt: previous_close(Exchange.SEHK, dt),
    ],
)
def test_naive_datetime_is_rejected(call) -> None:  # noqa: ANN001 — parametrized lambda
    with pytest.raises(ValueError, match="timezone-aware"):
        call(datetime(2026, 7, 23, 10, 0))


# ---------------------------------------------------------------------------
# next_window_change — what a scheduler sleeps until
# ---------------------------------------------------------------------------
def test_inside_a_window_next_change_is_its_close() -> None:
    assert next_window_change(Exchange.SEHK, hk(2026, 7, 23, 10, 0)) == hk(
        2026, 7, 23, 12, 0
    )


def test_in_the_lunch_gap_next_change_is_the_afternoon_open() -> None:
    assert next_window_change(Exchange.SEHK, hk(2026, 7, 23, 12, 30)) == hk(
        2026, 7, 23, 13, 0
    )


def test_after_the_close_next_change_is_tomorrows_open() -> None:
    assert next_window_change(Exchange.SEHK, hk(2026, 7, 23, 17, 0)) == hk(
        2026, 7, 24, 9, 0
    )


def test_friday_night_next_change_skips_the_weekend() -> None:
    friday_evening = hk(2026, 7, 24, 20, 0)
    assert next_window_change(Exchange.SEHK, friday_evening) == hk(2026, 7, 27, 9, 0)


def test_next_change_is_strictly_after_the_moment() -> None:
    """Standing exactly on an open must not return that same instant, or a
    scheduler sleeping until it would spin."""
    at_open = hk(2026, 7, 23, 9, 0)
    nxt = next_window_change(Exchange.SEHK, at_open)
    assert nxt is not None and nxt > at_open
    assert nxt == hk(2026, 7, 23, 12, 0)


def test_next_change_returns_none_when_nothing_opens_in_horizon() -> None:
    class NeverOpen:
        def is_trading_day(self, exchange: Exchange, day: date) -> bool:
            return False

    assert next_window_change(Exchange.SEHK, hk(2026, 7, 23, 10, 0), calendar=NeverOpen()) is None


def test_horizon_is_wide_enough_for_a_long_holiday_run() -> None:
    assert MAX_SCAN_DAYS >= 10


# ---------------------------------------------------------------------------
# previous_close — what should already be complete in the database
# ---------------------------------------------------------------------------
def test_previous_close_mid_session_is_the_earlier_window_end() -> None:
    assert previous_close(Exchange.SEHK, hk(2026, 7, 23, 14, 0)) == hk(
        2026, 7, 23, 12, 0
    )


def test_previous_close_on_sunday_is_fridays_close() -> None:
    assert previous_close(Exchange.SEHK, datetime(2026, 7, 26, 10, 0, tzinfo=HKT)) == hk(
        2026, 7, 24, 16, 10
    )


def test_previous_close_before_the_first_window_looks_at_earlier_days() -> None:
    assert previous_close(Exchange.SEHK, hk(2026, 7, 23, 8, 0)) == hk(
        2026, 7, 22, 16, 10
    )


# ---------------------------------------------------------------------------
# Calendars
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("day", [SAT, SUN])
def test_weekends_have_no_windows(day: date) -> None:
    assert windows(Exchange.SEHK, day) == ()
    assert windows(Exchange.SMART, day) == ()
    assert day_close(Exchange.SEHK, day) is None


def test_weekday_calendar_does_not_know_holidays() -> None:
    """Documented limitation, asserted so nobody assumes otherwise."""
    lunar_new_year_2026 = date(2026, 2, 17)  # a Tuesday, HKEX closed in reality
    assert WeekdayCalendar().is_trading_day(Exchange.SEHK, lunar_new_year_2026)


def test_static_holiday_calendar_closes_the_listed_day_for_that_exchange_only() -> None:
    calendar = StaticHolidayCalendar({Exchange.SEHK: [THU]})

    assert not calendar.is_trading_day(Exchange.SEHK, THU)
    assert calendar.is_trading_day(Exchange.SMART, THU)
    assert windows(Exchange.SEHK, THU, calendar=calendar) == ()
    assert windows(Exchange.SMART, THU, calendar=calendar) != ()
    assert calendar.holidays(Exchange.SEHK) == frozenset({THU})


def test_holiday_is_skipped_when_scheduling_the_next_open() -> None:
    calendar = StaticHolidayCalendar({Exchange.SEHK: [date(2026, 7, 24), MON]})
    friday_eve = hk(2026, 7, 23, 17, 0)
    assert next_window_change(Exchange.SEHK, friday_eve, calendar=calendar) == hk(
        2026, 7, 28, 9, 0
    )


def test_static_calendar_still_closes_weekends() -> None:
    calendar = StaticHolidayCalendar({Exchange.SEHK: []})
    assert not calendar.is_trading_day(Exchange.SEHK, SAT)


# ---------------------------------------------------------------------------
# open_seconds_between — the unit gap detection must use
# ---------------------------------------------------------------------------
def test_open_seconds_inside_one_window() -> None:
    got = open_seconds_between(
        Exchange.SEHK, hk(2026, 7, 23, 10, 0), hk(2026, 7, 23, 10, 30)
    )
    assert got == 30 * 60


def test_open_seconds_excludes_the_lunch_break() -> None:
    """11:55 → 13:05 is 70 wall-clock minutes but only 10 open ones."""
    got = open_seconds_between(
        Exchange.SEHK, hk(2026, 7, 23, 11, 55), hk(2026, 7, 23, 13, 5)
    )
    assert got == 10 * 60


def test_open_seconds_across_a_night_counts_only_the_two_sessions() -> None:
    got = open_seconds_between(
        Exchange.SEHK, hk(2026, 7, 23, 15, 55), hk(2026, 7, 24, 9, 35)
    )
    # 15:55–16:10 = 15 min, then 09:00–09:35 = 35 min the next morning.
    assert got == 50 * 60


def test_open_seconds_across_a_weekend_is_zero() -> None:
    got = open_seconds_between(
        Exchange.SEHK, hk(2026, 7, 24, 17, 0), hk(2026, 7, 27, 8, 0)
    )
    assert got == 0.0


def test_open_seconds_full_hk_regular_day() -> None:
    got = open_seconds_between(
        Exchange.SEHK,
        hk(2026, 7, 23, 0, 0),
        hk(2026, 7, 24, 0, 0),
        kinds=[SessionKind.REGULAR],
    )
    assert got == (150 + 180) * 60  # 09:30–12:00 + 13:00–16:00


def test_open_seconds_is_zero_for_reversed_or_equal_bounds() -> None:
    moment = hk(2026, 7, 23, 10, 0)
    assert open_seconds_between(Exchange.SEHK, moment, moment) == 0.0
    assert open_seconds_between(Exchange.SEHK, moment, moment - timedelta(hours=1)) == 0.0


def test_open_seconds_refuses_an_absurd_span() -> None:
    with pytest.raises(ValueError, match="max_span_days"):
        open_seconds_between(
            Exchange.SEHK,
            hk(2020, 1, 1, 10, 0),
            hk(2026, 1, 1, 10, 0),
        )


# ---------------------------------------------------------------------------
# Whole-day consistency sweep: walk a day minute by minute and check that
# is_open agrees with the merged windows at every step.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("exchange", [Exchange.SEHK, Exchange.SMART])
def test_is_open_agrees_with_windows_all_day(exchange: Exchange) -> None:
    day_windows = windows(exchange, THU)
    cursor = datetime.combine(THU, time(0, 0), tzinfo=day_windows[0][0].tzinfo)
    for _ in range(24 * 60):
        expected = any(start <= cursor < end for start, end in day_windows)
        assert is_open(exchange, cursor) is expected, cursor
        cursor += timedelta(minutes=1)
