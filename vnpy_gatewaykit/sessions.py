"""Exchange → trading-session windows: *when* a market is open, in its own
local time. The time-of-day companion to market_clock.py's timezone map.

market_clock answers "what timezone did this row come from"; this module
answers "is that market open right now, and when does that change". Both
belong to the same fact — a market's clock — so they live side by side in the
shared kit rather than being re-derived per consumer. Before this module the
US session boundaries existed as bare `time(9, 30)` / `time(16, 0)` constants
inside vnpy_app's chart code, with nothing for HK at all; anything else
needing them (a recorder, a scheduler, a session-aware strategy) would have
copied them, and a copy is how two components end up disagreeing about when
the market closed.

Every window here is expressed in the *market's own* local time and is
same-day: HK spans 09:00–16:10 HKT, US 04:00–20:00 ET, neither crosses local
midnight. That is a real property of both markets (it is not true of futures,
which is one reason this fork records equities only) and it keeps every
computation a plain `datetime.combine(day, t, tzinfo=market_tz)` with no
day-rollover arithmetic. US windows resolve through ZoneInfo per date, so DST
is handled: 09:30 ET is 13:30 UTC in summer and 14:30 UTC in winter. No US
boundary falls inside the 02:00–03:00 ET DST transition hour, so no window
start is ever a nonexistent or ambiguous local time.

Trading *days* are a separate concern from trading *hours*, so they come from
an injected TradingCalendar. The default (WeekdayCalendar) knows only that
Sat/Sun are closed — public holidays look like trading days to it. That is a
deliberate, documented floor rather than a hardcoded holiday table that would
silently rot: a caller with a real source (futu's request_trading_days, an
exchange calendar file) injects it, and StaticHolidayCalendar is provided for
the file case.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Protocol, runtime_checkable

from vnpy.trader.constant import Exchange

from .market_clock import market_tz

# How far next_window_change() will scan forward before giving up. Covers the
# longest realistic closure (a Lunar New Year / Golden Week style run plus a
# weekend) without looping forever if a calendar says every day is a holiday.
MAX_SCAN_DAYS: int = 14


class SessionKind(Enum):
    """What kind of trading a window carries.

    The distinction is operational, not cosmetic: an AUCTION window matches
    orders at a single price and pushes almost no quotes, an EXTENDED window
    is thin and needs an explicit opt-in from the data feed (futu only
    delivers US pre/after-hours ticks when subscribed with extended_time=True),
    and REGULAR is the continuous session. A consumer that wants "is there
    real two-sided liquidity right now" filters on REGULAR alone.
    """

    AUCTION = "auction"
    REGULAR = "regular"
    EXTENDED = "extended"


@dataclass(frozen=True, slots=True)
class Session:
    """One window of a trading day, in the exchange's local time.

    Half-open [start, end): a session that ends at 12:00 does not include
    12:00, so abutting windows (US pre-market ending 09:30 / regular starting
    09:30) never both claim the same instant.
    """

    name: str
    start: time
    end: time
    kind: SessionKind

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError(
                f"session {self.name!r} must not be empty or cross local midnight: "
                f"{self.start} >= {self.end}"
            )


# HK cash equities (SEHK) and US cash equities (SMART) — the only two markets
# this fork trades. Times are exchange-local (HKT / ET respectively).
#
# HK: the 12:00–13:00 lunch break is a genuine no-data hole, so it is modeled
# as two separate REGULAR windows rather than one 09:30–16:00 block — a
# consumer measuring "did we miss data" must not count lunch as missing.
# US: 04:00–09:30 and 16:00–20:00 are the extended-hours windows futu serves
# with extended_time=True (vnpy_futu.gateway passes that flag for SMART).
_SESSIONS: dict[Exchange, tuple[Session, ...]] = {
    Exchange.SEHK: (
        Session("开市前竞价", time(9, 0), time(9, 30), SessionKind.AUCTION),
        Session("上午连续", time(9, 30), time(12, 0), SessionKind.REGULAR),
        Session("下午连续", time(13, 0), time(16, 0), SessionKind.REGULAR),
        Session("收市竞价", time(16, 0), time(16, 10), SessionKind.AUCTION),
    ),
    Exchange.SMART: (
        Session("盘前", time(4, 0), time(9, 30), SessionKind.EXTENDED),
        Session("正常盘", time(9, 30), time(16, 0), SessionKind.REGULAR),
        Session("盘后", time(16, 0), time(20, 0), SessionKind.EXTENDED),
    ),
}

ALL_KINDS: frozenset[SessionKind] = frozenset(SessionKind)


@runtime_checkable
class TradingCalendar(Protocol):
    """Which dates a market trades at all. Hours come from _SESSIONS above."""

    def is_trading_day(self, exchange: Exchange, day: date) -> bool: ...


class WeekdayCalendar:
    """Mon–Fri are trading days. Public holidays are NOT known.

    The honest floor. Callers that treat "market should be open" as an alarm
    condition must inject a real calendar, or a holiday will read as an
    outage. Callers that only use it to decide when to wake up are fine: a
    holiday costs one idle wake-up, not a wrong answer.
    """

    def is_trading_day(self, exchange: Exchange, day: date) -> bool:
        return day.weekday() < 5


class StaticHolidayCalendar:
    """WeekdayCalendar plus an explicit per-exchange holiday set.

    Built for the "load a JSON file of exchange closures" case. An exchange
    absent from the mapping simply has no holidays recorded.
    """

    def __init__(self, holidays: dict[Exchange, Iterable[date]] | None = None) -> None:
        self._holidays: dict[Exchange, frozenset[date]] = {
            exchange: frozenset(days) for exchange, days in (holidays or {}).items()
        }

    def is_trading_day(self, exchange: Exchange, day: date) -> bool:
        if day.weekday() >= 5:
            return False
        return day not in self._holidays.get(exchange, frozenset())

    def holidays(self, exchange: Exchange) -> frozenset[date]:
        return self._holidays.get(exchange, frozenset())


DEFAULT_CALENDAR: TradingCalendar = WeekdayCalendar()


def supported_exchanges() -> tuple[Exchange, ...]:
    """Exchanges this module has session data for."""
    return tuple(_SESSIONS)


def sessions_for(exchange: Exchange) -> tuple[Session, ...]:
    """The exchange's session windows, in chronological order.

    Raises KeyError for an unmapped exchange — same policy as market_tz: a
    caller must not silently treat an unknown market as always-closed (it
    would stop recording it) nor as always-open (it would alarm forever).
    """
    sessions = _SESSIONS.get(exchange)
    if sessions is None:
        raise KeyError(
            f"no trading sessions mapped for {exchange!r}; add it to "
            f"vnpy_gatewaykit.sessions._SESSIONS"
        )
    return sessions


def _resolve_kinds(kinds: Iterable[SessionKind] | None) -> frozenset[SessionKind]:
    return ALL_KINDS if kinds is None else frozenset(kinds)


def _local(moment: datetime, exchange: Exchange) -> datetime:
    """The instant, expressed in the exchange's local time.

    Rejects naive datetimes outright. A naive "now" is the single most common
    way this kind of code goes wrong on a machine whose timezone differs from
    the market's — the whole point of market_clock is that a bare wall-clock
    reading carries no meaning until a zone is attached.
    """
    if moment.tzinfo is None:
        raise ValueError(
            "moment must be timezone-aware (use datetime.now(timezone.utc)); "
            "a naive datetime has no defined instant to compare against a market clock"
        )
    return moment.astimezone(market_tz(exchange))


def windows(
    exchange: Exchange,
    day: date,
    *,
    kinds: Iterable[SessionKind] | None = None,
    calendar: TradingCalendar = DEFAULT_CALENDAR,
) -> tuple[tuple[datetime, datetime], ...]:
    """Merged open windows for one local trading day, as aware datetimes.

    Abutting windows of the selected kinds are merged, so asking for all kinds
    on an HK day yields 09:00–12:00 and 13:00–16:10 (auction glued to
    continuous, lunch preserved) rather than four fragments. Returns () on a
    non-trading day.
    """
    selected = _resolve_kinds(kinds)
    if not calendar.is_trading_day(exchange, day):
        return ()

    tz = market_tz(exchange)
    merged: list[list[datetime]] = []
    for session in sessions_for(exchange):
        if session.kind not in selected:
            continue
        start = datetime.combine(day, session.start, tzinfo=tz)
        end = datetime.combine(day, session.end, tzinfo=tz)
        if merged and merged[-1][1] == start:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def active_session(
    exchange: Exchange,
    moment: datetime,
    *,
    kinds: Iterable[SessionKind] | None = None,
    calendar: TradingCalendar = DEFAULT_CALENDAR,
) -> Session | None:
    """The session containing `moment`, or None when the market is closed."""
    selected = _resolve_kinds(kinds)
    local = _local(moment, exchange)
    if not calendar.is_trading_day(exchange, local.date()):
        return None

    tz = market_tz(exchange)
    for session in sessions_for(exchange):
        if session.kind not in selected:
            continue
        start = datetime.combine(local.date(), session.start, tzinfo=tz)
        end = datetime.combine(local.date(), session.end, tzinfo=tz)
        if start <= local < end:
            return session
    return None


def is_open(
    exchange: Exchange,
    moment: datetime,
    *,
    kinds: Iterable[SessionKind] | None = None,
    calendar: TradingCalendar = DEFAULT_CALENDAR,
) -> bool:
    """Whether the market is inside one of the selected windows."""
    return active_session(exchange, moment, kinds=kinds, calendar=calendar) is not None


def next_window_change(
    exchange: Exchange,
    moment: datetime,
    *,
    kinds: Iterable[SessionKind] | None = None,
    calendar: TradingCalendar = DEFAULT_CALENDAR,
    horizon_days: int = MAX_SCAN_DAYS,
) -> datetime | None:
    """The next instant strictly after `moment` at which is_open() flips.

    This is what a scheduler sleeps until: inside a window it returns that
    window's close, outside one it returns the next window's open (skipping
    weekends and whatever the calendar calls a holiday). Returns None if
    nothing changes within horizon_days — a caller should then fall back to a
    bounded poll rather than sleeping forever.
    """
    local = _local(moment, exchange)
    for offset in range(horizon_days + 1):
        day = local.date() + timedelta(days=offset)
        for start, end in windows(exchange, day, kinds=kinds, calendar=calendar):
            if start > local:
                return start
            if end > local:
                return end
    return None


def day_close(
    exchange: Exchange,
    day: date,
    *,
    kinds: Iterable[SessionKind] | None = None,
    calendar: TradingCalendar = DEFAULT_CALENDAR,
) -> datetime | None:
    """When the market finishes for that local day, or None if it never opened.

    Used to schedule post-close work (a backfill, a report) against the
    market's own clock instead of the machine's.
    """
    day_windows = windows(exchange, day, kinds=kinds, calendar=calendar)
    if not day_windows:
        return None
    return day_windows[-1][1]


def open_seconds_between(
    exchange: Exchange,
    start: datetime,
    end: datetime,
    *,
    kinds: Iterable[SessionKind] | None = None,
    calendar: TradingCalendar = DEFAULT_CALENDAR,
    max_span_days: int = 366,
) -> float:
    """Seconds the market was actually open between two instants.

    Wall-clock elapsed time is the wrong unit for "how long was there no
    data": 13 hours of silence across an HK night is normal, 13 minutes
    during the continuous session is an outage, and the hour from 12:00 to
    13:00 is lunch either way. This subtracts every closed stretch so a
    caller can threshold on open time alone.

    Returns 0.0 when end <= start. Raises ValueError past max_span_days,
    which would mean the caller is measuring something other than a gap.
    """
    if end <= start:
        return 0.0

    local_start = _local(start, exchange)
    local_end = _local(end, exchange)
    span_days = (local_end.date() - local_start.date()).days
    if span_days > max_span_days:
        raise ValueError(
            f"span of {span_days} days exceeds max_span_days={max_span_days}; "
            f"open_seconds_between is for gap/coverage measurement, not bulk history"
        )

    total = 0.0
    for offset in range(span_days + 1):
        day = local_start.date() + timedelta(days=offset)
        for window_start, window_end in windows(
            exchange, day, kinds=kinds, calendar=calendar
        ):
            overlap_start = max(window_start, local_start)
            overlap_end = min(window_end, local_end)
            if overlap_end > overlap_start:
                total += (overlap_end - overlap_start).total_seconds()
    return total


def previous_close(
    exchange: Exchange,
    moment: datetime,
    *,
    kinds: Iterable[SessionKind] | None = None,
    calendar: TradingCalendar = DEFAULT_CALENDAR,
    horizon_days: int = MAX_SCAN_DAYS,
) -> datetime | None:
    """The most recent close at or before `moment`.

    The counterpart to next_window_change for catch-up logic: "which session
    should already be complete in the database".
    """
    local = _local(moment, exchange)
    for offset in range(horizon_days + 1):
        day = local.date() - timedelta(days=offset)
        for _start, end in reversed(windows(exchange, day, kinds=kinds, calendar=calendar)):
            if end <= local:
                return end
    return None
