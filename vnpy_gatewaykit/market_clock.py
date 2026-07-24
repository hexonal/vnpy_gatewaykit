"""Exchange → market timezone: the single source of truth every gateway uses
to localize the naive wall-clock timestamps its data feed reports.

Broker feeds (Futu, uSMART, …) stamp bars and ticks in the *market's own
local time* with no timezone attached — "10:00" on an HK bar means 10:00 in
Hong Kong, on a US bar 10:00 in New York. A machine running the app is often
in a different timezone (this project runs US Pacific/Eastern while trading
HK/US/CN), and vnpy's storage layer (`vnpy.trader.database.convert_tz`)
interprets a naive datetime as *machine-local*. So a naive HK 10:00 bar gets
written to the database as if it were 10:00 machine-time — a ~12h-wrong UTC,
and HK/US bars that are really 12h apart collide in stored UTC.

The fix is to attach the correct market timezone at the gateway boundary
(the one place that knows which market a row came from), making every
downstream datetime tz-aware. `convert_tz` then converts to true UTC, and
display (`strftime`) still shows the same market wall-clock because the aware
datetime carries the market's own time. Keeping the map here — in the shared
kit, not per-gateway — means every gateway localizes identically and a new
gateway inherits correct time handling for free.

ZoneInfo (not a fixed offset) is deliberate: America/New_York observes DST,
so the UTC offset depends on the date; ZoneInfo resolves that per-datetime.
Asia/Hong_Kong and Asia/Shanghai have no DST but are named for consistency.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from vnpy.trader.constant import Exchange

# Exchange → IANA timezone name. Extend as new markets are added; an unmapped
# exchange raises in market_tz rather than silently leaving data naive (which
# would reintroduce the wrong-UTC bug this module exists to prevent).
_MARKET_TZ_NAME: dict[Exchange, str] = {
    Exchange.SEHK: "Asia/Hong_Kong",
    Exchange.SMART: "America/New_York",
    Exchange.SSE: "Asia/Shanghai",
    Exchange.SZSE: "Asia/Shanghai",
}

# ZoneInfo instances are immutable and cheap to share; build once.
_MARKET_TZ: dict[Exchange, ZoneInfo] = {
    exchange: ZoneInfo(name) for exchange, name in _MARKET_TZ_NAME.items()
}


def market_tz(exchange: Exchange) -> ZoneInfo:
    """The timezone the given exchange stamps its data in. Raises KeyError for
    an unmapped exchange — a gateway must not localize data it can't place."""
    tz = _MARKET_TZ.get(exchange)
    if tz is None:
        raise KeyError(
            f"no market timezone mapped for {exchange!r}; add it to "
            f"vnpy_gatewaykit.market_clock._MARKET_TZ_NAME"
        )
    return tz


def localize(dt: datetime, exchange: Exchange) -> datetime:
    """Attach the exchange's market timezone to a naive datetime parsed from a
    feed's wall-clock string. If dt is already tz-aware it's returned
    unchanged (idempotent — never silently reinterprets an aware instant)."""
    if dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=market_tz(exchange))
