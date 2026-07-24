from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from vnpy.trader.constant import Exchange

from vnpy_gatewaykit import localize, market_tz


def test_market_tz_maps_known_exchanges() -> None:
    assert market_tz(Exchange.SEHK).key == "Asia/Hong_Kong"
    assert market_tz(Exchange.SMART).key == "America/New_York"
    assert market_tz(Exchange.SSE).key == "Asia/Shanghai"
    assert market_tz(Exchange.SZSE).key == "Asia/Shanghai"


def test_market_tz_raises_for_unmapped_exchange() -> None:
    with pytest.raises(KeyError):
        market_tz(Exchange.CFFEX)  # not a market this kit's gateways serve


def test_localize_attaches_market_tz_to_naive() -> None:
    naive = datetime(2026, 7, 24, 10, 0, 0)  # HK feed reports "10:00" naive
    aware = localize(naive, Exchange.SEHK)
    assert aware.tzinfo is not None
    assert aware.tzinfo.key == "Asia/Hong_Kong"
    # Wall clock preserved — 10:00 stays 10:00, now tagged HKT.
    assert aware.hour == 10 and aware.minute == 0


def test_localize_is_idempotent_on_aware() -> None:
    already = datetime(2026, 7, 24, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    out = localize(already, Exchange.SEHK)  # must NOT reinterpret
    assert out is already or out == already
    assert out.tzinfo.key == "America/New_York"


def test_localized_datetime_yields_correct_utc() -> None:
    # The whole point: an HK 10:00 bar must become the correct UTC instant
    # (HKT = UTC+8 → 02:00 UTC), not be read as machine-local time.
    aware = localize(datetime(2026, 7, 24, 10, 0, 0), Exchange.SEHK)
    utc = aware.astimezone(ZoneInfo("UTC"))
    assert (utc.hour, utc.minute) == (2, 0)
    assert utc.date() == aware.date()


def test_us_localization_respects_dst() -> None:
    # July → EDT (UTC-4): 10:00 ET = 14:00 UTC.
    summer = localize(datetime(2026, 7, 24, 10, 0, 0), Exchange.SMART)
    assert summer.astimezone(ZoneInfo("UTC")).hour == 14
    # January → EST (UTC-5): 10:00 ET = 15:00 UTC.
    winter = localize(datetime(2026, 1, 24, 10, 0, 0), Exchange.SMART)
    assert winter.astimezone(ZoneInfo("UTC")).hour == 15
