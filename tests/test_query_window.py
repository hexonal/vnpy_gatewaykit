"""``query_window`` — what a naive database query bound means.

The read-path counterpart to ``market_clock``: gateways attach the market's
zone to an incoming feed row, and this attaches the same zone to an outgoing
query bound, so both sides of the database sit on one clock. Without it,
``vnpy.trader.database.convert_tz`` reads a naive bound as the *host's* zone
and a window typed as exchange dates slides by the machine's UTC offset.

The two policies differ on purpose and both are pinned here: an unmapped
exchange makes ``market_clock`` raise (a gateway must not guess which clock it
is writing), while a query bound falls back to the configured
``database.timezone`` (a wrong guess costs one call's rows, nothing persists).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest
from vnpy.trader.constant import Exchange
from vnpy.trader.database import DB_TZ

import vnpy_gatewaykit
from vnpy_gatewaykit.query_window import localize_bound, query_tz

HK_TZ = ZoneInfo("Asia/Hong_Kong")
NY_TZ = ZoneInfo("America/New_York")

# One aware moment to derive everything from — the naive bounds below are this
# same wall clock with the zone stripped, which is exactly what a user types.
HK_MOMENT = datetime(2024, 1, 26, 9, 30, tzinfo=HK_TZ)
NAIVE_MOMENT = HK_MOMENT.replace(tzinfo=None)

# Upstream markets this fork's market_clock does not map. Exchange.CFFEX is a
# real vnpy exchange, so the fallback path is reachable by real users.
UNMAPPED = Exchange.CFFEX


def test_naive_bound_is_read_as_the_exchanges_wall_clock() -> None:
    localized: datetime = localize_bound(NAIVE_MOMENT, Exchange.SEHK)

    assert localized.utcoffset() == HK_TZ.utcoffset(NAIVE_MOMENT)


def test_localizing_keeps_the_wall_clock_it_was_given() -> None:
    """Attach a zone, do not shift the numbers: 09:30 stays 09:30."""
    localized: datetime = localize_bound(NAIVE_MOMENT, Exchange.SEHK)

    assert localized.replace(tzinfo=None) == NAIVE_MOMENT


def test_each_exchange_gets_its_own_clock() -> None:
    """The same typed date is a different instant in Hong Kong and New York."""
    hk: datetime = localize_bound(NAIVE_MOMENT, Exchange.SEHK)
    ny: datetime = localize_bound(NAIVE_MOMENT, Exchange.SMART)

    assert hk.utcoffset() == HK_TZ.utcoffset(NAIVE_MOMENT)
    assert ny.utcoffset() == NY_TZ.utcoffset(NAIVE_MOMENT)
    assert hk != ny


def test_aware_bound_is_returned_untouched() -> None:
    """An explicit instant names one moment; reinterpreting it would move it.

    This is what lets a caller with an open-ended window build an aware "now"
    and hand it through defensively without the bound being dragged into the
    market's wall clock.
    """
    moment: datetime = datetime(2024, 1, 26, 9, 30, tzinfo=timezone.utc)

    assert localize_bound(moment, Exchange.SEHK) is moment


def test_unmapped_exchange_falls_back_to_the_configured_database_zone() -> None:
    """Upstream markets still get a *declared* zone, not the host's."""
    assert query_tz(UNMAPPED) is DB_TZ
    assert localize_bound(NAIVE_MOMENT, UNMAPPED).tzinfo is DB_TZ


def test_mapped_exchange_uses_the_market_clock_table() -> None:
    assert query_tz(Exchange.SEHK) is HK_TZ


def test_helpers_are_exported_from_the_package_root() -> None:
    """Callers in the other packages import these; keep them on the surface."""
    assert vnpy_gatewaykit.localize_bound is localize_bound
    assert vnpy_gatewaykit.query_tz is query_tz
    assert {"localize_bound", "query_tz"} <= set(vnpy_gatewaykit.__all__)


@pytest.fixture
def pacific_host(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.skipif(
    not hasattr(time, "tzset"),
    reason="test pins the host timezone via TZ, which needs time.tzset (POSIX)",
)
def test_reading_does_not_depend_on_the_host_zone(pacific_host: None) -> None:
    """The whole point: two machines must read one typed date the same way.

    Both the mapped and the fallback branch are checked, because the fallback
    is the one that could quietly reintroduce host-dependence if it ever went
    back to leaving the bound naive.
    """
    assert localize_bound(NAIVE_MOMENT, Exchange.SEHK).utcoffset() == HK_TZ.utcoffset(
        NAIVE_MOMENT
    )
    assert localize_bound(NAIVE_MOMENT, UNMAPPED).utcoffset() == DB_TZ.utcoffset(
        NAIVE_MOMENT
    )
