"""HK/US minimum-tick lookup — the cases a single scalar ``pricetick`` gets wrong.

The reason this table exists at all is that ``ContractData.pricetick`` is one
number per contract while HK's tick is a function of price: the same symbol
trades on a 0.010 grid at HKD 8 and a 0.200 grid at HKD 400.  Rounding an order
price with the contract's scalar therefore produces a price that passes every
local check and is refused by the exchange.

The tests below pin three things: the band edges (where an off-by-one lands on
an illegal grid), that rounding never leaves the legal grid even when it crosses
a band boundary, and that an unmapped exchange raises rather than guessing.
"""

from __future__ import annotations

import pytest
from vnpy.trader.constant import Exchange

# Reaching into market_clock's private table on purpose: the invariant under
# test is "these two tables agree", and exporting an accessor only so a test
# can read it would put API surface on the wrong side of the assertion.
from vnpy_gatewaykit.market_clock import _MARKET_TZ_NAME
from vnpy_gatewaykit.spread_table import (
    finest_tick,
    price_tick,
    round_to_tick,
    supported_exchanges,
)


# ---------------------------------------------------------------------------
# Table sanity
# ---------------------------------------------------------------------------
def test_coverage_matches_market_clock_not_sessions() -> None:
    """Contracts are created for every market ``market_tz`` knows, so this table
    has to answer for all of them.

    ``sessions`` covers only HK+US because we trade only those; but ``vnpy_futu``
    is read-only on SSE/SZSE and still pushes ContractData for them, and a
    contract needs a ``pricetick``. Covering fewer exchanges than
    ``market_clock`` makes ``connect()`` raise KeyError mid-contract-push.
    """
    assert set(supported_exchanges()) == set(_MARKET_TZ_NAME)


def test_a_share_grid_is_a_flat_cent() -> None:
    for exchange in (Exchange.SSE, Exchange.SZSE):
        assert price_tick(exchange, 12.34) == pytest.approx(0.01)
        assert finest_tick(exchange) == pytest.approx(0.01)


def test_unmapped_exchange_raises_instead_of_guessing() -> None:
    """market_tz's fail-closed bias: refuse rather than invent a grid.

    A wrong tick is not a cosmetic problem — it is an order the broker refuses,
    or worse one it accepts at a price nobody intended.
    """
    with pytest.raises(KeyError, match="CFFEX"):
        price_tick(Exchange.CFFEX, 10.0)


# ---------------------------------------------------------------------------
# HKEX band table
#
# Official HKEX spread table, reproduced here so a reader can diff it against
# the exchange's published version without leaving the file.  Bands are
# (lower, upper] except the first, which includes its lower bound.
# ---------------------------------------------------------------------------
HK_BANDS: tuple[tuple[float, float], ...] = (
    (0.01, 0.001),
    (0.25, 0.001),      # upper edge of band 1 still trades on 0.001
    (0.251, 0.005),
    (0.50, 0.005),
    (0.501, 0.010),
    (10.00, 0.010),
    (10.01, 0.020),
    (20.00, 0.020),
    (20.01, 0.050),
    (100.00, 0.050),
    (100.01, 0.100),
    (200.00, 0.100),
    (200.01, 0.200),
    (500.00, 0.200),
    (500.01, 0.500),
    (1000.00, 0.500),
    (1000.01, 1.000),
    (2000.00, 1.000),
    (2000.01, 2.000),
    (5000.00, 2.000),
    (5000.01, 5.000),
    (9995.00, 5.000),
)


def test_hk_spread_table_matches_hkex() -> None:
    for price, expected in HK_BANDS:
        assert price_tick(Exchange.SEHK, price) == pytest.approx(expected), price


def test_tencent_around_400_is_two_tenths_not_a_thousandth() -> None:
    """The concrete case that motivated this module.

    700.SEHK near HKD 400 sits in the (200, 500] band, so the legal grid is
    0.200.  The flat 0.001 placeholder previously used for rounding produced
    prices like 400.123 — three illegal decimals.
    """
    assert price_tick(Exchange.SEHK, 400.0) == pytest.approx(0.200)


# ---------------------------------------------------------------------------
# US
# ---------------------------------------------------------------------------
def test_us_tick_is_a_cent_at_or_above_one_dollar() -> None:
    """SEC Rule 612: sub-penny quoting is prohibited at and above USD 1.00."""
    assert price_tick(Exchange.SMART, 1.00) == pytest.approx(0.01)
    assert price_tick(Exchange.SMART, 42.37) == pytest.approx(0.01)


def test_us_tick_is_a_hundredth_of_a_cent_below_one_dollar() -> None:
    assert price_tick(Exchange.SMART, 0.9999) == pytest.approx(0.0001)


# ---------------------------------------------------------------------------
# finest_tick — what a contract may honestly advertise before any quote
# ---------------------------------------------------------------------------
def test_finest_tick_is_the_smallest_band_so_it_never_calls_a_legal_price_illegal() -> None:
    """``ContractData.pricetick`` is one scalar; the only safe scalar is the floor.

    A contract advertising the *current* band would call a perfectly legal
    price illegal as soon as the symbol traded into a finer band.
    """
    assert finest_tick(Exchange.SEHK) == pytest.approx(0.001)
    assert finest_tick(Exchange.SMART) == pytest.approx(0.0001)


# ---------------------------------------------------------------------------
# round_to_tick
# ---------------------------------------------------------------------------
def test_rounding_lands_on_the_band_grid() -> None:
    assert round_to_tick(400.123, Exchange.SEHK) == pytest.approx(400.2)
    assert round_to_tick(400.09, Exchange.SEHK) == pytest.approx(400.0)
    assert round_to_tick(8.004, Exchange.SEHK) == pytest.approx(8.00)


def test_rounding_across_a_band_edge_still_lands_on_a_legal_price() -> None:
    """20.01 rounds on the (20,100] grid to 20.00, which lives in (10,20].

    Every HKEX band edge is a multiple of both adjacent ticks, so this is safe
    by construction — the test pins that property rather than trusting it.
    """
    rounded: float = round_to_tick(20.01, Exchange.SEHK)
    assert rounded == pytest.approx(20.00)
    assert price_tick(Exchange.SEHK, rounded) == pytest.approx(0.020)
    assert round(rounded / 0.020) * 0.020 == pytest.approx(rounded)


# (edge, tick below the edge, tick above it) — the actual band boundaries, not
# the probe prices in HK_BANDS above.
HK_EDGES: tuple[tuple[float, float, float], ...] = (
    (0.25, 0.001, 0.005),
    (0.50, 0.005, 0.010),
    (10.00, 0.010, 0.020),
    (20.00, 0.020, 0.050),
    (100.00, 0.050, 0.100),
    (200.00, 0.100, 0.200),
    (500.00, 0.200, 0.500),
    (1000.00, 0.500, 1.000),
    (2000.00, 1.000, 2.000),
    (5000.00, 2.000, 5.000),
)


def test_every_band_edge_is_a_multiple_of_both_adjacent_ticks() -> None:
    """The property the previous test relies on, checked across the whole table.

    If HKEX ever publishes an edge that breaks it, rounding at that edge could
    land on a price legal in neither band — this test is the tripwire.
    """
    for edge, below, above in HK_EDGES:
        for tick in (below, above):
            quotient: float = edge / tick
            assert quotient == pytest.approx(round(quotient)), (edge, tick)


def test_rounding_a_us_price_uses_the_cent_grid() -> None:
    assert round_to_tick(42.3749, Exchange.SMART) == pytest.approx(42.37)


def test_non_finite_price_is_refused_rather_than_rounded() -> None:
    """NaN makes every band comparison false; a silent fallthrough to the
    coarsest tick would price an order at an arbitrary level."""
    with pytest.raises(ValueError, match="非有限"):
        round_to_tick(float("nan"), Exchange.SEHK)
    with pytest.raises(ValueError, match="非有限"):
        price_tick(Exchange.SEHK, float("inf"))


def test_non_positive_price_is_refused() -> None:
    with pytest.raises(ValueError, match="必须为正"):
        price_tick(Exchange.SEHK, 0.0)


def test_price_above_the_published_table_refuses_instead_of_extrapolating() -> None:
    """HKEX publishes up to 9,995.00. Above that we have no authority to guess."""
    with pytest.raises(ValueError, match="超出"):
        price_tick(Exchange.SEHK, 12_000.0)
