"""Minimum tick as a function of price — what a scalar ``pricetick`` cannot express.

Why a table is needed at all
----------------------------

``ContractData.pricetick`` is one number per contract, but HK's minimum spread
is a function of the price: 700.SEHK trades on a 0.010 grid at HKD 8 and on a
0.200 grid at HKD 400.  A single scalar is therefore wrong for rounding at
almost every price — and rounding is exactly what the order path does with it
(``round_to(price, contract.pricetick)``).

The concrete failure that motivated this module: ``vnpy_futu`` advertised a flat
``0.001`` placeholder (its own comment said the value "MUST NOT be trusted for
real order price rounding"), so an order for 700.SEHK near HKD 400 was rounded
to three decimals — a price that passes every local check and is refused by the
exchange.  The gate said allowed, the broker said no.

The deliberate design choices
-----------------------------

* **Refuse rather than extrapolate.** An unmapped exchange raises ``KeyError``
  and a price above HKEX's published ceiling raises ``ValueError``.  This
  mirrors ``market_clock.market_tz`` — a guessed grid is not a cosmetic error,
  it is an order priced at a level nobody chose.

* **Non-finite refuses too.** NaN makes every band comparison false; falling
  through to the coarsest tick would price an order arbitrarily.  ``inf`` is
  refused for the same reason rather than being clamped to the top band.

* **``finest_tick`` is what a contract may advertise.** Before any quote
  arrives there is no price to look a band up with, and a contract claiming the
  *current* band would call a perfectly legal price illegal the moment the
  symbol traded into a finer one.  The floor of the table is the only scalar
  that is never wrong in that direction; it is a lower bound, not the tick.

* **Rounding may cross a band edge, and that is safe here.** Every HKEX edge is
  a multiple of both adjacent ticks, so a price rounded on one grid still lands
  on a legal point of the neighbouring band.  ``test_spread_table.py`` pins that
  property across the whole table rather than trusting it.
"""

from __future__ import annotations

import math
from decimal import Decimal

from vnpy.trader.constant import Exchange

# HKEX's published spread table (HKD).  Each entry is (inclusive upper bound of
# the band, minimum spread within it); bands are (previous bound, this bound].
# Reproduced in full so a reader can diff it against the exchange's version
# without leaving the file.
_HK_TABLE: tuple[tuple[float, float], ...] = (
    (0.25, 0.001),
    (0.50, 0.005),
    (10.00, 0.010),
    (20.00, 0.020),
    (100.00, 0.050),
    (200.00, 0.100),
    (500.00, 0.200),
    (1000.00, 0.500),
    (2000.00, 1.000),
    (5000.00, 2.000),
    (9995.00, 5.000),
)

# SEC Rule 612 bans sub-penny quoting at and above USD 1.00; below that the
# minimum increment is USD 0.0001.
_US_SUB_DOLLAR_TICK: float = 0.0001
_US_TICK: float = 0.01
_US_SUB_DOLLAR_CEILING: float = 1.00

# A-share equities quote on a flat CNY 0.01 grid across SSE and SZSE (main
# board, STAR and ChiNext alike — the boards differ in price *limits*, not in
# tick). This fork is read-only on both (see vnpy_futu's stop_supported note),
# but contracts are still created for them, so the table has to answer.
_CN_TICK: float = 0.01

_TABLES: dict[Exchange, tuple[tuple[float, float], ...]] = {
    Exchange.SEHK: _HK_TABLE,
    Exchange.SMART: (
        (_US_SUB_DOLLAR_CEILING, _US_SUB_DOLLAR_TICK),
        (math.inf, _US_TICK),
    ),
    Exchange.SSE: ((math.inf, _CN_TICK),),
    Exchange.SZSE: ((math.inf, _CN_TICK),),
}

# The US table's lower band is (0, 1.00] by the same (previous, this] rule, but
# Rule 612 draws the line at "at and above USD 1.00" — so 1.00 itself trades on
# the penny grid.  Encoded as a half-open exception rather than bent into the
# generic table, because bending it would be a silent lie about the rule.
_HALF_OPEN_LOWER_BAND: frozenset[Exchange] = frozenset({Exchange.SMART})


def supported_exchanges() -> tuple[Exchange, ...]:
    """Exchanges this table can price.  Same scope as ``market_clock``."""
    return tuple(_TABLES)


def _check_price(price: float) -> None:
    try:
        finite: bool = math.isfinite(price)
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"价格非有限数值: {price!r} —— 无法确定价位档, 拒绝取整")
    if price <= 0:
        raise ValueError(f"价格必须为正数, 收到 {price}")


def price_tick(exchange: Exchange, price: float) -> float:
    """The minimum legal price increment for ``price`` on ``exchange``."""
    _check_price(price)
    try:
        table: tuple[tuple[float, float], ...] = _TABLES[exchange]
    except KeyError:
        raise KeyError(
            f"{exchange.value} 无价位表 —— 新增市场必须在 spread_table 里显式登记, "
            f"猜一个价位档会生成交易所拒绝的委托价"
        ) from None

    half_open: bool = exchange in _HALF_OPEN_LOWER_BAND
    for index, (upper, tick) in enumerate(table):
        within: bool = price < upper if (half_open and index == 0) else price <= upper
        if within:
            return tick

    raise ValueError(
        f"{exchange.value} 价格 {price} 超出价位表上限 {table[-1][0]} —— "
        f"表外区间没有权威依据, 拒绝外推"
    )


def finest_tick(exchange: Exchange) -> float:
    """The smallest tick on ``exchange`` — the only honest scalar for a contract.

    See the module docstring: this is a lower bound advertised on
    ``ContractData.pricetick``, not the tick at any particular price.  Round
    order prices with :func:`round_to_tick`, never with this.
    """
    try:
        table: tuple[tuple[float, float], ...] = _TABLES[exchange]
    except KeyError:
        raise KeyError(f"{exchange.value} 无价位表 —— 新增市场必须显式登记") from None
    return min(tick for _, tick in table)


def round_to_tick(price: float, exchange: Exchange) -> float:
    """Round ``price`` to the legal grid of the band it falls in.

    Uses the same nearest-with-Decimal arithmetic as ``vnpy.trader.utility``'s
    ``round_to``, so switching a call site from one to the other changes which
    grid is used and nothing else.
    """
    tick: float = price_tick(exchange, price)
    decimal_price: Decimal = Decimal(str(price))
    decimal_tick: Decimal = Decimal(str(tick))
    return float(int(round(decimal_price / decimal_tick)) * decimal_tick)
