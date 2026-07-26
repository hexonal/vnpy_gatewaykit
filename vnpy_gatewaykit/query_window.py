"""Turn a database query bound into an instant that means the same thing
everywhere — the read-path counterpart to :mod:`vnpy_gatewaykit.market_clock`.

Why this file exists
--------------------
Every database driver funnels query bounds through
``vnpy.trader.database.convert_tz``, which calls ``datetime.astimezone()``.
On a *naive* datetime that method reads the value as the **host's** local
zone. So ``datetime(2024, 1, 26)`` is a different instant on a US-Pacific
laptop than on an HK server, and a window that should hold N bars silently
holds fewer: the host's UTC offset slides both bounds off the bar timestamps
at the edges.

Measured on this project's QuestDB (host in US Eastern, ``database.timezone``
= UTC), a 700-bar daily series queried with naive bounds returned 699 bars
from a single call, 693 through ``BacktestingEngine.load_data``, and 0 from a
naive single-day window over a day that has a bar.

What a naive bound should mean
------------------------------
The market's wall clock. "Backtest 700.SEHK from 2024-01-26", "replay the
2024-01-26 session", "show me 2024-01-26 in the data manager" all name the
Hong Kong session of that date; nobody types a bound meaning "midnight
wherever this process happens to run". :func:`market_clock.market_tz` is this
project's single source of truth for exchange -> timezone, and it is the same
clock the stored bars carry: gateways attach it to the feed's naive
wall-clock string on the way in, so attaching it to the query bound on the way
out puts both sides on one clock.

Why it lives in gatewaykit
--------------------------
Three packages read bars back out of the database with user-typed bounds —
``vnpy_ctastrategy`` (backtest windows), ``vnpy_replay`` (replay windows) and
``vnpy_app`` (the data manager and the replay chart). All three already depend
on this kit for :mod:`market_clock`; none of them should have to depend on
each other for a timezone reading. Keeping the write path (``market_clock``,
which localizes an incoming feed row) and the read path (this module, which
localizes an outgoing query bound) side by side in one package is what stops
the two from drifting apart.

The two paths differ in exactly one way, deliberately. ``market_clock.localize``
*raises* for an exchange it cannot place: a gateway that guesses would write
wrong-UTC rows, corrupting storage. A query bound has no such stake — a wrong
guess returns the wrong rows for one call and nothing persists — so exchanges
``market_clock`` does not map (upstream's CFFEX/SHFE/... users) fall back to
``DB_TZ``, i.e. the configured ``database.timezone``. That is a declared
setting rather than whatever zone the host sits in, so the reading is still
reproducible across machines; on a stock install where ``database.timezone``
is left at the host zone it also reproduces the previous behaviour exactly,
which keeps this a fix rather than a silent semantic change for markets we do
not map.

What this module is *not* for
-----------------------------
An open-ended bound — "up to now", "everything after X" — is an **instant**,
not a wall clock. ``datetime.now()`` is naive but already means the present
moment; handing it here would re-read it as the market's clock and shove it
hours into the past. Callers with an open end must build an aware "now"
(``datetime.now(query_tz(exchange))``) instead. Aware input is returned
untouched precisely so such a bound survives a defensive call unharmed.
"""

from __future__ import annotations

from datetime import datetime, tzinfo

from vnpy.trader.constant import Exchange
from vnpy.trader.database import DB_TZ

from .market_clock import market_tz


def query_tz(exchange: Exchange) -> tzinfo:
    """The zone a naive query bound for ``exchange`` is written in."""
    try:
        return market_tz(exchange)
    except KeyError:
        return DB_TZ


def localize_bound(moment: datetime, exchange: Exchange) -> datetime:
    """Make a query bound explicit about which instant it names.

    An already-aware bound is returned unchanged: it names one instant
    already, and reinterpreting it would silently move a caller's window.
    """
    if moment.tzinfo is not None:
        return moment
    return moment.replace(tzinfo=query_tz(exchange))
