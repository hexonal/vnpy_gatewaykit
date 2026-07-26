# vnpy_gatewaykit

Broker-agnostic plumbing for a [VeighNa (vnpy)](https://github.com/vnpy/vnpy)
fork — what every real gateway ends up re-deriving, separated out so the next
gateway package (e.g. a future `vnpy_longbridge`) doesn't copy-paste it.

## Why this exists, and what may live here

`vnpy_futu` is currently the only real gateway implementation in this
project. Everything broker-specific in it (symbol/exchange mapping,
enum conversions, the actual SDK calls) has no business being "shared" —
Futu's wire format and Longbridge's wire format are fundamentally
different, and abstracting over a sample size of one is how you end up
with the wrong abstraction. The bar for moving something here is therefore
evidence, not symmetry: the mixins were extracted from `vnpy_futu`'s working
tested code rather than designed speculatively, and the market/label modules
exist because their content is *measured* (session tables, cross-source label
comparisons) and would otherwise be re-measured, or worse re-guessed, per
gateway.

| Module | What it is |
|---|---|
| `nonblocking` | `NonBlockingConnectMixin` / `NonBlockingSubscribeMixin` — see below |
| `reject` | `RejectOrderMixin` — see below |
| `suppress` | `SuppressContractMixin`: a trade gateway in a split quote/trade setup pushes no `EVENT_CONTRACT` (contracts come from the quote gateway; see `vnpy_router`), optionally feeding a `contract_sink` instead |
| `market_clock` | Per-exchange timezone table + `localize()`: brokers stamp naive market-local time, the DB stores UTC |
| `sessions` | HK/US session windows (auction / regular / extended), holiday calendars, "when does the state next change" |
| `query_window` | Which timezone a gateway's history query bounds are expressed in |
| `tick_filter` | Dropping the non-ticks a feed emits (stale/empty snapshots) |
| `bar_label` | Measured label semantics per (source, interval) — futu's intraday bars are END-labelled, uSMART/longbridge are START — plus normalization to vnpy's START convention, `stored_label_version()`, and `LabelLedger`, the ledger that keeps a series' stored convention from silently changing under it |

`bar_label`'s normalization ships **off** (`VNPY_BAR_LABEL_NORMALIZE`): turning
it on changes new bars' timestamps, so a series stored under the old convention
must be migrated (`relabel_stored_bars`) first. `LabelLedger` is what makes the
flip safe — a writer records the convention it wrote (`stored_label_version()`)
and refuses a series stored under the other one. `vnpy_recorder` does this on
every backfill; any other writer into the same tables should too.

## `NonBlockingConnectMixin`

`BaseGateway`'s own docstring requires "all methods should be
non-blocked", but any real broker SDK's connect sequence (auth, contract
list, account/position queries) takes real synchronous network time —
and vnpy's stock `ConnectDialog` calls `main_engine.connect()` directly
from a Qt button-click slot, no thread of its own. A blocking `connect()`
freezes the whole GUI for however long that takes. This mixin wraps
`connect()` in a daemon thread; subclasses implement `_connect(self,
setting)` with the real logic instead of `connect()` itself.

## `RejectOrderMixin`

`BaseGateway.send_order()`'s docstring specifies the failure path
precisely: a rejected order still needs a locally-assigned unique
orderid, a pushed `OrderData` with `Status.REJECTED`, and a returned
`vt_orderid` — not an exception, not an empty string. `self._reject(req,
reason)` does exactly that bookkeeping.

## Usage

```python
from vnpy.trader.gateway import BaseGateway
from vnpy_gatewaykit import NonBlockingConnectMixin, RejectOrderMixin

class MyGateway(NonBlockingConnectMixin, RejectOrderMixin, BaseGateway):
    def _connect(self, setting: dict) -> None:
        ...  # the actual (potentially slow) connection logic

    def send_order(self, req: OrderRequest) -> str:
        if not self.connected:
            return self._reject(req, "not connected")
        ...
```

Mixin order matters for `RejectOrderMixin`'s cooperative `__init__` to
chain correctly to `BaseGateway.__init__` — put both mixins before
`BaseGateway` in the base list, in either relative order to each other
(neither mixin depends on the other), but always before `BaseGateway`.

## Running the tests

```bash
python -m pytest -q
```
