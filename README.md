# vnpy_gatewaykit

Broker-agnostic mixins extracted from `vnpy_futu` for a
[VeighNa (vnpy)](https://github.com/vnpy/vnpy) fork — the plumbing every
real gateway ends up re-deriving, separated out so the next gateway
package (e.g. a future `vnpy_longbridge`) doesn't copy-paste it.

## Why this exists, and why it's small

`vnpy_futu` is currently the only real gateway implementation in this
project. Everything broker-specific in it (symbol/exchange mapping,
enum conversions, the actual SDK calls) has no business being "shared" —
Futu's wire format and Longbridge's wire format are fundamentally
different, and abstracting over a sample size of one is how you end up
with the wrong abstraction. What *did* turn out to be broker-agnostic —
verified by literally extracting it out of `vnpy_futu`'s working,
tested code, not designed speculatively — is exactly the two mixins
below. Nothing else moved here on a guess.

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
