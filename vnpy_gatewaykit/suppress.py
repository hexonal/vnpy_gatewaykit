"""SuppressContractMixin — a trade-only gateway that does not push its
contracts onto the event bus.

In a split quote/trade architecture (see the vnpy_router design), contracts
are supplied by ONE quote gateway (e.g. FutuGateway), and a trade gateway
pushing its own EVENT_CONTRACT only overwrites the OMS entry for a symbol
with its own (usually wrong) size/pricetick/history_data — OmsEngine keys
contracts by vt_symbol alone, last writer wins. This mixin makes an
in-house trade gateway emit no EVENT_CONTRACT at all, so the quote gateway
stays the authoritative owner in the OMS.

The contracts a trade gateway does know are still useful for order-side
validation, so an optional `contract_sink` callback receives them (the
RouterEngine passes its private trade_contracts setter here). The mixin
overrides on_contract; every other gateway callback is unchanged.
"""

from __future__ import annotations

from typing import Any, Callable

from vnpy.trader.object import ContractData


class SuppressContractMixin:
    """Mix in before BaseGateway. Overrides on_contract to NOT emit
    EVENT_CONTRACT; optionally forwards the contract to contract_sink.

        class MyTradeGateway(SuppressContractMixin, BaseGateway):
            ...

    Pass contract_sink at construction to feed the router's trade-side cache:
        MyTradeGateway(event_engine, "USMART", contract_sink=router.add_trade_contract)
    """

    def __init__(
        self,
        *args: Any,
        contract_sink: Callable[[ContractData], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)  # cooperative — chains to BaseGateway.__init__
        self._contract_sink = contract_sink

    def on_contract(self, contract: ContractData) -> None:
        # Deliberately does NOT call on_event(EVENT_CONTRACT, ...) — the quote
        # gateway owns contracts in the OMS. Feed the router's private cache
        # for order-side validation instead, if a sink was provided.
        if self._contract_sink is not None:
            self._contract_sink(contract)
