"""
Extracted from vnpy_futu/gateway.py's _reject() helper. BaseGateway's own
send_order() docstring specifies the failure contract precisely: "if
request is failed to sent, OrderData.status should be set to
Status.REJECTED" and "return vt_orderid" — meaning even a rejected order
needs a locally-assigned, unique orderid and a pushed OrderData, not just
an exception or an empty string. Every gateway that can fail to send an
order (i.e. every real gateway) ends up re-deriving this exact same
bookkeeping; this mixin is that logic, extracted once.
"""

from __future__ import annotations

from typing import Any

from vnpy.trader.constant import OrderType, Status
from vnpy.trader.object import OrderData, OrderRequest


class RejectOrderMixin:
    """
    Mix in before BaseGateway and call self._reject(req, reason) from
    send_order() whenever the request can't actually be sent (no
    connection, unsupported field, broker returned an error, ...):

    class MyGateway(RejectOrderMixin, BaseGateway):
        def send_order(self, req: OrderRequest) -> str:
            if not self.connected:
                return self._reject(req, "not connected")
            ...
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # cooperative — chains to BaseGateway.__init__
        self._local_reject_count = 0

    def _reject(self, req: OrderRequest, reason: str) -> str:
        self._local_reject_count += 1
        order: OrderData = req.create_order_data(
            f"local-reject-{self._local_reject_count}", self.gateway_name  # type: ignore[attr-defined]
        )
        order.status = Status.REJECTED
        self.on_order(order)  # type: ignore[attr-defined]
        self.write_log(f"委托失败({req.vt_symbol}): {reason}")  # type: ignore[attr-defined]
        return order.vt_orderid

    def reject_if_invalid(self, req: OrderRequest) -> str | None:
        """Gateway-agnostic pre-send order sanity. EVERY trading gateway
        should call this at the TOP of send_order, before any gateway-specific
        work, so new gateways inherit the guard for free:

            def send_order(self, req: OrderRequest) -> str:
                invalid = self.reject_if_invalid(req)
                if invalid is not None:
                    return invalid
                ...  # gateway-specific send

        The gateway is the last chokepoint before a broker's place_order, and
        an order's fields can come from anywhere upstream — a strategy, a
        client-side stop, a UI. The concrete hazard this closes: vnpy's
        client-side stop (vnpy_ctastrategy.check_stop_order) prices a triggered
        order from tick.ask_price_5 / bid_price_5, which are 0 on any gateway
        whose order book arrives as a push SEPARATE from the quote tick (both
        Futu and uSMART). Markets without a daily price limit leave
        tick.limit_up/limit_down at 0 too, so the 0-price branch is the norm —
        unguarded, that becomes place_order(price=0) at the live broker.

        Rejects (via _reject, pushing a proper REJECTED OrderData) any order
        that must never reach a broker: non-positive volume, or a non-positive
        price on a LIMIT / STOP order. MARKET orders legitimately carry no
        price and are not price-checked. Returns the reject vt_orderid if the
        request is invalid, else None. Costs nothing on the happy path.
        """
        if req.volume <= 0:
            return self._reject(req, f"非法委托数量 {req.volume}(须 > 0)")
        if req.type in (OrderType.LIMIT, OrderType.STOP) and req.price <= 0:
            return self._reject(req, f"非法委托价 {req.price}(限价/止损单价须 > 0)")
        return None
