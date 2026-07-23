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

from vnpy.trader.constant import Status
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
