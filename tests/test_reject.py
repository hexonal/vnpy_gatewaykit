from __future__ import annotations

from vnpy.event import EventEngine
from vnpy.trader.constant import Direction, Exchange, Offset, OrderType, Status
from vnpy.trader.gateway import BaseGateway
from vnpy.trader.object import OrderRequest

from vnpy_gatewaykit import RejectOrderMixin


class _RejectableGateway(RejectOrderMixin, BaseGateway):
    default_name = "REJECTABLE"

    def connect(self, setting: dict) -> None: pass
    def close(self) -> None: pass
    def subscribe(self, req) -> None: pass  # noqa: ANN001
    def send_order(self, req: OrderRequest) -> str:
        return self._reject(req, "test: always unreachable")
    def cancel_order(self, req) -> None: pass  # noqa: ANN001
    def query_account(self) -> None: pass
    def query_position(self) -> None: pass


def _make_req() -> OrderRequest:
    return OrderRequest(
        symbol="700",
        exchange=Exchange.SEHK,
        direction=Direction.LONG,
        type=OrderType.LIMIT,
        volume=100,
        price=300.0,
        offset=Offset.OPEN,
    )


def test_reject_returns_a_valid_vt_orderid() -> None:
    gw = _RejectableGateway(EventEngine(), "REJECTABLE")
    vt_orderid = gw.send_order(_make_req())
    assert vt_orderid == "REJECTABLE.local-reject-1"


def test_reject_pushes_order_data_with_rejected_status() -> None:
    event_engine = EventEngine()
    event_engine.start()
    try:
        gw = _RejectableGateway(event_engine, "REJECTABLE")

        pushed = []
        from vnpy.trader.event import EVENT_ORDER

        event_engine.register(EVENT_ORDER, lambda e: pushed.append(e.data))

        gw.send_order(_make_req())
        import time

        time.sleep(0.1)

        assert len(pushed) == 1
        assert pushed[0].status == Status.REJECTED
        assert pushed[0].symbol == "700"
    finally:
        event_engine.stop()


def test_reject_count_increments_across_calls() -> None:
    gw = _RejectableGateway(EventEngine(), "REJECTABLE")

    first = gw.send_order(_make_req())
    second = gw.send_order(_make_req())

    assert first == "REJECTABLE.local-reject-1"
    assert second == "REJECTABLE.local-reject-2"


def test_reject_count_is_independent_per_gateway_instance() -> None:
    gw1 = _RejectableGateway(EventEngine(), "GW1")
    gw2 = _RejectableGateway(EventEngine(), "GW2")

    gw1.send_order(_make_req())
    gw1.send_order(_make_req())
    only_call = gw2.send_order(_make_req())

    assert only_call == "GW2.local-reject-1"
