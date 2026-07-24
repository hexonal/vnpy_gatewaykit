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


def _make_req(
    *, type: OrderType = OrderType.LIMIT, volume: float = 100, price: float = 300.0
) -> OrderRequest:
    return OrderRequest(
        symbol="700",
        exchange=Exchange.SEHK,
        direction=Direction.LONG,
        type=type,
        volume=volume,
        price=price,
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


# --- reject_if_invalid: the shared pre-send order sanity guard ---

def test_reject_if_invalid_passes_a_good_limit_order() -> None:
    gw = _RejectableGateway(EventEngine(), "REJECTABLE")
    assert gw.reject_if_invalid(_make_req(price=300.0, volume=100)) is None


def test_reject_if_invalid_rejects_zero_price_limit() -> None:
    # The client-side-stop price-0 hazard: a LIMIT/STOP at price 0 must never
    # reach a broker.
    gw = _RejectableGateway(EventEngine(), "REJECTABLE")
    result = gw.reject_if_invalid(_make_req(type=OrderType.LIMIT, price=0.0))
    assert result == "REJECTABLE.local-reject-1"


def test_reject_if_invalid_rejects_zero_price_stop() -> None:
    gw = _RejectableGateway(EventEngine(), "REJECTABLE")
    result = gw.reject_if_invalid(_make_req(type=OrderType.STOP, price=0.0))
    assert result is not None


def test_reject_if_invalid_rejects_negative_price() -> None:
    gw = _RejectableGateway(EventEngine(), "REJECTABLE")
    assert gw.reject_if_invalid(_make_req(price=-5.0)) is not None


def test_reject_if_invalid_rejects_nonpositive_volume() -> None:
    gw = _RejectableGateway(EventEngine(), "REJECTABLE")
    assert gw.reject_if_invalid(_make_req(volume=0)) is not None
    assert gw.reject_if_invalid(_make_req(volume=-10)) is not None


def test_reject_if_invalid_allows_market_order_without_price() -> None:
    # MARKET orders legitimately carry no price — must NOT be price-rejected.
    gw = _RejectableGateway(EventEngine(), "REJECTABLE")
    assert gw.reject_if_invalid(_make_req(type=OrderType.MARKET, price=0.0, volume=100)) is None
