from __future__ import annotations

import time

import pytest
from vnpy.event import EventEngine
from vnpy.trader.gateway import BaseGateway

from vnpy_gatewaykit import NonBlockingConnectMixin


class _MinimalGateway(BaseGateway):
    """The bare minimum to satisfy BaseGateway's abstract methods — this
    test is only about connect()'s threading behavior, not gateway logic."""

    default_name = "MINIMAL"

    def close(self) -> None: pass
    def subscribe(self, req) -> None: pass  # noqa: ANN001
    def send_order(self, req) -> str: return ""  # noqa: ANN001
    def cancel_order(self, req) -> None: pass  # noqa: ANN001
    def query_account(self) -> None: pass
    def query_position(self) -> None: pass


class SlowGateway(NonBlockingConnectMixin, _MinimalGateway):
    def __init__(self, event_engine: EventEngine, gateway_name: str, slow_seconds: float) -> None:
        super().__init__(event_engine, gateway_name)
        self.slow_seconds = slow_seconds
        self.connected = False

    def _connect(self, setting: dict) -> None:
        time.sleep(self.slow_seconds)
        self.connected = True


class BrokenGateway(NonBlockingConnectMixin, _MinimalGateway):
    """Never overrides _connect() — exercises the NotImplementedError path."""


@pytest.fixture()
def event_engine():
    engine = EventEngine()
    yield engine


def test_connect_returns_immediately_even_when_connect_work_is_slow(
    event_engine: EventEngine,
) -> None:
    SLOW_SECONDS = 0.3
    gw = SlowGateway(event_engine, "SLOW", SLOW_SECONDS)

    started = time.monotonic()
    gw.connect({})
    elapsed = time.monotonic() - started

    assert elapsed < SLOW_SECONDS / 2, "connect() blocked the calling thread"
    assert gw.connected is False  # background work hasn't finished yet

    time.sleep(SLOW_SECONDS + 0.2)
    assert gw.connected is True  # ...but it did complete, off-thread


def test_connect_runs_on_a_background_thread_not_the_caller(event_engine: EventEngine) -> None:
    import threading

    seen_thread_names: list[str] = []

    class RecordingGateway(NonBlockingConnectMixin, _MinimalGateway):
        def _connect(self, setting: dict) -> None:
            seen_thread_names.append(threading.current_thread().name)

    gw = RecordingGateway(event_engine, "RECORD")
    gw.connect({})
    time.sleep(0.1)

    assert seen_thread_names, "_connect() never ran"
    assert seen_thread_names[0] != threading.current_thread().name


def test_subclass_must_implement_connect_or_it_raises(event_engine: EventEngine) -> None:
    gw = BrokenGateway(event_engine, "BROKEN")
    with pytest.raises(NotImplementedError, match="must implement _connect"):
        gw._connect({})
