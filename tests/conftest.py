from __future__ import annotations

from collections.abc import Iterator

import pytest

from vnpy.event import Event, EventEngine
from vnpy.trader.constant import Exchange


@pytest.fixture()
def event_engine() -> Iterator[EventEngine]:
    engine = EventEngine()
    engine.start()
    yield engine
    engine.stop()


@pytest.fixture()
def collected_events(event_engine: EventEngine) -> list[Event]:
    from vnpy.trader.event import EVENT_LOG, EVENT_ORDER

    events: list[Event] = []

    def _collect(event: Event) -> None:
        events.append(event)

    for et in (EVENT_LOG, EVENT_ORDER):
        event_engine.register(et, _collect)

    return events


SEHK = Exchange.SEHK
