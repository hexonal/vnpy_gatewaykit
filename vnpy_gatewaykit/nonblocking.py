"""
Extracted from vnpy_futu/gateway.py's connect() fix (see that package's
README, "connect() is non-blocking"). BaseGateway's own docstring requires
"all methods should be non-blocked", but a real broker SDK's connect
sequence — auth, contract list, account/position queries — routinely
takes several seconds of synchronous network I/O. If a GUI's
ConnectDialog calls connect() directly from a Qt button-click slot (which
vnpy's stock ConnectDialog does), a blocking connect() freezes the whole
window for that duration. This mixin is the fix, generalized so any
future gateway package can reuse it instead of re-deriving the same
threading.Thread wrapper from scratch.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only: this kit stays importable without pulling vnpy at runtime,
    # and `from __future__ import annotations` keeps the annotations lazy.
    from vnpy.trader.object import SubscribeRequest


class NonBlockingConnectMixin:
    """
    Mix in before BaseGateway (or any class providing send_order/on_order/
    etc.) and implement _connect(self, setting) instead of connect() —
    connect() itself is provided here and should not be overridden.

    class MyGateway(NonBlockingConnectMixin, BaseGateway):
        def _connect(self, setting: dict[str, Any]) -> None:
            ...  # the actual (potentially slow) connection logic
    """

    def connect(self, setting: dict[str, Any]) -> None:
        threading.Thread(target=self._connect, args=(setting,), daemon=True).start()

    def _connect(self, setting: dict[str, Any]) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _connect(self, setting) "
            f"— NonBlockingConnectMixin only provides the non-blocking connect() wrapper."
        )


class NonBlockingSubscribeMixin:
    """
    Same rationale as NonBlockingConnectMixin, for subscribe(). A broker SDK's
    subscribe is a synchronous request/response round-trip to the daemon
    (futu OpenD, uSMART WS, …). subscribe() is routinely called from a Qt slot
    on the GUI thread — the ChartWizard subscribes the charted symbol in its
    EVENT_CHART_HISTORY handler (which runs on the GUI thread via a queued
    signal), and re-subscribes on every period switch; the trading/tick-
    monitor widgets subscribe on user input. A blocking subscribe there
    freezes the window for the round-trip — exactly the "GUI hangs when
    pulling data" symptom. Off-loading to a daemon thread keeps the GUI
    responsive; the broker SDK's own request context serializes concurrent
    calls, so thread-per-subscribe is safe.

    class MyGateway(NonBlockingSubscribeMixin, BaseGateway):
        def _subscribe(self, req: SubscribeRequest) -> None:
            ...  # the actual (blocking) subscribe call

    subscribe() is provided here and should not be overridden.
    """

    # Typed as SubscribeRequest, not object: a gateway's _subscribe naturally
    # takes SubscribeRequest, and narrowing a parameter in an override violates
    # substitutability — every gateway using this mixin was flagged for it.
    # This is also the true contract; BaseGateway.subscribe is declared the same
    # way, so the mixin now lines up with what it is standing in for.
    def subscribe(self, req: SubscribeRequest) -> None:
        threading.Thread(target=self._subscribe, args=(req,), daemon=True).start()

    def _subscribe(self, req: SubscribeRequest) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _subscribe(self, req) "
            f"— NonBlockingSubscribeMixin only provides the non-blocking subscribe() wrapper."
        )
