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


class NonBlockingConnectMixin:
    """
    Mix in before BaseGateway (or any class providing send_order/on_order/
    etc.) and implement _connect(self, setting) instead of connect() —
    connect() itself is provided here and should not be overridden.

    class MyGateway(NonBlockingConnectMixin, BaseGateway):
        def _connect(self, setting: dict) -> None:
            ...  # the actual (potentially slow) connection logic
    """

    def connect(self, setting: dict) -> None:
        threading.Thread(target=self._connect, args=(setting,), daemon=True).start()

    def _connect(self, setting: dict) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _connect(self, setting) "
            f"— NonBlockingConnectMixin only provides the non-blocking connect() wrapper."
        )
