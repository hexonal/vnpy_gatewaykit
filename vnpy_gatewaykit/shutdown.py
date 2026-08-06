"""
Take SIGINT back from the futu SDK, and give SIGTERM/SIGHUP the same exit path.

Why this module has to exist
----------------------------
``import futu`` rewrites the process-wide SIGINT disposition at *import* time,
unconditionally, from the main thread (site-packages/futu/__init__.py:126-132)::

    def quit_handler(sig, frame):
        os._exit(0)

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, quit_handler)

``os._exit`` skips ``finally`` blocks, skips ``atexit``, and exits **0**.
Measured end to end on this machine: a script that imports futu, raises SIGINT
at itself and has ``finally: print("ran")`` prints nothing and returns 0.

For a trading entry point that is the worst possible combination. In
``vnpy_app/run_live_alpha.py`` the ``finally`` block is ``main_engine.close()``,
which is what reaches ``AlphaLiveEngine.close()`` -> ``cancel_working_orders_on_exit()``.
So Ctrl-C leaves broker-side limit orders working with nobody watching them —
precisely the scenario that ``close()`` was written to prevent — and the exit
code says the run finished cleanly, so a supervisor neither restarts nor alerts.

SIGTERM and SIGHUP are a **separate** cause with the same symptom: no entry point
in this workspace installs a handler for them, so they take CPython's default and
terminate the process outright. Closing a tmux window sends SIGHUP, which is as
routine as Ctrl-C. Fixing only SIGINT would leave that half broken, so both are
handled here.

Why the handler raises KeyboardInterrupt instead of setting a flag
------------------------------------------------------------------
The call sites already have the machinery: ``try/finally`` around the engine and
``except KeyboardInterrupt`` around the sleep. A flag would need every loop to
poll it, and the loops that matter are the ones blocked in ``time.sleep`` — which
is exactly what a raising handler interrupts for free. Turning SIGTERM into a
KeyboardInterrupt is deliberate: "the operator wants us to stop" is one event, and
it deserves one exit path, not two that drift apart.

Why the second signal is not polite
-----------------------------------
Graceful shutdown here means cancelling orders against a broker, which can block.
An operator who presses Ctrl-C twice has decided they want out now, and a handler
that keeps politely raising KeyboardInterrupt would make the process unkillable by
the one key everybody reaches for. So the second signal restores the default
disposition and re-raises: the OS kills it, with the conventional 128+signum code.

The install must happen after `import futu`
-------------------------------------------
futu grabs SIGINT while its module body executes. Anything installed before that
is silently overwritten, and the overwrite leaves no trace — the process just
stops honouring ``finally`` again. ``verify_owns_signals`` exists so that this
ordering error fails loudly instead of at 3am on a live account.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from collections.abc import Callable
from types import FrameType

#: Signals that mean "the operator wants this process to stop". All three are
#: funnelled into KeyboardInterrupt so one ``finally`` covers them.
#:
#: SIGHUP is absent on Windows; resolved at import so the tuple only ever holds
#: signals this platform actually has.
SHUTDOWN_SIGNALS: tuple[signal.Signals, ...] = tuple(
    s for s in (
        getattr(signal, "SIGINT", None),
        getattr(signal, "SIGTERM", None),
        getattr(signal, "SIGHUP", None),
    ) if s is not None
)

#: Set once a shutdown signal has been seen. The second one bypasses graceful
#: shutdown entirely — see the module docstring.
_shutting_down = threading.Event()


def _log(message: str) -> None:
    """Default sink. stderr, not the vnpy log engine.

    A signal can arrive while the event engine is already tearing down, and a
    log call that reaches a dead queue would raise *inside the handler* — which
    replaces a clean shutdown with a traceback and an unpredictable exit code.
    stderr is always there.
    """
    print(message, file=sys.stderr, flush=True)


def _handler(log: Callable[[str], None]) -> Callable[[int, FrameType | None], None]:
    def handle(signum: int, _frame: FrameType | None) -> None:
        name = signal.Signals(signum).name
        if _shutting_down.is_set():
            # Second one: stop being graceful. Restoring the default and
            # re-raising (rather than os._exit) keeps the conventional
            # 128+signum exit code, so whatever supervises this can still tell
            # "killed" from "finished".
            log(f"再次收到 {name}，放弃优雅停机，立即退出")
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        _shutting_down.set()
        log(f"收到 {name}，开始优雅停机（再按一次强制退出）")
        raise KeyboardInterrupt(name)

    return handle


def install_shutdown_handlers(log: Callable[[str], None] | None = None) -> None:
    """Route SIGINT/SIGTERM/SIGHUP to KeyboardInterrupt.

    **Call this after every ``import`` in the entry point**, not at the top of
    ``main()`` — see the module docstring for why the ordering is load-bearing.

    Only effective from the main thread, which is also the only thread Python
    delivers signals on; called from anywhere else it raises rather than
    pretending to have worked.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "install_shutdown_handlers 必须在主线程调用 —— "
            "Python 只在主线程投递信号，别处调用会静默无效"
        )
    sink = log or _log
    handle = _handler(sink)
    for sig in SHUTDOWN_SIGNALS:
        signal.signal(sig, handle)


def verify_owns_signals() -> None:
    """Raise if something re-hijacked our handlers.

    The failure this catches is an import ordering mistake: a lazily imported
    module (futu, or anything that vendors the same trick) running its body
    *after* ``install_shutdown_handlers``. That leaves no log line and no
    exception — the process simply stops honouring ``finally`` again, and the
    next Ctrl-C exits 0 with orders still working.

    Cheap enough to call right before entering the trading loop.
    """
    stolen = [
        signal.Signals(sig).name
        for sig in SHUTDOWN_SIGNALS
        if getattr(signal.getsignal(sig), "__module__", None) != __name__
    ]
    if stolen:
        raise RuntimeError(
            f"信号处理器已被覆盖: {', '.join(stolen)} —— "
            "多半是某个模块在 install_shutdown_handlers 之后才被 import "
            "（futu 就是在 import 期无条件抢 SIGINT 的）。"
            "把 install_shutdown_handlers 移到全部 import 之后"
        )


def shutdown_requested() -> bool:
    """Whether a shutdown signal has already been seen.

    For loops that want to stop between units of work rather than be
    interrupted mid-order.
    """
    return _shutting_down.is_set()
