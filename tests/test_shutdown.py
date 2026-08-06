"""Signal handling: SIGINT/SIGTERM/SIGHUP must reach `finally`, not skip it.

Why the load-bearing cases are subprocesses
-------------------------------------------
The bug this module exists for is that `import futu` replaces SIGINT with a bare
``os._exit(0)``. **No in-process assertion can catch that**: ``os._exit`` does not
raise, does not run ``finally``, and does not return — a test that called it would
take pytest down with it and report nothing. The only way to observe "did the
``finally`` block run" is to look at what a separate process wrote before it died,
plus its exit code.

So the tests below split in two:

* **In-process** — handler bookkeeping that is safe to observe directly
  (which signals get claimed, main-thread guard, the re-hijack detector).
* **Subprocess** — the actual contract: a `finally` runs, and the exit code is
  not 0. These are the ones that would have caught the original defect; the
  in-process ones would all have passed against the broken build.

`futu` is only installed in the vnpy venv, so the tests that need it skip cleanly
elsewhere rather than failing for the wrong reason.
"""

from __future__ import annotations

import importlib.util
import signal
import subprocess
import sys
import textwrap
import threading

import pytest

from vnpy_gatewaykit.shutdown import (
    SHUTDOWN_SIGNALS,
    install_shutdown_handlers,
    shutdown_requested,
    verify_owns_signals,
)

HAS_FUTU = importlib.util.find_spec("futu") is not None

#: Long enough that the parent is definitely blocked when the signal lands,
#: short enough that a hung test is obvious rather than a CI timeout.
SLEEP = 30


def run_child(body: str, sig: int, *, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter and send it ``sig`` from inside itself.

    Self-signalling rather than `proc.send_signal` from the parent: it removes
    the race where the child has not reached the sleep yet, which would make the
    test flaky in exactly the direction that hides bugs (a signal delivered
    before the handler is installed looks like "the handler did not work").
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body).format(sig=sig, sleep=SLEEP)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# 端到端：finally 到底跑没跑
# ---------------------------------------------------------------------------

#: The shape of the real entry point: import futu (which steals SIGINT), install
#: handlers, do work, and rely on `finally` for the cleanup that cancels orders.
ENTRY = """
    import os, signal, sys
    import futu                                  # steals SIGINT at import time
    from vnpy_gatewaykit.shutdown import install_shutdown_handlers
    install_shutdown_handlers(log=lambda m: None)
    try:
        os.kill(os.getpid(), {sig})
        import time; time.sleep({sleep})
    except KeyboardInterrupt:
        sys.stderr.write("KBD\\n")
    finally:
        sys.stderr.write("FINALLY\\n"); sys.stderr.flush()
"""

#: Same thing without the fix — the regression baseline. Kept as an executable
#: statement of the defect: if futu ever stops hijacking SIGINT, this test goes
#: red and tells us the workaround can be dropped, instead of the workaround
#: quietly outliving its reason.
UNFIXED = """
    import os, signal, sys
    import futu
    try:
        os.kill(os.getpid(), {sig})
        import time; time.sleep({sleep})
    finally:
        sys.stderr.write("FINALLY\\n"); sys.stderr.flush()
"""


@pytest.mark.skipif(not HAS_FUTU, reason="futu 只装在 vnpy 的 venv 里")
def test_sigint_after_importing_futu_still_reaches_finally():
    """The whole point. Broken build: no output, exit 0."""
    r = run_child(ENTRY, signal.SIGINT)
    assert "FINALLY" in r.stderr, f"finally 没跑，stderr={r.stderr!r} rc={r.returncode}"
    assert "KBD" in r.stderr, "SIGINT 应该变成 KeyboardInterrupt"


@pytest.mark.skipif(not HAS_FUTU, reason="futu 只装在 vnpy 的 venv 里")
def test_the_unfixed_shape_really_does_skip_finally():
    """Pin the defect itself, so the fix cannot be silently deleted as useless.

    Exit code 0 is the nastiest part: a supervisor sees a clean run.
    """
    r = run_child(UNFIXED, signal.SIGINT)
    assert "FINALLY" not in r.stderr, (
        "futu 不再劫持 SIGINT 了？那 shutdown.py 的理由消失了，去核实并考虑删掉它"
    )
    assert r.returncode == 0, f"期望伪装成功的退出码 0，实得 {r.returncode}"


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="平台无 SIGHUP")
def test_sighup_reaches_finally_too():
    """Closing a tmux window is as routine as Ctrl-C, and futu is not involved —
    plain CPython默认 would terminate outright."""
    body = """
        import os, signal, sys
        from vnpy_gatewaykit.shutdown import install_shutdown_handlers
        install_shutdown_handlers(log=lambda m: None)
        try:
            os.kill(os.getpid(), {sig})
            import time; time.sleep({sleep})
        except KeyboardInterrupt:
            pass
        finally:
            sys.stderr.write("FINALLY\\n"); sys.stderr.flush()
    """
    r = run_child(body, signal.SIGHUP)
    assert "FINALLY" in r.stderr, f"stderr={r.stderr!r} rc={r.returncode}"


def test_sigterm_reaches_finally_too():
    body = """
        import os, signal, sys
        from vnpy_gatewaykit.shutdown import install_shutdown_handlers
        install_shutdown_handlers(log=lambda m: None)
        try:
            os.kill(os.getpid(), {sig})
            import time; time.sleep({sleep})
        except KeyboardInterrupt:
            pass
        finally:
            sys.stderr.write("FINALLY\\n"); sys.stderr.flush()
    """
    r = run_child(body, signal.SIGTERM)
    assert "FINALLY" in r.stderr, f"stderr={r.stderr!r} rc={r.returncode}"


def test_the_second_signal_kills_even_if_cleanup_swallows_interrupts():
    """Graceful shutdown cancels orders against a broker and can block. Ctrl-C
    twice has to work, or the one key everybody reaches for cannot stop it.

    **The cleanup here swallows KeyboardInterrupt on purpose.** An earlier version
    of this test just pressed Ctrl-C twice and asserted the exit code was
    ``-SIGINT`` — and it passed against a build with the second-press guard
    removed, because CPython restores SIG_DFL and re-raises when a
    KeyboardInterrupt reaches the top level, producing the very same exit code.
    The two behaviours are only distinguishable when something *catches* the
    interrupt, which is exactly the realistic hang: a cleanup path with a broad
    ``except`` around broker I/O. Found by mutation testing, not by reading.
    """
    body = """
        import os, signal, sys, time
        from vnpy_gatewaykit.shutdown import install_shutdown_handlers
        install_shutdown_handlers(log=lambda m: None)
        try:
            os.kill(os.getpid(), {sig})
            time.sleep({sleep})
        except KeyboardInterrupt:
            sys.stderr.write("FIRST\\n"); sys.stderr.flush()
            # 模拟一个吞掉中断、迟迟不退的停机路径
            for i in range(3):
                try:
                    os.kill(os.getpid(), {sig})
                    time.sleep(5)
                except KeyboardInterrupt:
                    sys.stderr.write("SWALLOWED%d\\n" % i); sys.stderr.flush()
            sys.stderr.write("SURVIVED\\n"); sys.stderr.flush()
    """
    r = run_child(body, signal.SIGINT, timeout=90)
    assert "FIRST" in r.stderr, f"stderr={r.stderr!r}"
    assert "SWALLOWED0" not in r.stderr, "第二次信号被吞了 —— 强制退出的守卫没生效"
    assert "SURVIVED" not in r.stderr, "进程扛过了三次 Ctrl-C，操作员将无法停止它"
    assert r.returncode != 0, f"强制退出不能报成功，实得 {r.returncode}"


# ---------------------------------------------------------------------------
# 进程内：登记与守卫
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_signals():
    """Put the process's dispositions back. Without this, installing handlers in
    one test leaks into every later test in the same pytest process."""
    saved = {s: signal.getsignal(s) for s in SHUTDOWN_SIGNALS}
    yield
    for s, h in saved.items():
        signal.signal(s, h)


def test_it_claims_every_shutdown_signal_this_platform_has(restore_signals):
    install_shutdown_handlers(log=lambda m: None)
    for sig in SHUTDOWN_SIGNALS:
        handler = signal.getsignal(sig)
        assert callable(handler)
        assert handler.__module__ == "vnpy_gatewaykit.shutdown"


def test_sigint_is_among_them():
    """Guards against a refactor that keeps the module but drops the one signal
    the whole thing was written for."""
    assert signal.SIGINT in SHUTDOWN_SIGNALS


def test_installing_off_the_main_thread_raises_instead_of_pretending(restore_signals):
    """Python only delivers signals on the main thread, so a call from anywhere
    else is a no-op that looks like success — the exact failure mode this module
    is meant to eliminate."""
    box: list[BaseException] = []

    def target() -> None:
        try:
            install_shutdown_handlers(log=lambda m: None)
        except BaseException as exc:  # noqa: BLE001 — 要的就是把它带回主线程断言
            box.append(exc)

    t = threading.Thread(target=target)
    t.start()
    t.join()
    assert box and isinstance(box[0], RuntimeError)
    assert "主线程" in str(box[0])


def test_verify_passes_right_after_install(restore_signals):
    install_shutdown_handlers(log=lambda m: None)
    verify_owns_signals()


def test_verify_catches_a_late_import_stealing_the_handler(restore_signals):
    """The failure this exists for: a lazily imported module running its body
    after install and grabbing SIGINT, leaving no trace at all."""
    install_shutdown_handlers(log=lambda m: None)
    signal.signal(signal.SIGINT, lambda *_: None)          # 冒充 futu 的抢夺
    with pytest.raises(RuntimeError, match="信号处理器已被覆盖"):
        verify_owns_signals()


def test_verify_names_the_stolen_signal(restore_signals):
    """A bare 'something was overwritten' would send the reader hunting through
    three signals; the message has to say which."""
    install_shutdown_handlers(log=lambda m: None)
    signal.signal(signal.SIGTERM, lambda *_: None)
    with pytest.raises(RuntimeError, match="SIGTERM"):
        verify_owns_signals()


def test_shutdown_requested_is_false_before_any_signal():
    assert shutdown_requested() is False


# ---------------------------------------------------------------------------
# 接线：run_live_alpha 的安装位置
# ---------------------------------------------------------------------------


def test_run_live_alpha_installs_after_importing_futu():
    """Ordering is load-bearing: futu overwrites SIGINT while its module body
    runs, so anything installed earlier is silently discarded.

    Asserted on source text rather than by running the entry point — running it
    needs a live OpenD. Textual, but it pins the one property that a reader
    cannot verify by looking at either file alone.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[2]
        / "vnpy_app"
        / "run_live_alpha.py"
    ).read_text(encoding="utf-8")

    import_at = src.index("from vnpy_futu import FutuGateway")
    install_at = src.index("install_shutdown_handlers()")
    assert import_at < install_at, "install_shutdown_handlers() 必须在 import futu 之后"

    # 且必须在进循环之前自检一次
    assert "verify_owns_signals()" in src
    assert src.index("verify_owns_signals()") < src.index("return run_loop(")
