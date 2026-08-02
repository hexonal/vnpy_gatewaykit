"""Stop-price transport on ``OrderRequest.reference``.

Why this lives in gatewaykit rather than in the package that invented it
----------------------------------------------------------------------

The declared stop price travels as a ``|stop=<price>`` suffix on
``OrderRequest.reference``.  ``reference`` is a free-form field that vnpy copies
straight onto the resulting ``OrderData``, so the stop reaches the OMS, the log
and ``on_trade`` without a side table that could fall out of sync — and it
survives the order crossing an engine boundary.

That design only holds while **every** producer and the single consumer agree
on one regex.  It did not hold: ``vnpy_alphakit`` defined the codec and was the
only caller, so orders from ``vnpy_ctastrategy`` (and from the GUI, and from the
MCP bridge) reached the mandatory-stop gate carrying a reference the gate could
not parse and were refused — silently, because ``CtaEngine.send_server_order``
treats an empty ``vt_orderid`` as a ``continue``.  A strategy showed "running"
on screen while not a single opening order left the process.

Moving the codec down here makes it reachable from every package that already
depends on gatewaykit — which is all of them — so a second, subtly different
regex never has to be written.  The alternative considered and rejected was
importing ``vnpy_alphakit`` from ``vnpy_ctastrategy``: that inverts the
dependency layering (two satellites importing each other) to share forty lines.

The deliberate design choices
-----------------------------

* **Anchored at end-of-string.** ``strategy_name`` is user-supplied text. A
  strategy literally named ``my|stop=999_strategy`` must not read back as a
  declared stop of 999.

* **Refuse a bad number at encode time, not decode time.** ``f"{nan:.10g}"``
  renders ``nan``, which the pattern cannot match — the stop would vanish and
  the order would then be refused downstream for the wrong reason ("no stop
  declared"), hiding a bad computation behind a plausible message.

* **The decoder does not validate.** ``1e400`` matches the pattern and
  ``float()``s to ``inf``. Reporting what the reference actually says, and
  leaving the finiteness check to the gate that has to fail closed, keeps one
  place responsible for the policy.
"""

from __future__ import annotations

import math
import re

#: Appended to every non-finite refusal so the message says why it matters.
NON_FINITE_HINT: str = "非有限数值(NaN/inf)会让所有比较返回 False, 等同于关闭该闸"

# Suffix appended to OrderRequest.reference, e.g. "alpha_live.Demo|stop=118.4".
# Anchored at the end of the string so a strategy name containing "stop=" or
# "|" cannot be mistaken for the marker.
STOP_PATTERN = re.compile(r"\|stop=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$")


def is_finite(value: object) -> bool:
    """``math.isfinite`` that answers False instead of raising.

    A gateway is free to put a string, ``None`` or a Decimal in a field this
    module reads.  ``math.isfinite`` raises ``TypeError`` on the first two,
    which would turn a guard into a crash on the order path; answering False
    routes it into the same fail-closed refusal as a NaN.
    """
    try:
        return math.isfinite(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def attach_stop(reference: str, stop_price: float) -> str:
    """Encode ``stop_price`` into an ``OrderRequest.reference`` string.

    Replaces any marker already present rather than appending a second one:
    the pattern is end-anchored, so two markers would mean the stale one is
    read back and the re-priced order carries the wrong protection.
    """
    if not is_finite(stop_price):
        raise ValueError(f"止损价非有限数值: {stop_price!r} —— {NON_FINITE_HINT}")
    if stop_price <= 0:
        raise ValueError(f"止损价必须为正数, 收到 {stop_price}")
    return f"{strip_stop(reference)}|stop={stop_price:.10g}"


def extract_stop(reference: str) -> float | None:
    """Decode the stop price from a reference string, or None if absent.

    None means "nothing was declared", which is a different claim from a
    declared stop of zero — callers must not conflate them.
    """
    match = STOP_PATTERN.search(reference or "")
    if not match:
        return None
    return float(match.group(1))


def strip_stop(reference: str) -> str:
    """Return ``reference`` with any trailing stop marker removed."""
    return STOP_PATTERN.sub("", reference or "")
