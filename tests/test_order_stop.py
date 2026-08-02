"""The ``|stop=`` reference marker — the wire format two packages must agree on.

The declared stop price rides on ``OrderRequest.reference`` rather than in a
field of its own, so it survives the trip into ``OrderData``, the OMS and the
log without a side table to fall out of sync.  That only works while the
producer (whoever attaches it) and the consumer (the mandatory-stop gate) share
one regex; this module is that single definition, and these tests pin the parts
a second implementation would get subtly wrong.
"""

from __future__ import annotations

import re

import pytest

from vnpy_gatewaykit.order_stop import (
    NON_FINITE_HINT,
    attach_stop,
    extract_stop,
    is_finite,
    strip_stop,
)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------
def test_attached_stop_reads_back_unchanged() -> None:
    reference: str = attach_stop("alpha_live.Demo", 118.4)
    assert extract_stop(reference) == pytest.approx(118.4)


def test_attaching_twice_replaces_rather_than_appends() -> None:
    """Otherwise a re-priced order carries two markers and the regex, being
    anchored at the end, silently reports the stale one."""
    once: str = attach_stop("alpha_live.Demo", 118.4)
    twice: str = attach_stop(once, 120.0)
    assert twice.count("|stop=") == 1
    assert extract_stop(twice) == pytest.approx(120.0)


def test_strip_returns_the_original_reference() -> None:
    assert strip_stop(attach_stop("CtaStrategy_aa", 40.2)) == "CtaStrategy_aa"


def test_absent_marker_reads_as_none_not_zero() -> None:
    """Zero would be a *declared* stop of zero — a different claim entirely."""
    assert extract_stop("CtaStrategy_aa") is None
    assert extract_stop("") is None


# ---------------------------------------------------------------------------
# Anchoring — a strategy name is user-supplied text
# ---------------------------------------------------------------------------
def test_a_strategy_named_like_the_marker_is_not_mistaken_for_one() -> None:
    """The pattern is anchored at end-of-string precisely for this input."""
    assert extract_stop("my|stop=999_strategy") is None


def test_marker_inside_a_longer_reference_is_ignored() -> None:
    assert extract_stop("a|stop=1.0 trailing text") is None


def test_real_marker_still_wins_when_the_name_also_contains_one() -> None:
    reference: str = attach_stop("my|stop=999_strategy", 40.5)
    assert extract_stop(reference) == pytest.approx(40.5)


# ---------------------------------------------------------------------------
# Refusals — encoding a bad number would hide it behind a plausible message
# ---------------------------------------------------------------------------
def test_non_finite_stop_is_refused_at_encode_time() -> None:
    """``f"{nan:.10g}"`` renders "nan", which the pattern cannot match, so the
    stop would vanish and the order would be refused downstream for the wrong
    reason — "no stop declared" instead of "your stop is NaN"."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="非有限"):
            attach_stop("s", bad)


def test_refusal_message_says_why_non_finite_matters() -> None:
    # re.escape because the hint contains "(NaN/inf)" — bare, that is a regex
    # capture group and matches a message that does not contain the text.
    with pytest.raises(ValueError, match=re.escape(NON_FINITE_HINT)):
        attach_stop("s", float("nan"))


def test_non_positive_stop_is_refused() -> None:
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="必须为正数"):
            attach_stop("s", bad)


# ---------------------------------------------------------------------------
# Decoding edge cases the consumer must still guard
# ---------------------------------------------------------------------------
def test_scientific_notation_round_trips() -> None:
    """``:.10g`` emits exponent form for small numbers, so the pattern has to
    accept what the encoder produces."""
    reference: str = attach_stop("s", 0.00001234)
    assert "e" in reference or "E" in reference
    assert extract_stop(reference) == pytest.approx(0.00001234)


def test_a_hand_written_overflow_literal_parses_to_inf() -> None:
    """1e400 matches the pattern and float()s to inf.

    Documented rather than fixed here: the decoder's job is to report what the
    reference says.  Callers must still check ``is_finite`` on the result — the
    gate does, and this test exists so nobody "simplifies" that check away.
    """
    assert extract_stop("s|stop=1e400") == float("inf")


# ---------------------------------------------------------------------------
# is_finite — shared so producer and consumer agree on what a number is
# ---------------------------------------------------------------------------
def test_is_finite_answers_false_instead_of_raising_on_junk() -> None:
    """A gateway is free to put a string or None in a field this reads; raising
    would turn a guard into a crash on the order path."""
    assert is_finite(1.0)
    assert not is_finite(float("nan"))
    assert not is_finite(None)
    assert not is_finite("40.2")
