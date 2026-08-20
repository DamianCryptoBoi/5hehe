"""Structured weight-submission failure details.

When the SDK reports failure with an empty message, the operator currently gets
`ok=False msg=''` and nothing else — a failure with no stated cause. This
behavior recovers the structured `error` and `data` the result object carries.

The final block-context fallback for when message, error, and data are all empty
is tested separately.

The EXACT rendered format is also not pinned — these cases assert that the
operator can read the SDK's words, not how they are punctuated.

CONSTRAINT this design works around: `_weight_result_status` has its exact tuple
return shape asserted at tests/test_decentralized.py:38,42,43. It must not
change shape, so the structured detail arrives as a separate function.

"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlvr.neurons.decentralized import (
    _MAX_WEIGHT_DETAIL_CHARS,
    _weight_failure_detail,
)


def result(**attrs):
    return SimpleNamespace(**attrs)


def test_nothing_to_report_yields_nothing():
    """A result carrying neither field must not manufacture a detail string."""
    assert _weight_failure_detail(result(success=False, message="")) == ""


def test_a_structured_error_is_reported():
    detail = _weight_failure_detail(result(error="InvalidTransaction", data=""))

    assert "InvalidTransaction" in detail


def test_both_fields_are_reported_when_present():
    detail = _weight_failure_detail(
        result(error="InvalidTransaction", data="Stale(0x1234)")
    )

    assert "InvalidTransaction" in detail
    assert "Stale(0x1234)" in detail


def test_a_present_field_is_reported_even_if_the_other_is_missing():
    """Legacy and partial SDK shapes must not suppress the half we do have."""
    only_data = _weight_failure_detail(result(data="Stale(0x1234)"))

    assert "Stale(0x1234)" in only_data


# --------------------------------------------------------------------------- #
# Log hygiene — the same bounding discipline the lease diagnostics use
# --------------------------------------------------------------------------- #
def test_detail_is_bounded():
    detail = _weight_failure_detail(result(error="x" * 5_000, data="y" * 5_000))

    assert len(detail) <= _MAX_WEIGHT_DETAIL_CHARS


def test_a_long_field_does_not_crowd_out_the_other_one():
    """`data` is usually the actionable half and is usually the SHORT one.

    A single shared budget lets a 5,000-character `error` consume all of it and
    silently drop `data`. Bound each field on its own.
    """
    detail = _weight_failure_detail(
        result(error="x" * 5_000, data="Stale(0x1234)")
    )

    assert "Stale(0x1234)" in detail
    assert len(detail) <= _MAX_WEIGHT_DETAIL_CHARS


def test_a_truncated_field_says_so():
    """An operator must not read a cut value as the SDK's complete words."""
    detail = _weight_failure_detail(result(error="x" * 5_000))

    assert "..." in detail


@pytest.mark.parametrize("value", [{}, [], ()])
def test_empty_containers_are_absence_not_content(value):
    """Empty containers must allow the block-context fallback.

    `data={}` rendered as "data={}" would satisfy "something to report" and
    preempt the block-context fallback with a string that says nothing.
    """
    assert _weight_failure_detail(result(error=value, data=value)) == ""


def test_detail_stays_on_one_log_line():
    """A multi-line SDK error must not fracture the validator's log format."""
    detail = _weight_failure_detail(
        result(error="line one\nline two\r\nline three", data="a\tb")
    )

    assert "\n" not in detail
    assert "\r" not in detail


def test_non_string_fields_are_rendered_rather_than_dropped():
    """`data` is frequently a dict or an enum, and it is the useful half."""
    detail = _weight_failure_detail(result(error=None, data={"index": 7}))

    assert "7" in detail


# --------------------------------------------------------------------------- #
# This runs on the weight-submission path, so it must never be the thing that
# breaks weight submission
# --------------------------------------------------------------------------- #
def test_an_attribute_that_raises_is_treated_as_absent():
    """A property raising on access must not take down the submission path.

    Reaching for diagnostics is not worth losing the round's weights over.
    """

    class Hostile:
        success = False
        message = ""

        @property
        def error(self):
            raise RuntimeError("exploding property")

        data = "Stale(0x1234)"

    detail = _weight_failure_detail(Hostile())

    assert "Stale(0x1234)" in detail


def test_an_unrenderable_field_does_not_propagate():
    class Unrenderable:
        def __repr__(self):
            raise RuntimeError("no repr for you")

    detail = _weight_failure_detail(result(error=Unrenderable(), data="usable"))

    assert "usable" in detail


@pytest.mark.parametrize("value", [None, "", 0, False])
def test_empty_and_falsey_fields_are_omitted(value):
    """`error=None` is absence, not a cause worth printing."""
    assert _weight_failure_detail(result(error=value, data=value)) == ""


def test_a_legacy_tuple_result_has_no_structured_detail():
    """Legacy shapes carry no structured fields; that is absence, not an error."""
    assert _weight_failure_detail((False, "rejected by chain")) == ""
    assert _weight_failure_detail(False) == ""
