"""Vectors for the rust-exact-token-v1 judge (docs/RUST_CHALLENGES.md).

The judge is deliberately dumb: tokenize both sides on runs of ASCII
whitespace, compare tokens as exact strings in order, require equal counts.
There are NO numeric semantics — several vectors below exist purely to pin
that ABSENCE, so tolerance or value-comparison cannot erode in later
"improvements" without going through a new judge version and capability token.

Expected seam (red until implemented):

    from rlvr.execution.rust_judge import JUDGE_VERSION, outputs_match

    outputs_match(actual: str, expected: str) -> bool
    JUDGE_VERSION == "rust-exact-token-v1"

The judge takes decoded text. Strict UTF-8 decoding of candidate bytes (and
the typed failure for invalid UTF-8) is the runner's job and is pinned in the
runner tests, not here.

Imports happen inside each test so a missing module reports per-test as the
absent feature rather than one collection error masking the suite.
"""

from __future__ import annotations


def _judge():
    from rlvr.execution.rust_judge import outputs_match

    return outputs_match


# --------------------------------------------------------------------------- #
# Version pin
# --------------------------------------------------------------------------- #
def test_judge_version_is_the_agreed_token():
    from rlvr.execution.rust_judge import JUDGE_VERSION

    assert JUDGE_VERSION == "rust-exact-token-v1"


# --------------------------------------------------------------------------- #
# Tokenization: runs of ASCII whitespace, and nothing else
# --------------------------------------------------------------------------- #
def test_identical_single_token_matches():
    assert _judge()("42", "42")


def test_all_six_ascii_delimiters_are_equivalent():
    """0x09 0x0A 0x0B 0x0C 0x0D 0x20 all delimit; runs collapse."""
    actual = "1 2\t3\n4\x0b5\x0c6\r7"
    expected = "1 2 3 4 5 6 7"
    assert _judge()(actual, expected)


def test_crlf_line_endings_are_delimiter_noise():
    assert _judge()("1\r\n2\r\n", "1\n2\n")


def test_leading_trailing_and_repeated_whitespace_are_ignored():
    assert _judge()("  \t a   b \n\n", "a b")


def test_extra_token_fails():
    assert not _judge()("1 2 3", "1 2")


def test_missing_token_fails():
    assert not _judge()("1", "1 2")


def test_empty_output_matches_empty_expected():
    assert _judge()("", "")


def test_whitespace_only_output_matches_empty_expected():
    assert _judge()(" \n\t ", "")


def test_empty_output_against_a_token_fails():
    assert not _judge()("", "0")


# --------------------------------------------------------------------------- #
# Exactness: the deliberate ABSENCE of numeric semantics
# --------------------------------------------------------------------------- #
def test_leading_zeros_fail():
    """007 != 7. Exact strings — the earlier value-compare rule was replaced
    by the exact-token judge in the agreed contract; this pins the change."""
    assert not _judge()("007", "7")


def test_negative_zero_fails():
    assert not _judge()("-0", "0")


def test_float_formatting_fails_against_integer():
    assert not _judge()("1.0", "1")


def test_scientific_notation_fails_against_plain_integer():
    assert not _judge()("1e3", "1000")


def test_close_floats_fail():
    """No tolerance of any kind: 0.3333333 != 0.3333334."""
    assert not _judge()("0.3333333", "0.3333334")


def test_inf_and_nan_are_ordinary_strings():
    """No special-casing: the judge has no numeric detector to trip. Authoring
    rules forbid non-finite EXPECTED values; the judge itself is agnostic."""
    assert _judge()("inf", "inf")
    assert _judge()("NaN", "NaN")
    assert not _judge()("inf", "infinity")


# --------------------------------------------------------------------------- #
# Content is unicode; only DELIMITERS are ASCII
# --------------------------------------------------------------------------- #
def test_non_ascii_whitespace_is_token_content_not_a_delimiter():
    """U+00A0 (NBSP) and U+2009 (thin space) are NOT delimiters. A token
    containing them is one token and compares by codepoint."""
    assert not _judge()("a\u00a0b", "a b")
    assert _judge()("a\u00a0b", "a\u00a0b")
    assert not _judge()("a\u2009b", "a b")


def test_unicode_content_compares_exactly():
    assert _judge()("héllo", "héllo")
    assert not _judge()("héllo", "hello")


def test_comparison_is_case_sensitive():
    assert not _judge()("ABC", "abc")
