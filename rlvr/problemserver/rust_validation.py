"""Validation for the Rust stdin/stdout test representation."""

from __future__ import annotations

from ..types import TestCase

_MAX_STDIN_BYTES = 65_536
_MAX_EXPECTED_BYTES = 32_768


def validate_rust_tests(
    cases: list[TestCase], *, require_nonempty: bool = True
) -> None:
    if require_nonempty and not cases:
        raise ValueError("Rust challenge requires at least one test")
    if len(cases) > 100:
        raise ValueError("Rust challenge exceeds the 100-test limit")
    for case in cases:
        if len(case.args) != 1 or not isinstance(case.args[0], str):
            raise ValueError("Rust tests require exactly one string stdin argument")
        if case.kwargs:
            raise ValueError("Rust tests do not support keyword arguments")
        if not isinstance(case.expected, str):
            raise ValueError("Rust expected stdout must be a string")
        if len(case.args[0].encode("utf-8")) > _MAX_STDIN_BYTES:
            raise ValueError("Rust test stdin exceeds 65536 bytes")
        if len(case.expected.encode("utf-8")) > _MAX_EXPECTED_BYTES:
            raise ValueError("Rust expected stdout exceeds 32768 bytes")
