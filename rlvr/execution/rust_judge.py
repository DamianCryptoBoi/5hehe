"""Exact-token comparison for Rust program output."""

from __future__ import annotations

import re

JUDGE_VERSION = "rust-exact-token-v1"
_ASCII_WHITESPACE = re.compile(r"[\x09-\x0d\x20]+")


def _tokens(value: str) -> list[str]:
    stripped = value.strip("\x09\x0a\x0b\x0c\x0d\x20")
    return [] if not stripped else _ASCII_WHITESPACE.split(stripped)


def outputs_match(actual: str, expected: str) -> bool:
    """Compare ordered tokens, ignoring only ASCII whitespace runs."""

    return _tokens(actual) == _tokens(expected)
