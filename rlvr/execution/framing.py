"""Strict parser for sandbox-runner stdout frames."""

from __future__ import annotations

import json
from typing import Any, Optional


def _reject_constant(value: str) -> None:
    """Reject Python's permissive NaN/Infinity JSON extensions."""
    raise ValueError(f"non-standard JSON constant: {value}")


def _valid_status(obj: Any) -> bool:
    """Accept only shapes emitted by ``_runner._emit``."""
    if not isinstance(obj, dict) or not isinstance(obj.get("status"), str):
        return False
    kind = obj["status"]
    if kind == "ok":
        return set(obj) == {"status", "value"}
    if kind in {"error", "compile_error"}:
        return set(obj) == {"status", "error"} and isinstance(obj["error"], str)
    if kind == "timeout":
        return set(obj) == {"status", "error"} and isinstance(obj["error"], str)
    if kind == "unserializable":
        return (
            set(obj) == {"status", "error", "actual_repr"}
            and isinstance(obj["error"], str)
            and isinstance(obj["actual_repr"], str)
        )
    if kind == "batch":
        return (
            set(obj) == {"status", "results"}
            and isinstance(obj["results"], list)
            and all(_valid_status(item) for item in obj["results"])
        )
    return False


def extract_framed(text: str, marker: str) -> Optional[dict[str, Any]]:
    """Pull the final marker-framed, strict-JSON verdict from stdout.

    The random marker distinguishes runner output from ordinary stdout; it is
    not an authentication secret. Selecting the final frame gives the runner's
    verdict precedence over earlier candidate noise in the normal return path.
    """
    if not text or not marker:
        return None
    end = text.rfind(marker)
    if end == -1:
        return None
    start = text.rfind(marker, 0, end)
    if start == -1:
        return None
    try:
        obj = json.loads(
            text[start + len(marker):end],
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return obj if _valid_status(obj) else None
