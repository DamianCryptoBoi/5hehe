"""In-sandbox runner: execute ONE candidate call and emit its return VALUE.

Security model (why this shape):
  - The runner is NEVER given the expected answer, so candidate code cannot learn
    the target and forge a matching value.
  - The runner does NOT decide pass/fail. It emits the candidate's return value
    (strict-JSON, so an object with an overridden __eq__ cannot ride along) and
    the TRUSTED PARENT compares it to `expected`.
  - The verdict is wrapped in a per-call random marker so the parent can find it
    amid ordinary candidate stdout. This marker is framing, NOT authentication:
    candidate code shares the interpreter and may inspect process state. Safety
    comes from the parent accepting only strict JSON values and independently
    comparing them with an expected value that never enters the sandbox.

Protocol:
  stdin : JSON {"code", "entrypoint", "args", "kwargs", "nonce"}   (NO expected)
  stdout: NONCE + JSON{"status","value"|"error"} + NONCE
          status in {"ok","error","compile_error","unserializable"}
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any


def _summarize_exc() -> str:
    exc_type, exc_val, _ = sys.exc_info()
    name = exc_type.__name__ if exc_type else "Error"
    msg = str(exc_val)
    tb = traceback.format_exc(limit=4)
    head = f"{name}: {msg}" if msg else name
    return f"{head}\n{tb}".strip()


def _safe_repr(obj: Any) -> str:
    try:
        return repr(obj)[:500]
    except Exception:  # noqa: BLE001
        return "<unreprable>"


def _emit(nonce: str, status: dict[str, Any]) -> None:
    # The random marker frames the verdict amid candidate-written noise. It is
    # intentionally not treated as a secret or an authentication boundary.
    try:
        body = json.dumps(status)
    except Exception:  # noqa: BLE001 - serializing our own status dict
        body = json.dumps({"status": "error", "error": "runner: status not serializable"})
    sys.stdout.write(nonce + body + nonce)
    sys.stdout.flush()


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        nonce = str(payload["nonce"])
    except Exception:  # noqa: BLE001 - no nonce => parent will see no valid frame
        return

    code = payload.get("code", "")
    entrypoint = payload.get("entrypoint", "")
    args = payload.get("args", []) or []
    kwargs = payload.get("kwargs", {}) or {}

    # --- Compile / import phase: failures here are COMPILE errors. ---
    sandbox_globals: dict[str, Any] = {"__name__": "__candidate__"}
    try:
        exec(compile(code, "<candidate>", "exec"), sandbox_globals)  # noqa: S102
    except BaseException:  # noqa: BLE001 - SystemExit too; candidate must not skip emit
        _emit(nonce, {"status": "compile_error", "error": _summarize_exc()})
        return

    fn = sandbox_globals.get(entrypoint)
    if not callable(fn):
        _emit(nonce, {"status": "compile_error",
                      "error": f"entrypoint {entrypoint!r} is not defined by the candidate"})
        return

    # --- Runtime phase. Catch BaseException so a candidate cannot use SystemExit
    #     to suppress the verdict (os._exit is uncatchable -> no frame -> fail). ---
    try:
        actual = fn(*args, **kwargs)
    except BaseException:  # noqa: BLE001
        _emit(nonce, {"status": "error", "error": _summarize_exc()})
        return

    # Strict JSON validation: rejects non-native types and objects with fake
    # __eq__/__repr__ (json serializes by value, never via candidate equality).
    try:
        json.dumps(actual, allow_nan=False)
    except (TypeError, ValueError):
        _emit(nonce, {"status": "unserializable",
                      "error": f"return value not JSON-serializable: {type(actual).__name__}",
                      "actual_repr": _safe_repr(actual)})
        return

    _emit(nonce, {"status": "ok", "value": actual})


if __name__ == "__main__":
    main()
