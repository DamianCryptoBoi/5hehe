"""Run one submitted Python program against a batch of test inputs.

The program is compiled once. Each test executes in a fresh child process, and
the supervising process enforces the per-test deadline. Expected answers never
enter the container; this runner returns JSON-native values to the host for
comparison.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import select
import signal
import sys
import time
import traceback
from typing import Any

_MAX_CASE_STATUS_BYTES = 256 * 1024
_MAX_BATCH_BYTES = 4 * 1024 * 1024
_POLL_INTERVAL_S = 0.02


def _summarize_exc() -> str:
    exc_type, exc_val, _ = sys.exc_info()
    name = exc_type.__name__ if exc_type else "Error"
    message = str(exc_val)
    head = f"{name}: {message}" if message else name
    return f"{head}\n{traceback.format_exc(limit=4)}".strip()[:4000]


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)[:500]
    except Exception:  # noqa: BLE001
        return "<unreprable>"


def _encode_status(
    status: dict[str, Any], nonce: str, max_status_bytes: int
) -> bytes:
    try:
        body = json.dumps(status, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        body = json.dumps(
            {
                "status": "unserializable",
                "error": "return value is not JSON-serializable",
                "actual_repr": _safe_repr(status.get("value")),
            },
            separators=(",", ":"),
        )
    framed = (nonce + body + nonce).encode("utf-8", errors="replace")
    if len(framed) > max_status_bytes:
        body = json.dumps(
            {"status": "error", "error": "serialized return value is too large"},
            separators=(",", ":"),
        )
        framed = (nonce + body + nonce).encode()
    return framed


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _child_main(
    write_fd: int,
    compiled,
    entrypoint: str,
    args: list[Any],
    kwargs: dict[str, Any],
    nonce: str,
    max_status_bytes: int,
) -> None:
    try:
        os.setsid()
    except OSError:
        pass
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        if devnull not in (1, 2):
            os.close(devnull)
    except OSError:
        pass

    namespace: dict[str, Any] = {"__name__": "__candidate__"}
    try:
        exec(compiled, namespace)  # noqa: S102
        function = namespace.get(entrypoint)
        if not callable(function):
            status = {
                "status": "compile_error",
                "error": f"entrypoint {entrypoint!r} is not defined by the candidate",
            }
        else:
            try:
                status = {"status": "ok", "value": function(*args, **kwargs)}
            except BaseException:  # noqa: BLE001
                status = {"status": "error", "error": _summarize_exc()}
    except BaseException:  # noqa: BLE001
        status = {"status": "compile_error", "error": _summarize_exc()}

    try:
        _write_all(write_fd, _encode_status(status, nonce, max_status_bytes))
    except BaseException:  # noqa: BLE001
        pass
    finally:
        try:
            os.close(write_fd)
        except OSError:
            pass
    os._exit(0)


def _terminate_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _extract_status(data: bytes, nonce: str) -> dict[str, Any] | None:
    text = data.decode("utf-8", errors="replace")
    start = text.rfind(nonce)
    if start < 0:
        return None
    previous = text.rfind(nonce, 0, start)
    if previous < 0:
        return None
    try:
        value = json.loads(text[previous + len(nonce) : start])
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _run_case(
    compiled,
    entrypoint: str,
    case: dict[str, Any],
    timeout_s: float,
    max_status_bytes: int,
):
    read_fd, write_fd = os.pipe()
    nonce = secrets.token_hex(16)
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        _child_main(
            write_fd,
            compiled,
            entrypoint,
            case.get("args", []) or [],
            case.get("kwargs", {}) or {},
            nonce,
            max_status_bytes,
        )

    os.close(write_fd)
    os.set_blocking(read_fd, False)
    deadline = time.monotonic() + timeout_s
    output = bytearray()
    timed_out = False
    child_done = False
    while not child_done:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            timed_out = True
            _terminate_group(pid)
            break
        readable, _, _ = select.select(
            [read_fd], [], [], min(_POLL_INTERVAL_S, remaining)
        )
        if readable:
            try:
                chunk = os.read(read_fd, 8192)
            except BlockingIOError:
                chunk = b""
            if chunk and len(output) < max_status_bytes:
                output.extend(chunk[: max_status_bytes - len(output)])
        try:
            waited, _ = os.waitpid(pid, os.WNOHANG)
            child_done = waited == pid
        except ChildProcessError:
            child_done = True

    _terminate_group(pid)
    if not child_done:
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
    try:
        while len(output) < max_status_bytes:
            chunk = os.read(read_fd, 8192)
            if not chunk:
                break
            output.extend(chunk[: max_status_bytes - len(output)])
    except (BlockingIOError, OSError):
        pass
    os.close(read_fd)

    if timed_out:
        return {"status": "timeout", "error": f"timed out after {timeout_s:.3f}s"}
    return _extract_status(bytes(output), nonce) or {
        "status": "error",
        "error": "test process produced no result",
    }


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        nonce = str(payload["nonce"])
        code = str(payload.get("code", ""))
        entrypoint = str(payload.get("entrypoint", ""))
        cases = list(payload.get("tests", []))
        timeout_s = float(payload.get("timeout_s", 0.0))
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("invalid timeout")
    except Exception:  # noqa: BLE001
        return

    try:
        compiled = compile(code, "<candidate>", "exec")
    except BaseException:  # noqa: BLE001
        status = {"status": "compile_error", "error": _summarize_exc()}
        results = [status for _ in cases]
    else:
        max_status_bytes = max(
            4096,
            min(
                _MAX_CASE_STATUS_BYTES,
                (_MAX_BATCH_BYTES - 4096) // max(1, len(cases)),
            ),
        )
        results = [
            _run_case(compiled, entrypoint, case, timeout_s, max_status_bytes)
            for case in cases
        ]

    body = json.dumps({"status": "batch", "results": results}, separators=(",", ":"))
    sys.stdout.write(nonce + body + nonce)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
