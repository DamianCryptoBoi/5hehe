"""Trusted in-container supervisor for Rust submissions."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

_MAX_CASE_FRAME_BYTES = 512_000
_MAX_BATCH_BYTES = 4 * 1024 * 1024
_MAX_OUTPUT_BYTES = 1_000_000


def _uniform(count: int, error: str) -> list[dict[str, str]]:
    return [{"status": "compile_error", "error": error} for _ in range(count)]


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _compile(command: list[str], timeout_s: float) -> tuple[int | None, str]:
    """Compile in a killable process group and retain bounded diagnostics."""

    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=errors,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.wait()
            return None, "compilation timed out"
        errors.seek(0, os.SEEK_END)
        size = errors.tell()
        errors.seek(max(0, size - 2000))
        detail = errors.read().decode("utf-8", errors="replace")[-500:]
    return proc.returncode, detail


def _run_case(
    artifact: Path,
    stdin: str,
    timeout_s: float,
    max_status_bytes: int,
) -> dict[str, str]:
    with tempfile.TemporaryFile() as output:
        try:
            proc = subprocess.Popen(
                [str(artifact)],
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as error:
            return {
                "status": "error",
                "error": f"could not execute compiled artifact: {error}",
            }
        try:
            proc.communicate(stdin.encode("utf-8"), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.wait()
            return {"status": "timeout", "error": "test timed out"}
        output.seek(0, os.SEEK_END)
        size = output.tell()
        if size > _MAX_OUTPUT_BYTES:
            return {"status": "error", "error": "candidate output is too large"}
        output.seek(0)
        raw = output.read()
    if proc.returncode != 0:
        return {
            "status": "error",
            "error": f"program exited with status {proc.returncode}",
        }
    try:
        value = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return {"status": "error", "error": "candidate output is not valid UTF-8"}
    encoded = json.dumps({"status": "ok", "value": value}, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > max_status_bytes:
        return {"status": "error", "error": "serialized candidate output is too large"}
    return {"status": "ok", "value": value}


def main() -> None:
    payload = json.load(sys.stdin)
    nonce = str(payload["nonce"])
    tests = payload["tests"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "main.rs"
        artifact = root / "submission"
        source.write_text(str(payload["code"]), encoding="utf-8")
        command = [*payload["compiler_argv"], str(source), "-o", str(artifact)]
        try:
            compile_rc, compile_detail = _compile(
                command,
                max(0.001, float(payload["compile_timeout_s"])),
            )
        except Exception as error:  # noqa: BLE001
            results = _uniform(
                len(tests),
                f"compiler failed: {type(error).__name__}: {error}",
            )
        else:
            if compile_rc is None:
                results = _uniform(len(tests), compile_detail)
            elif compile_rc != 0:
                results = _uniform(
                    len(tests), compile_detail or "compilation failed"
                )
            elif not artifact.is_file() or artifact.stat().st_size > int(
                payload["artifact_max_bytes"]
            ):
                results = _uniform(len(tests), "compiled artifact exceeds size limit")
            else:
                max_status_bytes = max(
                    4096,
                    min(
                        _MAX_CASE_FRAME_BYTES,
                        (_MAX_BATCH_BYTES - 4096) // max(1, len(tests)),
                    ),
                )
                results = [
                    _run_case(
                        artifact,
                        str(test["stdin"]),
                        max(0.001, float(payload["timeout_s"])),
                        max_status_bytes,
                    )
                    for test in tests
                ]
    body = json.dumps({"status": "batch", "results": results}, separators=(",", ":"))
    sys.stdout.write(nonce + body + nonce)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
