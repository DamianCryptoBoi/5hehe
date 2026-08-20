"""SubprocessExecutor: run candidate code in a fresh python subprocess per test.

Security posture (see SECURITY notes in docs/DESIGN.md):
  - The sandbox is NEVER given the expected answer; the parent (this class)
    compares the returned value via `compare.values_equal`. So a candidate
    cannot forge a passing verdict (it doesn't know the target) and an
    overridden __eq__ cannot ride along (we compare JSON-native values).
  - The verdict is wrapped in a per-call random marker so it can be found amid
    candidate stdout. The marker is framing, not a secret/authentication token.
  - Resource limits (no file writes, capped address space/CPU, no core dumps)
    are applied via setrlimit; the child is its own session so a timeout kills
    the WHOLE process group (no orphaned grandchildren). stdout is read with a
    hard cap so a flooding candidate cannot OOM the validator.
  - On macOS, a best-effort `sandbox-exec` profile additionally denies network
    and reads of the operator's home dir (where .env / wallets live).

This backend is DEV-GRADE. It does not prevent filesystem reads or direct
network access on Linux. For untrusted miner code use the Docker backend.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ..types import ExecutionResult, TestCase
from . import _runner
from .compare import values_equal
from .executor import Executor
from .framing import extract_framed as _extract_framed

_RUNNER_SRC = Path(_runner.__file__).read_text(encoding="utf-8")
_MAX_OUTPUT_BYTES = 256 * 1024  # cap captured stdout to prevent OOM-by-flood
_MEM_LIMIT_BYTES = 1024 * 1024 * 1024  # 1 GiB address-space cap (enforced on Linux)


def _make_limits(timeout_s: float):
    """preexec_fn applying resource limits in the child (POSIX only)."""

    def _apply() -> None:  # pragma: no cover - runs in the forked child
        try:
            import resource

            def _set(res: int, soft: int, hard: Optional[int] = None) -> None:
                try:
                    resource.setrlimit(res, (soft, hard if hard is not None else soft))
                except (ValueError, OSError):
                    pass

            _set(resource.RLIMIT_FSIZE, 0)  # no file writes (blocks exfil/OOM-by-write)
            _set(resource.RLIMIT_CORE, 0)  # no core dumps
            _set(resource.RLIMIT_CPU, int(timeout_s) + 1)  # CPU-seconds backstop
            if hasattr(resource, "RLIMIT_AS"):
                _set(resource.RLIMIT_AS, _MEM_LIMIT_BYTES)  # note: macOS ignores this
        except Exception:  # noqa: BLE001
            pass

    return _apply


class SubprocessExecutor(Executor):
    def __init__(self, settings: Any = None):
        self.settings = settings
        self._python = sys.executable or "python3"
        # default sandbox-exec ON for darwin dev; harmless if the binary is absent
        self._use_seatbelt = bool(getattr(settings, "use_sandbox_exec", sys.platform == "darwin"))

    # ------------------------------------------------------------------ #
    def run_tests(
        self, code: str, entrypoint: str, tests: list[TestCase], timeout_s: float
    ) -> list[ExecutionResult]:
        if not tests:
            return []
        with tempfile.TemporaryDirectory(prefix="rlvr_sbx_") as tmp:
            runner_path = Path(tmp) / "_sbx_runner.py"
            runner_path.write_text(_RUNNER_SRC, encoding="utf-8")
            env = self._sandbox_env(tmp)
            return [
                self._run_one(idx, t, code, entrypoint, runner_path, tmp, env, timeout_s)
                for idx, t in enumerate(tests)
            ]

    # ------------------------------------------------------------------ #
    def _run_one(
        self,
        idx: int,
        test: TestCase,
        code: str,
        entrypoint: str,
        runner_path: Path,
        tmp: str,
        env: dict[str, str],
        timeout_s: float,
    ) -> ExecutionResult:
        nonce = secrets.token_hex(16)
        # NOTE: expected is intentionally NOT sent into the sandbox.
        payload = json.dumps(
            {"code": code, "entrypoint": entrypoint, "args": test.args,
             "kwargs": test.kwargs, "nonce": nonce}
        )
        cmd = self._wrap_cmd([self._python, "-I", "-S", str(runner_path)], tmp)
        t0 = time.monotonic()
        captured, timed_out, launch_err = self._spawn(cmd, payload, tmp, env, timeout_s)
        runtime_ms = (time.monotonic() - t0) * 1000.0

        if launch_err is not None:
            return ExecutionResult(test_index=idx, passed=False, error=launch_err,
                                   error_kind="sandbox_error",
                                   timed_out=False, runtime_ms=runtime_ms)
        if timed_out:
            return ExecutionResult(test_index=idx, passed=False,
                                   error=f"timed out after {timeout_s:.3f}s",
                                   error_kind="timeout",
                                   timed_out=True, runtime_ms=runtime_ms)

        status = _extract_framed(captured, nonce)
        if status is None:
            return ExecutionResult(test_index=idx, passed=False,
                                   error="sandbox produced no verdict (crashed or exited early)",
                                   error_kind="sandbox_error",
                                   timed_out=False, runtime_ms=runtime_ms)

        kind = status.get("status")
        if kind == "ok":
            value = status.get("value")
            passed = values_equal(value, test.expected)
            try:
                rep = json.dumps(value)[:500]
            except Exception:  # noqa: BLE001
                rep = None
            return ExecutionResult(test_index=idx, passed=passed, value=value,
                                   value_ok=True, actual_repr=rep, runtime_ms=runtime_ms)
        if kind == "unserializable":
            return ExecutionResult(test_index=idx, passed=False, value_ok=False,
                                   actual_repr=status.get("actual_repr"),
                                   error=status.get("error", "unserializable return value"),
                                   error_kind="unserializable",
                                   runtime_ms=runtime_ms)
        # error / compile_error
        return ExecutionResult(test_index=idx, passed=False, value_ok=False,
                               error=status.get("error", kind or "execution error"),
                               error_kind=(
                                   "compile_error" if kind == "compile_error"
                                   else "runtime_error"
                               ),
                               runtime_ms=runtime_ms)

    # ------------------------------------------------------------------ #
    def _spawn(
        self, cmd: list[str], payload: str, tmp: str, env: dict[str, str], timeout_s: float
    ) -> tuple[str, bool, Optional[str]]:
        """Run cmd, feed payload on stdin, capture stdout (capped). Returns
        (captured_text, timed_out, launch_error)."""
        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, cwd=tmp, env=env, text=True,
                start_new_session=True,  # own process group -> killpg on timeout
                preexec_fn=_make_limits(timeout_s) if os.name == "posix" else None,
            )
        except Exception as e:  # noqa: BLE001
            return "", False, f"sandbox launch failed: {type(e).__name__}: {e}"

        # Retain the tail, not the head: the runner emits its result after the
        # candidate returns. Continuously drain everything so a noisy candidate
        # cannot block on a full stdout pipe.
        captured = ""

        def _reader() -> None:
            nonlocal captured
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(8192)
                if not chunk:
                    break
                if len(chunk) >= _MAX_OUTPUT_BYTES:
                    captured = chunk[-_MAX_OUTPUT_BYTES:]
                else:
                    captured = (captured + chunk)[-_MAX_OUTPUT_BYTES:]

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()
        try:
            if proc.stdin is not None:
                proc.stdin.write(payload)
                proc.stdin.close()
        except BrokenPipeError:
            pass

        timed_out = False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_group(proc)
            proc.wait()
        reader.join(timeout=1.0)
        return captured, timed_out, None

    @staticmethod
    def _kill_group(proc: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def _wrap_cmd(self, cmd: list[str], tmp: str) -> list[str]:
        """On macOS, confine with sandbox-exec: deny network and deny reads of the
        operator's home (where .env / wallets live), while allowing the temp dir."""
        if not (self._use_seatbelt and sys.platform == "darwin"):
            return cmd
        if not os.path.exists("/usr/bin/sandbox-exec"):
            return cmd
        home = os.path.realpath(os.path.expanduser("~"))
        # Deny reads of the operator's home (secrets/.env/wallets) but RE-ALLOW the
        # Python install + venv (which live under home but hold no secrets) so the
        # interpreter can still boot. A later rule overrides an earlier one.
        allow_read = {
            os.path.realpath(tmp),
            os.path.realpath(tempfile.gettempdir()),
            os.path.realpath(sys.prefix),
            os.path.realpath(sys.base_prefix),
            os.path.dirname(os.path.realpath(self._python)),
        }
        # Deny only file-read-DATA under home (blocks reading .env/wallet CONTENTS)
        # while leaving metadata/traversal allowed, so the interpreter can resolve
        # and load itself from the venv. Re-allow data reads for python/venv/tmp.
        parts = [
            "(version 1)(allow default)(deny network*)(deny file-write*)",
            f'(allow file-write* (subpath "{os.path.realpath(tmp)}"))',
            f'(allow file-write* (subpath "{os.path.realpath(tempfile.gettempdir())}"))',
            f'(deny file-read-data (subpath "{home}"))',
        ]
        parts += [f'(allow file-read-data (subpath "{p}"))' for p in sorted(allow_read)]
        return ["/usr/bin/sandbox-exec", "-p", "".join(parts), *cmd]

    @staticmethod
    def _sandbox_env(tmp: str) -> dict[str, str]:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "HOME": tmp,  # don't expose the real home to the child
            "TMPDIR": tmp,
            "no_proxy": "*",
            "NO_PROXY": "*",
            "http_proxy": "",
            "https_proxy": "",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
        }
        if "SYSTEMROOT" in os.environ:  # Windows interpreter boot
            env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        return env
