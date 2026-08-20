"""DockerExecutor: run candidate code inside a python:3.12-slim container.

Same `Executor` contract as SubprocessExecutor, but with real kernel-level
isolation suitable for untrusted code:
  - ``--network=none``        : no network at all.
  - ``--memory`` / ``--cpus`` : bounded memory and CPU.
  - ``--memory-swap``         : equal to ``--memory`` (swap DISABLED, so a
    candidate cannot escape the RAM cap by swapping to disk).
  - ``--read-only`` + tmpfs   : no persistent writes; only an ephemeral /tmp.
  - ``--pids-limit``          : fork-bomb resistance.
  - dropped Linux capabilities, no-new-privileges, and an unprivileged uid.
  - stdout/stderr are continuously drained but retained only up to hard caps.
  - One container per submitted program; each test runs in a fresh child
    process with its own timeout.

Security posture MIRRORS SubprocessExecutor (see that module + _runner.py):
  - The container is NEVER given the expected answer; the parent (this class)
    compares the returned value via `compare.values_equal`, so a candidate can
    neither forge a passing verdict nor smuggle a fake __eq__ across the JSON
    boundary.
  - The verdict is wrapped in a per-call random marker so the parent can locate
    it amid candidate stdout. The marker is framing, not authentication; the
    security boundary is strict JSON plus parent-side comparison to an expected
    value that never enters the container.
  - We do NOT trust any in-container ``passed`` flag (the runner never emits
    one): the parent decides pass/fail from the returned value.

If the Docker CLI / daemon is unavailable, construction raises a clear
`RuntimeError` so callers can fall back to the subprocess backend explicitly.

The parser, command construction, resource flags, and bounded-I/O runner are
covered without a daemon. A real-daemon smoke test remains a deployment check.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ..types import ExecutionResult, TestCase
from . import _batch_runner
from .compare import values_equal
from .executor import Executor
from .framing import extract_framed as _extract_framed

_RUNNER_SRC = Path(_batch_runner.__file__).read_text(encoding="utf-8")
_DOCKER_WATCHDOG_SLACK_S = 10.0
_CONTAINER_CLEANUP_TIMEOUT_S = 10.0
_MAX_STDOUT_BYTES = _batch_runner._MAX_BATCH_BYTES + 64 * 1024
_MAX_STDERR_BYTES = 64 * 1024


class DockerExecutor(Executor):
    def __init__(self, settings: Any = None):
        self.settings = settings
        self.image = getattr(settings, "docker_image", None) or "python:3.12-slim"
        self.memory = getattr(settings, "docker_memory", None) or "256m"
        self.cpus = str(getattr(settings, "docker_cpus", None) or "1.0")
        self.pids_limit = int(getattr(settings, "docker_pids_limit", None) or 128)
        self._docker = self._resolve_docker()

    @staticmethod
    def _resolve_docker() -> str:
        docker = shutil.which("docker")
        if not docker:
            raise RuntimeError(
                "DockerExecutor requires the 'docker' CLI on PATH, but it was not "
                "found. Install Docker or set settings.executor='subprocess'."
            )
        try:
            proc = subprocess.run(
                [docker, "info"],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"DockerExecutor could not contact the Docker daemon ('docker info' "
                f"failed: {type(e).__name__}: {e}). Is Docker running?"
            ) from e
        if proc.returncode != 0:
            raise RuntimeError(
                "DockerExecutor could not contact the Docker daemon "
                f"('docker info' rc={proc.returncode}: {proc.stderr.strip()[:300]}). "
                "Is Docker running?"
            )
        return docker

    def run_tests(
        self,
        code: str,
        entrypoint: str,
        tests: list[TestCase],
        timeout_s: float,
    ) -> list[ExecutionResult]:
        if not tests:
            return []
        return self._run_batch(code, entrypoint, tests, timeout_s)

    # ------------------------------------------------------------------ #
    def _run_batch(
        self,
        code: str,
        entrypoint: str,
        tests: list[TestCase],
        timeout_s: float,
    ) -> list[ExecutionResult]:
        nonce = secrets.token_hex(16)
        container_name = f"rlvr-sbx-{secrets.token_hex(12)}"
        timeout_s = max(0.001, float(timeout_s))
        # Expected answers are intentionally omitted. The host compares returned
        # values only after the container exits.
        payload = json.dumps(
            {
                "code": code,
                "entrypoint": entrypoint,
                "tests": [
                    {"args": test.args, "kwargs": test.kwargs} for test in tests
                ],
                "timeout_s": timeout_s,
                "nonce": nonce,
            }
        )
        # The runner source is too large for argv, so we pipe BOTH the runner src
        # and the JSON payload on stdin, separated by a sentinel. A tiny bootstrap
        # (small enough for `-c`) splits them, exec's the runner, then feeds the
        # payload to the runner's main() via a replaced stdin.
        sentinel = "\n#==RLVR_SBX_SENTINEL==\n"
        stdin_blob = _RUNNER_SRC + sentinel + payload
        bootstrap = (
            "import sys;"
            "d=sys.stdin.read();"
            f"i=d.index({sentinel!r});"
            "src=d[:i];"
            f"payload=d[i+len({sentinel!r}):];"
            "g={'__name__':'__sbx__'};"
            "exec(compile(src,'<runner>','exec'),g);"
            "import io;sys.stdin=io.StringIO(payload);"
            "g['main']()"
        )
        timeout_slack_s = min(60.0, 5.0 + 0.025 * len(tests))
        total_timeout_s = timeout_s * len(tests) + timeout_slack_s
        cmd = [
            self._docker,
            "run",
            "--name",
            container_name,
            "-i",
            "--network=none",
            f"--memory={self.memory}",
            # Swap == memory disables swap entirely, so the RAM cap is real and a
            # candidate cannot evade it (was --memory-swap=-1 / unlimited swap).
            f"--memory-swap={self.memory}",
            f"--cpus={self.cpus}",
            f"--pids-limit={self.pids_limit}",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user=65534:65534",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=64m",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-w",
            "/tmp",
            self.image,
            # Enforce the candidate deadline INSIDE the container. The outer
            # subprocess timeout below is only an emergency Docker-daemon
            # watchdog and must not be the candidate's effective deadline.
            "timeout",
            "--signal=TERM",
            "--kill-after=0.25",
            f"{total_timeout_s:.6f}",
            "python",
            "-I",
            "-S",
            "-c",
            bootstrap,
        ]
        t0 = time.monotonic()
        proc: Optional[subprocess.CompletedProcess[str]]
        launch_error: Optional[str]
        watchdog_timed_out: bool
        oom_killed = False
        try:
            proc, watchdog_timed_out, launch_error = self._run_bounded(
                cmd,
                stdin_blob.encode("utf-8"),
                total_timeout_s + _DOCKER_WATCHDOG_SLACK_S,
            )
            # GNU timeout normally exits 124. If candidate code ignores TERM,
            # --kill-after escalates to KILL and Docker reports 137—the same
            # status as an OOM kill. Inspect the still-named container before
            # cleanup so those cases remain distinguishable.
            if proc is not None and proc.returncode == 137:
                oom_killed = self._container_was_oom_killed(container_name)
        except Exception as e:  # noqa: BLE001
            proc = None
            watchdog_timed_out = False
            launch_error = f"docker launch failed: {type(e).__name__}: {e}"
        finally:
            cleanup_error = self._remove_container(container_name)

        runtime_ms = (time.monotonic() - t0) * 1000.0
        if cleanup_error is not None:
            return self._batch_failure(tests, cleanup_error, runtime_ms)
        if watchdog_timed_out or (
            proc is not None
            and (proc.returncode == 124 or (proc.returncode == 137 and not oom_killed))
        ):
            return self._batch_failure(
                tests,
                f"container timed out after {total_timeout_s:.3f}s",
                runtime_ms,
                timed_out=True,
            )
        if launch_error is not None or proc is None:
            return self._batch_failure(
                tests,
                launch_error or "docker launch failed without a result",
                runtime_ms,
            )
        return self._parse_batch(tests, nonce, proc, runtime_ms)

    @staticmethod
    def _batch_failure(
        tests: list[TestCase],
        error: str,
        runtime_ms: float,
        *,
        timed_out: bool = False,
    ) -> list[ExecutionResult]:
        return [
            ExecutionResult(
                test_index=index,
                passed=False,
                actual_repr=None,
                error=error,
                error_kind="timeout" if timed_out else "sandbox_error",
                timed_out=timed_out,
                runtime_ms=runtime_ms,
            )
            for index, _ in enumerate(tests)
        ]

    @classmethod
    def _parse_batch(
        cls,
        tests: list[TestCase],
        nonce: str,
        proc: subprocess.CompletedProcess[str],
        runtime_ms: float,
    ) -> list[ExecutionResult]:
        payload = _extract_framed(proc.stdout or "", nonce)
        if not isinstance(payload, dict) or payload.get("status") != "batch":
            stderr = (proc.stderr or "").strip()
            if proc.returncode == 137:
                detail = "container killed (likely OOM / memory limit)"
            else:
                detail = stderr[-500:] if stderr else f"no output (rc={proc.returncode})"
            return cls._batch_failure(
                tests,
                f"container produced no verdict: {detail}",
                runtime_ms,
            )
        statuses = payload.get("results")
        if not isinstance(statuses, list) or len(statuses) != len(tests):
            return cls._batch_failure(
                tests,
                "container returned an invalid result count",
                runtime_ms,
            )
        return [
            cls._parse_status(index, test, status, runtime_ms)
            for index, (test, status) in enumerate(zip(tests, statuses, strict=True))
        ]

    @staticmethod
    def _run_bounded(
        command: list[str], stdin_bytes: bytes, timeout_s: float
    ) -> tuple[
        Optional[subprocess.CompletedProcess[str]], bool, Optional[str]
    ]:
        """Run Docker while draining stdout/stderr into fixed-size tail buffers.

        ``subprocess.run(capture_output=True)`` buffers without limit. Candidate
        code controls both streams and can otherwise exhaust validator memory
        before its timeout. Reader threads continuously drain the pipes so the
        container cannot block on a full pipe, while only the last bounded
        segment is retained (the trusted runner verdict is emitted last).
        """
        try:
            child = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except Exception as error:  # noqa: BLE001
            return None, False, (
                f"docker launch failed: {type(error).__name__}: {error}"
            )

        stdout = bytearray()
        stderr = bytearray()
        truncated = {"stdout": False, "stderr": False}

        def drain(pipe, target: bytearray, limit: int, name: str) -> None:
            try:
                while True:
                    chunk = pipe.read(8192)
                    if not chunk:
                        break
                    overflow = len(target) + len(chunk) - limit
                    if overflow > 0:
                        truncated[name] = True
                        del target[: min(overflow, len(target))]
                        if len(chunk) >= limit:
                            target.clear()
                            chunk = chunk[-limit:]
                    target.extend(chunk)
            finally:
                pipe.close()

        def feed() -> None:
            try:
                if child.stdin is not None:
                    child.stdin.write(stdin_bytes)
                    child.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                if child.stdin is not None:
                    try:
                        child.stdin.close()
                    except OSError:
                        pass

        assert child.stdout is not None and child.stderr is not None
        stdout_thread = threading.Thread(
            target=drain,
            args=(child.stdout, stdout, _MAX_STDOUT_BYTES, "stdout"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=drain,
            args=(child.stderr, stderr, _MAX_STDERR_BYTES, "stderr"),
            daemon=True,
        )
        stdin_thread = threading.Thread(target=feed, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        stdin_thread.start()

        timed_out = False
        try:
            child.wait(timeout=max(0.001, float(timeout_s)))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(child.pid), signal.SIGKILL)
                else:  # pragma: no cover - production validator is Linux
                    child.kill()
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    child.kill()
                except OSError:
                    pass
            child.wait()

        stdin_thread.join(timeout=1.0)
        stdout_thread.join(timeout=2.0)
        stderr_thread.join(timeout=2.0)
        stdout_text = bytes(stdout).decode("utf-8", errors="replace")
        stderr_text = bytes(stderr).decode("utf-8", errors="replace")
        if truncated["stdout"]:
            stderr_text = "<stdout truncated; tail retained>\n" + stderr_text
        if truncated["stderr"]:
            stderr_text = "<stderr truncated; tail follows>\n" + stderr_text
        completed = subprocess.CompletedProcess(
            args=command,
            returncode=int(child.returncode),
            stdout=stdout_text,
            stderr=stderr_text,
        )
        return completed, timed_out, None

    def _container_was_oom_killed(self, container_name: str) -> bool:
        """Inspect a stopped container before cleanup to disambiguate rc=137."""
        try:
            proc = subprocess.run(
                [
                    self._docker,
                    "inspect",
                    "--format",
                    "{{.State.OOMKilled}}",
                    container_name,
                ],
                capture_output=True,
                text=True,
                timeout=_CONTAINER_CLEANUP_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001 - cleanup still runs; fail closed as timeout
            return False
        return proc.returncode == 0 and proc.stdout.strip().lower() == "true"

    def _remove_container(self, container_name: str) -> Optional[str]:
        """Force-remove this invocation's uniquely named container."""
        try:
            proc = subprocess.run(
                [self._docker, "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=_CONTAINER_CLEANUP_TIMEOUT_S,
            )
        except Exception as e:  # noqa: BLE001
            return f"docker cleanup failed for {container_name}: {type(e).__name__}: {e}"
        if proc.returncode == 0 or "No such container" in (proc.stderr or ""):
            return None
        detail = (proc.stderr or proc.stdout or "unknown error").strip()[-500:]
        return f"docker cleanup failed for {container_name}: {detail}"

    @classmethod
    def _parse(
        cls,
        idx: int,
        test: TestCase,
        nonce: str,
        proc: subprocess.CompletedProcess[str],
        runtime_ms: float,
    ) -> ExecutionResult:
        status = _extract_framed(proc.stdout or "", nonce)
        if status is None:
            stderr = (proc.stderr or "").strip()
            # OOM-killed containers exit 137; surface that clearly.
            if proc.returncode == 137:
                tail = "container killed (likely OOM / memory limit)"
            else:
                tail = stderr[-500:] if stderr else f"no output (rc={proc.returncode})"
            return ExecutionResult(
                test_index=idx,
                passed=False,
                value_ok=False,
                actual_repr=None,
                error=f"container produced no verdict: {tail}",
                error_kind="sandbox_error",
                timed_out=False,
                runtime_ms=runtime_ms,
            )

        return cls._parse_status(idx, test, status, runtime_ms)

    @staticmethod
    def _parse_status(
        idx: int,
        test: TestCase,
        status: object,
        runtime_ms: float,
    ) -> ExecutionResult:
        if not isinstance(status, dict):
            return ExecutionResult(
                test_index=idx,
                passed=False,
                value_ok=False,
                error="container returned an invalid test result",
                error_kind="sandbox_error",
                timed_out=False,
                runtime_ms=runtime_ms,
            )
        kind = status.get("status")
        if kind == "ok":
            value = status.get("value")
            # The TRUSTED PARENT decides pass/fail; the container never sees
            # `expected` and never reports `passed`.
            passed = values_equal(value, test.expected)
            try:
                rep = json.dumps(value)[:500]
            except Exception:  # noqa: BLE001
                rep = None
            return ExecutionResult(
                test_index=idx,
                passed=passed,
                value=value,
                value_ok=True,
                actual_repr=rep,
                timed_out=False,
                runtime_ms=runtime_ms,
            )
        if kind == "unserializable":
            return ExecutionResult(
                test_index=idx,
                passed=False,
                value_ok=False,
                actual_repr=status.get("actual_repr"),
                error=status.get("error", "unserializable return value"),
                error_kind="unserializable",
                timed_out=False,
                runtime_ms=runtime_ms,
            )
        if kind == "timeout":
            return ExecutionResult(
                test_index=idx,
                passed=False,
                value_ok=False,
                error=status.get("error", "test timed out"),
                error_kind="timeout",
                timed_out=True,
                runtime_ms=runtime_ms,
            )
        # error / compile_error
        return ExecutionResult(
            test_index=idx,
            passed=False,
            value_ok=False,
            error=status.get("error", kind or "execution error"),
            error_kind=(
                "compile_error" if kind == "compile_error" else "runtime_error"
            ),
            timed_out=False,
            runtime_ms=runtime_ms,
        )
