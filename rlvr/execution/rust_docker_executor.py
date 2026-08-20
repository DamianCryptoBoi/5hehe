"""Docker executor for complete Rust stdin/stdout programs."""

from __future__ import annotations

import json
import secrets
import subprocess
import time
from pathlib import Path

from ..types import ExecutionResult, TestCase
from ..policy import RELEASE_POLICY
from . import _rust_batch_runner
from .docker_executor import DockerExecutor, _DOCKER_WATCHDOG_SLACK_S
from .framing import extract_framed
from .rust_judge import outputs_match

_RUNNER_SRC = Path(_rust_batch_runner.__file__).read_text(encoding="utf-8")


class RustDockerExecutor(DockerExecutor):
    def __init__(self, settings=None):
        super().__init__(settings)
        self.image = RELEASE_POLICY.rust_image
        self.memory = RELEASE_POLICY.rust_memory
        self.compile_timeout_s = RELEASE_POLICY.rust_compile_timeout_s
        self.artifact_max_bytes = RELEASE_POLICY.rust_artifact_max_bytes

    def run_tests(self, code, entrypoint, tests, timeout_s):
        if not tests:
            return []
        nonce = secrets.token_hex(16)
        container_name = f"rlvr-rust-{secrets.token_hex(12)}"
        timeout_s = max(0.001, float(timeout_s))
        payload = json.dumps(
            {
                "compiler_argv": [
                    "rustc",
                    f"--edition={RELEASE_POLICY.rust_edition}",
                    *RELEASE_POLICY.rustc_flags,
                ],
                "code": code,
                "tests": [{"stdin": test.args[0]} for test in tests],
                "timeout_s": timeout_s,
                "compile_timeout_s": self.compile_timeout_s,
                "artifact_max_bytes": self.artifact_max_bytes,
                "nonce": nonce,
            }
        )
        sentinel = "\n#==RLVR_RUST_SENTINEL==\n"
        stdin_blob = _RUNNER_SRC + sentinel + payload
        bootstrap = (
            "import sys;d=sys.stdin.read();"
            f"i=d.index({sentinel!r});src=d[:i];payload=d[i+len({sentinel!r}):];"
            "g={'__name__':'__sbx__'};exec(compile(src,'<runner>','exec'),g);"
            "import io;sys.stdin=io.StringIO(payload);g['main']()"
        )
        slack = min(60.0, 5.0 + 0.025 * len(tests))
        total = self.compile_timeout_s + timeout_s * len(tests) + slack
        cmd = [
            self._docker, "run", "--name", container_name, "-i",
            "--network=none", f"--memory={self.memory}",
            f"--memory-swap={self.memory}", f"--cpus={self.cpus}",
            f"--pids-limit={self.pids_limit}", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--user=65534:65534",
            "--read-only", "--tmpfs",
            (
                "/tmp:rw,exec,nosuid,nodev,"
                f"size={RELEASE_POLICY.rust_tmpfs_bytes}"
            ),
            "-w", "/tmp",
            self.image, "timeout", "--signal=TERM", "--kill-after=0.25",
            f"{total:.6f}", "python3", "-I", "-S", "-c", bootstrap,
        ]
        started = time.monotonic()
        oom_killed = False
        try:
            proc, timed_out, launch_error = self._run_bounded(
                cmd, stdin_blob.encode("utf-8"), total + _DOCKER_WATCHDOG_SLACK_S
            )
            if proc is not None and proc.returncode == 137:
                oom_killed = self._container_was_oom_killed(container_name)
        except Exception as error:  # noqa: BLE001
            proc, timed_out, launch_error = None, False, str(error)
        finally:
            cleanup_error = self._remove_container(container_name)
        runtime_ms = (time.monotonic() - started) * 1000.0
        if cleanup_error or timed_out or launch_error or proc is None:
            return self._batch_failure(
                tests, cleanup_error or launch_error or "Rust container timed out",
                runtime_ms, timed_out=timed_out,
            )
        if proc.returncode == 124 or (proc.returncode == 137 and not oom_killed):
            return self._batch_failure(
                tests, "Rust container timed out", runtime_ms, timed_out=True
            )
        if proc.returncode != 0:
            if oom_killed:
                detail = "Rust container exceeded its memory limit"
            else:
                stderr = (proc.stderr or "").strip()
                detail = stderr[-500:] if stderr else f"Rust container exited {proc.returncode}"
            return self._batch_failure(tests, detail, runtime_ms)
        frame = extract_framed(proc.stdout or "", nonce)
        statuses = frame.get("results") if isinstance(frame, dict) else None
        if not isinstance(statuses, list) or len(statuses) != len(tests):
            return self._batch_failure(
                tests, "Rust container produced no verdict", runtime_ms
            )
        return [
            self._parse_rust_status(index, test, status, runtime_ms)
            for index, (test, status) in enumerate(zip(tests, statuses, strict=True))
        ]

    @staticmethod
    def _parse_rust_status(index, test, status, runtime_ms):
        if isinstance(status, dict) and status.get("status") == "ok":
            value = status.get("value")
            passed = isinstance(value, str) and outputs_match(value, test.expected)
            return ExecutionResult(
                test_index=index, passed=passed, value=value, value_ok=True,
                actual_repr=json.dumps(value)[:500], runtime_ms=runtime_ms,
            )
        kind = status.get("status") if isinstance(status, dict) else "error"
        error = (
            status.get("error", "invalid Rust result")
            if isinstance(status, dict)
            else "invalid Rust result"
        )
        error_kind = (
            "compile_error"
            if kind == "compile_error"
            else ("timeout" if kind == "timeout" else "runtime_error")
        )
        return ExecutionResult(
            test_index=index, passed=False, error=error,
            error_kind=error_kind,
            timed_out=kind == "timeout", runtime_ms=runtime_ms,
        )
