"""Red expectations for the Rust batch runner and executor seams.

Expected seams (docs/RUST_CHALLENGES.md; red until implemented):

  rlvr/execution/_rust_batch_runner.py
      Trusted in-container supervisor, stdlib-only, run as
      ``python -I -S _rust_batch_runner.py`` with a JSON payload on stdin:

        {"compiler_argv": [...],        # trusted host-built compiler command
         "code": "<rust source>",
         "tests": [{"stdin": "<text>"}, ...],
         "timeout_s": <per-test>,
         "compile_timeout_s": <compile deadline>,
         "artifact_max_bytes": <cap>,
         "nonce": "<hex>"}

      It compiles ONCE via ``compiler_argv + [source_path, "-o", artifact]``,
      then runs the artifact per test in a fresh process group, and emits one
      nonce-framed ``{"status":"batch","results":[...]}`` on stdout. Case
      statuses reuse the existing framing vocabulary:
        ok            {"status":"ok","value":"<stdout text>"}
        error         nonzero exit / invalid UTF-8 / oversized output
        timeout       per-test deadline
        compile_error uniform for compile failure, compile timeout, and
                      artifact-cap rejection (text distinguishes them)

  rlvr/execution/rust_docker_executor.py :: RustDockerExecutor
      Host side. The container payload NEVER carries expected values; the
      outer GNU-timeout budget is compile_timeout_s + timeout_s*N + slack(N)
      with the PR #4 slack formula min(60, 5 + 0.025*N).

  rlvr.execution.executor.get_executor(settings, language="python")
      Language-dispatched executor selection; unknown languages fail closed.

The fake toolchain (tests/fixtures/rust/) stands in for rustc and the
compiled artifact so the SUPERVISOR logic is fully exercised without a Rust
toolchain; the real-compiler axis is the deploy-host benchmark gate.
"""

from __future__ import annotations

import json
import pathlib
import secrets
import subprocess
import sys
import time

import pytest

from rlvr.config import Settings
from rlvr.execution.framing import extract_framed
from rlvr.types import TestCase as Case

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "rust"
FAKE_RUSTC = str(FIXTURES / "fake_rustc.py")

RUST_SOURCE = 'fn main() { println!("unused by the fake toolchain"); }\n'


def _runner_path() -> str:
    from rlvr.execution import _rust_batch_runner

    return _rust_batch_runner.__file__


def compiler(mode: str = "ok", size: int = 0) -> list[str]:
    argv = [sys.executable, FAKE_RUSTC, f"--mode={mode}"]
    if size:
        argv.append(f"--size={size}")
    return argv


def run_runner_raw(
    tests: list[str],
    *,
    mode: str = "ok",
    size: int = 0,
    timeout_s: float = 5.0,
    compile_timeout_s: float = 10.0,
    artifact_max_bytes: int = 10_000_000,
):
    """Invoke the real runner subprocess exactly as the container would."""
    nonce = secrets.token_hex(16)
    payload = json.dumps(
        {
            "compiler_argv": compiler(mode, size),
            "code": RUST_SOURCE,
            "tests": [{"stdin": stdin} for stdin in tests],
            "timeout_s": timeout_s,
            "compile_timeout_s": compile_timeout_s,
            "artifact_max_bytes": artifact_max_bytes,
            "nonce": nonce,
        }
    )
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-I", "-S", _runner_path()],
        input=payload.encode("utf-8"),
        capture_output=True,
        timeout=120,
    )
    wall = time.monotonic() - started
    frame = extract_framed(proc.stdout.decode("utf-8", errors="replace"), nonce)
    return frame, proc.stdout, wall


def run_runner(tests: list[str], **kwargs):
    frame, _, wall = run_runner_raw(tests, **kwargs)
    return frame, wall


def results_of(frame) -> list[dict]:
    assert frame is not None, "runner emitted no parseable batch frame"
    assert frame.get("status") == "batch"
    return frame["results"]


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_ok_batch_returns_stdout_values():
    frame, _ = run_runner(["echo\n3\n", "echo\nhello world\n"])
    results = results_of(frame)

    assert [r["status"] for r in results] == ["ok", "ok"]
    assert results[0]["value"] == "3\n"
    assert results[1]["value"] == "hello world\n"
    assert all(isinstance(r["value"], str) for r in results)


# --------------------------------------------------------------------------- #
# Compile phase: uniform, submission-level, separately bounded
# --------------------------------------------------------------------------- #
def test_compile_failure_is_uniform_for_all_cases():
    frame, _ = run_runner(["echo\na", "echo\nb", "echo\nc"], mode="fail")
    results = results_of(frame)

    assert [r["status"] for r in results] == ["compile_error"] * 3
    assert "E0999" in results[0]["error"]
    assert len({r["error"] for r in results}) == 1, "compile error must be uniform"


def test_compile_timeout_is_uniform_and_bounded():
    """The compile deadline is separate from the per-test budget and the
    runner enforces it: a 30s-hanging compiler dies at ~0.5s."""
    frame, wall = run_runner(
        ["echo\na", "echo\nb"], mode="sleep", compile_timeout_s=0.5
    )
    results = results_of(frame)

    assert wall < 10.0, "compile deadline was not enforced"
    assert [r["status"] for r in results] == ["compile_error"] * 2
    assert "time" in results[0]["error"].lower()


def test_oversized_artifact_fails_closed_uniformly():
    frame, _ = run_runner(
        ["echo\na", "echo\nb"],
        mode="bigout",
        size=200_000,
        artifact_max_bytes=50_000,
    )
    results = results_of(frame)

    assert [r["status"] for r in results] == ["compile_error"] * 2
    assert "artifact" in results[0]["error"].lower()


# --------------------------------------------------------------------------- #
# Per-case execution semantics
# --------------------------------------------------------------------------- #
def test_nonzero_exit_fails_the_case_regardless_of_stdout():
    """docs/RUST_CHALLENGES.md pin: a crash after printing is not a correct
    program. Correct payload + exit 3 must NOT be an ok status."""
    frame, _ = run_runner(["exitcode 3\n3\n", "echo\n3\n"])
    results = results_of(frame)

    assert results[0]["status"] == "error"
    assert "exit" in results[0]["error"].lower()
    assert results[1] == {"status": "ok", "value": "3\n"}


def test_per_test_timeout_kills_case_while_sibling_returns():
    frame, wall = run_runner(["sleep 8", "echo\nfast\n"], timeout_s=0.75)
    results = results_of(frame)

    assert wall < 6.0, "runner waited for the sleeping case's full sleep"
    assert results[0]["status"] == "timeout"
    assert results[1] == {"status": "ok", "value": "fast\n"}


def test_stdout_flood_fails_only_that_case():
    """A flooding case must be drained-and-bounded (not deadlocked on a full
    pipe), fail typed, and leave its sibling and the batch frame intact."""
    frame, _ = run_runner(["flood 5000000", "echo\nok\n"], timeout_s=10.0)
    results = results_of(frame)

    assert results[0]["status"] == "error"
    assert "large" in results[0]["error"].lower()
    assert results[1] == {"status": "ok", "value": "ok\n"}


def test_invalid_utf8_output_fails_the_case_typed():
    """Strict decode per the agreed contract — never errors='replace' here."""
    frame, _ = run_runner(["badutf8", "echo\nok\n"])
    results = results_of(frame)

    assert results[0]["status"] == "error"
    assert "utf" in results[0]["error"].lower()
    assert results[1] == {"status": "ok", "value": "ok\n"}


def test_detached_grandchild_does_not_hang_the_batch():
    """A candidate that setsids a 30s child must not stall the runner: the
    case's own result returns and the batch completes promptly."""
    frame, wall = run_runner(["spawn", "echo\nok\n"], timeout_s=2.0)
    results = results_of(frame)

    assert wall < 15.0
    assert results[0] == {"status": "ok", "value": "spawned"}
    assert results[1] == {"status": "ok", "value": "ok\n"}


# --------------------------------------------------------------------------- #
# The O1 class: output budgets are ESCAPED-size aware
# --------------------------------------------------------------------------- #
def test_legal_heavy_escaping_output_survives_intact():
    """Quote-heavy output doubles under JSON escaping. Four legal cases must
    come back byte-exact with the frame parseable."""
    frame, _ = run_runner(["quotes 100000"] * 4, timeout_s=10.0)
    results = results_of(frame)

    assert [r["status"] for r in results] == ["ok"] * 4
    assert all(r["value"] == '"' * 100000 for r in results)


def test_escaped_overflow_degrades_per_case_never_the_frame():
    """Six 650 KB quote outputs escape to ~7.8 MB — beyond any sane aggregate
    budget. The correct failure is PER-CASE typed errors with the batch frame
    still parseable; a raw-size-accounting runner would emit an oversized
    frame, lose it to retention, and zero the whole submission (the O1 bug).
    """
    frame, _ = run_runner(["quotes 650000"] * 6, timeout_s=15.0)
    results = results_of(frame)

    assert len(results) == 6
    assert [r["status"] for r in results] == ["error"] * 6
    assert all("large" in r["error"].lower() for r in results)


def test_aggregate_frame_always_fits_host_retention():
    """The per-case allowance must SCALE WITH N, not sit at a flat constant.

    Ten ~500 KB-escaped outputs each clear a flat 512,000-byte per-case cap,
    and the aggregate frame reaches ~5.0 MB — beyond the host's 4,259,840-byte
    stdout retention, so the leading nonce is truncated away and the WHOLE
    submission scores 0 (measured against the first implementation). With a
    budget-derived allowance the oversized cases degrade to typed errors and
    the frame provably fits retention."""
    from rlvr.execution import docker_executor as dockmod

    frame, raw, _ = run_runner_raw(["quotes 250000"] * 10, timeout_s=15.0)

    assert len(raw) <= dockmod._MAX_STDOUT_BYTES, (
        f"aggregate frame is {len(raw):,} bytes; the host retains only "
        f"{dockmod._MAX_STDOUT_BYTES:,} and the submission would mass-zero"
    )
    results = results_of(frame)
    assert len(results) == 10
    assert all(r["status"] in {"ok", "error"} for r in results)


def test_unrunnable_artifact_fails_closed_with_a_frame():
    """A size-legal artifact the OS refuses to exec must produce a framed,
    typed failure — never an unhandled OSError that kills the runner and turns
    into a frameless sandbox_error for the whole submission."""
    frame, _ = run_runner(["echo\nhi\n", "echo\nbye\n"], mode="noexec")
    results = results_of(frame)

    assert len(results) == 2
    assert all(r["status"] in {"error", "compile_error"} for r in results)


# --------------------------------------------------------------------------- #
# Host executor: payload hygiene, timeout math, language dispatch
# --------------------------------------------------------------------------- #
def _completed(stdout: str = "", rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")


def test_executor_payload_has_exact_keys_and_no_expected(monkeypatch):
    """The security invariant, rust edition: expected values never enter the
    container, and the payload carries exactly the agreed keys."""
    from rlvr.execution import rust_docker_executor as rustmod

    captured: dict = {}

    def fake_bounded(cmd, input_bytes, timeout):
        captured["cmd"] = cmd
        captured["input"] = input_bytes.decode("utf-8")
        captured["timeout"] = timeout
        return _completed(), False, None

    def fake_run(cmd, **kwargs):
        return _completed()

    ex = rustmod.RustDockerExecutor.__new__(rustmod.RustDockerExecutor)
    ex._docker = "docker"
    ex.image = "rust-pinned@sha256:0000000000000000000000000000000000000000"
    ex.memory = "2g"
    ex.cpus = "1.0"
    ex.pids_limit = 128
    ex.compile_timeout_s = 60.0
    ex.artifact_max_bytes = 10_000_000
    monkeypatch.setattr(ex, "_run_bounded", fake_bounded)
    monkeypatch.setattr(subprocess, "run", fake_run)

    ex.run_tests(
        RUST_SOURCE,
        "main",
        [Case(args=["1 2\n"], kwargs={}, expected="SECRET_EXPECTED_VALUE")],
        timeout_s=2.0,
    )

    sentinel = captured["input"].rindex("\n") + 1  # payload follows runner src
    payload = json.loads(captured["input"][sentinel:])
    assert set(payload) == {
        "compiler_argv",
        "code",
        "tests",
        "timeout_s",
        "compile_timeout_s",
        "artifact_max_bytes",
        "nonce",
    }
    assert payload["tests"] == [{"stdin": "1 2\n"}]
    assert "SECRET_EXPECTED_VALUE" not in captured["input"]

    cmd = captured["cmd"]
    assert "--network=none" in cmd
    assert ex.image in cmd
    # Outer budget: compile deadline + N*t + slack(N); watchdog +10 above it.
    assert f"{60.0 + 2.0 * 1 + 5.025:.6f}" in cmd
    assert captured["timeout"] == pytest.approx(60.0 + 2.0 + 5.025 + 10.0)


def test_get_executor_requires_an_explicit_language():
    """V2 cutover: no language default anywhere. A call site that does not
    say what it is grading is a bug, not a python assumption."""
    from rlvr.execution.executor import get_executor

    settings = Settings(_env_file=None, executor="subprocess")

    assert get_executor(settings, language="python") is not None

    with pytest.raises(TypeError):
        get_executor(settings)  # type: ignore[call-arg]

    # Unknown languages fail closed with a typed error, not a dispatch.
    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        get_executor(settings, language="cobol")
    assert "cobol" in str(excinfo.value)

    # Rust remains docker-only: a subprocess-configured validator must get a
    # loud refusal, never a silent python-executor fallback.
    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        get_executor(settings, language="rust")
    assert "rust" in str(excinfo.value).lower()


def test_rust_tmpfs_is_executable_with_explicit_posture(monkeypatch):
    """MEASURED on the benchmark host: Docker mounts the tmpfs noexec by
    default there, so rustc builds the artifact and exec returns Permission
    denied — every submission dies as a sandbox error. The rust tmpfs must
    request exec EXPLICITLY, and because we are now depending on mount
    options rather than defaults, the rest of the posture is spelled out
    too: nosuid and nodev stay, size stays bounded. exec grants nothing
    new in this container — the artifact IS candidate code, and
    no-new-privileges plus cap-drop plus nosuid means nothing executable
    can elevate."""
    from rlvr.execution import rust_docker_executor as rustmod

    captured: dict = {}

    def fake_bounded(cmd, input_bytes, timeout):
        captured["cmd"] = cmd
        return _completed(), False, None

    ex = rustmod.RustDockerExecutor.__new__(rustmod.RustDockerExecutor)
    ex._docker = "docker"
    ex.image = "rust-pinned@sha256:0000000000000000000000000000000000000000"
    ex.memory = "2g"
    ex.cpus = "1.0"
    ex.pids_limit = 128
    ex.compile_timeout_s = 60.0
    ex.artifact_max_bytes = 10_000_000
    monkeypatch.setattr(ex, "_run_bounded", fake_bounded)
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: _completed())

    ex.run_tests(RUST_SOURCE, "main", [Case(args=["1\n"], expected="1")], 2.0)

    cmd = captured["cmd"]
    tmpfs_value = cmd[cmd.index("--tmpfs") + 1]
    assert tmpfs_value.startswith("/tmp:")
    for option in ("exec", "nosuid", "nodev", "size="):
        assert option in tmpfs_value, f"tmpfs posture missing {option!r}"


def _rc_executor(monkeypatch, rc: int, oom: str):
    from rlvr.execution import rust_docker_executor as rustmod

    def fake_bounded(cmd, input_bytes, timeout):
        return _completed(stdout="", rc=rc), False, None

    def fake_run(cmd, **kwargs):
        if "inspect" in cmd:
            return _completed(stdout=f"{oom}\n")
        return _completed()

    ex = rustmod.RustDockerExecutor.__new__(rustmod.RustDockerExecutor)
    ex._docker = "docker"
    ex.image = "rust-pinned@sha256:0000000000000000000000000000000000000000"
    ex.memory = "2g"
    ex.cpus = "1.0"
    ex.pids_limit = 128
    ex.compile_timeout_s = 60.0
    ex.artifact_max_bytes = 10_000_000
    monkeypatch.setattr(ex, "_run_bounded", fake_bounded)
    monkeypatch.setattr(subprocess, "run", fake_run)
    return ex


def test_rust_container_timeout_rc_is_typed_as_timeout(monkeypatch):
    ex = _rc_executor(monkeypatch, rc=124, oom="false")

    results = ex.run_tests(RUST_SOURCE, "main", [Case(args=["1\n"], expected="1")], 2.0)

    assert all(r.timed_out for r in results)


def test_rust_oom_kill_is_typed_as_infrastructure(monkeypatch):
    ex = _rc_executor(monkeypatch, rc=137, oom="true")

    results = ex.run_tests(RUST_SOURCE, "main", [Case(args=["1\n"], expected="1")], 2.0)

    assert all(not r.timed_out for r in results)
    assert all(
        "memory" in (r.error or "").lower() or "oom" in (r.error or "").lower()
        for r in results
    )


def test_rust_kill_escalation_without_oom_is_a_timeout(monkeypatch):
    ex = _rc_executor(monkeypatch, rc=137, oom="false")

    results = ex.run_tests(RUST_SOURCE, "main", [Case(args=["1\n"], expected="1")], 2.0)

    assert all(r.timed_out for r in results)
