"""Tests for the sandbox execution module (SubprocessExecutor).

Exercises pass / fail / exception / timeout / compile-error, plus the structural
float-tolerant comparison and the get_executor selector. No network.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from rlvr.execution import (
    Executor,
    SubprocessExecutor,
    get_executor,
    values_equal,
)
from rlvr.execution import docker_executor as dockmod
from rlvr.execution import _batch_runner as batchmod
from rlvr.execution import subprocess_executor as submod
from rlvr.execution.executor import FLOAT_TOLERANCE
from rlvr.execution.framing import extract_framed
from rlvr.types import ExecutionResult
from rlvr.types import TestCase as Case  # aliased so pytest doesn't try to collect it


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _Settings:
    """Minimal stand-in for rlvr.config.Settings (only .executor is read)."""

    def __init__(self, executor: str = "subprocess"):
        self.executor = executor


def _run_batch_runner(code, tests, timeout_s=0.5):
    nonce = "batch-test-nonce"
    payload = {
        "code": code,
        "entrypoint": "f",
        "tests": tests,
        "timeout_s": timeout_s,
        "nonce": nonce,
    }
    proc = subprocess.run(
        [sys.executable, str(batchmod.__file__)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=max(5.0, timeout_s * len(tests) + 2.0),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = extract_framed(proc.stdout, nonce)
    assert result is not None
    return result


def test_batch_runner_uses_fresh_process_state_for_each_test():
    result = _run_batch_runner(
        "counter = 0\ndef f():\n    global counter\n    counter += 1\n    return counter\n",
        [{"args": [], "kwargs": {}}, {"args": [], "kwargs": {}}],
    )

    assert [item["value"] for item in result["results"]] == [1, 1]


def test_batch_runner_times_out_one_test_and_continues():
    result = _run_batch_runner(
        "import time\ndef f(x):\n    time.sleep(0.2) if x == 1 else None\n    return x\n",
        [{"args": [1]}, {"args": [2]}],
        timeout_s=0.05,
    )

    assert result["results"][0]["status"] == "timeout"
    assert result["results"][1] == {"status": "ok", "value": 2}


def test_batch_runner_reports_compile_failure_for_every_test():
    result = _run_batch_runner(
        "def broken(:\n    pass\n",
        [{"args": []}, {"args": [1]}],
    )

    assert [item["status"] for item in result["results"]] == [
        "compile_error",
        "compile_error",
    ]


@pytest.fixture()
def ex() -> SubprocessExecutor:
    return SubprocessExecutor(_Settings())


ADD_CODE = "def add(a, b):\n    return a + b\n"


# --------------------------------------------------------------------------- #
# comparison helper
# --------------------------------------------------------------------------- #
def test_values_equal_numbers_and_tolerance():
    assert values_equal(1, 1)
    assert values_equal(1.0, 1.0 + FLOAT_TOLERANCE / 10)
    assert not values_equal(1.0, 1.1)
    assert values_equal(0.1 + 0.2, 0.3)  # float fuzz
    assert values_equal(float("nan"), float("nan"))


def test_values_equal_strict_bool():
    # True must not equal 1; protects against type bugs in candidate code.
    assert not values_equal(True, 1)
    assert not values_equal(0, False)
    assert values_equal(True, True)


def test_values_equal_structural():
    assert values_equal([1, 2, 3], (1, 2, 3))  # JSON erases tuple/list
    assert values_equal({"a": 1, "b": [1, 2]}, {"b": [1, 2], "a": 1})
    assert not values_equal([1, 2], [2, 1])
    assert values_equal({1, 2, 3}, {3, 2, 1})


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def test_get_executor_subprocess():
    inst = get_executor(_Settings("subprocess"), language="python")
    assert isinstance(inst, SubprocessExecutor)
    assert isinstance(inst, Executor)


def test_get_executor_unknown_raises():
    with pytest.raises(ValueError):
        get_executor(_Settings("bogus"), language="python")


# --------------------------------------------------------------------------- #
# pass
# --------------------------------------------------------------------------- #
def test_all_tests_pass(ex: SubprocessExecutor):
    tests = [
        Case(args=[2, 3], expected=5),
        Case(args=[-1, 1], expected=0),
        Case(args=[10, 20], kwargs={}, expected=30),
    ]
    results = ex.run_tests(ADD_CODE, "add", tests, timeout_s=5.0)
    assert len(results) == 3
    assert all(isinstance(r, ExecutionResult) for r in results)
    assert all(r.passed for r in results)
    assert [r.test_index for r in results] == [0, 1, 2]
    assert all(r.error is None for r in results)
    assert all(not r.timed_out for r in results)
    assert results[0].actual_repr == "5"
    assert all(r.runtime_ms >= 0.0 for r in results)


def test_kwargs_are_passed(ex: SubprocessExecutor):
    code = "def greet(name, greeting='hi'):\n    return greeting + ' ' + name\n"
    tests = [
        Case(args=["bob"], kwargs={"greeting": "yo"}, expected="yo bob"),
        Case(args=["al"], expected="hi al"),
    ]
    results = ex.run_tests(code, "greet", tests, timeout_s=5.0)
    assert all(r.passed for r in results)


def test_float_tolerant_pass(ex: SubprocessExecutor):
    code = "import math\ndef f(x):\n    return math.sqrt(x)\n"
    tests = [Case(args=[2.0], expected=1.4142135623730951)]
    results = ex.run_tests(code, "f", tests, timeout_s=5.0)
    assert results[0].passed


# --------------------------------------------------------------------------- #
# fail (wrong answer)
# --------------------------------------------------------------------------- #
def test_wrong_answer_fails(ex: SubprocessExecutor):
    tests = [
        Case(args=[2, 3], expected=5),  # ok
        Case(args=[2, 3], expected=6),  # wrong expected -> fail
    ]
    results = ex.run_tests(ADD_CODE, "add", tests, timeout_s=5.0)
    assert results[0].passed is True
    assert results[1].passed is False
    assert results[1].error is None  # wrong answer is not an error
    assert results[1].actual_repr == "5"
    assert not results[1].timed_out


# --------------------------------------------------------------------------- #
# exception during the call
# --------------------------------------------------------------------------- #
def test_runtime_exception_is_failure_with_error(ex: SubprocessExecutor):
    code = "def boom(x):\n    return 1 / x\n"
    tests = [
        Case(args=[2], expected=0.5),  # ok
        Case(args=[0], expected=0),  # ZeroDivisionError
    ]
    results = ex.run_tests(code, "boom", tests, timeout_s=5.0)
    assert results[0].passed
    assert not results[1].passed
    assert results[1].error is not None
    assert "ZeroDivisionError" in results[1].error
    assert not results[1].timed_out


# --------------------------------------------------------------------------- #
# timeout
# --------------------------------------------------------------------------- #
def test_timeout(ex: SubprocessExecutor):
    code = "def slow(x):\n    while True:\n        pass\n"
    tests = [Case(args=[1], expected=1)]
    results = ex.run_tests(code, "slow", tests, timeout_s=1.0)
    assert len(results) == 1
    assert results[0].timed_out is True
    assert results[0].passed is False
    assert results[0].error is not None


def test_timeout_isolated_from_good_tests(ex: SubprocessExecutor):
    """A hanging test must not poison neighbouring tests (per-test subprocess)."""
    code = (
        "def f(x):\n"
        "    if x == 0:\n"
        "        while True:\n"
        "            pass\n"
        "    return x * 2\n"
    )
    tests = [
        Case(args=[3], expected=6),
        Case(args=[0], expected=0),  # hangs
        Case(args=[5], expected=10),
    ]
    results = ex.run_tests(code, "f", tests, timeout_s=1.0)
    assert results[0].passed
    assert results[1].timed_out and not results[1].passed
    assert results[2].passed


# --------------------------------------------------------------------------- #
# compile error
# --------------------------------------------------------------------------- #
def test_syntax_error_marks_all_tests_failed_with_error(ex: SubprocessExecutor):
    code = "def add(a, b)\n    return a + b\n"  # missing colon
    tests = [Case(args=[1, 2], expected=3), Case(args=[4, 5], expected=9)]
    results = ex.run_tests(code, "add", tests, timeout_s=5.0)
    assert len(results) == 2
    assert all(not r.passed for r in results)
    assert all(r.error is not None for r in results)
    assert any("SyntaxError" in (r.error or "") for r in results)


def test_missing_entrypoint_marks_all_tests_failed(ex: SubprocessExecutor):
    code = "def something_else(a, b):\n    return a + b\n"
    tests = [Case(args=[1, 2], expected=3)]
    results = ex.run_tests(code, "add", tests, timeout_s=5.0)
    assert not results[0].passed
    assert results[0].error is not None
    assert "add" in results[0].error


def test_import_error_is_compile_failure(ex: SubprocessExecutor):
    code = "import a_module_that_does_not_exist_xyz\ndef add(a, b):\n    return a + b\n"
    tests = [Case(args=[1, 2], expected=3)]
    results = ex.run_tests(code, "add", tests, timeout_s=5.0)
    assert not results[0].passed
    assert results[0].error is not None
    assert (
        "ModuleNotFoundError" in results[0].error
        or "ImportError" in results[0].error
    )


# --------------------------------------------------------------------------- #
# isolation / misc
# --------------------------------------------------------------------------- #
def test_empty_tests_returns_empty(ex: SubprocessExecutor):
    assert ex.run_tests(ADD_CODE, "add", [], timeout_s=5.0) == []


def test_no_state_leak_between_tests(ex: SubprocessExecutor):
    """Module-global mutation in one test must not be visible in the next."""
    code = (
        "_seen = []\n"
        "def f(x):\n"
        "    _seen.append(x)\n"
        "    return len(_seen)\n"
    )
    tests = [Case(args=[1], expected=1), Case(args=[2], expected=1)]
    results = ex.run_tests(code, "f", tests, timeout_s=5.0)
    assert all(r.passed for r in results)  # each subprocess starts fresh -> len==1


@pytest.mark.skipif(
    sys.platform != "darwin", reason="Linux subprocess backend is explicitly dev-only"
)
def test_network_is_blocked_by_macos_seatbelt(ex: SubprocessExecutor):
    code = (
        "import socket\n"
        "def f(x):\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=1)\n"
        "    return x\n"
    )
    tests = [Case(args=[1], expected=1)]
    results = ex.run_tests(code, "f", tests, timeout_s=5.0)
    assert not results[0].passed
    assert results[0].error is not None


# --------------------------------------------------------------------------- #
# DockerExecutor verdict-parsing contract (no docker daemon required)
#
# The Docker backend cannot run here (no daemon), but its security contract is
# pure: nonce-framed verdict extraction + PARENT-side comparison via
# values_equal, never trusting an in-container verdict. We exercise that logic
# directly against synthetic container stdout via the static `_parse` helper and
# the module-level `_extract_framed`. This keeps Docker in lockstep with the
# subprocess backend / _runner.py protocol.
# --------------------------------------------------------------------------- #
def _completed(stdout: str = "", stderr: str = "", rc: int = 0):
    return subprocess.CompletedProcess(args=["docker"], returncode=rc,
                                       stdout=stdout, stderr=stderr)


def _framed(nonce: str, status: dict) -> str:
    # Mirrors _runner._emit: nonce + json(status) + nonce, with candidate noise.
    return f"candidate noise\n{nonce}{json.dumps(status)}{nonce}trailing junk"


def test_docker_extract_framed_ignores_noise():
    nonce = "n0nce"
    out = _framed(nonce, {"status": "ok", "value": 42})
    assert dockmod._extract_framed(out, nonce) == {"status": "ok", "value": 42}
    # A fake verdict that lacks the nonce must NOT be extracted.
    assert dockmod._extract_framed('{"status":"ok","value":42}', nonce) is None
    assert dockmod._extract_framed("", nonce) is None
    # Only one occurrence of the nonce -> no closing frame -> None.
    assert dockmod._extract_framed(f"{nonce}{{}}", nonce) is None


def test_framed_parser_uses_final_strict_protocol_frame():
    nonce = "frame-marker"
    early = f'{nonce}{json.dumps({"status": "ok", "value": 999})}{nonce}'
    final = f'{nonce}{json.dumps({"status": "ok", "value": 42})}{nonce}'
    output = early + "candidate noise" + final

    assert dockmod._extract_framed(output, nonce) == {"status": "ok", "value": 42}
    assert submod._extract_framed(output, nonce) == {"status": "ok", "value": 42}

    nonstandard = f'{nonce}{{"status":"ok","value":NaN}}{nonce}'
    unknown = f'{nonce}{{"status":"surprise","value":42}}{nonce}'
    extra = f'{nonce}{{"status":"ok","value":42,"passed":true}}{nonce}'
    for malformed in (nonstandard, unknown, extra):
        assert dockmod._extract_framed(malformed, nonce) is None
        assert submod._extract_framed(malformed, nonce) is None


def test_subprocess_keeps_runner_verdict_after_bounded_candidate_noise(
    ex: SubprocessExecutor,
):
    code = (
        "import sys\n"
        "def f():\n"
        f"    sys.stdout.write('x' * {submod._MAX_OUTPUT_BYTES * 2})\n"
        "    return 42\n"
    )

    result = ex.run_tests(code, "f", [Case(expected=42)], timeout_s=5.0)[0]

    assert result.passed and result.value == 42


def test_docker_parse_ok_parent_compares_value():
    nonce = "abc123"
    out = _framed(nonce, {"status": "ok", "value": 5})
    # Correct expected -> parent says passed.
    r = dockmod.DockerExecutor._parse(0, Case(args=[2, 3], expected=5), nonce,
                                      _completed(out), 1.0)
    assert r.passed and r.value == 5 and r.value_ok and r.actual_repr == "5"
    # Wrong expected -> parent says NOT passed (value is still recorded).
    r2 = dockmod.DockerExecutor._parse(0, Case(args=[2, 3], expected=6), nonce,
                                       _completed(out), 1.0)
    assert not r2.passed and r2.value == 5 and r2.value_ok and r2.error is None


def test_docker_parse_does_not_trust_in_container_passed():
    # CRITICAL: even if the container claims it passed, the parent re-derives the
    # verdict from the value vs. the (never-shipped) expected. A forged value that
    # doesn't match expected must FAIL.
    nonce = "deadbeef"
    out = _framed(nonce, {"status": "ok", "value": 999999, "passed": True})
    r = dockmod.DockerExecutor._parse(0, Case(args=[1], expected=42), nonce,
                                      _completed(out), 1.0)
    assert not r.passed  # in-container 'passed' is ignored


def test_docker_parse_unframed_verdict_is_rejected():
    # A candidate printing a bare passing verdict without the nonce cannot forge
    # a verdict -> no frame -> treated as "no verdict" failure.
    r = dockmod.DockerExecutor._parse(
        0, Case(args=[1], expected=999999), "thenonce",
        _completed('{"status":"ok","value":999999}'), 1.0)
    assert not r.passed and not r.value_ok
    assert r.error and "no verdict" in r.error


def test_docker_parse_unserializable_is_failure():
    nonce = "ff00ff00"
    out = _framed(nonce, {"status": "unserializable",
                          "error": "return value not JSON-serializable: Evil",
                          "actual_repr": "<Evil>"})
    r = dockmod.DockerExecutor._parse(0, Case(args=[1], expected=42), nonce,
                                      _completed(out), 1.0)
    assert not r.passed and not r.value_ok
    assert r.actual_repr == "<Evil>" and "JSON-serializable" in r.error


def test_docker_parse_compile_and_runtime_errors():
    nonce = "ff00ff00"
    comp = _framed(nonce, {"status": "compile_error", "error": "SyntaxError: bad"})
    r = dockmod.DockerExecutor._parse(0, Case(args=[1], expected=1), nonce,
                                      _completed(comp), 1.0)
    assert not r.passed and not r.value_ok and "SyntaxError" in r.error
    assert r.error_kind == "compile_error"
    run = _framed(nonce, {"status": "error", "error": "ZeroDivisionError: x"})
    r2 = dockmod.DockerExecutor._parse(0, Case(args=[1], expected=1), nonce,
                                       _completed(run), 1.0)
    assert not r2.passed and "ZeroDivisionError" in r2.error
    assert r2.error_kind == "runtime_error"


def test_docker_parse_oom_and_no_output():
    r = dockmod.DockerExecutor._parse(0, Case(args=[1], expected=1), "n",
                                      _completed(stdout="", rc=137), 1.0)
    assert not r.passed and "OOM" in r.error
    r2 = dockmod.DockerExecutor._parse(0, Case(args=[1], expected=1), "n",
                                       _completed(stdout="", stderr="boom", rc=1), 1.0)
    assert not r2.passed and "boom" in r2.error


def test_docker_swap_disabled_and_expected_not_shipped(monkeypatch):
    # Build the exact docker argv + stdin a run would use, WITHOUT a daemon, by
    # capturing the subprocess.run call. Asserts the MEDIUM swap fix and the
    # core invariant that `expected` never crosses into the container.
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cleanup_cmd"] = cmd
        return _completed(rc=0)

    def fake_bounded(cmd, input_bytes, timeout):
        captured["cmd"] = cmd
        captured["input"] = input_bytes.decode()
        captured["timeout"] = timeout
        return _completed(stdout="", rc=0), False, None

    ex = dockmod.DockerExecutor.__new__(dockmod.DockerExecutor)  # skip _resolve_docker
    ex._docker = "docker"
    ex.image = "python:3.12-slim"
    ex.memory = "256m"
    ex.cpus = "1.0"
    ex.pids_limit = 128
    monkeypatch.setattr(ex, "_run_bounded", fake_bounded)
    monkeypatch.setattr(subprocess, "run", fake_run)

    ex.run_tests("def f(x):\n    return x\n", "f",
                 [Case(args=[1], expected="SECRET_EXPECTED_VALUE")], timeout_s=2.0)

    cmd = captured["cmd"]
    assert "--network=none" in cmd
    assert "--read-only" in cmd
    assert "--name" in cmd
    container_name = cmd[cmd.index("--name") + 1]
    assert container_name.startswith("rlvr-sbx-")
    assert "--rm" not in cmd
    assert "--memory=256m" in cmd
    assert "--memory-swap=256m" in cmd  # MEDIUM: swap disabled (was -1)
    assert "--memory-swap=-1" not in cmd
    assert any(c.startswith("--cpus=") for c in cmd)
    assert any(c.startswith("--pids-limit=") for c in cmd)
    assert "--cap-drop=ALL" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--user=65534:65534" in cmd
    assert cmd[cmd.index("timeout") + 1:cmd.index("python")] == [
        "--signal=TERM",
        "--kill-after=0.25",
        "7.025000",
    ]
    assert captured["timeout"] == 17.025
    assert captured["cleanup_cmd"] == ["docker", "rm", "-f", container_name]
    # The expected answer must NEVER cross into the container. The runner SOURCE
    # (piped before the sentinel) legitimately mentions "expected" in its
    # docstring, so inspect only the JSON payload fed to the runner.
    payload = captured["input"].split("#==RLVR_SBX_SENTINEL==")[-1]
    assert "SECRET_EXPECTED_VALUE" not in captured["input"]  # value never anywhere
    assert "expected" not in json.loads(payload.strip())  # no 'expected' key in payload
    parsed_payload = json.loads(payload.strip())
    assert parsed_payload.keys() == {
        "code", "entrypoint", "tests", "timeout_s", "nonce"
    }
    assert parsed_payload["tests"] == [{"args": [1], "kwargs": {}}]


def test_docker_batches_all_tests_into_one_container(monkeypatch):
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured.setdefault("cleanup", []).append(cmd)
        return _completed(rc=0)

    def fake_bounded(cmd, input_bytes, timeout):
        payload = json.loads(
            input_bytes.decode().split("#==RLVR_SBX_SENTINEL==")[-1]
        )
        captured["run_cmd"] = cmd
        captured["payload"] = payload
        output = _framed(
            payload["nonce"],
            {
                "status": "batch",
                "results": [
                    {"status": "ok", "value": 1},
                    {"status": "ok", "value": 4},
                    {"status": "ok", "value": 8},
                ],
            },
        )
        return _completed(stdout=output), False, None

    ex = dockmod.DockerExecutor.__new__(dockmod.DockerExecutor)
    ex._docker = "docker"
    ex.image = "python:3.12-slim"
    ex.memory = "256m"
    ex.cpus = "1.0"
    ex.pids_limit = 128
    monkeypatch.setattr(ex, "_run_bounded", fake_bounded)
    monkeypatch.setattr(subprocess, "run", fake_run)

    results = ex.run_tests(
        "def f(x): return x * x",
        "f",
        [
            Case(args=[1], expected=1),
            Case(args=[2], expected=4),
            Case(args=[3], expected=9),
        ],
        timeout_s=2.0,
    )

    assert len(captured["cleanup"]) == 1
    assert len(captured["payload"]["tests"]) == 3
    assert [result.passed for result in results] == [True, True, False]


def test_docker_batch_rejects_a_result_count_mismatch():
    nonce = "batch-count"
    proc = _completed(
        _framed(nonce, {"status": "batch", "results": [{"status": "ok", "value": 1}]})
    )

    results = dockmod.DockerExecutor._parse_batch(
        [Case(expected=1), Case(expected=2)], nonce, proc, 1.0
    )

    assert len(results) == 2
    assert all(not result.passed for result in results)
    assert all("invalid result count" in (result.error or "") for result in results)


def test_docker_inner_timeout_is_classified_and_cleaned(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _completed(rc=0)

    def fake_bounded(cmd, input_bytes, timeout):
        calls.append(cmd)
        return _completed(rc=124), False, None

    ex = dockmod.DockerExecutor.__new__(dockmod.DockerExecutor)
    ex._docker = "docker"
    ex.image = "python:3.12-slim"
    ex.memory = "256m"
    ex.cpus = "1.0"
    ex.pids_limit = 128
    monkeypatch.setattr(ex, "_run_bounded", fake_bounded)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ex.run_tests(
        "import time\ndef f():\n    time.sleep(1)\n",
        "f",
        [Case(expected=None)],
        timeout_s=0.1,
    )[0]

    assert result.timed_out and not result.passed
    run_cmd = calls[0]
    container_name = run_cmd[run_cmd.index("--name") + 1]
    assert "5.125000" in run_cmd
    assert calls[-1] == ["docker", "rm", "-f", container_name]


def test_docker_outer_watchdog_still_forces_cleanup(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _completed(rc=0)

    def fake_bounded(cmd, input_bytes, timeout):
        calls.append(cmd)
        return None, True, None

    ex = dockmod.DockerExecutor.__new__(dockmod.DockerExecutor)
    ex._docker = "docker"
    ex.image = "python:3.12-slim"
    ex.memory = "256m"
    ex.cpus = "1.0"
    ex.pids_limit = 128
    monkeypatch.setattr(ex, "_run_bounded", fake_bounded)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ex.run_tests("def f(): return 1", "f", [Case(expected=1)], 0.2)[0]

    assert result.timed_out and not result.passed
    container_name = calls[0][calls[0].index("--name") + 1]
    assert calls[-1] == ["docker", "rm", "-f", container_name]


def test_docker_process_output_is_bounded_and_keeps_tail():
    marker = "TRUSTED-TAIL"
    command = [
        sys.executable,
        "-c",
        f"import sys; sys.stdout.write('x' * {dockmod._MAX_STDOUT_BYTES * 2}); "
        f"sys.stdout.write({marker!r})",
    ]
    result, timed_out, error = dockmod.DockerExecutor._run_bounded(
        command, b"", 5.0
    )

    assert error is None and not timed_out and result is not None
    assert len(result.stdout.encode()) <= dockmod._MAX_STDOUT_BYTES
    assert result.stdout.endswith(marker)


def test_docker_batch_retains_multiple_large_results():
    nonce = "large-batch-nonce"
    expected = "x" * 30_000
    tests = [Case(args=[], expected=expected) for _ in range(10)]
    payload = json.dumps(
        {
            "code": "def f(): return 'x' * 30000",
            "entrypoint": "f",
            "tests": [{"args": [], "kwargs": {}} for _ in tests],
            "timeout_s": 0.5,
            "nonce": nonce,
        }
    ).encode()

    proc, timed_out, error = dockmod.DockerExecutor._run_bounded(
        [sys.executable, str(batchmod.__file__)], payload, 10.0
    )

    assert error is None and not timed_out and proc is not None
    assert len(proc.stdout.encode()) > 256 * 1024
    results = dockmod.DockerExecutor._parse_batch(tests, nonce, proc, 1.0)
    assert all(result.passed for result in results)
