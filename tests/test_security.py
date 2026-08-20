"""Regression tests for the verifier/sandbox exploits the adversarial review found.
Each test encodes an attack that MUST now fail to score reward."""

import os
import sys
import pytest

from rlvr.execution.subprocess_executor import SubprocessExecutor
from rlvr.types import TestCase as Case


def _run(code: str, entrypoint: str, tests, timeout=5.0):
    return SubprocessExecutor().run_tests(code, entrypoint, tests, timeout)


def test_correct_solution_passes():
    code = "def add(a, b):\n    return a + b\n"
    res = _run(code, "add", [Case(args=[2, 3], expected=5),
                             Case(args=[10, 20], expected=30)])
    assert all(r.passed for r in res)
    assert res[0].value == 5 and res[0].value_ok


def test_wrong_solution_fails():
    code = "def add(a, b):\n    return a - b\n"
    res = _run(code, "add", [Case(args=[2, 3], expected=5)])
    assert not res[0].passed
    assert res[0].value == -1  # parent saw the real (wrong) value


def test_verdict_forgery_via_stdout_and_early_exit_is_defeated():
    # CRITICAL #1: candidate prints a fake passing verdict and exits before the
    # real verdict is emitted. Unframed candidate stdout is not a result -> fail.
    code = (
        "import sys, os\n"
        "def solve(x):\n"
        "    sys.stdout.write('{\"status\":\"ok\",\"value\":999999}')\n"
        "    sys.stdout.flush()\n"
        "    os._exit(0)\n"
    )
    res = _run(code, "solve", [Case(args=[1], expected=999999)])
    assert not res[0].passed  # forged value must NOT be accepted


def test_eq_override_reward_hack_is_defeated():
    # CRITICAL #3: an object whose __eq__ always returns True. The sandbox
    # strict-JSON-serializes the value, so the fake __eq__ never runs in the
    # parent; a non-serializable object is rejected outright.
    code = (
        "class Evil:\n"
        "    def __eq__(self, other):\n"
        "        return True\n"
        "    def __ne__(self, other):\n"
        "        return False\n"
        "def solve(x):\n"
        "    return Evil()\n"
    )
    res = _run(code, "solve", [Case(args=[1], expected=42)])
    assert not res[0].passed
    assert not res[0].value_ok  # rejected as unserializable


def test_data_write_to_operator_home_is_blocked():
    # CRITICAL #2 (write side): untrusted code must not persist data into the
    # operator's home (seatbelt deny-write on macOS; RLIMIT_FSIZE=0 on Linux).
    if os.name != "posix":
        pytest.skip("POSIX only")
    target = os.path.join(os.path.expanduser("~"), ".rlvr_write_probe.txt")
    if os.path.exists(target):
        os.remove(target)
    code = (
        f"def solve(x):\n"
        f"    with open({target!r}, 'w') as f:\n"
        f"        f.write('pwned')\n"
        f"    return True\n"
    )
    try:
        res = _run(code, "solve", [Case(args=[1], expected=True)])
        persisted = os.path.exists(target) and open(target).read() == "pwned"
        assert not persisted  # the secret data must not be on disk
        assert not res[0].passed  # the write raised -> fn failed
    finally:
        if os.path.exists(target):
            os.remove(target)


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec read confinement is macOS-only")
def test_home_read_is_blocked_on_macos():
    # CRITICAL #2 (read side): on macOS the sandbox-exec profile denies reads of
    # the operator's home dir, so candidate code cannot exfiltrate ~/... secrets.
    secret_path = os.path.join(os.path.expanduser("~"), ".rlvr_secret_probe")
    with open(secret_path, "w") as f:
        f.write("TOP_SECRET")
    try:
        code = (
            f"def solve(x):\n"
            f"    with open({secret_path!r}) as f:\n"
            f"        return f.read()\n"
        )
        res = _run(code, "solve", [Case(args=[1], expected="TOP_SECRET")])
        assert not res[0].passed  # the read must be denied -> fn failed
    finally:
        os.remove(secret_path)


def test_timeout_is_enforced():
    code = "def solve(x):\n    while True:\n        pass\n"
    res = _run(code, "solve", [Case(args=[1], expected=1)], timeout=1.0)
    assert res[0].timed_out and not res[0].passed
