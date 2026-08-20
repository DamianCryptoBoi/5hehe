"""Sandbox executor contract.

An `Executor` runs candidate code against a `Problem`'s hidden tests in an
ISOLATED sandbox and reports one `ExecutionResult` per test. The contract is
intentionally narrow so the verifier can treat any backend identically.

Backends:
  - SubprocessExecutor: development-only process isolation with resource limits.
    It blocks network on macOS when sandbox-exec is available, but not on Linux.
  - DockerExecutor: same contract, inside a `python:3.12-slim` container with
    `--network=none` and mem/cpu limits.

Comparison semantics (shared by all backends): structural equality with a
float tolerance of 1e-6 (see `rlvr.execution.compare.values_equal`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import ExecutionResult, TestCase

# Float tolerance for structural comparison of actual vs. expected.
FLOAT_TOLERANCE = 1e-6


class Executor(ABC):
    """Runs candidate code against a list of tests in an isolated sandbox."""

    @abstractmethod
    def run_tests(
        self,
        code: str,
        entrypoint: str,
        tests: list[TestCase],
        timeout_s: float,
    ) -> list[ExecutionResult]:
        """Execute `code` in an isolated sandbox, locate the function named
        `entrypoint`, call ``entrypoint(*t.args, **t.kwargs)`` for each test, and
        compare the return value to ``t.expected`` using structural equality with
        a 1e-6 float tolerance.

        Returns exactly one `ExecutionResult` per test (same order), each carrying
        ``test_index``, ``passed``, ``actual_repr``, ``error``, ``timed_out`` and
        ``runtime_ms``. ``timeout_s`` bounds EACH test call.

        If the code fails to import/compile or does not define `entrypoint`, return
        one FAILED `ExecutionResult` per test, each with ``.error`` set, so the
        caller can detect a compile failure.
        """
        raise NotImplementedError


def get_executor(settings, language: str) -> Executor:
    """Construct the configured executor.

    ``settings.executor == 'subprocess'`` -> SubprocessExecutor (dev only).
    ``settings.executor == 'docker'``     -> DockerExecutor (default).
    """
    if language not in {"python", "rust"}:
        raise ValueError(f"Unknown challenge language {language!r}.")
    kind = getattr(settings, "executor", "docker")
    if language == "rust":
        if kind != "docker":
            raise RuntimeError("Rust challenge execution requires the Docker executor")
        from .rust_docker_executor import RustDockerExecutor

        return RustDockerExecutor(settings)
    if kind == "subprocess":
        from .subprocess_executor import SubprocessExecutor

        return SubprocessExecutor(settings)
    if kind == "docker":
        from .docker_executor import DockerExecutor

        return DockerExecutor(settings)
    raise ValueError(
        f"Unknown executor {kind!r}; expected 'subprocess' or 'docker'."
    )
