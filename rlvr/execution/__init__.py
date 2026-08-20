"""Sandboxed execution of candidate code against verifiable tests.

Exported surface:
    Executor              - the backend-agnostic ABC.
    get_executor(settings, language) - language-aware backend factory.
    SubprocessExecutor    - explicit development backend; one subprocess/test.
    DockerExecutor        - production default; container with --network=none.
    values_equal          - structural, float-tolerant comparison helper.
"""

from __future__ import annotations

from .compare import FLOAT_TOLERANCE, values_equal
from .executor import Executor, get_executor
from .subprocess_executor import SubprocessExecutor

__all__ = [
    "Executor",
    "get_executor",
    "SubprocessExecutor",
    "FLOAT_TOLERANCE",
    "values_equal",
    "DockerExecutor",
]


def __getattr__(name: str):
    # Lazily expose DockerExecutor so importing the package never requires Docker.
    if name == "DockerExecutor":
        from .docker_executor import DockerExecutor

        return DockerExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
