"""Shared data contracts for the whole subnet.

These models connect the problem source, validators, responders, sandbox,
scoring, and chain logic. Keep I/O JSON-serializable.
"""

from __future__ import annotations

import enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class NeuronType(str, enum.Enum):
    VALIDATOR = "validator"
    MINER = "miner"


class DifficultyBand(str, enum.Enum):
    EASY = "easy"  # pool pass-rate above band_high
    IN_BAND = "in_band"  # within [band_low, band_high] — the useful zone
    HARD = "hard"  # below band_low (toward 0)
    UNKNOWN = "unknown"  # not yet measured


# --------------------------------------------------------------------------- #
# Problems & tests
# --------------------------------------------------------------------------- #
class TestCase(BaseModel):
    """One authored test: call entrypoint(*args, **kwargs), compare to `expected`.

    Values must be JSON-serializable. Hidden cases stay private until the
    validator has committed the miner responses for a challenge.
    """

    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    expected: Any = None


class Problem(BaseModel):
    """A self-contained coding problem with a hidden, verifiable test suite.

    Deliberately contains no implementation or answer source. Solvability is
    established later by an independent iterative pre-solver.
    """

    model_config = ConfigDict(extra="forbid")

    problem_id: str
    language: Literal["python", "rust"]
    statement: str  # natural-language task shown to the solver
    entrypoint: str  # function name the solver must implement
    tests: list[TestCase]  # hidden tests, never shipped to solvers
    public_examples: list[TestCase] = Field(default_factory=list)  # shown to solver

    created_at: float = 0.0  # unix ts, stamped by caller (not in workflow scripts)

    # measured difficulty (filled after dispatch)
    measured_pass_rate: Optional[float] = None
    band: DifficultyBand = DifficultyBand.UNKNOWN
    num_attempts: int = 0
    # Prompt-policy variant used for the current dispatch (sampled by the
    # private problem service). It is retained in the local audit record.
    prompt_variant: int = 0


# --------------------------------------------------------------------------- #
# Solver responses
# --------------------------------------------------------------------------- #
class SolutionResponse(BaseModel):
    """A miner's answer to a TaskRequest."""

    problem_id: str
    code: str  # extracted source defining `entrypoint`
    raw_response: str = ""  # full model output (incl. fences/prose) for the dataset
    latency_ms: float = 0.0


# --------------------------------------------------------------------------- #
# Execution & verification
# --------------------------------------------------------------------------- #
class ExecutionResult(BaseModel):
    """Outcome of running a candidate against ONE test in the sandbox.

    `passed` is decided by the TRUSTED PARENT (the executor), never by code
    running inside the sandbox. `value` is the JSON-native return value the
    sandbox produced; it is None when execution failed or the value was not
    JSON-serializable."""

    test_index: int
    passed: bool
    value: Any = None  # JSON-native return value (parent-parsed), if any
    value_ok: bool = False  # True iff the sandbox returned a serializable value
    actual_repr: Optional[str] = None
    error: Optional[str] = None  # exception text / traceback summary
    # Machine-readable origin used to distinguish compile/definition failures
    # from a callable that raises for every test input.
    error_kind: Optional[str] = None
    timed_out: bool = False
    runtime_ms: float = 0.0


class Verification(BaseModel):
    """Aggregate verdict for a single solution against the hidden suite."""

    problem_id: str
    num_tests: int
    num_passed: int
    all_passed: bool
    reward: float  # near-binary: 1.0 if all_passed else 0.0 (unless partial-credit)
    compile_error: Optional[str] = None  # syntax/import failure -> reward 0
    results: list[ExecutionResult] = Field(default_factory=list)
    flagged_flaky: bool = False  # verdict changed across verification passes


# --------------------------------------------------------------------------- #
# Round aggregation & farmed data
# --------------------------------------------------------------------------- #
class MinerOutcome(BaseModel):
    """One miner's full result on one problem in one round."""

    uid: int
    hotkey: str = ""
    solution: SolutionResponse
    verification: Verification
    reward: float


class ChallengeResult(BaseModel):
    """Aggregate of a problem dispatched to the miner pool: the basis for the
    pool pass-rate p, the difficulty band, and per-miner scoring."""

    problem_id: str
    outcomes: list[MinerOutcome]
    pass_rate: float  # fraction of responding miners that fully passed = p
    band: DifficultyBand
    num_responses: int
    # Reserved for report compatibility. V1 does not compute this signal.
    dup_ratio: float = 0.0


class Rollout(BaseModel):
    """One validator-local response and its independently verified outcome."""

    problem_id: str
    prompt: str  # canonical rendering of statement + public examples
    system_prompt: str = ""  # the system prompt variant served with it
    prompt_variant: int = 0  # template index (diversity bookkeeping)
    completion: str  # the miner's raw_response
    reward: float
    all_passed: bool
    num_passed: int
    num_tests: int
    pool_pass_rate: float  # difficulty signal at farming time
    band: DifficultyBand
    miner_uid: int
    miner_hotkey: str = ""
    created_at: float = 0.0
