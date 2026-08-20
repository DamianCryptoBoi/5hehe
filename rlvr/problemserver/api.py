"""Versioned wire contract for the private problem server.

The security-critical ordering is public task -> committed miner responses ->
hidden tests. A validator never receives tests early enough to pass them into
the miner requests whose signed responses unlock the reveal.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..types import DifficultyBand, Problem, TestCase


class _WireModel(BaseModel):
    """Fail closed if either side accidentally adds private wire fields."""

    model_config = ConfigDict(extra="forbid")


def derive_request_id(challenge_id: str, uid: int, hotkey: str) -> str:
    """Deterministic miner request id verifiable by the server."""
    material = f"rlvr-challenge-v1\0{challenge_id}\0{int(uid)}\0{hotkey}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class PublicChallenge(_WireModel):
    """Everything a miner may see, plus a validator lease id."""

    challenge_id: str = Field(min_length=1, max_length=128)
    problem_id: str = Field(min_length=1, max_length=256)
    language: Literal["python", "rust"]
    statement: str = Field(min_length=1, max_length=500_000)
    entrypoint: str = Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    public_examples: list[TestCase] = Field(default_factory=list, max_length=1000)
    prompt_variant: int = Field(default=0, ge=0, le=2**31 - 1)
    deadline_s: float = Field(default=300.0, gt=0.0, le=3600.0)
    commit_min_responses: int = Field(default=1, ge=1, le=1024)
    commit_min_signed_responses: int = Field(default=1, ge=1, le=1024)
    created_at: float = Field(default=0.0, ge=0.0)
    expires_at: float = Field(default=0.0, ge=0.0)

    def to_problem(self, tests: list[TestCase]) -> Problem:
        """Materialize the validator-local problem after the test reveal."""
        return Problem(
            problem_id=self.problem_id,
            language=self.language,
            statement=self.statement,
            entrypoint=self.entrypoint,
            tests=tests,
            public_examples=self.public_examples,
            created_at=self.created_at,
            prompt_variant=self.prompt_variant,
        )


class ChallengeLeaseRequest(_WireModel):
    """Stable request id makes an ambiguous lease response retry-safe."""

    request_id: str = Field(min_length=1, max_length=128)


class MinerSubmission(_WireModel):
    """One response committed before hidden tests can be revealed."""

    uid: int = Field(ge=0)
    hotkey: str = Field(min_length=1, max_length=256)
    request_id: str = Field(min_length=1, max_length=256)
    response_body: str = ""
    response_headers: dict[str, str] = Field(default_factory=dict, max_length=32)
    error: str = Field(default="", max_length=4096)
    latency_ms: float = Field(default=0.0, ge=0.0)


class ChallengeCommitRequest(_WireModel):
    challenge_id: str = Field(min_length=1, max_length=128)
    submissions: list[MinerSubmission] = Field(default_factory=list, max_length=1024)


class ChallengeCommitResponse(_WireModel):
    accepted: bool = False
    commit_token: str = Field(default="", max_length=128)
    num_submissions: int = Field(default=0, ge=0, le=1024)
    detail: str = Field(default="", max_length=4096)


class ChallengeRevealRequest(_WireModel):
    challenge_id: str = Field(min_length=1, max_length=128)
    commit_token: str = Field(min_length=1, max_length=128)


class ChallengeRevealResponse(_WireModel):
    challenge_id: str = Field(min_length=1, max_length=128)
    language: Literal["python", "rust"]
    tests: list[TestCase] = Field(default_factory=list, max_length=1000)


class FeedbackVerdict(_WireModel):
    """One authenticated miner response graded by the validator."""

    uid: int = Field(ge=0)
    hotkey: str = Field(min_length=1, max_length=256)
    passed: bool


class ChallengeFeedback(_WireModel):
    """Validator-reported curriculum signal; never used as authoritative grading."""

    challenge_id: str = Field(min_length=1, max_length=128)
    pass_rate: float = Field(ge=0.0, le=1.0)
    band: DifficultyBand
    num_responses: int = Field(ge=0)
    dup_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    verdicts: list[FeedbackVerdict] = Field(
        default_factory=list,
        max_length=1024,
        exclude_if=lambda value: not value,
    )
