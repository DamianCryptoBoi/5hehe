"""Rust and Python wire contract tests.

The contract requires:

  * `language` ("python" | "rust") is required and explicit on
    PublicChallenge, ChallengeRevealResponse, TaskRequest, and Problem.
    No default, no omission semantics, no exclude_if, no before-validators.
  * Python serializes with its language field present.
  * The lease request is exactly {"request_id": ...}; capability
    negotiation does not exist in V2.
  * Unknown languages still fail closed at parse.
  * TestCase and SolutionPayload shapes are unchanged.

"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import rlvr.protocol as protocol_module
from rlvr.problemserver.api import (
    ChallengeLeaseRequest,
    ChallengeRevealResponse,
    PublicChallenge,
)
from rlvr.protocol import SolutionPayload, TaskRequest, sign_message, verify_signature
from rlvr.types import Problem, TestCase


def challenge_payload(**overrides) -> dict:
    payload = {
        "challenge_id": "chal-v2-1",
        "problem_id": "prob-v2-1",
        "language": "python",
        "statement": "Given n, print n squared.",
        "entrypoint": "solve",
        "public_examples": [{"args": [3], "kwargs": {}, "expected": 9}],
        "prompt_variant": 7,
        "deadline_s": 120.0,
        "commit_min_responses": 2,
        "commit_min_signed_responses": 1,
        "created_at": 1000.0,
        "expires_at": 2000.0,
    }
    payload.update(overrides)
    return payload


def reveal_payload(**overrides) -> dict:
    payload = {
        "challenge_id": "chal-v2-1",
        "language": "python",
        "tests": [{"args": [3], "kwargs": {}, "expected": 9}],
    }
    payload.update(overrides)
    return payload


def task_request_kwargs(**overrides) -> dict:
    kwargs = dict(
        problem_id="prob-v2-1",
        language="python",
        statement="Given n, print n squared.",
        entrypoint="solve",
        public_examples=[TestCase(args=[3], kwargs={}, expected=9)],
        deadline_s=120.0,
        prompt_variant=7,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture(autouse=True)
def force_chain_free_signatures(monkeypatch):
    """Keep signature tests hermetic when a chain-signing package is installed."""

    monkeypatch.setattr(protocol_module, "_HAVE_CRYPTO", False)
    monkeypatch.setattr(protocol_module, "_Keypair", None)


# --------------------------------------------------------------------------- #
# Language is required and explicit on every problem wire.
# --------------------------------------------------------------------------- #
def test_language_is_required_on_the_lease_wire():
    PublicChallenge.model_validate(challenge_payload())

    absent = challenge_payload()
    del absent["language"]
    with pytest.raises(ValidationError):
        PublicChallenge.model_validate(absent)


def test_language_is_required_on_the_reveal_wire():
    ChallengeRevealResponse.model_validate(reveal_payload())

    absent = reveal_payload()
    del absent["language"]
    with pytest.raises(ValidationError):
        ChallengeRevealResponse.model_validate(absent)


def test_language_is_required_on_the_miner_wire():
    TaskRequest(**task_request_kwargs())

    absent = task_request_kwargs()
    del absent["language"]
    with pytest.raises(ValidationError):
        TaskRequest(**absent)


def test_language_is_required_on_problem():
    kwargs = dict(
        problem_id="prob-v2-1",
        statement="s",
        entrypoint="solve",
        tests=[TestCase(args=[1], kwargs={}, expected=1)],
    )
    Problem(**kwargs, language="python")
    with pytest.raises(ValidationError):
        Problem(**kwargs)


def test_python_serializes_present_like_any_other_value():
    """No exclude_if, no omission: the wire says what it means."""
    encoded = json.loads(TaskRequest(**task_request_kwargs()).model_dump_json())
    assert encoded["language"] == "python"

    rust = json.loads(
        TaskRequest(
            **task_request_kwargs(language="rust", entrypoint="main")
        ).model_dump_json()
    )
    assert rust["language"] == "rust"


def test_rust_parses_on_both_response_wires():
    challenge = PublicChallenge.model_validate(
        challenge_payload(language="rust", entrypoint="main")
    )
    reveal = ChallengeRevealResponse.model_validate(reveal_payload(language="rust"))

    assert challenge.language == "rust"
    assert reveal.language == "rust"


def test_unknown_language_fails_closed():
    with pytest.raises(ValidationError):
        PublicChallenge.model_validate(challenge_payload(language="zig"))
    with pytest.raises(ValidationError):
        TaskRequest(**task_request_kwargs(language="zig"))


# --------------------------------------------------------------------------- #
# The lease request contains only its idempotency key.
# --------------------------------------------------------------------------- #
def test_lease_request_is_exactly_request_id():
    body = ChallengeLeaseRequest(request_id="lease-v2-1").model_dump_json()
    assert json.loads(body) == {"request_id": "lease-v2-1"}


def test_lease_request_has_no_capability_field():
    """Reject capability fields that are not part of the lease contract."""
    with pytest.raises(ValidationError):
        ChallengeLeaseRequest(
            request_id="lease-v2-1", supported_languages=["python"]
        )


# --------------------------------------------------------------------------- #
# Stable shapes and signing
# --------------------------------------------------------------------------- #
def test_testcase_and_solution_payload_shapes_unchanged():
    case = TestCase(args=["1 2\n"], kwargs={}, expected="3\n")
    assert json.loads(case.model_dump_json()) == {
        "args": ["1 2\n"],
        "kwargs": {},
        "expected": "3\n",
    }

    payload = SolutionPayload(problem_id="p", code="fn main() {}", raw_response="")
    assert set(json.loads(payload.model_dump_json())) == {
        "problem_id",
        "code",
        "raw_response",
    }


def test_v2_task_request_signs_and_verifies():
    body = TaskRequest(**task_request_kwargs()).model_dump_json().encode("utf-8")

    headers = sign_message("sim-validator", body, signed_for="sim-miner")

    assert verify_signature(headers, body, expected_signed_for="sim-miner")
    assert not verify_signature(headers, body + b" ", expected_signed_for="sim-miner")


def test_statement_field_limit():
    """Keep statements within both lease and miner request body limits."""
    PublicChallenge.model_validate(challenge_payload(statement="x" * 500_000))

    with pytest.raises(ValidationError):
        PublicChallenge.model_validate(
            challenge_payload(statement="x" * 500_001)
        )
    with pytest.raises(ValidationError):
        TaskRequest(**task_request_kwargs(statement="x" * 500_001))
