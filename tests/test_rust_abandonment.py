"""Failure disposition for Rust challenges: abandon, never mass-zero.

docs/RUST_CHALLENGES.md: a malformed rust challenge, a lease/reveal language
mismatch, or an unsupported profile is a PROTOCOL/SERVER fault. The validator
must abandon the challenge — no dispatch (for lease-time faults), no grading,
no score-state mutation, no feedback — because grading a challenge nobody can
pass would zero every honest miner. Miner faults (their code failing) grade
normally; that split is the whole point.

Harness adapted from tests/test_feedback_verdicts.py: drive the REAL
_evaluate_one with a recording client/orchestrator and a recording dispatch.

Also pins the pure validation seam (red until implemented):

    rlvr.problemserver.rust_validation.validate_rust_tests(
        cases, *, require_nonempty=True)   # raises ValueError on violation

Caps are UTF-8 BYTE counts from the agreed contract: stdin <= 65536 bytes per
test, expected stdout <= 32768 bytes per test. One vector uses multi-byte
characters precisely to pin byte (not character) semantics.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rlvr.neurons.decentralized as decentralized
from rlvr.config import Settings
from rlvr.neurons.decentralized import _evaluate_one
from rlvr.problemserver.api import MinerSubmission, PublicChallenge
from rlvr.problemserver.client import LeaseCategory, LeaseOutcome
from rlvr.types import ChallengeResult, DifficultyBand, TestCase


@pytest.fixture(autouse=True)
def assume_rust_environment_ready(monkeypatch):
    """Disposition tests exercise gate LOGIC, so the environment preflight is
    forced ready here; the two readiness-specific tests override it. Without
    this, a real _rust_ready would consult this host's docker state and every
    disposition test would fail for an unrelated reason."""
    from rlvr.neurons import decentralized as dec

    monkeypatch.setattr(dec, "_rust_ready", lambda settings: True, raising=False)


GOOD_EXAMPLE = TestCase(args=["1 2\n"], kwargs={}, expected="3\n")
GOOD_TEST = TestCase(args=["3 4\n"], kwargs={}, expected="7\n")
# args holds two ints instead of one stdin string: the python TestCase shape,
# which is malformed FOR RUST and must abandon, not grade.
MALFORMED_CASE = TestCase(args=[1, 2], kwargs={}, expected=3)


def rust_challenge(examples=None) -> PublicChallenge:
    return PublicChallenge(
        challenge_id="rust-challenge-1",
        problem_id="rust-problem-1",
        statement="Read two integers from stdin; print their sum.",
        entrypoint="main",
        language="rust",
        public_examples=[GOOD_EXAMPLE] if examples is None else examples,
        commit_min_responses=1,
        commit_min_signed_responses=1,
    )


def reveal(tests, language="rust") -> SimpleNamespace:
    return SimpleNamespace(
        challenge_id="rust-challenge-1", tests=tests, language=language
    )


class RecordingClient:
    def __init__(self, challenge, reveal_payload):
        self.sent_feedback = None
        self.commits = []
        self._challenge = challenge
        self._reveal = reveal_payload

    async def lease(self):
        return LeaseOutcome(
            category=LeaseCategory.LEASED, status=200, challenge=self._challenge
        )

    async def commit(self, challenge_id, submissions):
        self.commits.append((challenge_id, len(submissions)))
        return SimpleNamespace(
            accepted=True,
            commit_token="token",
            num_submissions=len(submissions),
            detail="",
        )

    async def reveal(self, challenge_id, commit_token):
        return self._reveal

    async def feedback(self, feedback):
        self.sent_feedback = feedback
        return True


class RecordingOrchestrator:
    """Records grading calls; returns a plausible result so a WRONGLY
    proceeding path completes and the assertion failure stays readable."""

    def __init__(self):
        self.evaluated = False
        self.scored = False

    async def evaluate(self, problem, captured, semaphore):
        self.evaluated = True
        return ChallengeResult(
            problem_id=problem.problem_id,
            outcomes=[],
            pass_rate=0.0,
            band=DifficultyBand.IN_BAND,
            num_responses=0,
        )

    def score_and_export(self, problem, challenge_result, active_uids=None):
        self.scored = True


def install_recording_dispatch(monkeypatch, record: dict):
    async def fake_dispatch(public, subset, concurrency):
        record["dispatched"] = True
        submissions = [
            MinerSubmission(
                uid=1,
                hotkey="hotkey-1",
                request_id="req-1",
                response_body="{}",
                response_headers={"Epistula-Signed-By": "hotkey-1"},
                error="",
            )
        ]
        captured = [SimpleNamespace(uid=1, hotkey="hotkey-1")]
        return submissions, captured

    monkeypatch.setattr(decentralized, "_dispatch_committed", fake_dispatch)


async def run(client, orchestrator, settings=None):
    return await _evaluate_one(
        client,
        orchestrator,
        SimpleNamespace(sample=lambda pool, _k: pool),
        [SimpleNamespace(uid=1, hotkey="hotkey-1")],
        settings
        or Settings(_env_file=None, validator_dispatch_concurrency=4),
    )


# --------------------------------------------------------------------------- #
# Disposition through the real _evaluate_one
# --------------------------------------------------------------------------- #
async def test_malformed_public_example_abandons_before_dispatch(monkeypatch):
    """A rust lease whose public example is not [stdin_str]/stdout_str is a
    malformed challenge: nothing may be dispatched to miners at all."""
    record: dict = {}
    install_recording_dispatch(monkeypatch, record)
    client = RecordingClient(
        rust_challenge(examples=[MALFORMED_CASE]), reveal([GOOD_TEST])
    )
    orchestrator = RecordingOrchestrator()

    await run(client, orchestrator)

    assert "dispatched" not in record, "malformed rust challenge was dispatched"
    assert client.commits == []
    assert not orchestrator.evaluated
    assert not orchestrator.scored
    assert client.sent_feedback is None


async def test_malformed_hidden_tests_abandon_after_reveal_without_scoring(
    monkeypatch,
):
    """Dispatch and commit already happened by reveal time — that is fine and
    unavoidable. What must NOT happen afterward: grading, scoring, feedback."""
    record: dict = {}
    install_recording_dispatch(monkeypatch, record)
    client = RecordingClient(rust_challenge(), reveal([MALFORMED_CASE]))
    orchestrator = RecordingOrchestrator()

    await run(client, orchestrator)

    assert record.get("dispatched"), "dispatch before reveal is expected"
    assert client.commits, "commit before reveal is expected"
    assert not orchestrator.evaluated, "malformed rust tests were graded"
    assert not orchestrator.scored, "malformed rust tests mutated scores"
    assert client.sent_feedback is None, "abandoned challenge sent feedback"


async def test_language_mismatch_at_reveal_abandons(monkeypatch):
    """Leased rust, revealed python: the transactional binding is violated;
    the challenge is abandoned without scoring anyone."""
    record: dict = {}
    install_recording_dispatch(monkeypatch, record)
    client = RecordingClient(
        rust_challenge(), reveal([GOOD_TEST], language="python")
    )
    orchestrator = RecordingOrchestrator()

    await run(client, orchestrator)

    assert not orchestrator.evaluated
    assert not orchestrator.scored
    assert client.sent_feedback is None


async def test_valid_rust_challenge_still_grades(monkeypatch):
    """The control: abandonment logic must not over-trigger. A well-formed
    rust challenge grades and reports feedback normally."""
    record: dict = {}
    install_recording_dispatch(monkeypatch, record)
    client = RecordingClient(rust_challenge(), reveal([GOOD_TEST]))
    orchestrator = RecordingOrchestrator()

    await run(client, orchestrator)

    assert record.get("dispatched")
    assert orchestrator.evaluated
    assert orchestrator.scored
    assert client.sent_feedback is not None


async def test_rust_without_docker_abandons_before_dispatch(monkeypatch):
    """A validator that CANNOT grade rust must abandon the challenge, never
    dispatch it. The failure mode this forbids: the verifier's lazy
    get_executor raises per solution, the orchestrator converts each raise
    into a VERIFY_ERROR zero, and every miner gets a RECORDED zero — the
    mass-zero-by-misconfiguration the design document rules out."""
    record: dict = {}
    install_recording_dispatch(monkeypatch, record)
    client = RecordingClient(rust_challenge(), reveal([GOOD_TEST]))
    orchestrator = RecordingOrchestrator()

    await run(
        client,
        orchestrator,
        Settings(
            _env_file=None,
            executor="subprocess",
            validator_dispatch_concurrency=4,
        ),
    )

    assert "dispatched" not in record, (
        "a rust challenge was dispatched by a validator that cannot grade rust"
    )
    assert not orchestrator.evaluated
    assert not orchestrator.scored
    assert client.sent_feedback is None


async def test_rust_not_ready_abandons_before_dispatch(monkeypatch):
    """Belt and braces below the advertisement: even with the docker
    executor, a failed image-readiness preflight must abandon a rust
    challenge before dispatch — an image that cannot start the supervisor
    turns every submission into a frameless sandbox_error zero otherwise."""
    from rlvr.neurons import decentralized as dec

    monkeypatch.setattr(dec, "_rust_ready", lambda settings: False, raising=False)
    record: dict = {}
    install_recording_dispatch(monkeypatch, record)
    client = RecordingClient(rust_challenge(), reveal([GOOD_TEST]))
    orchestrator = RecordingOrchestrator()

    await run(
        client,
        orchestrator,
        Settings(
            _env_file=None, executor="docker", validator_dispatch_concurrency=4
        ),
    )

    assert "dispatched" not in record
    assert not orchestrator.evaluated
    assert not orchestrator.scored
    assert client.sent_feedback is None


# --------------------------------------------------------------------------- #
# The pure validation seam and its byte-semantics caps
# --------------------------------------------------------------------------- #
def _validate():
    from rlvr.problemserver.rust_validation import validate_rust_tests

    return validate_rust_tests


def test_well_formed_rust_cases_validate():
    _validate()([GOOD_TEST, GOOD_EXAMPLE])


def test_empty_hidden_tests_are_invalid_but_empty_examples_are_not():
    with pytest.raises(ValueError):
        _validate()([])
    _validate()([], require_nonempty=False)  # public-examples mode


@pytest.mark.parametrize(
    "case",
    [
        TestCase(args=[], kwargs={}, expected="x"),
        TestCase(args=["a", "b"], kwargs={}, expected="x"),
        TestCase(args=[7], kwargs={}, expected="x"),
        TestCase(args=["in"], kwargs={"k": 1}, expected="x"),
        TestCase(args=["in"], kwargs={}, expected=7),
        TestCase(args=["in"], kwargs={}, expected=None),
    ],
    ids=[
        "no-args",
        "two-args",
        "non-string-arg",
        "non-empty-kwargs",
        "non-string-expected",
        "null-expected",
    ],
)
def test_malformed_shapes_raise(case):
    with pytest.raises(ValueError):
        _validate()([case])


def test_size_caps_are_utf8_bytes_not_characters():
    """The contract caps are serialized UTF-8 bytes. 33,000 'é' is 33,000
    characters but 66,000 bytes — over the 65,536-byte stdin cap."""
    at_cap = TestCase(args=["a" * 65536], kwargs={}, expected="x")
    over_by_bytes = TestCase(args=["é" * 33000], kwargs={}, expected="x")
    over_ascii = TestCase(args=["a" * 65537], kwargs={}, expected="x")

    _validate()([at_cap])
    with pytest.raises(ValueError):
        _validate()([over_ascii])
    with pytest.raises(ValueError):
        _validate()([over_by_bytes])


def test_expected_stdout_cap_is_32768_bytes():
    at_cap = TestCase(args=["in"], kwargs={}, expected="b" * 32768)
    over = TestCase(args=["in"], kwargs={}, expected="b" * 32769)

    _validate()([at_cap])
    with pytest.raises(ValueError):
        _validate()([over])


def test_hidden_test_count_cap_is_100():
    """The agreed contract caps rust challenges at 100 hidden tests; the
    validator repeats authoring checks, so an over-count reveal is a
    malformed challenge, not a bigger workload. (The wire model alone allows
    1000 — the rust ceiling is the VALIDATION seam's job.)"""
    at_cap = [GOOD_TEST] * 100
    over = [GOOD_TEST] * 101

    _validate()(at_cap)
    with pytest.raises(ValueError):
        _validate()(over)


# --------------------------------------------------------------------------- #
# Sandbox fault attribution
# --------------------------------------------------------------------------- #
def _passing_probe_verification(problem):
    from rlvr.types import ExecutionResult, Verification

    return Verification(
        problem_id=problem.problem_id,
        num_tests=1,
        num_passed=1,
        all_passed=True,
        reward=1.0,
        results=[
            ExecutionResult(
                test_index=0, passed=True, value=7, value_ok=True, runtime_ms=1.0
            )
        ],
    )


def _failing_probe_verification(problem):
    from rlvr.types import ExecutionResult, Verification

    return Verification(
        problem_id=problem.problem_id,
        num_tests=1,
        num_passed=0,
        all_passed=False,
        reward=0.0,
        results=[
            ExecutionResult(
                test_index=0,
                passed=False,
                error="container produced no verdict",
                error_kind="sandbox_error",
                runtime_ms=1.0,
            )
        ],
    )


class LanguageProbeVerifier:
    def __init__(self, rust_healthy: bool):
        self.rust_healthy = rust_healthy
        self.probed_languages: list[str] = []

    def verify(self, problem, solution):
        self.probed_languages.append(problem.language)
        healthy = problem.language == "python" or self.rust_healthy
        if healthy:
            return _passing_probe_verification(problem)
        return _failing_probe_verification(problem)


def _sandbox_error_outcome(uid: int, problem_id: str):
    from rlvr.types import ExecutionResult, MinerOutcome, SolutionResponse, Verification

    return MinerOutcome(
        uid=uid,
        hotkey=f"hotkey-{uid}",
        solution=SolutionResponse(
            problem_id=problem_id, code="fn main() {}", raw_response=""
        ),
        verification=Verification(
            problem_id=problem_id,
            num_tests=1,
            num_passed=0,
            all_passed=False,
            reward=0.0,
            results=[
                ExecutionResult(
                    test_index=0,
                    passed=False,
                    error="container produced no verdict",
                    error_kind="sandbox_error",
                    runtime_ms=1.0,
                )
            ],
        ),
        reward=0.0,
    )


class SandboxErrorOrchestrator(RecordingOrchestrator):
    def __init__(self, verifier, problem_id: str):
        super().__init__()
        self.verifier = verifier
        self._problem_id = problem_id

    async def evaluate(self, problem, captured, semaphore):
        self.evaluated = True
        outcomes = [_sandbox_error_outcome(1, self._problem_id)]
        return ChallengeResult(
            problem_id=problem.problem_id,
            outcomes=outcomes,
            pass_rate=0.0,
            band=DifficultyBand.IN_BAND,
            num_responses=1,
        )


async def test_rust_sandbox_errors_probe_rust_and_discard(monkeypatch):
    record: dict = {}
    install_recording_dispatch(monkeypatch, record)
    verifier = LanguageProbeVerifier(rust_healthy=False)
    orchestrator = SandboxErrorOrchestrator(verifier, "rust-problem-1")
    client = RecordingClient(rust_challenge(), reveal([GOOD_TEST]))

    await run(client, orchestrator)

    assert "rust" in verifier.probed_languages, "probe never exercised rust"
    assert not orchestrator.scored, "rust sandbox failure was scored as miner fault"
    assert client.sent_feedback is None


async def test_python_sandbox_errors_still_probe_python(monkeypatch):
    record: dict = {}
    install_recording_dispatch(monkeypatch, record)
    verifier = LanguageProbeVerifier(rust_healthy=True)
    orchestrator = SandboxErrorOrchestrator(verifier, "python-problem-1")
    challenge = PublicChallenge(
        challenge_id="rust-challenge-1",
        problem_id="python-problem-1",
        language="python",
        statement="Return a + b.",
        entrypoint="add",
        commit_min_responses=1,
        commit_min_signed_responses=1,
    )
    client = RecordingClient(
        challenge, reveal([TestCase(args=[1, 2], kwargs={}, expected=3)],
                          language="python")
    )

    await run(client, orchestrator)

    assert verifier.probed_languages == ["python"]
    assert orchestrator.scored


async def test_failed_rust_probe_downgrades_cached_readiness(monkeypatch):
    from rlvr.neurons import decentralized as dec

    record: dict = {}
    install_recording_dispatch(monkeypatch, record)
    monkeypatch.setattr(dec, "_RUST_READINESS", True, raising=False)
    verifier = LanguageProbeVerifier(rust_healthy=False)
    orchestrator = SandboxErrorOrchestrator(verifier, "rust-problem-1")
    client = RecordingClient(rust_challenge(), reveal([GOOD_TEST]))

    await run(client, orchestrator)

    assert dec._RUST_READINESS is False


def test_sandbox_healthcheck_probes_the_requested_language():
    from rlvr.neurons.decentralized import _sandbox_healthcheck
    from types import SimpleNamespace as NS

    verifier = LanguageProbeVerifier(rust_healthy=False)
    orchestrator = NS(verifier=verifier)

    assert _sandbox_healthcheck(orchestrator, language="python") is True
    assert _sandbox_healthcheck(orchestrator, language="rust") is False
    assert verifier.probed_languages == ["python", "rust"]
