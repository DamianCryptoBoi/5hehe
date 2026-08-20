"""Focused behavior checks for the live decentralized validator."""

import asyncio
from types import SimpleNamespace

from rlvr.config import Settings
from rlvr.neurons.decentralized import (
    _dispatch_committed,
    _evaluate_one,
    _sandbox_error_rate,
    _sandbox_healthcheck,
    _validator_http_limits,
    _weight_result_status,
)
from rlvr.problemserver.api import PublicChallenge
from rlvr.problemserver.client import LeaseCategory, LeaseOutcome
from rlvr.types import (
    ChallengeResult,
    DifficultyBand,
    ExecutionResult,
    MinerOutcome,
    SolutionResponse,
    TestCase as Case,
    Verification,
)


class _ExtrinsicResponse:
    def __init__(self, success: bool, message: str):
        self.success = success
        self.message = message


def test_weight_result_status_uses_extrinsic_success_not_object_truthiness():
    result = _ExtrinsicResponse(False, "rejected by chain")

    assert bool(result)
    assert _weight_result_status(result) == (False, "rejected by chain")


def test_weight_result_status_retains_legacy_return_shapes():
    assert _weight_result_status((True, "included")) == (True, "included")
    assert _weight_result_status(False) == (False, "")


def test_http_pool_can_dispatch_the_complete_miner_wave():
    limits = _validator_http_limits(
        Settings(_env_file=None, validator_dispatch_concurrency=256)
    )

    assert limits.max_connections == 264
    assert limits.max_keepalive_connections == 64


def _outcome(uid, *, error_kind=None, compile_error=None):
    return MinerOutcome(
        uid=uid,
        hotkey=f"hotkey-{uid}",
        solution=SolutionResponse(problem_id="problem", code=""),
        verification=Verification(
            problem_id="problem",
            num_tests=1,
            num_passed=0,
            all_passed=False,
            reward=0.0,
            compile_error=compile_error,
            results=[
                ExecutionResult(
                    test_index=0,
                    passed=False,
                    error="failed",
                    error_kind=error_kind,
                )
            ],
        ),
        reward=0.0,
    )


def test_sandbox_error_rate_counts_executor_and_verifier_failures():
    result = ChallengeResult(
        problem_id="problem",
        outcomes=[
            _outcome(0, error_kind="sandbox_error"),
            _outcome(1, compile_error="VERIFY_ERROR: daemon unavailable"),
            _outcome(2, error_kind="runtime_error"),
            _outcome(3),
        ],
        pass_rate=0.0,
        band=DifficultyBand.HARD,
        num_responses=4,
    )

    assert _sandbox_error_rate(result) == 0.5


def test_sandbox_healthcheck_requires_a_clean_known_good_result():
    class Verifier:
        def __init__(self, verification):
            self.verification = verification
            self.calls = []

        def verify(self, problem, solution):
            self.calls.append((problem, solution))
            return self.verification

    good = ExecutionResult(test_index=0, passed=True, value=7, value_ok=True)
    verifier = Verifier(
        Verification(
            problem_id="__rlvr_sandbox_healthcheck__",
            num_tests=1,
            num_passed=1,
            all_passed=True,
            reward=1.0,
            results=[good],
        )
    )
    orchestrator = type(
        "Orchestrator",
        (),
        {"verifier": verifier},
    )()

    assert _sandbox_healthcheck(orchestrator, "python") is True
    assert verifier.calls[0][0].entrypoint == "__rlvr_sandbox_healthcheck__"
    assert verifier.calls[0][1].code.endswith("    return value\n")

    verifier.verification = Verification(
        problem_id="__rlvr_sandbox_healthcheck__",
        num_tests=1,
        num_passed=0,
        all_passed=False,
        reward=0.0,
        results=[
            ExecutionResult(
                test_index=0,
                passed=False,
                error="daemon unavailable",
                error_kind="sandbox_error",
            )
        ],
    )
    assert _sandbox_healthcheck(orchestrator, "python") is False


async def test_broken_sandbox_probe_prevents_score_update_and_feedback(monkeypatch):
    import rlvr.neurons.decentralized as decentralized

    result = ChallengeResult(
        problem_id="problem",
        outcomes=[_outcome(7, error_kind="sandbox_error")],
        pass_rate=0.0,
        band=DifficultyBand.HARD,
        num_responses=1,
    )

    class Client:
        feedback_called = False

        async def lease(self):
            return LeaseOutcome(
                category=LeaseCategory.LEASED,
                status=200,
                challenge=PublicChallenge(
                    challenge_id="challenge",
                    problem_id="problem",
                    language="python",
                    statement="Return the input.",
                    entrypoint="solve",
                ),
            )

        async def commit(self, challenge_id, submissions):
            return SimpleNamespace(
                accepted=True,
                commit_token="token",
                num_submissions=len(submissions),
                detail="",
            )

        async def reveal(self, challenge_id, commit_token):
            return SimpleNamespace(
                challenge_id=challenge_id,
                language="python",
                tests=[Case(args=[7], expected=7)],
            )

        async def feedback(self, feedback):
            self.feedback_called = True

    class Orchestrator:
        score_called = False

        async def evaluate(self, problem, captured, semaphore):
            return result

        def score_and_export(self, problem, challenge_result, active_uids):
            self.score_called = True

    async def fake_dispatch(public, solvers, concurrency):
        return [SimpleNamespace()], [SimpleNamespace()]

    monkeypatch.setattr(decentralized, "_dispatch_committed", fake_dispatch)
    monkeypatch.setattr(
        decentralized,
        "_sandbox_healthcheck",
        lambda orchestrator, language: False,
    )
    client = Client()
    orchestrator = Orchestrator()
    solver = SimpleNamespace(uid=7, hotkey="hotkey-7")

    evaluated = await _evaluate_one(
        client,
        orchestrator,
        SimpleNamespace(sample=lambda solvers, _k: solvers),
        [solver],
        Settings(
            _env_file=None,
            dispatch_subset_k=0,
            validator_dispatch_concurrency=1,
            validator_verify_concurrency=1,
        ),
    )

    assert evaluated is None
    assert orchestrator.score_called is False
    assert client.feedback_called is False


async def test_dispatch_has_a_total_deadline_and_cancels_stalling_miner(
    monkeypatch,
):
    import rlvr.neurons.decentralized as decentralized

    class StallingSolver:
        uid = 7
        hotkey = "stalling-miner"
        cancelled = False

        async def solve_signed(self, problem, request_id):
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True

    monkeypatch.setattr(decentralized, "_MINER_RESPONSE_GRACE_S", 0.0)
    public = PublicChallenge(
        challenge_id="challenge",
        problem_id="problem",
        language="python",
        statement="Return the input.",
        entrypoint="solve",
        deadline_s=0.02,
    )
    solver = StallingSolver()

    submissions, captured = await asyncio.wait_for(
        _dispatch_committed(public, [solver], concurrency=1),
        timeout=1,
    )

    assert solver.cancelled
    assert len(submissions) == len(captured) == 1
    assert submissions[0].uid == solver.uid
    assert submissions[0].response_body == ""
    assert submissions[0].response_headers == {}
    assert submissions[0].error == "<miner total response deadline exceeded>"
    assert captured[0].solution.code == ""
