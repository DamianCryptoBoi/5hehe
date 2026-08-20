"""Tests for validator-side dispatch, verification, and scoring."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from rlvr.config import get_settings
from rlvr.orchestrator import Orchestrator, RotationSampler
from rlvr.types import DifficultyBand, Problem, SolutionResponse, Verification
from rlvr.types import TestCase as Case


def make_problem(pid: str = "p1", n_tests: int = 3) -> Problem:
    return Problem(
        problem_id=pid,
        language="python",
        statement="implement f",
        entrypoint="f",
        tests=[Case(args=[i], expected=i) for i in range(n_tests)],
    )


class FakeSolver:
    def __init__(self, uid: int, code: str | None = None, raise_exc=None):
        self.uid = uid
        self.hotkey = f"hk{uid}"
        self._code = code or f"def f(x):\n    return x + {uid} - {uid}\n"
        self._raise = raise_exc

    async def solve(self, problem: Problem, prompt: str) -> SolutionResponse:
        if self._raise is not None:
            raise self._raise
        return SolutionResponse(
            problem_id=problem.problem_id,
            code=self._code,
            raw_response=self._code,
        )


class FakeVerifier:
    def __init__(self, reward: float = 1.0, throw_first: bool = False):
        self.reward = reward
        self.throw_first = throw_first
        self.calls = 0

    def verify(self, problem: Problem, solution: SolutionResponse) -> Verification:
        self.calls += 1
        if self.throw_first and self.calls == 1:
            raise RuntimeError("executor exploded")
        passed = self.reward > 0
        n = len(problem.tests)
        return Verification(
            problem_id=problem.problem_id,
            num_tests=n,
            num_passed=n if passed else 0,
            all_passed=passed,
            reward=self.reward,
        )


class ThrowEveryTimeVerifier:
    def verify(self, problem: Problem, solution: SolutionResponse) -> Verification:
        raise ValueError("verifier always blows up")


class FakeEvalEngine:
    def __init__(self):
        self.updates: list[dict] = []
        self.dispatched: list[set] = []
        self.nonserving_active: list[set] = []

    def update(self, rewards, hotkeys=None, dispatched=None):
        self.updates.append(dict(rewards))
        self.dispatched.append(set(dispatched or ()))
        return 123.0

    def record_nonserving(self, active_uids, observed_at=None) -> None:
        assert observed_at == 123.0
        self.nonserving_active.append(set(active_uids))


class FakeWriter:
    def __init__(self):
        self.written: list[list] = []
        self.problems: list[Problem] = []

    def build_prompt(self, problem: Problem) -> str:
        return problem.statement

    def to_rollouts(self, problem, result):
        return [(problem.problem_id, item.uid, item.reward) for item in result.outcomes]

    def write(self, rollouts) -> None:
        self.written.append(list(rollouts))

    def write_problem(self, problem: Problem) -> None:
        self.problems.append(problem)


def make_orchestrator(verifier, *, settings=None, engine=None, writer=None):
    settings = settings or get_settings()
    return Orchestrator(
        verifier=verifier,
        eval_engine=engine or FakeEvalEngine(),
        writer=writer or FakeWriter(),
        settings=settings,
    ), settings


def solvers(count: int, **kwargs) -> list[FakeSolver]:
    return [FakeSolver(uid=index, **kwargs) for index in range(count)]


def evaluate(orch: Orchestrator, problem: Problem, pool: list[FakeSolver]):
    return asyncio.run(orch.evaluate(problem, pool, asyncio.Semaphore(4)))


def test_throwing_verifier_isolated_to_one_zero_reward_outcome():
    problem = make_problem()
    orch, settings = make_orchestrator(FakeVerifier(throw_first=True))
    result = evaluate(orch, problem, solvers(settings.difficulty_samples_m))

    assert len(result.outcomes) == settings.difficulty_samples_m
    failed = [outcome for outcome in result.outcomes if outcome.reward == 0.0]
    assert len(failed) == 1
    assert "VERIFY_ERROR" in failed[0].verification.compile_error


def test_verifier_that_always_throws_does_not_crash_dispatch():
    problem = make_problem()
    orch, settings = make_orchestrator(ThrowEveryTimeVerifier())
    result = evaluate(orch, problem, solvers(settings.difficulty_samples_m))

    assert result.pass_rate == 0.0
    assert all(outcome.reward == 0.0 for outcome in result.outcomes)


def test_concurrency_limit_covers_verification_not_only_dispatch():
    class TrackingVerifier(FakeVerifier):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.maximum = 0
            self.lock = threading.Lock()

        def verify(self, problem, solution):
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            try:
                time.sleep(0.03)
                return super().verify(problem, solution)
            finally:
                with self.lock:
                    self.active -= 1

    verifier = TrackingVerifier()
    orch, _ = make_orchestrator(verifier)
    asyncio.run(
        orch.evaluate(make_problem(), solvers(8), asyncio.Semaphore(2))
    )

    assert verifier.maximum == 2


def test_throwing_responder_becomes_a_normal_failed_attempt():
    problem = make_problem()
    orch, settings = make_orchestrator(FakeVerifier(reward=0.0))
    pool = solvers(settings.difficulty_samples_m)
    pool[0] = FakeSolver(0, raise_exc=RuntimeError("offline"))

    result = evaluate(orch, problem, pool)

    assert len(result.outcomes) == len(pool)
    assert "SOLVER_ERROR" in result.outcomes[0].solution.raw_response


def test_small_response_pool_is_not_assigned_a_difficulty_band():
    problem = make_problem()
    orch, settings = make_orchestrator(FakeVerifier())
    result = evaluate(orch, problem, solvers(settings.difficulty_samples_m - 1))

    assert result.pass_rate == 1.0
    assert result.band == DifficultyBand.UNKNOWN


def test_full_independent_response_pool_is_banded_normally():
    problem = make_problem()
    orch, settings = make_orchestrator(FakeVerifier())
    result = evaluate(orch, problem, solvers(settings.difficulty_samples_m))

    assert result.dup_ratio == 0.0
    assert result.band == DifficultyBand.EASY


def test_identical_passing_answers_do_not_override_v1_difficulty_band():
    problem = make_problem()
    orch, settings = make_orchestrator(FakeVerifier())
    shared = "def f(x):\n    return x\n"
    pool = [FakeSolver(index, code=shared) for index in range(settings.difficulty_samples_m)]

    result = evaluate(orch, problem, pool)

    assert result.pass_rate == 1.0
    assert result.dup_ratio == 0.0
    assert result.band == DifficultyBand.EASY


def test_verified_results_update_payments_and_are_persisted():
    settings = get_settings()
    problem = make_problem(n_tests=2)

    class OneWinnerVerifier:
        def verify(self, problem, solution):
            passed = solution.code.startswith("# uid0")
            n = len(problem.tests)
            return Verification(
                problem_id=problem.problem_id,
                num_tests=n,
                num_passed=n if passed else 0,
                all_passed=passed,
                reward=1.0 if passed else 0.0,
            )

    engine, writer = FakeEvalEngine(), FakeWriter()
    orch, _ = make_orchestrator(
        OneWinnerVerifier(), settings=settings, engine=engine, writer=writer
    )
    pool = [
        FakeSolver(index, code=f"# uid{index}\ndef f(x):\n    return x\n")
        for index in range(settings.difficulty_samples_m)
    ]

    result = evaluate(orch, problem, pool)
    active_uids = {solver.uid for solver in pool}
    orch.score_and_export(problem, result, active_uids=active_uids)

    assert engine.updates[0][0] == pytest.approx(1.0)
    assert all(engine.updates[0][uid] == 0.0 for uid in range(1, len(pool)))
    assert engine.dispatched == [active_uids]
    assert engine.nonserving_active == [active_uids]
    assert writer.written and len(writer.problems) == 1
    assert writer.problems[0].measured_pass_rate == pytest.approx(1 / len(pool))


def test_score_rejects_dispatched_uid_missing_from_active_set():
    engine, writer = FakeEvalEngine(), FakeWriter()
    orch, _ = make_orchestrator(FakeVerifier(), engine=engine, writer=writer)
    problem = make_problem()
    result = evaluate(orch, problem, solvers(2))

    with pytest.raises(ValueError, match="active uid set"):
        orch.score_and_export(problem, result, active_uids={0})

    assert engine.updates == []
    assert writer.written == []


def test_rotation_sampler_is_even_and_handles_membership_changes():
    sampler = RotationSampler()
    pool = solvers(16)
    counts: dict[int, int] = {}
    for _ in range(8):
        for responder in sampler.sample(pool, 4):
            counts[responder.uid] = counts.get(responder.uid, 0) + 1
    assert all(count == 2 for count in counts.values())

    remaining = pool[:3]
    assert all(item.uid < 3 for item in sampler.sample(remaining, 2))
    assert sampler.sample(remaining, 99) == remaining


def test_rotation_sampler_deduplicates_malformed_uid_pool_without_spinning():
    sampler = RotationSampler()
    duplicate_uid_pool = [FakeSolver(0), FakeSolver(0), FakeSolver(1)]
    selected = sampler.sample(duplicate_uid_pool, 2)
    assert {solver.uid for solver in selected} == {0, 1}
