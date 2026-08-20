"""Validator-side dispatch, verification, scoring, and reward logic.

Problems arrive exclusively through the private problem-source protocol. This
module has no problem-construction, model-inference, or curriculum-authoring
dependency.
"""

from __future__ import annotations

import asyncio
import random
from typing import Iterable, Optional, Protocol, Sequence, runtime_checkable

from .config import Settings
from .dataset.writer import RolloutWriter
from .policy import RELEASE_POLICY, ValidatorPolicy
from .scoring.difficulty import classify_band, pass_rate
from .scoring.eval_engine import EvalEngine
from .scoring.payment import compute_payments
from .scoring.verifier import Verifier
from .types import (
    ChallengeResult,
    DifficultyBand,
    MinerOutcome,
    Problem,
    SolutionResponse,
    Verification,
)


@runtime_checkable
class SolverClient(Protocol):
    """Anything that can answer a task dispatched by a validator."""

    uid: int
    hotkey: str

    async def solve(self, problem: Problem, prompt: str) -> SolutionResponse: ...


class RotationSampler:
    """Deal solver subsets off a reshuffled queue so per-epoch load is even.

    Uniform per-problem sampling is fine in expectation but lumpy in practice:
    payment flows only through problems a miner was sampled into, so lumpy
    sampling injects earnings variance miners can't control. Dealing (like
    cards) bounds every miner's load to ±1 problem per epoch. Handles pool
    membership changes between draws (dereg / new axons).
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self._queue: list[int] = []
        self._rng = rng or random.Random()

    def sample(
        self, solvers: Sequence[SolverClient], k: int
    ) -> list[SolverClient]:
        by_uid = {s.uid: s for s in solvers}
        # UIDs are expected to be unique in a metagraph, but fail safely if a
        # malformed caller supplies duplicates. Otherwise k > unique UIDs can
        # make the refill loop below spin forever.
        unique = list(by_uid.values())
        if k <= 0 or k >= len(unique):
            return unique
        self._queue = [u for u in self._queue if u in by_uid]
        picked: list[int] = []
        while len(picked) < k:
            if not self._queue:
                fresh = list(by_uid)
                self._rng.shuffle(fresh)
                self._queue = fresh
            uid = self._queue.pop(0)
            if uid in picked:
                # Preserve this epoch slot for the next problem rather than
                # deleting it at a boundary crossed mid-sample.
                self._queue.append(uid)
                continue
            picked.append(uid)
        return [by_uid[u] for u in picked]


class Orchestrator:
    def __init__(
        self,
        verifier: Verifier,
        eval_engine: EvalEngine,
        writer: RolloutWriter,
        settings: Settings,
        policy: ValidatorPolicy = RELEASE_POLICY,
    ):
        self.verifier = verifier
        self.eval_engine = eval_engine
        self.writer = writer
        self.settings = settings
        self.policy = policy

    async def _attempt_one(
        self, solver: SolverClient, problem: Problem, prompt: str, sem: asyncio.Semaphore
    ) -> MinerOutcome:
        async with sem:
            try:
                solution = await solver.solve(problem, prompt)
            except Exception as e:  # noqa: BLE001 - a dead miner scores 0, never crashes the round
                solution = SolutionResponse(
                    problem_id=problem.problem_id, code="", raw_response=f"SOLVER_ERROR: {e}"
                )
            # Miners are adversarial: a verifier/executor blowup on ONE solution
            # must not crash the whole round. Verification is subprocess-bound,
            # so run it in a worker thread while RETAINING the concurrency permit;
            # otherwise a setting of 4 could still launch dozens of Docker
            # containers simultaneously after the HTTP calls finish.
            try:
                verification = await asyncio.to_thread(
                    self.verifier.verify, problem, solution
                )
            except Exception as e:  # noqa: BLE001
                verification = Verification(
                    problem_id=problem.problem_id,
                    num_tests=len(problem.tests),
                    num_passed=0,
                    all_passed=False,
                    reward=0.0,
                    compile_error=f"VERIFY_ERROR: {e}",
                )
        return MinerOutcome(
            uid=solver.uid,
            hotkey=solver.hotkey,
            solution=solution,
            verification=verification,
            reward=verification.reward,
        )

    async def evaluate(
        self, problem: Problem, solvers: Sequence[SolverClient], sem: asyncio.Semaphore
    ) -> ChallengeResult:
        """Dispatch one problem to the pool, verify, aggregate into p + band."""
        prompt = self.writer.build_prompt(problem)
        # return_exceptions=True so a single straggler escaping _attempt_one cannot
        # abort the whole gather (and with it the round). _attempt_one already maps
        # solver AND verifier failures to reward-0 outcomes; this is the last belt.
        raw = await asyncio.gather(
            *[self._attempt_one(s, problem, prompt, sem) for s in solvers],
            return_exceptions=True,
        )
        outcomes = [
            o for s, o in zip(solvers, raw) if isinstance(o, MinerOutcome)
        ] + [
            self._dead_outcome(problem, s, o)
            for s, o in zip(solvers, raw)
            if isinstance(o, BaseException)
        ]
        p = pass_rate(outcomes)

        # With too few responders, the pass-rate is statistical noise.
        if len(outcomes) < self.settings.difficulty_samples_m:
            band = DifficultyBand.UNKNOWN
        else:
            band = classify_band(p, self.settings.band_low, self.settings.band_high)

        problem.measured_pass_rate = p
        problem.band = band
        problem.num_attempts = len(outcomes)
        return ChallengeResult(
            problem_id=problem.problem_id,
            outcomes=outcomes,
            pass_rate=p,
            band=band,
            num_responses=len(outcomes),
            # Reserved for report compatibility. V1 does not infer a duplicate
            # ratio from submitted source code.
            dup_ratio=0.0,
        )

    @staticmethod
    def _dead_outcome(
        problem: Problem, solver: SolverClient, exc: BaseException
    ) -> MinerOutcome:
        """Reward-0 outcome for an attempt whose task raised before returning one."""
        return MinerOutcome(
            uid=solver.uid,
            hotkey=solver.hotkey,
            solution=SolutionResponse(
                problem_id=problem.problem_id, code="", raw_response=f"ATTEMPT_ERROR: {exc}"
            ),
            verification=Verification(
                problem_id=problem.problem_id,
                num_tests=len(problem.tests),
                num_passed=0,
                all_passed=False,
                reward=0.0,
                compile_error=f"ATTEMPT_ERROR: {exc}",
            ),
            reward=0.0,
        )

    def score_and_export(
        self,
        problem: Problem,
        cr: ChallengeResult,
        active_uids: Optional[Iterable[int]] = None,
    ) -> None:
        """Apply verified outcomes to local scores and persist the audit record."""
        payments = compute_payments(
            cr.outcomes,
            speed_half_life_ms=self.policy.payment_speed_half_life_ms,
            speed_floor=self.policy.payment_speed_floor,
        )
        dispatched = {o.uid for o in cr.outcomes}
        active = None if active_uids is None else {int(uid) for uid in active_uids}
        if active is not None and not dispatched.issubset(active):
            raise ValueError("every dispatched uid must be in the active uid set")
        observed_at = self.eval_engine.update(
            payments,
            hotkeys={o.uid: o.hotkey for o in cr.outcomes if o.hotkey},
            # Subset dispatch: only the miners actually sampled for this problem
            # may decay for silence; unsampled miners are frozen until their
            # rotation turn. record_nonserving handles uids with no serving axon.
            dispatched=dispatched,
        )
        if active is not None:
            # This is the other half of the completed-challenge partition:
            # update() handled dispatched uids; only non-serving histories get
            # a zero here. Paced/empty rounds never reach this method.
            self.eval_engine.record_nonserving(active, observed_at=observed_at)
        rollouts = self.writer.to_rollouts(problem, cr)
        self.writer.write(rollouts)
        # Persist the locally evaluated challenge for validator auditing. This
        # record contains revealed cases and must never ship to miners.
        self.writer.write_problem(problem)
