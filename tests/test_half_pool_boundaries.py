"""Boundary tests for the half-pool dispatch default.

Each problem now goes to HALF of the currently serving miners instead of all of
them. These tests exercise the observable boundary rather than any one
helper's shape:

  * sizing at the pool edges (1/2/3/255/256) and the empty pool;
  * the challenge commit quorum raising the half, capped by the serving pool,
    while the pre-lease viability gate keeps reading the SERVING count;
  * release-controlled half-pool sizing that legacy environment values cannot
    override;
  * multi-round fairness: full coverage every two rounds, +/-1 load per epoch;
  * the no-decay boundary. Half the pool is silent every round BY DESIGN, so
    an unsampled SERVING miner must be frozen, while a dispatched-but-silent
    miner and a non-serving observed miner must still decay.

Sizing assertions are made twice on purpose: once through ``_evaluate_one``,
which is implementation-agnostic and valid against any implementation, and once
against the sizing helper directly.

"""

from __future__ import annotations

import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

import rlvr.neurons.decentralized as decentralized
from rlvr.config import Settings
from rlvr.neurons.decentralized import (
    _dispatch_subset_size,
    _evaluate_one,
    _weight_observation_count,
)
from rlvr.orchestrator import Orchestrator, RotationSampler
from rlvr.policy import LEGACY_SCORE_WINDOW_SECONDS, RELEASE_POLICY
from rlvr.problemserver.api import MinerSubmission, PublicChallenge
from rlvr.problemserver.client import LeaseCategory, LeaseOutcome
from rlvr.scoring.eval_engine import EvalEngine
from rlvr.types import (
    ChallengeResult,
    DifficultyBand,
    MinerOutcome,
    Problem,
    SolutionResponse,
    TestCase,
    Verification,
)

DEFAULTS = Settings(_env_file=None)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _size(
    pool: int,
    required: int = 1,
    fraction: float = 0.5,
) -> int:
    return _dispatch_subset_size(pool, required, fraction)


# --------------------------------------------------------------------------- #
# 1. Dynamic half: pool-size boundaries
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("serving", "expected"),
    [(1, 1), (2, 1), (3, 2), (255, 128), (256, 128)],
)
def test_default_sample_is_ceil_half_the_serving_pool(serving, expected):
    """1/2/3/255/256 are the pools operators actually hit: a solo testnet
    miner, an even pair, the first odd pool, and the two sides of a full 256.
    Rounding is UP: floor sends a one-miner pool nothing, and 0 already means
    "everyone" to RotationSampler, so floor is ambiguous at the small end."""
    assert _size(serving) == expected
    assert expected == math.ceil(serving / 2)


def test_a_single_serving_miner_still_receives_the_problem():
    assert _size(1) == 1
    assert _size(1, required=1) == 1


def test_an_empty_serving_pool_targets_nobody():
    """No serving miners is not a fallback to full pool."""
    assert _size(0) == 0
    assert _size(0, required=4) == 0


def test_shipped_default_configuration_selects_the_dynamic_half():
    """An unset fixed-count override selects the fractional default."""
    assert RELEASE_POLICY.dispatch_fraction == 0.5
    assert _size(256) == 128
    assert _size(256) != 256


# --------------------------------------------------------------------------- #
# 2. Quorum override
# --------------------------------------------------------------------------- #
def test_challenge_quorum_raises_the_half_sample():
    """A challenge needing more committed responses than half the pool must
    widen the dispatch; otherwise every such round commits under quorum after
    the problem has already been burned on lease."""
    assert _size(10, required=3) == 5  # below half: no effect
    assert _size(10, required=5) == 5  # exactly half
    assert _size(10, required=6) == 6  # one above half
    assert _size(10, required=10) == 10  # whole pool required


def test_quorum_never_widens_the_dispatch_beyond_the_serving_pool():
    """Refusing an under-quorum round is the caller's pre-lease job; the size
    itself must stay a valid subset size."""
    assert _size(4, required=9) == 4
    assert _size(255, required=1024) == 255


# --------------------------------------------------------------------------- #
# 3. Wiring: the round actually dispatches to the half
# --------------------------------------------------------------------------- #
class _RecordingRotation:
    """Records the size _evaluate_one asks for and deals the first k solvers."""

    def __init__(self):
        self.requested: list[int] = []

    def sample(self, solvers, k):
        self.requested.append(k)
        return list(solvers)[:k] if k > 0 else list(solvers)


class _Client:
    def __init__(self, *, min_responses: int = 1, min_signed: int = 1):
        self._min_responses = min_responses
        self._min_signed = min_signed
        self.leases = 0

    async def lease(self):
        self.leases += 1
        return LeaseOutcome(
            category=LeaseCategory.LEASED,
            status=200,
            challenge=PublicChallenge(
                challenge_id="challenge-1",
                problem_id="problem-1",
                language="python",
                statement="Return a + b.",
                entrypoint="add",
                commit_min_responses=self._min_responses,
                commit_min_signed_responses=self._min_signed,
            ),
        )

    async def commit(self, challenge_id, submissions):
        return SimpleNamespace(
            accepted=True,
            commit_token="commit-token",
            num_submissions=len(submissions),
            detail="",
        )

    async def reveal(self, challenge_id, commit_token):
        return SimpleNamespace(
            challenge_id=challenge_id,
            language="python",
            tests=[TestCase(args=[1, 2], expected=3)],
        )

    async def feedback(self, feedback):
        return True


class _Orchestrator:
    def __init__(self):
        self.active_uids: list[set[int]] = []
        self.evaluated: list[int] = []

    async def evaluate(self, problem, captured, semaphore):
        self.evaluated.append(len(captured))
        return ChallengeResult(
            problem_id=problem.problem_id,
            outcomes=[],
            pass_rate=0.0,
            band=DifficultyBand.UNKNOWN,
            num_responses=len(captured),
        )

    def score_and_export(self, problem, challenge_result, active_uids=None):
        self.active_uids.append(set(active_uids or ()))


def _install_dispatch(monkeypatch):
    """Echo whatever subset it is handed, so contacted == the sampled subset."""

    async def fake_dispatch(public, subset, concurrency):
        submissions = [
            MinerSubmission(
                uid=solver.uid,
                hotkey=solver.hotkey,
                request_id=f"request-{solver.uid}",
                response_body="{}",
                response_headers={"Epistula-Request-Signature": "signature"},
            )
            for solver in subset
        ]
        captured = [
            SimpleNamespace(uid=solver.uid, hotkey=solver.hotkey) for solver in subset
        ]
        return submissions, captured

    monkeypatch.setattr(decentralized, "_dispatch_committed", fake_dispatch)


def _pool(count: int):
    return [SimpleNamespace(uid=uid, hotkey=f"hotkey-{uid}") for uid in range(count)]


def _settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        validator_dispatch_concurrency=4,
        validator_verify_concurrency=1,
        **overrides,
    )


async def test_round_dispatches_to_half_an_even_serving_pool(monkeypatch, capsys):
    _install_dispatch(monkeypatch)
    rotation = _RecordingRotation()

    await _evaluate_one(_Client(), _Orchestrator(), rotation, _pool(8), _settings())

    assert rotation.requested == [4]
    assert "contacted=4" in capsys.readouterr().out


async def test_round_halves_an_odd_serving_pool_upward(monkeypatch, capsys):
    _install_dispatch(monkeypatch)
    rotation = _RecordingRotation()

    await _evaluate_one(_Client(), _Orchestrator(), rotation, _pool(9), _settings())

    assert rotation.requested == [5]
    assert "contacted=5" in capsys.readouterr().out


async def test_round_raises_the_dispatch_to_the_challenge_quorum(monkeypatch, capsys):
    """Half of 8 is 4, but the challenge needs 6 committed signed responses.
    Eight are serving, so the round is viable and must widen, not skip."""
    _install_dispatch(monkeypatch)
    rotation = _RecordingRotation()

    result = await _evaluate_one(
        _Client(min_responses=2, min_signed=6),
        _Orchestrator(),
        rotation,
        _pool(8),
        _settings(),
    )

    assert rotation.requested == [6]
    assert "contacted=6" in capsys.readouterr().out
    assert result is not None


async def test_round_ignores_a_retired_fixed_count_setting(monkeypatch, capsys):
    _install_dispatch(monkeypatch)
    rotation = _RecordingRotation()

    await _evaluate_one(
        _Client(),
        _Orchestrator(),
        rotation,
        _pool(8),
        _settings(dispatch_subset_k=3),
    )

    assert rotation.requested == [4]
    assert "contacted=4" in capsys.readouterr().out


async def test_prelease_viability_still_reads_the_full_serving_count(monkeypatch):
    """Halving what we contact must not halve what we consider viable: 8
    serving against a quorum of 6 is a valid round, and only a pool genuinely
    below quorum may refuse the lease."""
    _install_dispatch(monkeypatch)
    client = _Client(min_responses=6, min_signed=6)

    viable = await _evaluate_one(
        client, _Orchestrator(), _RecordingRotation(), _pool(8), _settings()
    )
    under = await _evaluate_one(
        client, _Orchestrator(), _RecordingRotation(), _pool(4), _settings()
    )

    assert viable is not None
    assert under is None


async def test_scoring_still_receives_every_serving_uid_as_active(monkeypatch):
    """The dispatched half shrinks; the active set must stay the whole serving
    pool or the unsampled half is misread as non-serving and decays."""
    _install_dispatch(monkeypatch)
    orchestrator = _Orchestrator()

    await _evaluate_one(
        _Client(), orchestrator, _RecordingRotation(), _pool(8), _settings()
    )

    assert orchestrator.active_uids == [set(range(8))]
    assert orchestrator.evaluated == [4]


# --------------------------------------------------------------------------- #
# 5. Multi-round fairness under the dynamic half
# --------------------------------------------------------------------------- #
def _deal(pool_size: int, rounds: int, seed: int = 1234) -> list[list[int]]:
    """Deal `rounds` half-pool samples off ONE sampler, as a validator does."""
    sampler = RotationSampler(random.Random(seed))
    pool = _pool(pool_size)
    k = _size(pool_size)
    return [[s.uid for s in sampler.sample(pool, k)] for _ in range(rounds)]


@pytest.mark.parametrize("pool_size", [2, 3, 9, 10, 255, 256])
def test_every_serving_miner_is_contacted_within_the_first_two_rounds(pool_size):
    """Halving per-problem cost must not halve coverage. Because the sample is
    ceil(n/2), the leftovers of round one exactly fill round two."""
    first, second = _deal(pool_size, 2)

    assert set(first) | set(second) == set(range(pool_size))


@pytest.mark.parametrize("pool_size", [2, 3, 9, 10, 255, 256])
def test_half_pool_load_stays_within_one_problem_per_miner(pool_size):
    """Payment flows only through problems a miner was sampled into, so lumpy
    dealing is earnings variance a miner cannot control."""
    counts = {uid: 0 for uid in range(pool_size)}
    for dealt in _deal(pool_size, 8):
        assert len(dealt) == len(set(dealt)), "a miner was dealt twice in one problem"
        assert len(dealt) == math.ceil(pool_size / 2)
        for uid in dealt:
            counts[uid] += 1

    assert max(counts.values()) - min(counts.values()) <= 1


def test_half_pool_dealing_survives_deregistration_between_problems():
    """A pool that shrinks mid-epoch must not strand the remaining miners or
    deal a slot to a departed uid."""
    sampler = RotationSampler(random.Random(7))
    pool = _pool(10)
    sampler.sample(pool, _size(10))

    shrunk = pool[:5]
    dealt = sampler.sample(shrunk, _size(len(shrunk)))

    assert len(dealt) == 3
    assert {s.uid for s in dealt} <= {s.uid for s in shrunk}


def test_half_pool_dealing_admits_a_new_registration_without_a_reshuffle():
    sampler = RotationSampler(random.Random(11))
    pool = _pool(8)
    sampler.sample(pool, _size(8))

    grown = _pool(9)
    dealt = sampler.sample(grown, _size(9))

    assert len(dealt) == 5
    assert len({solver.uid for solver in dealt}) == 5


# --------------------------------------------------------------------------- #
# 6. No-decay boundary for the unsampled half
# --------------------------------------------------------------------------- #
class _FakeWriter:
    def __init__(self):
        self.written: list[list] = []
        self.problems: list[Problem] = []

    def build_prompt(self, problem: Problem) -> str:
        return problem.statement

    def to_rollouts(self, problem, result):
        return [(problem.problem_id, o.uid, o.reward) for o in result.outcomes]

    def write(self, rollouts) -> None:
        self.written.append(list(rollouts))

    def write_problem(self, problem: Problem) -> None:
        self.problems.append(problem)


def _problem() -> Problem:
    return Problem(
        problem_id="p1",
        language="python",
        statement="implement f",
        entrypoint="f",
        tests=[TestCase(args=[1], expected=1)],
    )


def _outcome(uid: int, reward: float) -> MinerOutcome:
    return MinerOutcome(
        uid=uid,
        hotkey=f"hotkey-{uid}",
        solution=SolutionResponse(problem_id="p1", code="def f(x):\n    return x\n"),
        verification=Verification(
            problem_id="p1",
            num_tests=1,
            num_passed=1 if reward > 0 else 0,
            all_passed=reward > 0,
            reward=reward,
        ),
        reward=reward,
    )


def _engine(num_uids: int, decay: bool = True) -> EvalEngine:
    return EvalEngine(
        num_uids,
        LEGACY_SCORE_WINDOW_SECONDS,
        RELEASE_POLICY.score_window_max_samples,
        RELEASE_POLICY.score_window_min_samples,
        decay=decay,
        clock=lambda: 1000.0,
    )


def _half_round(engine: EvalEngine, serving, sampled, rewards=None) -> None:
    """Score one completed challenge dealt to `sampled` out of `serving`."""
    assert len(sampled) == _size(len(serving)), "test must dispatch exactly the half"
    orchestrator = Orchestrator(
        verifier=None, eval_engine=engine, writer=_FakeWriter(), settings=DEFAULTS
    )
    outcomes = [
        _outcome(uid, 1.0 if rewards is None else rewards[uid]) for uid in sampled
    ]
    result = ChallengeResult(
        problem_id="p1",
        outcomes=outcomes,
        pass_rate=sum(o.reward for o in outcomes) / len(outcomes),
        band=DifficultyBand.UNKNOWN,
        num_responses=len(outcomes),
    )
    orchestrator.score_and_export(_problem(), result, active_uids=set(serving))


def test_an_unsampled_serving_miner_is_frozen_not_zeroed():
    """The central risk of halving dispatch: half the pool is silent every
    round BY DESIGN. Recording a zero for them halves an honest miner's score
    for being off-rotation."""
    engine = _engine(5)
    engine.update({uid: 1.0 for uid in range(5)}, dispatched=set(range(5)))
    before = {uid: float(engine.scores[uid]) for uid in range(5)}

    _half_round(engine, serving=[0, 1, 2, 3], sampled=[0, 1])

    for uid in (2, 3):
        assert len(engine.histories[uid]) == 1, "unsampled miner gained an entry"
        assert engine.scores[uid] == pytest.approx(before[uid])


def test_a_dispatched_but_silent_miner_still_decays():
    """Freezing the unsampled must not also freeze a miner that WAS asked and
    did not answer."""
    engine = _engine(5)
    engine.update({uid: 1.0 for uid in range(5)}, dispatched=set(range(5)))

    _half_round(engine, serving=[0, 1, 2, 3], sampled=[0, 1], rewards={0: 1.0, 1: 0.0})

    assert [reward for _, reward in engine.histories[1]] == [1.0, 0.0]
    assert [reward for _, reward in engine.histories[2]] == [1.0]


def test_a_non_serving_observed_miner_still_decays():
    """Unchanged semantics for a miner with no serving axon: it is absent, not
    off-rotation."""
    engine = _engine(5)
    engine.update({uid: 1.0 for uid in range(5)}, dispatched=set(range(5)))

    _half_round(engine, serving=[0, 1, 2, 3], sampled=[0, 1])

    assert [reward for _, reward in engine.histories[4]] == [1.0, 0.0]


def test_an_unsampled_miner_stays_frozen_across_consecutive_rounds():
    """A miner that sits out several problems in a row must be exactly as
    scored as before, not merely decayed slowly."""
    engine = _engine(4)
    engine.update({uid: 1.0 for uid in range(4)}, dispatched=set(range(4)))
    before = float(engine.scores[3])

    for _ in range(3):
        _half_round(engine, serving=[0, 1, 2, 3], sampled=[0, 1])

    assert len(engine.histories[3]) == 1
    assert engine.scores[3] == pytest.approx(before)


def test_a_never_observed_unsampled_miner_gains_no_history():
    """A newly registered miner awaiting its rotation turn must not accumulate
    pre-participation failures."""
    engine = _engine(4)
    engine.update({0: 1.0, 1: 1.0}, dispatched={0, 1})

    _half_round(engine, serving=[0, 1, 2, 3], sampled=[0, 1])

    assert 2 not in engine.histories
    assert 3 not in engine.histories


def test_freezing_is_unconditional_when_decay_is_disabled():
    engine = _engine(5, decay=False)
    engine.update({uid: 1.0 for uid in range(5)}, dispatched=set(range(5)))

    _half_round(engine, serving=[0, 1, 2, 3], sampled=[0, 1], rewards={0: 1.0, 1: 0.0})

    assert len(engine.histories[4]) == 1
    assert len(engine.histories[2]) == 1


def test_one_new_registration_does_not_close_an_established_weight_gate():
    engine = _engine(8)
    for _ in range(4):
        engine.update({uid: 1.0 for uid in range(4)}, dispatched=set(range(4)))

    engine.update({7: 1.0}, dispatched={7})

    assert _weight_observation_count(engine) >= 4


# --------------------------------------------------------------------------- #
# 7. Operator-facing defaults
# --------------------------------------------------------------------------- #
def test_env_example_does_not_pin_release_policy():
    text = (REPO_ROOT / ".env.example").read_text()

    assert "DISPATCH_SUBSET_FRACTION=" not in text
    assert "DISPATCH_SUBSET_K=" not in text
    assert "FULL pool" not in text
    assert "All serving miners receive every problem" not in text
