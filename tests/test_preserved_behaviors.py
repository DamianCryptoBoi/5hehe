"""Coverage for response-count logging and current-block recovery.

Miner response-count logging: contacted vs responded vs signed are three
    different numbers and an operator diagnoses a different fault from each.
Current-block recovery: one transient RPC blip must not kill the validator.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import rlvr.neurons.decentralized as decentralized
import rlvr.neurons.validator as validator_mod
from rlvr.config import Settings
from rlvr.neurons.decentralized import _evaluate_one
from rlvr.neurons.validator import ValidatorNeuron
from rlvr.problemserver.api import MinerSubmission, PublicChallenge
from rlvr.problemserver.client import LeaseCategory, LeaseOutcome
from rlvr.types import ChallengeResult, DifficultyBand, TestCase


# --------------------------------------------------------------------------- #
# 5a — response-count logging
# --------------------------------------------------------------------------- #
class _Client:
    def __init__(self, *, accepted=True, count=None, detail=""):
        self._accepted = accepted
        self._count = count
        self._detail = detail
        self.commits = 0

    async def lease(self):
        return LeaseOutcome(
            category=LeaseCategory.LEASED,
            status=200,
            challenge=PublicChallenge(
                challenge_id="challenge-1",
                problem_id="problem-1",
                language="python",
                statement="Return a + b.",
                entrypoint="add",
                commit_min_responses=1,
                commit_min_signed_responses=1,
            ),
        )

    async def commit(self, challenge_id, submissions):
        self.commits += 1
        return SimpleNamespace(
            accepted=self._accepted,
            commit_token="commit-token",
            num_submissions=(
                len(submissions) if self._count is None else self._count
            ),
            detail=self._detail,
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
    async def evaluate(self, problem, captured, semaphore):
        return ChallengeResult(
            problem_id=problem.problem_id,
            outcomes=[],
            pass_rate=0.0,
            band=DifficultyBand.UNKNOWN,
            num_responses=len(captured),
        )

    def score_and_export(self, problem, challenge_result, active_uids=None):
        return None


def _install_dispatch(monkeypatch, uids, *, unsigned=(), errored=()):
    async def fake_dispatch(public, subset, concurrency):
        submissions = [
            MinerSubmission(
                uid=uid,
                hotkey=f"hotkey-{uid}",
                request_id=f"request-{uid}",
                response_body="" if uid in errored else "{}",
                response_headers=(
                    {}
                    if uid in unsigned or uid in errored
                    else {"Epistula-Request-Signature": "signature"}
                ),
                error="<miner total response deadline exceeded>"
                if uid in errored
                else "",
            )
            for uid in uids
        ]
        captured = [SimpleNamespace(uid=uid, hotkey=f"hotkey-{uid}") for uid in uids]
        return submissions, captured

    monkeypatch.setattr(decentralized, "_dispatch_committed", fake_dispatch)


async def _run(client, *, dispatched, serving=None):
    pool = [
        SimpleNamespace(uid=uid, hotkey=f"hotkey-{uid}")
        for uid in (serving if serving is not None else dispatched)
    ]
    chosen = set(dispatched)
    return await _evaluate_one(
        client,
        _Orchestrator(),
        SimpleNamespace(sample=lambda pool_, _k: [s for s in pool_ if s.uid in chosen]),
        pool,
        Settings(
            _env_file=None,
            dispatch_subset_k=0,
            validator_dispatch_concurrency=4,
            validator_verify_concurrency=1,
        ),
    )


async def test_response_counts_distinguish_contacted_responded_and_signed(
    monkeypatch, capsys
):
    """Three separate numbers. Collapsing any two hides a distinct fault:
    contacted<pool is a rotation question, responded<contacted is a miner
    liveness question, signed<responded is an authentication question."""
    _install_dispatch(monkeypatch, [0, 1, 2, 3], unsigned=[2], errored=[3])

    await _run(_Client(), dispatched=[0, 1, 2, 3])
    out = capsys.readouterr().out

    assert "contacted=4" in out, out
    assert "responses=3" in out, out
    assert "signed=2" in out, out


async def test_a_miner_that_failed_is_contacted_but_not_counted_as_responding(
    monkeypatch, capsys
):
    _install_dispatch(monkeypatch, [0, 1], errored=[1])

    await _run(_Client(), dispatched=[0, 1])
    out = capsys.readouterr().out

    assert "contacted=2" in out, out
    assert "responses=1" in out, out


async def test_an_unsigned_response_counts_as_a_response_but_not_as_signed(
    monkeypatch, capsys
):
    """An unsigned answer is a live miner with a broken Epistula path — a
    different diagnosis from a miner that never answered."""
    _install_dispatch(monkeypatch, [0, 1], unsigned=[1])

    await _run(_Client(), dispatched=[0, 1])
    out = capsys.readouterr().out

    assert "responses=2" in out, out
    assert "signed=1" in out, out


async def test_contacted_is_the_dispatched_subset_not_the_whole_serving_pool(
    monkeypatch, capsys
):
    """The pool is what is registered and serving; contacted is what rotation
    actually sampled. Reporting the pool would hide an undersized dispatch."""
    _install_dispatch(monkeypatch, [0, 1])

    await _run(_Client(), dispatched=[0, 1], serving=[0, 1, 2, 3, 4])
    out = capsys.readouterr().out

    assert "contacted=2" in out, out


async def test_response_counts_are_logged_even_when_the_commit_is_rejected(
    monkeypatch, capsys
):
    """The counts are the primary evidence for WHY a commit was rejected, so
    they cannot be reported only on the success path."""
    _install_dispatch(monkeypatch, [0, 1], errored=[1])
    client = _Client(accepted=False, detail="quorum not met")

    result = await _run(client, dispatched=[0, 1])
    out = capsys.readouterr().out

    assert result is None
    assert "contacted=2" in out and "responses=1" in out, out


async def test_response_counts_are_logged_before_the_commit_is_attempted(
    monkeypatch, capsys
):
    """A commit that hangs or crashes must still leave the counts on stdout."""

    class Exploding(_Client):
        async def commit(self, challenge_id, submissions):
            raise RuntimeError("commit blew up")

    _install_dispatch(monkeypatch, [0, 1])

    try:
        await _run(Exploding(), dispatched=[0, 1])
    except RuntimeError:
        pass
    out = capsys.readouterr().out

    assert "contacted=2" in out, out


async def test_a_fully_silent_wave_reports_zero_responses_rather_than_nothing(
    monkeypatch, capsys
):
    _install_dispatch(monkeypatch, [0, 1], errored=[0, 1])
    client = _Client(accepted=False, detail="no responses")

    await _run(client, dispatched=[0, 1])
    out = capsys.readouterr().out

    assert "contacted=2" in out and "responses=0" in out and "signed=0" in out, out


# --------------------------------------------------------------------------- #
# 5b — bounded current-block RPC recovery
# --------------------------------------------------------------------------- #
def _neuron(monkeypatch) -> ValidatorNeuron:
    """A ValidatorNeuron with a live-looking subtensor and no real chain."""
    monkeypatch.setattr(validator_mod, "_CURRENT_BLOCK_RETRY_DELAY_S", 0.0)
    neuron = ValidatorNeuron(Settings(_env_file=None))
    neuron.subtensor = SimpleNamespace()
    neuron.metagraph = SimpleNamespace(hotkeys=[], uids=[])
    return neuron


class _Blocks:
    """Scripted current_block(): entries are either an int or an exception."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        value = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(value, BaseException):
            raise value
        return value


async def test_a_transient_block_read_failure_is_retried_and_recovers(monkeypatch):
    neuron = _neuron(monkeypatch)
    neuron.current_block = _Blocks(ConnectionError("websocket closed"), 4321)

    assert await neuron._current_block_with_recovery() == 4321
    assert neuron.current_block.calls == 2


async def test_block_read_recovery_is_bounded_and_gives_up(monkeypatch):
    """Unbounded retry inside the poll loop is just a hang with extra logging."""
    neuron = _neuron(monkeypatch)
    neuron.current_block = _Blocks(ConnectionError("chain down"))

    assert await neuron._current_block_with_recovery() is None
    assert neuron.current_block.calls == validator_mod._CURRENT_BLOCK_MAX_ATTEMPTS


async def test_every_block_read_failure_is_reported(monkeypatch, capsys):
    """A silent retry loop looks identical to a stalled validator."""
    neuron = _neuron(monkeypatch)
    neuron.current_block = _Blocks(ConnectionError("chain down"))

    await neuron._current_block_with_recovery()
    out = capsys.readouterr().out

    assert out.lower().count("current-block") == validator_mod._CURRENT_BLOCK_MAX_ATTEMPTS
    assert "chain down" in out


async def test_a_block_read_failure_does_not_escape_the_run_loop(monkeypatch):
    """base.py current_block() raises straight through and validator.py called
    it bare inside the loop — one RPC blip used to kill the validator."""
    neuron = _neuron(monkeypatch)
    neuron.current_block = _Blocks(ConnectionError("blip"), 100, 200)
    rounds = []

    async def round_callback(v):
        rounds.append(v.current_block.calls)
        return {}

    neuron.set_round_callback(round_callback)
    monkeypatch.setattr(validator_mod.asyncio, "sleep", _no_sleep())

    await asyncio.wait_for(neuron.run(max_rounds=1), timeout=5)

    assert rounds, "the run loop exited on a transient current-block failure"


async def test_a_totally_failed_block_read_skips_the_round_without_exiting(monkeypatch):
    """Give-up must mean 'poll again', not 'run a round on a block we never
    read' and not 'raise out of run()'."""
    neuron = _neuron(monkeypatch)
    script = [ConnectionError("down")] * validator_mod._CURRENT_BLOCK_MAX_ATTEMPTS
    neuron.current_block = _Blocks(*script, 900)
    rounds = []

    async def round_callback(v):
        rounds.append(True)
        return {}

    neuron.set_round_callback(round_callback)
    monkeypatch.setattr(validator_mod.asyncio, "sleep", _no_sleep())

    await asyncio.wait_for(neuron.run(max_rounds=1), timeout=5)

    assert len(rounds) == 1
    assert neuron.current_block.calls == validator_mod._CURRENT_BLOCK_MAX_ATTEMPTS + 1


async def test_a_failed_block_read_does_not_advance_the_round_gate(monkeypatch):
    """should_run() is STATEFUL — it assigns _last_run_block as a side effect of
    returning True. Anchoring the gate on a block we failed to read would skip
    a real round later."""
    neuron = _neuron(monkeypatch)
    neuron.current_block = _Blocks(ConnectionError("down"))

    await neuron._current_block_with_recovery()

    assert neuron._last_run_block is None


async def test_the_weight_gate_is_not_evaluated_on_a_block_we_could_not_read(
    monkeypatch,
):
    """Submitting weights against a stale or invented block number is worse than
    submitting late."""
    neuron = _neuron(monkeypatch)
    script = [ConnectionError("down")] * validator_mod._CURRENT_BLOCK_MAX_ATTEMPTS
    neuron.current_block = _Blocks(*script, 900)
    submissions = []

    async def round_callback(v):
        return {}

    neuron.set_round_callback(round_callback)
    neuron.set_weight_setter(lambda v: submissions.append(v.current_block.calls))
    monkeypatch.setattr(validator_mod.asyncio, "sleep", _no_sleep())

    await asyncio.wait_for(neuron.run(max_rounds=1), timeout=5)

    assert all(call > validator_mod._CURRENT_BLOCK_MAX_ATTEMPTS for call in submissions)


def _no_sleep():
    async def sleep(_seconds):
        return None

    return sleep
