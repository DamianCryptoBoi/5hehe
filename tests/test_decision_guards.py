"""Guards for commit retry and score-state compatibility decisions."""

from __future__ import annotations

import json
from types import SimpleNamespace

import rlvr.neurons.decentralized as decentralized
from rlvr.config import Settings
from rlvr.neurons.decentralized import _evaluate_one, _save_scores
from rlvr.problemserver.api import MinerSubmission, PublicChallenge
from rlvr.problemserver.client import LeaseCategory, LeaseOutcome
from rlvr.scoring.eval_engine import EvalEngine
from rlvr.types import TestCase


class _Client:
    def __init__(self):
        self.commits = 0

    async def lease(self):
        return LeaseOutcome(
            category=LeaseCategory.LEASED,
            status=200,
            challenge=PublicChallenge(
                challenge_id="c1",
                problem_id="p1",
                language="python",
                statement="Return a + b.",
                entrypoint="add",
                commit_min_responses=2,
                commit_min_signed_responses=2,
            ),
        )

    async def commit(self, challenge_id, submissions):
        self.commits += 1
        return SimpleNamespace(
            accepted=False,
            commit_token="",
            num_submissions=len(submissions),
            detail="uid 2 deregistered mid-round",
        )

    async def reveal(self, challenge_id, commit_token):
        return SimpleNamespace(
            challenge_id=challenge_id,
            language="python",
            tests=[TestCase(args=[1, 2], expected=3)],
        )

    async def feedback(self, feedback):
        return True


async def test_a_rejected_commit_is_sent_exactly_once(monkeypatch):
    """A rejected commit is reported and never retried from free-text detail."""

    async def fake_dispatch(public, subset, concurrency):
        submissions = [
            MinerSubmission(
                uid=u, hotkey=f"hotkey-{u}", request_id=f"req-{u}", response_body="{}"
            )
            for u in (0, 1, 2)
        ]
        captured = [SimpleNamespace(uid=u, hotkey=f"hotkey-{u}") for u in (0, 1, 2)]
        return submissions, captured

    monkeypatch.setattr(decentralized, "_dispatch_committed", fake_dispatch)
    client = _Client()

    result = await _evaluate_one(
        client,
        None,
        SimpleNamespace(sample=lambda pool, _k: pool),
        [SimpleNamespace(uid=u, hotkey=f"hotkey-{u}") for u in (0, 1, 2)],
        Settings(_env_file=None, validator_dispatch_concurrency=4),
    )

    assert result is None
    assert client.commits == 1, (
        f"commit was sent {client.commits} times; churn retry is ablated and a "
        "retry must not be reintroduced on the server's free-text detail"
    )


def test_saved_state_still_carries_window_seconds_for_rollback(tmp_path):
    """`window_seconds` is inert for scoring but must stay in persisted state.

    The scoring window is count-only now, so this value influences nothing. It
    is still written because `_load_scores` requires the key to be present,
    finite and positive: a build that stops emitting it produces state files no
    earlier build can read. A rollback would then discard every miner's history
    and submit no weights until four fresh observations accumulate.

    Keeping it costs one dictionary key. Removing it is a one-way upgrade.
    """
    engine = EvalEngine(
        num_uids=4, window_seconds=57_600.0, max_samples=200, min_samples=4
    )
    for i in range(3):
        engine.update({0: 1.0}, observed_at=1_000.0 + i)
    engine.set_hotkeys({0: "miner-a"})

    path = tmp_path / "scores.json"
    _save_scores(engine, str(path))
    saved = json.loads(path.read_text())

    assert "window_seconds" in saved, (
        "persisted state dropped window_seconds; earlier builds can no longer "
        "read this file and a rollback silently wipes all miner history"
    )
    assert saved["window_seconds"] > 0
