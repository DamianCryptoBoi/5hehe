"""Commit receipt correctness without registration-churn retry behavior."""

from types import SimpleNamespace

import rlvr.neurons.decentralized as decentralized
from rlvr.config import Settings
from rlvr.neurons.decentralized import _evaluate_one
from rlvr.problemserver.api import MinerSubmission, PublicChallenge
from rlvr.problemserver.client import LeaseCategory, LeaseOutcome


class Client:
    def __init__(self, receipt):
        self.receipt = receipt
        self.commits = 0
        self.reveals = 0

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
        self.commits += 1
        return self.receipt

    async def reveal(self, challenge_id, commit_token):
        self.reveals += 1
        raise AssertionError("failed commits must not reveal")


async def evaluate(monkeypatch, receipt):
    async def dispatch(public, solvers, concurrency):
        submission = MinerSubmission(
            uid=7,
            hotkey="hotkey-7",
            request_id="request-7",
            response_body="{}",
        )
        captured = SimpleNamespace(uid=7, hotkey="hotkey-7")
        return [submission], [captured]

    monkeypatch.setattr(decentralized, "_dispatch_committed", dispatch)
    client = Client(receipt)
    result = await _evaluate_one(
        client,
        SimpleNamespace(),
        SimpleNamespace(sample=lambda solvers, _k: solvers),
        [SimpleNamespace(uid=7, hotkey="hotkey-7")],
        Settings(_env_file=None, validator_dispatch_concurrency=1),
    )
    return result, client


async def test_rejected_receipt_reports_server_detail_before_count(
    monkeypatch, capsys
):
    result, client = await evaluate(
        monkeypatch,
        SimpleNamespace(
            accepted=False,
            commit_token="",
            num_submissions=99,
            detail="registration changed during dispatch",
        ),
    )

    output = capsys.readouterr().out
    assert result is None
    assert "registration changed during dispatch" in output
    assert "protocol error" not in output
    assert client.commits == 1
    assert client.reveals == 0


async def test_rejected_commit_is_never_retried(monkeypatch):
    _, client = await evaluate(
        monkeypatch,
        SimpleNamespace(
            accepted=False,
            commit_token="",
            num_submissions=1,
            detail="commit refused",
        ),
    )

    assert client.commits == 1


async def test_accepted_count_mismatch_is_a_protocol_error(monkeypatch, capsys):
    result, client = await evaluate(
        monkeypatch,
        SimpleNamespace(
            accepted=True,
            commit_token="token",
            num_submissions=2,
            detail="",
        ),
    )

    output = capsys.readouterr().out.lower()
    assert result is None
    assert "protocol error" in output
    assert "rejected" not in output
    assert client.commits == 1
    assert client.reveals == 0


async def test_missing_receipt_reports_server_unavailable(monkeypatch, capsys):
    result, client = await evaluate(monkeypatch, None)

    assert result is None
    assert "server unavailable" in capsys.readouterr().out
    assert client.commits == 1
    assert client.reveals == 0
