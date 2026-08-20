"""Per-miner verdicts on challenge feedback.

The validator grades locally after the test reveal; the problem server never
learns per-miner pass/fail unless feedback carries it. These expectations pin
the wire shape and — most importantly — WHO gets a verdict.

THE PREDICATE: a verdict is emitted exactly when the dispatch produced an
authenticated, parseable response — submission.error == "", equivalently
response_headers retained, which in live.py happens only on the full success
path. This is deliberately NOT "solution code is non-empty":

  * a miner may return a VALID signed payload whose code is empty — that is a
    deliberate answer, it was verified (and failed), and it earns
    passed=false. Empty code is a wrong answer, not silence.
  * a signed-but-unparseable payload retains no headers on the submission, so
    the problem server's own signature re-verification cannot count it as an
    authenticated response. Excluding it from verdicts keeps verdict coverage
    consistent with the server-derived authenticated response rate — the two
    denominators must agree about which responses "count".
  * transport failures, timeouts and HTTP errors carry an error and no
    headers: committed (their silence belongs in the response rate) but never
    verdicted (nothing of theirs was graded).

Privacy boundary: a verdict is {uid, hotkey, passed} and nothing else.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import rlvr.neurons.decentralized as decentralized
from rlvr.config import Settings
from rlvr.neurons.decentralized import _evaluate_one
from rlvr.problemserver.api import (
    ChallengeFeedback,
    MinerSubmission,
    PublicChallenge,
)
from rlvr.problemserver.client import LeaseCategory, LeaseOutcome
from rlvr.types import (
    ChallengeResult,
    DifficultyBand,
    MinerOutcome,
    SolutionResponse,
    TestCase,
    Verification,
)


def challenge() -> PublicChallenge:
    return PublicChallenge(
        challenge_id="challenge-1",
        problem_id="problem-1",
        language="python",
        statement="Return a + b.",
        entrypoint="add",
        commit_min_responses=1,
        commit_min_signed_responses=1,
    )


class Client:
    """Happy-path lease/commit/reveal; captures the feedback it receives."""

    def __init__(self):
        self.sent_feedback = None

    async def lease(self):
        return LeaseOutcome(
            category=LeaseCategory.LEASED, status=200, challenge=challenge()
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
            tests=[TestCase(args=[1, 2], expected=3)],
        )

    async def feedback(self, feedback):
        self.sent_feedback = feedback
        return True


def outcome(uid: int, *, code: str, passed: bool) -> MinerOutcome:
    return MinerOutcome(
        uid=uid,
        hotkey=f"hotkey-{uid}",
        solution=SolutionResponse(
            problem_id="problem-1",
            code=code,
            raw_response="" if code else "graded empty submission",
        ),
        verification=Verification(
            problem_id="problem-1",
            num_tests=1,
            num_passed=1 if passed else 0,
            all_passed=passed,
            reward=1.0 if passed else 0.0,
        ),
        reward=1.0 if passed else 0.0,
    )


class Orchestrator:
    def __init__(self, outcomes):
        self._outcomes = outcomes

    async def evaluate(self, problem, captured, semaphore):
        return ChallengeResult(
            problem_id=problem.problem_id,
            outcomes=self._outcomes,
            pass_rate=0.5,
            band=DifficultyBand.IN_BAND,
            num_responses=len(self._outcomes),
        )

    def score_and_export(self, problem, cr, active_uids=None):
        pass


def install_dispatch(monkeypatch, uids, unresponsive=()):
    """Committed submissions mirroring live.py's evidence exactly.

    A responsive miner's submission has error == "" and retained Epistula
    headers (live.py keeps headers only on the success path). An unresponsive
    one carries an error, an empty body, and NO headers — the shape every
    live.py failure path produces.
    """

    async def fake_dispatch(public, subset, concurrency):
        submissions = []
        for u in uids:
            if u in unresponsive:
                submissions.append(
                    MinerSubmission(
                        uid=u,
                        hotkey=f"hotkey-{u}",
                        request_id=f"req-{u}",
                        response_body="",
                        error="<miner total response deadline exceeded>",
                    )
                )
            else:
                submissions.append(
                    MinerSubmission(
                        uid=u,
                        hotkey=f"hotkey-{u}",
                        request_id=f"req-{u}",
                        response_body="{}",
                        response_headers={"Epistula-Signed-By": f"hotkey-{u}"},
                        error="",
                    )
                )
        captured = [SimpleNamespace(uid=u, hotkey=f"hotkey-{u}") for u in uids]
        return submissions, captured

    monkeypatch.setattr(decentralized, "_dispatch_committed", fake_dispatch)


async def run(client, orchestrator, uids):
    return await _evaluate_one(
        client,
        orchestrator,
        SimpleNamespace(sample=lambda pool, _k: pool),
        [SimpleNamespace(uid=u, hotkey=f"hotkey-{u}") for u in uids],
        Settings(_env_file=None, validator_dispatch_concurrency=4),
    )


# --------------------------------------------------------------------------- #
# Wire shape
# --------------------------------------------------------------------------- #
def test_feedback_accepts_a_bounded_verdict_list():
    from rlvr.problemserver.api import FeedbackVerdict

    feedback = ChallengeFeedback(
        challenge_id="challenge-1",
        pass_rate=0.5,
        band=DifficultyBand.IN_BAND,
        num_responses=2,
        verdicts=[
            FeedbackVerdict(uid=1, hotkey="hotkey-1", passed=True),
            FeedbackVerdict(uid=2, hotkey="hotkey-2", passed=False),
        ],
    )

    payload = json.loads(feedback.model_dump_json())
    assert payload["verdicts"] == [
        {"uid": 1, "hotkey": "hotkey-1", "passed": True},
        {"uid": 2, "hotkey": "hotkey-2", "passed": False},
    ]


def test_a_verdict_carries_nothing_but_uid_hotkey_passed():
    """The privacy boundary at the field level: exactly three keys, ever."""
    from rlvr.problemserver.api import FeedbackVerdict

    encoded = json.loads(
        FeedbackVerdict(uid=1, hotkey="hotkey-1", passed=True).model_dump_json()
    )

    assert set(encoded) == {"uid", "hotkey", "passed"}


def test_verdictless_feedback_serializes_exactly_as_before():
    """Legacy compatibility: an empty list must VANISH from the wire.

    extra=forbid on the server side means an unexpected field rejects the whole
    request. A validator with nothing to report must emit bytes a pre-verdict
    server accepts — the field omitted, not present-and-empty.
    """
    feedback = ChallengeFeedback(
        challenge_id="challenge-1",
        pass_rate=0.5,
        band=DifficultyBand.IN_BAND,
        num_responses=2,
    )

    assert "verdicts" not in json.loads(feedback.model_dump_json())


# --------------------------------------------------------------------------- #
# Who gets a verdict: dispatch evidence, not code emptiness
# --------------------------------------------------------------------------- #
async def test_graded_outcomes_reach_the_server_as_verdicts(monkeypatch):
    install_dispatch(monkeypatch, [1, 2])
    client = Client()
    orchestrator = Orchestrator(
        [outcome(1, code="def add(a,b): return a+b", passed=True),
         outcome(2, code="def add(a,b): return a-b", passed=False)]
    )

    await run(client, orchestrator, [1, 2])

    sent = {(v.uid, v.passed) for v in client.sent_feedback.verdicts}
    assert sent == {(1, True), (2, False)}


async def test_an_unresponsive_miner_is_committed_but_never_verdicted(monkeypatch):
    """UID 3 timed out: in the committed submissions, absent from verdicts.

    Its outcome exists with empty code and all_passed=False — the naive mapping
    over result.outcomes would send it as failed, rebranding silence as
    wrong-answering. Silence already lives in the response rate.
    """
    install_dispatch(monkeypatch, [1, 3], unresponsive={3})
    client = Client()
    orchestrator = Orchestrator(
        [outcome(1, code="def add(a,b): return a+b", passed=True),
         outcome(3, code="", passed=False)]
    )

    await run(client, orchestrator, [1, 3])

    uids = {v.uid for v in client.sent_feedback.verdicts}
    assert 3 not in uids, "an unresponsive miner received a verdict"
    assert uids == {1}


async def test_an_authenticated_empty_answer_is_a_graded_failure(monkeypatch):
    """The case that distinguishes the two candidate predicates.

    UID 4 returned a VALID signed payload whose code is empty: submission has
    error == "" and retained headers, but the graded solution has no code. That
    is a deliberate answer, it was verified, and it failed — passed=false. A
    code-is-empty predicate would wrongly file it with the silent miners; the
    dispatch-evidence predicate must not.
    """
    install_dispatch(monkeypatch, [1, 4])
    client = Client()
    orchestrator = Orchestrator(
        [outcome(1, code="def add(a,b): return a+b", passed=True),
         outcome(4, code="", passed=False)]  # responded, answered nothing
    )

    await run(client, orchestrator, [1, 4])

    sent = {(v.uid, v.passed) for v in client.sent_feedback.verdicts}
    assert (4, False) in sent, (
        "an authenticated empty answer got no verdict; empty code from a "
        "responsive miner is a wrong answer, not silence"
    )
    assert sent == {(1, True), (4, False)}


async def test_every_verdict_corresponds_to_a_committed_submission(monkeypatch):
    """The server validates membership; we must never fail it."""
    install_dispatch(monkeypatch, [1, 2, 3], unresponsive={3})
    client = Client()
    orchestrator = Orchestrator(
        [outcome(1, code="x = 1", passed=False),
         outcome(2, code="y = 2", passed=True),
         outcome(3, code="", passed=False)]
    )

    await run(client, orchestrator, [1, 2, 3])

    assert {v.uid for v in client.sent_feedback.verdicts} <= {1, 2, 3}


async def test_passed_reflects_verification_not_payment(monkeypatch):
    """all_passed is the fact; reward blends payment policy on top of it."""
    install_dispatch(monkeypatch, [1])
    client = Client()
    graded = outcome(1, code="def add(a,b): return a+b", passed=True)
    graded.reward = 0.0  # payment zeroed by policy; the code still passed
    orchestrator = Orchestrator([graded])

    await run(client, orchestrator, [1])

    assert client.sent_feedback.verdicts[0].passed is True
