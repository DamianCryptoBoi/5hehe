"""Server-directed pacing wiring.

This is the first behavior that touches decentralized.py. Two halves:
  * `_evaluate_one` REPORTS server-directed pacing instead of discarding it;
  * the per-round challenge loop ACTS on it — installing the deadline and
    stopping, rather than leasing again inside the same callback.

The second half is why the loop must move out of the round_callback closure and
become an addressable function. A closure defined inside
`_run_decentralized_validator_async` cannot be exercised without standing up the
whole live runner, and an untestable break is how this regresses later.

Deliberately EXCLUDED: cadence defaults (75/180), and any change to how
non-paced causes drive the loop.

"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlvr.config import Settings
from rlvr.neurons.decentralized import _evaluate_one, _run_challenge_round
from rlvr.problemserver.client import (
    _RETRY_AFTER_CUSHION_S,
    LeaseCategory,
    LeaseOutcome,
)
from rlvr.policy import ValidatorPolicy

NOW = 5_000.0


def settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


class LeaseClient:
    """Serves scripted lease outcomes and counts how many were requested."""

    def __init__(self, *outcomes: LeaseOutcome):
        self._outcomes = list(outcomes)
        self.leases = 0

    async def lease(self) -> LeaseOutcome:
        outcome = self._outcomes[min(self.leases, len(self._outcomes) - 1)]
        self.leases += 1
        return outcome


class RecordingValidator:
    """Captures deferral installs without needing a chain or a run loop."""

    def __init__(self):
        self.deferrals: list[float] = []

    def defer_rounds_until(self, deadline: float) -> None:
        self.deferrals.append(deadline)


def solvers(n: int = 2):
    return [SimpleNamespace(uid=i, hotkey=f"hk-{i}") for i in range(n)]


def rotation():
    return SimpleNamespace(sample=lambda pool, _k: pool)


PACED_WITH_DIRECTIVE = LeaseOutcome(
    category=LeaseCategory.PACED, status=429, detail="slow down", retry_after_s=45
)
PACED_NO_DIRECTIVE = LeaseOutcome(
    category=LeaseCategory.PACED, status=429, detail="slow down"
)
EMPTY_POOL = LeaseOutcome(category=LeaseCategory.EMPTY, status=204)
DENIED = LeaseOutcome(category=LeaseCategory.DENIED, status=403, detail="nope")
TRANSPORT = LeaseOutcome(category=LeaseCategory.TRANSPORT, detail="refused")


# --------------------------------------------------------------------------- #
# Half one: _evaluate_one reports pacing instead of swallowing it
# --------------------------------------------------------------------------- #
async def test_a_paced_lease_reports_a_converted_deadline():
    pacing: dict = {}
    result = await _evaluate_one(
        LeaseClient(PACED_WITH_DIRECTIVE),
        None,
        rotation(),
        solvers(),
        settings(),
        pacing=pacing,
        now=NOW,
    )

    assert result is None
    assert pacing["paced"] is True
    assert pacing["not_before"] == NOW + 45 + _RETRY_AFTER_CUSHION_S


async def test_a_paced_lease_without_a_usable_directive_reports_no_deadline():
    """A 429 with a malformed Retry-After still means 'stop', but not 'how long'.

    Inventing a duration here would be exactly the guessing the parser refuses
    to do; the caller falls back to its ordinary cadence instead.
    """
    pacing: dict = {}
    await _evaluate_one(
        LeaseClient(PACED_NO_DIRECTIVE), None, rotation(), solvers(),
        settings(), pacing=pacing, now=NOW,
    )

    assert pacing["paced"] is True
    assert "not_before" not in pacing


@pytest.mark.parametrize("outcome", [EMPTY_POOL, DENIED, TRANSPORT])
async def test_non_paced_failures_never_defer(outcome):
    """An empty pool is not a pacing signal, and neither is a dead socket.

    Deferring on these would suppress leasing for reasons the server never
    asked for — the validator would pace itself out of work.
    """
    pacing: dict = {}
    await _evaluate_one(
        LeaseClient(outcome), None, rotation(), solvers(),
        settings(), pacing=pacing, now=NOW,
    )

    assert not pacing.get("paced")
    assert "not_before" not in pacing


# --------------------------------------------------------------------------- #
# Half two: the challenge loop acts on it
# --------------------------------------------------------------------------- #
async def test_pacing_stops_the_multi_challenge_loop_immediately():
    """The hazard: the deadline is useless until the callback returns.

    With five challenges configured, the run-loop gate cannot intervene between
    iterations. Four more leases would be fired into a server that just asked
    for silence.
    """
    client = LeaseClient(PACED_WITH_DIRECTIVE)
    v = RecordingValidator()

    completed = await _run_challenge_round(
        v, client, None, rotation(), solvers(),
        settings(), now=NOW, policy=ValidatorPolicy(challenges_per_round=5),
    )

    assert client.leases == 1, f"leased {client.leases} times while paced"
    assert completed == 0
    assert v.deferrals == [NOW + 45 + _RETRY_AFTER_CUSHION_S]


async def test_pacing_without_a_directive_stops_the_loop_without_deferring():
    client = LeaseClient(PACED_NO_DIRECTIVE)
    v = RecordingValidator()

    await _run_challenge_round(
        v, client, None, rotation(), solvers(),
        settings(), now=NOW, policy=ValidatorPolicy(challenges_per_round=5),
    )

    assert client.leases == 1
    assert v.deferrals == [], "deferred without a server directive"


async def test_a_non_paced_failure_does_not_shorten_the_round():
    """Control: only pacing stops the loop. Existing behavior is preserved."""
    client = LeaseClient(EMPTY_POOL)
    v = RecordingValidator()

    await _run_challenge_round(
        v, client, None, rotation(), solvers(),
        settings(), now=NOW, policy=ValidatorPolicy(challenges_per_round=5),
    )

    assert client.leases == 5
    assert v.deferrals == []


# --------------------------------------------------------------------------- #
# Composition — every behavior above is tested in isolation; this is the seam
# --------------------------------------------------------------------------- #
async def test_a_real_429_over_http_reaches_the_validator_as_a_deadline():
    """The whole chain, with no stubbed LeaseOutcome anywhere in it.

    Behaviors 1-5 were each verified alone, and the multi-challenge hammering
    defect still slipped between them. This exercises a genuine HTTP 429 through
    parsing, categorization, conversion and installation in one pass, so a break
    anywhere along that seam fails here even when every unit test still passes.
    """
    pytest.importorskip("httpx")
    import httpx

    from rlvr.problemserver.client import ProblemServerClient

    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(
            429,
            content=b'{"detail": "global lease slot busy"}',
            headers={"Retry-After": "45"},
        )

    client = ProblemServerClient(
        url="https://problems.invalid",
        wallet="validator-hotkey",
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    v = RecordingValidator()
    try:
        completed = await _run_challenge_round(
            v, client, None, rotation(), solvers(),
            settings(), now=NOW, policy=ValidatorPolicy(challenges_per_round=5),
        )
    finally:
        await client._http.aclose()

    assert attempts["n"] == 1, (
        f"{attempts['n']} leases fired at a server that returned 429"
    )
    assert completed == 0
    assert v.deferrals == [NOW + 45 + _RETRY_AFTER_CUSHION_S]
