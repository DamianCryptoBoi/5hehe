"""The contacted, responded, and signed counts must remain distinct.

The spec asks the validator to "distinguish contact count, responses, and signed
responses". Three labels are only worth printing if they can hold three
different numbers.

They currently cannot. In live.py every failure path builds a SignedSolution
with an error and WITHOUT response_headers, and the success path is the only one
that sets headers — so `error == ""` and `headers != {}` are the same condition,
and `responses` and `signed` are equal by construction for every possible input.

The distinction that is being lost is the operationally useful one. A miner
answering HTTP 500 is UP AND BROKEN; a miner that never answers is UNREACHABLE.
Those call for different operator responses — restart a service versus chase a
network — and today both are reported identically.

"""

from __future__ import annotations

import httpx
import pytest

from rlvr.protocol import SignedSolution

def counts(submissions):
    """Mirror of the counting in _evaluate_one, kept in step deliberately.

    If the production expression changes, this must change with it — that is the
    point. It exists so the SEMANTICS can be asserted, not the formatting.
    """
    responded = sum(bool(getattr(s, "responded", False)) for s in submissions)
    signed = sum(bool(getattr(s, "response_headers", {})) for s in submissions)
    return responded, signed


def test_a_miner_that_answers_badly_is_not_the_same_as_one_that_is_absent():
    """HTTP 500 from a miner is a response. It must count as one.

    Reporting an answering-but-failing miner identically to an unreachable one
    sends an operator to debug the network when the miner's own service is the
    fault.
    """
    answered_badly = SignedSolution(
        error="HTTP 500: internal error", latency_ms=12.0, responded=True
    )
    never_answered = SignedSolution(error="<dispatch error: connect refused>")

    responded, signed = counts([answered_badly, never_answered])

    assert responded == 1, (
        "a miner that returned HTTP 500 is not counted as having responded; "
        "'unreachable' and 'up but broken' are indistinguishable in the log"
    )
    assert signed == 0


def test_the_three_counts_can_hold_three_different_values():
    """The whole purpose of printing three labels.

    Two contacted, one of which answered unverifiably: contacted=2,
    responses=1, signed=0 — three distinct numbers from one dispatch.
    """
    verified = SignedSolution(
        response_body="{}",
        response_headers={"Epistula-Signed-By": "hotkey-0"},
        latency_ms=5.0,
        responded=True,
    )
    unverifiable = SignedSolution(
        error="<invalid or unauthenticated miner response>",
        latency_ms=7.0,
        responded=True,
    )
    absent = SignedSolution(error="<miner total response deadline exceeded>")

    responded, signed = counts([verified, unverifiable, absent])

    assert (len([verified, unverifiable, absent]), responded, signed) == (3, 2, 1)


def test_a_verified_response_counts_in_both():
    """Control: the happy path must remain counted everywhere."""
    verified = SignedSolution(
        response_body="{}",
        response_headers={"Epistula-Signed-By": "hotkey-0"},
        latency_ms=5.0,
        responded=True,
    )

    assert counts([verified]) == (1, 1)


def test_an_unreachable_miner_counts_in_neither():
    """Control: nothing came back, so nothing is counted."""
    absent = SignedSolution(error="<dispatch error: connection refused>")

    assert counts([absent]) == (0, 0)


@pytest.mark.parametrize(
    "status", [400, 401, 403, 404, 429, 500, 502, 503]
)
def test_every_http_status_from_a_miner_is_a_response(status):
    """Any HTTP status means the miner's server accepted and answered."""
    answered = SignedSolution(
        error=f"HTTP {status}: refused", latency_ms=3.0, responded=True
    )

    responded, signed = counts([answered])

    assert (responded, signed) == (1, 0)


def test_an_oversized_body_is_a_response_too():
    """The miner answered; we declined to read it. That is not silence."""
    oversized = SignedSolution(
        error="<miner response exceeds byte limit>",
        latency_ms=9.0,
        responded=True,
    )

    responded, _ = counts([oversized])

    assert responded == 1
