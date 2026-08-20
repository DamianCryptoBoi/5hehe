"""Lease diagnostic behavior.

These assertions exercise the public client boundary and describe required
behavior rather than whatever the code happens to do. Each
case is one an operator would act on differently.

"""

from __future__ import annotations

import json

import pytest

from rlvr.problemserver.client import (
    LeaseCategory,
    LeaseOutcome,
    ProblemServerClient,
)

httpx = pytest.importorskip("httpx")


_CHALLENGE = {
    "challenge_id": "challenge-1",
    "problem_id": "problem-1",
    "language": "python",
    "statement": "Return a + b.",
    "entrypoint": "add",
}


class Scripted:
    """Transport that plays a scripted sequence and counts attempts."""

    def __init__(self, steps):
        self._steps = steps
        self.attempts = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        step = self._steps[min(self.attempts, len(self._steps) - 1)]
        self.attempts += 1
        if isinstance(step, BaseException):
            raise type(step)(str(step), request=request)
        return step


def make_client(handler, *, retries: int = 3, max_response_bytes: int = 512_000):
    return ProblemServerClient(
        url="https://problems.invalid",
        wallet="validator-hotkey",
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retries=retries,
        max_response_bytes=max_response_bytes,
    )


async def lease_with(handler, **kwargs) -> LeaseOutcome:
    client = make_client(handler, **kwargs)
    try:
        return await client.lease()
    finally:
        await client._http.aclose()


# --------------------------------------------------------------------------- #
# 1A — a final transport failure must not erase HTTP evidence we already have
# --------------------------------------------------------------------------- #
async def test_transport_after_server_error_keeps_the_observed_status():
    """503 then a dead socket: the server DID answer, and the log must say so.

    Reporting only "unreachable" sends an operator to check DNS and firewalls
    when the actual signal is that the problem source was failing requests.
    """
    handler = Scripted(
        [
            httpx.Response(503, content=b"upstream down"),
            httpx.ConnectError("connection refused"),
            httpx.ConnectError("connection refused"),
        ]
    )
    outcome = await lease_with(handler)

    # The final attempt had no HTTP response, so the category is TRANSPORT.
    assert outcome.category is LeaseCategory.TRANSPORT
    # ...but the 503 we did observe is evidence, not noise.
    assert outcome.status == 503
    described = outcome.describe()
    assert "503" in described
    assert "connection refused" in described
    # It must not read as though the server never answered at all.
    assert described != LeaseOutcome(category=LeaseCategory.TRANSPORT).describe()


async def test_transport_only_failure_reports_no_status():
    """No HTTP response ever arrived, so there is no status to invent."""
    handler = Scripted([httpx.ConnectError("connection refused")])
    outcome = await lease_with(handler)

    assert outcome.category is LeaseCategory.TRANSPORT
    assert outcome.status is None
    assert "connection refused" in outcome.detail


async def test_server_error_on_the_final_attempt_stays_unavailable():
    """The mirror of the 1A case: last attempt IS an HTTP response."""
    handler = Scripted(
        [
            httpx.ConnectError("connection refused"),
            httpx.Response(503, content=b"upstream down"),
            httpx.Response(503, content=b"upstream down"),
        ]
    )
    outcome = await lease_with(handler)

    assert outcome.category is LeaseCategory.UNAVAILABLE
    assert outcome.status == 503
    assert outcome.detail == "upstream down"


# --------------------------------------------------------------------------- #
# 1B — a completed HTTP response with an unusable body is a protocol failure
# --------------------------------------------------------------------------- #
async def test_oversized_body_is_malformed_not_unreachable():
    """HTTP 200 arrived. Calling that "unreachable" points at the wrong system."""
    handler = Scripted([httpx.Response(200, content=b"x" * 5000)])
    outcome = await lease_with(handler, max_response_bytes=100)

    assert outcome.category is LeaseCategory.MALFORMED
    assert outcome.status == 200
    assert outcome.detail


async def test_invalid_content_length_is_malformed_not_unreachable():
    handler = Scripted(
        [
            httpx.Response(
                200, content=b"x" * 50, headers={"Content-Length": "999999"}
            )
        ]
    )
    outcome = await lease_with(handler, max_response_bytes=100)

    assert outcome.category is LeaseCategory.MALFORMED
    assert outcome.status == 200


async def test_unusable_body_is_not_retried_because_a_lease_burns_a_problem():
    """The decisive case: retrying cannot help and is not free.

    A 200 means the problem source already committed a problem off its durable
    cursor. The body is deterministic, so a second and third attempt re-read the
    same unusable bytes while burning two more problems from a scarce pool.
    """
    handler = Scripted([httpx.Response(200, content=b"x" * 5000)])
    outcome = await lease_with(handler, retries=3, max_response_bytes=100)

    assert outcome.category is LeaseCategory.MALFORMED
    assert handler.attempts == 1, (
        f"an unusable body was fetched {handler.attempts} times; each 200 "
        "lease burns a problem server-side"
    )


async def test_retryable_status_still_retries():
    """Guard against over-correcting 1B into disabling legitimate retries."""
    handler = Scripted(
        [
            httpx.Response(503, content=b"down"),
            httpx.Response(200, content=json.dumps(_CHALLENGE).encode()),
        ]
    )
    outcome = await lease_with(handler, retries=3)

    assert outcome.category is LeaseCategory.LEASED
    assert handler.attempts == 2


# --------------------------------------------------------------------------- #
# 1C — a log line an operator reads must be readable
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("seconds", "rendered"),
    [
        (0, "0s"),
        (45, "45s"),
        (3600, "3600s"),
        (1209600, "1209600s"),
        (999999999, "999999999s"),
    ],
)
def test_retry_after_renders_as_exact_whole_seconds(seconds, rendered):
    """`:g` turns 1209600 into 1.2096e+06, which no operator can act on."""
    outcome = LeaseOutcome(
        category=LeaseCategory.PACED, status=429, retry_after_s=seconds
    )
    described = outcome.describe()

    assert rendered in described
    assert "e+" not in described


# --------------------------------------------------------------------------- #
# 1B follow-up — the no-retry rule must be scoped by what a response COST us
# --------------------------------------------------------------------------- #
async def test_unusable_body_on_a_retryable_status_still_retries():
    """A 5xx burns no problem, so declining to retry it forfeits free recovery.

    The reason an unusable 200 must not be retried is that it already consumed a
    problem from the durable FIFO cursor, and re-reading it consumes more. That
    reasoning is specific to a response that cost us something. A 503 costs
    nothing, and here the very next attempt returns a valid challenge — so
    stopping at the first attempt turns a recoverable blip into a dead round.
    """
    handler = Scripted(
        [
            httpx.Response(503, content=b"x" * 5000),
            httpx.Response(200, content=json.dumps(_CHALLENGE).encode()),
        ]
    )
    outcome = await lease_with(handler, retries=3, max_response_bytes=1000)

    assert outcome.category is LeaseCategory.LEASED
    assert handler.attempts == 2, (
        f"gave up after {handler.attempts} attempt(s); an unusable body on a "
        "status that burns no problem must still be retried"
    )


async def test_unusable_body_on_a_retryable_status_reports_the_status_when_it_never_recovers():
    """Exhausting retries on that path still reports the server's own status."""
    handler = Scripted([httpx.Response(503, content=b"x" * 5000)])
    outcome = await lease_with(handler, retries=3, max_response_bytes=1000)

    assert outcome.category is LeaseCategory.UNAVAILABLE
    assert outcome.status == 503
    assert handler.attempts == 3


# --------------------------------------------------------------------------- #
# 1D — an unreadable body must not erase what the STATUS already told us
#
# The whole point of item 1 was to stop collapsing distinct causes into one
# opaque failure. A local read refusal must not re-collapse them one layer
# down: the status is evidence we already hold, and it keeps its meaning
# whether or not the body was readable. MALFORMED belongs to 200 alone — the
# only status that promised a challenge in the first place.
# --------------------------------------------------------------------------- #
async def test_a_paced_lease_stays_paced_when_the_body_is_unreadable():
    """The costliest misfiling: a pacing directive we received and parsed.

    The server told us to slow down, and we have the directive in hand. Filing
    it as MALFORMED means the item-2 gate never defers, so we keep hammering a
    source that just asked us to stop — while the log tells the operator their
    problem server is emitting garbage.
    """
    handler = Scripted(
        [httpx.Response(429, content=b"x" * 5000, headers={"Retry-After": "300"})]
    )
    outcome = await lease_with(handler, retries=3, max_response_bytes=100)

    assert outcome.category is LeaseCategory.PACED
    assert outcome.status == 429
    assert outcome.retry_after_s == 300
    # 429 is not retryable, and re-asking a source that said "slow down" is the
    # one thing this outcome exists to prevent.
    assert handler.attempts == 1
    # The reason the body was refused is ours, not the server's words.
    assert outcome.detail_is_server is False
    assert outcome.detail
    described = outcome.describe()
    assert "retry after 300s" in described
    assert "unusable challenge" not in described
    # Same attempt supplied the directive, so nothing was carried forward.
    assert outcome.retry_after_from_prior_response is False


@pytest.mark.parametrize("status", [401, 403])
async def test_an_unauthorized_lease_stays_denied_when_the_body_is_unreadable(status):
    """"Unusable challenge" sends an operator to debug the wrong system.

    A validator that is not authorized needs a credential, not a bug report
    against the problem server. Retrying is equally wrong: the request was
    refused, and refusal is not transient.
    """
    handler = Scripted([httpx.Response(status, content=b"x" * 5000)])
    outcome = await lease_with(handler, retries=3, max_response_bytes=100)

    assert outcome.category is LeaseCategory.DENIED
    assert outcome.status == status
    assert handler.attempts == 1
    assert outcome.detail_is_server is False
    assert "unusable challenge" not in outcome.describe()


# --------------------------------------------------------------------------- #
# 1E — Retry-After provenance
#
# Carrying a directive forward from an earlier attempt is deliberate and errs
# toward obeying the server. But the value is deliberately unbounded, so a
# stale 1209600 attached to a final response that asked for nothing is a
# two-week deferral nobody currently requested. The log must say which attempt
# it came from, and the flag must be false when the final response said it.
# --------------------------------------------------------------------------- #
async def test_retry_after_from_the_final_response_is_not_marked_as_prior():
    """Guard against blanket-setting the flag: this directive is current."""
    handler = Scripted(
        [httpx.Response(429, content=b"slow down", headers={"Retry-After": "300"})]
    )
    outcome = await lease_with(handler, retries=3)

    assert outcome.category is LeaseCategory.PACED
    assert outcome.retry_after_s == 300
    assert outcome.retry_after_from_prior_response is False
    assert "earlier" not in outcome.describe()


async def test_retry_after_carried_from_an_earlier_response_is_marked_as_prior():
    """The final 503 withdrew the directive; we still obey the one we got."""
    handler = Scripted(
        [
            httpx.Response(503, content=b"down", headers={"Retry-After": "600"}),
            httpx.Response(503, content=b"down"),
        ]
    )
    outcome = await lease_with(handler, retries=3)

    assert outcome.category is LeaseCategory.UNAVAILABLE
    assert outcome.status == 503
    assert outcome.retry_after_s == 600
    assert handler.attempts == 3
    assert outcome.retry_after_from_prior_response is True
    assert "earlier" in outcome.describe()


async def test_a_replaced_directive_is_not_marked_as_prior():
    """The flag must not be sticky once a fresher directive supersedes the old.

    Attempt 1 asks for 600s; the final attempt asks for 30s. We obey 30 — the
    server is entitled to shorten its own pacing — and the log must not claim
    that value came from an earlier attempt when it did not.
    """
    handler = Scripted(
        [
            httpx.Response(503, content=b"down", headers={"Retry-After": "600"}),
            httpx.Response(503, content=b"down", headers={"Retry-After": "30"}),
        ]
    )
    outcome = await lease_with(handler, retries=3)

    assert outcome.category is LeaseCategory.UNAVAILABLE
    assert outcome.retry_after_s == 30
    assert outcome.retry_after_from_prior_response is False
    assert "earlier" not in outcome.describe()


async def test_retry_after_carried_across_a_transport_failure_is_marked_as_prior():
    """No final response at all cannot have supplied the directive."""
    handler = Scripted(
        [
            httpx.Response(503, content=b"down", headers={"Retry-After": "600"}),
            httpx.ConnectError("connection refused"),
        ]
    )
    outcome = await lease_with(handler, retries=3)

    assert outcome.category is LeaseCategory.TRANSPORT
    assert outcome.status == 503
    assert outcome.retry_after_s == 600
    assert outcome.retry_after_from_prior_response is True


async def test_a_directive_on_an_unreadable_body_survives_a_later_transport_loss():
    """The mixed case that has slipped past both of us in every earlier pass.

    Attempt 1 answers 503 with a pacing directive and a body we refuse to read;
    the socket then dies. Three separate things must hold at once: the refused
    read does not stop the retry budget on a status that burns no problem, the
    directive we parsed off that attempt is retained, and the category still
    names transport as what failed last. A stale local read error must not
    outrank the socket failure and re-file this as a server-side failure.
    """
    handler = Scripted(
        [
            httpx.Response(
                503, content=b"x" * 5000, headers={"Retry-After": "60"}
            ),
            httpx.ConnectError("connection refused"),
        ]
    )
    outcome = await lease_with(handler, retries=3, max_response_bytes=100)

    assert outcome.category is LeaseCategory.TRANSPORT
    assert outcome.status == 503
    assert outcome.retry_after_s == 60
    assert outcome.retry_after_from_prior_response is True
    assert handler.attempts == 3
    described = outcome.describe()
    assert "connection refused" in described
    assert "retry after 60s" in described


def test_provenance_is_rendered_only_for_a_carried_directive():
    """Pins the rendering itself, so neither branch can silently lose its label."""
    carried = LeaseOutcome(
        category=LeaseCategory.UNAVAILABLE,
        status=503,
        retry_after_s=1209600,
        retry_after_from_prior_response=True,
    ).describe()
    current = LeaseOutcome(
        category=LeaseCategory.PACED,
        status=429,
        retry_after_s=1209600,
    ).describe()

    assert "1209600s" in carried and "1209600s" in current
    assert "earlier" in carried
    assert "earlier" not in current


# --------------------------------------------------------------------------- #
# 1F — retained per-attempt state must not outlive the attempt that produced it
# --------------------------------------------------------------------------- #
async def test_recovery_does_not_leak_the_earlier_attempt_s_failure():
    """A successful attempt must not inherit a previous attempt's error text.

    Retained per-attempt state is what makes evidence survive; it is also how a
    stale failure leaks onto a success if the happy path forgets to clear it.
    """
    handler = Scripted(
        [
            httpx.Response(503, content=b"x" * 5000),
            httpx.Response(200, content=json.dumps(_CHALLENGE).encode()),
        ]
    )
    outcome = await lease_with(handler, retries=3, max_response_bytes=1000)

    assert outcome.category is LeaseCategory.LEASED
    assert outcome.challenge is not None
    assert outcome.detail == "", f"stale failure leaked onto a success: {outcome.detail!r}"
