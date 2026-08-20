"""Lease diagnostics for the problem-source client.

Every case here is one of the causes that used to collapse into a bare ``None``
return, so the assertions are about telling them apart.
"""

from __future__ import annotations

import json

import pytest

from rlvr.problemserver.api import ChallengeFeedback, DifficultyBand, PublicChallenge
from rlvr.problemserver.client import (
    _MAX_DETAIL_CHARS,
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


def make_client(
    handler, *, retries: int = 1, max_response_bytes: int = 512_000
) -> ProblemServerClient:
    """A client whose transport is fully mocked; no network, no crypto stack."""
    return ProblemServerClient(
        url="https://problems.invalid",
        # A plain string signs via the HMAC fallback path in sign_message.
        wallet="validator-hotkey",
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retries=retries,
        max_response_bytes=max_response_bytes,
    )


def responder(status_code: int, body: bytes = b"", headers=None):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, content=body, headers=headers or {})

    return handler


# --- multi-attempt harness ------------------------------------------------
#
# Every test above runs at retries=1, so attempt 1 is always the final attempt
# and the retry loop is never actually exercised. The mixed-sequence defects
# only exist across attempts, so they need a handler that varies by attempt.

REFUSED = object()
"""Sentinel step: the attempt dies in transport, with no HTTP response at all."""


def served(status_code: int, body: bytes = b"", headers=None):
    """One HTTP step for :func:`sequence`."""
    return (status_code, body, headers or {})


def sequence(*steps):
    """Replay `steps` one per attempt; the last step repeats once exhausted.

    The handler carries `.attempts`, so a test can assert how many requests the
    retry loop actually issued rather than assuming.
    """
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        step = steps[min(state["n"], len(steps) - 1)]
        state["n"] += 1
        if step is REFUSED:
            raise httpx.ConnectError("connection refused", request=request)
        status_code, body, headers = step
        return httpx.Response(status_code, content=body, headers=headers)

    handler.attempts = state
    return handler


async def call_legacy(client: ProblemServerClient, method: str):
    """Invoke one of the three non-lease callers with a valid request body."""
    if method == "commit":
        return await client.commit("challenge-1", [])
    if method == "reveal":
        return await client.reveal("challenge-1", "commit-token")
    return await client.feedback(
        ChallengeFeedback(
            challenge_id="challenge-1",
            pass_rate=0.5,
            band=DifficultyBand.IN_BAND,
            num_responses=1,
        )
    )


# The failure shape each legacy caller has always returned, and must keep
# returning: preserving richer lease evidence must not leak a failed response
# into callers that treat "not None" as success.
LEGACY_FAILURE = {"commit": None, "reveal": None, "feedback": False}


async def lease_once(
    handler, *, retries: int = 1, max_response_bytes: int = 512_000
) -> LeaseOutcome:
    client = make_client(
        handler, retries=retries, max_response_bytes=max_response_bytes
    )
    try:
        return await client.lease()
    finally:
        await client._http.aclose()


async def test_successful_lease_carries_the_challenge_and_status():
    outcome = await lease_once(responder(200, json.dumps(_CHALLENGE).encode()))

    assert outcome.category is LeaseCategory.LEASED
    assert outcome.leased
    assert outcome.challenge == PublicChallenge(**_CHALLENGE)
    assert outcome.status == 200
    assert outcome.retry_after_s is None


async def test_paced_lease_is_distinguishable_from_an_empty_pool():
    paced = await lease_once(
        responder(
            429,
            json.dumps({"detail": "lease budget exhausted"}).encode(),
            {"Retry-After": "45"},
        )
    )
    empty = await lease_once(responder(204))

    assert paced.category is LeaseCategory.PACED
    assert paced.retry_after_s == 45.0
    assert paced.detail == "lease budget exhausted"
    assert empty.category is LeaseCategory.EMPTY
    assert empty.retry_after_s is None
    # The distinction the flat log line could not make.
    assert paced.describe() != empty.describe()
    assert not paced.leased and not empty.leased


@pytest.mark.parametrize("status_code", [401, 403])
async def test_unauthorized_lease_reports_denial_not_absence(status_code):
    outcome = await lease_once(responder(status_code, b"hotkey not registered"))

    assert outcome.category is LeaseCategory.DENIED
    assert outcome.status == status_code
    assert outcome.detail == "hotkey not registered"
    assert outcome.challenge is None


async def test_other_client_errors_report_rejection():
    outcome = await lease_once(responder(400, b"bad request"))

    assert outcome.category is LeaseCategory.REJECTED
    assert outcome.status == 400


async def test_transport_failure_has_no_status_and_keeps_the_cause():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    outcome = await lease_once(refuse)

    assert outcome.category is LeaseCategory.TRANSPORT
    # No HTTP exchange happened, so there is no status to report.
    assert outcome.status is None
    assert outcome.transport_error == "connection refused"


async def test_persistent_server_error_is_reported_as_unavailable():
    outcome = await lease_once(responder(503, b"upstream down"))

    assert outcome.category is LeaseCategory.UNAVAILABLE
    assert outcome.status == 503
    assert outcome.detail == "upstream down"


async def test_unparseable_challenge_is_malformed_not_missing():
    outcome = await lease_once(responder(200, b'{"challenge_id": ""}'))

    assert outcome.category is LeaseCategory.MALFORMED
    assert outcome.status == 200
    assert outcome.challenge is None
    assert outcome.detail


@pytest.mark.parametrize(
    "raw",
    [
        "not-a-number",
        "-5",
        "1.5",
        "",
        "30s",
        "+5",
        "1_0",
        "Wed, 21 Oct 2026 07:28:00 GMT",
    ],
)
async def test_malformed_retry_after_fails_closed(raw):
    outcome = await lease_once(responder(429, b"", {"Retry-After": raw}))

    assert outcome.category is LeaseCategory.PACED
    assert outcome.retry_after_s is None


async def test_large_valid_retry_after_is_preserved():
    outcome = await lease_once(responder(429, b"", {"Retry-After": "7200"}))

    assert outcome.retry_after_s == 7200


async def test_zero_retry_after_is_valid():
    outcome = await lease_once(responder(429, b"", {"Retry-After": "0"}))

    assert outcome.retry_after_s == 0


async def test_server_detail_is_bounded_and_stays_on_one_log_line():
    hostile = "x" * 5000 + "\nEpistula-Request-Signature: 0xdeadbeef\r\n\x1b[31m"
    outcome = await lease_once(responder(400, hostile.encode()))

    assert len(outcome.detail) <= _MAX_DETAIL_CHARS
    assert "\n" not in outcome.detail and "\r" not in outcome.detail
    assert "\x1b" not in outcome.detail
    assert "\n" not in outcome.describe()


async def test_non_json_and_unlabeled_json_details_both_survive():
    plain = await lease_once(responder(400, b"  plain   text\n  reason  "))
    unlabeled = await lease_once(responder(400, b'{"code": 17}'))

    assert plain.detail == "plain text reason"
    assert unlabeled.detail == '{"code": 17}'


async def test_describe_names_the_cause_and_the_wire_facts():
    outcome = await lease_once(
        responder(429, json.dumps({"error": "slow down"}).encode(), {"Retry-After": "5"})
    )

    described = outcome.describe()
    assert "paced" in described
    assert "HTTP 429" in described
    assert "5s" in described
    assert "slow down" in described


# --- retry sequences ------------------------------------------------------


async def test_late_transport_failure_keeps_the_earlier_http_evidence():
    """A server that answered and then went away did not "never answer"."""
    handler = sequence(
        served(503, b"upstream down", {"Retry-After": "60"}),
        served(503, b"upstream down"),
        REFUSED,
    )
    outcome = await lease_once(handler, retries=3)

    assert handler.attempts["n"] == 3
    # The final attempt failed in transport, so that is the category...
    assert outcome.category is LeaseCategory.TRANSPORT
    # ...but everything the server did say survives it.
    assert outcome.status == 503
    assert outcome.detail == "upstream down"
    assert outcome.retry_after_s == 60
    assert outcome.transport_error == "connection refused"

    described = outcome.describe()
    assert "503" in described
    assert "upstream down" in described
    assert "connection refused" in described
    assert "\n" not in described


async def test_early_transport_failure_is_not_claimed_once_the_server_answers():
    """The mirror case: a recovered socket must not leave a stale claim."""
    handler = sequence(REFUSED, served(503, b"upstream down"))
    outcome = await lease_once(handler, retries=2)

    assert outcome.category is LeaseCategory.UNAVAILABLE
    assert outcome.status == 503
    assert outcome.detail == "upstream down"
    # The server was reachable on the final attempt; saying otherwise would send
    # the operator to check DNS and firewalls for an application failure.
    assert outcome.transport_error == ""
    assert "connection refused" not in outcome.describe()


async def test_retry_exhaustion_preserves_status_and_detail():
    """The preservation this whole fix is about, over a real multi-attempt run."""
    handler = sequence(served(503, b"upstream down"))
    outcome = await lease_once(handler, retries=2)

    assert handler.attempts["n"] == 2
    assert outcome.category is LeaseCategory.UNAVAILABLE
    assert outcome.status == 503
    assert outcome.detail == "upstream down"


async def test_retryable_status_then_success_recovers():
    handler = sequence(served(503, b"upstream down"), served(200, json.dumps(_CHALLENGE).encode()))
    outcome = await lease_once(handler, retries=2)

    assert handler.attempts["n"] == 2
    assert outcome.category is LeaseCategory.LEASED
    assert outcome.challenge == PublicChallenge(**_CHALLENGE)


# --- a completed response we refuse to read is not an unreachable server ---


async def test_oversized_success_body_is_not_reported_as_unreachable():
    handler = responder(200, json.dumps(_CHALLENGE).encode())
    outcome = await lease_once(handler, max_response_bytes=64)

    # The server was reachable and answered 200; the refusal to read is ours.
    assert outcome.category is LeaseCategory.MALFORMED
    assert outcome.status == 200
    assert "byte limit" in outcome.detail
    # And our own limit must not be quoted back as something the server said.
    assert "server said" not in outcome.describe()
    assert "byte limit" in outcome.describe()


async def test_invalid_content_length_is_not_reported_as_unreachable():
    handler = responder(
        200, json.dumps(_CHALLENGE).encode(), {"Content-Length": "not-a-number"}
    )
    outcome = await lease_once(handler, max_response_bytes=512_000)

    assert outcome.category is LeaseCategory.MALFORMED
    assert outcome.status == 200
    assert "content-length" in outcome.detail.lower()


async def test_oversized_error_body_keeps_its_http_failure_category():
    """An unreadable body does not change WHY the request failed: it was a 503."""
    handler = sequence(served(503, b"y" * 5000))
    outcome = await lease_once(handler, retries=3, max_response_bytes=64)

    # A retryable HTTP failure consumes no problem and keeps its retry budget.
    assert handler.attempts["n"] == 3
    assert outcome.status == 503
    assert outcome.category is LeaseCategory.UNAVAILABLE
    assert "byte limit" in outcome.detail


async def test_malformed_challenge_does_not_attribute_the_parse_error_to_the_server():
    """`server said: 4 validation errors for PublicChallenge` — it said no such thing."""
    outcome = await lease_once(responder(200, b'{"challenge_id": ""}'))

    described = outcome.describe()
    assert outcome.category is LeaseCategory.MALFORMED
    assert "server said" not in described
    # The reason still has to reach the log, just not in the server's voice.
    assert "validation error" in described


# --- exact rendering ------------------------------------------------------


async def test_long_retry_after_renders_exactly():
    """Refusing to clamp a valid value is pointless if the log mangles it anyway."""
    outcome = await lease_once(responder(429, b"slow down", {"Retry-After": "1209600"}))

    assert outcome.retry_after_s == 1209600
    described = outcome.describe()
    assert "1209600s" in described
    assert "e+" not in described


# --- the non-lease callers must not notice any of this --------------------


@pytest.mark.parametrize("method", ["commit", "reveal", "feedback"])
async def test_legacy_callers_stay_failure_shaped_on_exhausted_retryable_status(method):
    handler = sequence(served(503, b"upstream down"))
    client = make_client(handler, retries=2)
    try:
        assert await call_legacy(client, method) is LEGACY_FAILURE[method]
    finally:
        await client._http.aclose()
    assert handler.attempts["n"] == 2


@pytest.mark.parametrize("method", ["commit", "reveal", "feedback"])
async def test_legacy_callers_stay_failure_shaped_when_a_response_precedes_transport_loss(
    method,
):
    """Retaining the earlier 503 for lease() must not hand it to these callers.

    They treat any response as an answer to parse, so a retained failure would
    surface as "invalid commit receipt" instead of a failed POST.
    """
    handler = sequence(served(503, b"upstream down"), REFUSED)
    client = make_client(handler, retries=2)
    try:
        assert await call_legacy(client, method) is LEGACY_FAILURE[method]
    finally:
        await client._http.aclose()
    assert handler.attempts["n"] == 2


@pytest.mark.parametrize(
    ("method", "success_body"),
    [
        (
            "commit",
            b'{"accepted":true,"commit_token":"token","num_submissions":0}',
        ),
        (
            "reveal",
            b'{"challenge_id":"challenge-1","language":"python","tests":[]}',
        ),
        ("feedback", b'{"accepted":true}'),
    ],
)
async def test_legacy_callers_retry_an_oversized_retryable_error(
    method, success_body
):
    """A large 503 body must not shrink the existing retry budget."""
    handler = sequence(served(503, b"x" * 5000), served(200, success_body))
    client = make_client(handler, retries=2, max_response_bytes=1000)
    try:
        result = await call_legacy(client, method)
    finally:
        await client._http.aclose()

    assert handler.attempts["n"] == 2
    assert result is not LEGACY_FAILURE[method]


# --------------------------------------------------------------------------- #
# Canonical V2 lease request: no capability negotiation.
# --------------------------------------------------------------------------- #
def _capturing_client(captured: dict, max_response_bytes: int = 2 * 1024 * 1024):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        body = captured.get("response_body") or json.dumps(
            {**_CHALLENGE, "language": "python"}
        ).encode()
        return httpx.Response(200, content=body)

    return ProblemServerClient(
        url="https://problems.invalid",
        wallet="validator-hotkey",
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retries=1,
        max_response_bytes=max_response_bytes,
    )


def test_client_has_no_capability_parameter():
    """The server's canonical V2 removed capability negotiation; a client
    still accepting the parameter would tempt a call site into serializing
    a field the server rejects wholesale."""
    with pytest.raises(TypeError):
        ProblemServerClient(
            url="https://problems.invalid",
            wallet="validator-hotkey",
            http=None,
            supported_languages=["python"],
        )


async def test_lease_body_is_exactly_request_id():
    captured: dict = {}
    client = _capturing_client(captured)

    await client.lease()

    assert set(captured["body"]) == {"request_id"}


async def test_large_python_lease_is_accepted_under_the_single_v2_cap():
    """V1 held python responses to 512,000 bytes to protect deployed old
    validators; the cutover retires those readers, and the server's V2
    contract states one >=2 MiB read limit for BOTH languages on BOTH
    wires. A ~560 KB python lease — max-length statement plus example
    bulk, every FIELD within its unchanged V1 limit — must now lease,
    not report MALFORMED. (The statement field limit itself stays at
    500,000: see test_rust_wire_compat's field-limit pin.)"""
    captured: dict = {}
    big_statement = "x" * 500_000
    examples = [
        {"args": ["y" * 2000], "kwargs": {}, "expected": "z"} for _ in range(30)
    ]
    body = json.dumps(
        {
            **_CHALLENGE,
            "language": "python",
            "statement": big_statement,
            "public_examples": examples,
        }
    ).encode()
    assert len(body) > 512_000, "fixture must exceed the retired V1 cap"
    captured["response_body"] = body
    client = _capturing_client(captured)

    outcome = await client.lease()

    assert outcome.category is LeaseCategory.LEASED
    assert outcome.challenge is not None
    assert outcome.challenge.statement == big_statement
