"""Validator-side signed HTTP transport coverage against a protocol stub."""

from __future__ import annotations

import asyncio
import json

import pytest

from rlvr.config import get_settings
from rlvr.neurons.decentralized import _dispatch_committed
from rlvr.neurons.live import LiveSolverClient
from rlvr.problemserver.api import PublicChallenge
from rlvr.protocol import SolutionPayload, TaskRequest, crypto_available, sign_message
from rlvr.types import Problem, TestCase as Case

httpx = pytest.importorskip("httpx")
fastapi = pytest.importorskip("fastapi")
pytestmark = pytest.mark.skipif(
    not crypto_available(), reason="no sr25519 crypto stack installed"
)


class WalletLike:
    def __init__(self, hotkey):
        self.hotkey = hotkey


def setup_transport(
    *,
    oversized: bool = False,
    dripping: bool = False,
    dripping_status: int = 200,
):
    from bittensor_wallet import Keypair

    validator = Keypair.create_from_uri("//Alice")
    responder = Keypair.create_from_uri("//Dave")
    app = fastapi.FastAPI()
    captured = {}

    @app.post("/solve")
    async def solve(request: fastapi.Request):
        request_body = await request.body()
        captured["request"] = json.loads(request_body)
        task = TaskRequest.model_validate_json(request_body)
        if dripping:
            from starlette.responses import StreamingResponse

            async def never_finishes():
                try:
                    yield b"{"
                    await asyncio.Event().wait()
                finally:
                    captured["stream_cancelled"] = True

            return StreamingResponse(
                never_finishes(),
                media_type="application/json",
                status_code=dripping_status,
            )
        code = "x" * 2048 if oversized else "def add(a, b):\n    return a + b"
        body = json.dumps(
            SolutionPayload(
                problem_id=task.problem_id,
                code=code,
                raw_response=code,
            ).model_dump(),
            separators=(",", ":"),
        ).encode()
        response = fastapi.Response(body, media_type="application/json")
        response.headers.update(
            sign_message(
                WalletLike(responder),
                body,
                signed_for=request.headers["Epistula-Signed-By"],
            )
        )
        return response

    settings = get_settings()
    if oversized:
        settings.miner_max_response_bytes = 128
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://responder"
    )
    client = LiveSolverClient(
        uid=0,
        hotkey=responder.ss58_address,
        url="http://responder",
        wallet=WalletLike(validator),
        settings=settings,
        http=http,
    )
    problem = Problem(
        problem_id="p1",
        language="python",
        statement="Return a + b.",
        entrypoint="add",
        tests=[Case(args=[20, 22], expected=42)],
        public_examples=[Case(args=[1, 2], expected=3)],
    )
    return http, client, problem, captured


async def test_validator_transport_signed_roundtrip():
    http, client, problem, captured = setup_transport()
    try:
        response = await client.solve(problem, problem.statement)
    finally:
        await http.aclose()

    assert response.code == "def add(a, b):\n    return a + b"
    assert "tests" not in captured["request"]
    assert captured["request"]["public_examples"] == [
        {"args": [1, 2], "kwargs": {}, "expected": 3}
    ]


async def test_validator_transport_rejects_oversized_response():
    http, client, problem, _captured = setup_transport(oversized=True)
    try:
        response = await client.solve(problem, problem.statement)
    finally:
        await http.aclose()

    assert response.code == ""
    assert "exceeds byte limit" in response.raw_response


@pytest.mark.parametrize("status_code", [200, 500])
async def test_dispatch_total_deadline_stops_drip_fed_http_response(
    monkeypatch, status_code
):
    import rlvr.neurons.decentralized as decentralized

    http, client, problem, captured = setup_transport(
        dripping=True,
        dripping_status=status_code,
    )
    monkeypatch.setattr(decentralized, "_MINER_RESPONSE_GRACE_S", 0.0)
    public = PublicChallenge(
        challenge_id="drip-challenge",
        problem_id=problem.problem_id,
        language=problem.language,
        statement=problem.statement,
        entrypoint=problem.entrypoint,
        public_examples=problem.public_examples,
        deadline_s=0.02,
    )
    try:
        submissions, _captured = await asyncio.wait_for(
            _dispatch_committed(public, [client], concurrency=1),
            timeout=1,
        )
    finally:
        await http.aclose()

    assert captured["stream_cancelled"]
    assert submissions[0].error == "<miner total response deadline exceeded>"
