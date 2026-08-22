"""OpenAI-compatible provider client and signed miner integration coverage."""

from __future__ import annotations

import json

import httpx
import pytest

import rlvr.protocol as protocol_module
from rlvr.config import Settings
from rlvr.neurons.live import LiveSolverClient
from rlvr.neurons.openai_miner import (
    OpenAICompatibleClient,
    OpenAICompatibleMiner,
    OpenAICompatibleSettings,
    build_openai_miner_app,
)
from rlvr.types import Problem, TestCase as Case

fastapi = pytest.importorskip("fastapi")


@pytest.fixture(autouse=True)
def force_chain_free_signatures(monkeypatch):
    monkeypatch.setattr(protocol_module, "_HAVE_CRYPTO", False)
    monkeypatch.setattr(protocol_module, "_Keypair", None)


class WalletLike:
    class Hotkey:
        def __init__(self, address: str):
            self.ss58_address = address

    def __init__(self, address: str):
        self.hotkey = self.Hotkey(address)


class FakeMetagraph:
    def __init__(self, validator: str):
        self.hotkeys = [validator]
        self.S = [100.0]
        self.validator_permit = [True]


def miner_settings(**updates) -> OpenAICompatibleSettings:
    values = {
        "openai_api_key": "test-key",
        "openai_model": "provider/code-model",
        "miner_self_test": False,
    }
    values.update(updates)
    return OpenAICompatibleSettings(
        _env_file=None,
        **values,
    )


async def test_chat_completion_contract_and_auth_header():
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "```python\ndef add(a, b):\n    return a + b\n```",
                        }
                    }
                ]
            },
        )

    settings = miner_settings(openai_base_url="https://provider.example/v1")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(settings, http=http)
    content = await client.complete(
        [{"role": "user", "content": "Implement add."}],
        timeout_s=30.0,
    )
    await http.aclose()

    assert content.startswith("```python")
    assert captured["url"] == "https://provider.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["body"] == {
        "model": "provider/code-model",
        "messages": [{"role": "user", "content": "Implement add."}],
        "stream": False,
        "max_tokens": 16_384,
    }


@pytest.mark.parametrize(
    "provider_content, expected",
    [
        (
            "<think>private chain of thought</think>\n"
            "```python\ndef f():\n    return '<think>code data</think>'\n```\n"
            "The implementation is above.",
            "```python\ndef f():\n    return '<think>code data</think>'\n```",
        ),
        (
            "<analysis>private analysis</analysis>\ndef f():\n    return 1",
            "def f():\n    return 1",
        ),
        (
            "Planning artifact:\n```json\n{\"approach\": \"discard\"}\n```\n"
            "```rust\nfn main() {}\n```\n<reasoning>also private</reasoning>",
            "```rust\nfn main() {}\n```",
        ),
        (
            "<think>private</think>\n"
            "```self-tests\n"
            '{"tests":[{"args":[1,2],"kwargs":{},"expected":3}]}\n'
            "```\n"
            "```python\ndef add(a, b):\n    return a + b\n```",
            "```self-tests\n"
            '{"tests":[{"args":[1,2],"kwargs":{},"expected":3}]}\n'
            "```\n\n"
            "```python\ndef add(a, b):\n    return a + b\n```",
        ),
        (
            "```json\n"
            '{"tests":[{"args":[1,2],"kwargs":{},"expected":3}]}\n'
            "```\n"
            "```python\ndef add(a, b):\n    return a + b\n```",
            "```json\n"
            '{"tests":[{"args":[1,2],"kwargs":{},"expected":3}]}\n'
            "```\n\n"
            "```python\ndef add(a, b):\n    return a + b\n```",
        ),
    ],
)
async def test_reasoning_text_is_suppressed_from_completion(
    provider_content,
    expected,
):
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": provider_content,
                            "reasoning_content": "separate private reasoning",
                        }
                    }
                ]
            },
        )

    settings = miner_settings()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(settings, http=http)
    try:
        content = await client.complete([], timeout_s=30.0)
    finally:
        await http.aclose()

    assert content == expected
    assert "private" not in content


async def test_reasoning_without_a_solution_is_a_provider_failure():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "<think>reasoning only</think>"}}
                ]
            },
        )

    settings = miner_settings()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(settings, http=http)
    try:
        with pytest.raises(RuntimeError, match="invalid response"):
            await client.complete([], timeout_s=30.0)
    finally:
        await http.aclose()


async def test_optional_compatible_fields_and_provider_extensions():
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "fn main() {}"}}]},
        )

    settings = miner_settings(
        openai_base_url="https://provider.example/v1/chat/completions",
        openai_max_tokens=4096,
        openai_max_tokens_param="max_completion_tokens",
        openai_temperature=0.3,
        openai_reasoning_effort="high",
        openai_extra_headers_json='{"X-Provider-Feature":"enabled"}',
        openai_extra_body_json='{"top_p":0.95}',
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(settings, http=http)
    await client.complete(
        [{"role": "user", "content": "Implement it."}],
        timeout_s=30.0,
    )
    await http.aclose()

    assert captured["headers"]["x-provider-feature"] == "enabled"
    assert captured["body"]["max_completion_tokens"] == 4096
    assert "max_tokens" not in captured["body"]
    assert captured["body"]["temperature"] == 0.3
    assert captured["body"]["reasoning_effort"] == "high"
    assert captured["body"]["top_p"] == 0.95


async def test_insecure_http_and_missing_api_key_fail_closed():
    settings = miner_settings(openai_base_url="http://provider.example/v1")
    client = OpenAICompatibleClient(settings)
    try:
        with pytest.raises(RuntimeError, match="must use HTTPS"):
            _ = client.completion_url
    finally:
        await client.aclose()

    no_key = miner_settings(openai_api_key="")
    client = OpenAICompatibleClient(no_key)
    try:
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            client._headers()
    finally:
        await client.aclose()


async def test_explicit_local_server_can_run_without_authentication():
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "def f(): return 1"}}]},
        )

    settings = miner_settings(
        openai_api_key="",
        openai_require_api_key=False,
        openai_base_url="http://127.0.0.1:8000/v1",
        openai_allow_insecure_http=True,
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(settings, http=http)
    content = await client.complete([], timeout_s=30.0)
    await http.aclose()

    assert content == "def f(): return 1"
    assert captured["authorization"] is None


async def test_fallback_runs_only_after_primary_retries_are_exhausted(monkeypatch):
    calls = []

    async def no_retry_delay(_seconds: float) -> None:
        return None

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(
            (
                request.url.host,
                body["model"],
                request.headers.get("authorization"),
            )
        )
        if request.url.host == "primary.example":
            return httpx.Response(503, json={"error": "primary unavailable"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "fallback code"}}]},
        )

    monkeypatch.setattr(
        "rlvr.neurons.openai_miner.asyncio.sleep",
        no_retry_delay,
    )
    settings = miner_settings(
        openai_base_url="https://primary.example/v1",
        openai_model="primary/model",
        openai_max_retries=2,
        openai_fallback_base_url="https://fallback.example/v1",
        openai_fallback_model="fallback/model",
        openai_fallback_api_key="fallback-key",
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(settings, http=http)
    try:
        content = await client.complete([], timeout_s=30.0)
    finally:
        await http.aclose()

    assert content == "fallback code"
    assert calls == [
        ("primary.example", "primary/model", "Bearer test-key"),
        ("primary.example", "primary/model", "Bearer test-key"),
        ("primary.example", "primary/model", "Bearer test-key"),
        ("fallback.example", "fallback/model", "Bearer fallback-key"),
    ]


async def test_successful_primary_never_calls_configured_fallback():
    hosts = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "primary code"}}]},
        )

    settings = miner_settings(
        openai_base_url="https://primary.example/v1",
        openai_fallback_base_url="https://fallback.example/v1",
        openai_fallback_model="fallback/model",
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(settings, http=http)
    try:
        content = await client.complete([], timeout_s=30.0)
    finally:
        await http.aclose()

    assert content == "primary code"
    assert hosts == ["primary.example"]


async def test_failure_is_terminal_after_fallback_retries_are_exhausted(
    monkeypatch,
):
    hosts = []

    async def no_retry_delay(_seconds: float) -> None:
        return None

    async def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(503, json={"error": "unavailable"})

    monkeypatch.setattr(
        "rlvr.neurons.openai_miner.asyncio.sleep",
        no_retry_delay,
    )
    settings = miner_settings(
        openai_base_url="https://primary.example/v1",
        openai_max_retries=1,
        openai_fallback_base_url="https://fallback.example/v1",
        openai_fallback_model="fallback/model",
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(settings, http=http)
    try:
        with pytest.raises(RuntimeError, match="primary and fallback"):
            await client.complete([], timeout_s=30.0)
    finally:
        await http.aclose()

    assert hosts == [
        "primary.example",
        "primary.example",
        "fallback.example",
        "fallback.example",
    ]


@pytest.mark.parametrize(
    "updates",
    [
        {"openai_fallback_base_url": "https://fallback.example/v1"},
        {"openai_fallback_model": "fallback/model"},
    ],
)
def test_fallback_url_and_model_must_be_configured_together(updates):
    with pytest.raises(ValueError, match="must be configured together"):
        miner_settings(**updates)


@pytest.mark.parametrize(
    "setting, value, message",
    [
        ("openai_extra_headers_json", "[]", "JSON object"),
        (
            "openai_extra_headers_json",
            '{"Authorization":"replacement"}',
            "cannot override",
        ),
        ("openai_extra_body_json", '{"model":"replacement"}', "protected fields"),
    ],
)
async def test_invalid_provider_extensions_are_rejected(setting, value, message):
    settings = miner_settings(**{setting: value})
    client = OpenAICompatibleClient(settings)
    try:
        with pytest.raises(RuntimeError, match=message):
            await client.complete([], timeout_s=30.0)
    finally:
        await client.aclose()


async def test_signed_validator_to_openai_miner_roundtrip_and_health():
    validator_address = "validator-test-address"
    miner_address = "miner-test-address"

    async def provider_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "```python\ndef add(a, b):\n    return a + b\n```"
                            )
                        }
                    }
                ]
            },
        )

    settings = miner_settings()
    provider_http = httpx.AsyncClient(
        transport=httpx.MockTransport(provider_handler)
    )
    miner = OpenAICompatibleMiner(
        settings,
        OpenAICompatibleClient(settings, http=provider_http),
        wallet=WalletLike(miner_address),
        metagraph=FakeMetagraph(validator_address),
    )
    app = build_openai_miner_app(miner)
    miner_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://miner",
    )
    validator_client = LiveSolverClient(
        uid=0,
        hotkey=miner_address,
        url="http://miner",
        wallet=WalletLike(validator_address),
        settings=Settings(_env_file=None),
        http=miner_http,
    )
    problem = Problem(
        problem_id="canonical-problem-id",
        language="python",
        statement="Add two integers.",
        entrypoint="add",
        tests=[],
        public_examples=[Case(args=[1, 2], expected=3)],
    )

    try:
        solution = await validator_client.solve(problem, problem.statement)
        health = await miner_http.get("/health")
    finally:
        await miner_http.aclose()
        await provider_http.aclose()

    assert solution.code == "def add(a, b):\n    return a + b"
    assert health.json() == {
        "status": "ok",
        "model": "provider/code-model",
    }
