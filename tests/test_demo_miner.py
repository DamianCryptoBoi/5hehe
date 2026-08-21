"""GLM demo miner protocol and provider-client coverage."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import rlvr.protocol as protocol_module
from rlvr.neurons.demo_miner import (
    DemoMiner,
    DemoMinerSettings,
    GLM52Client,
    build_demo_miner_app,
    build_model_messages,
    build_verification_messages,
    extract_python,
)
from rlvr.neurons.live import LiveSolverClient
from rlvr.protocol import TaskRequest
from rlvr.types import Problem, TestCase as Case

fastapi = pytest.importorskip("fastapi")


@pytest.fixture(autouse=True)
def force_chain_free_signatures(monkeypatch):
    """Keep protocol tests hermetic when a chain-signing package is installed."""

    monkeypatch.setattr(protocol_module, "_HAVE_CRYPTO", False)
    monkeypatch.setattr(protocol_module, "_Keypair", None)


class WalletLike:
    class Hotkey:
        def __init__(self, address: str):
            self.ss58_address = address

    def __init__(self, address: str):
        self.hotkey = self.Hotkey(address)


class FakeMetagraph:
    def __init__(self, validator: str, *, permit: bool = True):
        self.hotkeys = [validator]
        self.S = [100.0]
        self.validator_permit = [permit]


class FakeGLM:
    def __init__(
        self, content: str = "```python\ndef add(a, b):\n    return a + b\n```"
    ):
        self.content = content
        self.messages = None

    async def complete(self, messages, *, timeout_s):
        self.messages = messages
        return self.content

    async def aclose(self):
        return None


class FailingGLM(FakeGLM):
    async def complete(self, messages, *, timeout_s):
        request = httpx.Request("POST", "https://provider.example/v4")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError(
            "quota unavailable",
            request=request,
            response=response,
        )


class SequencedGLM(FakeGLM):
    def __init__(self, *outputs: str | Exception):
        self.outputs = list(outputs)
        self.calls: list[tuple[list[dict[str, str]], float]] = []

    async def complete(self, messages, *, timeout_s):
        self.calls.append((messages, timeout_s))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class SlowReviewGLM(SequencedGLM):
    async def complete(self, messages, *, timeout_s):
        self.calls.append((messages, timeout_s))
        if len(self.calls) == 1:
            return self.outputs.pop(0)
        await asyncio.sleep(timeout_s * 10.0)
        return self.outputs.pop(0)


def demo_settings(**updates) -> DemoMinerSettings:
    return DemoMinerSettings(
        _env_file=None,
        glm_api_key="test-key",
        miner_require_validator_permit=True,
        **updates,
    )


def test_extract_python_prefers_python_fence():
    reply = "Text\n```json\n{}\n```\n```python\ndef f():\n    return 1\n```"
    assert extract_python(reply) == "def f():\n    return 1"
    assert extract_python("def f():\n    return 1") == "def f():\n    return 1"


def test_model_prompt_contains_only_public_task_data():
    request = TaskRequest(
        problem_id="request-id",
        language="python",
        statement="Add two integers.",
        entrypoint="add",
        public_examples=[Case(args=[20, 22], expected=42)],
    )
    messages = build_model_messages(request)
    rendered = json.dumps(messages)

    assert "Add two integers." in rendered
    assert "entrypoint: add" in rendered
    assert "42" in rendered
    assert "hidden" not in rendered.lower()
    assert "request-id" not in rendered
    assert "<problem_statement>" in rendered
    assert "<public_examples_json>" in rendered


def test_verification_prompt_reuses_task_and_keeps_candidate_separate():
    request = TaskRequest(
        problem_id="request-id",
        language="python",
        statement="Add two integers, including negative values.",
        entrypoint="add",
        public_examples=[Case(args=[-2, 5], expected=3)],
    )

    messages = build_verification_messages(
        request,
        "def add(a, b):\n    return abs(a) + abs(b)",
    )
    rendered = json.dumps(messages)

    assert messages[-2]["role"] == "assistant"
    assert "return abs(a) + abs(b)" in messages[-2]["content"]
    assert messages[-1]["role"] == "user"
    assert "independent verification" in messages[-1]["content"]
    assert "Add two integers, including negative values." in rendered
    assert '"expected": 3' in messages[1]["content"]
    assert "request-id" not in rendered


async def test_self_verification_corrects_the_initial_draft():
    corrected = "```python\ndef add(a, b):\n    return a + b\n```"
    client = SequencedGLM(
        "```python\ndef add(a, b):\n    return abs(a) + abs(b)\n```",
        corrected,
    )
    miner = DemoMiner(demo_settings(), client)
    request = TaskRequest(
        problem_id="p",
        language="python",
        statement="Return the sum of two integers.",
        entrypoint="add",
        public_examples=[Case(args=[-2, 5], expected=3)],
    )

    payload = await miner.solve(request, timeout_s=30.0)

    assert payload.code == "def add(a, b):\n    return a + b"
    assert payload.raw_response == corrected
    assert len(client.calls) == 2
    assert client.calls[0][1] == pytest.approx(15.0)
    assert "Return the sum of two integers." in client.calls[1][0][1]["content"]
    assert "return abs(a) + abs(b)" in client.calls[1][0][-2]["content"]


@pytest.mark.parametrize(
    "review",
    [
        "The candidate looks correct.",
        "```python\ndef different_name(a, b):\n    return a + b\n```",
        TimeoutError("review timed out"),
    ],
)
async def test_failed_or_invalid_review_preserves_the_initial_draft(review):
    draft = "```python\ndef add(a, b):\n    return a + b\n```"
    client = SequencedGLM(draft, review)
    miner = DemoMiner(demo_settings(), client)
    request = TaskRequest(
        problem_id="p",
        language="python",
        statement="Return the sum.",
        entrypoint="add",
    )

    payload = await miner.solve(request, timeout_s=30.0)

    assert payload.code == "def add(a, b):\n    return a + b"
    assert payload.raw_response == draft
    assert len(client.calls) == 2


async def test_review_timeout_returns_the_unreviewed_draft_before_outer_deadline():
    draft = "```python\ndef add(a, b):\n    return a + b\n```"
    client = SlowReviewGLM(draft, "```python\ndef add(a, b):\n    return a - b\n```")
    miner = DemoMiner(
        demo_settings(miner_self_verify_reserve_s=0.1),
        client,
    )
    request = TaskRequest(
        problem_id="p",
        language="python",
        statement="Return the sum.",
        entrypoint="add",
    )

    started = asyncio.get_running_loop().time()
    payload = await miner.solve(request, timeout_s=0.2)
    elapsed = asyncio.get_running_loop().time() - started

    assert payload.code == "def add(a, b):\n    return a + b"
    assert payload.raw_response == draft
    assert len(client.calls) == 2
    assert client.calls[1][1] < 0.2
    assert elapsed < 0.2


async def test_self_verification_can_be_disabled_for_one_pass_operation():
    draft = "```python\ndef add(a, b):\n    return a + b\n```"
    client = SequencedGLM(draft)
    miner = DemoMiner(demo_settings(miner_self_verify=False), client)
    request = TaskRequest(
        problem_id="p",
        language="python",
        statement="Return the sum.",
        entrypoint="add",
    )

    payload = await miner.solve(request, timeout_s=30.0)

    assert payload.code == "def add(a, b):\n    return a + b"
    assert len(client.calls) == 1
    assert client.calls[0][1] == pytest.approx(30.0)


async def test_glm_client_uses_configured_chat_completion_contract():
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "```python\ndef f(): return 1\n```"}}
                ]
            },
        )

    settings = demo_settings(
        glm_base_url="https://provider.example/v4",
        glm_model="glm-5.2",
        glm_reasoning_effort="max",
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = GLM52Client(settings, http=http)
    content = await client.complete(
        [{"role": "user", "content": "Implement f."}],
        timeout_s=30.0,
    )
    await http.aclose()

    assert content.startswith("```python")
    assert captured["url"] == "https://provider.example/v4/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == "glm-5.2"
    assert captured["body"]["thinking"] == {"type": "enabled"}
    assert captured["body"]["reasoning_effort"] == "max"


async def test_glm_client_requires_https():
    settings = demo_settings(glm_base_url="http://provider.example/v4")
    client = GLM52Client(settings)
    try:
        with pytest.raises(RuntimeError, match="HTTPS"):
            _ = client.completion_url
    finally:
        await client.aclose()


def setup_roundtrip(*, permit: bool = True, request_limit: int = 1_000_000):
    validator_address = "validator-test-address"
    miner_address = "miner-test-address"
    settings = demo_settings(miner_max_request_bytes=request_limit)
    fake_glm = FakeGLM()
    miner = DemoMiner(
        settings,
        fake_glm,  # type: ignore[arg-type]
        wallet=WalletLike(miner_address),
        metagraph=FakeMetagraph(validator_address, permit=permit),
    )
    app = build_demo_miner_app(miner)
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://miner",
    )

    from rlvr.config import Settings

    validator_settings = Settings(_env_file=None)
    client = LiveSolverClient(
        uid=0,
        hotkey=miner_address,
        url="http://miner",
        wallet=WalletLike(validator_address),
        settings=validator_settings,
        http=http,
    )
    problem = Problem(
        problem_id="canonical-problem-id",
        language="python",
        statement="Add two integers.",
        entrypoint="add",
        tests=[],
        public_examples=[Case(args=[1, 2], expected=3)],
    )
    return http, client, problem, fake_glm


async def test_signed_validator_to_demo_miner_roundtrip():
    http, client, problem, fake_glm = setup_roundtrip()
    try:
        solution = await client.solve(problem, problem.statement)
    finally:
        await http.aclose()

    assert solution.code == "def add(a, b):\n    return a + b"
    assert "<problem_statement>\nAdd two integers." in fake_glm.messages[1][
        "content"
    ]


async def test_provider_http_failure_returns_signed_zero_code(capsys):
    validator_address = "validator-test-address"
    miner_address = "miner-test-address"
    miner = DemoMiner(
        demo_settings(),
        FailingGLM(),  # type: ignore[arg-type]
        wallet=WalletLike(miner_address),
        metagraph=FakeMetagraph(validator_address),
    )
    request = TaskRequest(
        problem_id="request-id",
        language="python",
        statement="Implement f.",
        entrypoint="f",
    )

    payload = await miner.solve(request, timeout_s=30.0)

    assert payload.code == ""
    assert payload.raw_response == "<model request failed>"
    assert "HTTP 429" in capsys.readouterr().out


async def test_demo_miner_rejects_unpermitted_validator():
    http, client, problem, _fake_glm = setup_roundtrip(permit=False)
    try:
        solution = await client.solve(problem, problem.statement)
    finally:
        await http.aclose()

    assert solution.code == ""
    assert "HTTP 403" in solution.raw_response


async def test_demo_miner_rejects_oversized_request():
    http, client, problem, _fake_glm = setup_roundtrip(request_limit=16)
    try:
        solution = await client.solve(problem, problem.statement)
    finally:
        await http.aclose()

    assert solution.code == ""
    assert "HTTP 413" in solution.raw_response


# --------------------------------------------------------------------------- #
# Rust prompting and extraction (docs/RUST_CHALLENGES.md)
# --------------------------------------------------------------------------- #
def test_rust_task_prompts_for_a_complete_program():
    from rlvr.neurons.demo_miner import RUST_SYSTEM_PROMPT

    request = TaskRequest(
        problem_id="p",
        statement="Sum the integers on stdin.",
        entrypoint="main",
        language="rust",
    )

    messages = build_model_messages(request)

    assert messages[0]["content"] == RUST_SYSTEM_PROMPT
    assert "Rust" in RUST_SYSTEM_PROMPT
    assert "standard input" in RUST_SYSTEM_PROMPT
    assert "held-out-test" in RUST_SYSTEM_PROMPT
    assert "boundary cases" in RUST_SYSTEM_PROMPT
    assert "exactly one complete fenced `rust`" in RUST_SYSTEM_PROMPT
    # Program mode names main as metadata without Python-specific function prose.
    assert "Required function name" not in messages[1]["content"]
    assert "entrypoint: main" in messages[1]["content"]


def test_python_prompt_is_unchanged_when_language_is_omitted():
    from rlvr.neurons.demo_miner import PYTHON_SYSTEM_PROMPT

    request = TaskRequest(
        problem_id="p",
        statement="Return a + b.",
        entrypoint="add",
        language="python",
    )

    messages = build_model_messages(request)

    assert messages[0]["content"] == PYTHON_SYSTEM_PROMPT
    assert "entrypoint: add" in messages[1]["content"]
    assert "held-out-test" in PYTHON_SYSTEM_PROMPT
    assert "boundary cases" in PYTHON_SYSTEM_PROMPT
    assert "deterministic" in PYTHON_SYSTEM_PROMPT
    assert "exactly one complete fenced `python`" in PYTHON_SYSTEM_PROMPT
    assert "<public_examples_json>[]</public_examples_json>" in messages[1][
        "content"
    ]


def test_rust_fence_is_preferred_over_other_fences():
    from rlvr.neurons.demo_miner import extract_rust

    both = "```python\nx = 1\n```\n\n```rust\nfn main() {}\n```"
    assert extract_rust(both) == "fn main() {}"
    assert extract_rust("```\nfn main() {}\n```") == "fn main() {}"
    assert extract_python("```python\nx = 1\n```") == "x = 1"
