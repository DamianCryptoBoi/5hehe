"""Reference miner backed by the GLM-5.2 chat-completions API.

The demo is a self-contained example of the validator/miner wire protocol. It
verifies and authorizes signed task requests, asks GLM-5.2 for a Python
solution, and signs the exact response bytes with the miner hotkey.
"""

import ast
import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Mapping, Optional

import httpx
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..protocol import (
    NonceCache,
    SolutionPayload,
    TaskRequest,
    sign_message,
    verify_signature,
)

PYTHON_SYSTEM_PROMPT = (
    "You are an expert competitive-programming solver. Produce a correct and "
    "efficient solution for a held-out-test coding challenge.\n\n"
    "Before answering, reason privately through the exact specification, input "
    "and output types, algorithm, complexity, and likely boundary cases. Check "
    "the proposed solution against every public example. Treat those examples "
    "as illustrations, not as the complete specification, and never hard-code "
    "their answers.\n\n"
    "Requirements:\n"
    "- Target Python 3.12 and define the requested callable entrypoint; helper "
    "functions and classes are allowed.\n"
    "- Return the specified JSON-serializable value. Do not add stdin/stdout "
    "handling, example calls, tests, or debug output.\n"
    "- Use only the Python standard library. The solution must be deterministic "
    "and must not use network access, files, environment variables, randomness, "
    "or persistent state.\n"
    "- Handle all applicable degenerate and boundary cases, including empty or "
    "minimal inputs, duplicates, ordering and tie rules, signs, indexing, and "
    "numeric precision. Do not invent behavior that contradicts the statement.\n"
    "- Avoid unnecessary superlinear work and fit within 5 seconds and 256 MiB.\n\n"
    "Output exactly one complete fenced `python` code block and no other text."
)
RUST_SYSTEM_PROMPT = (
    "You are an expert competitive-programming solver. Produce a correct and "
    "efficient solution for a held-out-test coding challenge.\n\n"
    "Before answering, reason privately through the exact specification, input "
    "grammar, output grammar, algorithm, complexity, and likely boundary cases. "
    "Check the proposed solution against every public example. Treat those "
    "examples as illustrations, not as the complete specification, and never "
    "hard-code their answers.\n\n"
    "Requirements:\n"
    "- Target Rust 1.89, edition 2021, and return one complete source unit with "
    "`fn main()`.\n"
    "- Read the complete test case from standard input and write only the "
    "requested answer to standard output. Output tokens must use the exact "
    "specified spelling and order; only ASCII whitespace differences are ignored.\n"
    "- Use only the Rust standard library: no Cargo manifest, external crates, "
    "nightly features, files, environment variables, network access, randomness, "
    "or persistent state.\n"
    "- Handle all applicable degenerate and boundary cases, including empty or "
    "minimal inputs, duplicates, ordering and tie rules, signs, indexing, and "
    "integer bounds. Do not invent behavior that contradicts the statement.\n"
    "- Avoid unnecessary superlinear work and fit within 5 seconds per case.\n\n"
    "Output exactly one complete fenced `rust` code block and no other text."
)

SELF_VERIFICATION_PROMPT = (
    "Perform an independent verification of the candidate solution immediately "
    "above against the original problem statement and public examples. Re-derive "
    "the required behavior from the statement instead of trusting the candidate's "
    "approach. Reason privately and:\n"
    "- simulate the candidate on every public example;\n"
    "- construct likely boundary and adversarial cases from the stated contract;\n"
    "- check return/output shape, ordering and tie rules, numeric bounds, and "
    "complexity; and\n"
    "- repair every defect you find without weakening or inventing requirements.\n\n"
    "Return the complete replacement solution even when no change is needed. Do "
    "not describe the review or output tests. Follow the original system contract "
    "and output exactly one fenced source block."
)

_PYTHON_FENCE_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL
)
_RUST_FENCE_RE = re.compile(r"```rust\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
_REVIEW_RESPONSE_MARGIN_S = 0.5


class DemoMinerSettings(BaseSettings):
    """Environment configuration for the reference miner."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    glm_api_key: str = ""
    glm_base_url: str = "https://api.z.ai/api/paas/v4"
    glm_model: str = "glm-5.2"
    glm_max_tokens: int = Field(default=16_384, ge=1, le=131_072)
    glm_temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    glm_thinking: bool = True
    glm_reasoning_effort: str = Field(
        default="high",
        pattern="^(|none|minimal|low|medium|high|max)$",
    )
    glm_request_timeout_s: float = Field(default=280.0, gt=0.0, le=3600.0)
    glm_max_retries: int = Field(default=2, ge=0, le=10)

    # Reserve part of the signed request deadline for a second, independent
    # prompt-aware review. A failed review never discards the first draft.
    miner_self_verify: bool = True
    miner_self_verify_reserve_s: float = Field(default=90.0, ge=0.0, le=1800.0)

    netuid: int = Field(default=0, ge=0)
    subtensor_network: str = "test"
    subtensor_chain_endpoint: str = ""
    wallet_name: str = "default"
    wallet_hotkey: str = "default"

    axon_host: str = "0.0.0.0"
    axon_port: int = Field(default=8091, ge=1, le=65_535)
    axon_external_ip: str = ""
    miner_max_concurrent_requests: int = Field(default=4, ge=1, le=256)
    miner_max_request_bytes: int = Field(default=1_000_000, ge=1, le=10_000_000)
    miner_metagraph_sync_s: float = Field(default=300.0, gt=0.0)
    miner_min_stake: float = Field(default=0.0, ge=0.0)
    miner_require_validator_permit: bool = True


def extract_python(text: str) -> str:
    """Extract the first Python fence, then any fence, or use the whole reply."""

    match = _PYTHON_FENCE_RE.search(text) or _ANY_FENCE_RE.search(text)
    return (match.group(1) if match else text).strip()


def extract_rust(text: str) -> str:
    """Extract the first Rust fence, then any fence, or use the whole reply."""

    match = _RUST_FENCE_RE.search(text) or _ANY_FENCE_RE.search(text)
    return (match.group(1) if match else text).strip()


def build_model_messages(request: TaskRequest) -> list[dict[str, str]]:
    """Render a task using only fields in the request wire model."""

    prompt = (
        "Solve the following challenge according to the system contract.\n\n"
        "<problem_statement>\n"
        f"{request.statement.strip()}\n"
        "</problem_statement>\n\n"
        "<task_metadata>\n"
        f"language: {request.language}\n"
        f"entrypoint: {request.entrypoint}\n"
        "</task_metadata>"
    )
    if request.public_examples:
        examples = [case.model_dump(mode="json") for case in request.public_examples]
        prompt += "\n\n<public_examples_json>\n" + json.dumps(
            examples,
            ensure_ascii=False,
            indent=2,
        )
        prompt += "\n</public_examples_json>"
    else:
        prompt += "\n\n<public_examples_json>[]</public_examples_json>"
    return [
        {
            "role": "system",
            "content": (
                RUST_SYSTEM_PROMPT
                if request.language == "rust"
                else PYTHON_SYSTEM_PROMPT
            ),
        },
        {"role": "user", "content": prompt},
    ]


def build_verification_messages(
    request: TaskRequest, candidate_code: str
) -> list[dict[str, str]]:
    """Ask for a fresh audit while retaining the exact public task context."""

    language = "rust" if request.language == "rust" else "python"
    return [
        *build_model_messages(request),
        {
            "role": "assistant",
            "content": f"```{language}\n{candidate_code.strip()}\n```",
        },
        {"role": "user", "content": SELF_VERIFICATION_PROMPT},
    ]


def _extract_code(request: TaskRequest, raw: str) -> str:
    return extract_rust(raw) if request.language == "rust" else extract_python(raw)


def _is_well_formed_replacement(request: TaskRequest, code: str) -> bool:
    """Cheap, non-executing guard against accepting review prose as source."""

    if not code.strip():
        return False
    if request.language == "rust":
        return re.search(r"\bfn\s+main\s*\(", code) is not None
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    for statement in tree.body:
        if (
            isinstance(statement, ast.FunctionDef)
            and statement.name == request.entrypoint
        ):
            return True
        if (
            isinstance(statement, ast.ClassDef)
            and statement.name == request.entrypoint
        ):
            return True
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            if any(
                isinstance(target, ast.Name) and target.id == request.entrypoint
                for target in targets
            ):
                return True
    return False


class GLM52Client:
    """Small async client for Z.ai's chat-completions endpoint."""

    def __init__(
        self,
        settings: DemoMinerSettings,
        http: Optional[httpx.AsyncClient] = None,
    ):
        self.settings = settings
        self._http = http or httpx.AsyncClient()
        self._owns_http = http is None

    @property
    def model_name(self) -> str:
        return self.settings.glm_model

    @property
    def request_timeout_s(self) -> float:
        return self.settings.glm_request_timeout_s

    @property
    def log_prefix(self) -> str:
        return "demo-miner"

    @property
    def completion_url(self) -> str:
        base = self.settings.glm_base_url.rstrip("/")
        if not base.startswith("https://"):
            raise RuntimeError("GLM_BASE_URL must use HTTPS")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        timeout_s: float,
    ) -> str:
        if not self.settings.glm_api_key:
            raise RuntimeError("GLM_API_KEY is not configured")

        request: dict[str, Any] = {
            "model": self.settings.glm_model,
            "messages": messages,
            "stream": False,
            "max_tokens": self.settings.glm_max_tokens,
            "temperature": self.settings.glm_temperature,
            "thinking": {
                "type": "enabled" if self.settings.glm_thinking else "disabled"
            },
        }
        if self.settings.glm_thinking and self.settings.glm_reasoning_effort:
            request["reasoning_effort"] = self.settings.glm_reasoning_effort

        deadline = time.monotonic() + min(
            timeout_s, self.settings.glm_request_timeout_s
        )
        for attempt in range(self.settings.glm_max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("GLM request deadline exceeded")
            try:
                response = await self._http.post(
                    self.completion_url,
                    headers={
                        "Authorization": f"Bearer {self.settings.glm_api_key}",
                        "Content-Type": "application/json",
                        "Accept-Language": "en-US,en",
                    },
                    json=request,
                    timeout=remaining,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.settings.glm_max_retries:
                        await asyncio.sleep(min(2.0**attempt, max(0.0, remaining)))
                        continue
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("GLM returned an empty completion")
                return content
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt >= self.settings.glm_max_retries:
                    raise
                remaining = deadline - time.monotonic()
                await asyncio.sleep(min(2.0**attempt, max(0.0, remaining)))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise RuntimeError("GLM returned an invalid response") from exc

        raise RuntimeError("GLM request failed")

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()


class DemoMiner:
    """Verify subnet requests and turn them into signed model solutions."""

    def __init__(
        self,
        settings: DemoMinerSettings,
        client: Any,
        *,
        wallet: Any = None,
        subtensor: Any = None,
        metagraph: Any = None,
    ):
        self.settings = settings
        self.client = client
        self.wallet = wallet
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.nonces = NonceCache(window_ms=8000)
        self.solve_slots = asyncio.Semaphore(settings.miner_max_concurrent_requests)

    @property
    def model_name(self) -> str:
        return str(getattr(self.client, "model_name", "unknown"))

    @property
    def log_prefix(self) -> str:
        return str(getattr(self.client, "log_prefix", "demo-miner"))

    @property
    def hotkey_address(self) -> str:
        try:
            return str(self.wallet.hotkey.ss58_address)
        except Exception:  # noqa: BLE001
            return ""

    def authorize(self, signed_by: str) -> bool:
        """Require the caller to satisfy the configured metagraph policy."""

        hotkeys = getattr(self.metagraph, "hotkeys", None)
        if not hotkeys:
            return False
        try:
            uid = list(hotkeys).index(signed_by)
        except ValueError:
            return False

        if self.settings.miner_min_stake > 0.0:
            stakes = getattr(self.metagraph, "S", None)
            try:
                if stakes is None or float(stakes[uid]) < self.settings.miner_min_stake:
                    return False
            except (IndexError, TypeError, ValueError):
                return False

        if self.settings.miner_require_validator_permit:
            permits = getattr(self.metagraph, "validator_permit", None)
            try:
                if permits is None or not bool(permits[uid]):
                    return False
            except (IndexError, TypeError):
                return False
        return True

    async def solve(self, request: TaskRequest, timeout_s: float) -> SolutionPayload:
        """Generate, independently review, and return the strongest valid draft."""

        deadline = time.monotonic() + timeout_s
        self_verify = bool(getattr(self.settings, "miner_self_verify", True))
        reserve_s = min(
            float(getattr(self.settings, "miner_self_verify_reserve_s", 90.0)),
            timeout_s / 2.0,
        )
        draft_timeout_s = timeout_s - reserve_s if self_verify else timeout_s
        try:
            raw = await self.client.complete(
                build_model_messages(request),
                timeout_s=draft_timeout_s,
            )
        except httpx.HTTPStatusError as exc:
            print(
                f"[{self.log_prefix}] model request failed: "
                f"HTTP {exc.response.status_code}"
            )
            return SolutionPayload(
                problem_id=request.problem_id,
                code="",
                raw_response="<model request failed>",
            )
        except Exception as exc:  # noqa: BLE001 - provider failure scores zero
            print(f"[{self.log_prefix}] model request failed: {type(exc).__name__}")
            return SolutionPayload(
                problem_id=request.problem_id,
                code="",
                raw_response="<model request failed>",
            )
        draft_code = _extract_code(request, raw)
        if not self_verify or not draft_code:
            return SolutionPayload(
                problem_id=request.problem_id,
                code=draft_code,
                raw_response=raw,
            )

        remaining_s = deadline - time.monotonic()
        review_timeout_s = remaining_s - min(
            _REVIEW_RESPONSE_MARGIN_S,
            max(0.0, remaining_s / 4.0),
        )
        if review_timeout_s > 0.0:
            try:
                reviewed_raw = await asyncio.wait_for(
                    self.client.complete(
                        build_verification_messages(request, draft_code),
                        timeout_s=review_timeout_s,
                    ),
                    timeout=review_timeout_s,
                )
                reviewed_code = _extract_code(request, reviewed_raw)
                if _is_well_formed_replacement(request, reviewed_code):
                    return SolutionPayload(
                        problem_id=request.problem_id,
                        code=reviewed_code,
                        raw_response=reviewed_raw,
                    )
                print(
                    f"[{self.log_prefix}] self-verification returned invalid "
                    "source; using initial draft"
                )
            except Exception as exc:  # noqa: BLE001 - retain the usable draft
                print(
                    f"[{self.log_prefix}] self-verification failed; using "
                    f"initial draft ({type(exc).__name__})"
                )
        return SolutionPayload(
            problem_id=request.problem_id,
            code=draft_code,
            raw_response=raw,
        )

    async def handle_request(
        self, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, SolutionPayload | dict[str, str]]:
        expected_recipient = self.hotkey_address or None
        if not verify_signature(
            headers,
            body,
            expected_signed_for=expected_recipient,
        ):
            return 401, {"error": "invalid signature"}

        if not self.nonces.check_and_add(headers.get("Epistula-Uuid", "")):
            return 409, {"error": "replayed request"}

        signed_by = headers.get("Epistula-Signed-By", "")
        if self.metagraph is not None and not self.authorize(signed_by):
            return 403, {"error": "unauthorized signer"}

        try:
            request = TaskRequest.model_validate_json(body)
        except Exception:  # noqa: BLE001
            return 400, {"error": "invalid task request"}

        provider_timeout_s = float(
            getattr(self.client, "request_timeout_s", request.deadline_s)
        )
        timeout_s = min(request.deadline_s, provider_timeout_s)

        async def solve_with_slot() -> SolutionPayload:
            async with self.solve_slots:
                return await self.solve(request, timeout_s)

        try:
            payload = await asyncio.wait_for(solve_with_slot(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return 504, {"error": "solve deadline exceeded"}
        return 200, payload

    async def aclose(self) -> None:
        await self.client.aclose()


def build_demo_miner_app(miner: DemoMiner):
    """Build the FastAPI surface used by validators."""

    from fastapi import FastAPI, Request, Response

    sync_state = {"last": time.monotonic()}
    sync_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await miner.aclose()

    app = FastAPI(title=f"rlvr-{miner.log_prefix}", lifespan=lifespan)

    async def maybe_sync_metagraph() -> None:
        if miner.metagraph is None or miner.subtensor is None:
            return
        if (
            time.monotonic() - sync_state["last"]
            < miner.settings.miner_metagraph_sync_s
        ):
            return
        async with sync_lock:
            if (
                time.monotonic() - sync_state["last"]
                < miner.settings.miner_metagraph_sync_s
            ):
                return
            try:
                await asyncio.to_thread(miner.metagraph.sync, subtensor=miner.subtensor)
            except Exception as exc:  # noqa: BLE001 - use last known chain view
                print(
                    f"[{miner.log_prefix}] metagraph refresh failed; "
                    f"using cached view ({type(exc).__name__})"
                )
            finally:
                # Back off after failures too; otherwise every incoming request
                # would trigger another chain RPC while the endpoint is unhealthy.
                sync_state["last"] = time.monotonic()

    async def read_bounded(request: Request) -> Optional[bytes]:
        limit = miner.settings.miner_max_request_bytes
        try:
            declared = int(request.headers.get("content-length", "0") or 0)
        except ValueError:
            return None
        if declared < 0 or declared > limit:
            return None
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > limit:
                return None
        return bytes(body)

    @app.post("/solve")
    async def solve_endpoint(request: Request) -> Response:
        await maybe_sync_metagraph()
        body = await read_bounded(request)
        if body is None:
            return Response(
                content=b'{"error":"request body too large"}',
                status_code=413,
                media_type="application/json",
            )

        status, payload = await miner.handle_request(request.headers, body)
        if isinstance(payload, SolutionPayload):
            response_body = payload.model_dump_json().encode("utf-8")
        else:
            response_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        response = Response(
            content=response_body,
            status_code=status,
            media_type="application/json",
        )
        if status == 200 and miner.wallet is not None:
            response.headers.update(
                sign_message(
                    miner.wallet,
                    response_body,
                    signed_for=request.headers.get("Epistula-Signed-By", ""),
                )
            )
        return response

    @app.get("/solve")
    async def solve_endpoint(request: Request) -> Response:
        await maybe_sync_metagraph()
        body = await read_bounded(request)
        if body is None:
            return Response(
                content=b'{"error":"request body too large"}',
                status_code=413,
                media_type="application/json",
            )

        status, payload = await miner.handle_request(request.headers, body)
        if isinstance(payload, SolutionPayload):
            response_body = payload.model_dump_json().encode("utf-8")
        else:
            response_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        response = Response(
            content=response_body,
            status_code=status,
            media_type="application/json",
        )
        if status == 200 and miner.wallet is not None:
            response.headers.update(
                sign_message(
                    miner.wallet,
                    response_body,
                    signed_for=request.headers.get("Epistula-Signed-By", ""),
                )
            )
        return response
    
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": miner.model_name}

    return app


def run_demo_miner(settings: Optional[DemoMinerSettings] = None) -> None:
    """Set up the wallet, advertise the endpoint, and serve HTTP."""

    settings = settings or DemoMinerSettings()
    if not settings.glm_api_key:
        raise SystemExit("set GLM_API_KEY before starting the demo miner")

    import bittensor as bt  # type: ignore[import-not-found]
    import uvicorn

    wallet = bt.Wallet(name=settings.wallet_name, hotkey=settings.wallet_hotkey)
    network = settings.subtensor_chain_endpoint or settings.subtensor_network
    subtensor = bt.Subtensor(network=network)
    if not subtensor.is_hotkey_registered(
        netuid=settings.netuid,
        hotkey_ss58=wallet.hotkey.ss58_address,
    ):
        raise SystemExit(
            f"hotkey {wallet.hotkey.ss58_address} is not registered "
            f"on netuid {settings.netuid}"
        )
    metagraph = subtensor.metagraph(settings.netuid)

    axon_kwargs: dict[str, Any] = {
        "wallet": wallet,
        "port": settings.axon_port,
    }
    if settings.axon_external_ip:
        axon_kwargs["external_ip"] = settings.axon_external_ip
    axon = bt.Axon(**axon_kwargs)
    axon.serve(netuid=settings.netuid, subtensor=subtensor)

    print(
        f"[demo-miner] serving netuid={settings.netuid} "
        f"wallet={settings.wallet_name}/{settings.wallet_hotkey} "
        f"model={settings.glm_model} port={settings.axon_port}"
    )
    miner = DemoMiner(
        settings,
        GLM52Client(settings),
        wallet=wallet,
        subtensor=subtensor,
        metagraph=metagraph,
    )
    uvicorn.run(
        build_demo_miner_app(miner),
        host=settings.axon_host,
        port=settings.axon_port,
        log_level="info",
    )
