"""Reference miner backed by the GLM-5.2 chat-completions API.

The demo is a self-contained example of the validator/miner wire protocol. It
verifies and authorizes signed task requests, asks GLM-5.2 for a Python
solution, and signs the exact response bytes with the miner hotkey.
"""

import ast
import asyncio
import hashlib
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Literal, Mapping, Optional

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
from ..types import ExecutionResult, TestCase

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
_SELF_TEST_FENCE_RE = re.compile(
    r"```(?:self-tests|self_tests)\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_REVIEW_RESPONSE_MARGIN_S = 0.5
_CODE_ONLY_OUTPUT_INSTRUCTION = "Output exactly one complete fenced"


def _response_margin(remaining_s: float) -> float:
    return min(_REVIEW_RESPONSE_MARGIN_S, max(0.0, remaining_s / 4.0))


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

    # Reserve part of the signed request deadline for prompt-aware repairs. An
    # untested draft remains a fallback, but a draft that failed tests is gated.
    miner_self_verify: bool = True
    miner_self_verify_reserve_s: float = Field(default=90.0, ge=0.0, le=1800.0)
    # Number of repair/retest attempts allowed after a candidate fails local
    # tests. The deadline remains the hard upper bound for the loop.
    miner_self_verify_max_attempts: int = Field(default=3, ge=1, le=16)

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
    # Append validated public tasks for offline benchmark construction. The
    # archive contains no hidden tests; set an empty value to disable capture.
    miner_task_archive_file: str = "data/miner_tasks.jsonl"
    # Local preflight. Docker is the safe default; subprocess is available only
    # for local development and is not a Linux security boundary.
    miner_self_test: bool = True
    miner_self_test_file: str = "data/miner_tests.jsonl"
    miner_self_test_max_generated_cases: int = Field(default=8, ge=1, le=64)
    miner_self_test_executor: Literal["docker", "subprocess"] = "docker"
    miner_self_test_timeout_s: float = Field(default=5.0, gt=0.0, le=300.0)
    miner_self_test_docker_image: str = "python:3.12-slim"
    miner_self_test_docker_memory: str = "256m"
    miner_self_test_docker_cpus: float = Field(default=1.0, gt=0.0, le=256.0)
    miner_self_test_docker_pids_limit: int = Field(default=128, ge=16, le=4096)
    miner_metagraph_sync_s: float = Field(default=300.0, gt=0.0)
    miner_min_stake: float = Field(default=0.0, ge=0.0)
    miner_require_validator_permit: bool = True


def semantic_task_fingerprint(request: TaskRequest) -> str:
    """Fingerprint the public task while ignoring per-request identifiers."""

    task = request.model_dump(mode="json")
    fingerprint_input = {
        key: task[key]
        for key in (
            "language",
            "statement",
            "entrypoint",
            "public_examples",
            "prompt_variant",
        )
        if key in task
    }
    canonical = json.dumps(
        fingerprint_input,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def extract_python(text: str) -> str:
    """Extract the first Python fence, then any fence, or use the whole reply."""

    match = _PYTHON_FENCE_RE.search(text) or _ANY_FENCE_RE.search(text)
    return (match.group(1) if match else text).strip()


class TaskArchive:
    """Append validated public tasks as JSONL without affecting inference."""

    def __init__(self, path: str) -> None:
        self.path = str(path or "").strip()
        self._lock = threading.Lock()

    def append(self, request: TaskRequest, headers: Mapping[str, str]) -> None:
        if not self.path:
            return
        task = request.model_dump(mode="json")
        # The request body is the benchmark input. Receipt fields make repeated
        # observations traceable without retaining signatures or API headers.
        record = {
            "schema_version": 1,
            "captured_at": time.time(),
            "task_fingerprint": semantic_task_fingerprint(request),
            "request": task,
            "receipt": {
                "signed_by": headers.get("Epistula-Signed-By", ""),
                "request_nonce": headers.get("Epistula-Uuid", ""),
                "signed_at_ms": headers.get("Epistula-Timestamp", ""),
            },
        }
        encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        try:
            with self._lock:
                parent = os.path.dirname(self.path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(self.path, "ab+") as handle:
                    handle.seek(0, os.SEEK_END)
                    if handle.tell():
                        handle.seek(-1, os.SEEK_END)
                        if handle.read(1) != b"\n":
                            handle.seek(0, os.SEEK_END)
                            handle.write(b"\n")
                    handle.seek(0, os.SEEK_END)
                    handle.write(encoded)
        except OSError as exc:
            # Capturing is observability. A full or unavailable disk must not
            # turn an otherwise valid miner request into a failed solve.
            print(f"[demo-miner] task archive write failed ({type(exc).__name__})")


class LocalTestCorpus:
    """Load operator-owned test cases keyed by semantic task fingerprint."""

    def __init__(self, path: str) -> None:
        self.path = str(path or "").strip()
        self._tests: dict[str, list[TestCase]] = {}
        if not self.path:
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        fingerprint = str(record["task_fingerprint"])
                        raw_tests = record.get("tests", [])
                        if not isinstance(raw_tests, list):
                            raise ValueError("tests must be a list")
                        tests = [TestCase.model_validate(item) for item in raw_tests]
                    except Exception as exc:  # noqa: BLE001 - one bad record is isolated
                        print(
                            f"[demo-miner] ignoring local test record "
                            f"{self.path}:{line_number} ({type(exc).__name__})"
                        )
                        continue
                    self._tests.setdefault(fingerprint, []).extend(tests)
        except FileNotFoundError:
            return
        except OSError as exc:
            print(f"[demo-miner] local test corpus unavailable ({type(exc).__name__})")

    def for_request(self, request: TaskRequest) -> list[TestCase]:
        return list(self._tests.get(semantic_task_fingerprint(request), ()))


class LocalSelfTester:
    """Run public and operator-owned tests through the repository executor."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.corpus = LocalTestCorpus(
            getattr(settings, "miner_self_test_file", "data/miner_tests.jsonl")
        )
        self._executors: dict[str, Any] = {}

    def _get_executor(self, language: str):
        if language in self._executors:
            return self._executors[language]
        from types import SimpleNamespace

        from ..execution.executor import get_executor

        execution_settings = SimpleNamespace(
            executor=getattr(self.settings, "miner_self_test_executor", "docker"),
            docker_image=getattr(
                self.settings, "miner_self_test_docker_image", "python:3.12-slim"
            ),
            docker_memory=getattr(
                self.settings, "miner_self_test_docker_memory", "256m"
            ),
            docker_cpus=getattr(self.settings, "miner_self_test_docker_cpus", 1.0),
            docker_pids_limit=getattr(
                self.settings, "miner_self_test_docker_pids_limit", 128
            ),
        )
        executor = get_executor(execution_settings, language=language)
        self._executors[language] = executor
        return executor

    def cases_for(
        self,
        request: TaskRequest,
        generated_tests: Optional[list[TestCase]] = None,
    ) -> list[TestCase]:
        return [
            *request.public_examples,
            *self.corpus.for_request(request),
            *(generated_tests or ()),
        ]

    def run(
        self,
        request: TaskRequest,
        code: str,
        generated_tests: Optional[list[TestCase]] = None,
    ) -> list[ExecutionResult]:
        tests = self.cases_for(request, generated_tests)
        if not tests:
            return []
        executor = self._get_executor(request.language)
        return executor.run_tests(
            code,
            request.entrypoint,
            tests,
            float(getattr(self.settings, "miner_self_test_timeout_s", 5.0)),
        )


def extract_rust(text: str) -> str:
    """Extract the first Rust fence, then any fence, or use the whole reply."""

    match = _RUST_FENCE_RE.search(text) or _ANY_FENCE_RE.search(text)
    return (match.group(1) if match else text).strip()


def _self_test_output_contract(request: TaskRequest, max_cases: int) -> str:
    case_shape = (
        '{"args":["complete stdin"],"kwargs":{},"expected":"complete stdout"}'
        if request.language == "rust"
        else '{"args":[],"kwargs":{},"expected":null}'
    )
    language = "rust" if request.language == "rust" else "python"
    return (
        "Generate a compact self-test suite from the statement before writing "
        "the implementation. Cover normal, boundary, and adversarial behavior "
        "not already covered by the public examples. Derive expected results "
        "from the specification, independently of the implementation. Use only "
        "JSON-serializable values and do not include invalid inputs unless the "
        f"statement defines their behavior. Generate between 1 and {max_cases} "
        "cases.\n\n"
        "Output exactly two fenced blocks and no prose. The first must be a "
        "`self-tests` JSON block of this form:\n"
        f'{{"tests":[{case_shape}]}}\n'
        f"The second must be the complete fenced `{language}` solution."
    )


def build_model_messages(
    request: TaskRequest,
    *,
    generate_self_tests: bool = False,
    max_self_test_cases: int = 8,
) -> list[dict[str, str]]:
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
    system_prompt = (
        RUST_SYSTEM_PROMPT if request.language == "rust" else PYTHON_SYSTEM_PROMPT
    )
    if generate_self_tests:
        # Replace the normal one-source-block output instruction with the
        # initial generation contract. Review calls remain source-only.
        output_start = system_prompt.rfind(_CODE_ONLY_OUTPUT_INSTRUCTION)
        if output_start >= 0:
            system_prompt = system_prompt[:output_start].rstrip()
        system_prompt += "\n\n" + _self_test_output_contract(
            request,
            max_self_test_cases,
        )
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]


def extract_generated_tests(
    request: TaskRequest,
    raw: str,
    *,
    max_cases: int,
) -> list[TestCase]:
    """Parse and validate the model's optional fixed self-test suite."""

    match = _SELF_TEST_FENCE_RE.search(raw)
    if match is None:
        return []
    try:
        payload = json.loads(match.group(1))
        raw_tests = payload.get("tests") if isinstance(payload, dict) else None
        if not isinstance(raw_tests, list):
            return []
        tests: list[TestCase] = []
        for item in raw_tests[:max_cases]:
            case = TestCase.model_validate(item)
            if request.language == "rust" and not (
                len(case.args) == 1
                and isinstance(case.args[0], str)
                and not case.kwargs
                and isinstance(case.expected, str)
            ):
                continue
            tests.append(case)
        return tests
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def build_verification_messages(
    request: TaskRequest,
    candidate_code: str,
    local_test_feedback: Optional[list[str]] = None,
    generated_tests: Optional[list[TestCase]] = None,
    *,
    require_generated_tests: bool = False,
    max_self_test_cases: int = 8,
) -> list[dict[str, str]]:
    """Ask for a fresh audit while retaining the exact public task context."""

    language = "rust" if request.language == "rust" else "python"
    review_prompt = SELF_VERIFICATION_PROMPT
    if require_generated_tests:
        review_prompt = (
            "The candidate response omitted a valid generated self-test suite. "
            "Independently derive a compact suite from the original statement, "
            "review and repair the candidate against it, then follow the system "
            "output contract. Return the fixed `self-tests` block followed by "
            "the complete replacement source block, with no prose."
        )
    if local_test_feedback:
        review_prompt += (
            "\n\nThe miner's local preflight found these failures. Treat them as "
            "debugging evidence and repair the candidate before returning it:\n- "
            + "\n- ".join(local_test_feedback)
        )
    if generated_tests:
        review_prompt += (
            "\n\nKeep this fixed self-test suite as the acceptance gate; do not "
            "change its inputs or expected results:\n"
            + json.dumps(
                [case.model_dump(mode="json") for case in generated_tests],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return [
        *build_model_messages(
            request,
            generate_self_tests=require_generated_tests,
            max_self_test_cases=max_self_test_cases,
        ),
        {
            "role": "assistant",
            "content": f"```{language}\n{candidate_code.strip()}\n```",
        },
        {"role": "user", "content": review_prompt},
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
        self.task_archive = TaskArchive(
            getattr(settings, "miner_task_archive_file", "data/miner_tasks.jsonl")
        )
        self.self_tester = (
            LocalSelfTester(settings)
            if bool(getattr(settings, "miner_self_test", False))
            else None
        )

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
        """Generate, test, repair failed candidates, and return a solution.

        Local tests are a gate: a passing candidate is sent immediately. A
        failing candidate is given to the model with concrete failure evidence
        and is retested after each replacement while both the attempt cap and
        request deadline permit. If the local test harness is disabled, the
        legacy independent review pass is retained.
        """

        deadline = time.monotonic() + timeout_s
        self_verify = bool(getattr(self.settings, "miner_self_verify", True))
        max_generated_cases = int(
            getattr(self.settings, "miner_self_test_max_generated_cases", 8)
        )
        reserve_s = min(
            float(getattr(self.settings, "miner_self_verify_reserve_s", 90.0)),
            timeout_s / 2.0,
        )
        draft_timeout_s = timeout_s - reserve_s if self_verify else timeout_s
        try:
            raw = await self.client.complete(
                build_model_messages(
                    request,
                    generate_self_tests=self.self_tester is not None,
                    max_self_test_cases=max_generated_cases,
                ),
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
        if not draft_code:
            return SolutionPayload(
                problem_id=request.problem_id,
                code=draft_code,
                raw_response=raw,
            )

        generated_tests = extract_generated_tests(
            request,
            raw,
            max_cases=max_generated_cases,
        )
        require_generated_tests = self.self_tester is not None
        draft_failures = await self._local_test_failures(
            request,
            draft_code,
            generated_tests=generated_tests,
            budget_s=max(
                0.001,
                (remaining_s := deadline - time.monotonic())
                - _response_margin(remaining_s),
            ),
        )
        if require_generated_tests and not generated_tests:
            draft_failures = [
                *(draft_failures or ()),
                "required generated self-test suite was missing or malformed",
            ]

        # Once a configured test suite passes, do not spend the remaining
        # deadline on an unnecessary review call.
        if draft_failures == []:
            return SolutionPayload(
                problem_id=request.problem_id,
                code=draft_code,
                raw_response=raw,
            )
        if not self_verify:
            if draft_failures:
                print(
                    f"[{self.log_prefix}] local self-test rejected the draft "
                    f"({len(draft_failures)} failure(s))"
                )
                return SolutionPayload(
                    problem_id=request.problem_id,
                    code="",
                    raw_response="<local self-test failed>\n" + raw,
                )
            return SolutionPayload(
                problem_id=request.problem_id,
                code=draft_code,
                raw_response=raw,
            )

        candidate_code = draft_code
        candidate_failures = draft_failures or []
        # Without an executed failing suite this is the legacy independent
        # review. Actual test failures unlock the bounded repair loop.
        max_attempts = (
            int(getattr(self.settings, "miner_self_verify_max_attempts", 3))
            if draft_failures
            else 1
        )
        for attempt in range(max_attempts):
            remaining_s = deadline - time.monotonic()
            response_margin_s = _response_margin(remaining_s)
            if remaining_s <= response_margin_s:
                break
            # Each repair may use the remaining budget. If it returns quickly,
            # the loop can spend what remains on another repair and retest.
            review_timeout_s = max(0.001, remaining_s - response_margin_s)
            try:
                reviewed_raw = await asyncio.wait_for(
                    self.client.complete(
                        build_verification_messages(
                            request,
                            candidate_code,
                            local_test_feedback=candidate_failures,
                            generated_tests=generated_tests,
                            require_generated_tests=not generated_tests,
                            max_self_test_cases=max_generated_cases,
                        ),
                        timeout_s=review_timeout_s,
                    ),
                    timeout=review_timeout_s,
                )
            except Exception as exc:  # noqa: BLE001 - retain the usable draft
                action = (
                    "rejecting failed draft"
                    if draft_failures
                    else "using initial draft"
                )
                print(
                    f"[{self.log_prefix}] self-verification failed; {action} "
                    f"({type(exc).__name__})"
                )
                break

            reviewed_code = _extract_code(request, reviewed_raw)
            if not _is_well_formed_replacement(request, reviewed_code):
                action = (
                    "rejecting failed draft"
                    if draft_failures
                    else "using initial draft"
                )
                print(
                    f"[{self.log_prefix}] self-verification returned invalid "
                    f"source; {action}"
                )
                break

            if not generated_tests:
                generated_tests = extract_generated_tests(
                    request,
                    reviewed_raw,
                    max_cases=max_generated_cases,
                )
            reviewed_failures = await self._local_test_failures(
                request,
                reviewed_code,
                generated_tests=generated_tests,
                budget_s=max(
                    0.001,
                    (remaining_s := deadline - time.monotonic())
                    - _response_margin(remaining_s),
                ),
            )
            if require_generated_tests and not generated_tests:
                reviewed_failures = [
                    *(reviewed_failures or ()),
                    "required generated self-test suite was missing or malformed",
                ]
            if reviewed_failures == [] or (
                reviewed_failures is None and draft_failures is None
            ):
                return SolutionPayload(
                    problem_id=request.problem_id,
                    code=reviewed_code,
                    raw_response=reviewed_raw,
                )
            if reviewed_failures is None:
                print(
                    f"[{self.log_prefix}] repaired candidate could not be "
                    "retested before the deadline"
                )
                break
            candidate_code = reviewed_code
            candidate_failures = reviewed_failures
            print(
                f"[{self.log_prefix}] self-verification replacement failed "
                f"local self-test ({len(reviewed_failures)} failure(s)); "
                f"retry {attempt + 1}/{max_attempts}"
            )
        if draft_failures:
            print(
                f"[{self.log_prefix}] local self-test rejected the final draft "
                f"({len(draft_failures)} failure(s))"
            )
            return SolutionPayload(
                problem_id=request.problem_id,
                code="",
                raw_response="<local self-test failed>\n" + raw,
            )
        return SolutionPayload(problem_id=request.problem_id, code=draft_code, raw_response=raw)

    async def _local_test_failures(
        self,
        request: TaskRequest,
        code: str,
        *,
        generated_tests: Optional[list[TestCase]] = None,
        budget_s: Optional[float] = None,
    ) -> Optional[list[str]]:
        """Run configured local tests off the event loop and summarize failures."""

        if self.self_tester is None:
            return None
        try:
            run = asyncio.to_thread(
                self.self_tester.run,
                request,
                code,
                generated_tests,
            )
            if budget_s is None:
                results = await run
            else:
                results = await asyncio.wait_for(run, timeout=max(0.001, budget_s))
        except asyncio.TimeoutError:
            print(f"[{self.log_prefix}] local self-test exceeded its remaining budget")
            return None
        except Exception as exc:  # noqa: BLE001 - unavailable self-test is fail-open
            print(
                f"[{self.log_prefix}] local self-test unavailable; preserving candidate "
                f"({type(exc).__name__})"
            )
            return None
        if not results:
            return None
        failures: list[str] = []
        for result in results:
            if result.passed:
                continue
            detail = result.error or result.actual_repr or "failed"
            failures.append(f"test {result.test_index}: {detail[:300]}")
        return failures

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

        # Capture only after authentication, replay protection, authorization,
        # and schema validation. Hidden tests are never present in this request.
        self.task_archive.append(request, headers)

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
