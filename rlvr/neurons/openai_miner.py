"""Hone miner backed by an OpenAI-compatible chat-completions API.

The provider boundary deliberately uses plain HTTP/JSON instead of a provider
SDK. This keeps the miner compatible with hosted services and self-hosted
servers that implement ``POST /chat/completions`` while preserving Hone's
signed validator/miner wire protocol.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional
from urllib.parse import urlsplit

import httpx
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .demo_miner import DemoMiner, build_demo_miner_app


_FENCED_CODE_RE = re.compile(
    r"```(?P<label>[^\n`]*)\n.*?```",
    re.DOTALL,
)
_REASONING_BLOCK_RE = re.compile(
    r"<(think|thinking|thought|reasoning|analysis)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)


class OpenAICompatibleSettings(BaseSettings):
    """Environment configuration for the OpenAI-compatible miner."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str = ""
    openai_require_api_key: bool = True
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = ""
    openai_max_tokens: int = Field(default=16_384, ge=1, le=131_072)
    openai_max_tokens_param: Literal[
        "max_tokens", "max_completion_tokens"
    ] = "max_tokens"
    openai_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    openai_reasoning_effort: str = Field(
        default="",
        pattern="^(|none|minimal|low|medium|high|max)$",
    )
    openai_request_timeout_s: float = Field(default=280.0, gt=0.0, le=3600.0)
    openai_max_retries: int = Field(default=2, ge=0, le=10)
    miner_self_verify: bool = True
    miner_self_verify_reserve_s: float = Field(default=90.0, ge=0.0, le=1800.0)
    openai_fallback_base_url: str = ""
    openai_fallback_model: str = ""
    openai_fallback_api_key: str = ""
    openai_fallback_reserve_s: float = Field(default=60.0, ge=0.0, le=1800.0)
    openai_allow_insecure_http: bool = False
    openai_extra_headers_json: str = "{}"
    openai_extra_body_json: str = "{}"

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
    miner_task_archive_file: str = "data/miner_tasks.jsonl"
    miner_self_test: bool = False
    miner_self_test_file: str = "data/miner_tests.jsonl"
    miner_self_test_executor: Literal["docker", "subprocess"] = "docker"
    miner_self_test_timeout_s: float = Field(default=5.0, gt=0.0, le=300.0)
    miner_self_test_docker_image: str = "python:3.12-slim"
    miner_self_test_docker_memory: str = "256m"
    miner_self_test_docker_cpus: float = Field(default=1.0, gt=0.0, le=256.0)
    miner_self_test_docker_pids_limit: int = Field(default=128, ge=16, le=4096)
    miner_metagraph_sync_s: float = Field(default=300.0, gt=0.0)
    miner_min_stake: float = Field(default=0.0, ge=0.0)
    miner_require_validator_permit: bool = True

    @field_validator("openai_temperature", mode="before")
    @classmethod
    def empty_temperature_is_none(cls, value):
        return None if value in (None, "", "none", "null") else value

    @model_validator(mode="after")
    def fallback_url_and_model_are_paired(self) -> "OpenAICompatibleSettings":
        has_url = bool(self.openai_fallback_base_url.strip())
        has_model = bool(self.openai_fallback_model.strip())
        if has_url != has_model:
            raise ValueError(
                "OPENAI_FALLBACK_BASE_URL and OPENAI_FALLBACK_MODEL must be "
                "configured together"
            )
        return self


def _json_object(raw: str, setting_name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{setting_name} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{setting_name} must contain a JSON object")
    return value


def _suppress_reasoning_text(content: str) -> str:
    """Keep solution fences while removing provider-visible reasoning text."""

    fences = list(_FENCED_CODE_RE.finditer(content))
    if fences:
        preferred_labels = {"py", "python", "python3", "rs", "rust"}
        solution = next(
            (
                match
                for match in fences
                if match.group("label").strip().lower() in preferred_labels
            ),
            fences[0],
        )
        return solution.group(0).strip()
    return _REASONING_BLOCK_RE.sub("", content).strip()


@dataclass(frozen=True)
class _ProviderTarget:
    label: str
    base_url: str
    model: str
    api_key: str
    api_key_setting: str


class OpenAICompatibleClient:
    """Async client for the common OpenAI ``/chat/completions`` contract."""

    def __init__(
        self,
        settings: OpenAICompatibleSettings,
        http: Optional[httpx.AsyncClient] = None,
    ):
        self.settings = settings
        self._http = http or httpx.AsyncClient()
        self._owns_http = http is None

    @property
    def model_name(self) -> str:
        return self.settings.openai_model

    @property
    def request_timeout_s(self) -> float:
        return self.settings.openai_request_timeout_s

    @property
    def log_prefix(self) -> str:
        return "openai-miner"

    @property
    def completion_url(self) -> str:
        return self._completion_url(
            self.settings.openai_base_url,
            setting_name="OPENAI_BASE_URL",
        )

    def _completion_url(self, base_url: str, *, setting_name: str) -> str:
        base = base_url.rstrip("/")
        parsed = urlsplit(base)
        if not parsed.scheme or not parsed.netloc:
            raise RuntimeError(
                f"{setting_name} must be an absolute HTTP(S) URL"
            )
        if parsed.scheme != "https":
            if parsed.scheme != "http" or not self.settings.openai_allow_insecure_http:
                raise RuntimeError(
                    f"{setting_name} must use HTTPS unless "
                    "OPENAI_ALLOW_INSECURE_HTTP=true"
                )
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _headers(
        self,
        api_key: Optional[str] = None,
        *,
        api_key_setting: str = "OPENAI_API_KEY",
    ) -> dict[str, str]:
        key = self.settings.openai_api_key if api_key is None else api_key
        if self.settings.openai_require_api_key and not key:
            raise RuntimeError(f"{api_key_setting} is not configured")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        extras = _json_object(
            self.settings.openai_extra_headers_json,
            "OPENAI_EXTRA_HEADERS_JSON",
        )
        protected = {"authorization", "content-type"}
        for name, value in extras.items():
            if str(name).lower() in protected:
                raise RuntimeError(
                    "OPENAI_EXTRA_HEADERS_JSON cannot override authorization "
                    "or content-type"
                )
            if not isinstance(value, str):
                raise RuntimeError(
                    "OPENAI_EXTRA_HEADERS_JSON values must all be strings"
                )
            headers[str(name)] = value
        return headers

    def _request_body(
        self,
        messages: list[dict[str, str]],
        *,
        model: Optional[str] = None,
    ) -> dict[str, Any]:
        model_name = self.settings.openai_model if model is None else model
        if not model_name:
            raise RuntimeError("OPENAI_MODEL is not configured")
        request: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            self.settings.openai_max_tokens_param: self.settings.openai_max_tokens,
        }
        if self.settings.openai_temperature is not None:
            request["temperature"] = self.settings.openai_temperature
        if self.settings.openai_reasoning_effort:
            request["reasoning_effort"] = self.settings.openai_reasoning_effort

        extras = _json_object(
            self.settings.openai_extra_body_json,
            "OPENAI_EXTRA_BODY_JSON",
        )
        protected = {
            "model",
            "messages",
            "stream",
            "max_tokens",
            "max_completion_tokens",
        }
        overlap = protected.intersection(extras)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise RuntimeError(
                f"OPENAI_EXTRA_BODY_JSON cannot override protected fields: {names}"
            )
        request.update(extras)
        return request

    def _primary_target(self) -> _ProviderTarget:
        return _ProviderTarget(
            label="primary",
            base_url=self.settings.openai_base_url,
            model=self.settings.openai_model,
            api_key=self.settings.openai_api_key,
            api_key_setting="OPENAI_API_KEY",
        )

    def _fallback_target(self) -> Optional[_ProviderTarget]:
        if not self.settings.openai_fallback_base_url:
            return None
        fallback_key = (
            self.settings.openai_fallback_api_key
            or self.settings.openai_api_key
        )
        key_setting = (
            "OPENAI_FALLBACK_API_KEY"
            if self.settings.openai_fallback_api_key
            else "OPENAI_API_KEY"
        )
        return _ProviderTarget(
            label="fallback",
            base_url=self.settings.openai_fallback_base_url,
            model=self.settings.openai_fallback_model,
            api_key=fallback_key,
            api_key_setting=key_setting,
        )

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code in {408, 409, 429} or status_code >= 500

    async def _complete_target(
        self,
        messages: list[dict[str, str]],
        *,
        target: _ProviderTarget,
        deadline: float,
    ) -> str:
        request = self._request_body(messages, model=target.model)
        headers = self._headers(
            target.api_key,
            api_key_setting=target.api_key_setting,
        )
        completion_url = self._completion_url(
            target.base_url,
            setting_name=(
                "OPENAI_BASE_URL"
                if target.label == "primary"
                else "OPENAI_FALLBACK_BASE_URL"
            ),
        )

        for attempt in range(self.settings.openai_max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"OpenAI-compatible {target.label} request deadline exceeded"
                )
            try:
                response = await self._http.post(
                    completion_url,
                    headers=headers,
                    json=request,
                    timeout=remaining,
                )
                if self._is_retryable_status(response.status_code):
                    if attempt < self.settings.openai_max_retries:
                        await asyncio.sleep(
                            min(
                                2.0**attempt,
                                max(0.0, deadline - time.monotonic()),
                            )
                        )
                        continue
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("provider returned an empty completion")
                sanitized = _suppress_reasoning_text(content)
                if not sanitized:
                    raise ValueError(
                        "provider returned reasoning without a solution"
                    )
                return sanitized
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt >= self.settings.openai_max_retries:
                    raise
                await asyncio.sleep(
                    min(
                        2.0**attempt,
                        max(0.0, deadline - time.monotonic()),
                    )
                )
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"OpenAI-compatible {target.label} provider returned an "
                    "invalid response"
                ) from exc

        raise RuntimeError(
            f"OpenAI-compatible {target.label} request failed"
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        timeout_s: float,
    ) -> str:
        budget_s = min(timeout_s, self.request_timeout_s)
        if budget_s <= 0:
            raise TimeoutError("OpenAI-compatible request deadline exceeded")
        overall_deadline = time.monotonic() + budget_s
        fallback = self._fallback_target()
        primary_deadline = overall_deadline
        if fallback is not None:
            reserve_s = min(
                self.settings.openai_fallback_reserve_s,
                budget_s / 2.0,
            )
            primary_deadline -= reserve_s

        try:
            return await self._complete_target(
                messages,
                target=self._primary_target(),
                deadline=primary_deadline,
            )
        except Exception as primary_error:  # noqa: BLE001 - fail over once
            if fallback is None:
                raise
            print(
                "[openai-miner] primary provider exhausted; trying fallback "
                f"({type(primary_error).__name__})"
            )
            try:
                return await self._complete_target(
                    messages,
                    target=fallback,
                    deadline=overall_deadline,
                )
            except Exception as fallback_error:  # noqa: BLE001 - terminal failure
                raise RuntimeError(
                    "primary and fallback OpenAI-compatible providers failed"
                ) from fallback_error

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()


class OpenAICompatibleMiner(DemoMiner):
    """Hone protocol server using an OpenAI-compatible completion client."""


def build_openai_miner_app(miner: OpenAICompatibleMiner):
    return build_demo_miner_app(miner)


def run_openai_miner(
    settings: Optional[OpenAICompatibleSettings] = None,
) -> None:
    """Advertise the miner endpoint on-chain and serve signed solutions."""

    settings = settings or OpenAICompatibleSettings()
    if not settings.openai_model:
        raise SystemExit("set OPENAI_MODEL before starting the miner")
    if settings.openai_require_api_key and not settings.openai_api_key:
        raise SystemExit("set OPENAI_API_KEY before starting the miner")

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
        f"[openai-miner] serving netuid={settings.netuid} "
        f"wallet={settings.wallet_name}/{settings.wallet_hotkey} "
        f"model={settings.openai_model} port={settings.axon_port} "
        f"fallback={settings.openai_fallback_model or 'disabled'}"
    )
    miner = OpenAICompatibleMiner(
        settings,
        OpenAICompatibleClient(settings),
        wallet=wallet,
        subtensor=subtensor,
        metagraph=metagraph,
    )
    uvicorn.run(
        build_openai_miner_app(miner),
        host=settings.axon_host,
        port=settings.axon_port,
        log_level="info",
    )


__all__ = [
    "OpenAICompatibleClient",
    "OpenAICompatibleMiner",
    "OpenAICompatibleSettings",
    "build_openai_miner_app",
    "run_openai_miner",
]
