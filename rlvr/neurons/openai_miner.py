"""Hone miner backed by an OpenAI-compatible chat-completions API.

The provider boundary deliberately uses plain HTTP/JSON instead of a provider
SDK. This keeps the miner compatible with hosted services and self-hosted
servers that implement ``POST /chat/completions`` while preserving Hone's
signed validator/miner wire protocol.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal, Optional
from urllib.parse import urlsplit

import httpx
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .demo_miner import DemoMiner, build_demo_miner_app


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
    miner_metagraph_sync_s: float = Field(default=300.0, gt=0.0)
    miner_min_stake: float = Field(default=0.0, ge=0.0)
    miner_require_validator_permit: bool = True

    @field_validator("openai_temperature", mode="before")
    @classmethod
    def empty_temperature_is_none(cls, value):
        return None if value in (None, "", "none", "null") else value


def _json_object(raw: str, setting_name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{setting_name} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{setting_name} must contain a JSON object")
    return value


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
        base = self.settings.openai_base_url.rstrip("/")
        parsed = urlsplit(base)
        if not parsed.scheme or not parsed.netloc:
            raise RuntimeError("OPENAI_BASE_URL must be an absolute HTTP(S) URL")
        if parsed.scheme != "https":
            if parsed.scheme != "http" or not self.settings.openai_allow_insecure_http:
                raise RuntimeError(
                    "OPENAI_BASE_URL must use HTTPS unless "
                    "OPENAI_ALLOW_INSECURE_HTTP=true"
                )
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _headers(self) -> dict[str, str]:
        if self.settings.openai_require_api_key and not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.settings.openai_api_key:
            headers["Authorization"] = f"Bearer {self.settings.openai_api_key}"
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

    def _request_body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if not self.settings.openai_model:
            raise RuntimeError("OPENAI_MODEL is not configured")
        request: dict[str, Any] = {
            "model": self.settings.openai_model,
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

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        timeout_s: float,
    ) -> str:
        request = self._request_body(messages)
        headers = self._headers()
        deadline = time.monotonic() + min(timeout_s, self.request_timeout_s)

        for attempt in range(self.settings.openai_max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("OpenAI-compatible request deadline exceeded")
            try:
                response = await self._http.post(
                    self.completion_url,
                    headers=headers,
                    json=request,
                    timeout=remaining,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.settings.openai_max_retries:
                        await asyncio.sleep(
                            min(2.0**attempt, max(0.0, deadline - time.monotonic()))
                        )
                        continue
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("provider returned an empty completion")
                return content
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt >= self.settings.openai_max_retries:
                    raise
                await asyncio.sleep(
                    min(2.0**attempt, max(0.0, deadline - time.monotonic()))
                )
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "OpenAI-compatible provider returned an invalid response"
                ) from exc

        raise RuntimeError("OpenAI-compatible request failed")

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
        f"model={settings.openai_model} port={settings.axon_port}"
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
