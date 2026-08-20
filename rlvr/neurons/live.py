"""Validator-side live HTTP transport for dispatching signed tasks."""

from __future__ import annotations

import asyncio
import time
from typing import Optional
from uuid import uuid4

import httpx

from ..config import Settings
from ..protocol import SignedSolution, SolutionPayload, TaskRequest, sign_message, verify_signature
from ..types import Problem, SolutionResponse
from .validator import ValidatorNeuron

_SEND_START_BUDGET_S = 6.0


class _SendPermit:
    """One idempotently releasable slot from a :class:`SendGate`."""

    def __init__(self, semaphore: asyncio.BoundedSemaphore):
        self._semaphore = semaphore
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._semaphore.release()


class SendGate:
    """Bound signatures waiting to hand their request body to HTTPX."""

    def __init__(self, capacity: int):
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise ValueError("send gate capacity must be a positive integer")
        self._semaphore = asyncio.BoundedSemaphore(int(capacity))

    async def acquire(self) -> _SendPermit:
        await self._semaphore.acquire()
        return _SendPermit(self._semaphore)

    @property
    def available(self) -> int:
        """Currently available slots, exposed for health checks and tests."""
        return int(self._semaphore._value)


async def _read_bounded_response(
    response: httpx.Response, limit: int
) -> Optional[bytes]:
    """Read a miner response up to ``limit`` bytes, then fail closed."""
    declared = response.headers.get("content-length")
    if declared:
        try:
            if int(declared) < 0 or int(declared) > limit:
                return None
        except ValueError:
            return None
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > limit:
            return None
    return bytes(body)


# --------------------------------------------------------------------------- #
# Validator side: a SolverClient that queries a registered miner's axon.
# --------------------------------------------------------------------------- #
class LiveSolverClient:
    """Queries one miner's HTTP axon with an epistula-signed TaskRequest."""

    def __init__(
        self,
        uid: int,
        hotkey: str,
        url: str,
        wallet,
        settings: Settings,
        http: httpx.AsyncClient,
        gate: Optional[SendGate] = None,
    ):
        self.uid = uid
        self.hotkey = hotkey
        self._url = url.rstrip("/")
        self._wallet = wallet
        self._settings = settings
        self._http = http
        self._gate = gate or SendGate(settings.validator_send_concurrency)

    async def solve(self, problem: Problem, prompt: str) -> SolutionResponse:
        artifact = await self.solve_signed(problem, request_id=uuid4().hex)
        return artifact.to_solution(problem.problem_id)

    async def solve_signed(self, problem: Problem, request_id: str) -> SignedSolution:
        """Return exact miner-signed bytes for commit-before-test-reveal."""
        req = TaskRequest(
            problem_id=request_id,
            language=problem.language,
            statement=problem.statement,
            entrypoint=problem.entrypoint,
            public_examples=problem.public_examples,
            deadline_s=self._settings.solve_deadline_s,
            prompt_variant=problem.prompt_variant,
        )
        body = req.model_dump_json().encode("utf-8")
        permit = await self._gate.acquire()
        signed_at = time.monotonic()
        started = signed_at
        body_consumed = asyncio.Event()

        async def body_stream():
            try:
                yield body
                body_consumed.set()
            finally:
                # HTTPX has consumed the body. Miner solve time must not retain
                # this short-lived send-start slot.
                permit.release()

        # Sign only after send capacity is available, bound to this miner.
        try:
            headers = sign_message(self._wallet, body, signed_for=self.hotkey)
        except Exception as e:  # noqa: BLE001 - signing failures fail this attempt
            permit.release()
            return SignedSolution(
                error=f"<dispatch error: {e}>",
                latency_ms=(time.monotonic() - started) * 1000.0,
            )
        headers["Content-Type"] = "application/json"
        # Async request content otherwise switches HTTP framing to chunked.
        headers["Content-Length"] = str(len(body))

        async def exchange() -> SignedSolution:
            responded = False
            try:
                async with self._http.stream(
                    "POST",
                    f"{self._url}/solve",
                    content=body_stream(),
                    headers=headers,
                    # Slack past the miner's deadline so a response finishing right
                    # at the deadline isn't dropped on the wire.
                    timeout=req.deadline_s + 10.0,
                ) as resp:
                    responded = True
                    if resp.status_code != 200:
                        signed_age_s = (
                            time.monotonic() - signed_at
                            if resp.status_code == 401
                            else None
                        )
                        error_body = await _read_bounded_response(resp, 512)
                        detail = (
                            error_body.decode("utf-8", errors="replace")[:200]
                            if error_body is not None
                            else "<oversized error response>"
                        )
                        if signed_age_s is not None:
                            detail = (
                                f"{detail} (rejected {signed_age_s:.3f}s "
                                "after signing)"
                            )
                        return SignedSolution(
                            error=f"HTTP {resp.status_code}: {detail}",
                            latency_ms=(time.monotonic() - started) * 1000.0,
                            responded=True,
                            signed_age_s=signed_age_s,
                        )
                    response_body = await _read_bounded_response(
                        resp, max(1, self._settings.miner_max_response_bytes)
                    )
                    if response_body is None:
                        return SignedSolution(
                            error="<miner response exceeds byte limit>",
                            latency_ms=(time.monotonic() - started) * 1000.0,
                            responded=True,
                        )
                    response_headers = {
                        name: resp.headers.get(name, "")
                        for name in (
                            "Epistula-Version",
                            "Epistula-Timestamp",
                            "Epistula-Uuid",
                            "Epistula-Signed-By",
                            "Epistula-Signed-For",
                            "Epistula-Request-Signature",
                        )
                    }
                if (
                    response_headers["Epistula-Signed-By"] != self.hotkey
                    or not verify_signature(
                        response_headers,
                        response_body,
                        expected_signed_for=headers["Epistula-Signed-By"],
                    )
                ):
                    return SignedSolution(
                        error="<invalid or unauthenticated miner response>",
                        latency_ms=(time.monotonic() - started) * 1000.0,
                        responded=True,
                    )
                payload = SolutionPayload.model_validate_json(response_body)
                if payload.problem_id != req.problem_id:
                    return SignedSolution(
                        error="<miner response request-id mismatch>",
                        latency_ms=(time.monotonic() - started) * 1000.0,
                        responded=True,
                    )
                return SignedSolution(
                    response_body=response_body.decode("utf-8"),
                    response_headers=response_headers,
                    latency_ms=(time.monotonic() - started) * 1000.0,
                    responded=True,
                )
            except Exception as e:  # noqa: BLE001 - unreachable miner scores 0
                return SignedSolution(
                    error=f"<dispatch error: {e}>",
                    latency_ms=(time.monotonic() - started) * 1000.0,
                    responded=responded,
                )

        exchange_task = asyncio.create_task(exchange())
        consumed_task = asyncio.create_task(body_consumed.wait())
        try:
            done, _ = await asyncio.wait(
                {exchange_task, consumed_task},
                timeout=_SEND_START_BUDGET_S,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if exchange_task in done:
                return await exchange_task
            if consumed_task in done:
                return await exchange_task
            exchange_task.cancel()
            await asyncio.gather(exchange_task, return_exceptions=True)
            return SignedSolution(
                error="<request body did not reach transport before signature deadline>",
                latency_ms=(time.monotonic() - started) * 1000.0,
            )
        finally:
            consumed_task.cancel()
            if not exchange_task.done():
                exchange_task.cancel()
            await asyncio.gather(
                consumed_task, exchange_task, return_exceptions=True
            )
            # Connection setup can fail or cancellation can arrive before HTTPX
            # consumes the request iterator. Return the slot exactly once.
            permit.release()


def _solver_clients(
    validator: ValidatorNeuron,
    wallet,
    settings: Settings,
    http: httpx.AsyncClient,
    gate: Optional[SendGate] = None,
) -> list[LiveSolverClient]:
    """Build a LiveSolverClient for every serving miner in the metagraph."""
    mg = validator.metagraph
    clients: list[LiveSolverClient] = []
    for uid in range(len(mg.hotkeys)):
        axon = mg.axons[uid]
        ip, port = getattr(axon, "ip", "0.0.0.0"), getattr(axon, "port", 0)
        if not port or ip in ("0.0.0.0", "", None):
            continue  # miner not serving an axon
        clients.append(
            LiveSolverClient(
                uid,
                mg.hotkeys[uid],
                f"http://{ip}:{port}",
                wallet,
                settings,
                http,
                gate=gate,
            )
        )
    return clients
