"""Signature freshness: a request must not be signed before it can be sent.

Epistula stamps a wall-clock timestamp at signing and miners reject anything
outside an eight-second window. Signing therefore starts a clock that the
connection setup, the event loop, and any queueing in front of the miner all
spend. Under a wide fan-out the earliest-signed requests can reach the wire long
after their timestamps, and are rejected as replays despite being genuine.

The fix is to sign only once a send slot is held, and to give that slot back the
moment the body reaches the transport — not when the miner finishes thinking. A
miner is allowed to take the full solve deadline; it must not hold a send slot
for that whole time.

Two properties here are about the mechanism failing safely rather than about the
feature working:

  * the slot count must return exactly to its starting value after any mix of
    successes, connection failures and timeouts. A leaked slot shrinks capacity
    until dispatch deadlocks; a doubly-returned slot raises the ceiling, so the
    limit silently stops limiting and the original defect returns under load.
  * the request must stay byte-identical on the wire. Streaming a body to learn
    when it is consumed switches HTTP framing from Content-Length to chunked
    unless the length is set explicitly, and that reaches every miner.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from rlvr.config import Settings
from rlvr.neurons import live as live_module
from rlvr.neurons.live import LiveSolverClient, SendGate
from rlvr.types import Problem

BODY_PROBLEM = Problem(
    problem_id="p1",
    language="python",
    statement="Return a + b.",
    entrypoint="add",
    tests=[],
)


def client(handler, *, gate=None, settings=None):
    return LiveSolverClient(
        0,
        "hotkey-0",
        "http://miner.invalid",
        "validator-hotkey",
        settings or Settings(_env_file=None),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        gate=gate,
    )


async def solve(handler, *, gate=None, request_id="req-0"):
    c = client(handler, gate=gate)
    try:
        return await c.solve_signed(BODY_PROBLEM, request_id)
    finally:
        await c._http.aclose()


def ok(request):
    return httpx.Response(200, content=b"{}")


# --------------------------------------------------------------------------- #
# Wire fidelity — the framing must not change
# --------------------------------------------------------------------------- #
async def test_the_request_keeps_an_explicit_content_length():
    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        seen["body"] = request.read()
        return ok(request)

    await solve(handler)

    assert "content-length" in seen["headers"]


async def test_the_request_is_never_chunked():
    """Streaming the body to observe consumption must not change the framing.

    A miner behind a proxy that will not accept chunked request bodies would
    start failing for a reason unrelated to anything it did.
    """
    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        return ok(request)

    await solve(handler)

    assert "transfer-encoding" not in seen["headers"]


async def test_the_signed_body_is_what_is_actually_sent():
    """The signature covers a hash of the body; the two must not diverge."""
    seen = {}

    def handler(request):
        seen["body"] = request.read()
        seen["headers"] = dict(request.headers)
        return ok(request)

    await solve(handler)

    assert seen["body"]
    assert seen["headers"]["content-length"] == str(len(seen["body"]))


# --------------------------------------------------------------------------- #
# Signing order — the whole point
# --------------------------------------------------------------------------- #
async def test_a_request_waiting_for_a_send_slot_is_not_signed_yet(monkeypatch):
    """A task queued behind a full gate must not have started its clock.

    Signing on entry and then waiting is exactly the defect: the timestamp ages
    while the request sits in a queue it cannot see.
    """
    signings: list[float] = []
    real_sign = live_module.sign_message

    def recording_sign(*args, **kwargs):
        signings.append(asyncio.get_running_loop().time())
        return real_sign(*args, **kwargs)

    monkeypatch.setattr(live_module, "sign_message", recording_sign)

    gate = SendGate(1)
    first_arrived = asyncio.Event()
    consume_first = asyncio.Event()

    class HeldBodyTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.calls = 0

        async def handle_async_request(self, request):
            self.calls += 1
            if self.calls == 1:
                first_arrived.set()
                await consume_first.wait()
            await request.aread()
            return ok(request)

    http = httpx.AsyncClient(transport=HeldBodyTransport())
    settings = Settings(_env_file=None)
    clients = [
        LiveSolverClient(
            uid,
            f"hotkey-{uid}",
            "http://miner.invalid",
            "validator-hotkey",
            settings,
            http,
            gate=gate,
        )
        for uid in range(2)
    ]
    tasks = [
        asyncio.create_task(client.solve_signed(BODY_PROBLEM, f"req-{uid}"))
        for uid, client in enumerate(clients)
    ]
    await first_arrived.wait()
    await asyncio.sleep(0)

    assert len(signings) == 1, "the queued request was signed before a slot opened"

    consume_first.set()
    try:
        await asyncio.gather(*tasks)
    finally:
        await http.aclose()

    assert len(signings) == 2
    assert signings[1] > signings[0]


# --------------------------------------------------------------------------- #
# Slot accounting — failing safe matters more than the happy path
# --------------------------------------------------------------------------- #
async def test_a_slow_miner_does_not_hold_a_send_slot():
    """The slot covers reaching the wire, not the miner's thinking time.

    Holding it for the full solve deadline would serialise the fan-out into
    deadline-length waves, which is worse than the problem being solved.
    """
    gate = SendGate(1)
    started = asyncio.Event()

    async def slow_handler(request):
        started.set()
        await asyncio.sleep(0.2)
        return ok(request)

    task = asyncio.create_task(solve(slow_handler, gate=gate))
    await started.wait()
    await asyncio.sleep(0.05)

    assert gate.available == 1, "a thinking miner is still holding a send slot"
    await task


async def test_a_miner_that_never_reads_the_body_cannot_hold_a_slot_forever():
    """A slot must be bounded by the freshness window, not the solve deadline.

    A miner can accept a connection and simply never consume the request. The
    slot is then held until the whole-response deadline expires, which is two
    orders of magnitude longer than the window the slot exists to protect.
    Enough miners doing that — deliberately or merely by being overloaded —
    hold every slot and stall dispatch for the rest of the pool.

    Bounding the hold costs nothing real: a body that has not reached the
    transport within the freshness window is already carrying a signature that
    will be rejected on arrival, so continuing to hold capacity for it buys a
    request that cannot succeed.
    """
    gate = SendGate(1)

    class NeverReadsBody(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            await asyncio.sleep(30)
            return ok(request)

    http = httpx.AsyncClient(transport=NeverReadsBody())
    settings = Settings(_env_file=None)
    stalling = LiveSolverClient(
        0, "hotkey-0", "http://miner.invalid", "validator-hotkey",
        settings, http, gate=gate,
    )
    task = asyncio.create_task(stalling.solve_signed(BODY_PROBLEM, "req-0"))
    await asyncio.sleep(0.05)
    assert gate.available == 0

    try:
        await asyncio.wait_for(gate.acquire(), timeout=live_module._SEND_START_BUDGET_S + 1.0)
    except asyncio.TimeoutError:  # pragma: no cover - the failure being asserted
        pytest.fail(
            "a miner that never read the request body held its send slot past "
            "the freshness budget; enough such miners stall the whole dispatch"
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await http.aclose()


async def test_a_send_that_misses_the_freshness_window_fails_fast():
    """The doomed request should report, not linger.

    Its signature is already stale, so waiting out the full solve deadline
    consumes a dispatch slot to deliver something guaranteed to be rejected.
    """
    gate = SendGate(1)

    class NeverReadsBody(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            await asyncio.sleep(30)
            return ok(request)

    http = httpx.AsyncClient(transport=NeverReadsBody())
    settings = Settings(_env_file=None)
    stalling = LiveSolverClient(
        0, "hotkey-0", "http://miner.invalid", "validator-hotkey",
        settings, http, gate=gate,
    )
    try:
        artifact = await asyncio.wait_for(
            stalling.solve_signed(BODY_PROBLEM, "req-0"),
            timeout=live_module._SEND_START_BUDGET_S + 2.0,
        )
    finally:
        await http.aclose()

    assert artifact.error
    assert not artifact.response_body


async def test_a_connection_failure_returns_its_slot():
    gate = SendGate(2)

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    await solve(refuse, gate=gate)

    assert gate.available == 2


async def test_a_cancelled_send_returns_its_slot():
    gate = SendGate(2)

    async def hang(request):
        await asyncio.sleep(10)
        return ok(request)

    c = client(hang, gate=gate)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                c.solve_signed(BODY_PROBLEM, "req-0"), timeout=0.05
            )
    finally:
        await c._http.aclose()

    assert gate.available == 2


async def test_slot_count_returns_to_its_start_after_a_mixed_batch():
    """Successes, refusals and timeouts together — the realistic case."""
    gate = SendGate(4)
    calls = {"n": 0}

    def mixed(request):
        calls["n"] += 1
        if calls["n"] % 3 == 1:
            raise httpx.ConnectError("refused", request=request)
        return ok(request)

    await asyncio.gather(
        *(solve(mixed, gate=gate, request_id=f"req-{i}") for i in range(9))
    )

    assert gate.available == 4


async def test_returning_a_slot_twice_is_refused_rather_than_widening_the_gate():
    """A plain semaphore would raise its own ceiling and stop limiting.

    That fails open and silently: the limit disappears and the only symptom is
    the original defect returning under load.
    """
    gate = SendGate(1)
    permit = await gate.acquire()
    permit.release()

    assert gate.available == 1
    permit.release()  # idempotent on the lease, never a second real release

    assert gate.available == 1


# --------------------------------------------------------------------------- #
# Observability — a rejection must say how stale it was
# --------------------------------------------------------------------------- #
async def test_a_rejected_signature_reports_its_age():
    """Without this, a stale-signature 401 is indistinguishable from a clock
    problem on the miner's host, and the two need opposite responses."""

    def unauthorized(request):
        return httpx.Response(401, content=b"invalid signature")

    artifact = await solve(unauthorized)

    assert artifact.signed_age_s is not None
    assert artifact.signed_age_s >= 0.0


async def test_rejection_age_is_captured_before_reading_its_body(monkeypatch):
    """A slow error body must not make a prompt rejection look stale."""
    clock = {"now": 100.0}
    monkeypatch.setattr(
        live_module.time, "monotonic", lambda: clock["now"]
    )

    class DelayedErrorBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            clock["now"] = 110.0
            yield b'invalid signature'

    def unauthorized(request):
        return httpx.Response(401, stream=DelayedErrorBody())

    artifact = await solve(unauthorized)

    assert artifact.signed_age_s == 0.0
    assert "rejected 0.000s after signing" in artifact.error


async def test_an_ordinary_failure_does_not_claim_an_age():
    """Only a rejection that could plausibly be staleness carries the number."""

    def server_error(request):
        return httpx.Response(500, content=b"boom")

    artifact = await solve(server_error)

    assert artifact.signed_age_s is None


async def test_signature_age_is_measured_monotonically(monkeypatch):
    """A wall-clock step mid-dispatch must not corrupt the measurement."""
    import time as time_module

    def unauthorized(request):
        return httpx.Response(401, content=b"invalid signature")

    monkeypatch.setattr(time_module, "time", lambda: 0.0)
    artifact = await solve(unauthorized)

    assert artifact.signed_age_s is not None
    assert artifact.signed_age_s < 60.0
