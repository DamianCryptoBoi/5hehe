"""Deferred lease retry behavior.

When the deadline opens, the retry fires on its own authority, clears the
deadline, and anchors the ordinary round gate at the retry block so the gate
cannot fire again on the very next poll.

This file deliberately uses a LARGE round interval and a deferral long enough to
outlast it. That combination is the only way the anchor is observable: with a
short deferral the ordinary gate would not have fired anyway, and a missing
anchor would look identical to a correct one.

"""

from __future__ import annotations

import asyncio

from rlvr.config import Settings
from rlvr.neurons import validator as validator_module
from rlvr.neurons.validator import ValidatorNeuron

START = 1_000.0
INTERVAL = 100


class Clock:
    """Exposes monotonic and nothing else, so a wall-clock read fails loudly."""

    def __init__(self, start: float = START):
        self.t = start

    def monotonic(self) -> float:
        return self.t


def build(round_callback, clock) -> ValidatorNeuron:
    v = ValidatorNeuron(
        Settings(_env_file=None),
        round_callback=round_callback,
        weight_setter=lambda _v: None,
    )
    v.subtensor = object()
    v.current_block = lambda: int(clock.t - START)
    # Large, so an ordinary gate firing is a real event and not background noise.
    v.round_interval_blocks = INTERVAL
    v.weights_interval_blocks = INTERVAL
    return v


async def drive(v, clock, monkeypatch, *, ticks: int):
    """One tick == one second == one block."""
    seen = {"sleeps": 0}

    async def fake_sleep(seconds):
        seen["sleeps"] += 1
        clock.t += float(seconds)
        if seen["sleeps"] >= ticks:
            v.stop()

    async def fake_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(validator_module, "time", clock, raising=False)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    v._should_exit = False
    await v.run()
    return seen


def paces_once(seconds: float, rounds: list):
    """Round callback that requests one deferral on its first invocation."""

    async def round_callback(v):
        rounds.append(v.current_block())
        if len(rounds) == 1:
            v.defer_rounds_until(v_clock.monotonic() + seconds)
        return {}

    return round_callback


# The callback needs the clock; bind it per test rather than globally.
v_clock: Clock


async def test_the_retry_fires_even_though_the_block_gate_would_not(monkeypatch):
    """A short pace must not cost a whole round interval.

    Three blocks elapse during a 3s deferral, far short of the 100-block gate.
    Without a retry of its own the validator would idle until block 100 — the
    pacing would cost far more than the server asked for.
    """
    global v_clock
    v_clock = clock = Clock()
    rounds: list[int] = []

    v = build(paces_once(3.0, rounds), clock)
    await drive(v, clock, monkeypatch, ticks=10)

    assert len(rounds) >= 2, f"the deferred retry never fired: {rounds}"
    assert rounds[1] == 3, f"retry fired at the wrong block: {rounds}"


async def test_the_retry_clears_the_deadline(monkeypatch):
    global v_clock
    v_clock = clock = Clock()
    rounds: list[int] = []

    v = build(paces_once(3.0, rounds), clock)
    await drive(v, clock, monkeypatch, ticks=10)

    assert v.rounds_deferred_until is None


async def test_the_retry_anchors_the_gate_at_the_retry_block(monkeypatch):
    """150 blocks pass during the deferral, so the anchor is load-bearing."""
    global v_clock
    v_clock = clock = Clock()
    rounds: list[int] = []

    v = build(paces_once(150.0, rounds), clock)
    await drive(v, clock, monkeypatch, ticks=152)

    assert rounds[:2] == [0, 150], f"unexpected round schedule: {rounds}"
    assert v._last_run_block == 150, (
        f"gate anchored at {v._last_run_block}, not the retry block 150"
    )


async def test_the_ordinary_gate_cannot_fire_on_the_poll_after_the_retry(monkeypatch):
    """The defect this behavior exists to prevent.

    150 blocks elapsed while deferred. If the retry does not re-anchor, the
    ordinary gate sees 151 blocks since its last anchor at 0 and fires again one
    second later — two rounds back to back, burning two problems.
    """
    global v_clock
    v_clock = clock = Clock()
    rounds: list[int] = []

    v = build(paces_once(150.0, rounds), clock)
    await drive(v, clock, monkeypatch, ticks=155)

    assert rounds == [0, 150], (
        f"expected exactly one ordinary round and one deferred retry, got {rounds}"
    )


async def test_a_retry_that_is_paced_again_keeps_the_new_deadline(monkeypatch):
    """Ordering trap: the deadline must be cleared BEFORE the round runs.

    If it is cleared afterwards, a retry that the server paces again has its
    fresh directive wiped, and the validator leases straight back into a server
    that just told it to wait.
    """
    global v_clock
    v_clock = clock = Clock()
    rounds: list[int] = []

    async def round_callback(v):
        rounds.append(v.current_block())
        if len(rounds) == 1:
            v.defer_rounds_until(clock.monotonic() + 3.0)
        elif len(rounds) == 2:
            v.defer_rounds_until(clock.monotonic() + 50.0)
        return {}

    v = build(round_callback, clock)
    await drive(v, clock, monkeypatch, ticks=10)

    assert len(rounds) == 2, f"the second deferral was not honored: {rounds}"
    assert v.rounds_deferred_until == START + 3.0 + 50.0, (
        f"the re-pacing directive was lost: {v.rounds_deferred_until}"
    )


async def test_an_unpaced_loop_keeps_the_ordinary_cadence(monkeypatch):
    """Control: with no directive, nothing about the gate changes."""
    global v_clock
    v_clock = clock = Clock()
    rounds: list[int] = []

    async def round_callback(v):
        rounds.append(v.current_block())
        return {}

    v = build(round_callback, clock)
    await drive(v, clock, monkeypatch, ticks=205)

    assert rounds == [0, 100, 200], f"ordinary cadence disturbed: {rounds}"
