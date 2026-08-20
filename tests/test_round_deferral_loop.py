"""Run-loop behavior during lease deferral.

Round work is suppressed until the stored monotonic deadline while the weight
scheduler keeps running, and the loop keeps polling rather than sleeping the
deferral off inside one iteration.

Covered separately:
  * deferred-retry gate anchoring — nothing here asserts what
    happens to ``_last_run_block`` or to the stored deadline at the moment the
    deadline opens, beyond "a suppressed poll must not consume the anchor";
  * integration wiring — no lease, no LeaseOutcome, no decentralized.py. The
    deferral is set directly through the deadline setter;
  * cadence defaults — every interval here is forced to 1 so the ordinary block
    gates are always eligible and the deferral is the only suppressor under test.

"""

from __future__ import annotations

import asyncio

from rlvr.config import Settings
from rlvr.neurons import validator as validator_module
from rlvr.neurons.validator import ValidatorNeuron

START = 1_000.0


class Clock:
    """Controllable stand-in for the run loop's ``time`` module.

    It exposes ``monotonic`` and nothing else on purpose: a run loop that
    reaches for ``time.time()`` fails here with AttributeError rather than
    quietly passing while measuring a steppable wall clock.
    """

    def __init__(self, start: float = START):
        self.t = start

    def monotonic(self) -> float:
        return self.t


def build(round_callback, weight_setter, clock) -> ValidatorNeuron:
    """A chain-free validator whose block advances one per loop iteration."""
    v = ValidatorNeuron(
        Settings(_env_file=None),
        round_callback=round_callback,
        weight_setter=weight_setter,
    )
    v.subtensor = object()  # run() refuses to start without one
    v.current_block = lambda: int(clock.t - START)
    # Intervals of 1 keep both ordinary gates permanently eligible, so anything
    # that does not happen here was suppressed by the deferral and nothing else.
    v.round_interval_blocks = 1
    v.weights_interval_blocks = 1
    return v


async def drive(v, clock, monkeypatch, *, ticks: int, on_tick=None):
    """Run the loop for a bounded number of iterations against the fake clock.

    The loop's own ``asyncio.sleep`` is what advances the clock, so one tick is
    one second and one block. No test here sleeps for real.
    """
    seen = {"sleeps": 0, "longest": 0.0}

    async def fake_sleep(seconds):
        seen["sleeps"] += 1
        seen["longest"] = max(seen["longest"], float(seconds))
        clock.t += float(seconds)
        if on_tick is not None:
            on_tick()
        if seen["sleeps"] >= ticks:
            v.stop()

    async def fake_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    # raising=False so the two baseline cases below can run against today's
    # loop, which has no clock at all. Red must point at behavior.
    monkeypatch.setattr(validator_module, "time", clock, raising=False)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    v._should_exit = False  # a prior drive() ended by calling stop()
    await v.run()
    return seen


# --------------------------------------------------------------------------- #
# The central requirement: pacing stops leasing, not the whole validator
# --------------------------------------------------------------------------- #
async def test_round_work_is_suppressed_while_weights_keep_flowing(monkeypatch):
    """A validator that stalls its weight scheduler to wait out a lease
    deferral stops earning for the entire window — a far worse outcome than the
    pacing it is obeying."""
    clock = Clock()
    rounds, weights = [], []

    async def round_callback(v):
        rounds.append(clock.t)
        v.defer_rounds_until(clock.t + 30.0)
        return {}

    v = build(round_callback, lambda _v: weights.append(clock.t), clock)
    await drive(v, clock, monkeypatch, ticks=10)

    # One round ran and asked for 30s; the clock only advanced 10.
    assert rounds == [START]
    # ...and the weight scheduler kept running straight through the deferral.
    assert len(weights) >= 5, f"weights starved during the deferral: {weights}"


async def test_a_deferral_before_the_first_round_starves_weights_deliberately(
    monkeypatch,
):
    """The one case where weights SHOULD stall, and it must stay that way.

    Weights are gated on a completed round precisely so the initial all-zero
    score vector is never submitted. A deferral that lands before any round has
    run therefore holds weights too. This is a guard against "fixing" that
    starvation by loosening the rounds gate.
    """
    clock = Clock()
    rounds, weights = [], []

    async def round_callback(v):
        rounds.append(clock.t)
        return {}

    v = build(round_callback, lambda _v: weights.append(clock.t), clock)
    v.defer_rounds_until(clock.t + 30.0)
    await drive(v, clock, monkeypatch, ticks=5)

    assert rounds == []
    assert weights == [], "the initial all-zero weight vector must never ship"


# --------------------------------------------------------------------------- #
# The trap: should_run() anchors as a side effect of returning True
# --------------------------------------------------------------------------- #
async def test_a_suppressed_poll_does_not_consume_the_round_gate_anchor(monkeypatch):
    """``should_run()`` mutates ``_last_run_block`` when it opens the gate.

    If the deferral is checked AFTER the gate, every suppressed poll silently
    re-anchors it, so round intervals are consumed by rounds that never
    happened. The deferral has to short-circuit BEFORE should_run is consulted.
    """
    clock = Clock()
    anchors = []

    async def round_callback(v):
        v.defer_rounds_until(clock.t + 30.0)
        return {}

    v = build(round_callback, lambda _v: None, clock)
    await drive(
        v,
        clock,
        monkeypatch,
        ticks=10,
        on_tick=lambda: anchors.append(v._last_run_block),
    )

    # Tick 1 runs the only round and anchors the gate. The nine suppressed polls
    # after it must leave that anchor exactly where the round left it.
    assert anchors[0] is not None, "the first round never ran"
    assert set(anchors) == {anchors[0]}, f"anchor moved while suppressed: {anchors}"


# --------------------------------------------------------------------------- #
# The deadline is a not-before instant: never early, and not a tick late
# --------------------------------------------------------------------------- #
async def test_round_work_never_resumes_before_the_deadline(monkeypatch):
    clock = Clock()
    rounds = []

    async def round_callback(v):
        rounds.append(clock.t)
        if len(rounds) == 1:
            v.defer_rounds_until(clock.t + 3.0)
        return {}

    v = build(round_callback, lambda _v: None, clock)
    await drive(v, clock, monkeypatch, ticks=10)

    assert len(rounds) > 1, "round work never resumed after the deadline passed"
    assert rounds[1] >= rounds[0] + 3.0, f"leased inside the paced window: {rounds}"


async def test_round_work_resumes_at_the_deadline_instant(monkeypatch):
    """Pins ``now < deadline`` rather than ``<=``.

    The deadline already carries the parsing cushion, so the stored instant is
    the first moment leasing is permitted — holding one extra poll past it is an
    off-by-one, not extra safety.
    """
    clock = Clock()
    rounds = []

    async def round_callback(v):
        rounds.append(clock.t)
        if len(rounds) == 1:
            v.defer_rounds_until(clock.t + 3.0)
        return {}

    v = build(round_callback, lambda _v: None, clock)
    await drive(v, clock, monkeypatch, ticks=10)

    assert rounds[1] == rounds[0] + 3.0, f"resumed off the deadline instant: {rounds}"


# --------------------------------------------------------------------------- #
# Guards: these pass against today's loop and must stay passing
# --------------------------------------------------------------------------- #
async def test_the_loop_polls_through_the_deferral_instead_of_sleeping_it_off(
    monkeypatch,
):
    """Waiting the directive out inside one iteration would freeze the weight
    scheduler and make the validator deaf to stop() for the whole window."""
    clock = Clock()

    async def round_callback(v):
        v.defer_rounds_until(clock.t + 30.0)
        return {}

    v = build(round_callback, lambda _v: None, clock)
    seen = await drive(v, clock, monkeypatch, ticks=10)

    assert seen["sleeps"] == 10
    assert seen["longest"] <= 1.0, (
        f"loop slept {seen['longest']}s in one iteration; the deferral must be "
        "polled through, not waited out"
    )


async def test_an_unset_deferral_suppresses_nothing(monkeypatch):
    """Control: with no directive the loop behaves exactly as it does today."""
    clock = Clock()
    rounds = []

    async def round_callback(v):
        rounds.append(clock.t)
        return {}

    v = build(round_callback, lambda _v: None, clock)
    await drive(v, clock, monkeypatch, ticks=5)

    assert len(rounds) == 5
