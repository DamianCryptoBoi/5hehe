"""Lease deferral deadline behavior.

Storage and replacement semantics on ValidatorNeuron. Run-loop suppression and
gate re-anchoring are covered independently.

"""

from __future__ import annotations

import math

import pytest

from rlvr.config import Settings
from rlvr.neurons.validator import ValidatorNeuron


def neuron() -> ValidatorNeuron:
    """Chain-free construction; nothing here touches bittensor."""
    return ValidatorNeuron(Settings(_env_file=None))


def test_a_fresh_validator_has_no_deferral():
    assert neuron().rounds_deferred_until is None


def test_a_deadline_is_stored_as_given():
    v = neuron()
    v.defer_rounds_until(1_500.0)

    assert v.rounds_deferred_until == 1_500.0


# --------------------------------------------------------------------------- #
# Replacement policy: a deferral may be extended, never quietly shortened
# --------------------------------------------------------------------------- #
def test_a_later_deadline_extends_the_deferral():
    v = neuron()
    v.defer_rounds_until(1_000.0)
    v.defer_rounds_until(1_060.0)

    assert v.rounds_deferred_until == 1_060.0


def test_an_earlier_deadline_does_not_shorten_an_active_deferral():
    """With challenges_per_round > 1, two paced leases can land in one round.

    Honoring the shorter of two live directives would lease into a server that
    had already asked for the longer wait, which is the exact behavior the
    pacing work exists to stop.
    """
    v = neuron()
    v.defer_rounds_until(1_060.0)
    v.defer_rounds_until(1_010.0)

    assert v.rounds_deferred_until == 1_060.0


def test_an_expired_deadline_is_superseded_by_a_fresh_one():
    """Absolute monotonic deadlines make this fall out of the same rule.

    An expired deadline is by definition smaller than any deadline derived from
    a later 'now', so 'never shorten' needs no separate expiry branch — but it
    must actually behave that way, hence the case.
    """
    v = neuron()
    v.defer_rounds_until(1_000.0)  # long past by the time the next one arrives
    v.defer_rounds_until(9_000.0)

    assert v.rounds_deferred_until == 9_000.0


# --------------------------------------------------------------------------- #
# Fail closed: a deadline that cannot be compared silently disables pacing
# --------------------------------------------------------------------------- #
def test_nan_deadline_is_rejected():
    """`now < nan` is False, so a NaN deadline disables pacing invisibly.

    Silently ignoring server pacing is worse than crashing: the validator keeps
    leasing into a server that asked it to stop, and nothing in the log says so.
    """
    v = neuron()
    with pytest.raises(ValueError):
        v.defer_rounds_until(math.nan)

    assert v.rounds_deferred_until is None


def test_infinite_deadline_is_rejected():
    """An infinite deferral parks the validator permanently with no diagnosis."""
    v = neuron()
    with pytest.raises(ValueError):
        v.defer_rounds_until(math.inf)

    assert v.rounds_deferred_until is None


def test_a_rejected_deadline_leaves_an_existing_deferral_intact():
    v = neuron()
    v.defer_rounds_until(1_060.0)
    with pytest.raises(ValueError):
        v.defer_rounds_until(math.nan)

    assert v.rounds_deferred_until == 1_060.0


# --------------------------------------------------------------------------- #
# Scope guards: deadline storage must not alter unrelated scheduling state
# --------------------------------------------------------------------------- #
def test_deferring_does_not_anchor_the_round_gate():
    """Re-anchoring belongs to the deferred RETRY, not to setting the deadline.

    If storage anchored the gate, the deferral would silently consume a round
    interval that was never run.
    """
    v = neuron()
    assert v._last_run_block is None
    v.defer_rounds_until(1_060.0)
    assert v._last_run_block is None

    v._last_run_block = 100
    v.defer_rounds_until(2_000.0)
    assert v._last_run_block == 100


def test_deferring_does_not_disturb_the_configured_intervals():
    """Pacing suppresses round WORK; it must not rewrite the cadence config."""
    v = neuron()
    before = (v.round_interval_blocks, v.weights_interval_blocks)
    v.defer_rounds_until(1_060.0)

    assert (v.round_interval_blocks, v.weights_interval_blocks) == before
