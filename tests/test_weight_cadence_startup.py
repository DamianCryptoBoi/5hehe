"""Startup application of the chain-derived weight cadence.

The cadence derivation is inert until startup installs it on the neuron the run
loop reads. Startup also records the chain's answer for truthful diagnostics.

The 180 default and structured extrinsic error logging are tested separately.

"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlvr.config import Settings
from rlvr.neurons.decentralized import (
    _WEIGHTS_RATE_LIMIT_MARGIN,
    _apply_weights_rate_limit,
)
from rlvr.neurons.validator import ValidatorNeuron


def validator_with(chain_answer, *, configured: int, netuid: int = 5):
    """A chain-free ValidatorNeuron whose subtensor reports a rate limit."""
    settings = Settings(
        _env_file=None, weights_interval_blocks=configured, netuid=netuid
    )
    v = ValidatorNeuron(settings)
    seen: dict = {}

    def weights_rate_limit(uid):
        seen["netuid"] = uid
        if isinstance(chain_answer, Exception):
            raise chain_answer
        return chain_answer

    v.subtensor = SimpleNamespace(weights_rate_limit=weights_rate_limit)
    return v, settings, seen


def test_a_higher_chain_floor_is_installed_on_the_neuron():
    """The run loop reads weights_interval_blocks; that is the only value that
    governs anything, so the derivation has to land there."""
    v, settings, _ = validator_with(500, configured=180)

    _apply_weights_rate_limit(v, settings)

    assert v.weights_interval_blocks == 500 + _WEIGHTS_RATE_LIMIT_MARGIN


def test_a_configured_interval_already_above_the_floor_is_left_alone():
    v, settings, _ = validator_with(100, configured=180)

    _apply_weights_rate_limit(v, settings)

    assert v.weights_interval_blocks == 180


def test_the_chain_answer_is_recorded_for_later_diagnostics():
    """Diagnostics must be able to say what the chain claimed.

    Reporting an effective interval without the limit it was derived from
    leaves an operator unable to tell a config problem from a chain change.
    """
    v, settings, _ = validator_with(100, configured=180)

    _apply_weights_rate_limit(v, settings)

    assert v.chain_weights_rate_limit == 100


def test_the_limit_is_read_for_the_configured_netuid():
    v, settings, seen = validator_with(100, configured=180, netuid=5)

    _apply_weights_rate_limit(v, settings)

    assert seen["netuid"] == 5


# --------------------------------------------------------------------------- #
# Failure: startup must survive a chain that will not answer
# --------------------------------------------------------------------------- #
def test_an_unreadable_limit_leaves_the_configured_cadence_in_place():
    v, settings, _ = validator_with(
        RuntimeError("substrate connection refused"), configured=180
    )

    _apply_weights_rate_limit(v, settings)

    assert v.weights_interval_blocks == 180
    assert v.chain_weights_rate_limit is None


def test_an_unreadable_limit_does_not_raise():
    """A validator that will not start because a diagnostic read failed is a
    worse outcome than one running at the operator's configured cadence."""
    v, settings, _ = validator_with(RuntimeError("boom"), configured=180)

    _apply_weights_rate_limit(v, settings)  # must not raise


@pytest.mark.parametrize("nonsense", [True, 0, -5, "100", 1.5, None])
def test_an_untrustworthy_answer_leaves_the_configured_cadence_in_place(nonsense):
    v, settings, _ = validator_with(nonsense, configured=180)

    _apply_weights_rate_limit(v, settings)

    assert v.weights_interval_blocks == 180
    assert v.chain_weights_rate_limit is None


# --------------------------------------------------------------------------- #
# Scope: this behavior governs the weight cadence and nothing else
# --------------------------------------------------------------------------- #
def test_applying_the_floor_does_not_disturb_the_round_cadence():
    v, settings, _ = validator_with(500, configured=180)
    before = v.round_interval_blocks

    _apply_weights_rate_limit(v, settings)

    assert v.round_interval_blocks == before


def test_a_fresh_validator_reports_no_chain_limit_yet():
    """Before startup applies anything, the field must not claim a value."""
    assert ValidatorNeuron(Settings(_env_file=None)).chain_weights_rate_limit is None
