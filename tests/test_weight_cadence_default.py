"""Default weight-submission cadence.

Two of these cases are not really about the number. They pin the REASONS 180 was
chosen, because a bare `== 180` assertion tells a future reader nothing about
what breaks if they change it — and this is a value someone will eventually be
tempted to tune.

"""

from __future__ import annotations

from types import SimpleNamespace

from rlvr.config import Settings
from rlvr.neurons.decentralized import (
    _apply_weights_rate_limit,
    effective_weights_interval,
)
from rlvr.neurons.validator import ValidatorNeuron
from rlvr.policy import RELEASE_POLICY

SN5_WEIGHTS_INTERVAL_BLOCKS = 180
# The on-chain per-validator floor this default is chosen to clear.
KNOWN_CHAIN_RATE_LIMIT = 100
# Emission tempo the cadence has to fit inside.
TEMPO_BLOCKS = 360


def test_the_default_weight_interval_is_the_sn5_value():
    assert RELEASE_POLICY.weights_interval_blocks == SN5_WEIGHTS_INTERVAL_BLOCKS


def test_the_validator_adopts_the_configured_weight_interval():
    v = ValidatorNeuron(Settings(_env_file=None))

    assert v.weights_interval_blocks == SN5_WEIGHTS_INTERVAL_BLOCKS


def test_operator_environment_cannot_override_the_weight_cadence(monkeypatch):
    monkeypatch.setenv("WEIGHTS_INTERVAL_BLOCKS", "240")

    assert ValidatorNeuron(Settings(_env_file=None)).weights_interval_blocks == 180


def test_the_round_cadence_is_not_disturbed_by_the_weight_default():
    """The weight default must not alter the independent round interval."""
    assert RELEASE_POLICY.round_interval_blocks == 38


# --------------------------------------------------------------------------- #
# Why 180 — the two properties that make it the right number
# --------------------------------------------------------------------------- #
def test_the_default_already_clears_the_known_chain_floor_unaided():
    """This is what makes 'degrade to configured' a SAFE fallback.

    If the chain read fails at startup we keep the configured value. That is
    only defensible because the configured value is independently above the
    floor: max(180, 100 + 20) is still 180, so a failed read costs diagnostics,
    not correctness. Lower this default below 120 and the fallback silently
    becomes a cadence the chain will reject.
    """
    assert effective_weights_interval(
        SN5_WEIGHTS_INTERVAL_BLOCKS, KNOWN_CHAIN_RATE_LIMIT
    ) == SN5_WEIGHTS_INTERVAL_BLOCKS


def test_an_unreadable_chain_leaves_a_cadence_that_is_still_safe():
    """The same property, exercised through the real startup path."""
    settings = Settings(_env_file=None)
    v = ValidatorNeuron(settings)

    def explode(_netuid):
        raise RuntimeError("substrate connection refused")

    v.subtensor = SimpleNamespace(weights_rate_limit=explode)
    _apply_weights_rate_limit(v, settings)

    assert v.chain_weights_rate_limit is None
    assert v.weights_interval_blocks == SN5_WEIGHTS_INTERVAL_BLOCKS
    assert v.weights_interval_blocks > KNOWN_CHAIN_RATE_LIMIT


def test_the_default_permits_two_attempts_per_tempo():
    """180 buys a second attempt inside each 360-block tempo.

    One attempt per tempo means a single failed extrinsic costs the whole
    window's emissions. Anything above 180 drops back to one, which is the
    concrete cost of "just raise it a bit to be safer".
    """
    assert TEMPO_BLOCKS // SN5_WEIGHTS_INTERVAL_BLOCKS == 2
