"""Default challenge-round cadence.

These tests cover the validator round interval. Weight cadence has its own
independent policy and tests.

"""

from __future__ import annotations

from rlvr.config import Settings
from rlvr.neurons.validator import ValidatorNeuron
from rlvr.policy import RELEASE_POLICY

SN5_ROUND_INTERVAL_BLOCKS = 38


def test_the_default_round_interval_is_the_sn5_value():
    assert RELEASE_POLICY.round_interval_blocks == SN5_ROUND_INTERVAL_BLOCKS


def test_the_validator_adopts_the_configured_round_interval():
    """The default is worthless if the run loop reads something else."""
    v = ValidatorNeuron(Settings(_env_file=None))

    assert v.round_interval_blocks == SN5_ROUND_INTERVAL_BLOCKS


def test_operator_environment_cannot_override_the_cadence(monkeypatch):
    monkeypatch.setenv("ROUND_INTERVAL_BLOCKS", "200")

    assert ValidatorNeuron(Settings(_env_file=None)).round_interval_blocks == 38


def test_another_environment_value_cannot_override_the_cadence(monkeypatch):
    monkeypatch.setenv("ROUND_INTERVAL_BLOCKS", "1")

    assert ValidatorNeuron(Settings(_env_file=None)).round_interval_blocks == 38
