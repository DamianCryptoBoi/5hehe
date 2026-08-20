"""Chain-derived weight cadence floor.

Startup reads the chain's own weights rate limit and derives an effective
interval of max(configured, chain_limit + 20). The 180 default and structured
error logging are tested separately.

The failure modes matter more than the happy path. This value governs how often
the validator touches the chain, and both directions of error are expensive:
too fast earns rejections, too slow forfeits emissions. So an unreadable or
nonsensical chain answer must degrade to the operator's configured value rather
than to a guess.

"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlvr.neurons.decentralized import (
    _WEIGHTS_RATE_LIMIT_MARGIN,
    _read_weights_rate_limit,
    effective_weights_interval,
)


# --------------------------------------------------------------------------- #
# The derivation itself
# --------------------------------------------------------------------------- #
def test_the_margin_is_a_real_safety_gap():
    assert _WEIGHTS_RATE_LIMIT_MARGIN == 20


def test_a_configured_interval_above_the_chain_floor_is_kept():
    """180 configured against a 100-block chain limit stays 180.

    The floor is a minimum, not a target; raising the interval to 120 here would
    submit LESS often than the operator asked for, for no reason.
    """
    assert effective_weights_interval(180, 100) == 180


def test_a_configured_interval_below_the_chain_floor_is_raised():
    """88 configured against a 100-block limit must become 120, not 88.

    Submitting inside the chain's rate limit is rejected outright, so an
    operator who lowers this below the floor otherwise silently stops setting
    weights at all.
    """
    assert effective_weights_interval(88, 100) == 100 + _WEIGHTS_RATE_LIMIT_MARGIN


def test_a_raised_chain_limit_overrides_a_previously_safe_configuration():
    """The chain can move; a config written against an old floor must not stick."""
    assert effective_weights_interval(180, 500) == 500 + _WEIGHTS_RATE_LIMIT_MARGIN


def test_an_unknown_chain_limit_falls_back_to_the_configured_value():
    """A failed read must not invent a cadence in either direction."""
    assert effective_weights_interval(180, None) == 180


@pytest.mark.parametrize("nonsense", [0, -1, -500])
def test_a_non_positive_chain_limit_is_ignored(nonsense):
    """A zero or negative limit is not a floor; treating it as one would let
    `chain_limit + 20` become an absurdly small interval."""
    assert effective_weights_interval(180, nonsense) == 180


# --------------------------------------------------------------------------- #
# Reading it from the chain — every failure is the operator's problem, not a crash
# --------------------------------------------------------------------------- #
def validator_returning(value):
    def weights_rate_limit(netuid):
        return value

    return SimpleNamespace(subtensor=SimpleNamespace(weights_rate_limit=weights_rate_limit))


def test_a_chain_limit_is_read_for_the_configured_netuid():
    seen = {}

    def weights_rate_limit(netuid):
        seen["netuid"] = netuid
        return 100

    v = SimpleNamespace(
        subtensor=SimpleNamespace(weights_rate_limit=weights_rate_limit)
    )

    assert _read_weights_rate_limit(v, 5) == 100
    assert seen["netuid"] == 5


def test_an_rpc_failure_yields_no_limit_rather_than_raising():
    """A diagnostic read must never take the validator down at startup."""

    def explode(netuid):
        raise RuntimeError("substrate connection refused")

    v = SimpleNamespace(subtensor=SimpleNamespace(weights_rate_limit=explode))

    assert _read_weights_rate_limit(v, 5) is None


@pytest.mark.parametrize("value", [None, "100", 1.5, float("nan")])
def test_a_non_integer_chain_answer_is_refused(value):
    """An SDK that changes shape must not silently become a cadence.

    Anything that is not a plain integer is discarded, so the configured value
    governs and the operator sees a log line instead of a mystery interval.
    """
    assert _read_weights_rate_limit(validator_returning(value), 5) is None


def test_a_boolean_is_not_an_integer_here():
    """bool is an int subclass in Python; True would otherwise mean 1 block."""
    assert _read_weights_rate_limit(validator_returning(True), 5) is None


def test_a_missing_sdk_method_yields_no_limit():
    """Older SDKs may not expose it at all; that is not a startup failure."""
    v = SimpleNamespace(subtensor=SimpleNamespace())

    assert _read_weights_rate_limit(v, 5) is None
