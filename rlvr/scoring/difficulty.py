"""Dependency-free validator pass-rate and difficulty classification."""

from __future__ import annotations

from typing import Iterable

from ..types import DifficultyBand, MinerOutcome


def pass_rate(outcomes: Iterable[MinerOutcome]) -> float:
    """Fraction of outcomes that FULLY passed the hidden suite.

    Uses ``verification.all_passed`` rather than ``reward > 0`` so that under
    ``reward_partial_credit`` a partial solve (0 < reward < 1) does NOT count as
    a pass. Counting partials as passes would inflate p and corrupt the
    difficulty band + saturation signal.
    Returns 0.0 for an empty pool (no responders -> no measurable difficulty).
    """
    outcomes = list(outcomes)
    n = len(outcomes)
    if n == 0:
        return 0.0
    passed = sum(1 for o in outcomes if o.verification.all_passed)
    return passed / n


def classify_band(p: float, low: float, high: float) -> DifficultyBand:
    """Place a measured pass-rate into a difficulty band.

    p > high  -> EASY  (too easy; most miners solve it)
    p < low   -> HARD  (too hard; toward 0)
    otherwise -> IN_BAND (the useful [low, high] training zone)
    """
    if p > high:
        return DifficultyBand.EASY
    if p < low:
        return DifficultyBand.HARD
    return DifficultyBand.IN_BAND
