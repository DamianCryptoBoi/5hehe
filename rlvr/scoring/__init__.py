"""Scoring subsystem: verification, difficulty placement, and weight aggregation.

- `verifier.Verifier`   -> runs a solution against hidden tests, filters flaky
                           cases, and emits correctness plus compile errors.
- `difficulty`          -> pool pass-rate and band classification.
- `payment`             -> complete-suite correctness gate and speed tiebreaker.
- `eval_engine.EvalEngine` -> bounded score history -> L1-normalized, nan-safe weights
                           (adapted from the BitMind EvalEngine pattern).
"""

from __future__ import annotations

from .difficulty import classify_band, pass_rate
from .eval_engine import EvalEngine
from .payment import compute_payments
from .verifier import Verifier

__all__ = [
    "Verifier",
    "EvalEngine",
    "pass_rate",
    "classify_band",
    "compute_payments",
]
