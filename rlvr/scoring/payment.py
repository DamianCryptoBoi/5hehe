"""Convert verified correctness outcomes into chain payments.

Only responses that pass the complete hidden test suite receive a nonzero
payment. Among those responses, the fastest validator-measured latency receives
a 1.0 multiplier. Each additional ``speed_half_life_ms`` halves the above-floor
share, flattening out at a configurable floor. Defaults make speed a very light,
continuous tiebreaker (three-minute half-life, floor 0.95), never the dominant
term.
Correctness is a hard gate: a fast failing answer receives nothing. If a
non-networked harness supplies no valid latency, speed weighting is skipped so
offline evaluation stays meaningful.

TODO(V2): Revisit duplication-resistant scoring only with a signal that cannot
be cheaply gamed. Simple AST fingerprints encourage irrelevant structural
changes in otherwise valid solutions, so V1 deliberately does not use them for
scoring or difficulty classification.
"""

from __future__ import annotations

import math
from typing import Iterable

from ..types import MinerOutcome

DEFAULT_SPEED_HALF_LIFE_MS = 180_000.0
DEFAULT_SPEED_FLOOR = 0.95


def compute_payments(
    outcomes: Iterable[MinerOutcome],
    *,
    speed_half_life_ms: float = DEFAULT_SPEED_HALF_LIFE_MS,
    speed_floor: float = DEFAULT_SPEED_FLOOR,
) -> dict[int, float]:
    """Map one problem's outcomes to {uid: payment} for the EvalEngine.

    Only ``verification.all_passed`` earns payment. A failing or partial result
    remains in the mapping as zero so it enters the recent score history.

    ``speed_half_life_ms`` controls a relative latency multiplier. For a
    rewarded response arriving one half-life after the fastest, the portion
    above ``speed_floor`` is halved; after two half-lives that portion is
    quartered, and so on. Latency is taken from
    ``SolutionResponse.latency_ms``, which the live validator measures around
    the signed HTTP exchange. When no rewarded response has a finite positive
    latency, the multiplier is 1.0 for everyone.
    """
    outcomes = list(outcomes)
    if not outcomes:
        return {}
    latency_floor = min(1.0, max(0.0, float(speed_floor)))
    half_life = float(speed_half_life_ms)

    rewarded_latencies = [
        float(o.solution.latency_ms)
        for o in outcomes
        if o.verification.all_passed
        and math.isfinite(float(o.solution.latency_ms))
        and float(o.solution.latency_ms) > 0
    ]
    fastest_latency = min(rewarded_latencies) if rewarded_latencies else None

    def speed_multiplier(outcome: MinerOutcome) -> float:
        if fastest_latency is None or not math.isfinite(half_life) or half_life <= 0:
            return 1.0
        latency = float(outcome.solution.latency_ms)
        if not math.isfinite(latency) or latency <= 0:
            return latency_floor
        delay = max(0.0, latency - fastest_latency)
        return latency_floor + (1.0 - latency_floor) * (2.0 ** (-delay / half_life))

    payments: dict[int, float] = {}
    for o in outcomes:
        if not o.verification.all_passed:
            payments[o.uid] = 0.0
            continue
        payments[o.uid] = speed_multiplier(o)
    return payments
