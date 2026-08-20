"""Payment based on all-pass correctness and a small latency adjustment.

The scoring rule: a fully correct response pays its latency multiplier in
[speed_floor, 1.0]; anything else pays zero; a miner's score is the mean of its
latest observations. Peer pass-rates do not scale payment. The latency
multiplier uses the configured floor and half-life.
"""

from __future__ import annotations

import inspect

from rlvr.config import Settings
from rlvr.scoring.eval_engine import EvalEngine
from rlvr.scoring.payment import compute_payments
from rlvr.types import MinerOutcome, SolutionResponse, Verification


def outcome(uid: int, *, passed: bool, latency_ms: float = 1_000.0) -> MinerOutcome:
    return MinerOutcome(
        uid=uid,
        hotkey=f"hotkey-{uid}",
        solution=SolutionResponse(
            problem_id="p", code="x = 1", latency_ms=latency_ms
        ),
        verification=Verification(
            problem_id="p",
            num_tests=1,
            num_passed=1 if passed else 0,
            all_passed=passed,
            reward=1.0 if passed else 0.0,
        ),
        reward=1.0 if passed else 0.0,
    )


# --------------------------------------------------------------------------- #
# Peer independence
# --------------------------------------------------------------------------- #
def test_peer_pass_rate_does_not_change_a_correct_miners_payment():
    """The same correct response pays the same whether peers passed or failed."""
    alone_correct = compute_payments(
        [outcome(1, passed=True), outcome(2, passed=False), outcome(3, passed=False)]
    )
    all_correct = compute_payments(
        [outcome(1, passed=True), outcome(2, passed=True), outcome(3, passed=True)]
    )

    assert alone_correct[1] == all_correct[1], (
        f"peer results changed miner 1's payment: alone={alone_correct[1]} "
        f"vs everyone-correct={all_correct[1]}"
    )


def test_a_universally_solved_problem_pays_in_the_latency_band():
    """Equal-latency passes receive the same full payment."""
    payments = compute_payments(
        [outcome(1, passed=True), outcome(2, passed=True), outcome(3, passed=True)]
    )

    assert payments[1] == payments[2] == payments[3] == 1.0


def test_every_correct_payment_lies_in_the_configured_latency_band():
    payments = compute_payments(
        [
            outcome(1, passed=True, latency_ms=1_000.0),
            outcome(2, passed=True, latency_ms=50_000.0),
            outcome(3, passed=True, latency_ms=3_000_000.0),
            outcome(4, passed=False),
        ]
    )

    for uid in (1, 2, 3):
        assert 0.95 <= payments[uid] <= 1.0, f"uid {uid}: {payments[uid]}"
    assert payments[4] == 0.0


# --------------------------------------------------------------------------- #
# Latency adjustment
# --------------------------------------------------------------------------- #
def test_the_fastest_correct_response_pays_exactly_one():
    payments = compute_payments(
        [
            outcome(1, passed=True, latency_ms=10_000.0),
            outcome(2, passed=True, latency_ms=90_000.0),
        ]
    )

    assert payments[1] == 1.0


def test_one_half_life_of_delay_halves_the_share_above_the_floor():
    """180s behind the fastest: 0.95 + 0.5 * 0.05 = 0.975 exactly.

    This verifies the configured latency adjustment directly.
    """
    payments = compute_payments(
        [
            outcome(1, passed=True, latency_ms=20_000.0),
            outcome(2, passed=True, latency_ms=200_000.0),
        ]
    )

    assert payments[2] == 0.95 + 0.5 * (1.0 - 0.95)


# --------------------------------------------------------------------------- #
# Zeros and controls
# --------------------------------------------------------------------------- #
def test_failed_outcomes_pay_zero():
    payments = compute_payments(
        [
            outcome(1, passed=False),                       # wrong answer
            outcome(2, passed=False, latency_ms=0.0),       # no usable latency
            outcome(3, passed=True),
        ]
    )

    assert payments[1] == 0.0
    assert payments[2] == 0.0
    assert payments[3] > 0.0


def test_a_lone_correct_responder_pays_full():
    """A pool containing one correct response pays 1.0."""
    assert compute_payments([outcome(1, passed=True)])[1] == 1.0


def test_eval_engine_normalizes_averaged_observations():
    """Averages of 1.0 and 0.5 produce weights in a 2:1 ratio."""
    e = EvalEngine(num_uids=4, window_seconds=57_600.0, max_samples=200, min_samples=4)
    for i in range(4):
        e.update({0: 1.0}, observed_at=1_000.0 + i, dispatched={0})
        e.update({1: 1.0 if i % 2 == 0 else 0.0}, observed_at=1_000.0 + i, dispatched={1})

    weights = e.get_weights(n=4)

    assert abs(weights[0] / weights[1] - 2.0) < 1e-9


# --------------------------------------------------------------------------- #
# Configuration compatibility
# --------------------------------------------------------------------------- #
def test_difficulty_floor_is_not_an_active_setting():
    """The obsolete difficulty setting is not exposed in active configuration."""
    assert not hasattr(Settings(_env_file=None), "payment_difficulty_floor")


def test_a_stale_difficulty_floor_env_entry_is_ignored_not_fatal(monkeypatch):
    """Operators with PAYMENT_DIFFICULTY_FLOOR in .env must not break."""
    monkeypatch.setenv("PAYMENT_DIFFICULTY_FLOOR", "0.25")
    settings = Settings(_env_file=None)

    assert settings.solve_deadline_s > 0  # loaded fine; entry ignored


def test_compute_payments_has_no_difficulty_floor_parameter():
    """The payment API does not expose the obsolete difficulty setting."""
    assert "difficulty_floor" not in inspect.signature(compute_payments).parameters
