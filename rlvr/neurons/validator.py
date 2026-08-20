"""ValidatorNeuron — block-interval run loop with injected V1 callbacks.

The validator owns the chain identity and run cadence. The V1 evaluation and
weight-submission functions are injected to keep this chain loop independent of
the problem-server and scoring modules.

Callbacks:
  * ``round_callback`` — ``async (validator) -> dict[int, float]`` returning
    {uid: reward} for the round. Set via the constructor or :meth:`set_round_callback`.
  * ``weight_setter`` — computes and submits current validator-local weights.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Awaitable, Callable, Optional

from ..config import Settings
from ..policy import RELEASE_POLICY, ValidatorPolicy
from ..types import NeuronType
from .base import BaseNeuron

# Type of the injected orchestrator round callback.
RoundCallback = Callable[["ValidatorNeuron"], Awaitable[dict[int, float]]]
WeightSetter = Callable[["ValidatorNeuron"], None]
_CURRENT_BLOCK_MAX_ATTEMPTS = 3
_CURRENT_BLOCK_RETRY_DELAY_S = 1.0


class ValidatorNeuron(BaseNeuron):
    """Validator skeleton: chain identity + run loop + weight setting.

    The heavy lifting is delegated to an injected ``round_callback`` so the
    orchestrator and this neuron stay decoupled (no import cycle).
    """

    neuron_type = NeuronType.VALIDATOR

    def __init__(
        self,
        settings: Optional[Settings] = None,
        round_callback: Optional[RoundCallback] = None,
        weight_setter: Optional[WeightSetter] = None,
        policy: ValidatorPolicy = RELEASE_POLICY,
    ):
        super().__init__(settings)
        self.policy = policy
        self._round_callback: Optional[RoundCallback] = round_callback
        self._weight_setter: Optional[WeightSetter] = weight_setter

        # How often (in blocks) to run a round / set weights when live.
        self.round_interval_blocks: int = policy.round_interval_blocks
        self.weights_interval_blocks: int = policy.weights_interval_blocks
        self.chain_weights_rate_limit: Optional[int] = None
        self._rounds_deferred_until: Optional[float] = None

        self._should_exit = False

    # ------------------------------------------------------------------ #
    # Injection seams
    # ------------------------------------------------------------------ #
    def set_round_callback(self, callback: RoundCallback) -> None:
        """Inject the orchestrator's async round function.

        Signature: ``async (validator: ValidatorNeuron) -> dict[int, float]``.
        """
        self._round_callback = callback

    def set_weight_setter(
        self, setter: WeightSetter
    ) -> None:
        """Inject the validator-local weight computation/submission function."""
        self._weight_setter = setter

    @property
    def rounds_deferred_until(self) -> Optional[float]:
        """Current monotonic lease deadline, if server pacing is active."""
        return self._rounds_deferred_until

    def defer_rounds_until(self, monotonic_deadline: float) -> None:
        """Extend server-directed round pacing without touching block gates."""
        try:
            deadline = float(monotonic_deadline)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("lease deferral deadline must be finite") from error
        if not math.isfinite(deadline):
            raise ValueError("lease deferral deadline must be finite")
        if (
            self._rounds_deferred_until is None
            or deadline > self._rounds_deferred_until
        ):
            self._rounds_deferred_until = deadline

    # ------------------------------------------------------------------ #
    # Round driver
    # ------------------------------------------------------------------ #
    async def run_round(self) -> dict[int, float]:
        """Run one orchestrated round via the injected callback.

        Returns {uid: reward}. Raises if no callback has been injected — this is
        the explicit seam the live orchestrator must fill.
        """
        if self._round_callback is None:
            raise RuntimeError(
                "No round_callback injected. The orchestrator must call "
                "validator.set_round_callback(...) before run_round(). "
                "(This is the orchestrator seam; orchestrator is wired separately.)"
            )
        return await self._round_callback(self)

    # ------------------------------------------------------------------ #
    # Weight setting hook
    # ------------------------------------------------------------------ #
    def submit_weights(self) -> None:
        """Compute and submit weights through the configured V1 callback."""
        if self._weight_setter is None:
            raise RuntimeError("No weight_setter configured")
        self._weight_setter(self)

    # ------------------------------------------------------------------ #
    # Run loop
    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        """Signal the run loop to exit after the current iteration."""
        self._should_exit = True

    async def _current_block_with_recovery(self) -> Optional[int]:
        """Bound transient current-block RPC failures without ending the loop."""
        for attempt in range(1, _CURRENT_BLOCK_MAX_ATTEMPTS + 1):
            try:
                return int(self.current_block())
            except Exception as exc:  # noqa: BLE001 - chain RPCs vary by SDK
                print(
                    "[validator] WARN: current-block RPC failed "
                    f"(attempt {attempt}/{_CURRENT_BLOCK_MAX_ATTEMPTS}: {exc})"
                )
                if attempt < _CURRENT_BLOCK_MAX_ATTEMPTS:
                    await asyncio.sleep(_CURRENT_BLOCK_RETRY_DELAY_S)
        return None

    async def run(self, max_rounds: Optional[int] = None) -> None:
        """Block-interval run loop (live).

        Polls the current block; on each ``round_interval_blocks`` boundary it
        runs an orchestrated round, and on each ``weights_interval_blocks``
        boundary it sets weights. ``max_rounds`` bounds the loop for tests/sim.
        """
        if self.subtensor is None:
            # Not set up for chain — the orchestrator/sim should drive run_round
            # directly. Surface a clear error rather than silently no-op.
            raise RuntimeError(
                "run() needs setup_bittensor(); for offline use call run_round() directly."
            )

        rounds_done = 0
        weights_gate = BaseNeuron(self.settings)  # separate interval bookkeeping
        while not self._should_exit:
            block = await self._current_block_with_recovery()
            if block is None:
                await asyncio.sleep(1.0)
                continue
            now = time.monotonic()
            deadline = self._rounds_deferred_until
            deferred = deadline is not None and now < deadline
            retry_due = deadline is not None and not deferred
            if not deferred and (
                retry_due
                or self.should_run(block, self.round_interval_blocks)
            ):
                if retry_due:
                    # Clear before the callback so a repeated pacing response
                    # can install a new deadline without being wiped afterward.
                    self._rounds_deferred_until = None
                    # The retry trigger short-circuits should_run(), so anchor
                    # the ordinary cadence explicitly at the retry block.
                    self._last_run_block = block
                await self.run_round()
                rounds_done += 1
                if max_rounds is not None and rounds_done >= max_rounds:
                    break
            if rounds_done and weights_gate.should_run(block, self.weights_interval_blocks):
                # The callback owns the EvalEngine. Gate on rounds_done so we
                # never submit the initial all-zero score vector.
                if self._weight_setter is not None:
                    # Chain inclusion can take a block. Keep that synchronous SDK
                    # wait off the event loop so unrelated async work is not blocked.
                    await asyncio.to_thread(self.submit_weights)
            await asyncio.sleep(1.0)
