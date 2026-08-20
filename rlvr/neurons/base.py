"""BaseNeuron — shared validator-side chain plumbing.

Bittensor is imported **lazily** (inside ``setup_bittensor``), so this module —
and everything that imports it — loads fine without the chain stack. The wallet,
subtensor, and metagraph are constructed from :class:`rlvr.config.Settings`.

A small block-interval helper (:meth:`should_run`) mirrors BitMind's
``on_block_interval`` pattern: the validator's run loop polls the current block
and fires work every ``interval`` blocks.
"""

from __future__ import annotations

from typing import Any, Optional

from ..config import Settings, get_settings
from ..types import NeuronType


def _import_bittensor() -> Any:
    """Lazy, guarded import of bittensor. Raises a clear error if missing."""
    try:
        import bittensor as bt  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "bittensor is not installed. Live chain operations are unavailable; "
            "run in simulation mode instead. Install the 'bittensor' extra to go live."
        ) from e
    return bt


class BaseNeuron:
    """Common wallet/subtensor/metagraph setup, driven by Settings.

    Construction is cheap and chain-free: nothing touches bittensor until
    :meth:`setup_bittensor` is called. This lets the simulation and tests build
    neurons (and call their pure helpers) with no chain stack present.
    """

    neuron_type: NeuronType = NeuronType.VALIDATOR

    def __init__(self, settings: Optional[Settings] = None):
        self.settings: Settings = settings or get_settings()

        # Chain handles — populated by setup_bittensor() when running live.
        self.bt: Any = None
        self.wallet: Any = None
        self.subtensor: Any = None
        self.metagraph: Any = None
        self.axon: Any = None
        self.uid: Optional[int] = None

        # Block-interval bookkeeping. None = never fired (first check fires).
        self._last_run_block: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Live chain setup (lazy bittensor import)
    # ------------------------------------------------------------------ #
    def setup_bittensor(self) -> None:
        """Construct wallet / subtensor / metagraph from Settings.

        Only call this when running on-chain. Imports bittensor lazily so the
        module stays importable without it.
        """
        bt = _import_bittensor()
        self.bt = bt

        s = self.settings
        self.wallet = bt.Wallet(name=s.wallet_name, hotkey=s.wallet_hotkey)

        network = s.subtensor_chain_endpoint or s.subtensor_network
        self.subtensor = bt.Subtensor(network=network)
        self.metagraph = self.subtensor.metagraph(s.netuid)

        self.check_registered()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

    def check_registered(self) -> None:
        """Verify the hotkey is registered on ``netuid``; raise if not."""
        if self.subtensor is None or self.wallet is None:
            raise RuntimeError("setup_bittensor() must run before check_registered()")
        registered = self.subtensor.is_hotkey_registered(
            netuid=self.settings.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        )
        if not registered:
            raise RuntimeError(
                f"Hotkey {self.wallet.hotkey.ss58_address} is not registered on "
                f"netuid {self.settings.netuid}. Register with "
                "`btcli subnets register` before running live."
            )

    @property
    def hotkey_address(self) -> str:
        """ss58 address of this neuron's hotkey, or '' if not set up."""
        if self.wallet is None:
            return ""
        try:
            return str(self.wallet.hotkey.ss58_address)
        except Exception:  # noqa: BLE001
            return ""

    def current_block(self) -> int:
        """Current chain block. Requires a live subtensor."""
        if self.subtensor is None:
            raise RuntimeError("setup_bittensor() must run before current_block()")
        return int(self.subtensor.get_current_block())

    def sync_metagraph(self) -> None:
        """Resync the metagraph against the chain. Requires live setup."""
        if self.metagraph is None or self.subtensor is None:
            raise RuntimeError("setup_bittensor() must run before sync_metagraph()")
        self.metagraph.sync(subtensor=self.subtensor)

    # ------------------------------------------------------------------ #
    # Block-interval helper (pure; no chain needed to evaluate)
    # ------------------------------------------------------------------ #
    def should_run(self, block: int, interval: int) -> bool:
        """Return True when at least ``interval`` blocks have elapsed since the
        last firing (and on the first check). Stateful.

        Deliberately NOT a modulo-boundary check: the run loop polls between
        rounds that can span many blocks, so an exact ``block % interval == 0``
        boundary is routinely skipped over and would never fire.
        """
        if interval <= 0:
            return True
        if self._last_run_block is None or block - self._last_run_block >= interval:
            self._last_run_block = block
            return True
        return False
