"""Bounded recovery for transient current-block RPC failures."""

from rlvr.config import Settings
from rlvr.neurons.validator import ValidatorNeuron


async def test_current_block_recovers_before_the_attempt_bound(monkeypatch):
    neuron = ValidatorNeuron(Settings(_env_file=None))
    outcomes = iter([RuntimeError("rpc down"), RuntimeError("rpc down"), 1234])
    neuron.current_block = lambda: (
        (_ for _ in ()).throw(value) if isinstance(value := next(outcomes), Exception)
        else value
    )
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("rlvr.neurons.validator.asyncio.sleep", fake_sleep)

    assert await neuron._current_block_with_recovery() == 1234
    assert len(sleeps) == 2


async def test_current_block_stops_after_three_failures(monkeypatch):
    neuron = ValidatorNeuron(Settings(_env_file=None))
    attempts = 0

    def fail():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("rpc down")

    neuron.current_block = fail

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("rlvr.neurons.validator.asyncio.sleep", no_sleep)

    assert await neuron._current_block_with_recovery() is None
    assert attempts == 3
