"""Release policy cannot be changed by an operator's persistent environment."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rlvr.config import (
    Settings,
    ignored_release_policy_keys,
    nondefault_settings_summary,
    release_policy_summary,
)
from rlvr.neurons.decentralized import _dispatch_subset_size
from rlvr.neurons.validator import ValidatorNeuron
from rlvr.policy import RELEASE_POLICY, RELEASE_POLICY_ENV_KEYS


def test_policy_is_frozen():
    with pytest.raises(FrozenInstanceError):
        RELEASE_POLICY.dispatch_fraction = 1.0  # type: ignore[misc]


def test_policy_keys_are_not_settings_fields():
    assert RELEASE_POLICY_ENV_KEYS.isdisjoint(Settings.model_fields)


def test_legacy_environment_cannot_restore_full_pool(monkeypatch):
    monkeypatch.setenv("DISPATCH_SUBSET_K", "0")
    monkeypatch.setenv("DISPATCH_SUBSET_FRACTION", "1")

    Settings(_env_file=None)
    sample = _dispatch_subset_size(
        246,
        required=0,
        fraction=RELEASE_POLICY.dispatch_fraction,
    )

    assert sample == 123


def test_legacy_env_file_keys_are_reported_without_values(tmp_path, monkeypatch):
    monkeypatch.delenv("DISPATCH_SUBSET_K", raising=False)
    path = tmp_path / ".env"
    secret_value = "do-not-print-this-value"
    path.write_text(
        f"DISPATCH_SUBSET_K=0\nROUND_INTERVAL_BLOCKS={secret_value}\n",
        encoding="utf-8",
    )

    found = ignored_release_policy_keys(str(path))

    assert found == ["DISPATCH_SUBSET_K", "ROUND_INTERVAL_BLOCKS"]
    assert secret_value not in repr(found)


def test_startup_summary_is_stable_and_nonsecret():
    summary = release_policy_summary()

    assert f"version={RELEASE_POLICY.version}" in summary
    assert f"hash={RELEASE_POLICY.fingerprint}" in summary
    assert "dispatch_fraction=0.5" in summary
    assert "score_samples=200" in summary
    assert summary.endswith("owner_burn=0")


def test_machine_summary_is_allowlisted_and_omits_identity():
    settings = Settings(
        _env_file=None,
        validator_send_concurrency=7,
        wallet_name="private-wallet-name",
        problem_server_url="https://private.example",
    )

    summary = nondefault_settings_summary(settings)

    assert summary == "VALIDATOR_SEND_CONCURRENCY=7"
    assert "private-wallet-name" not in summary
    assert "private.example" not in summary


def test_legacy_cadence_environment_cannot_change_neuron(monkeypatch):
    monkeypatch.setenv("ROUND_INTERVAL_BLOCKS", "1")
    monkeypatch.setenv("WEIGHTS_INTERVAL_BLOCKS", "999")

    validator = ValidatorNeuron(Settings(_env_file=None))

    assert validator.round_interval_blocks == RELEASE_POLICY.round_interval_blocks
    assert validator.weights_interval_blocks == RELEASE_POLICY.weights_interval_blocks


# --------------------------------------------------------------------------- #
# Rust grading pins (docs/RUST_CHALLENGES.md): everything that changes a
# verdict is release policy, never operator environment.
# --------------------------------------------------------------------------- #
def test_rust_grading_is_release_pinned_not_operator_configured():
    policy = RELEASE_POLICY

    assert policy.rust_judge_version == "rust-exact-token-v1"
    # "Digest-pinned image" means a digest, not a mutable tag.
    assert "@sha256:" in policy.rust_image
    assert isinstance(policy.rustc_version, str) and policy.rustc_version
    assert policy.rust_edition in {"2021", "2024"}
    assert isinstance(policy.rustc_flags, tuple)
    assert policy.rust_compile_timeout_s > 0
    assert policy.rust_artifact_max_bytes > 0
    # The reveal reader must admit the agreed rust cap regardless of env.
    assert policy.problem_response_read_bytes >= 2 * 1024 * 1024

    # No Settings field may shadow any of it: a rust knob in operator env is
    # a consensus split waiting for a toolchain skew.
    assert not any("rust" in name for name in Settings.model_fields)


def test_rust_executor_draws_from_release_policy(monkeypatch):
    """The executor's image, compile deadline, and artifact cap come from
    RELEASE_POLICY — not from settings attributes, not from inline literals."""
    from rlvr.execution import rust_docker_executor as rustmod

    monkeypatch.setattr(
        rustmod.RustDockerExecutor,
        "_resolve_docker",
        staticmethod(lambda: "docker"),
    )

    executor = rustmod.RustDockerExecutor(Settings(_env_file=None))

    assert executor.image == RELEASE_POLICY.rust_image
    assert executor.compile_timeout_s == RELEASE_POLICY.rust_compile_timeout_s
    assert executor.artifact_max_bytes == RELEASE_POLICY.rust_artifact_max_bytes

def test_rust_verify_concurrency_is_bounded_by_host_memory(monkeypatch):
    from rlvr.neurons import decentralized as dec

    gib = 1024**3
    settings = Settings(_env_file=None, validator_verify_concurrency=16)

    monkeypatch.setattr(dec, "_host_memory_bytes", lambda: 16 * gib, raising=False)
    slots = dec._rust_verify_concurrency(settings)
    assert 1 <= slots <= 16
    assert slots * 2 * gib <= 14 * gib

    monkeypatch.setattr(dec, "_host_memory_bytes", lambda: 4 * gib, raising=False)
    assert dec._rust_verify_concurrency(settings) == 1

    monkeypatch.setattr(dec, "_host_memory_bytes", lambda: 64 * gib, raising=False)
    assert dec._rust_verify_concurrency(settings) == 16
