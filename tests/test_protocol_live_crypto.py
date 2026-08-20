"""Real-crypto (live) signing coverage for the epistula protocol.

Unlike tests/test_protocol.py (which pins the chain-free HMAC fallback), these
tests exercise the REAL sr25519 path that the live validator<->miner transport
uses: a bittensor Keypair signs, and verify_signature reconstructs the signer
from its ss58 address and checks the signature. They are skipped when no crypto
stack is importable.

This is the coverage that was missing — it directly catches the two live-path
regressions we hit on bittensor 10.x: (a) Keypair imported from the wrong module
(crypto silently unavailable -> every ss58 request fail-closed), and (b)
sign_message handed a Wallet (no .sign of its own) dropping to HMAC.
"""

from __future__ import annotations

import pytest

from rlvr.protocol import (
    crypto_available,
    sign_message,
    verify_signature,
)

pytestmark = pytest.mark.skipif(
    not crypto_available(), reason="no sr25519 crypto stack installed"
)


def _keypair(uri: str):
    from bittensor_wallet import Keypair

    return Keypair.create_from_uri(uri)


class _WalletLike:
    """Mimics a bittensor Wallet: exposes a `.hotkey` keypair but has no `.sign`
    of its own — the shape sign_message must resolve through."""

    def __init__(self, keypair):
        self.hotkey = keypair


# --------------------------------------------------------------------------- #
# Sign / verify roundtrip with a real Keypair
# --------------------------------------------------------------------------- #
def test_real_sign_verify_roundtrip():
    kp = _keypair("//Alice")
    body = b'{"problem_id":"p1"}'
    headers = sign_message(kp, body, signed_for="miner-hk")
    assert headers["Epistula-Signed-By"] == kp.ss58_address
    assert headers["Epistula-Request-Signature"].startswith("0x")
    assert verify_signature(headers, body, expected_signed_for="miner-hk") is True


def test_real_sign_rejects_tampered_body():
    kp = _keypair("//Bob")
    body = b"original"
    headers = sign_message(kp, body)
    assert verify_signature(headers, b"tampered") is False


def test_real_sign_through_wallet_like_object():
    """A Wallet (no `.sign`) must sign via its hotkey keypair, not drop to HMAC."""
    kp = _keypair("//Charlie")
    body = b"payload"
    headers = sign_message(_WalletLike(kp), body, signed_for="rcpt")
    assert headers["Epistula-Signed-By"] == kp.ss58_address
    # An HMAC fallback would NOT verify against the real ss58 key, so a True here
    # proves a genuine sr25519 signature was produced.
    assert verify_signature(headers, body, expected_signed_for="rcpt") is True


def test_real_verify_rejects_wrong_signer():
    """A signature from Alice must not verify when Signed-By claims Bob."""
    alice, bob = _keypair("//Alice"), _keypair("//Bob")
    body = b"x"
    headers = sign_message(alice, body)
    headers["Epistula-Signed-By"] = bob.ss58_address  # claim a different signer
    assert verify_signature(headers, body) is False
