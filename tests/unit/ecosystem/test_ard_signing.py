"""Ed25519 ARD datapoint signing tests (CONCEPT:AU-OS.identity.ard-datapoint-signing)."""

from __future__ import annotations

import pytest

from agent_utilities.security import ard_signing

pytestmark = pytest.mark.skipif(
    not ard_signing.signing_available(), reason="cryptography not installed"
)


def test_sign_verify_roundtrip() -> None:
    dp = {"b": 2, "a": 1, "name": "x"}
    sig = ard_signing.sign_datapoint(dp)
    assert sig
    assert ard_signing.verify_datapoint(dp, sig, ard_signing.public_key_b64())


def test_tampered_datapoint_fails() -> None:
    dp = {"a": 1}
    sig = ard_signing.sign_datapoint(dp)
    assert not ard_signing.verify_datapoint({"a": 2}, sig, ard_signing.public_key_b64())


def test_canonical_is_order_independent() -> None:
    assert ard_signing.canonical({"a": 1, "b": 2}) == ard_signing.canonical(
        {"b": 2, "a": 1}
    )


def test_verify_rejects_bad_key() -> None:
    dp = {"a": 1}
    sig = ard_signing.sign_datapoint(dp)
    assert not ard_signing.verify_datapoint(dp, sig, "not-a-key")


def test_configured_key_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    import base64

    seed = base64.urlsafe_b64encode(b"0" * 32).decode().rstrip("=")
    monkeypatch.setenv("ARD_SIGNING_PRIVATE_KEY", seed)
    pub1 = ard_signing.public_key_b64()
    dp = {"x": 1}
    sig = ard_signing.sign_datapoint(dp)
    assert ard_signing.verify_datapoint(dp, sig, pub1)
    assert ard_signing.is_configured()
