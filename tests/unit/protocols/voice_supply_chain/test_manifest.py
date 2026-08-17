"""Tests for the voice-model supply-chain manifest schemas (GOC-36)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_utilities.protocols.voice_supply_chain.manifest import (
    VoiceLicenseDecision,
    VoiceManifestStatus,
    VoiceModelManifest,
    VoiceModelProvider,
)

_VALID_REVISION = "a" * 40
_VALID_SHA = "b" * 64


def _manifest(**overrides: object) -> VoiceModelManifest:
    fields: dict[str, object] = {
        "manifest_id": _VALID_SHA,
        "source_host": "huggingface.co",
        "source_repository": "rhasspy/piper-voices",
        "source_revision": _VALID_REVISION,
        "source_path": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "source_url": "https://huggingface.co/rhasspy/piper-voices/resolve/"
        f"{_VALID_REVISION}/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "byte_length": 1024,
        "sha256": _VALID_SHA,
    }
    fields.update(overrides)
    return VoiceModelManifest(**fields)


def test_manifest_defaults_to_piper_and_quarantined() -> None:
    m = _manifest()
    assert m.provider == VoiceModelProvider.PIPER
    assert m.status == VoiceManifestStatus.QUARANTINED
    assert m.format == "onnx"


def test_manifest_rejects_mutable_revision() -> None:
    """A branch/tag name ('main') must never pass as a pinned revision — DEF-017/lane
    doc "Authority and invariants": a source pin must be immutable."""
    with pytest.raises(ValidationError, match="commit SHA"):
        _manifest(source_revision="main")


def test_manifest_rejects_short_revision() -> None:
    with pytest.raises(ValidationError, match="commit SHA"):
        _manifest(source_revision="a" * 39)


def test_manifest_normalizes_sha256_case() -> None:
    m = _manifest(sha256="B" * 64, manifest_id="B" * 64)
    assert m.sha256 == "b" * 64


def test_manifest_rejects_malformed_sha256() -> None:
    with pytest.raises(ValidationError, match="sha256"):
        _manifest(sha256="not-hex")


def test_license_decision_defaults_pending_never_approved() -> None:
    """A license decision must never default to approved — absence of an explicit
    reviewer decision is a blocking gap, not an implicit allow (fail-closed)."""
    decision = VoiceLicenseDecision(asset_manifest_id=_VALID_SHA)
    assert decision.counsel_decision == "pending"
    assert decision.reviewer == ""


def test_license_decision_rejects_unknown_counsel_value() -> None:
    with pytest.raises(ValidationError):
        VoiceLicenseDecision(asset_manifest_id=_VALID_SHA, counsel_decision="maybe")
