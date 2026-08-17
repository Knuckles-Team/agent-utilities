"""Tests for voice-model license/consent decision recording (GOC-36 acceptance gate 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.protocols.voice_supply_chain import acquisition as acq
from agent_utilities.protocols.voice_supply_chain import license_registry as lr
from agent_utilities.protocols.voice_supply_chain.manifest import (
    VoiceLicenseDecision,
    VoiceManifestStatus,
    VoiceModelManifest,
)

_REVISION = "a" * 40
_SHA = "c" * 64


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acq, "data_dir", lambda: tmp_path)


def _manifest(status: VoiceManifestStatus = VoiceManifestStatus.VERIFIED) -> VoiceModelManifest:
    return VoiceModelManifest(
        manifest_id=_SHA,
        source_host="huggingface.co",
        source_repository="rhasspy/piper-voices",
        source_revision=_REVISION,
        source_path="en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        source_url="https://huggingface.co/x",
        byte_length=1,
        sha256=_SHA,
        status=status,
    )


def test_no_decision_blocks_promotion_handoff() -> None:
    ready, reason = lr.is_ready_for_promotion_handoff(_manifest())
    assert ready is False
    assert "no license decision" in reason


def test_pending_decision_blocks_promotion_handoff() -> None:
    lr.record_license_decision(
        VoiceLicenseDecision(asset_manifest_id=_SHA, declared_license="MIT")
    )
    ready, reason = lr.is_ready_for_promotion_handoff(_manifest())
    assert ready is False
    assert "pending" in reason


def test_approved_decision_without_reviewer_still_blocks() -> None:
    """An 'approved' decision with no recorded reviewer is not evidence of anything —
    never treat it as a real approval."""
    lr.record_license_decision(
        VoiceLicenseDecision(
            asset_manifest_id=_SHA, declared_license="MIT", counsel_decision="approved"
        )
    )
    ready, reason = lr.is_ready_for_promotion_handoff(_manifest())
    assert ready is False
    assert "reviewer" in reason


def test_fully_approved_decision_on_verified_manifest_is_ready() -> None:
    lr.record_license_decision(
        VoiceLicenseDecision(
            asset_manifest_id=_SHA,
            declared_license="MIT",
            counsel_decision="approved",
            reviewer="counsel@example.invalid",
        )
    )
    ready, reason = lr.is_ready_for_promotion_handoff(_manifest())
    assert ready is True
    assert reason == ""


def test_quarantined_manifest_never_ready_even_with_approved_license() -> None:
    """License approval alone is not enough — the asset must also be VERIFIED (pair
    validated). A quarantined-only asset is never ready for promotion handoff."""
    lr.record_license_decision(
        VoiceLicenseDecision(
            asset_manifest_id=_SHA,
            counsel_decision="approved",
            reviewer="counsel@example.invalid",
        )
    )
    ready, reason = lr.is_ready_for_promotion_handoff(
        _manifest(status=VoiceManifestStatus.QUARANTINED)
    )
    assert ready is False
    assert "quarantined" in reason


def test_gpl_flag_recorded_and_still_requires_explicit_approval() -> None:
    """The lane doc's GPL/static-eSpeak blocker is just a fact + pending decision —
    flagging it does not change the fail-closed default."""
    decision = lr.record_license_decision(
        VoiceLicenseDecision(
            asset_manifest_id=_SHA,
            declared_license="GPL-3.0-or-later (eSpeak-ng, statically linked)",
            is_gpl_or_copyleft_flagged=True,
        )
    )
    assert decision.is_gpl_or_copyleft_flagged is True
    ready, _ = lr.is_ready_for_promotion_handoff(_manifest())
    assert ready is False


def test_mark_verified_transitions_status_and_persists() -> None:
    manifest = _manifest(status=VoiceManifestStatus.QUARANTINED)
    (acq.manifest_index_dir() / f"{manifest.manifest_id}.json").write_text(
        manifest.model_dump_json()
    )
    updated = lr.mark_verified(manifest)
    assert updated.status == VoiceManifestStatus.VERIFIED


def test_mark_verified_refuses_rejected_manifest() -> None:
    manifest = _manifest(status=VoiceManifestStatus.REJECTED)
    with pytest.raises(ValueError, match="REJECTED"):
        lr.mark_verified(manifest)
