"""Voice-model license/consent decision recording (GOC-36).

Implements the lane doc's acceptance gate 3: *"Dataset/voice/model/runtime licenses and
consent records have an approved scope, reviewer, expiry/review date, and
counsel/compliance decision; the GPL/static eSpeak linking question remains blocked
until explicitly resolved."*

This module makes **no legal determination**. It only:

1. Durably records the license facts an operator observed for an acquired asset
   (:func:`record_license_decision`), and
2. Answers whether a manifest is allowed to be handed off toward promotion
   (:func:`is_ready_for_promotion_handoff`) — which is ``False`` whenever a decision is
   absent or still ``"pending"``. There is no code path in this package that treats an
   absent license decision as an implicit approval (the fail-closed discipline this
   repo documents extensively: a degraded/missing read must never grant permission).

The GPL/static-eSpeak-linking question the lane doc calls out explicitly is not special
here — it is recorded the same way as any other license fact, with
``is_gpl_or_copyleft_flagged=True`` and ``counsel_decision="pending"`` until a reviewer
sets it, which keeps it blocked exactly as the lane doc requires without hardcoding a
one-off rule for eSpeak specifically.
"""

from __future__ import annotations

from pathlib import Path

from agent_utilities.protocols.voice_supply_chain.acquisition import manifest_index_dir
from agent_utilities.protocols.voice_supply_chain.manifest import (
    VoiceLicenseDecision,
    VoiceManifestStatus,
    VoiceModelManifest,
)


def _decision_path(asset_manifest_id: str) -> Path:
    return manifest_index_dir() / f"{asset_manifest_id}.license.json"


def record_license_decision(decision: VoiceLicenseDecision) -> VoiceLicenseDecision:
    """Durably record a license/consent decision for an acquired asset.

    Overwrites any prior decision for the same ``asset_manifest_id`` — a license
    decision is a reviewable, updatable record (a reviewer may move it from
    ``pending`` to ``approved``/``blocked``), unlike the asset manifest itself which is
    immutable once quarantined.
    """
    _decision_path(decision.asset_manifest_id).write_text(
        decision.model_dump_json(indent=2)
    )
    return decision


def get_license_decision(asset_manifest_id: str) -> VoiceLicenseDecision | None:
    path = _decision_path(asset_manifest_id)
    if not path.exists():
        return None
    return VoiceLicenseDecision.model_validate_json(path.read_text())


def is_ready_for_promotion_handoff(manifest: VoiceModelManifest) -> tuple[bool, str]:
    """Whether ``manifest`` may be handed off toward the (not-yet-existing) EG registry
    for promotion.

    Returns ``(True, "")`` only when the manifest is ``VERIFIED`` AND an explicit
    ``"approved"`` license decision exists. Every other case returns ``(False,
    "<reason>")`` — never ``(True, ...)`` on missing/ambiguous evidence, matching the
    lane doc's "A license/consent gap ... yields BLOCKED/REJECTED, never a warning-only
    fallback."
    """
    if manifest.status != VoiceManifestStatus.VERIFIED:
        return False, f"manifest status is {manifest.status.value!r}, not 'verified'"
    decision = get_license_decision(manifest.manifest_id)
    if decision is None:
        return False, "no license decision recorded for this manifest"
    if decision.counsel_decision != "approved":
        return (
            False,
            f"license decision is {decision.counsel_decision!r}, not 'approved'",
        )
    if not decision.reviewer:
        return False, "license decision has no recorded reviewer"
    return True, ""


def mark_verified(manifest: VoiceModelManifest) -> VoiceModelManifest:
    """Transition a ``QUARANTINED`` manifest to ``VERIFIED`` after pair/format checks
    passed (see ``acquisition.acquire_voice_config``'s Piper schema validation).

    Never transitions a manifest that is already ``REJECTED`` — rejection is terminal
    (see ``manifest.VoiceManifestStatus``'s docstring).
    """
    if manifest.status == VoiceManifestStatus.REJECTED:
        raise ValueError(
            f"manifest {manifest.manifest_id} is REJECTED (terminal); cannot verify"
        )
    updated = manifest.model_copy(update={"status": VoiceManifestStatus.VERIFIED})
    (manifest_index_dir() / f"{updated.manifest_id}.json").write_text(
        updated.model_dump_json(indent=2)
    )
    return updated


__all__ = [
    "get_license_decision",
    "is_ready_for_promotion_handoff",
    "mark_verified",
    "record_license_decision",
]
