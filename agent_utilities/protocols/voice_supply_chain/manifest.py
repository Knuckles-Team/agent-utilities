"""Voice-model supply-chain manifest schemas (GOC-36 — voice model supply chain qualification).

Typed, versioned records for the Piper voice-model acquisition path this package owns.
These mirror the ``voice.model.manifest.v1`` / ``voice.config.manifest.v1`` /
``voice.license.decision.v1`` contracts defined in the lane doc
(``plans/graph-os-completion-program/lanes/GOC-36-voice-model-supply-chain-qualification.md``
§"Schemas and APIs") — the AU-side subset this repo is authorized to own
(``agent-utilities governed voice-model registry/acquisition adapter and evidence consumer
only`` per the lane contract table).

**Authority note (read before wiring a caller):** per the lane doc's "Authority and
invariants", *the EG registry is the promotion/revocation authority; the external HF
repository is the source record only.* No such EG registry exists yet (confirmed by
reading ``crates/eg-audio``: its ``ingress``/``asr``/``tts`` modules are frozen
contract-and-validation code with no I/O, no `epistemic-graph-voice-worker`, and their own
doc comments say model acquisition/licensing/manifests are GOC-36's scope, not theirs).
So :data:`VoiceManifestStatus` in this module intentionally stops at ``VERIFIED`` — it
never reaches ``APPROVED``/``STAGED``, because this repo has no authority to grant worker
trust. A manifest this module writes is quarantine-and-verification evidence staged for
handoff to that registry once it exists, never a promotion.

Scope guard (DEF-017): the only supported ``provider`` is ``piper`` — an exact
ONNX-graph-plus-JSON-config pair loaded by a Piper-compatible runtime (``piper-rs``).
This is NOT a generic Hugging Face / Transformers model loader; any other asset shape is
rejected as ``unsupported_format`` by :mod:`.acquisition`, never heuristically converted.

Out of scope (DEF-018): nothing in this module models speaker diarization or
voice-biometric identity — the lane's manifests describe an acquired TTS/ASR MODEL
ASSET, never a person.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

#: Exact 40-hex-char commit SHA length Hugging Face uses for a pinned revision. A
#: branch/tag name ("main", "v1") is mutable and is rejected — see
#: :meth:`PinnedVoiceSource.validate` in ``acquisition.py``.
HF_COMMIT_SHA_LEN = 40


class VoiceModelProvider(enum.StrEnum):
    """The only acquisition provider this lane supports (DEF-017 scope guard)."""

    PIPER = "piper"


class VoiceManifestStatus(enum.StrEnum):
    """Local (AU-side) lifecycle of an acquired asset.

    Stops at ``VERIFIED`` — see the module docstring's Authority note. ``REJECTED`` is
    terminal and never reused for a corrected re-acquisition (a corrected asset gets a
    new ``manifest_id``/digest per the lane doc's "Promotion is monotonic... re-promoting
    a corrected artifact creates a new manifest/digest, never mutates history").
    """

    QUARANTINED = "quarantined"
    VERIFIED = "verified"
    REJECTED = "rejected"


class VoiceModelManifest(BaseModel):
    """``voice.model.manifest.v1`` (AU acquisition-side subset).

    One immutable record per acquired Piper ONNX model file, content-addressed by
    ``sha256``. Two acquisitions of the same ``(source_repository, source_revision,
    source_path)`` with the same digest are idempotent (same ``manifest_id``); a digest
    mismatch against a previously recorded manifest for that same source coordinate is a
    hard error, never silently overwritten (append-only per the lane doc).
    """

    manifest_id: str
    provider: VoiceModelProvider = VoiceModelProvider.PIPER
    format: Literal["onnx"] = "onnx"

    source_host: str
    source_repository: str
    source_revision: str
    source_path: str
    source_url: str

    byte_length: int = Field(ge=0)
    sha256: str

    config_manifest_id: str | None = None
    config_sha256: str | None = None

    language: str = ""
    voice: str = ""
    speaker_count: int = Field(default=1, ge=1)

    model_license_ref: str = ""
    dataset_license_ref: str = ""
    voice_consent_ref: str = ""

    status: VoiceManifestStatus = VoiceManifestStatus.QUARANTINED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notes: str = ""

    @field_validator("sha256", "config_sha256")
    @classmethod
    def _validate_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value.lower()

    @field_validator("source_revision")
    @classmethod
    def _validate_revision(cls, value: str) -> str:
        if len(value) != HF_COMMIT_SHA_LEN or any(
            c not in "0123456789abcdef" for c in value.lower()
        ):
            raise ValueError(
                "source_revision must be a 40-char Hugging Face commit SHA, not a "
                "mutable branch/tag name — see the lane doc's Authority and invariants"
            )
        return value.lower()


class VoiceConfigManifest(BaseModel):
    """``voice.config.manifest.v1`` — the Piper JSON config paired with a model manifest."""

    manifest_id: str
    model_manifest_id: str
    sha256: str
    byte_length: int = Field(ge=0)

    sample_rate: int = Field(gt=0)
    espeak_voice: str = ""
    phoneme_id_map_present: bool = False
    piper_schema_present: bool = False

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value.lower()


class VoiceLicenseDecision(BaseModel):
    """``voice.license.decision.v1``.

    Records the license/consent FACTS an operator observed and an explicit
    counsel/compliance disposition. This module makes no legal determination —
    ``counsel_decision`` defaults to ``"pending"`` and NOTHING in this package promotes a
    manifest past ``VERIFIED`` while it is pending (see
    :func:`.acquisition.is_ready_for_promotion_handoff`). The GPL/static-eSpeak linking
    question the lane doc calls out as an explicit compliance blocker is recorded the same
    way: as a fact plus a pending/approved/blocked decision, never inferred.
    """

    asset_manifest_id: str
    declared_license: str = ""
    spdx_id: str | None = None
    is_gpl_or_copyleft_flagged: bool = False
    counsel_decision: Literal["approved", "blocked", "pending"] = "pending"
    reviewer: str = ""
    decided_at: datetime | None = None
    rationale: str = ""


__all__ = [
    "HF_COMMIT_SHA_LEN",
    "VoiceConfigManifest",
    "VoiceLicenseDecision",
    "VoiceManifestStatus",
    "VoiceModelManifest",
    "VoiceModelProvider",
]
