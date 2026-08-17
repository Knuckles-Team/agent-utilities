"""Audio ingestion via the governed sidecar delegate (CONCEPT:AU-KG.ingest.media-sidecar-delegation).

GOC-07 (audio/video modality stack) wires the adapter the W4.6 ADR
(``reports/wave4/ADR-media-sidecar.md``) declared as a design stub: the
``sidecar_contract.SIDECAR_CAPABILITIES['audio']`` manifest entry (default
provider ``audio-transcriber-mcp``) already existed with
``produces={"AudioSegment"}`` but no adapter module. Same reusable delegate
loop as ``pdf_sidecar.py``/``image_sidecar.py``
(:func:`~agent_utilities.media.sidecar_delegate.delegate_extract`) — this
module owns only the audio-specific write-back mapping. Writes:

* the raw audio bytes as the artifact bundle (opaque blob storage — decode
  bounds/malformed-input rejection is the Rust ``eg-audio`` native runtime's
  job, already proven there (``crates/eg-audio/src/runtime.rs``'s
  ``malformed_or_unsupported_audio_is_rejected``); this module never decodes
  audio itself, it only ships bytes to the sidecar and writes back its typed
  result);
* one ``AudioSegment`` evidence locus per transcribed segment the sidecar
  returns, each carrying the segment's own transcript text as its stored
  bytes (never a bare opaque reference masquerading as resolved text, per
  GOC-07's invariant 3) plus confidence.

A malformed sidecar payload (a segment missing ``start_ms``/``end_ms``, or a
non-dict entry) is skipped rather than guessed at or fabricated — the same
"skip the bad entry, keep the rest" discipline ``image_sidecar.py`` uses for
malformed region dicts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.knowledge_graph.etl import lineage

from .sidecar_contract import SidecarContractError, capability_for, is_capable
from .sidecar_delegate import claim_engine, delegate_extract

logger = logging.getLogger(__name__)

_MODALITY = "audio"


@dataclass(frozen=True)
class AudioSidecarResult:
    """One audio recording's sidecar-delegated ingestion outcome — the
    conformance chain's summary: artifact bundle -> transcript segments ->
    AudioSegment loci -> queryable."""

    available: bool
    audio_id: str
    occurrence_id: str | None
    blob_id: str | None
    claim_id: str | None
    segment_count: int
    segment_evidence_ids: list[str] = field(default_factory=list)
    error: str | None = None


def _segment_bounds(segment: dict[str, Any]) -> tuple[int, int] | None:
    try:
        start_ms = int(segment.get("start_ms"))
        end_ms = int(segment.get("end_ms"))
    except (TypeError, ValueError):
        return None
    if start_ms < 0 or end_ms <= start_ms:
        return None
    return start_ms, end_ms


def ingest_audio_via_sidecar(
    data: bytes,
    *,
    audio_id: str,
    mime_type: str = "audio/wav",
    source: str = "",
    provider: str = "",
    session: Any = None,
    media_store: Any = None,
) -> AudioSidecarResult:
    """Store ``data`` as the audio artifact bundle, delegate transcription to
    the governed sidecar, and through-write one ``AudioSegment`` evidence
    locus per returned segment.

    Never raises — same "best-effort, never raises" contract as
    :func:`~agent_utilities.media.pdf_sidecar.ingest_pdf_via_sidecar`.

    Args:
        data: The audio recording's raw bytes (any container/codec the
            configured sidecar accepts — this function never decodes audio
            itself).
        audio_id: The owning recording's id — becomes the ``AudioSegment``
            loci's ``audio_id`` and the ``:SourceObject`` key.
        mime_type: The artifact's real MIME type, forwarded to both
            ``store_media`` and the sidecar delegation as ``media_type``.
        source: Provenance source tag forwarded to ``store_media``.
        provider: Optional non-default sidecar (see
            ``sidecar_contract.SIDECAR_CAPABILITIES``).
        session: Optional explicit ``GraphSession`` forwarded to every
            ``MediaStore`` write.
        media_store: Optional injected ``MediaStore`` (tests inject a fake;
            defaults to ``native_ingest.media_store()``).
    """
    from ..knowledge_graph.memory import native_ingest

    store = media_store if media_store is not None else native_ingest.media_store()

    bundle = store.store_media(
        data,
        media_type=mime_type,
        mime_type=mime_type,
        name=audio_id,
        source=source,
        session=session,
    )
    if bundle is None:
        return AudioSidecarResult(
            available=False,
            audio_id=audio_id,
            occurrence_id=None,
            blob_id=None,
            claim_id=None,
            segment_count=0,
            error="failed to store the audio artifact bundle",
        )

    try:
        capability = capability_for(_MODALITY, provider=provider)
    except SidecarContractError as exc:
        return AudioSidecarResult(
            available=False,
            audio_id=audio_id,
            occurrence_id=bundle.occurrence_id,
            blob_id=bundle.blob_id,
            claim_id=None,
            segment_count=0,
            error=str(exc),
        )

    result = delegate_extract(
        data,
        digest=bundle.digest,
        media_type=mime_type,
        modality=_MODALITY,
        provider=provider,
    )

    if not result.available:
        return AudioSidecarResult(
            available=False,
            audio_id=audio_id,
            occurrence_id=bundle.occurrence_id,
            blob_id=bundle.blob_id,
            claim_id=None,
            segment_count=0,
            error=result.error,
        )

    segments = result.raw.get("segments")
    segments = segments if isinstance(segments, list) else []

    claim_id: str | None = None
    if segments:
        try:
            claim_id = lineage.record_media_sidecar_claim(
                claim_engine(),
                sidecar=result.provider,
                modality=_MODALITY,
                artifact_id=audio_id,
                summary=(
                    f"{result.provider} transcribed {len(segments)} "
                    f"segment(s) from {audio_id}"
                ),
                activity_id=result.activity_id,
            )
        except Exception as exc:  # noqa: BLE001 - provenance is best-effort
            logger.debug(
                "ingest_audio_via_sidecar: claim record failed for %s: %s",
                audio_id,
                exc,
            )

    segment_evidence_ids: list[str] = []
    can_audio_segment = is_capable(capability, "AudioSegment")

    for segment in segments:
        if not can_audio_segment or not isinstance(segment, dict):
            continue
        bounds = _segment_bounds(segment)
        if bounds is None:
            logger.debug(
                "ingest_audio_via_sidecar: skipping segment with missing/"
                "invalid start_ms/end_ms for %s: %r",
                audio_id,
                segment,
            )
            continue
        start_ms, end_ms = bounds
        text = str(segment.get("text") or "")
        try:
            confidence = float(segment.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        segment_evidence = store.store_audio_segment_evidence(
            text.encode("utf-8"),
            audio_id=audio_id,
            start_ms=start_ms,
            end_ms=end_ms,
            mime_type="text/plain",
            source=result.provider,
            claim_id=claim_id,
            confidence=confidence,
            session=session,
        )
        if segment_evidence is not None:
            segment_evidence_ids.append(segment_evidence.evidence_id)

    return AudioSidecarResult(
        available=True,
        audio_id=audio_id,
        occurrence_id=bundle.occurrence_id,
        blob_id=bundle.blob_id,
        claim_id=claim_id,
        segment_count=len(segments),
        segment_evidence_ids=segment_evidence_ids,
    )
