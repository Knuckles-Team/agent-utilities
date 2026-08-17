"""Video ingestion via the governed sidecar delegate (CONCEPT:AU-KG.ingest.media-sidecar-delegation).

GOC-07 (audio/video modality stack) wires the adapter the W4.6 ADR
(``reports/wave4/ADR-media-sidecar.md``) declared as a design stub: the
``sidecar_contract.SIDECAR_CAPABILITIES['video']`` manifest entry (default
provider ``data-science-mcp``) already existed with
``produces={"VideoShot", "VideoFrameRange"}`` but no adapter module. Same
reusable delegate loop as ``pdf_sidecar.py``/``image_sidecar.py``
(:func:`~agent_utilities.media.sidecar_delegate.delegate_extract`) — this
module owns only the video-specific write-back mapping. Writes:

* the raw video bytes as the artifact bundle (opaque blob storage — container
  parsing/malformed-input rejection is the Rust ``eg-video`` native runtime's
  job, already proven there (``crates/eg-video/src/runtime.rs``'s
  ``malformed_or_metadata_only_container_is_rejected``); this module never
  parses the container itself, it only ships bytes to the sidecar and writes
  back its typed result);
* one ``VideoShot`` evidence locus per detected shot boundary (a wall-clock
  time range, mirroring the Rust ``VideoData.shots`` evidence-address shape);
* one ``VideoFrameRange`` evidence locus per returned keyframe (a decoded-
  frame index range, distinct from the wall-clock shot range per GOC-07's
  invariant 2 — time coordinates and frame coordinates are explicit and
  never conflated).

A malformed sidecar payload (a shot/keyframe missing its required bounds, or
a non-dict entry) is skipped rather than guessed at or fabricated — the same
"skip the bad entry, keep the rest" discipline ``image_sidecar.py`` uses for
malformed region dicts. A keyframe with no returned pixel bytes stores a
compact JSON descriptor instead of fabricating pixels (the same honest-degrade
convention ``image_sidecar.py`` uses for a region with no crop bytes).
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.knowledge_graph.etl import lineage

from .sidecar_contract import SidecarContractError, capability_for, is_capable
from .sidecar_delegate import claim_engine, delegate_extract

logger = logging.getLogger(__name__)

_MODALITY = "video"


@dataclass(frozen=True)
class VideoSidecarResult:
    """One video's sidecar-delegated ingestion outcome — the conformance
    chain's summary: artifact bundle -> shots/keyframes -> VideoShot/
    VideoFrameRange loci -> queryable."""

    available: bool
    video_id: str
    occurrence_id: str | None
    blob_id: str | None
    claim_id: str | None
    shot_count: int
    keyframe_count: int
    shot_evidence_ids: list[str] = field(default_factory=list)
    frame_range_evidence_ids: list[str] = field(default_factory=list)
    error: str | None = None


def _shot_bounds(shot: dict[str, Any]) -> tuple[int, int] | None:
    try:
        start_ms = int(shot.get("start_ms"))
        end_ms = int(shot.get("end_ms"))
    except (TypeError, ValueError):
        return None
    if start_ms < 0 or end_ms <= start_ms:
        return None
    return start_ms, end_ms


def _frame_bounds(keyframe: dict[str, Any]) -> tuple[int, int] | None:
    try:
        start_frame = int(keyframe.get("start_frame", keyframe.get("frame_number")))
        end_frame = int(keyframe.get("end_frame", keyframe.get("frame_number")))
    except (TypeError, ValueError):
        return None
    if start_frame <= 0 or end_frame < start_frame:
        return None
    return start_frame, end_frame


def ingest_video_via_sidecar(
    data: bytes,
    *,
    video_id: str,
    mime_type: str = "video/mp4",
    source: str = "",
    provider: str = "",
    session: Any = None,
    media_store: Any = None,
) -> VideoSidecarResult:
    """Store ``data`` as the video artifact bundle, delegate shot/keyframe
    extraction to the governed sidecar, and through-write ``VideoShot``/
    ``VideoFrameRange`` evidence loci.

    Never raises — same "best-effort, never raises" contract as
    :func:`~agent_utilities.media.pdf_sidecar.ingest_pdf_via_sidecar`.

    Args:
        data: The video's raw bytes (any container/codec the configured
            sidecar accepts — this function never parses the container
            itself).
        video_id: The owning video's id — becomes the ``VideoShot``/
            ``VideoFrameRange`` loci's ``video_id`` and the ``:SourceObject``
            key.
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
        name=video_id,
        source=source,
        session=session,
    )
    if bundle is None:
        return VideoSidecarResult(
            available=False,
            video_id=video_id,
            occurrence_id=None,
            blob_id=None,
            claim_id=None,
            shot_count=0,
            keyframe_count=0,
            error="failed to store the video artifact bundle",
        )

    try:
        capability = capability_for(_MODALITY, provider=provider)
    except SidecarContractError as exc:
        return VideoSidecarResult(
            available=False,
            video_id=video_id,
            occurrence_id=bundle.occurrence_id,
            blob_id=bundle.blob_id,
            claim_id=None,
            shot_count=0,
            keyframe_count=0,
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
        return VideoSidecarResult(
            available=False,
            video_id=video_id,
            occurrence_id=bundle.occurrence_id,
            blob_id=bundle.blob_id,
            claim_id=None,
            shot_count=0,
            keyframe_count=0,
            error=result.error,
        )

    shots = result.raw.get("shots")
    shots = shots if isinstance(shots, list) else []
    keyframes = result.raw.get("keyframes")
    keyframes = keyframes if isinstance(keyframes, list) else []

    claim_id: str | None = None
    if shots or keyframes:
        try:
            claim_id = lineage.record_media_sidecar_claim(
                claim_engine(),
                sidecar=result.provider,
                modality=_MODALITY,
                artifact_id=video_id,
                summary=(
                    f"{result.provider} detected {len(shots)} shot(s) and "
                    f"{len(keyframes)} keyframe(s) in {video_id}"
                ),
                activity_id=result.activity_id,
            )
        except Exception as exc:  # noqa: BLE001 - provenance is best-effort
            logger.debug(
                "ingest_video_via_sidecar: claim record failed for %s: %s",
                video_id,
                exc,
            )

    shot_evidence_ids: list[str] = []
    can_video_shot = is_capable(capability, "VideoShot")
    for shot in shots:
        if not can_video_shot or not isinstance(shot, dict):
            continue
        bounds = _shot_bounds(shot)
        if bounds is None:
            logger.debug(
                "ingest_video_via_sidecar: skipping shot with missing/"
                "invalid start_ms/end_ms for %s: %r",
                video_id,
                shot,
            )
            continue
        start_ms, end_ms = bounds
        label = str(shot.get("label") or "")
        try:
            confidence = float(shot.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        descriptor = json.dumps({"label": label}).encode("utf-8")
        shot_evidence = store.store_video_shot_evidence(
            descriptor,
            video_id=video_id,
            start_ms=start_ms,
            end_ms=end_ms,
            mime_type="application/json",
            source=result.provider,
            claim_id=claim_id,
            confidence=confidence,
            session=session,
        )
        if shot_evidence is not None:
            shot_evidence_ids.append(shot_evidence.evidence_id)

    frame_range_evidence_ids: list[str] = []
    can_frame_range = is_capable(capability, "VideoFrameRange")
    for keyframe in keyframes:
        if not can_frame_range or not isinstance(keyframe, dict):
            continue
        bounds = _frame_bounds(keyframe)
        if bounds is None:
            logger.debug(
                "ingest_video_via_sidecar: skipping keyframe with missing/"
                "invalid frame bounds for %s: %r",
                video_id,
                keyframe,
            )
            continue
        start_frame, end_frame = bounds
        png_b64 = keyframe.get("png_b64")
        frame_bytes = b""
        if isinstance(png_b64, str) and png_b64:
            try:
                frame_bytes = base64.b64decode(png_b64)
            except (ValueError, TypeError):
                frame_bytes = b""
        if frame_bytes:
            frame_mime = "image/png"
        else:
            # Honest degrade (no invented pixels): a compact JSON descriptor
            # of the keyframe, mirroring image_sidecar.py's region-with-no-
            # crop-bytes precedent rather than fabricating pixels the
            # sidecar didn't return.
            frame_bytes = json.dumps(
                {"start_frame": start_frame, "end_frame": end_frame}
            ).encode("utf-8")
            frame_mime = "application/json"
        frame_evidence = store.store_video_frame_range_evidence(
            frame_bytes,
            video_id=video_id,
            start_frame=start_frame,
            end_frame=end_frame,
            mime_type=frame_mime,
            source=result.provider,
            claim_id=claim_id,
            session=session,
        )
        if frame_evidence is not None:
            frame_range_evidence_ids.append(frame_evidence.evidence_id)

    return VideoSidecarResult(
        available=True,
        video_id=video_id,
        occurrence_id=bundle.occurrence_id,
        blob_id=bundle.blob_id,
        claim_id=claim_id,
        shot_count=len(shots),
        keyframe_count=len(keyframes),
        shot_evidence_ids=shot_evidence_ids,
        frame_range_evidence_ids=frame_range_evidence_ids,
    )
