"""JPEG ingestion via the governed sidecar delegate (CONCEPT:AU-KG.ingest.media-sidecar-delegation).

Same reusable delegate loop as ``pdf_sidecar.py``
(:func:`~agent_utilities.media.sidecar_delegate.delegate_extract`),
targeting ``data-science-mcp``'s image-decode sidecar
(``sidecar_contract.SIDECAR_CAPABILITIES['jpeg']``) for the codec/embedding
work the pure-Rust engine deliberately never does in-process
(``reports/wave4/ADR-media-sidecar.md``'s permanent boundary — JPEG decode
never runs in-engine). Writes:

* the raw JPEG bytes as the artifact bundle (opaque blob storage — a JPEG's
  bytes need no decode to be content-addressed and stored);
* the sidecar's decoded thumbnail as a PNG ``:Rendition`` (re-entering the
  engine's own native codec, per the ADR);
* the perceptual hash (+ optional whole-image embedding) as an extraction
  property on the occurrence, through ``MediaStore.record_extraction`` — the
  EXISTING "async extraction/embedding seam" that method's own docstring
  describes ("the seam an async extraction/embedding pipeline calls back
  into"); no new engine surface;
* one ``ImageRegion`` evidence locus per detected region.
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

_MODALITY = "jpeg"


@dataclass(frozen=True)
class ImageSidecarResult:
    """One JPEG's sidecar-delegated ingestion outcome — the conformance
    chain's summary: artifact bundle -> pHash/regions -> embedding ->
    queryable."""

    available: bool
    image_id: str
    occurrence_id: str | None
    blob_id: str | None
    rendition_id: str | None
    phash: str | None
    claim_id: str | None
    region_evidence_ids: list[str] = field(default_factory=list)
    error: str | None = None


def ingest_jpeg_via_sidecar(
    data: bytes,
    *,
    image_id: str,
    source: str = "",
    provider: str = "",
    session: Any = None,
    media_store: Any = None,
) -> ImageSidecarResult:
    """Store ``data`` as the JPEG artifact bundle, delegate decode to the
    governed sidecar, and through-write its pHash + decoded thumbnail +
    detected-region evidence loci.

    Never raises — same "best-effort, never raises" contract as
    :func:`~agent_utilities.media.pdf_sidecar.ingest_pdf_via_sidecar`.

    Args:
        data: The JPEG's raw bytes.
        image_id: The owning image's id — becomes the ``ImageRegion`` loci's
            ``image_id`` and the ``:SourceObject`` key.
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
        media_type="image/jpeg",
        mime_type="image/jpeg",
        name=image_id,
        source=source,
        session=session,
    )
    if bundle is None:
        return ImageSidecarResult(
            available=False,
            image_id=image_id,
            occurrence_id=None,
            blob_id=None,
            rendition_id=None,
            phash=None,
            claim_id=None,
            error="failed to store the JPEG artifact bundle",
        )

    try:
        capability = capability_for(_MODALITY, provider=provider)
    except SidecarContractError as exc:
        return ImageSidecarResult(
            available=False,
            image_id=image_id,
            occurrence_id=bundle.occurrence_id,
            blob_id=bundle.blob_id,
            rendition_id=None,
            phash=None,
            claim_id=None,
            error=str(exc),
        )

    result = delegate_extract(
        data,
        digest=bundle.digest,
        media_type="image/jpeg",
        modality=_MODALITY,
        provider=provider,
    )

    if not result.available:
        return ImageSidecarResult(
            available=False,
            image_id=image_id,
            occurrence_id=bundle.occurrence_id,
            blob_id=bundle.blob_id,
            rendition_id=None,
            phash=None,
            claim_id=None,
            error=result.error,
        )

    phash_raw = result.raw.get("phash")
    phash = str(phash_raw) if phash_raw else None
    regions = result.raw.get("regions")
    regions = regions if isinstance(regions, list) else []
    embedding = result.raw.get("embedding")
    embedding = list(embedding) if isinstance(embedding, list) else None

    claim_id: str | None = None
    try:
        claim_id = lineage.record_media_sidecar_claim(
            claim_engine(),
            sidecar=result.provider,
            modality=_MODALITY,
            artifact_id=image_id,
            summary=(
                f"{result.provider} decoded image {image_id} "
                f"(phash={phash or 'n/a'}, {len(regions)} region(s))"
            ),
            activity_id=result.activity_id,
        )
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort
        logger.debug(
            "ingest_jpeg_via_sidecar: claim record failed for %s: %s", image_id, exc
        )

    if phash is not None or embedding is not None:
        try:
            store.record_extraction(
                bundle.occurrence_id,
                model=result.provider,
                embedding=embedding,
                extra={"phash": phash} if phash is not None else None,
                session=session,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort, matches MediaStore's own posture
            logger.debug(
                "ingest_jpeg_via_sidecar: record_extraction failed for %s: %s",
                image_id,
                exc,
            )

    rendition_id: str | None = None
    thumbnail_b64 = result.raw.get("thumbnail_png_b64")
    if isinstance(thumbnail_b64, str) and thumbnail_b64:
        try:
            thumbnail_bytes = base64.b64decode(thumbnail_b64)
        except (ValueError, TypeError) as exc:
            logger.debug(
                "ingest_jpeg_via_sidecar: bad thumbnail_png_b64 for %s: %s",
                image_id,
                exc,
            )
            thumbnail_bytes = b""
        if thumbnail_bytes:
            rendition = store.store_rendition(
                thumbnail_bytes,
                source_digest=bundle.digest,
                rendition_type="thumbnail",
                mime_type="image/png",
                occurrence_id=bundle.occurrence_id,
                model=result.provider,
                session=session,
            )
            rendition_id = rendition.rendition_id if rendition is not None else None

    region_evidence_ids: list[str] = []
    if regions and not is_capable(capability, "ImageRegion"):
        regions = []

    for region in regions:
        if not isinstance(region, dict):
            continue
        crop_b64 = region.get("crop_png_b64")
        region_bytes = b""
        if isinstance(crop_b64, str) and crop_b64:
            try:
                region_bytes = base64.b64decode(crop_b64)
            except (ValueError, TypeError):
                region_bytes = b""
        if region_bytes:
            region_mime = "image/png"
        else:
            # Honest degrade (no invented pixels): a compact JSON descriptor
            # of the detected box, mirroring store_metric_window_evidence's
            # own "data = a real, if minimal, serialized descriptor"
            # precedent rather than fabricating crop bytes the sidecar
            # didn't return.
            region_bytes = json.dumps(
                {"label": region.get("label"), "confidence": region.get("confidence")}
            ).encode("utf-8")
            region_mime = "application/json"
        region_evidence = store.store_image_region_evidence(
            region_bytes,
            image_id=image_id,
            x=float(region.get("x") or 0.0),
            y=float(region.get("y") or 0.0),
            width=float(region.get("width") or 0.0),
            height=float(region.get("height") or 0.0),
            mime_type=region_mime,
            source=result.provider,
            claim_id=claim_id,
            session=session,
        )
        if region_evidence is not None:
            region_evidence_ids.append(region_evidence.evidence_id)

    return ImageSidecarResult(
        available=True,
        image_id=image_id,
        occurrence_id=bundle.occurrence_id,
        blob_id=bundle.blob_id,
        rendition_id=rendition_id,
        phash=phash,
        claim_id=claim_id,
        region_evidence_ids=region_evidence_ids,
    )
