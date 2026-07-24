"""PDF ingestion via the governed sidecar delegate (CONCEPT:AU-KG.ingest.media-sidecar-delegation).

Ships a PDF's bytes to the fleet OCR sidecar (``stirlingpdf-mcp`` by default —
see ``sidecar_contract.SIDECAR_CAPABILITIES['pdf']``) and writes its
pages/text-spans/OCR-word-boxes back through the EXISTING ``MediaStore``
evidence-spine surfaces (``agent_utilities/knowledge_graph/memory/
media_store.py``) — no new engine code (``reports/wave4/ADR-media-sidecar.md``).
The fleet call itself runs through ONE reusable delegate loop
(:func:`~agent_utilities.media.sidecar_delegate.delegate_extract`); this
module owns only the PDF-specific write-back mapping. ``image_sidecar.py``
is the SAME shape for JPEG.

Volume discipline (matches every other locus producer in this codebase — see
``docs/architecture/evidence_spine_convergence.md``'s ``TableCellRange``/
``PageBox`` "whole known extent, not per-cell/per-shape" convention): ONE
``DocumentSpan`` per PAGE (the full page's text), not per line/sentence — a
large PDF would otherwise mean thousands of writes for a single ingest. OCR
word/line boxes (the sidecar's own per-box granularity) map onto
``ImageRegion``, mirroring ``readers_media.py``'s local RapidOCR producer
exactly (one locus per recognised box) — the sidecar path and the local-OCR
path share the SAME locus-kind convention, just a different ``source`` tag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.knowledge_graph.etl import lineage

from .sidecar_contract import SidecarContractError, capability_for, is_capable
from .sidecar_delegate import claim_engine, delegate_extract

logger = logging.getLogger(__name__)

_MODALITY = "pdf"


@dataclass(frozen=True)
class PdfSidecarResult:
    """One PDF's sidecar-delegated ingestion outcome — the conformance
    chain's summary: artifact bundle -> loci -> embeddings -> queryable."""

    available: bool
    document_id: str
    occurrence_id: str | None
    blob_id: str | None
    claim_id: str | None
    page_count: int
    page_evidence_ids: list[str] = field(default_factory=list)
    span_evidence_ids: list[str] = field(default_factory=list)
    ocr_box_evidence_ids: list[str] = field(default_factory=list)
    error: str | None = None


def ingest_pdf_via_sidecar(
    data: bytes,
    *,
    document_id: str,
    source: str = "",
    provider: str = "",
    session: Any = None,
    media_store: Any = None,
) -> PdfSidecarResult:
    """Store ``data`` as the PDF artifact bundle, delegate OCR/text
    extraction to the governed sidecar, and through-write ``PageBox``/
    ``DocumentSpan``/``ImageRegion`` evidence loci for every page.

    Never raises: a delegation failure, a malformed sidecar response, or an
    underlying ``MediaStore`` write failure all return ``available=False``
    with ``error`` set — the SAME "best-effort, never raises" contract
    ``MediaStore`` itself keeps throughout.

    Args:
        data: The PDF's raw bytes.
        document_id: The owning document's id — becomes the ``PageBox``/
            ``DocumentSpan`` loci's ``document_id`` and the ``:SourceObject``
            key (matches every other document-modality producer).
        source: Provenance source tag forwarded to ``store_media``.
        provider: Optional non-default sidecar (see
            ``sidecar_contract.SIDECAR_CAPABILITIES``), e.g.
            ``"pdf_documents"`` to route through paperless-ngx-mcp instead.
        session: Optional explicit ``GraphSession`` forwarded to every
            ``MediaStore`` write.
        media_store: Optional injected ``MediaStore`` (tests inject a fake;
            defaults to ``native_ingest.media_store()``).
    """
    from ..knowledge_graph.memory import native_ingest

    store = media_store if media_store is not None else native_ingest.media_store()

    bundle = store.store_media(
        data,
        media_type="application/pdf",
        mime_type="application/pdf",
        name=document_id,
        source=source,
        session=session,
    )
    if bundle is None:
        return PdfSidecarResult(
            available=False,
            document_id=document_id,
            occurrence_id=None,
            blob_id=None,
            claim_id=None,
            page_count=0,
            error="failed to store the PDF artifact bundle",
        )

    try:
        capability = capability_for(_MODALITY, provider=provider)
    except SidecarContractError as exc:
        return PdfSidecarResult(
            available=False,
            document_id=document_id,
            occurrence_id=bundle.occurrence_id,
            blob_id=bundle.blob_id,
            claim_id=None,
            page_count=0,
            error=str(exc),
        )

    result = delegate_extract(
        data,
        digest=bundle.digest,
        media_type="application/pdf",
        modality=_MODALITY,
        provider=provider,
    )

    if not result.available:
        return PdfSidecarResult(
            available=False,
            document_id=document_id,
            occurrence_id=bundle.occurrence_id,
            blob_id=bundle.blob_id,
            claim_id=None,
            page_count=0,
            error=result.error,
        )

    pages = result.raw.get("pages")
    pages = pages if isinstance(pages, list) else []

    claim_id: str | None = None
    if pages:
        try:
            claim_id = lineage.record_media_sidecar_claim(
                claim_engine(),
                sidecar=result.provider,
                modality=_MODALITY,
                artifact_id=document_id,
                summary=f"{result.provider} extracted {len(pages)} page(s) from {document_id}",
                activity_id=result.activity_id,
            )
        except Exception as exc:  # noqa: BLE001 - provenance is best-effort
            logger.debug(
                "ingest_pdf_via_sidecar: claim record failed for %s: %s",
                document_id,
                exc,
            )

    page_evidence_ids: list[str] = []
    span_evidence_ids: list[str] = []
    ocr_box_evidence_ids: list[str] = []

    can_page_box = is_capable(capability, "PageBox")
    can_document_span = is_capable(capability, "DocumentSpan")
    can_image_region = is_capable(capability, "ImageRegion")

    for page in pages:
        if not isinstance(page, dict):
            continue
        try:
            page_no = int(page.get("page") or 0)
        except (TypeError, ValueError):
            continue
        if page_no <= 0:
            continue
        width = float(page.get("width") or 0.0)
        height = float(page.get("height") or 0.0)
        text = str(page.get("text") or "")
        embedding = page.get("embedding")
        embedding = list(embedding) if isinstance(embedding, list) else None

        if can_page_box:
            page_evidence = store.store_document_page_evidence(
                text.encode("utf-8"),
                document_id=document_id,
                page=page_no,
                x=0.0,
                y=0.0,
                width=width,
                height=height,
                mime_type="text/plain",
                source=result.provider,
                claim_id=claim_id,
                embedding=embedding,
                session=session,
            )
            if page_evidence is not None:
                page_evidence_ids.append(page_evidence.evidence_id)

        if can_document_span and text:
            span_evidence = store.store_document_span_evidence(
                text.encode("utf-8"),
                document_id=document_id,
                start=0,
                end=len(text),
                mime_type="text/plain",
                source=result.provider,
                claim_id=claim_id,
                embedding=embedding,
                session=session,
            )
            if span_evidence is not None:
                span_evidence_ids.append(span_evidence.evidence_id)

        boxes = page.get("boxes")
        if can_image_region and isinstance(boxes, list):
            image_id = f"{document_id}:page{page_no}"
            for box in boxes:
                if not isinstance(box, dict):
                    continue
                box_text = str(box.get("text") or "")
                region_evidence = store.store_image_region_evidence(
                    box_text.encode("utf-8"),
                    image_id=image_id,
                    x=float(box.get("x") or 0.0),
                    y=float(box.get("y") or 0.0),
                    width=float(box.get("width") or 0.0),
                    height=float(box.get("height") or 0.0),
                    mime_type="text/plain",
                    source=result.provider,
                    claim_id=claim_id,
                    session=session,
                )
                if region_evidence is not None:
                    ocr_box_evidence_ids.append(region_evidence.evidence_id)

    return PdfSidecarResult(
        available=True,
        document_id=document_id,
        occurrence_id=bundle.occurrence_id,
        blob_id=bundle.blob_id,
        claim_id=claim_id,
        page_count=len(pages),
        page_evidence_ids=page_evidence_ids,
        span_evidence_ids=span_evidence_ids,
        ocr_box_evidence_ids=ocr_box_evidence_ids,
    )
