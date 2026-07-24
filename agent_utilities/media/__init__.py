"""The governed media-sidecar pattern (CONCEPT:AU-KG.ingest.media-sidecar-delegation, W4.6).

Standardizes the ``graph_mine_deep``-style delegate loop for heavy media
decode (OCR, image analysis, and — as design stubs — Whisper/keyframe
extraction) the pure-Rust engine deliberately never does in-process
(``reports/wave4/ADR-media-sidecar.md``):

* :mod:`.sidecar_contract` — the typed CAS-blob-ref input identity, the
  fail-closed sidecar capability manifest, and the PROV-O governance
  helpers (``agent_utilities.knowledge_graph.etl.lineage.
  record_media_sidecar_activity``/``record_media_sidecar_claim``).
* :mod:`.sidecar_delegate` — :func:`~.sidecar_delegate.delegate_extract`,
  the ONE reusable fleet-call loop every modality adapter below shares.
* :mod:`.pdf_sidecar` / :mod:`.image_sidecar` — the two modalities shipped
  this wave, each mapping a decoded sidecar result onto ``MediaStore``'s
  existing ``ArtifactBundle``/``EvidenceLocus`` write methods.

Reached from the fleet via the ``graph_media_sidecar`` MCP tool
(``agent_utilities/mcp/tools/media_sidecar_tools.py``); reached in-process by
any future ingestion pipeline by importing ``ingest_pdf_via_sidecar``/
``ingest_jpeg_via_sidecar`` directly.
"""

from __future__ import annotations

from .image_sidecar import ImageSidecarResult, ingest_jpeg_via_sidecar
from .pdf_sidecar import PdfSidecarResult, ingest_pdf_via_sidecar
from .sidecar_contract import (
    IMPLEMENTED_MODALITIES,
    LOCUS_KINDS,
    SIDECAR_CAPABILITIES,
    STUB_MODALITIES,
    SidecarBlobRef,
    SidecarCapability,
    SidecarContractError,
    assert_capable,
    capability_for,
    is_capable,
)
from .sidecar_delegate import SidecarDelegationResult, claim_engine, delegate_extract

__all__ = [
    "IMPLEMENTED_MODALITIES",
    "LOCUS_KINDS",
    "SIDECAR_CAPABILITIES",
    "STUB_MODALITIES",
    "ImageSidecarResult",
    "PdfSidecarResult",
    "SidecarBlobRef",
    "SidecarCapability",
    "SidecarContractError",
    "SidecarDelegationResult",
    "assert_capable",
    "capability_for",
    "claim_engine",
    "delegate_extract",
    "ingest_jpeg_via_sidecar",
    "ingest_pdf_via_sidecar",
    "is_capable",
]
