"""The governed media-sidecar pattern (CONCEPT:AU-KG.ingest.media-sidecar-delegation).

Standardizes the ``graph_mine_deep``-style delegate loop for heavy media
decode (OCR, image analysis, transcription, keyframe/shot extraction) the
pure-Rust engine deliberately never does in-process
(``reports/wave4/ADR-media-sidecar.md``; audio/video adapters landed in
GOC-07, the audio/video modality stack lane):

* :mod:`.sidecar_contract` — the typed CAS-blob-ref input identity, the
  fail-closed sidecar capability manifest, and the PROV-O governance
  helpers (``agent_utilities.knowledge_graph.etl.lineage.
  record_media_sidecar_activity``/``record_media_sidecar_claim``).
* :mod:`.sidecar_delegate` — :func:`~.sidecar_delegate.delegate_extract`,
  the ONE reusable fleet-call loop every modality adapter below shares.
* :mod:`.pdf_sidecar` / :mod:`.image_sidecar` / :mod:`.audio_sidecar` /
  :mod:`.video_sidecar` — the four modalities, each mapping a decoded
  sidecar result onto ``MediaStore``'s existing ``ArtifactBundle``/
  ``EvidenceLocus`` write methods.

Reached from the fleet via the ``graph_media_sidecar`` MCP tool
(``agent_utilities/mcp/tools/media_sidecar_tools.py``); reached in-process by
any future ingestion pipeline by importing ``ingest_pdf_via_sidecar``/
``ingest_jpeg_via_sidecar``/``ingest_audio_via_sidecar``/
``ingest_video_via_sidecar`` directly.
"""

from __future__ import annotations

from .audio_sidecar import AudioSidecarResult, ingest_audio_via_sidecar
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
from .video_sidecar import VideoSidecarResult, ingest_video_via_sidecar

__all__ = [
    "IMPLEMENTED_MODALITIES",
    "LOCUS_KINDS",
    "SIDECAR_CAPABILITIES",
    "STUB_MODALITIES",
    "AudioSidecarResult",
    "ImageSidecarResult",
    "PdfSidecarResult",
    "SidecarBlobRef",
    "SidecarCapability",
    "SidecarContractError",
    "SidecarDelegationResult",
    "VideoSidecarResult",
    "assert_capable",
    "capability_for",
    "claim_engine",
    "delegate_extract",
    "ingest_audio_via_sidecar",
    "ingest_jpeg_via_sidecar",
    "ingest_pdf_via_sidecar",
    "ingest_video_via_sidecar",
    "is_capable",
]
