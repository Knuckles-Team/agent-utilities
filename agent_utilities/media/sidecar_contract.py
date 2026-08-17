"""The governed media-sidecar contract (CONCEPT:AU-KG.ingest.media-sidecar-delegation) — CAS blob-ref input, fail-closed capability manifest, PROV-O governance.

**ADR:** ``reports/wave4/ADR-media-sidecar.md`` (HG-7, accepted 2026-07-23) —
the engine stays pure-Rust; all heavy media decode (OCR, JPEG, MP3/AAC,
H.264/VP9, Whisper) runs in governed fleet **delegate agents**, never
in-engine. This module is the typed contract that makes that delegation
standardized rather than ad hoc:

1. **Input identity** — a CAS blob ref (:class:`SidecarBlobRef`: digest +
   media type). A sidecar is never handed a raw path into the engine's own
   blob store (that path is private storage layout, often unreachable
   cross-host); it receives the identity plus the bytes themselves
   (base64-encoded over the MCP JSON wire — the only thing a JSON-RPC
   transport can actually carry across a process/host boundary).
2. **Capability manifest** (:data:`SIDECAR_CAPABILITIES`) — a fail-closed,
   declarative table of which sidecar/tool may produce which
   ``eg_modality::EvidenceSpan`` locus kind, "same style as the
   connector-manifest gate" (``knowledge_graph/ontology/
   connector_manifest_gate.py``) in *spirit* — fail-closed, additive,
   never-guessed — but deliberately NOT that gate's Ed25519-signed
   OWL-compile pipeline: a sidecar here is an already-trusted in-fleet MCP
   agent reached through the exact same ``call_tool_once`` transport
   ``graph_mine_deep`` uses (CONCEPT:AU-KG.mining.dsm-forecast-delegation), not an
   external, independently-onboarded data source. :func:`assert_capable` is
   the fail-closed guard every per-locus write-back in ``pdf_sidecar.py`` /
   ``image_sidecar.py`` calls before writing.
3. **Governance** — every delegation call is recorded as ONE PROV-O
   ``:PROVENANCE_ACTIVITY`` node and, when it produced locus writes, ONE
   directly-verified ``:Claim`` (``was_generated_by``/``generated_at_time``
   metadata, the same convention ``ops_causal_tools.py``'s claim
   materialization and ``owlready2_backend``'s PROV-O edge-alias table
   already recognize) that every locus write-back's ``claim_id`` links to via
   the EXISTING ``SUPPORTS`` edge convention — see
   :func:`agent_utilities.knowledge_graph.etl.lineage.record_media_sidecar_activity`
   / :func:`~agent_utilities.knowledge_graph.etl.lineage.record_media_sidecar_claim`.
   This is what makes a sidecar's result challengeable through the standard
   why/why-not machinery (``Method::ExplainEvidence`` / ``evidence_citations``)
   with no new resolver.

**Reference sidecars** (ADR item 5 — reuse the fleet, build nothing new where
an agent already exists): PDF/OCR primary is ``stirlingpdf-mcp`` (runs OCR,
returns page/word boxes); ``paperless-ngx-mcp`` is the declared alternate
(its consume pipeline OCRs a document but exposes no word-box detail — useful
for documents already living in a Paperless-ngx library). Image decode +
embeddings route to ``data-science-mcp``, the SAME server ``graph_mine_deep``
already delegates torch-dependent work to (CONCEPT:AU-KG.mining.dsm-forecast-delegation — same
server, same pure-Rust-engine boundary, a new action). **Neither
``stirlingpdf-mcp``'s OCR action nor ``data-science-mcp``'s image-decode
action exists in the live fleet yet** (verified against both repos'
checked-in tool surfaces as of this wave) — the wire contracts below
(:data:`SIDECAR_CAPABILITIES`' ``action``s and the ``artifact_b64``/``digest``/
``media_type`` request shape :func:`~agent_utilities.media.sidecar_delegate.delegate_extract`
builds) are what those tools should implement; tracked as follow-ups in
``reports/issue-register.md``. This wave proves the AU-side contract +
write-back with the fleet call MOCKED at the seam (no live sidecar required),
per the task's own acceptance bar.

**Audio/video adapters landed in GOC-07** (audio/video modality stack) —
``audio_sidecar.py``/``video_sidecar.py`` are the ``pdf_sidecar.py``/
``image_sidecar.py`` siblings for the two modalities this module previously
only declared as design stubs. Both are proven the SAME way pdf/jpeg were:
the AU-side contract + write-back with the fleet call MOCKED at the seam
(``audio-transcriber-mcp``'s ``transcribe_media`` action and
``data-science-mcp``'s ``video_keyframes`` action still do not exist in the
live fleet as of this wave — tracked in ``reports/issue-register.md`` as
follow-ups for the fleet-side tool implementation, same posture pdf/jpeg's
own not-yet-live sidecar actions were shipped under).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: The eleven ``eg_modality::EvidenceSpan`` located-locus variants MediaStore
#: already has a typed ``store_<locus>_evidence`` write-back method for (see
#: ``docs/architecture/evidence_spine_convergence.md``). A sidecar's declared
#: ``produces`` set MUST be a subset of this — a capability can never claim a
#: locus kind the write-back path doesn't support.
LOCUS_KINDS: frozenset[str] = frozenset(
    {
        "PageBox",
        "DocumentSpan",
        "TableCellRange",
        "ImageRegion",
        "AudioSegment",
        "VideoShot",
        "VideoFrameRange",
        "MetricWindow",
        "RowVersion",
        "CodeSymbol",
        "TraceSpan",
    }
)


class SidecarContractError(RuntimeError):
    """A sidecar delegation was refused, or its result did not match the
    declared contract (unknown modality/provider, or an undeclared locus
    kind at write-back time) — always caught at the call site and turned
    into a clean ``available=False`` degrade, never left to propagate."""


@dataclass(frozen=True)
class SidecarBlobRef:
    """The CAS input identity handed to a sidecar delegation (ADR item 1):
    a content digest + media type — never a raw path into the engine's own
    blob store."""

    digest: str
    media_type: str


@dataclass(frozen=True)
class SidecarCapability:
    """One governed sidecar's declared surface for one modality.

    Attributes:
        modality: The short capability-manifest key family (``"pdf"``,
            ``"jpeg"``, ``"audio"``, ``"video"``) — the value every
            ``ingest_*`` adapter passes to
            :func:`~agent_utilities.media.sidecar_delegate.delegate_extract`.
        server: The fleet MCP server name (``mcp_config.json`` key).
        tool: The MCP tool name on that server.
        action: The action-routed value that tool dispatches on (matches the
            fleet's dominant ``action`` + ``params_json`` convention — see
            ``paperless_ngx_mcp/mcp/mcp_documents.py``'s ``document_operations``
            and ``stirlingpdf_agent/mcp/mcp_pdf.py``'s ``pdf_action``).
        produces: The locus kinds (:data:`LOCUS_KINDS` subset) this sidecar
            is manifest-declared to produce — the fail-closed gate
            :func:`assert_capable` enforces before any write-back.
        description: Human-readable summary (surfaced in error/degrade
            payloads, never used for routing).
    """

    modality: str
    server: str
    tool: str
    action: str
    produces: frozenset[str]
    description: str = ""

    def __post_init__(self) -> None:
        unknown = self.produces - LOCUS_KINDS
        if unknown:
            raise SidecarContractError(
                f"capability {self.server}/{self.tool} declares unknown locus "
                f"kind(s) {sorted(unknown)} — not in LOCUS_KINDS"
            )


#: Fail-closed capability manifest — the ONE declarative table every
#: delegation call resolves against. Keyed by a stable capability id: the
#: bare modality (``"pdf"``, ``"jpeg"``) is the DEFAULT provider for that
#: modality; an alternate provider for the SAME modality gets its own key
#: (e.g. ``"pdf_documents"``) and is only selected when a caller passes
#: ``provider=<key>`` explicitly — the default routing never guesses between
#: two declared providers for one modality.
SIDECAR_CAPABILITIES: dict[str, SidecarCapability] = {
    "pdf": SidecarCapability(
        modality="pdf",
        server="stirlingpdf-mcp",
        tool="pdf_action",
        action="ocr_pdf",
        produces=frozenset({"PageBox", "DocumentSpan", "ImageRegion"}),
        description=(
            "Stirling PDF OCR: page text + char spans + OCR word/line boxes "
            "(boxes map onto ImageRegion, one per recognised box — the same "
            "granularity convention readers_media.py's local RapidOCR "
            "producer already uses for the ImageRegion locus)."
        ),
    ),
    "pdf_documents": SidecarCapability(
        modality="pdf",
        server="paperless-ngx-mcp",
        tool="document_operations",
        action="post_document",
        produces=frozenset({"PageBox", "DocumentSpan"}),
        description=(
            "Paperless-ngx consume pipeline: OCR'd page text via the "
            "already-deployed document library; no word-box detail, so "
            "ImageRegion is NOT declared for this provider. Alternate to "
            "'pdf' — selected via provider='pdf_documents'."
        ),
    ),
    "jpeg": SidecarCapability(
        modality="jpeg",
        server="data-science-mcp",
        tool="image_decode",
        action="analyze_image",
        produces=frozenset({"ImageRegion"}),
        description=(
            "Decode JPEG -> PNG thumbnail (re-entering the engine's native "
            "codec) + perceptual hash + detected regions. Same server "
            "graph_mine_deep already delegates torch-dependent work to "
            "(CONCEPT:AU-KG.mining.dsm-forecast-delegation) — same boundary, a new action."
        ),
    ),
    "audio": SidecarCapability(
        modality="audio",
        server="audio-transcriber-mcp",
        tool="transcribe_media",
        action="transcribe_segments",
        produces=frozenset({"AudioSegment"}),
        description=(
            "Whisper transcript segments with start_ms/end_ms timing -> "
            "AudioSegment loci, mirroring messaging/router.py's existing "
            "local-transcription AudioSegment producer but via the fleet "
            "sidecar instead of the in-process faster-whisper reader. "
            "GOC-07/BUG-271: adapter wired in audio_sidecar.py. The fleet tool "
            "action (audio-transcriber-mcp's transcribe_media, "
            "action=transcribe_segments) is LIVE: it verifies the caller's "
            "digest before transcribing, delegates to the SAME AudioTranscriber/ "
            "Whisper backend the server's transcribe_audio tool uses (no second "
            "transcription implementation), and never fabricates a transcript on "
            "failure."
        ),
    ),
    "video": SidecarCapability(
        modality="video",
        server="data-science-mcp",
        tool="video_keyframes",
        action="extract_keyframes",
        produces=frozenset({"VideoShot", "VideoFrameRange"}),
        description=(
            "Keyframe PNGs (re-entering the engine's native codec) + "
            "shot-boundary timestamps -> VideoShot/VideoFrameRange loci. "
            "GOC-07: adapter wired in video_sidecar.py; the fleet tool "
            "action (data-science-mcp's video_keyframes) is not yet live — "
            "tracked in reports/issue-register.md."
        ),
    ),
}

#: Modalities with a wired ``ingest_*_via_sidecar`` adapter.
IMPLEMENTED_MODALITIES: frozenset[str] = frozenset({"pdf", "jpeg", "audio", "video"})

#: Modalities with a declared :class:`SidecarCapability` contract but no
#: adapter — design stubs. Empty as of GOC-07 (audio/video landed); kept as
#: a typed set so a future modality can declare itself here again without a
#: shape change to media_sidecar_tools.py's stub-degrade branch.
STUB_MODALITIES: frozenset[str] = frozenset()


def capability_for(modality: str, *, provider: str = "") -> SidecarCapability:
    """Resolve the fail-closed capability-manifest entry for ``modality``.

    ``provider`` optionally selects a NON-default provider registered for the
    same modality (e.g. ``provider="pdf_documents"`` routes PDF ingest
    through paperless-ngx-mcp instead of the default stirlingpdf-mcp).
    Raises :class:`SidecarContractError` — never guesses, never falls back
    silently — when neither the requested provider nor the bare modality
    resolves, or when an explicit ``provider`` names a capability registered
    under a DIFFERENT modality.
    """
    key = provider or modality
    capability = SIDECAR_CAPABILITIES.get(key)
    if capability is None or (provider and capability.modality != modality):
        raise SidecarContractError(
            f"no declared sidecar capability for modality={modality!r} "
            f"provider={provider!r}"
        )
    return capability


def assert_capable(capability: SidecarCapability, locus_kind: str) -> None:
    """Fail-closed guard: refuse a write-back the sidecar's manifest entry
    did not declare it can produce.

    Called at each per-locus write-back site (``pdf_sidecar.py`` /
    ``image_sidecar.py``), never at the transport layer — the ADR's
    governance clause gates the WRITE, not merely the delegated call.
    """
    if locus_kind not in capability.produces:
        raise SidecarContractError(
            f"{capability.server}/{capability.tool} is not manifest-declared "
            f"to produce {locus_kind!r} (declared: {sorted(capability.produces)})"
        )


def is_capable(capability: SidecarCapability, locus_kind: str) -> bool:
    """Non-raising twin of :func:`assert_capable` for a write-back loop that
    wants to SKIP (and log) an under-declared locus kind rather than abort
    the whole artifact — the shape both ``pdf_sidecar.py`` and
    ``image_sidecar.py`` use so one mis-declared locus kind never loses the
    others."""
    try:
        assert_capable(capability, locus_kind)
    except SidecarContractError as exc:
        logger.warning(
            "%s not manifest-declared to produce %s, skipping: %s",
            capability.server,
            locus_kind,
            exc,
        )
        return False
    return True
