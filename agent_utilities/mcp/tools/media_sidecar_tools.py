"""``graph_media_sidecar`` — the W4.6 media-sidecar-delegation MCP surface
(CONCEPT:AU-KG.ingest.media-sidecar-delegation).

Two-surface (MCP + REST — ``agent_utilities/mcp/kg_server.py`` auto-mounts
the REST twin from ``ACTION_TOOL_ROUTES``) entry point for the sidecar
delegate pattern: ``ingest_pdf``/``ingest_jpeg`` each store the artifact,
delegate extraction to the governed fleet sidecar, and through-write the
resulting pages/spans/loci/embeddings — see ``agent_utilities/media/
pdf_sidecar.py`` / ``image_sidecar.py`` for the actual write-back, and
``reports/wave4/ADR-media-sidecar.md`` for the design. ``ingest_audio``/
``ingest_video`` are DESIGN STUBS this wave (per the ADR's audio/video
scope, tracked in ``reports/issue-register.md``) — they degrade cleanly with
a clear "not implemented" payload rather than raising or pretending to work.
"""

from __future__ import annotations

import base64
import dataclasses
import json

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.media.image_sidecar import (
    ImageSidecarResult,
    ingest_jpeg_via_sidecar,
)
from agent_utilities.media.pdf_sidecar import PdfSidecarResult, ingest_pdf_via_sidecar
from agent_utilities.media.sidecar_contract import (
    IMPLEMENTED_MODALITIES,
    STUB_MODALITIES,
)
from agent_utilities.security.error_surface import public_error_json

#: action name -> capability-manifest modality key, derived from the ONE
#: source of truth (sidecar_contract.py) so this tool's action list can never
#: drift from the declared capability manifest.
_IMPLEMENTED_ACTIONS: dict[str, str] = {
    f"ingest_{modality}": modality for modality in sorted(IMPLEMENTED_MODALITIES)
}
_STUB_ACTIONS: dict[str, str] = {
    f"ingest_{modality}": modality for modality in sorted(STUB_MODALITIES)
}
_ALL_ACTIONS: tuple[str, ...] = tuple(sorted({*_IMPLEMENTED_ACTIONS, *_STUB_ACTIONS}))


def _as_str(value: object, default: str = "") -> str:
    """Coerce an omitted FastMCP ``Field``-defaulted str param back to its
    declared default.

    The internal/REST dispatcher (``kg_server._execute_tool``) calls a
    registered tool function directly, bypassing FastMCP's own pydantic
    validation — so a str param the caller omitted arrives as the
    unresolved ``pydantic.fields.FieldInfo`` object rather than its
    declared default (the same class of gap ``write_ingest_tools.py``'s
    ``_as_dict`` coerces for dict params). Every str param below is routed
    through this before use so a REST body that omits an optional field
    degrades to that field's real default instead of raising deep inside
    ``base64.b64decode``/``str.strip``.
    """
    return value if isinstance(value, str) else default


def register_media_sidecar_tools(mcp) -> None:
    """Register ``graph_media_sidecar`` (tags ``media``, ``delegation``) on ``mcp``."""

    @mcp.tool(
        name="graph_media_sidecar",
        description=(
            "Delegate a media artifact (PDF, JPEG) to a governed fleet sidecar "
            "for heavy decode (OCR / image analysis) the pure-Rust engine "
            "deliberately does not do in-process, and write the result back as "
            "an ArtifactBundle + EvidenceLocus chain + embeddings through the "
            "EXISTING evidence-spine surfaces (reports/wave4/ADR-media-sidecar.md). "
            "Actions: 'ingest_pdf' (stirlingpdf-mcp OCR -> pages/spans/OCR-word-"
            "boxes as PageBox/DocumentSpan/ImageRegion loci), 'ingest_jpeg' "
            "(data-science-mcp decode -> pHash/thumbnail/region loci). "
            "'ingest_audio'/'ingest_video' are DESIGN STUBS this wave (the "
            "Whisper transcript-segment / keyframe+shot-boundary contracts are "
            "declared in agent_utilities/media/sidecar_contract.py but no "
            "adapter is wired yet) and return a clear not-implemented payload "
            "rather than raising. Every action is best-effort/never-raises: a "
            "delegation failure or malformed sidecar response returns "
            "available=false with error set, never an exception."
        ),
        tags=["graph-os", "media", "delegation", "evidence"],
    )
    def graph_media_sidecar(
        action: str = Field(
            default="ingest_pdf",
            description=f"One of: {', '.join(_ALL_ACTIONS)}.",
        ),
        artifact_id: str = Field(
            default="",
            description="The document_id (PDF) or image_id (JPEG) this artifact is filed under.",
        ),
        data_b64: str = Field(default="", description="Base64-encoded artifact bytes."),
        source: str = Field(default="", description="Provenance source tag."),
        provider: str = Field(
            default="",
            description=(
                "Optional non-default sidecar capability key (see "
                "sidecar_contract.SIDECAR_CAPABILITIES), e.g. 'pdf_documents' "
                "to route PDF ingest through paperless-ngx-mcp instead of the "
                "default stirlingpdf-mcp."
            ),
        ),
    ) -> str:
        """Delegate one media artifact to its governed sidecar and write back pages/loci/embeddings."""
        action = _as_str(action).strip() or "ingest_pdf"
        artifact_id = _as_str(artifact_id).strip()
        data_b64 = _as_str(data_b64).strip()
        source = _as_str(source)
        provider = _as_str(provider)
        if action not in _ALL_ACTIONS:
            return json.dumps(
                {
                    "surface": "media_sidecar",
                    "action": action,
                    "error": f"unknown action {action!r}; choose one of {_ALL_ACTIONS}",
                }
            )
        if action in _STUB_ACTIONS:
            modality = _STUB_ACTIONS[action]
            return json.dumps(
                {
                    "surface": "media_sidecar",
                    "action": action,
                    "available": False,
                    "stub": True,
                    "error": (
                        f"{modality} ingestion is a DESIGN STUB this wave (W4.6) — "
                        "the sidecar contract is declared in "
                        "agent_utilities/media/sidecar_contract.py "
                        f"(SIDECAR_CAPABILITIES[{modality!r}]) but no adapter is "
                        "wired yet; see reports/issue-register.md for the "
                        "tracked follow-up."
                    ),
                }
            )

        if not artifact_id:
            return json.dumps(
                {
                    "surface": "media_sidecar",
                    "action": action,
                    "error": "artifact_id is required",
                }
            )
        try:
            data = base64.b64decode(data_b64) if data_b64 else b""
        except (ValueError, TypeError) as exc:
            return public_error_json(
                exc, code="invalid_request", context={"action": action}
            )
        if not data:
            return json.dumps(
                {
                    "surface": "media_sidecar",
                    "action": action,
                    "error": "data_b64 is required",
                }
            )

        result: PdfSidecarResult | ImageSidecarResult
        try:
            if action == "ingest_pdf":
                result = ingest_pdf_via_sidecar(
                    data, document_id=artifact_id, source=source, provider=provider
                )
            else:
                result = ingest_jpeg_via_sidecar(
                    data, image_id=artifact_id, source=source, provider=provider
                )
        except Exception as exc:  # noqa: BLE001 - a delegation/write-back failure degrades cleanly
            return public_error_json(
                exc,
                code="operation_failed",
                context={"action": action, "artifact_id": artifact_id},
            )

        payload = {
            "surface": "media_sidecar",
            "action": action,
            **dataclasses.asdict(result),
        }
        return json.dumps(payload, default=str)

    kg_server.REGISTERED_TOOLS["graph_media_sidecar"] = graph_media_sidecar
    kg_server.ACTION_TOOL_ROUTES["graph_media_sidecar"] = "/media/sidecar"
