"""The reusable sidecar delegate loop (CONCEPT:AU-KG.ingest.media-sidecar-delegation).

Generalizes ``graph_mine_deep``'s delegate shape
(``agent_utilities/mcp/tools/engine_surface_tools.py``, CONCEPT:AU-KG.mining.dsm-forecast-delegation) for
MEDIA: hand a CAS-addressed artifact's bytes to a governed fleet sidecar over
the exact same ``call_tool_once``/``McpToolSourceConnector`` transport, record
ONE PROV-O Activity node for the call, decode its typed result (the same
defensive str-vs-dict handling that transport's BUG-7 fix established), and
return the raw payload for the caller's modality-specific write-back
(``pdf_sidecar.py`` / ``image_sidecar.py``) to fold into
``MediaStore``'s existing ``ArtifactBundle``/``EvidenceLocus`` write methods.

:func:`delegate_extract` is the ONE reusable component both shipped
modalities call — the "standardize the sidecar delegate pattern" deliverable
(``reports/wave4/ADR-media-sidecar.md``, W4.6). It never raises: an unknown
modality, an unreachable sidecar, or a malformed response all degrade to
``SidecarDelegationResult(available=False, error=...)``, matching every
other delegate tool in this codebase (``graph_mine_deep``, the connector
source presets).
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Any

# CONCEPT:AU-KG.mining.dsm-forecast-delegation — the SAME fleet write-side connector
# graph_mine_deep uses to reach data-science-mcp: one synchronous, decoded MCP
# tool call. Reused here, not reinvented, so every media modality delegates
# through the exact same primitive as every other fleet-delegation call site.
from agent_utilities.protocols.source_connectors.connectors.mcp_package import (
    _run_async,
)
from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
    call_tool_once,
)

from .sidecar_contract import SidecarContractError, capability_for

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SidecarDelegationResult:
    """The outcome of one :func:`delegate_extract` call.

    Attributes:
        available: Whether the sidecar answered with usable data. ``False``
            covers every failure mode (unknown modality, unreachable
            sidecar, malformed response, or the sidecar's own
            ``available: false``) — the caller never needs to distinguish
            them to decide whether to proceed with write-back.
        modality: The requested capability-manifest modality key.
        provider: The RESOLVED sidecar server name (the actual server used,
            not the ``provider=`` override key that selected it).
        tool: The resolved MCP tool name.
        action: The resolved action-routed value.
        activity_id: The PROV-O Activity node id recorded for this call (see
            :func:`agent_utilities.knowledge_graph.etl.lineage.record_media_sidecar_activity`),
            or ``None`` when the modality/provider didn't resolve or the
            provenance write itself failed (best-effort — never blocks the
            delegation).
        raw: The decoded sidecar response (empty when ``available`` is
            ``False`` and no response was ever parsed).
        error: A human-readable failure reason, or ``None`` on success.
    """

    available: bool
    modality: str
    provider: str
    tool: str
    action: str
    activity_id: str | None
    raw: dict[str, Any]
    error: str | None = None


def claim_engine() -> Any:
    """The ``IntelligenceGraphEngine`` PROV-O Activity/Claim nodes land on.

    Deliberately NOT ``MediaStore``'s own bound ``GraphComputeEngine`` (a
    different, lower-level engine currency reached via
    ``native_ingest.media_store()`` — MediaStore talks straight to the wire
    client, bypassing ``IntelligenceGraphEngine.add_node``'s
    ``(node_id, node_type, properties)`` signature entirely). Claim/PROV-O
    provenance elsewhere in this codebase (``etl/lineage.py``,
    ``mcp/tools/ops_causal_tools.py``) is written through THIS higher-level
    engine, resolved the same way ``graph_write``'s default (non-fan-out)
    target does — reused here so a sidecar-produced ``:Claim`` lands exactly
    where every other claim in the codebase does.
    """
    from agent_utilities.mcp import kg_server

    return kg_server.get_connection_registry().get_engine("")


def _decode_result(raw: Any) -> dict[str, Any]:
    """Normalize a fleet tool result exactly like ``graph_mine_deep``'s
    BUG-7 fix: a delegate's own ``json.dumps(...)``-returning tool
    convention means ``call_tool_once``'s decoder can hand back an
    already-JSON-encoded STRING rather than the parsed object — parse it
    here instead of treating it as a shape error. Raises
    :class:`SidecarContractError` only when the (possibly re-parsed) result
    genuinely isn't an object.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):  # noqa: BLE001 - genuinely non-JSON is a normal branch here, not a failure: it falls through to the isinstance(raw, dict) shape check below, which reports the exact type back to the caller as the real error
            pass
    if not isinstance(raw, dict):
        raise SidecarContractError(
            f"unexpected sidecar response shape: {type(raw).__name__}"
        )
    return raw


def delegate_extract(
    data: bytes,
    *,
    digest: str,
    media_type: str,
    modality: str,
    provider: str = "",
    extra_params: dict[str, Any] | None = None,
) -> SidecarDelegationResult:
    """Hand ``data`` to ``modality``'s governed sidecar and return its
    decoded result.

    The ONE reusable delegate loop shared by every modality adapter
    (``pdf_sidecar.py`` / ``image_sidecar.py`` this wave): resolve the
    fail-closed capability-manifest entry (:func:`~agent_utilities.media.sidecar_contract.capability_for`),
    record a PROV-O Activity node for the call, ship ``data`` base64-encoded
    over ``call_tool_once`` (action-routed: ``action`` + ``params_json``,
    matching the fleet's dominant tool convention — see
    ``stirlingpdf_agent/mcp/mcp_pdf.py``'s ``pdf_action`` and
    ``paperless_ngx_mcp/mcp/mcp_documents.py``'s ``document_operations``),
    and decode the response.

    Args:
        data: The artifact's raw bytes (already stored as a
            ``:Blob``/``:AssetOccurrence`` by the caller — this function
            does not store anything itself).
        digest: The artifact's content digest (correlation identity only;
            ``data`` is sent directly, never re-fetched by digest).
        media_type: The artifact's real MIME type (``"application/pdf"``,
            ``"image/jpeg"``), sent to the sidecar as-is.
        modality: The capability-manifest key family (``"pdf"``, ``"jpeg"``).
        provider: Optional non-default provider key (see
            :data:`~agent_utilities.media.sidecar_contract.SIDECAR_CAPABILITIES`).
        extra_params: Modality-specific tuning knobs merged into the wire
            payload alongside ``digest``/``media_type``/``artifact_b64``
            (e.g. ``{"languages": ["eng"]}`` for PDF OCR).

    Never raises — every failure mode returns
    ``SidecarDelegationResult(available=False, error=...)``.
    """
    try:
        capability = capability_for(modality, provider=provider)
    except SidecarContractError as exc:
        return SidecarDelegationResult(
            available=False,
            modality=modality,
            provider=provider or modality,
            tool="",
            action="",
            activity_id=None,
            raw={},
            error=str(exc),
        )

    activity_id: str | None = None
    try:
        from agent_utilities.knowledge_graph.etl import lineage

        activity_id = lineage.record_media_sidecar_activity(
            claim_engine(),
            sidecar=capability.server,
            tool=capability.tool,
            modality=modality,
            action=capability.action,
        )
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort, never blocks delegation
        logger.debug(
            "delegate_extract: activity record failed for %s/%s: %s",
            capability.server,
            capability.tool,
            exc,
        )

    params: dict[str, Any] = {
        "digest": digest,
        "media_type": media_type,
        "artifact_b64": base64.b64encode(data).decode("ascii"),
    }
    if extra_params:
        params.update(extra_params)

    try:
        raw = _run_async(
            call_tool_once(
                server=capability.server,
                tool=capability.tool,
                action=capability.action,
                params=params,
                params_style="json",
            )
        )
    except Exception as exc:  # noqa: BLE001 - an unreachable sidecar degrades cleanly, never raises
        logger.warning(
            "delegate_extract: %s/%s unreachable: %s",
            capability.server,
            capability.tool,
            exc,
        )
        return SidecarDelegationResult(
            available=False,
            modality=modality,
            provider=capability.server,
            tool=capability.tool,
            action=capability.action,
            activity_id=activity_id,
            raw={},
            error=f"{type(exc).__name__}: sidecar unreachable",
        )

    try:
        decoded = _decode_result(raw)
    except SidecarContractError as exc:
        return SidecarDelegationResult(
            available=False,
            modality=modality,
            provider=capability.server,
            tool=capability.tool,
            action=capability.action,
            activity_id=activity_id,
            raw={},
            error=str(exc),
        )

    if not decoded.get("available", True):
        return SidecarDelegationResult(
            available=False,
            modality=modality,
            provider=capability.server,
            tool=capability.tool,
            action=capability.action,
            activity_id=activity_id,
            raw=decoded,
            error=str(
                decoded.get("error") or f"{capability.server} reported unavailable"
            ),
        )

    return SidecarDelegationResult(
        available=True,
        modality=modality,
        provider=capability.server,
        tool=capability.tool,
        action=capability.action,
        activity_id=activity_id,
        raw=decoded,
        error=None,
    )
