"""CONCEPT:KG-2.24 — Live Artifact gateway routes (Wire-First entry point).

Exposes create / get / refresh for Live Artifacts over the shared :class:`LiveArtifactStore` and
:class:`RefreshService`. Mounted in ``server/app.py``.

Refresh sourcing:
  * If the request body carries ``data``, that becomes the new derivation (manual refresh).
  * Otherwise a registered **source resolver** re-derives from the KG (see
    :func:`register_artifact_source`); the default resolver preserves prior data (no-op) so the route
    is always live even before a KG source is wired.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent_utilities.knowledge_graph.live_artifacts import (
    BoundedJSONError,
    LiveArtifact,
    RefreshService,
    get_live_artifact_store,
)

logger = logging.getLogger(__name__)

artifacts_router = APIRouter(tags=["live-artifacts"])

_refresh_service: RefreshService | None = None
# Resolver: given an artifact, return fresh data (e.g. by querying the KG over source_node_ids).
_source_resolver: Callable[[LiveArtifact], dict[str, Any]] = lambda a: dict(a.data)


def _service() -> RefreshService:
    global _refresh_service
    if _refresh_service is None:
        _refresh_service = RefreshService(get_live_artifact_store())
    return _refresh_service


def register_artifact_source(resolver: Callable[[LiveArtifact], dict[str, Any]]) -> None:
    """Register the KG-backed source resolver used when a refresh omits inline ``data``."""
    global _source_resolver
    _source_resolver = resolver


class CreateArtifactRequest(BaseModel):
    name: str = ""
    template: str
    data: dict[str, Any] = Field(default_factory=dict)
    source_query: str = ""
    source_node_ids: list[str] = Field(default_factory=list)
    model: str = ""


@artifacts_router.post("/api/artifacts", summary="Create a Live Artifact")
async def create_artifact(req: CreateArtifactRequest) -> dict[str, Any]:
    art = LiveArtifact(
        name=req.name,
        template=req.template,
        data=req.data,
        source_query=req.source_query,
        source_node_ids=req.source_node_ids,
    )
    art.provenance.model = req.model
    art.provenance.source_query = req.source_query
    art.provenance.evidence_node_ids = list(req.source_node_ids)
    try:
        get_live_artifact_store().create(art)
    except BoundedJSONError as exc:
        raise HTTPException(status_code=400, detail=f"bounded-JSON violation: {exc}") from exc
    return {"artifact_id": art.artifact_id, "rendered": art.last_rendered}


@artifacts_router.get("/api/artifacts/{artifact_id}", summary="Get a Live Artifact")
async def get_artifact(artifact_id: str) -> dict[str, Any]:
    art = get_live_artifact_store().get(artifact_id)
    if art is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return art.model_dump()


@artifacts_router.post("/api/artifacts/{artifact_id}/refresh", summary="Refresh a Live Artifact")
async def refresh_artifact(artifact_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Re-derive the artifact's data and re-render; preserves prior render on failure."""
    body = body or {}
    if "data" in body:
        inline = body["data"]
        source = lambda _a: inline  # noqa: E731 - manual refresh with provided data
    else:
        source = _source_resolver
    result = _service().refresh(artifact_id, source)
    if result.reason == "not found":
        raise HTTPException(status_code=404, detail="artifact not found")
    return {"ok": result.ok, "reason": result.reason, "rendered": result.rendered}
