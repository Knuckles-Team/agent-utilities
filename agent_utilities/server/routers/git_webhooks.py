"""CONCEPT:AU-ECO.connector.git-task-resolver — Git webhook ingress: issue/PR -> ingested KG object -> enqueued SWE task."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Git"], prefix="/api/git")


@router.post("/webhook")
async def webhook(payload: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    """Resolve a GitHub/GitLab issue/PR webhook into a dispatched SWE task."""
    from agent_utilities.integrations.git_resolver import resolve_and_dispatch
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    result = resolve_and_dispatch(payload, engine, source=source)
    if result is None:
        raise HTTPException(
            status_code=422, detail="unsupported or unparseable webhook payload"
        )
    return result


@router.get("/suggested-tasks")
async def suggested(repo: str | None = None, kind: str | None = None) -> dict[str, Any]:
    """List open GitTasks (the suggested-tasks taxonomy) as a graph query."""
    from agent_utilities.integrations.git_resolver import suggested_tasks
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    return {"tasks": suggested_tasks(engine, repo=repo, kind=kind)}
