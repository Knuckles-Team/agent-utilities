#!/usr/bin/python
"""Workspace Synchronization Phase (Phase 14).

Uses the repository-manager (as a library or MCP) to ensure all projects
defined in workspace.yml are cloned and available, then triggers graph ingestion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from agent_utilities.core.workspace import get_agent_workspace

from ..types import PhaseResult, PipelineContext, PipelinePhase

logger = logging.getLogger(__name__)


async def execute_workspace_sync(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Phase 14: Synchronize workspace repositories defined in workspace.yml."""
    config = ctx.config
    if not getattr(config, "enable_workspace_sync", True):
        return {"status": "skipped", "reason": "workspace sync disabled"}

    # Look for workspace.yml in the repository root (parent of agent_data)
    # or in the current agent workspace if it somehow ended up there.
    ws = get_agent_workspace()
    yml_path = ws.parent / "workspace.yml"
    if not yml_path.exists():
        yml_path = ws / "workspace.yml"

    if not yml_path.exists():
        return {"status": "skipped", "reason": "workspace.yml missing from root"}

    try:
        with open(yml_path, encoding="utf-8") as f:
            ws_data = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to parse workspace.yml: {e}")
        return {"status": "failed", "error": f"Failed to parse workspace.yml: {e}"}

    projects = ws_data.get("projects", [])
    if not projects:
        return {"status": "skipped", "reason": "no projects defined"}

    results: dict[str, Any] = {"projects_synced": 0, "auto_ingested": 0}

    # Attempt to use repository-manager
    try:
        # Strategy 1: Direct import if installed in the same environment
        from repository_manager.repository_manager import Git

        repo_mgr = Git()
        logger.info(f"Using local repository_manager to sync {len(projects)} projects.")

        # Extract project URLs
        urls = []
        for p in projects:
            if isinstance(p, str):
                urls.append(p)
            elif isinstance(p, dict) and "url" in p:
                urls.append(p["url"])

        if urls:
            repo_mgr.clone_projects(urls)
            results["projects_synced"] = len(urls)
            logger.info("Workspace sync complete via local repository_manager.")

    except ImportError:
        logger.debug("repository_manager not available as library. Skipping sync...")
        results["sync_status"] = "skipped (library missing)"

    # After sync, the repositories are on disk.
    if getattr(config, "kb_auto_ingest_cloned_repos", True):
        # Trigger KB ingestion for each project path
        from ...kb.ingestion import KBIngestionEngine

        # We need an 'engine' here, but ctx usually contains what we need.
        # However, KBIngestionEngine expects a 'graph' and 'backend' in the new pattern
        # but the old one might expect an 'engine'.
        # Let's check KBIngestionEngine signature if possible.
        # For now, I'll use the pattern from knowledge_base.py.
        kb_engine = KBIngestionEngine(
            graph=ctx.nx_graph,
            backend=ctx.backend,
        )

        for project in projects:
            p_path = None
            p_name = "unknown"

            if isinstance(project, str):
                p_name = project.split("/")[-1].replace(".git", "")
                p_path = Path.cwd() / "Workspace" / p_name
            elif isinstance(project, dict):
                p_name = project.get("name", "unknown")
                p_path = Path(project.get("path", Path.cwd() / "Workspace" / p_name))

            if p_path and p_path.exists():
                logger.info(f"Auto-ingesting cloned project: {p_name}")
                await kb_engine.ingest_directory(
                    path=str(p_path), kb_name="workspace-repos", topic=p_name
                )
                results["auto_ingested"] += 1

    return results


workspace_sync_phase = PipelinePhase(
    name="workspace_sync",
    deps=["sync"],
    execute_fn=execute_workspace_sync,
)
