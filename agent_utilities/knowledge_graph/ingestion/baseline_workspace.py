"""Workspace code in the baseline (spec: baseline-ingestion).

One ``codebase`` WorkItem per checked-out agent-packages repository in the
configured scope, capped by ``KG_BASELINE_MAX_CODEBASES``. The repository list
comes from ``workspace.yml``: first the configured agent workspace, then the
XDG copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .baseline_items import BaselineItem, csv_names


def load_workspace_manifest() -> dict[str, Any] | None:
    """Read the first ``workspace.yml`` that exists, or ``None``."""
    import yaml

    from agent_utilities.core.workspace import get_agent_workspace
    from agent_utilities.core.workspace_config import get_workspace_yml_path

    for path in (get_agent_workspace() / "workspace.yml", get_workspace_yml_path()):
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
    return None


def _scoped_subtree(agent_packages: dict[str, Any], scope: str) -> dict[str, Any]:
    if scope != "core":
        return agent_packages
    children = agent_packages.get("subdirectories") or {}
    return {
        "repositories": agent_packages.get("repositories") or [],
        "subdirectories": {"skills": children.get("skills") or {}},
    }


def _checked_out_repositories(
    data: dict[str, Any], scope: str
) -> list[tuple[str, Path]]:
    from agent_utilities.core.workspace_config import (
        _extract_repositories,
        _workspace_base_path,
    )

    agent_packages = (data.get("subdirectories") or {}).get("agent-packages") or {}
    root = _workspace_base_path(data, require_resolved=False) / "agent-packages"
    pairs = _extract_repositories(_scoped_subtree(agent_packages, scope), root)
    return [
        (path.name, path)
        for path, _url in pairs
        if not path.name.startswith(".") and path.is_dir()
    ]


def workspace_repositories(data: dict[str, Any], scope: str) -> list[tuple[str, Path]]:
    """Return ``(name, path)`` for agent-packages repositories in ``scope``.

    ``core`` covers the agent-packages top level plus the skills subtree.
    ``all`` covers the whole subtree. A comma-separated list filters the whole
    subtree to those names. ``none`` or empty selects nothing. Only
    checked-out directories qualify.
    """
    if scope in {"", "none"}:
        return []
    named = set() if scope in {"core", "all"} else set(csv_names(scope))
    return [
        item
        for item in _checked_out_repositories(data, scope)
        if not named or item[0] in named
    ]


def codebase_items(scope: str, limit: int) -> list[BaselineItem]:
    """One ``codebase`` item per in-scope repository, at most ``limit``."""
    data = load_workspace_manifest()
    if data is None or limit <= 0:
        return []
    return [
        BaselineItem(
            leg="codebase",
            name=name,
            target=str(path),
            task_type="codebase",
            is_codebase=True,
        )
        for name, path in workspace_repositories(data, scope)[:limit]
    ]


__all__ = ["codebase_items", "load_workspace_manifest", "workspace_repositories"]
