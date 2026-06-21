#!/usr/bin/python
from __future__ import annotations

"""Deployment-status context provider (CONCEPT:KG-2.138).

The ``deploy`` domain of the context plane: answers the #1 build-loop papercut —
*"where does this code run, and is my change live?"* — by synthesizing the
deployment reality from git (the canonical checkout's HEAD + dirty state), the
mount-alias map (``/au`` → canonical), the active worktrees, and (when present) the
KG ``serves``/``servedBy`` route graph. It is honest about what it cannot see: a
served daemon's *loaded* revision is unknown from here, so it says so and points at
the restart that guarantees liveness, rather than guessing.

Pure read / best-effort (git shell-outs are guarded) so it never raises.
"""

import subprocess
from pathlib import Path
from typing import Any

from agent_utilities.core.source_paths import MOUNT_ALIASES, normalize_path
from agent_utilities.knowledge_graph.retrieval.context_plane import read_rows

VALID_INTENTS = ("status", "how", "impact")


def _repo_root() -> Path:
    """The agent-utilities canonical checkout root (this package's repo)."""
    # …/agent-utilities/agent_utilities/knowledge_graph/retrieval/deploy_context.py
    return Path(__file__).resolve().parents[3]


def _git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git best-effort
        return ""


def _routes_for(engine: Any, name: str) -> list[dict[str, Any]]:
    if not name:
        return []
    rows = read_rows(
        engine,
        "MATCH (h:Code)-[r2]->(rt:Route) WHERE type(r2) IN ['SERVES','serves'] "
        "AND h.name = $n OPTIONAL MATCH (rt)-[r3]->(svc) "
        "WHERE type(r3) IN ['SERVED_BY','served_by'] "
        "RETURN rt.method AS method, rt.path AS path, svc.id AS service",
        {"n": name},
    )
    return [
        {"method": r.get("method"), "path": r.get("path"), "service": r.get("service")}
        for r in rows
    ]


def deploy_status(
    engine: Any,
    *,
    query: str = "",
    intent: str = "status",
    **_opts: Any,
) -> dict[str, Any]:
    """Synthesize 'where does this run / is my change live' (see module docstring)."""
    intent = (intent or "status").strip().lower()
    if intent not in VALID_INTENTS:
        intent = "status"
    root = _repo_root()

    head = _git(root, "rev-parse", "--short", "HEAD")
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_git(root, "status", "--porcelain"))
    ahead = _git(root, "rev-list", "--count", "origin/main..HEAD") or "?"
    worktrees = [
        ln.split()[0]
        for ln in _git(root, "worktree", "list").splitlines()
        if ln.strip()
    ]
    # A symbol named in the query → its served routes (if the KG has them).
    import re as _re

    tok = next(iter(_re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", query or "")), "")
    routes = _routes_for(engine, tok)

    sections = {
        "canonical": [
            {
                "root": str(root),
                "branch": branch,
                "head": head,
                "dirty": dirty,
                "ahead_of_origin": ahead,
            }
        ],
        "mounts": [{"alias": a, "canonical": c} for a, c in MOUNT_ALIASES.items()],
        "worktrees": [{"path": w} for w in worktrees],
        "routes": routes,
    }
    answer = _synthesize(root, branch, head, dirty, ahead, worktrees, routes)
    citations = [{"type": "rev", "id": head, "branch": branch}] + [
        {"type": "route", **r} for r in routes
    ]
    return {
        "status": "ok",
        "domain": "deploy",
        "intent": intent,
        "query": query,
        "answer": answer,
        "citations": citations,
        "sections": sections,
        "capability_id": f"deploy:{intent}:{tok or 'repo'}",
        "used_primitives": ["git", "mounts"] + (["routes"] if routes else []),
    }


def _synthesize(root, branch, head, dirty, ahead, worktrees, routes) -> str:
    parts = [
        f"Canonical checkout {root} is on '{branch}' at {head or '?'}"
        + (" (working tree DIRTY)" if dirty else "")
        + f", {ahead} commit(s) ahead of origin/main (unpushed)."
    ]
    parts.append(
        f"Source mounts: {', '.join(f'{a}→canonical' for a in MOUNT_ALIASES)} — "
        "code edited at canonical (or its mount) is picked up by a served process "
        "only on restart."
    )
    if len(worktrees) > 1:
        parts.append(
            f"{len(worktrees) - 1} active worktree(s) — uncommitted work there is "
            "NOT on canonical until merged."
        )
    if routes:
        parts.append(
            "Served route(s): "
            + ", ".join(
                f"{r.get('method')} {r.get('path')}"
                + (f"→{r['service']}" if r.get("service") else "")
                for r in routes[:4]
            )
            + "."
        )
    parts.append(
        "Served daemon's LOADED revision is not observable from here — to guarantee a "
        "merged change is live, restart the served graph-os (the mount only updates "
        "files, not the running process)."
    )
    return " ".join(parts)
