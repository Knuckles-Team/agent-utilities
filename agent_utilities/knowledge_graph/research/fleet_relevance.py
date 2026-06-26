#!/usr/bin/python
from __future__ import annotations

"""Fleet-wide relevance grading (CONCEPT:AHE-3.63).

The feature matrix (KG-2.173) says WHAT each ingested source contributes; this
says WHICH of the fleet's 80+ ``agent-packages/*`` each source could improve. For
every research source (paper / repo) it scores keyword overlap against every
package's profile (name + description from ``workspace.yml``) and surfaces EVERY
match above a low threshold (default **5%**) as a *consideration* — a deliberately
wide net so a cross-cutting improvement opportunity for a package isn't dropped
just because it isn't the single best fit. The >5% list is the breadth signal; the
feature matrix's leverage ranking is the depth signal.

Deterministic by design — no LLM, no embedder: one overlap coefficient per
(source, target) pair, with the target profiles built once from the manifest and
reused across all sources. Cheap enough to grade N sources × 80+ targets inline.

Concept: fleet-relevance
"""

import re
from typing import Any

from ...core.workspace_config import load_workspace_yml

#: tokens too generic to discriminate one package from another (they appear in
#: most names/descriptions, so keeping them would make everything match everything).
_STOP = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "over",
    "per",
    "are",
    "via",
    "its",
    "agent",
    "agents",
    "mcp",
    "api",
    "server",
    "client",
    "service",
    "services",
    "tool",
    "tools",
    "package",
    "packages",
    "library",
    "standard",
    "support",
    "based",
    "using",
    "use",
    "used",
    "common",
    "core",
    "data",
    "management",
    "manager",
    "system",
    "systems",
    "ai",
    "automation",
}


def _tokenize(text: str) -> set[str]:
    """Discriminating keyword set: alnum tokens (len ≥ 3) minus generic stopwords."""
    return {
        t
        for t in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(t) >= 3 and t not in _STOP
    }


def _walk_repos(node: Any, out: dict[str, set[str]]) -> None:
    """Collect ``name → keyword-profile`` from a workspace.yml (sub)tree."""
    if not isinstance(node, dict):
        return
    for repo in node.get("repositories", []) or []:
        if not isinstance(repo, dict):
            continue
        url = str(repo.get("url", "")).rstrip("/")
        name = url.split("/")[-1].removesuffix(".git")
        if not name:
            continue
        desc = str(repo.get("description", ""))
        out[name] = _tokenize(name.replace("-", " ") + " " + desc)
    for sub in (node.get("subdirectories") or {}).values():
        _walk_repos(sub, out)


def fleet_target_profiles(yml_path: str | None = None) -> dict[str, set[str]]:
    """Every ``agent-packages/*`` (and sibling) target as a name → keyword profile,
    sourced from the workspace manifest (no disk reads of each package)."""
    out: dict[str, set[str]] = {}
    _walk_repos(load_workspace_yml(yml_path) or {}, out)
    return out


def _overlap_pct(a: set[str], b: set[str]) -> float:
    """Overlap coefficient as a 0–100 percentage (``|a∩b| / min(|a|,|b|)``)."""
    if not a or not b:
        return 0.0
    denom = min(len(a), len(b))
    return round(100.0 * len(a & b) / denom, 2) if denom else 0.0


def grade_item(
    item_kw: set[str],
    target_profiles: dict[str, set[str]],
    *,
    threshold_pct: float = 5.0,
) -> list[dict[str, Any]]:
    """All fleet targets a source matches at ≥ ``threshold_pct``, best first."""
    matches = [
        {"target": name, "score": score}
        for name, kw in target_profiles.items()
        if (score := _overlap_pct(item_kw, kw)) >= threshold_pct
    ]
    matches.sort(key=lambda m: (m["score"], m["target"]), reverse=True)
    return matches


_SOURCE_TYPES: tuple[str, ...] = ("article", "codebase", "document")
_TEXT_FIELDS = ("name", "title", "summary", "abstract", "content")


def grade_fleet(
    engine: Any,
    *,
    threshold_pct: float = 5.0,
    source_types: tuple[str, ...] = _SOURCE_TYPES,
    yml_path: str | None = None,
    max_sources: int = 500,
) -> dict[str, Any]:
    """Grade every ingested research source against the whole fleet.

    Returns ``{targets, sources_graded, threshold_pct, considerations}`` where each
    consideration is ``{source, source_name, match_count, matches:[{target,score}]}``
    for a source with at least one >threshold target — the fleet-wide breadth view.
    """
    targets = fleet_target_profiles(yml_path)
    graph = getattr(engine, "graph", None)
    if graph is None or not targets:
        return {
            "targets": len(targets),
            "sources_graded": 0,
            "threshold_pct": threshold_pct,
            "considerations": [],
        }
    # Bounded per-type fetch (CONCEPT:KG-2.261) — never a whole-graph node pull.
    from ..core.bounded_read import iter_nodes_by_types

    considerations: list[dict[str, Any]] = []
    graded = 0
    for nid, data in iter_nodes_by_types(graph, *source_types):
        if not isinstance(data, dict):
            continue
        if graded >= max_sources:
            break
        graded += 1
        item_kw = _tokenize(" ".join(str(data.get(k, "")) for k in _TEXT_FIELDS))
        matches = grade_item(item_kw, targets, threshold_pct=threshold_pct)
        if matches:
            considerations.append(
                {
                    "source": nid,
                    "source_name": str(data.get("name") or data.get("title") or nid),
                    "match_count": len(matches),
                    "matches": matches[:25],
                }
            )
    considerations.sort(key=lambda c: c["match_count"], reverse=True)
    return {
        "targets": len(targets),
        "sources_graded": graded,
        "threshold_pct": threshold_pct,
        "considerations": considerations,
    }


__all__ = [
    "fleet_target_profiles",
    "grade_item",
    "grade_fleet",
]
