"""GraphRAG-style community summarization (CONCEPT:KG-2.258).

The ``communities`` phase detects communities (native Rust Louvain) and tags each
node with a ``community`` index, but nothing *summarizes* them. This phase closes
sift-kg's community-narrative gap (``narrate/generator.py``): for each community it
asks the lite LLM for a distinctive **theme name + summary**, then persists a
first-class ``CommunityReport`` node linked to its members
(``PART_OF_COMMUNITY``) — so global-theme questions ("what are the main topics
about X?") are answered from report-grounded nodes through the existing
``graph_query``/``graph_search`` surface, no new store.

Two levels: per-community reports (level 0) and, when there are several, one global
report (level 1) summarizing the level-0 themes — the GraphRAG hierarchical view.

Default-on in the full profile; excluded from the structural bulk profile (LLM
cost). Bounded: only communities ≥ ``_MIN_COMMUNITY_SIZE`` are summarized and at
most ``_MAX_COMMUNITIES`` (largest first), so cost stays controlled on big graphs.
Best-effort: a missing/failed LLM degrades to a deterministic theme — the report
nodes still appear so the graph topology is queryable.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from ..types import PhaseResult, PipelineContext, PipelinePhase

logger = logging.getLogger(__name__)

LLMFn = Callable[[str], str]

# sift-kg parity: communities below this size are noise; cap how many we summarize.
_MIN_COMMUNITY_SIZE = 8
_MAX_COMMUNITIES = 50
# How many representative members to put in a summary prompt (highest-degree).
_MAX_MEMBERS_IN_PROMPT = 25
_MAX_EDGES_IN_PROMPT = 30


def _node_label(props: dict[str, Any], node_id: str) -> str:
    return str(props.get("label") or props.get("name") or props.get("title") or node_id)


def group_by_community(
    nodes: list[tuple[str, dict[str, Any]]],
) -> dict[int, list[tuple[str, dict[str, Any]]]]:
    """Group ``(node_id, props)`` by their ``community`` tag (untagged dropped)."""
    groups: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for node_id, props in nodes:
        community = props.get("community")
        if community is None:
            continue
        try:
            groups[int(community)].append((node_id, props))
        except (TypeError, ValueError):
            continue
    return groups


def build_summary_prompt(member_labels: list[str], edge_descriptions: list[str]) -> str:
    """Render the community-summary prompt (theme + summary, JSON out)."""
    members = "\n".join(f"- {m}" for m in member_labels[:_MAX_MEMBERS_IN_PROMPT])
    rels = "\n".join(f"- {e}" for e in edge_descriptions[:_MAX_EDGES_IN_PROMPT])
    return (
        "You are analyzing one cluster of related entities in a knowledge graph.\n"
        "Give it a SHORT, DISTINCTIVE theme label (not generic — prefer "
        '"Epstein Inner Circle" over "People and Organizations") and a 2-3 sentence '
        "summary of what connects these entities.\n\n"
        f"Entities:\n{members}\n\n"
        f"Relationships:\n{rels or '- (none captured)'}\n\n"
        'Return ONLY JSON: {"theme": "<label>", "summary": "<2-3 sentences>"}'
    )


def _parse_theme_summary(raw: str, fallback_theme: str) -> tuple[str, str]:
    """Parse the LLM JSON; fall back to a deterministic theme on any failure."""
    if raw:
        text = raw.strip()
        if "```" in text:
            text = text.split("```")[1] if text.count("```") >= 2 else text
            text = text.removeprefix("json").strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                theme = str(obj.get("theme") or "").strip()
                summary = str(obj.get("summary") or "").strip()
                if theme:
                    return theme, summary
            except (json.JSONDecodeError, TypeError):
                pass
    return fallback_theme, ""


def summarize_community(
    member_labels: list[str],
    edge_descriptions: list[str],
    llm_fn: LLMFn | None,
) -> tuple[str, str]:
    """Return ``(theme, summary)`` for a community — LLM if available, else fallback.

    Deterministic fallback theme = the top member + count, so the report node is
    always meaningful even with no LLM (best-effort, native-by-default).
    """
    fallback = (
        f"{member_labels[0]} cluster ({len(member_labels)} entities)"
        if member_labels
        else "community"
    )
    if llm_fn is None or not member_labels:
        return fallback, ""
    try:
        raw = llm_fn(build_summary_prompt(member_labels, edge_descriptions))
    except Exception:  # noqa: BLE001 — summarization never breaks the pipeline
        return fallback, ""
    return _parse_theme_summary(raw, fallback)


def _degree_map(graph: Any) -> dict[str, int]:
    degree: dict[str, int] = defaultdict(int)
    try:
        for u, v, _data in graph.edges(data=True):
            degree[u] += 1
            degree[v] += 1
    except Exception:  # noqa: BLE001
        pass
    return degree


async def execute_community_reports(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Summarize each detected community into a queryable ``CommunityReport`` node."""
    graph = ctx.graph
    try:
        if graph.node_count() == 0:
            return {"community_reports": 0}
        nodes = list(graph.nodes(data=True))
    except Exception:  # noqa: BLE001
        return {"community_reports": 0}

    groups = group_by_community(nodes)
    if not groups:
        return {"community_reports": 0}

    degree = _degree_map(graph)

    # Largest communities first; bound total summarized (cost control).
    ranked = sorted(
        (g for g in groups.items() if len(g[1]) >= _MIN_COMMUNITY_SIZE),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )[:_MAX_COMMUNITIES]
    if not ranked:
        return {"community_reports": 0}

    # Edge descriptions per community (inter-member edges only).
    members_by_community = {c: {nid for nid, _ in mem} for c, mem in ranked}
    edges_by_community: dict[int, list[str]] = defaultdict(list)
    try:
        for u, v, data in graph.edges(data=True):
            for c, member_ids in members_by_community.items():
                if u in member_ids and v in member_ids:
                    rel = str(data.get("rel_type") or data.get("type") or "related_to")
                    edges_by_community[c].append(f"{u} {rel} {v}")
                    break
    except Exception:  # noqa: BLE001
        pass

    llm_fn: LLMFn | None
    try:
        from ...enrichment.cards import make_lite_llm_fn

        llm_fn = make_lite_llm_fn()
    except Exception:  # noqa: BLE001 — degrade to deterministic themes
        llm_fn = None

    written = 0
    level0_themes: list[str] = []
    for community_idx, members in ranked:
        members_sorted = sorted(
            members, key=lambda m: degree.get(m[0], 0), reverse=True
        )
        labels = [_node_label(props, nid) for nid, props in members_sorted]
        theme, summary = summarize_community(
            labels, edges_by_community.get(community_idx, []), llm_fn
        )
        report_id = f"community_report:{community_idx}"
        graph.add_node(
            report_id,
            {
                "type": "CommunityReport",
                "community": community_idx,
                "level": 0,
                "member_count": len(members),
                "theme": theme,
                "summary": summary,
                "label": theme,
            },
        )
        for nid, _props in members:
            graph.add_edge(nid, report_id, type="PART_OF_COMMUNITY")
        level0_themes.append(theme)
        written += 1

    # Level 1: a single global report over the level-0 themes (GraphRAG global view).
    if len(level0_themes) >= 2:
        theme, summary = summarize_community(
            [f"Theme: {t}" for t in level0_themes], [], llm_fn
        )
        global_id = "community_report:global"
        graph.add_node(
            global_id,
            {
                "type": "CommunityReport",
                "level": 1,
                "member_count": len(level0_themes),
                "theme": theme or "Global themes",
                "summary": summary,
                "label": theme or "Global themes",
            },
        )
        for community_idx, _members in ranked:
            graph.add_edge(
                f"community_report:{community_idx}",
                global_id,
                type="PART_OF_COMMUNITY",
            )
        written += 1

    return {"community_reports": written}


community_reports_phase = PipelinePhase(
    name="community_reports",
    deps=["communities"],
    execute_fn=execute_community_reports,
)
