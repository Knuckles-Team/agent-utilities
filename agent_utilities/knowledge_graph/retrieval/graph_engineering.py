from __future__ import annotations

"""Graph Engineering / GraphRAG retrieval surface (CONCEPT:AU-KG.retrieval.graph-engineering-canonical-prompts).

Gap-fill over the Microsoft GraphRAG method, NOT a from-scratch implementation:
community detection already runs natively in the Rust engine
(``eg-compute::mining::community`` — Louvain + Label Propagation) and
community-*report* generation already exists as an ingest-time pipeline phase
(:mod:`..pipeline.phases.community_reports`). What was missing was (1) an
on-demand/live path to (re)build those reports over an already-ingested graph,
with an embedding so they are semantically findable, and (2) the two GraphRAG
query modes that read them back:

* :func:`local_search` — an entity + its relationship-path neighborhood
  (GraphRAG "local search"). Seed resolution tries a bounded, parameterized
  native-Cypher exact match first (guided by the ``kg_graph_query`` canonical
  prompt — never raw LLM-authored Cypher text), falling back to semantic
  search. The neighborhood is then handed to the EXISTING
  :class:`~.context_compiler.ContextCompiler` (policy enforcement, MMR
  diversity, budget-fit, citations, proof graph — all reused, nothing
  reimplemented) via a small static-candidate adapter, and the final answer is
  synthesized with the ``kg_grounded_answer`` canonical prompt.
* :func:`global_search` — bounded map-reduce over ``:CommunityReport`` nodes
  (GraphRAG "global search"). MAP: rank reports by embedding-cosine similarity
  to the query, then a bounded per-report LLM call produces a scored partial
  answer. REDUCE: the top-scored partial answers go back through the SAME
  ``ContextCompiler`` + ``kg_grounded_answer`` prompt for final synthesis.

:func:`narrate_maintenance_action` wires the fifth canonical prompt
(``kg_graph_maintenance``) onto the EXISTING contradiction/TMS path
(:mod:`..adaptation.contradiction_detector`) as a best-effort LLM
recommendation layered on top of the deterministic detector — propose-only,
same contract as the detector itself.

Dependency discipline: no new ML dependency — embeddings reuse
``core.embedding_utilities.create_embedding_model`` (remote, OpenAI-compatible);
community detection stays 100% inside the Rust engine; every native-Cypher call
goes through the engine backend's existing parameterized/escaped
``execute_read`` (never a hand-inlined literal).
"""

import json
import logging
from string import Template
from typing import Any

from agent_utilities.prompts.canonical import load_canonical_prompt

from ..core.engine import cosine_similarity
from ..pipeline.phases.community_reports import summarize_community
from .context_compiler import ContextCompiler

logger = logging.getLogger(__name__)

__all__ = [
    "build_community_reports",
    "local_search",
    "global_search",
    "narrate_maintenance_action",
    "resolve_llm_fn",
]

# Kept in sync with pipeline/phases/community_reports.py's bounds (same
# cost-control rationale: only summarize communities big enough to matter, cap
# the total summarized on a big graph).
_MIN_COMMUNITY_SIZE = 8
_MAX_COMMUNITIES = 50

# GraphRAG local search is always a BOUNDED neighborhood, never a whole-graph
# walk -- caps total candidate nodes regardless of branching factor/depth.
_MAX_LOCAL_NEIGHBORS = 40
# Probing every member's edges for the community-report prompt would be an
# unbounded per-community fan-out over the out-of-process engine transport
# ("batch, never per-element" -- see epistemic-graph/AGENTS.md); cap both the
# probed members and the collected descriptions.
_MAX_EDGE_PROBE_MEMBERS = 15
_MAX_EDGE_DESCRIPTIONS = 30

_DEFAULT_MAX_COMMUNITIES_GLOBAL = 8

# ---------------------------------------------------------------------------
# Fallback literals for the packaged canonical prompts (used only if the
# packaged JSON can't be loaded -- see agent_utilities/prompts/canonical.py).
# ---------------------------------------------------------------------------

_GRAPH_QUERY_PROMPT_DEFAULT = (
    "Extract a structured lookup target from a natural-language knowledge-graph "
    "question. Do NOT write Cypher or any query syntax yourself -- output ONLY "
    "the JSON object below, which the caller turns into a bounded, parameterized "
    "native-Cypher exact-match lookup.\n\n"
    "Question: $question\n\n"
    "Known node labels in this graph (best-effort hint, may be incomplete): "
    "$node_labels\n\n"
    "Return ONLY JSON with this exact shape:\n"
    "{\n"
    '  "entity_name": "<the canonical name/id of the entity the question is '
    'about, verbatim from the question, or best guess>",\n'
    '  "node_label": "<one of the known node labels above if identifiable, '
    'else null>"\n'
    "}\n\n"
    "If the question does not name a specific entity (e.g. it is a broad/"
    'thematic question), return {"entity_name": null, "node_label": null}.'
)

_GROUNDED_ANSWER_PROMPT_DEFAULT = (
    "Answer the question using ONLY the numbered context items below -- never "
    "introduce a fact that is not present in them. Cite the supporting item(s) "
    "inline with their [n] marker after each claim you make. If the context "
    "does not contain enough information to answer, say so explicitly instead "
    "of guessing.\n\n"
    "Question: $question\n\n"
    "Context:\n$context\n\n"
    "Answer (with inline [n] citations):"
)

_GRAPH_MAINTENANCE_PROMPT_DEFAULT = (
    "Two knowledge-graph claims have been flagged as contradicting one another "
    "by a deterministic friction detector. Recommend a maintenance action "
    "grounded ONLY in the two claim texts below -- do not invent facts outside "
    "them.\n\n"
    "New claim: $new_claim\n"
    "Existing claim: $existing_claim\n"
    "Topical similarity: $similarity\n"
    "Detector severity: $severity\n"
    "Detector reason: $reason\n\n"
    "Return ONLY JSON with this exact shape:\n"
    "{\n"
    '  "recommendation": "<one of: keep_new | keep_existing | merge | '
    'needs_human_review>",\n'
    '  "rationale": "<one sentence, grounded only in the two claim texts '
    'above>"\n'
    "}\n\n"
    'Use "needs_human_review" whenever the two claims are not clearly '
    "resolvable from their text alone (e.g. both plausible, missing context, "
    "high-stakes topic)."
)

# Intentionally NOT one of the 5 packaged canonical prompts: this is the MAP
# step's own small per-community scoring prompt (mirrors how
# pipeline/phases/community_reports.py has its own build_summary_prompt for a
# distinct, non-canonical purpose), used only internally by global_search.
_MAP_PARTIAL_ANSWER_PROMPT = Template(
    "You are one of several analysts, each given ONE community summary from a "
    "larger knowledge graph. Given ONLY the community summary below, decide "
    "whether it helps answer the question. If it does, give a short partial "
    "answer grounded ONLY in the summary; if it does not, say so.\n\n"
    "Question: $question\n\n"
    "Community theme: $theme\n"
    "Community summary: $summary\n\n"
    'Return ONLY JSON: {"partial_answer": "<partial answer, or empty string '
    'if irrelevant>", "score": <0-100 integer relevance score>}'
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def resolve_llm_fn() -> Any:
    """Best-effort lite-LLM completion fn, or ``None`` (never raises).

    Shared resolver so a caller iterating many findings/reports (e.g. the
    ``contradictions`` MCP action narrating several friction findings) resolves
    the client ONCE rather than per item.
    """
    try:
        from ..enrichment.cards import make_lite_llm_fn

        return make_lite_llm_fn()
    except Exception as exc:  # noqa: BLE001 — LLM enrichment is always optional
        logger.debug("graph_engineering: no LLM available: %s", exc)
        return None


def _resolve_embed_model() -> Any:
    """Best-effort embedding model via the standard factory, or ``None``."""
    try:
        from agent_utilities.core.embedding_utilities import create_embedding_model

        return create_embedding_model()
    except Exception as exc:  # noqa: BLE001 — embeddings are always optional here
        logger.debug("graph_engineering: no embedding model available: %s", exc)
        return None


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """Best-effort JSON object extraction from an LLM response (handles code fences)."""
    if not raw:
        return None
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) >= 2 else text
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _node_label(props: dict[str, Any], node_id: str) -> str:
    return str(props.get("label") or props.get("name") or props.get("title") or node_id)


def _graph_compute(engine: Any) -> Any:
    """The low-level ``GraphComputeEngine`` behind a top-level engine object."""
    return getattr(engine, "graph_compute", engine)


class _StaticCandidateRetriever:
    """Adapter exposing a fixed candidate list as ``retrieve_hybrid``.

    Lets :class:`~.context_compiler.ContextCompiler` assemble a policy-aware,
    scored, cited bundle over graph-traversal / community-report results
    instead of running its own semantic retrieval — the "reuse the
    ContextCompiler for grounded assembly" seam this module is built on.
    """

    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self._candidates = candidates

    def retrieve_hybrid(
        self,
        query: str,  # noqa: ARG002 — candidates are precomputed by the caller
        *,
        context_window: int = 0,
        as_of: str | None = None,  # noqa: ARG002 — precomputed candidates only
        skip_quality_gate: bool = True,  # noqa: ARG002 — precomputed candidates only
    ) -> list[dict[str, Any]]:
        if context_window and context_window > 0:
            return list(self._candidates[:context_window])
        return list(self._candidates)


def _as_candidate(
    node_id: str,
    props: dict[str, Any],
    *,
    score: float,
    relationship: str = "",
    related_to: str = "",
) -> dict[str, Any]:
    """Render one graph node as a ContextCompiler-shaped candidate dict."""
    name = _node_label(props, node_id)
    description = str(props.get("description") or props.get("summary") or "")
    text_desc = description
    if relationship and related_to:
        prefix = f"[{related_to} --{relationship}--> {name}] "
        text_desc = f"{prefix}{description}".strip()
    out = dict(props)
    out["id"] = node_id
    out["name"] = name
    out["description"] = text_desc or name
    out["score"] = score
    return out


# ---------------------------------------------------------------------------
# Community report generation (gap-fill over the existing ingest-time phase)
# ---------------------------------------------------------------------------


def _community_edge_descriptions(graph: Any, members_sorted: list[str]) -> list[str]:
    """Best-effort intra-community edge descriptions for the summary prompt.

    Bounded: probes only the top-degree members and stops once enough
    descriptions are collected, rather than scanning every edge (the
    out-of-process engine transport charges one round trip per call).
    """
    member_set = set(members_sorted)
    probe = members_sorted[:_MAX_EDGE_PROBE_MEMBERS]
    descriptions: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    for node_id in probe:
        if len(descriptions) >= _MAX_EDGE_DESCRIPTIONS:
            break
        try:
            neighbors = graph.get_neighbors(node_id) or []
        except Exception:  # noqa: BLE001 — best-effort enrichment only
            continue
        for neighbor_id in neighbors:
            if neighbor_id not in member_set or neighbor_id == node_id:
                continue
            pair = tuple(sorted((node_id, neighbor_id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            try:
                edge_props = graph._get_edge_properties(node_id, neighbor_id) or {}  # noqa: SLF001
            except Exception:  # noqa: BLE001
                edge_props = {}
            relationship = str(edge_props.get("relationship") or "related_to")
            descriptions.append(f"{node_id} {relationship} {neighbor_id}")
            if len(descriptions) >= _MAX_EDGE_DESCRIPTIONS:
                break
    return descriptions


def build_community_reports(
    engine: Any,
    *,
    resolution: float = 1.0,
    min_size: int = _MIN_COMMUNITY_SIZE,
    max_communities: int = _MAX_COMMUNITIES,
    embed: bool = True,
) -> dict[str, Any]:
    """(Re)build ``:CommunityReport`` nodes over an already-ingested LIVE graph.

    The on-demand counterpart to :func:`..pipeline.phases.community_reports.
    execute_community_reports` (which only runs automatically during
    ingestion): calls the SAME Rust community-detection kernel via
    ``engine.graph_compute.community_detection()`` and reuses
    :func:`~..pipeline.phases.community_reports.summarize_community` (which in
    turn builds the exact same prompt ``community_reports.py`` already uses)
    for the theme+summary (no duplicated prompt logic), then additionally
    embeds each report's ``theme + summary`` via the standard embedding
    factory so it is semantically findable — closing the gap that starved
    :func:`global_search`.

    Returns ``{"community_reports": <written>, "communities_considered": <n>}``.
    """
    graph = _graph_compute(engine)
    try:
        communities = graph.community_detection(resolution=resolution)
    except Exception as exc:  # noqa: BLE001 — degrade, never raise on a report build
        logger.warning("build_community_reports: community_detection failed: %s", exc)
        return {"community_reports": 0, "error": str(exc)}
    if not communities:
        return {"community_reports": 0}

    sized = [
        (idx, list(members))
        for idx, members in enumerate(communities)
        if len(members) >= min_size
    ]
    sized.sort(key=lambda kv: len(kv[1]), reverse=True)
    ranked = sized[:max_communities]
    if not ranked:
        return {"community_reports": 0}

    member_props: dict[str, dict[str, Any]] = {}
    degree: dict[str, int] = {}
    for _idx, members in ranked:
        for node_id in members:
            if node_id in member_props:
                continue
            try:
                member_props[node_id] = graph._get_node_properties(node_id) or {}  # noqa: SLF001
            except Exception:  # noqa: BLE001
                member_props[node_id] = {}
            try:
                degree[node_id] = graph.degree(node_id)
            except Exception:  # noqa: BLE001
                degree[node_id] = 0

    llm_fn = resolve_llm_fn()

    report_meta: list[dict[str, Any]] = []
    report_texts: list[str] = []
    level0_themes: list[str] = []
    for community_idx, members in ranked:
        members_sorted = sorted(members, key=lambda m: degree.get(m, 0), reverse=True)
        labels = [_node_label(member_props.get(nid, {}), nid) for nid in members_sorted]
        edge_descriptions = _community_edge_descriptions(graph, members_sorted)
        theme, summary = summarize_community(labels, edge_descriptions, llm_fn)
        report_id = f"community_report:{community_idx}"
        level0_themes.append(theme)
        report_texts.append(f"{theme}. {summary}".strip(". ") or theme)
        report_meta.append(
            {
                "report_id": report_id,
                "community": community_idx,
                "member_count": len(members),
                "theme": theme,
                "summary": summary,
                "members": members,
            }
        )

    embeddings: list[list[float] | None] = [None] * len(report_texts)
    if embed and report_texts:
        embed_model = _resolve_embed_model()
        if embed_model is not None:
            try:
                embeddings = list(embed_model.get_text_embedding_batch(report_texts))
            except Exception as exc:  # noqa: BLE001 — embedding is additive, never fatal
                logger.debug("build_community_reports: embedding batch failed: %s", exc)

    written = 0
    for meta, embedding in zip(report_meta, embeddings, strict=False):
        props: dict[str, Any] = {
            "node_type": "CommunityReport",
            "community": meta["community"],
            "level": 0,
            "member_count": meta["member_count"],
            "theme": meta["theme"],
            "summary": meta["summary"],
            "label": meta["theme"],
        }
        if embedding:
            props["embedding"] = embedding
        try:
            graph.add_node(meta["report_id"], props)
            for nid in meta["members"]:
                graph.add_edge(nid, meta["report_id"], relationship="PART_OF_COMMUNITY")
            written += 1
        except Exception as exc:  # noqa: BLE001 — one bad report shouldn't abort the rest
            logger.warning(
                "build_community_reports: failed to write %s: %s",
                meta["report_id"],
                exc,
            )

    if len(level0_themes) >= 2:
        theme, summary = summarize_community(
            [f"Theme: {t}" for t in level0_themes], [], llm_fn
        )
        global_id = "community_report:global"
        global_embedding = None
        if embed:
            embed_model = _resolve_embed_model()
            if embed_model is not None:
                try:
                    global_embedding = embed_model.get_text_embedding(
                        f"{theme}. {summary}".strip(". ") or theme
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "build_community_reports: global embedding failed: %s", exc
                    )
        props = {
            "node_type": "CommunityReport",
            "level": 1,
            "member_count": len(level0_themes),
            "theme": theme or "Global themes",
            "summary": summary,
            "label": theme or "Global themes",
        }
        if global_embedding:
            props["embedding"] = global_embedding
        try:
            graph.add_node(global_id, props)
            for meta in report_meta:
                graph.add_edge(
                    meta["report_id"], global_id, relationship="PART_OF_COMMUNITY"
                )
            written += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "build_community_reports: failed to write global report: %s", exc
            )

    return {"community_reports": written, "communities_considered": len(ranked)}


# ---------------------------------------------------------------------------
# Local search — entity + its relationship-path neighborhood
# ---------------------------------------------------------------------------


def _resolve_seed_ids(
    engine: Any,
    query: str,
    node_id: str,
    *,
    top_k: int,
    llm_fn: Any,
) -> list[str]:
    """Resolve one or more seed node ids for :func:`local_search`.

    ``node_id`` wins outright. Otherwise, in order: (1) the ``kg_graph_query``
    canonical prompt extracts a structured ``{entity_name, node_label}``
    target, executed as a bounded, parameterized native-Cypher EXACT match
    (never raw LLM-authored Cypher text); (2) the same exact match tried
    against the RAW query text (no LLM required — covers the common "the
    query IS the entity name" case, e.g. ``query="Acme Corp"``, and keeps
    seed resolution useful with no LLM configured at all); (3) on a miss,
    fall back to ``engine.semantic_search`` over the free-text query.
    """
    if node_id:
        return [node_id]
    if not query:
        return []
    graph = _graph_compute(engine)

    entity_name = None
    if llm_fn is not None:
        try:
            rendered = load_canonical_prompt(
                "kg_graph_query",
                fallback=_GRAPH_QUERY_PROMPT_DEFAULT,
                question=query,
                node_labels="(unknown)",
            )
            parsed = _extract_json_object(llm_fn(rendered))
            if parsed:
                name = parsed.get("entity_name")
                if isinstance(name, str) and name.strip():
                    entity_name = name.strip()
        except Exception as exc:  # noqa: BLE001 — falls through to semantic search
            logger.debug("local_search: graph-query prompt failed: %s", exc)

    backend = getattr(engine, "backend", None)
    for candidate_name in dict.fromkeys(n for n in (entity_name, query.strip()) if n):
        try:
            if backend is not None and hasattr(backend, "execute_read"):
                rows = backend.execute_read(
                    "MATCH (n) WHERE n.name = $name OR n.label = $name "
                    "OR n.id = $name RETURN n.id AS id LIMIT $limit",
                    {"name": candidate_name, "limit": max(top_k, 1)},
                )
                ids = [
                    str(r["id"]) for r in rows if isinstance(r, dict) and r.get("id")
                ]
                if ids:
                    return ids
        except Exception as exc:  # noqa: BLE001 — falls through to semantic search
            logger.debug("local_search: exact-match lookup failed: %s", exc)

    embed_model = _resolve_embed_model()
    if embed_model is not None:
        try:
            query_embedding = embed_model.get_text_embedding(entity_name or query)
            hits = graph.semantic_search(query_embedding, n_results=max(top_k, 1)) or []
            return [str(hit_id) for hit_id, _score in hits]
        except Exception as exc:  # noqa: BLE001 — no seed resolvable
            logger.debug("local_search: semantic_search fallback failed: %s", exc)
    return []


def _fetch_entity_neighborhood(
    graph: Any, seed_ids: list[str], *, depth: int
) -> list[dict[str, Any]]:
    """Seed entities + their relationship-path neighborhood, as
    ContextCompiler-shaped candidates. Bounded BFS: ``depth`` hops, capped at
    ``_MAX_LOCAL_NEIGHBORS`` total nodes.
    """
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    seed_set = set(seed_ids)
    frontier = list(dict.fromkeys(seed_ids))
    for hop in range(max(1, depth)):
        next_frontier: list[str] = []
        for node_id in frontier:
            if node_id in seen:
                continue
            seen.add(node_id)
            try:
                props = graph._get_node_properties(node_id) or {}  # noqa: SLF001
            except Exception:  # noqa: BLE001
                props = {}
            candidates.append(
                _as_candidate(
                    node_id,
                    props,
                    score=1.0 if node_id in seed_set else max(0.3, 0.9 - 0.2 * hop),
                )
            )
            if len(candidates) >= _MAX_LOCAL_NEIGHBORS:
                return candidates
            try:
                neighbors = graph.get_neighbors(node_id) or []
            except Exception:  # noqa: BLE001
                neighbors = []
            for neighbor_id in neighbors:
                if neighbor_id in seen:
                    continue
                try:
                    edge_props = graph._get_edge_properties(node_id, neighbor_id) or {}  # noqa: SLF001
                except Exception:  # noqa: BLE001
                    edge_props = {}
                relationship = str(edge_props.get("relationship") or "related_to")
                try:
                    neighbor_props = graph._get_node_properties(neighbor_id) or {}  # noqa: SLF001
                except Exception:  # noqa: BLE001
                    neighbor_props = {}
                candidates.append(
                    _as_candidate(
                        neighbor_id,
                        neighbor_props,
                        score=max(0.2, 0.8 - 0.2 * hop),
                        relationship=relationship,
                        related_to=node_id,
                    )
                )
                next_frontier.append(neighbor_id)
                if len(candidates) >= _MAX_LOCAL_NEIGHBORS:
                    return candidates
        frontier = next_frontier
    return candidates


def local_search(
    engine: Any,
    query: str,
    *,
    node_id: str = "",
    depth: int = 1,
    top_k: int = 8,
    token_budget: int = 2000,
    session: Any = None,
    synthesize_answer: bool = True,
) -> dict[str, Any]:
    """GraphRAG-style local search: one entity + its relationship-path neighborhood.

    Args:
        engine: the top-level graph engine (e.g. ``kg_server._get_engine()``).
        query: the natural-language question; also used for seed resolution
            when ``node_id`` is not supplied.
        node_id: an explicit seed node id, skipping seed resolution entirely.
        depth: neighborhood hops to walk from the seed (bounded, default 1).
        top_k / token_budget: passed straight through to
            :meth:`~.context_compiler.ContextCompiler.compile`.
        session: optional verified ``GraphSession``; ``None`` resolves the
            ambient MCP/REST caller session.
        synthesize_answer: render the ``kg_grounded_answer`` canonical prompt
            over the assembled bundle (best-effort — degrades to
            ``answer: None`` with no LLM configured).

    Returns:
        ``{"seed_ids": [...], "answer": str | None, "bundle": <ContextBundle dict>}``.
    """
    graph = _graph_compute(engine)
    llm_fn = resolve_llm_fn() if (synthesize_answer or not node_id) else None
    seed_ids = _resolve_seed_ids(
        engine, query, node_id, top_k=max(top_k, 1), llm_fn=llm_fn
    )
    if not seed_ids:
        return {
            "seed_ids": [],
            "answer": None,
            "bundle": None,
            "reason": "no seed entity resolved",
        }

    candidates = _fetch_entity_neighborhood(graph, seed_ids, depth=max(1, depth))
    if not candidates:
        return {
            "seed_ids": seed_ids,
            "answer": None,
            "bundle": None,
            "reason": "seed entity has no readable properties/neighbors",
        }

    compiler = ContextCompiler(
        engine, hybrid_retriever=_StaticCandidateRetriever(candidates)
    )
    bundle = compiler.compile(
        query or seed_ids[0],
        session,
        top_k=top_k,
        candidate_pool=max(top_k, len(candidates)),
        token_budget=token_budget,
    )

    answer = None
    if synthesize_answer and llm_fn is not None and bundle.items:
        rendered = load_canonical_prompt(
            "kg_grounded_answer",
            fallback=_GROUNDED_ANSWER_PROMPT_DEFAULT,
            question=query or f"What do we know about {seed_ids[0]}?",
            context=bundle.as_text(),
        )
        try:
            answer = llm_fn(rendered)
        except Exception as exc:  # noqa: BLE001 — the bundle is still useful without prose
            logger.debug("local_search: grounded-answer synthesis failed: %s", exc)

    return {"seed_ids": seed_ids, "answer": answer, "bundle": bundle.to_dict()}


# ---------------------------------------------------------------------------
# Global search — bounded map-reduce over :CommunityReport nodes
# ---------------------------------------------------------------------------


def _load_community_reports(engine: Any, *, level: int) -> list[dict[str, Any]]:
    backend = getattr(engine, "backend", None)
    if backend is None or not hasattr(backend, "execute_read"):
        return []
    try:
        rows = backend.execute_read(
            "MATCH (r) WHERE r.node_type = $node_type AND r.level = $level "
            "RETURN r.id AS id, r.theme AS theme, r.summary AS summary, "
            "r.member_count AS member_count, r.embedding AS embedding LIMIT $limit",
            {"node_type": "CommunityReport", "level": level, "limit": 500},
        )
    except Exception as exc:  # noqa: BLE001 — degrade to "no reports available"
        logger.debug("global_search: community-report lookup failed: %s", exc)
        return []
    return [dict(row) for row in rows if isinstance(row, dict) and row.get("id")]


def _rank_reports(reports: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Rank community reports by embedding-cosine similarity to ``query``.

    Falls back to a deterministic largest-community-first order when no
    report carries an embedding (e.g. reports written before this feature) or
    no embedding model is configured — never silently drops candidates.
    """
    embed_model = _resolve_embed_model() if query else None
    query_embedding = None
    if embed_model is not None:
        try:
            query_embedding = embed_model.get_text_embedding(query)
        except Exception as exc:  # noqa: BLE001
            logger.debug("global_search: query embedding failed: %s", exc)

    scored: list[dict[str, Any]] = []
    for report in reports:
        similarity = 0.0
        embedding = report.get("embedding")
        if query_embedding is not None and isinstance(embedding, list) and embedding:
            try:
                similarity = cosine_similarity(query_embedding, embedding)
            except Exception:  # noqa: BLE001
                similarity = 0.0
        item = dict(report)
        item["_similarity"] = similarity
        scored.append(item)

    if query_embedding is not None and any(r["_similarity"] > 0 for r in scored):
        scored.sort(key=lambda r: r["_similarity"], reverse=True)
    else:
        scored.sort(key=lambda r: _as_int(r.get("member_count")), reverse=True)
    return scored


def _map_partial_answers(
    query: str, reports: list[dict[str, Any]], llm_fn: Any
) -> list[dict[str, Any]]:
    """MAP step: a bounded per-community LLM call producing a scored partial
    answer (GraphRAG global search). Best-effort — degrades to the report's
    own theme+summary (no extra LLM cost) when no LLM is configured.
    """
    results: list[dict[str, Any]] = []
    for report in reports:
        report_id = str(report.get("id") or "")
        theme = str(report.get("theme") or "")
        summary = str(report.get("summary") or "")
        if llm_fn is None:
            results.append(
                {
                    "report_id": report_id,
                    "theme": theme,
                    "partial_answer": summary,
                    "score": float(report.get("_similarity", 0.0)) * 100,
                }
            )
            continue
        rendered = _MAP_PARTIAL_ANSWER_PROMPT.safe_substitute(
            question=query, theme=theme, summary=summary
        )
        partial_answer, score = summary, 0.0
        try:
            parsed = _extract_json_object(llm_fn(rendered)) or {}
            partial_answer = str(parsed.get("partial_answer") or "").strip() or summary
            score = float(parsed.get("score") or 0.0)
        except Exception as exc:  # noqa: BLE001 — degrade to the raw summary
            logger.debug("global_search: map step failed for %s: %s", report_id, exc)
        results.append(
            {
                "report_id": report_id,
                "theme": theme,
                "partial_answer": partial_answer,
                "score": score,
            }
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def global_search(
    engine: Any,
    query: str,
    *,
    level: int = 0,
    max_communities: int = _DEFAULT_MAX_COMMUNITIES_GLOBAL,
    token_budget: int = 4000,
    session: Any = None,
    auto_build_reports: bool = True,
) -> dict[str, Any]:
    """GraphRAG-style global search: bounded map-reduce over ``:CommunityReport`` nodes.

    Args:
        engine: the top-level graph engine (e.g. ``kg_server._get_engine()``).
        query: the natural-language question.
        level: report level to search (0 = per-community, 1 = the single
            global rollup report).
        max_communities: MAP-step fan-out bound (LLM calls are bounded by this,
            never by the total community count).
        token_budget: passed to :meth:`~.context_compiler.ContextCompiler.compile`.
        session: optional verified ``GraphSession``; ``None`` resolves the
            ambient MCP/REST caller session.
        auto_build_reports: when no ``:CommunityReport`` nodes exist yet,
            build them once on demand (Native-by-default — a caller shouldn't
            have to remember to call :func:`build_community_reports` first).

    Returns:
        ``{"answer": str | None, "bundle": <ContextBundle dict> | None,
        "communities_used": [report_id, ...]}``.
    """
    reports = _load_community_reports(engine, level=level)
    if not reports and auto_build_reports:
        try:
            build_community_reports(engine)
        except Exception as exc:  # noqa: BLE001 — fall through to "no reports"
            logger.debug("global_search: auto build_community_reports failed: %s", exc)
        reports = _load_community_reports(engine, level=level)
    if not reports:
        return {
            "answer": None,
            "bundle": None,
            "communities_used": [],
            "reason": "no community reports available",
        }

    ranked = _rank_reports(reports, query)
    top = ranked[: max(1, max_communities)]

    llm_fn = resolve_llm_fn()
    partial_answers = _map_partial_answers(query, top, llm_fn)

    candidates = [
        {
            "id": pa["report_id"],
            "name": pa["theme"] or pa["report_id"],
            "description": pa["partial_answer"],
            "score": pa["score"] / 100.0,
        }
        for pa in partial_answers
        if pa["partial_answer"]
    ]
    if not candidates:
        # Nothing scored (e.g. no LLM AND no embeddings): fall back to the raw
        # report summaries themselves so global_search still degrades usefully.
        candidates = [
            {
                "id": r["id"],
                "name": r.get("theme") or r["id"],
                "description": r.get("summary") or "",
                "score": float(r.get("_similarity", 0.0)),
            }
            for r in top
        ]

    compiler = ContextCompiler(
        engine, hybrid_retriever=_StaticCandidateRetriever(candidates)
    )
    bundle = compiler.compile(
        query,
        session,
        top_k=len(candidates),
        candidate_pool=len(candidates),
        token_budget=token_budget,
    )

    answer = None
    if llm_fn is not None and bundle.items:
        rendered = load_canonical_prompt(
            "kg_grounded_answer",
            fallback=_GROUNDED_ANSWER_PROMPT_DEFAULT,
            question=query,
            context=bundle.as_text(),
        )
        try:
            answer = llm_fn(rendered)
        except Exception as exc:  # noqa: BLE001 — the bundle is still useful without prose
            logger.debug("global_search: grounded-answer synthesis failed: %s", exc)

    return {
        "answer": answer,
        "bundle": bundle.to_dict(),
        "communities_used": [c["id"] for c in candidates],
    }


# ---------------------------------------------------------------------------
# Graph maintenance — LLM narration on top of the contradiction/TMS path
# ---------------------------------------------------------------------------


def narrate_maintenance_action(
    finding: Any,
    *,
    new_text: str = "",
    existing_text: str = "",
    llm_fn: Any = None,
) -> dict[str, Any] | None:
    """Best-effort LLM maintenance recommendation for a detected friction finding.

    Layers an LLM-articulated recommendation on top of the EXISTING
    deterministic ``ContradictionDetector``
    (:class:`~..adaptation.contradiction_detector.FrictionFinding`) — never
    resolves/mutates anything itself, same propose-only contract as the
    detector. Returns ``None`` (not a raised error) when no LLM is configured
    or the call fails — a friction finding is always usable without this
    enrichment.

    Args:
        finding: a ``FrictionFinding`` (or anything exposing ``.new_id``/
            ``.conflict_id``/``.similarity``/``.severity``/``.reason``).
        new_text / existing_text: the two claims' actual text (the finding
            itself only carries ids).
        llm_fn: reuse an already-resolved completion fn (see
            :func:`resolve_llm_fn`) when narrating several findings, to avoid
            re-resolving the client per finding.
    """
    fn = llm_fn if llm_fn is not None else resolve_llm_fn()
    if fn is None:
        return None
    rendered = load_canonical_prompt(
        "kg_graph_maintenance",
        fallback=_GRAPH_MAINTENANCE_PROMPT_DEFAULT,
        new_claim=new_text or getattr(finding, "new_id", ""),
        existing_claim=existing_text or getattr(finding, "conflict_id", ""),
        similarity=f"{float(getattr(finding, 'similarity', 0.0)):.2f}",
        severity=getattr(finding, "severity", ""),
        reason=getattr(finding, "reason", ""),
    )
    try:
        raw = fn(rendered)
    except Exception as exc:  # noqa: BLE001 — the finding is still usable without this
        logger.debug("narrate_maintenance_action: LLM call failed: %s", exc)
        return None
    parsed = _extract_json_object(raw)
    if not parsed:
        return None
    recommendation = str(parsed.get("recommendation") or "").strip()
    rationale = str(parsed.get("rationale") or "").strip()
    if not recommendation:
        return None
    return {"recommendation": recommendation, "rationale": rationale}
