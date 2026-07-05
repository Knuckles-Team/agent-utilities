#!/usr/bin/python
from __future__ import annotations

"""Hierarchical (tree-navigation) document retriever.

CONCEPT:AU-KG.retrieval.tree-navigation — a retriever that *walks* a document's
section tree by node relevance instead of embedding-similarity over flat chunks.

Provenance: PageIndex "vectorless" reasoning-tree RAG (``pageindex/retrieve.py``
``get_document_structure`` + the ``query_agent`` tree-navigation loop). Where the
:class:`~agent_utilities.knowledge_graph.retrieval.hybrid_retriever.HybridRetriever`
is similarity-first (overlap chunks → ANN/BM25 + rerank, bounded by the embedder's
recall ceiling), this retriever reasons over the per-document section tree
(CONCEPT:AU-KG.retrieval.section-tree): it scores nodes on their *title + summary*
(the text-free map), prunes irrelevant subtrees with a beam walk, and returns the
surviving sections with their cited ``char_start..char_end`` (and ``page`` for
PDFs) ranges. It **complements** — does not replace — vector/community retrieval:
route long single-document queries (a manual, a contract, a spec) here, where
"similar ≠ relevant" and an embedder's recall ceiling hurts most.

Two navigators, the first always available offline:

  * **Lexical beam walk** (default) — the dependency-free
    :class:`~agent_utilities.knowledge_graph.retrieval.reasoning_reranker.LexicalRelevanceScorer`
    scores each node's ``title + summary``; the walk keeps the top ``beam_width``
    per level and descends only into promising subtrees. No model, no network.
  * **LLM navigation** (opt-in) — hand the text-free structure to an ``llm_fn`` and
    let it pick the relevant ``node_id``s (PageIndex ``query_agent``). Falls back
    to the lexical walk when no ``llm_fn`` is configured or it errors.

The tree is loaded from the graph (a document's ``Section`` nodes, rebuilt via
:func:`~agent_utilities.knowledge_graph.ontology.document_processing.rebuild_section_tree`)
or passed in-memory for offline use.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..ontology.document_processing import (
    SECTION_NODE_TYPE,
    SectionNode,
    iter_sections,
    rebuild_section_tree,
)
from .reasoning_reranker import LexicalRelevanceScorer, RerankScorer

logger = logging.getLogger(__name__)

#: Retrieval-strategy name — the sibling to ``hybrid`` / ``semantic`` strategies,
#: selected for long-single-document queries (CONCEPT:AU-KG.retrieval.tree-navigation).
STRATEGY_NAME = "hierarchical_document"


@dataclass
class SectionMatch:
    """A relevant section with its cited source range.

    CONCEPT:AU-KG.retrieval.tree-navigation. ``char_start``/``char_end`` cite the
    node's own span (fetchable via the ``get_page_content`` verb); ``path`` is the
    title breadcrumb from the root so the answer is traceable to its place in the
    document's structure.
    """

    node_id: str
    title: str
    score: float
    char_start: int
    char_end: int
    summary: str = ""
    page_start: int | None = None
    page_end: int | None = None
    path: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "node_id": self.node_id,
            "title": self.title,
            "score": round(self.score, 6),
            "char_start": self.char_start,
            "char_end": self.char_end,
            "range": f"{self.char_start}..{self.char_end}",
            "summary": self.summary,
            "path": list(self.path),
        }
        if self.page_start is not None:
            out["page_start"] = self.page_start
            out["page_end"] = self.page_end
        return out


class HierarchicalDocumentRetriever:
    """Walk a document's section tree by node relevance (CONCEPT:AU-KG.retrieval.tree-navigation).

    Args:
        engine: Optional graph engine (``query_cypher`` or ``backend.execute``)
            used by :meth:`load_tree` to pull a document's ``Section`` nodes. When
            ``None`` the retriever operates purely on an in-memory tree (offline).
        scorer: The node relevance scorer; defaults to the dependency-free
            :class:`LexicalRelevanceScorer`.
        llm_fn: Optional ``(prompt) -> completion`` enabling LLM navigation.
    """

    STRATEGY = STRATEGY_NAME

    def __init__(
        self,
        engine: Any = None,
        *,
        scorer: RerankScorer | None = None,
        llm_fn: Any = None,
    ) -> None:
        self.engine = engine
        self.scorer: RerankScorer = scorer or LexicalRelevanceScorer()
        self.llm_fn = llm_fn

    # ── tree loading ─────────────────────────────────────────────────────

    def load_tree(self, document_id: str) -> list[SectionNode]:
        """Load a document's section tree from the graph (empty when absent).

        Reads the document's ``Section`` nodes and rebuilds the nesting from each
        node's stored ``parent_id`` (CONCEPT:AU-KG.retrieval.section-tree).
        """
        rows = self._section_rows(document_id)
        return rebuild_section_tree(rows) if rows else []

    def _section_rows(self, document_id: str) -> list[dict[str, Any]]:
        if self.engine is None or not document_id:
            return []
        cypher = (
            f"MATCH (s:{SECTION_NODE_TYPE}) WHERE s.document_id = $doc "
            "RETURN s.id AS id, s.tree_node_id AS tree_node_id, "
            "s.parent_id AS parent_id, "
            "s.title AS title, s.level AS level, s.char_start AS char_start, "
            "s.char_end AS char_end, s.line_start AS line_start, "
            "s.page_start AS page_start, s.page_end AS page_end, "
            "s.summary AS summary, s.content AS content"
        )
        return _rows(self.engine, cypher, {"doc": document_id})

    # ── retrieval ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        *,
        document_id: str = "",
        tree: list[SectionNode] | None = None,
        top_k: int = 5,
        beam_width: int = 3,
        instruction: str = "",
        use_llm: bool = False,
    ) -> list[SectionMatch]:
        """Return the most relevant sections with cited ranges.

        Args:
            query: The natural-language query.
            document_id: Load the tree from the graph for this document (ignored
                when ``tree`` is supplied).
            tree: An in-memory section tree (roots) to search directly.
            top_k: Max sections to return.
            beam_width: Sibling fan kept per level during the lexical walk.
            instruction: Optional task/intent for instruction-aware scoring.
            use_llm: Try LLM navigation first (falls back to the lexical walk).
        """
        roots = tree if tree is not None else self.load_tree(document_id)
        if not roots:
            return []

        if use_llm and self.llm_fn is not None:
            matches = self._navigate_llm(query, roots, top_k=top_k)
            if matches:
                return matches

        return self._walk(
            query,
            roots,
            top_k=top_k,
            beam_width=beam_width,
            instruction=instruction,
        )

    def _walk(
        self,
        query: str,
        roots: list[SectionNode],
        *,
        top_k: int,
        beam_width: int,
        instruction: str,
    ) -> list[SectionMatch]:
        """Beam walk: score node title+summary, descend only into promising subtrees.

        Every node that survives a kept branch is a candidate; the final ranking
        is by relevance across all visited candidates (PageIndex tree navigation
        with a deterministic lexical scorer instead of an LLM).
        """
        scored: list[tuple[float, list[str], SectionNode]] = []

        def _score(node: SectionNode) -> float:
            return self.scorer.score(query, _node_text(node), instruction)

        def _descend(nodes: list[SectionNode], path: list[str]) -> None:
            ranked = sorted(nodes, key=_score, reverse=True)
            kept = ranked[:beam_width] if beam_width > 0 else ranked
            for node in kept:
                node_path = [*path, node.title]
                scored.append((_score(node), node_path, node))
                if node.children:
                    _descend(node.children, node_path)

        _descend(roots, [])
        scored.sort(key=lambda t: t[0], reverse=True)

        out: list[SectionMatch] = []
        seen: set[str] = set()
        for score, path, node in scored:
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            out.append(_to_match(node, score, path[:-1]))
            if len(out) >= top_k:
                break
        return out

    def _navigate_llm(
        self, query: str, roots: list[SectionNode], *, top_k: int
    ) -> list[SectionMatch]:
        """LLM navigation over the text-free structure (PageIndex ``query_agent``)."""
        by_id = {n.node_id: n for n in iter_sections(roots)}
        structure = json.dumps(structure_view(roots), ensure_ascii=False)
        prompt = _NAV_PROMPT.format(
            query=query, structure=structure[:_MAX_STRUCT_CHARS]
        )
        try:
            out = self.llm_fn(prompt)
        except Exception as exc:  # noqa: BLE001 — fall back to lexical walk
            logger.debug("[tree-navigation] llm_fn failed: %s", exc)
            return []
        node_ids = _parse_node_ids(out)
        path_of = _paths(roots)
        matches: list[SectionMatch] = []
        for nid in node_ids:
            node = by_id.get(nid)
            if node is None:
                continue
            matches.append(_to_match(node, 1.0, path_of.get(nid, [])[:-1]))
            if len(matches) >= top_k:
                break
        return matches


# ── module helpers ────────────────────────────────────────────────────────────


def structure_view(roots: list[SectionNode]) -> list[dict[str, Any]]:
    """Text-free tree view: ``{node_id, title, summary, range, children}`` (map).

    CONCEPT:AU-KG.retrieval.tree-navigation — the token-cheap map an agent (or an
    LLM navigator) reasons over before fetching any body text (PageIndex
    ``get_document_structure`` = the tree with ``text`` removed).
    """

    def _one(node: SectionNode) -> dict[str, Any]:
        view: dict[str, Any] = {
            "node_id": node.node_id,
            "title": node.title,
            "summary": node.summary,
            "range": f"{node.char_start}..{node.char_end}",
        }
        if node.page_start is not None:
            view["pages"] = f"{node.page_start}-{node.page_end}"
        if node.children:
            view["children"] = [_one(c) for c in node.children]
        return view

    return [_one(r) for r in roots]


def content_for_ranges(
    roots: list[SectionNode], ranges: list[tuple[int, int]]
) -> list[dict[str, Any]]:
    """Return section bodies whose span intersects any requested char range.

    CONCEPT:AU-KG.retrieval.tree-navigation — the fetch half of map-then-fetch
    (PageIndex ``get_page_content``): the agent reads the structure, then asks for
    the exact ranges it wants. Sections are returned in document order.
    """
    hits: list[dict[str, Any]] = []
    for node in iter_sections(roots):
        for lo, hi in ranges:
            if node.char_start < hi and node.char_end > lo:
                hits.append(
                    {
                        "node_id": node.node_id,
                        "title": node.title,
                        "char_start": node.char_start,
                        "char_end": node.char_end,
                        "content": node.text,
                    }
                )
                break
    hits.sort(key=lambda h: h["char_start"])
    return hits


#: Cap on section body used for scoring when a node has no summary — keeps the
#: lexical walk cheap while still giving it real signal on unsummarized trees.
_SCORE_TEXT_CHARS = 600


def _node_text(node: SectionNode) -> str:
    """Scoreable text for a node — the text-free map surface (title + summary).

    Prefers ``title + summary`` (the navigation map). When a node carries no
    summary (e.g. a tree built without summarization, or loaded from the graph),
    it falls back to a bounded slice of the node's own body so the lexical walk
    still has content-word signal to rank on.
    """
    if node.summary:
        return f"{node.title} {node.summary}"
    if node.text:
        return f"{node.title} {node.text[:_SCORE_TEXT_CHARS]}"
    return node.title


def _to_match(node: SectionNode, score: float, path: list[str]) -> SectionMatch:
    return SectionMatch(
        node_id=node.node_id,
        title=node.title,
        score=float(score),
        char_start=node.char_start,
        char_end=node.char_end,
        summary=node.summary,
        page_start=node.page_start,
        page_end=node.page_end,
        path=path,
    )


def _paths(roots: list[SectionNode]) -> dict[str, list[str]]:
    """Map every ``node_id`` → its title breadcrumb from the root."""
    out: dict[str, list[str]] = {}

    def _walk(nodes: list[SectionNode], prefix: list[str]) -> None:
        for n in nodes:
            p = [*prefix, n.title]
            out[n.node_id] = p
            _walk(n.children, p)

    _walk(roots, [])
    return out


def _rows(engine: Any, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Run a read-only query, tolerant of either engine read API; never raises."""
    try:
        qc = getattr(engine, "query_cypher", None)
        if callable(qc):
            return list(qc(cypher, params) or [])
        backend = getattr(engine, "backend", None)
        if backend is not None and hasattr(backend, "execute"):
            return list(backend.execute(cypher, params) or [])
    except Exception:  # pragma: no cover - read best-effort by design
        return []
    return []


def _parse_node_ids(out: str | None) -> list[str]:
    """Extract a JSON list of node ids from an LLM completion (best-effort)."""
    if not out:
        return []
    raw = out[out.find("[") : out.rfind("]") + 1] if "[" in out else ""
    try:
        data = json.loads(raw) if raw else []
    except Exception:  # noqa: BLE001
        return []
    return [str(x).zfill(4) for x in data if isinstance(data, list) and x is not None]


_NAV_PROMPT = (
    "You navigate a document by its table of contents to answer a query. Given the "
    "query and the tree of sections (each with a node_id, title and summary), return "
    "a JSON list of the node_id strings of the sections most likely to contain the "
    "answer, most relevant first. Return ONLY the JSON list.\n\n"
    "Query: {query}\n\n<structure>\n{structure}\n</structure>"
)
_MAX_STRUCT_CHARS = 16_000


def build_hierarchical_retriever(
    engine: Any = None, *, llm_fn: Any = None
) -> HierarchicalDocumentRetriever:
    """Factory for the ``hierarchical_document`` retrieval strategy.

    CONCEPT:AU-KG.retrieval.tree-navigation — the registration seam mirroring how
    the other retrievers are instantiated; the MCP/REST document-tree verb and the
    ingestion path build the retriever through this.
    """
    return HierarchicalDocumentRetriever(engine, llm_fn=llm_fn)


__all__ = [
    "HierarchicalDocumentRetriever",
    "SectionMatch",
    "STRATEGY_NAME",
    "structure_view",
    "content_for_ranges",
    "build_hierarchical_retriever",
]
