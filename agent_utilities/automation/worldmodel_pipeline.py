"""Relevance-gated FreshRSS → KG world-model ingestion (CONCEPT:KG-2.116).

The news/finance/tech sibling of the KG-2.6 research pipeline. Where the research
pipeline acquires academic papers, this builds a *world model* from curated RSS
items: an AI agent curates which FreshRSS feeds are subscribed (the
``freshrss_subscriptions`` MCP tool), and this runner decides which *items* are
worth ingesting — only those relevant to the existing KG (taxonomy score OR
concept-novelty), or that the agent explicitly deems worthy (``agent_force``).

Tiering (per item drained from the ``freshrss`` mcp_tool preset):

* **relevant** — score ≥ threshold AND not already-covered → full native ingest via
  the KG-2.48 ``DocumentProcessor`` (Document + Chunk + downstream OWL enrichment).
* **marginal** — score ≥ marginal threshold, or highly novel but low score → a
  lightweight ``ArticleNode`` footprint so the item is remembered, not re-scored.
* **skipped** — below threshold and not novel.
* **research** — items under a "Research (ScholarX)" / arXiv feed route to the
  research path instead, unifying RSS intake (CONCEPT:KG-2.117).

Reuses ``score_text`` (world-model taxonomy profile) and ``concept_novelty`` from
``research_pipeline`` — one scorer, two profiles. Best-effort throughout: a single
bad item never aborts the sweep, and a missing engine/embedder degrades gracefully.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .research_pipeline import WORLD_MODEL_TAXONOMY, concept_novelty, score_text

logger = logging.getLogger(__name__)


@dataclass
class WorldModelConfig:
    """Gate thresholds for world-model ingestion (CONCEPT:KG-2.116)."""

    relevant_threshold: float = 3.0
    marginal_threshold: float = 1.0
    # An item the keyword score misses but that is highly novel AND touches the KG
    # still earns a marginal footprint.
    novelty_floor: float = 0.6
    # Below this novelty an otherwise-relevant item is demoted to marginal (we
    # already model it) — mirrors the research pipeline's 0.25 redundancy gate.
    redundancy_floor: float = 0.25
    # Concept-novelty (the embedding probe vs the KG concept registry) is a
    # refinement on top of the keyword score: it demotes already-covered items and
    # rescues highly-novel low-score ones. It requires a populated registry with
    # stored embeddings; building that index can be expensive, so it is OPT-IN
    # (FRESHRSS_USE_NOVELTY / use_novelty). Off → tier on keyword score alone (the
    # taxonomy + KG-grounded company terms still ground "relevant to the KG").
    use_novelty: bool = False


@dataclass
class WorldModelReport:
    """Outcome counts for one gated sweep."""

    items_seen: int = 0
    relevant: int = 0
    marginal: int = 0
    research: int = 0
    skipped: int = 0
    domains: dict[str, int] = field(default_factory=dict)

    @property
    def ingested(self) -> int:
        return self.relevant + self.marginal + self.research


@dataclass
class WorldModelPipelineRunner:
    """Gate + ingest curated FreshRSS items into the KG world model (KG-2.116)."""

    engine: Any = None
    config: WorldModelConfig = field(default_factory=WorldModelConfig)

    # ── public entrypoint ────────────────────────────────────────────────────

    def run_gated_ingest(self, docs: list[Any]) -> WorldModelReport:
        """Score, tier, and ingest a batch of drained ``SourceDocument`` items."""
        report = WorldModelReport(items_seen=len(docs))
        taxonomy = self._live_taxonomy()
        for doc in docs:
            try:
                self._gate_one(doc, taxonomy, report)
            except Exception:  # noqa: BLE001 — one bad item never aborts the sweep
                logger.debug(
                    "world-model gate failed for %s",
                    getattr(doc, "id", "?"),
                    exc_info=True,
                )
                report.skipped += 1
        return report

    def _gate_one(
        self, doc: Any, taxonomy: dict[str, dict[str, Any]], report: WorldModelReport
    ) -> None:
        rec = (getattr(doc, "metadata", None) or {}).get("record") or {}
        if self._is_known(doc, rec):
            report.skipped += 1
            return

        # Unified intake: arXiv/Research-feed items take the research path (KG-2.117).
        if self._is_research(rec):
            self._ingest_research(doc, rec)
            report.research += 1
            return

        forced = bool(rec.get("agent_force"))
        title = getattr(doc, "title", "") or ""
        text = getattr(doc, "text", "") or ""
        score, domains = score_text(title, text, taxonomy)
        # Novelty is opt-in and only consulted for items the score already makes
        # ingestable/borderline (it can only demote a relevant item or rescue a
        # novel low-score one) — never pay the embedding probe for the skip-majority.
        novelty = None
        if self.config.use_novelty and (score > 0 or forced):
            novelty = concept_novelty(
                self.engine, f"{title} — {text[:2000]}", holder=self
            )
        tier = self._tier(score, novelty, forced)

        if tier == "relevant":
            self._ingest_full(doc, rec, score, domains)
            report.relevant += 1
        elif tier == "marginal":
            self._ingest_marginal(doc, rec, score, domains)
            report.marginal += 1
        else:
            report.skipped += 1
            return
        for d in domains:
            report.domains[d] = report.domains.get(d, 0) + 1

    # ── tiering ──────────────────────────────────────────────────────────────

    def _tier(self, score: float, novelty: float | None, forced: bool) -> str:
        if forced:
            return "relevant"
        if score >= self.config.relevant_threshold and (
            novelty is None or novelty >= self.config.redundancy_floor
        ):
            return "relevant"
        if score >= self.config.marginal_threshold:
            return "marginal"
        if novelty is not None and novelty >= self.config.novelty_floor and score > 0:
            return "marginal"
        return "skipped"

    def _live_taxonomy(self) -> dict[str, dict[str, Any]]:
        """World-model taxonomy with the ``companies`` set augmented from the KG.

        Articles that mention an organization we already model score higher — the
        "relevant to existing KG data" requirement made concrete and cheap.
        """
        taxonomy = copy.deepcopy(WORLD_MODEL_TAXONOMY)
        graph = getattr(self.engine, "graph", None)
        if graph is None:
            return taxonomy
        names: set[str] = set()
        try:
            for _nid, data in graph.nodes(data=True):
                ntype = str(data.get("type", "")).lower()
                name = data.get("name")
                if name and ("organization" in ntype):
                    names.add(str(name).lower())
        except Exception:  # noqa: BLE001 — augmentation is best-effort
            return taxonomy
        if names:
            existing = set(taxonomy["companies"]["keywords"])
            taxonomy["companies"]["keywords"] = sorted(existing | names)
        return taxonomy

    # ── routing helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _is_research(rec: dict[str, Any]) -> bool:
        """A GReader item belongs to the research path if its feed/category is arXiv/ScholarX."""
        cats = rec.get("categories") or []
        blob_parts: list[str] = []
        for c in cats:
            if isinstance(c, dict):
                blob_parts.append(str(c.get("label") or c.get("id") or ""))
            else:
                blob_parts.append(str(c))
        origin = rec.get("origin") or {}
        blob_parts.append(str(origin.get("htmlUrl", "")))
        blob_parts.append(str(origin.get("streamId", "")))
        blob = " ".join(blob_parts).lower()
        return any(k in blob for k in ("scholarx", "arxiv", "research"))

    @staticmethod
    def _node_id(prefix: str, raw: str) -> str:
        safe = str(raw).replace(":", "-").replace("/", "-")
        return f"{prefix}{safe[:160]}"

    @staticmethod
    def _canonical(rec: dict[str, Any]) -> str:
        canonical = rec.get("canonical") or []
        if isinstance(canonical, list) and canonical and isinstance(canonical[0], dict):
            return str(canonical[0].get("href") or "")
        origin = rec.get("origin") or {}
        return str(origin.get("htmlUrl") or "")

    @staticmethod
    def _arxiv_id(rec: dict[str, Any]) -> str:
        """Canonical ``arxiv:<id>`` from a feed record, or ``""``.

        Lets the SAME arXiv paper arriving via native ScholarX RSS AND via FreshRSS
        collapse to one node (CONCEPT:KG-2.121 dedup convergence).
        """
        import re as _re

        origin = rec.get("origin") or {}
        blob = " ".join(
            str(x)
            for x in (
                rec.get("id", ""),
                WorldModelPipelineRunner._canonical(rec),
                origin.get("htmlUrl", ""),
            )
        )
        m = _re.search(r"(\d{4}\.\d{4,5})", blob)
        return f"arxiv:{m.group(1)}" if m else ""

    def _is_known(self, doc: Any, rec: dict[str, Any] | None = None) -> bool:
        graph = getattr(self.engine, "graph", None)
        if graph is None:
            return False
        rid = getattr(doc, "id", "")
        keys = [
            self._node_id(p, rid)
            for p in ("doc:freshrss:", "news:freshrss:", "doc:scholarx:")
        ]
        aid = self._arxiv_id(rec or {})
        if aid:
            safe = aid.replace(":", "-")
            # The research path lands papers at article:scholarx:<safe> (full/marginal).
            keys.append(f"article:scholarx:{safe}")
            keys.append(self._node_id("doc:scholarx:", aid))
        try:
            nodes = graph.nodes
            return any(k in nodes for k in keys)
        except Exception:  # noqa: BLE001
            return False

    # ── ingestion tiers ──────────────────────────────────────────────────────

    def _processor(self) -> Any:
        proc = getattr(self, "_proc", None)
        if proc is None:
            from ..knowledge_graph.ontology.document_processing import (
                ChunkingConfig,
                DocumentProcessor,
            )

            proc = DocumentProcessor(
                getattr(self.engine, "backend", None),
                chunking=ChunkingConfig(),
                contextual=True,
            )
            self._proc = proc
        return proc

    def _ingest_full(
        self, doc: Any, rec: dict[str, Any], score: float, domains: list[str]
    ) -> None:
        """Native KG-2.48 Document + Chunk ingestion of the full article body."""
        if self.engine is None:
            return
        url = self._canonical(rec) or getattr(doc, "source_uri", "")
        origin = rec.get("origin") or {}
        item_node_id = self._node_id("doc:freshrss:", doc.id)
        try:
            self._processor().process(
                getattr(doc, "text", "") or "",
                document_id=item_node_id,
                title=getattr(doc, "title", "") or doc.id,
                doc_type="news_article",
                source=url,
                metadata={
                    **(getattr(doc, "metadata", None) or {}),
                    "source_system": "freshrss",
                    "importance_score": 0.8,
                    "relevance_score": score,
                    "domains": domains,
                    "feed": origin.get("title"),
                    "published": rec.get("published"),
                },
            )
            self._link_feed_source(item_node_id, doc, rec)
        except Exception as exc:  # noqa: BLE001 — degrade to a marginal footprint
            logger.warning("[KG-2.116] full ingest failed for %s: %s", doc.id, exc)
            self._ingest_marginal(doc, rec, score, domains)

    def _ingest_marginal(
        self, doc: Any, rec: dict[str, Any], score: float, domains: list[str]
    ) -> None:
        """A lightweight ``ArticleNode`` footprint so the item is remembered."""
        if self.engine is None or getattr(self.engine, "graph", None) is None:
            return
        from ..models.knowledge_graph import ArticleNode

        article_id = self._node_id("news:freshrss:", doc.id)
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        body = getattr(doc, "text", "") or ""
        node = ArticleNode(
            id=article_id,
            name=getattr(doc, "title", "") or doc.id,
            description=body[:500],
            summary=body[:500],
            content=body[:2000],
            importance_score=0.5,
            timestamp=timestamp,
            tags=domains or [],
        )
        self.engine.graph.add_node(article_id, **node.model_dump())
        if getattr(self.engine, "backend", None):
            try:
                self.engine._upsert_node(
                    "Article",
                    article_id,
                    self.engine._serialize_node(node, "Article"),
                )
            except Exception:  # noqa: BLE001
                logger.debug("[KG-2.116] marginal upsert failed for %s", doc.id)
        self._link_feed_source(article_id, doc, rec)

    def _ingest_research(self, doc: Any, rec: dict[str, Any]) -> None:
        """Route a Research/arXiv item to the unified research path (CONCEPT:KG-2.117/2.121).

        Grades the item and enqueues a prioritized ``research_paper_fetch`` task (the
        KG-2.114 path) — so a research item from ANY feed (native RSS, ScholarX,
        FreshRSS-arXiv) is graded and fetched the SAME way, best-graded first. The
        node-id keys off the canonical arXiv id so duplicates across feeds collapse.
        """
        if self.engine is None:
            return
        from ..knowledge_graph.research.feed_grading import grade_and_enqueue_paper

        aid = self._arxiv_id(rec) or getattr(doc, "id", "")
        url = self._canonical(rec) or getattr(doc, "source_uri", "")
        paper = {
            "id": aid,
            "title": getattr(doc, "title", "") or doc.id,
            "abstract": getattr(doc, "text", "") or "",
            "authors": rec.get("authors", []),
            "url": url,
            "pdf_url": rec.get("pdf_url", ""),
        }
        try:
            grade_and_enqueue_paper(self.engine, paper)
        except Exception as exc:  # noqa: BLE001 — one item never aborts the sweep
            logger.warning("[KG-2.117] research grade/enqueue failed for %s: %s", aid, exc)

    def _link_feed_source(self, item_node_id: str, doc: Any, rec: dict[str, Any]) -> None:
        """Best-effort ``:ingestedFrom`` edge item → its :FeedSource node (KG-2.122)."""
        engine = self.engine
        if engine is None or not hasattr(engine, "add_edge"):
            return
        from .feed_sources import _feed_node_id

        system = str((getattr(doc, "metadata", None) or {}).get("source_system") or "")
        stream = str((rec.get("origin") or {}).get("streamId") or "")
        if system == "rss" and stream:
            feed_id = _feed_node_id("rss", stream)
        elif system == "freshrss":
            feed_id = _feed_node_id("freshrss", "freshrss")
        else:
            return  # only link where the FeedSource node is deterministically registered
        try:
            engine.add_edge(item_node_id, feed_id, "INGESTED_FROM")
        except Exception:  # noqa: BLE001 — provenance link is best-effort
            logger.debug("feed-source link failed for %s", item_node_id)
