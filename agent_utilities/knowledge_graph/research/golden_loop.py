"""The self-evolution "golden loop" controller (propose-only v1).

CONCEPT:KG-2.7 / KG-2.10 — research assimilation + orchestration synthesis.

Composes existing primitives into one cycle that makes the KG self-improving
WITHOUT auto-merging anything (propose-only):

    intake  → unresolved ``Concept`` topics (no ``ADDRESSED_BY``)
    acquire → semantically related sources for each topic (research/search)
    resolve → ``ADDRESSES`` edges source→topic so the loop converges
    distill → ``SpecDraft`` markdown into ``.specify/specs/kg-distilled/`` (gated)
    synth   → a ``TeamSpec``/``AgentSpec`` proposal persisted to the KG

Every artifact is a DRAFT/proposal: spec markdown under ``.specify/`` and KG
proposal nodes. No code execution, no PR merge, no edits outside ``.specify``.
Exposed on-demand (skill-workflow / MCP) and via a throttled daemon tick.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..adaptation.topic_resolver import mark_addressed, unresolved_topics
from .search import acquire_for_topic

logger = logging.getLogger(__name__)


class GoldenLoopController:
    """Run one propose-only self-evolution cycle over the KG."""

    def __init__(
        self,
        engine: Any,
        *,
        codebase_root: str | None = None,
        propose_only: bool = True,
    ) -> None:
        self.engine = engine
        self.codebase_root = codebase_root or os.getenv("WORKSPACE_PATH") or "."
        # propose_only is always True in v1 — kept explicit so a future
        # human-approved apply path is a deliberate flip, never accidental.
        self.propose_only = propose_only

    # ------------------------------------------------------------------
    def _capability_search(self):
        """Build a ``(query, top_k) -> list[dict]`` capability search fn."""
        backend = getattr(self.engine, "backend", None)
        search = getattr(backend, "semantic_search", None)
        if not callable(search):
            return None
        from ..enrichment.semantic import make_embed_fn

        embed = make_embed_fn()

        def _fn(query: str, top_k: int = 5) -> list[dict[str, Any]]:
            try:
                return search(embed([query])[0], top_k) or []
            except Exception:  # noqa: BLE001
                return []

        return _fn

    def run_one_cycle(
        self,
        *,
        max_topics: int = 5,
        distill: bool | None = None,
        synthesize: bool = True,
    ) -> dict[str, Any]:
        """Execute one cycle. Returns a structured, JSON-able report.

        ``distill`` defaults to the ``KG_GOLDEN_DISTILL`` env (off by default —
        it's LLM + full-graph heavy). All stages are best-effort: one failing
        stage never aborts the cycle.
        """
        if distill is None:
            distill = os.getenv("KG_GOLDEN_DISTILL", "0") == "1"

        report: dict[str, Any] = {
            "propose_only": self.propose_only,
            "topics_intake": 0,
            "topics_resolved": 0,
            "sources_linked": 0,
            "spec_drafts": [],
            "team": None,
            "errors": [],
        }

        # 1. INTAKE — open topics the loop should address.
        topics = unresolved_topics(self.engine, max_topics)
        report["topics_intake"] = len(topics)
        if not topics:
            return report

        # 2–3. ACQUIRE related sources + RESOLVE (ADDRESSES) so the loop converges.
        for t in topics:
            try:
                srcs = acquire_for_topic(self.engine, t)
                if srcs:
                    n = mark_addressed(self.engine, t["id"], srcs, source="golden_loop")
                    if n:
                        report["topics_resolved"] += 1
                        report["sources_linked"] += n
            except Exception as e:  # noqa: BLE001
                report["errors"].append(f"acquire/resolve {t.get('id')}: {e}")

        # 4. DISTILL spec drafts (gated; propose-only → .specify/).
        if distill:
            try:
                report["spec_drafts"] = self._distill_specs(topics)
            except Exception as e:  # noqa: BLE001
                report["errors"].append(f"distill: {e}")

        # 5. SYNTHESIZE a team proposal for the open topics (propose-only).
        if synthesize:
            try:
                report["team"] = self._synthesize_team(topics)
            except Exception as e:  # noqa: BLE001
                report["errors"].append(f"synthesize: {e}")

        return report

    # ------------------------------------------------------------------
    def _distill_specs(self, topics: list[dict[str, Any]]) -> list[str]:
        """Distil ``SpecDraft`` markdown into ``.specify/specs/kg-distilled/``."""
        from ..enrichment.cards import make_lite_llm_fn
        from ..enrichment.distill import what_specs_could_we_build, write_spec_drafts
        from ..enrichment.extractors.document import Concept

        # Bounded inputs: the intake topics as concepts; edges/code maps left
        # empty so distillation stays cheap (candidates come from concept value).
        concepts = [
            Concept(id=t["id"], name=t["name"], kind="topic", summary="", source_ids=[])
            for t in topics
        ]
        specs = what_specs_could_we_build(
            self.codebase_root, concepts, [], {}, make_lite_llm_fn(), limit=3
        )
        if not specs:
            return []
        # propose_only: write DRAFTS under .specify/ only.
        return write_spec_drafts(specs, self.codebase_root)

    def _synthesize_team(self, topics: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Synthesize a team proposal addressing the open topics; persist nodes."""
        cap = self._capability_search()
        if cap is None:
            return None
        from ..enrichment.cards import make_lite_llm_fn
        from ..enrichment.synthesize import persist_synthesis, synthesize_team

        names = ", ".join(t["name"] for t in topics[:5]) or "open KG topics"
        goal = f"Propose how to address these open knowledge-graph topics: {names}"
        team, members = synthesize_team(goal, cap, make_lite_llm_fn(), max_members=4)
        nodes = edges = 0
        if self.propose_only:
            # Persist the PROPOSAL (TeamSpec/AgentSpec nodes) — not executed.
            try:
                nodes, edges = persist_synthesis(self.engine.backend, team, *members)
            except Exception as e:  # noqa: BLE001
                logger.debug("persist_synthesis failed: %s", e)
        return {
            "goal": goal,
            "lead": getattr(team, "lead", None) or getattr(team, "name", None),
            "members": [getattr(m, "name", "?") for m in members],
            "persisted_nodes": nodes,
            "persisted_edges": edges,
        }


def run_golden_loop_cycle(engine: Any = None, **kwargs: Any) -> dict[str, Any]:
    """Convenience entry: run one cycle against the active (or given) engine."""
    if engine is None:
        from ..core.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine.get_active() or IntelligenceGraphEngine()
    return GoldenLoopController(engine).run_one_cycle(**kwargs)
