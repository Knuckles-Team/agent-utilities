#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:ORCH-1.2 — Global Workspace Attention.

Always-on attention mechanism for specialist output quality filtering.
Inspired by Global Workspace Theory (GWT), adaptive_agent_router submit proposals
that are scored and ranked before integration into the final response.

Architecture:
    - Specialists produce outputs normally
    - Each output is wrapped as a ``Proposal`` with tri-score:
      relevance (embedding similarity), confidence (self-reported),
      and track record (from self-model CONCEPT:KG-2.1)
    - Top-K proposals are selected and broadcast to the KG
    - Low-scoring proposals are filtered out

Cost: ~50ms per query (embedding comparison + sort). No LLM round-trip.
Always-on for consistent quality improvement.

Integrates with:
    - CONCEPT:KG-2.0 (OGM): Proposal persistence via ``KGMapper``
    - CONCEPT:KG-2.1 (Self-Model): Track record scoring
    - Existing engine: ``cosine_similarity()`` for relevance scoring

See docs/pillars/architecture_c4.md §CONCEPT:ORCH-1.2
"""


import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ..knowledge_graph.core.engine import cosine_similarity
from ..knowledge_graph.core.ogm import KGMapper
from ..models.knowledge_graph import ProposalNode, RegistryEdgeType

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine
    from ..knowledge_graph.retrieval.memory_retriever import MemoryRetriever

logger = logging.getLogger(__name__)


# ── GWT loop telemetry ────────────────────────────────────────────────
# The write side (``broadcast_to_kg``) and the read side (``get_attention_score``)
# must operate on the *same* engine instance for the loop to reinforce. If they
# ever hold different engines (e.g. per-subsystem sharding), broadcasts are written
# but no read ever resolves one — silently. These process-wide counters surface
# that: ``suspected_engine_mismatch`` flips True when proposals were broadcast yet
# reads keep missing with zero hits. CONCEPT:ORCH-1.2.
_MISMATCH_WARN_AFTER_MISSES = 3
# Strict mode (tests/CI): raise instead of warn when a mismatch is detected.
_STRICT = os.getenv("AGENT_UTILITIES_GWT_STRICT", "").lower() in ("1", "true", "yes")


@dataclass
class _GwtTelemetry:
    broadcasts_written: int = 0
    attention_reads: int = 0
    attention_hits: int = 0
    attention_misses: int = 0
    warned_mismatch: bool = False


_TELEMETRY = _GwtTelemetry()


def workspace_attention_telemetry() -> dict[str, int | bool]:
    """Process-wide GWT loop counters (CONCEPT:ORCH-1.2).

    ``suspected_engine_mismatch`` is True when proposals were broadcast and reads
    have happened but *none* resolved a proposal — the signature of the writer and
    reader holding different engine instances.
    """
    t = _TELEMETRY
    return {
        "broadcasts_written": t.broadcasts_written,
        "attention_reads": t.attention_reads,
        "attention_hits": t.attention_hits,
        "attention_misses": t.attention_misses,
        "suspected_engine_mismatch": (
            t.broadcasts_written > 0 and t.attention_reads > 0 and t.attention_hits == 0
        ),
    }


def reset_workspace_attention_telemetry() -> None:
    """Reset the GWT counters (intended for tests)."""
    global _TELEMETRY
    _TELEMETRY = _GwtTelemetry()


class Proposal(BaseModel):
    """A specialist's output proposal competing for broadcast.

    Used internally by ``WorkspaceAttention`` to score and rank
    specialist contributions before they are integrated.
    """

    specialist_id: str
    specialist_name: str = ""
    output: str
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    track_record_score: float = Field(default=0.5, ge=0.0, le=1.0)
    composite_score: float = Field(default=0.0, ge=0.0, le=1.0)


class WorkspaceAttention:
    """Always-on attention mechanism for specialist output quality.

    CONCEPT:ORCH-1.2 — Global Workspace Attention

    Instead of accepting all specialist outputs equally, this mechanism
    scores each output by relevance, confidence, and track record,
    then selects the top-K for integration into the final response.

    Scoring signals:
        1. **Relevance**: Embedding cosine similarity between the
           specialist's output and the original query
        2. **Confidence**: Self-reported confidence from the specialist
           (parsed from output if available, otherwise 0.5)
        3. **Track record**: Historical success rate from the persistent
           self-model (CONCEPT:KG-2.1)

    Composite score: ``0.5 * relevance + 0.3 * track_record + 0.2 * confidence``

    Args:
        max_broadcast_slots: Maximum number of proposals to select.
        relevance_weight: Weight for relevance scoring (default 0.5).
        track_record_weight: Weight for track record scoring (default 0.3).
        confidence_weight: Weight for confidence scoring (default 0.2).
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        max_broadcast_slots: int = 5,
        relevance_weight: float = 0.5,
        track_record_weight: float = 0.3,
        confidence_weight: float = 0.2,
    ) -> None:
        # ``engine`` is the shared knowledge engine the GWT loop reads/writes
        # proposals through; optional so the pure scoring helpers work standalone.
        self.engine = engine
        self.max_broadcast_slots = max_broadcast_slots
        self.w_relevance = relevance_weight
        self.w_track_record = track_record_weight
        self.w_confidence = confidence_weight

    # ── Proposal Collection ───────────────────────────────────────────

    def collect_proposals(
        self,
        specialist_outputs: dict[str, str],
        query: str,
        engine: IntelligenceGraphEngine | None = None,
        memory_retriever: MemoryRetriever | None = None,
    ) -> list[Proposal]:
        """Convert specialist outputs into scored proposals.

        Args:
            specialist_outputs: Mapping of specialist_id → output text.
            query: The original user query for relevance scoring.
            engine: Optional engine for embedding-based relevance.
            memory_retriever: Optional self-model for track record scoring.

        Returns:
            List of scored ``Proposal`` objects, sorted by composite score.
        """
        engine = engine or self.engine
        proposals: list[Proposal] = []
        query_embedding: list[float] | None = None

        # Generate query embedding once
        if engine:
            try:
                from agent_utilities.core.embedding_utilities import (
                    create_embedding_model,
                )

                embed_model = create_embedding_model()
                query_embedding = embed_model.get_text_embedding(query)
            except Exception:
                pass  # Fall back to keyword matching # nosec B110

        for specialist_id, output in specialist_outputs.items():
            # 1. Relevance scoring
            relevance = self._score_relevance(output, query, query_embedding, engine)

            # 2. Confidence scoring (parse from output if available)
            confidence = self._extract_confidence(output)

            # 3. Track record scoring (from self-model)
            track_record = self._score_track_record(specialist_id, memory_retriever)

            # Composite score
            composite = (
                self.w_relevance * relevance
                + self.w_track_record * track_record
                + self.w_confidence * confidence
            )

            # Get specialist name
            name = specialist_id
            if engine and specialist_id in engine.graph:
                name = engine.graph.nodes[specialist_id].get("name", specialist_id)

            proposals.append(
                Proposal(
                    specialist_id=specialist_id,
                    specialist_name=name,
                    output=output,
                    relevance_score=relevance,
                    confidence_score=confidence,
                    track_record_score=track_record,
                    composite_score=composite,
                )
            )

        # Sort by composite score descending
        proposals.sort(key=lambda p: p.composite_score, reverse=True)

        logger.debug(
            "GWT collected %d proposals (top: %s = %.3f)",
            len(proposals),
            proposals[0].specialist_name if proposals else "none",
            proposals[0].composite_score if proposals else 0.0,
        )

        return proposals

    # ── Selection ─────────────────────────────────────────────────────

    def select_winners(self, proposals: list[Proposal]) -> list[Proposal]:
        """Select the top-K proposals by composite score.

        Args:
            proposals: Scored proposals from ``collect_proposals()``.

        Returns:
            The winning proposals (at most ``max_broadcast_slots``).
        """
        winners = proposals[: self.max_broadcast_slots]

        if len(proposals) > self.max_broadcast_slots:
            filtered = len(proposals) - len(winners)
            logger.info(
                "GWT selected %d/%d proposals (filtered %d low-quality)",
                len(winners),
                len(proposals),
                filtered,
            )

        # CONCEPT:ORCH-1.3 — named-aggregation consensus over winners' scores,
        # via the coordination layer's aggregation registry (STRATEGY synergy #2).
        if winners:
            logger.debug(
                "GWT winner consensus (mean composite) = %.3f",
                self.consensus_score(winners),
            )

        return winners

    def consensus_score(
        self, proposals: list[Proposal], operator: str = "mean"
    ) -> float:
        """Named-operator consensus over proposals' composite scores (CONCEPT:ORCH-1.3).

        Delegates to the coordination layer's aggregation registry so winner
        consensus, coordination aggregation, and selection share one taxonomy.
        """
        from .coordination import aggregate_scores

        return aggregate_scores([p.composite_score for p in proposals], operator)

    # ── KG Broadcast ──────────────────────────────────────────────────

    def broadcast_to_kg(
        self,
        winners: list[Proposal],
        engine: IntelligenceGraphEngine | None = None,
        task_id: str = "",
    ) -> list[str]:
        """Persist winning proposals to the KG for global visibility.

        Creates ``ProposalNode`` entries and links them to their adaptive_agent_router.
        These serve as training signal for the self-model (CONCEPT:KG-2.1) and are
        read back by :meth:`get_attention_score`.

        Args:
            winners: The selected winning proposals.
            engine: The engine for KG persistence (defaults to ``self.engine``).
            task_id: Optional task ID to link proposals to.

        Returns:
            List of persisted proposal node IDs (empty if no engine is available).
        """
        engine = engine or self.engine
        if engine is None:
            return []
        ogm = KGMapper(engine)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node_ids: list[str] = []

        for proposal in winners:
            node_id = f"prop:{uuid.uuid4().hex[:8]}"
            node = ProposalNode(
                id=node_id,
                name=f"Proposal: {proposal.specialist_name}",
                specialist_id=proposal.specialist_id,
                output=proposal.output[:1000],  # Truncate for storage
                relevance_score=proposal.relevance_score,
                confidence_score=proposal.confidence_score,
                track_record_score=proposal.track_record_score,
                composite_score=proposal.composite_score,
                selected=True,
                timestamp=ts,
            )
            ogm.upsert(node)

            # Link to specialist
            ogm.upsert_edge(
                node_id,
                proposal.specialist_id,
                RegistryEdgeType.PROPOSED_FOR,
            )

            node_ids.append(node_id)

        _TELEMETRY.broadcasts_written += len(node_ids)
        logger.debug("GWT broadcast %d proposals to KG", len(node_ids))
        return node_ids

    def select_and_broadcast(
        self,
        specialist_outputs: dict[str, str],
        query: str,
        *,
        engine: IntelligenceGraphEngine | None = None,
        memory_retriever: MemoryRetriever | None = None,
        task_id: str = "",
    ) -> list[Proposal]:
        """Run the full global-workspace loop on a set of specialist outputs.

        collect → score → select top-K winners → broadcast to the KG. This is the
        one-call live entry point the orchestrator invokes after a multi-agent wave;
        the broadcast it produces is what :meth:`get_attention_score` later reads
        back as each specialist's runtime standing (CONCEPT:ORCH-1.2 / KG-2.1).

        Returns the winning proposals (already persisted).
        """
        engine = engine or self.engine
        proposals = self.collect_proposals(
            specialist_outputs, query, engine=engine, memory_retriever=memory_retriever
        )
        if not proposals:
            return []
        winners = self.select_winners(proposals)
        self.broadcast_to_kg(winners, engine, task_id=task_id)
        return winners

    def get_attention_score(
        self, node_id: str, engine: IntelligenceGraphEngine | None = None
    ) -> float | None:
        """Runtime attention score for a specialist from prior broadcasts.

        Reads back the ``ProposalNode``\\ s written by :meth:`broadcast_to_kg` and
        returns the most recent **selected** proposal's composite score (∈[0, 1])
        for ``node_id`` — i.e. that specialist's current standing in the global
        workspace — or ``None`` when it has no broadcast history (callers then fall
        back to a neutral prior). CONCEPT:KG-2.1.
        """
        engine = engine or self.engine
        _TELEMETRY.attention_reads += 1
        graph = getattr(engine, "graph", None)
        best_score: float | None = None
        if graph is not None:
            try:
                node_iter = graph.nodes(data=True)
            except TypeError:  # graph.nodes is not a callable view
                node_iter = None
            if node_iter is not None:
                best_ts = ""
                for _nid, data in node_iter:
                    if not isinstance(data, dict):
                        continue
                    if (
                        data.get("specialist_id") != node_id
                        or "composite_score" not in data
                    ):
                        continue
                    if not data.get("selected", False):
                        continue
                    ts = str(data.get("timestamp", ""))
                    if best_score is None or ts >= best_ts:
                        best_ts, best_score = ts, float(data["composite_score"])
        if best_score is None:
            _TELEMETRY.attention_misses += 1
            self._maybe_flag_engine_mismatch()
        else:
            _TELEMETRY.attention_hits += 1
        return best_score

    @staticmethod
    def _maybe_flag_engine_mismatch() -> None:
        """Surface the write-but-never-read failure mode (CONCEPT:ORCH-1.2).

        When proposals have been broadcast but reads keep missing with zero hits, the
        writer and reader almost certainly hold different engine instances. Warn once
        (or raise under ``AGENT_UTILITIES_GWT_STRICT``).
        """
        t = _TELEMETRY
        suspect = (
            t.broadcasts_written > 0
            and t.attention_hits == 0
            and t.attention_misses >= _MISMATCH_WARN_AFTER_MISSES
        )
        if not suspect or t.warned_mismatch:
            return
        t.warned_mismatch = True
        msg = (
            f"WorkspaceAttention: {t.broadcasts_written} proposal(s) broadcast but "
            f"{t.attention_misses} read(s) all missed with 0 hits — the writer and "
            "reader likely hold different engine instances (CONCEPT:ORCH-1.2)."
        )
        if _STRICT:
            raise AssertionError(msg)
        logger.warning(msg)

    # ── Scoring Helpers ───────────────────────────────────────────────

    def _score_relevance(
        self,
        output: str,
        query: str,
        query_embedding: list[float] | None,
        engine: IntelligenceGraphEngine | None,
    ) -> float:
        """Score output relevance to the original query.

        Uses embedding cosine similarity if available, otherwise falls
        back to simple keyword overlap.
        """
        if query_embedding and engine:
            try:
                from agent_utilities.core.embedding_utilities import (
                    create_embedding_model,
                )

                embed_model = create_embedding_model()
                output_emb = embed_model.get_text_embedding(output[:500])
                return max(
                    0.0, min(1.0, cosine_similarity(query_embedding, output_emb))
                )
            except Exception:
                pass  # nosec B110

        # Keyword overlap fallback
        query_words = set(query.lower().split())
        output_words = set(output.lower().split())
        if not query_words:
            return 0.5
        overlap = len(query_words & output_words)
        return min(1.0, overlap / len(query_words))

    def _extract_confidence(self, output: str) -> float:
        """Extract self-reported confidence from specialist output.

        Looks for patterns like "Confidence: 0.8" or "I'm 85% sure".
        Falls back to 0.5 if no confidence signal is found.
        """
        import re

        # Pattern 1: "Confidence: X.X" or "confidence: X.X"
        match = re.search(r"[Cc]onfidence:\s*([\d.]+)", output)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass

        # Pattern 2: "X% sure" or "X% confident"
        match = re.search(r"(\d+)%\s*(?:sure|confident|certain)", output)
        if match:
            try:
                return max(0.0, min(1.0, int(match.group(1)) / 100.0))
            except ValueError:
                pass

        return 0.5  # Default neutral confidence

    def _score_track_record(
        self,
        specialist_id: str,
        memory_retriever: MemoryRetriever | None,
    ) -> float:
        """Score a specialist's historical track record.

        Uses the persistent self-model (CONCEPT:KG-2.1) if available.
        Falls back to 0.5 (neutral) if no history exists.
        """
        if memory_retriever:
            current = memory_retriever.get_current()
            if current:
                # Use tool proficiency as track record proxy
                return current.tool_proficiency.get(specialist_id, 0.5)

        return 0.5

    # ── Group-Level Metrics (CONCEPT:ORCH-1.2) ─────────────────────────────────

    def compute_group_confidence(self, proposals: list[Proposal]) -> float:
        """Mean confidence across a group of proposals (CONCEPT:AHE-3.2).

        Implements the group-level confidence from Squeeze Evolve Eq. 4:
        ``GC(g) = (1/K) * Σ C(τ) for τ ∈ g``

        Args:
            proposals: A group of scored proposals.

        Returns:
            Mean confidence score ``[0.0, 1.0]``, or ``0.5`` if empty.
        """
        if not proposals:
            return 0.5
        return sum(p.confidence_score for p in proposals) / len(proposals)

    def compute_group_diversity(self, proposals: list[Proposal]) -> int:
        """Count unique final answer patterns (CONCEPT:AHE-3.2).

        Implements Eq. 5 from Squeeze Evolve:
        ``D(g) = |{answer(τ) : τ ∈ g}|``

        Uses first 200 characters of output as a diversity fingerprint.
        When embeddings are available in a future iteration, this can
        be upgraded to semantic clustering.

        Args:
            proposals: A group of scored proposals.

        Returns:
            Number of unique answer clusters (minimum 0).
        """
        if not proposals:
            return 0
        fingerprints = {p.output[:200].strip().lower() for p in proposals}
        return len(fingerprints)

    def deliberation_score(self, proposals: list[Proposal]) -> dict[str, float]:
        """Cross-trajectory critical analysis for heavy thinking integration.

        CONCEPT:AHE-3.4 — Analyzes a group of proposals (representing
        parallel reasoning trajectories) to determine whether they
        warrant sequential deliberation.

        Computes:
            - ``consensus``: Fraction of proposals that agree on the answer
            - ``diversity``: Normalized unique answer count
            - ``confidence``: Mean confidence across proposals
            - ``deliberation_needed``: Score indicating how much the
              trajectories would benefit from deliberation (higher = more needed)

        Deliberation is most beneficial when confidence is moderate but
        diversity is high (trajectories disagree with uncertain reasoning).

        Args:
            proposals: Scored proposals representing parallel trajectories.

        Returns:
            Dict with ``consensus``, ``diversity``, ``confidence``,
            and ``deliberation_needed`` scores.
        """
        if not proposals:
            return {
                "consensus": 0.0,
                "diversity": 0.0,
                "confidence": 0.0,
                "deliberation_needed": 0.0,
            }

        # Compute consensus (fraction agreeing on majority answer)
        fingerprints: dict[str, int] = {}
        for p in proposals:
            key = p.output[:200].strip().lower()
            fingerprints[key] = fingerprints.get(key, 0) + 1

        majority = max(fingerprints.values()) if fingerprints else 0
        consensus = majority / len(proposals) if proposals else 0.0

        # Compute diversity (normalized unique count)
        diversity = len(fingerprints) / max(len(proposals), 1)

        # Compute mean confidence
        confidence = self.compute_group_confidence(proposals)

        # Deliberation need: high when diversity is high + confidence is moderate
        # Formula: deliberation_needed = diversity * (1 - |confidence - 0.5| * 2)
        # Peaks when confidence ≈ 0.5 and diversity is high
        confidence_uncertainty = 1.0 - abs(confidence - 0.5) * 2.0
        deliberation_needed = min(1.0, diversity * max(0.0, confidence_uncertainty))

        return {
            "consensus": consensus,
            "diversity": diversity,
            "confidence": confidence,
            "deliberation_needed": deliberation_needed,
        }
