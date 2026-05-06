#!/usr/bin/python
"""CONCEPT:ORCH-1.2 — Global Workspace Attention.

Always-on attention mechanism for specialist output quality filtering.
Inspired by Global Workspace Theory (GWT), specialists submit proposals
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

See docs/emergent-architecture.md §CONCEPT:ORCH-1.2.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ..knowledge_graph.engine import cosine_similarity
from ..knowledge_graph.ogm import KGMapper
from ..models.knowledge_graph import ProposalNode, RegistryEdgeType

if TYPE_CHECKING:
    from ..knowledge_graph.engine import IntelligenceGraphEngine
    from ..knowledge_graph.memory_retriever import MemoryRetriever

logger = logging.getLogger(__name__)


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
        max_broadcast_slots: int = 5,
        relevance_weight: float = 0.5,
        track_record_weight: float = 0.3,
        confidence_weight: float = 0.2,
    ) -> None:
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

        return winners

    # ── KG Broadcast ──────────────────────────────────────────────────

    def broadcast_to_kg(
        self,
        winners: list[Proposal],
        engine: IntelligenceGraphEngine,
        task_id: str = "",
    ) -> list[str]:
        """Persist winning proposals to the KG for global visibility.

        Creates ``ProposalNode`` entries and links them to their specialists.
        These serve as training signal for the self-model (CONCEPT:KG-2.1).

        Args:
            winners: The selected winning proposals.
            engine: The engine for KG persistence.
            task_id: Optional task ID to link proposals to.

        Returns:
            List of persisted proposal node IDs.
        """
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

        logger.debug("GWT broadcast %d proposals to KG", len(node_ids))
        return node_ids

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

        CONCEPT:AHE-3.7 — Analyzes a group of proposals (representing
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
