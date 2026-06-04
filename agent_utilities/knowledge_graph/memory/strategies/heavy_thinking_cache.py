#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AHE-3.4 — Serialized Memory Cache for Heavy Thinking.

Implements the structured cache that bridges parallel reasoning trajectories
into the sequential deliberation phase.  The cache provides:

- **Trajectory storage**: Accumulates outputs from K parallel thinkers
- **Thinking token pruning**: Strips verbose chain-of-thought markers to
  prevent context overflow in the deliberation prompt
- **Shuffling**: Randomizes trajectory order to prevent position bias in
  the deliberation model (first-trajectory-wins effect)
- **Iterative augmentation**: Feeds deliberation results back as additional
  trajectories for convergence refinement
- **KG persistence**: Serializes all trajectories as ``TrajectoryNode``
  instances for cross-session reuse (free value-add from CONCEPT:KG-2.0)

Integrates with:
    - CONCEPT:AHE-3.4 (Heavy Thinking): Core data structure for the pipeline
    - CONCEPT:KG-2.0 (OGM): KG persistence via ``KGMapper``
    - CONCEPT:KG-2.4 (Hypergraphs): EncPI metadata on trajectory interactions

See docs/overview.md §CONCEPT:AHE-3.4
"""


import hashlib
import logging
import random
import re
import time
import uuid
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ...core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class TrajectoryEntry(BaseModel):
    """A single trajectory from a parallel thinker.

    Attributes:
        thinker_id: Unique identifier for this thinker instance.
        raw_output: The complete output including thinking tokens.
        answer: The extracted final answer (post-pruning).
        reasoning_summary: Pruned reasoning summary (thinking tokens removed).
        model_id: The LLM model used by this thinker.
        score: Evaluation score (set during deliberation).
        success: Whether this trajectory completed without error.
    """

    thinker_id: str
    raw_output: str = ""
    answer: str = ""
    reasoning_summary: str = ""
    model_id: str = ""
    score: float = 0.0
    success: bool = True


class MemoryCache(BaseModel):
    """Serialized Memory Cache for bridging parallel → deliberation phases.

    CONCEPT:AHE-3.4 — Heavy Thinking Orchestration

    Accumulates parallel thinker outputs, provides pruning/shuffling
    for the deliberation prompt, and supports iterative augmentation
    for refinement loops.

    Attributes:
        query: The original query being reasoned about.
        query_hash: SHA-256 hash for deduplication.
        trajectories: Accumulated trajectory entries.
        deliberation_results: Deliberation outputs from iterative passes.
    """

    query: str = ""
    query_hash: str = ""
    trajectories: list[TrajectoryEntry] = Field(default_factory=list)
    deliberation_results: list[str] = Field(default_factory=list)

    def add_trajectory(
        self,
        thinker_id: str,
        output: str,
        thinking_content: str | None = None,
        model_id: str = "",
        success: bool = True,
    ) -> None:
        """Add a trajectory from a parallel thinker.

        Args:
            thinker_id: Unique identifier for this thinker.
            output: The raw output from the thinker.
            thinking_content: Optional extracted thinking/reasoning content.
            model_id: The LLM model ID used.
            success: Whether the trajectory completed successfully.
        """
        if thinking_content:
            logger.debug(
                "MemoryCache: Captured thinking content for %s (%d chars)",
                thinker_id,
                len(thinking_content),
            )

        pruned = TrajectoryPruner.prune(output)
        answer = TrajectoryPruner.extract_answer(pruned)

        entry = TrajectoryEntry(
            thinker_id=thinker_id,
            raw_output=output,
            answer=answer,
            reasoning_summary=pruned,
            model_id=model_id,
            success=success,
        )
        self.trajectories.append(entry)
        logger.debug(
            "MemoryCache: Added trajectory from %s (answer=%s, success=%s)",
            thinker_id,
            answer[:50] if answer else "N/A",
            success,
        )

    def serialize(self, prune: bool = True, shuffle: bool = True) -> str:
        """Produce the serialized prompt for the deliberation model.

        Formats all trajectories into a structured prompt that the
        deliberation agent can consume for critical analysis.

        Args:
            prune: Whether to strip thinking tokens (use summaries vs raw).
            shuffle: Whether to randomize trajectory order.

        Returns:
            Formatted string for the deliberation prompt.
        """
        entries = list(self.trajectories)

        if shuffle:
            entries = TrajectoryShuffler.shuffle(entries)

        blocks: list[str] = []
        for i, entry in enumerate(entries, 1):
            content = entry.reasoning_summary if prune else entry.raw_output
            block = (
                f"### Thinker {i} ({entry.thinker_id})\n"
                f"**Model**: {entry.model_id or 'default'}\n"
                f"**Status**: {'SUCCESS' if entry.success else 'FAILURE'}\n"
                f"**Answer**: {entry.answer or '(no answer extracted)'}\n\n"
                f"**Reasoning**:\n{content}\n"
            )
            blocks.append(block)

        # Include prior deliberation results for iterative refinement
        delib_section = ""
        if self.deliberation_results:
            delib_section = (
                "\n---\n## Prior Deliberation Results\n"
                + "\n\n".join(
                    f"### Iteration {i + 1}\n{r}"
                    for i, r in enumerate(self.deliberation_results)
                )
                + "\n---\n"
            )

        header = (
            f"## Serialized Memory Cache\n"
            f"**Query**: {self.query}\n"
            f"**Trajectories**: {len(entries)}\n"
            f"**Iteration**: {len(self.deliberation_results) + 1}\n\n"
        )

        return header + "\n".join(blocks) + delib_section

    def augment(self, deliberation_result: str) -> None:
        """Add a deliberation result for iterative refinement.

        Feeds the deliberation output back as context for the next
        deliberation pass, enabling convergence without losing prior
        reasoning.

        Args:
            deliberation_result: The output from the deliberation agent.
        """
        self.deliberation_results.append(deliberation_result)
        logger.debug(
            "MemoryCache: Augmented with deliberation result (iteration %d)",
            len(self.deliberation_results),
        )

    def to_kg_nodes(
        self,
        engine: IntelligenceGraphEngine,
    ) -> list[str]:
        """Persist all trajectories as KG nodes for cross-session reuse.

        Creates ``TrajectoryNode`` instances linked via ``TRAJECTORY_OF``
        edges and computes EncPI metadata for hypergraph generalization.

        Args:
            engine: The Intelligence Graph Engine for persistence.

        Returns:
            List of persisted node IDs.
        """
        from ....models.knowledge_graph import (
            RegistryEdgeType,
            TrajectoryNode,
        )
        from ...core.hypergraph import PositionalInteractionEncoder
        from ...core.ogm import KGMapper

        ogm = KGMapper(engine)
        encoder = PositionalInteractionEncoder()
        node_ids: list[str] = []
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Create a query anchor node for trajectory linking
        query_anchor_id = f"query:{self.query_hash[:12]}"
        engine.graph.add_node(
            query_anchor_id,
            type="memory",
            name=f"Query: {self.query[:60]}",
            description=self.query[:300],
            timestamp=ts,
        )

        for i, entry in enumerate(self.trajectories):
            node_id = f"traj:{uuid.uuid4().hex[:8]}"

            # Compute EncPI for trajectory position interactions
            enc_pi = encoder.encode_interaction(i, len(self.trajectories))

            node = TrajectoryNode(
                id=node_id,
                name=f"Trajectory: {entry.thinker_id}",
                description=f"Parallel reasoning trajectory for: {self.query[:100]}",
                thinker_id=entry.thinker_id,
                query_hash=self.query_hash,
                answer=entry.answer,
                reasoning_summary=entry.reasoning_summary[:2000],
                score=entry.score,
                is_correct=None,  # Set during deliberation
                model_id=entry.model_id,
                timestamp=ts,
                metadata={
                    "enc_pi": enc_pi,
                    "source": "heavy_thinking",
                    "iteration": len(self.deliberation_results),
                },
            )

            ogm.upsert(node)
            ogm.upsert_edge(
                node_id,
                query_anchor_id,
                RegistryEdgeType.TRAJECTORY_OF,
                {"position": i},
            )
            node_ids.append(node_id)

        logger.info("MemoryCache: Persisted %d trajectory nodes to KG", len(node_ids))
        return node_ids

    @staticmethod
    def from_query(query: str) -> MemoryCache:
        """Create a new MemoryCache for a query.

        Args:
            query: The query to reason about.

        Returns:
            A new ``MemoryCache`` instance.
        """
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        return MemoryCache(query=query, query_hash=query_hash)


class TrajectoryPruner:
    """Strips thinking tokens while preserving answer content.

    CONCEPT:AHE-3.4 — Prevents context overflow in the deliberation
    prompt by removing verbose chain-of-thought markers, internal
    planning tokens, and self-talk patterns.
    """

    # Patterns matching common thinking token wrappers
    THINKING_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"<think>.*?</think>", re.DOTALL),
        re.compile(r"<thinking>.*?</thinking>", re.DOTALL),
        re.compile(r"<internal_thought>.*?</internal_thought>", re.DOTALL),
        re.compile(r"\[THINKING\].*?\[/THINKING\]", re.DOTALL),
        re.compile(r"```thinking\n.*?```", re.DOTALL),
    ]

    # Patterns for answer extraction
    ANSWER_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\\boxed\{(.*?)\}", re.DOTALL),
        re.compile(r"\*\*(?:Final )?Answer\*\*[:\s]*(.+?)(?:\n|$)", re.DOTALL),
        re.compile(r"(?:The answer is|ANSWER)[:\s]*(.+?)(?:\n|$)", re.IGNORECASE),
    ]

    @classmethod
    def prune(cls, text: str) -> str:
        """Remove thinking tokens from raw output.

        Args:
            text: The raw thinker output.

        Returns:
            The output with thinking sections removed.
        """
        result = text
        for pattern in cls.THINKING_PATTERNS:
            result = pattern.sub("", result)

        # Clean up excessive whitespace from removal
        result = re.sub(r"\n{3,}", "\n\n", result).strip()
        return result

    @classmethod
    def extract_answer(cls, text: str) -> str:
        """Extract the final answer from a trajectory output.

        Checks for boxed answers, markdown bold answers, and plain-text
        answer markers.

        Args:
            text: The raw or pruned thinker output.

        Returns:
            The extracted answer string, or empty string if none found.
        """
        for pattern in cls.ANSWER_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()

        # Fallback: last non-empty line
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        return lines[-1] if lines else ""


class TrajectoryShuffler:
    """Randomizes trajectory order to prevent position bias.

    CONCEPT:AHE-3.4 — The deliberation model may exhibit position bias
    (favoring the first or last trajectory).  Shuffling mitigates this
    while preserving all trajectory content.
    """

    @staticmethod
    def shuffle(entries: list[TrajectoryEntry]) -> list[TrajectoryEntry]:
        """Return a shuffled copy of the trajectory entries.

        Args:
            entries: The trajectory entries to shuffle.

        Returns:
            A new list with randomized order.
        """
        shuffled = list(entries)
        random.shuffle(shuffled)  # nosec B311
        return shuffled
