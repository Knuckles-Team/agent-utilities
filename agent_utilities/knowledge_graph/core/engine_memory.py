from __future__ import annotations

"""Memory management mixin for IntelligenceGraphEngine.

Extracted from engine.py. Contains CRUD operations for memory nodes.
"""
# CONCEPT:ORCH-1.2 — Memory Management


import typing

if typing.TYPE_CHECKING:
    from .._engine_protocol import _EngineProtocol

    _Base = _EngineProtocol
else:
    _Base = object


import logging
import time
import uuid
from datetime import UTC
from typing import Any

from ...models.knowledge_graph import MemoryNode

logger = logging.getLogger(__name__)


# ── Context Budget Optimizer (Research: 2604.20874v1) ──────────────


class ContextBudgetOptimizer:
    """Applies the Root Theorem of Context Engineering to memory recall.

    CONCEPT:KG-2.1 — Research: 2604.20874v1

    Core insight from the paper: context_quality = f(relevance_density × coverage).
    When relevance density drops below a threshold, the context should be
    *compacted* (summarized) rather than simply truncated. When coverage
    gaps exist, additional retrieval is warranted.

    This optimizer sits between memory retrieval and context assembly:
    1. Estimate available token budget from task complexity.
    2. After retrieval, check if results exceed budget.
    3. If so, compact by dropping lowest-relevance items first.
    4. Track quality metrics for AHE evaluation.

    Attributes:
        default_budget_tokens: Default context budget in tokens.
        min_density_threshold: Below this, trigger compaction.
        chars_per_token: Approximate characters per token.
    """

    def __init__(
        self,
        default_budget_tokens: int = 8_000,
        min_density_threshold: float = 0.3,
        chars_per_token: float = 3.5,
    ) -> None:
        self.default_budget_tokens = default_budget_tokens
        self.min_density_threshold = min_density_threshold
        self.chars_per_token = chars_per_token

    def allocate_budget(
        self,
        task_complexity: float = 0.5,
        available_tokens: int = 0,
    ) -> dict[str, Any]:
        """Allocate context budget based on task complexity.

        CONCEPT:KG-2.1 — Research: 2604.20874v1 §Root Theorem

        The Root Theorem states that optimal context size scales with
        task complexity: simple tasks need focused context, complex
        tasks need broader coverage.

        Args:
            task_complexity: Task complexity (0.0=trivial, 1.0=maximum).
            available_tokens: Total available context window. If 0, uses default.

        Returns:
            Dict with ``budget_tokens``, ``max_items``, and ``quality_target``.
        """
        base = available_tokens or self.default_budget_tokens
        # Scale: simple tasks get 40% of budget, complex tasks get 100%
        scale_factor = 0.4 + (task_complexity * 0.6)
        budget = int(base * scale_factor)

        return {
            "budget_tokens": budget,
            "budget_chars": int(budget * self.chars_per_token),
            "max_items": max(3, int(budget / 500)),  # ~500 tokens per memory item
            "quality_target": 0.5 + (task_complexity * 0.3),
            "task_complexity": task_complexity,
        }

    def compact_results(
        self,
        memories: list[dict[str, Any]],
        budget_tokens: int,
        score_key: str = "decay_adjusted_score",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Compact memory results to fit within token budget.

        CONCEPT:KG-2.1 — Research: 2604.20874v1

        Instead of simple truncation, this method:
        1. Estimates token count per memory item.
        2. Greedily selects highest-scoring items that fit.
        3. Computes relevance density of the final context.

        Args:
            memories: Retrieved memory items (already scored).
            budget_tokens: Maximum token budget.
            score_key: Key to use for relevance scoring.

        Returns:
            Tuple of (compacted_memories, quality_metrics).
        """
        budget_chars = int(budget_tokens * self.chars_per_token)
        selected: list[dict[str, Any]] = []
        total_chars = 0
        total_score = 0.0

        for mem in memories:
            content = str(mem.get("description", mem.get("content", "")))
            item_chars = len(content)
            if total_chars + item_chars > budget_chars and selected:
                break
            selected.append(mem)
            total_chars += item_chars
            total_score += float(mem.get(score_key, 0.0))

        # Compute quality metrics (Root Theorem)
        density = total_score / max(1, len(selected))
        coverage = len(selected) / max(1, len(memories))

        metrics = {
            "items_before": len(memories),
            "items_after": len(selected),
            "tokens_estimated": int(total_chars / self.chars_per_token),
            "budget_tokens": budget_tokens,
            "relevance_density": round(density, 4),
            "coverage": round(coverage, 4),
            "context_quality": round(density * coverage, 4),
            "compacted": len(selected) < len(memories),
        }

        if (
            metrics["relevance_density"] < self.min_density_threshold
            and len(selected) > 1
        ):
            # Density too low — further compact by keeping only top items
            half = max(1, len(selected) // 2)
            selected = selected[:half]
            metrics["items_after"] = len(selected)
            metrics["further_compacted"] = True

        return selected, metrics

    def should_expand(
        self,
        memories: list[dict[str, Any]],
        budget_tokens: int,
        score_key: str = "decay_adjusted_score",
    ) -> bool:
        """Check if context has coverage gaps warranting expansion.

        CONCEPT:KG-2.1 — Research: 2604.20874v1

        Returns True if we're under-utilizing the budget AND the items
        have high relevance (suggesting more relevant items may exist).

        Args:
            memories: Current memory results.
            budget_tokens: Available token budget.
            score_key: Scoring key.

        Returns:
            True if additional retrieval is recommended.
        """
        if not memories:
            return True
        total_chars = sum(
            len(str(m.get("description", m.get("content", "")))) for m in memories
        )
        usage = total_chars / max(1, budget_tokens * self.chars_per_token)
        avg_score = sum(float(m.get(score_key, 0)) for m in memories) / len(memories)
        # Expand if using <50% of budget AND average relevance is high
        return usage < 0.5 and avg_score > 0.5


# Default optimizer instance
_context_optimizer = ContextBudgetOptimizer()


class MemoryMixin(_Base):
    """Memory node CRUD capabilities for the KG engine."""

    def add_memory(
        self,
        content: str,
        name: str = "",
        category: str = "general",
        tags: list[str] | None = None,
    ) -> str:
        """Add a new memory to the unified graph."""
        memory_id = f"mem:{uuid.uuid4().hex[:8]}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = MemoryNode(
            id=memory_id,
            name=name or f"Memory {timestamp}",
            description=content,
            timestamp=timestamp,
            category=category,
            tags=tags or [],
        )

        # Generate embedding if model available
        if self.hybrid_retriever.embed_model:
            try:
                node.embedding = self.hybrid_retriever.embed_model.get_text_embedding(
                    node.description or node.name
                )
            except Exception as e:
                logger.warning(
                    f"Failed to generate embedding for memory {node.id}: {e}"
                )

        # Tiered write: backend is source of truth, NX is fallback
        if self.backend:
            data = self._serialize_node(node, label="Memory")
            self._upsert_node("Memory", node.id, data)

        # Update graph compute cache
        self.graph.add_node(node.id, **node.model_dump())

        return memory_id

    def delete_memory(self, memory_id: str):
        """Delete a memory from the graph."""
        if memory_id in self.graph:
            self.graph.remove_node(memory_id)
        if self.backend:
            self.backend.execute(
                "MATCH (n {id: $id}) SET n.status = 'ARCHIVED'", {"id": memory_id}
            )

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """Retrieve a specific memory by ID from the graph."""
        # Check graph compute first (in-memory)
        if memory_id in self.graph:
            return {"id": memory_id, **self.graph.nodes[memory_id]}
        # Fallback to persistent backend
        if self.backend:
            results = self.backend.execute(
                "MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id}
            )
            if results:
                return results[0].get("m", results[0])
        return None

    def update_memory(self, memory_id: str, **kwargs):
        """Update properties of an existing memory."""
        if memory_id in self.graph:
            self.graph.nodes[memory_id].update(kwargs)
        if self.backend:
            set_clause = self._get_set_clause(kwargs, "n", label="Memory")
            self.backend.execute(
                f"MATCH (n {{id: $id}}){set_clause}",
                {"id": memory_id, **kwargs},
            )

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ):
        """Create a relationship between two nodes in the graph."""
        if rel_type:
            rel_type = rel_type.upper()
        props = properties or {}
        if source_id in self.graph and target_id in self.graph:
            self.graph.add_edge(source_id, target_id, type=rel_type, **props)

        if self.backend:
            set_clause = self._get_set_clause(props, alias="r", label=rel_type)
            query = (
                f"MATCH (s {{id: $sid}}), (t {{id: $tid}}) "
                f"MERGE (s)-[r:{rel_type}]->(t){set_clause}"
            )
            params = {"sid": source_id, "tid": target_id}
            params.update(props)
            self.backend.execute(query, params)

    def add_memory_node(self, memory: MemoryNode):
        """Add a MemoryNode object to the graph."""
        if self.backend:
            data = self._serialize_node(memory, label="Memory")
            self._upsert_node("Memory", memory.id, data)
        else:
            self.graph.add_node(memory.id, **memory.model_dump())

    def get_memory_node(self, memory_id: str) -> MemoryNode | None:
        """Retrieve a MemoryNode object by ID."""
        data = self.get_memory(memory_id)
        if data:
            return MemoryNode(
                **{k: v for k, v in data.items() if not k.startswith("_")}
            )
        return None

    def update_memory_node(self, memory_id: str, memory: MemoryNode):
        """Update a memory using a MemoryNode object."""
        self.update_memory(memory_id, **memory.model_dump(exclude={"id"}))

    def delete_memory_node(self, memory_id: str):
        """Delete a memory node."""
        self.delete_memory(memory_id)

    # --- Enhanced Memory & Ingestion Tools ---
    # CONCEPT:KG-2.1 — Research: MEMO Survey (2504.01990v2), ParamMem (2604.27707v1)

    def store_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        name: str = "",
        tags: list[str] | None = None,
        trust_score: float = 0.8,
        agent_id: str = "",
    ) -> str:
        """Store a tiered memory with trust scoring and provenance.

        CONCEPT:KG-2.1 — Research: ParamMem (2604.27707v1) §6.2

        Args:
            content: Memory content text.
            memory_type: One of 'episodic', 'semantic', 'procedural', 'working'.
            name: Optional human-readable name.
            tags: Optional categorization tags.
            trust_score: Provenance trust (0.0–1.0). Memories below 0.3 are
                quarantined from default recall. Default 0.8.
            agent_id: Source agent for provenance tracking.

        Returns:
            Memory node ID.
        """
        memory_id = f"mem:{uuid.uuid4().hex[:8]}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = MemoryNode(
            id=memory_id,
            name=name or f"Memory {timestamp}",
            description=content,
            timestamp=timestamp,
            category=memory_type,
            tags=tags or [],
        )

        # Generate embedding
        if self.hybrid_retriever.embed_model:
            try:
                node.embedding = self.hybrid_retriever.embed_model.get_text_embedding(
                    node.description or node.name
                )
            except Exception as e:
                logger.warning(
                    f"Failed to generate embedding for memory {node.id}: {e}"
                )

        # Add trust and provenance metadata
        data = (
            self._serialize_node(node, label="Memory")
            if self.backend
            else node.model_dump()
        )
        data["memory_type"] = memory_type
        data["trust_score"] = trust_score
        data["access_count"] = 0
        data["last_accessed"] = timestamp
        if agent_id:
            data["agent_id"] = agent_id

        if self.backend:
            self._upsert_node("Memory", node.id, data)
        else:
            self.graph.add_node(node.id, **data)

        return memory_id

    def recall_memory(
        self,
        query: str,
        memory_type: str = "",
        top_k: int = 5,
        apply_decay: bool = True,
        include_untrusted: bool = False,
        task_context: str = "",
    ) -> list[dict[str, Any]]:
        """Recall memories with Ebbinghaus time-decay scoring.

        CONCEPT:KG-2.1 — Research: MEMO Survey (2504.01990v2) §3.2

        Performs hybrid search for memories, then applies time-decay
        scoring based on the Ebbinghaus forgetting curve. Memories
        with low trust scores are excluded by default.

        Args:
            query: Search query for memory retrieval.
            memory_type: Filter by memory tier ('episodic', 'semantic', etc.).
            top_k: Maximum results.
            apply_decay: Whether to apply Ebbinghaus time-decay. Default True.
            include_untrusted: Include memories with trust_score < 0.3.
            task_context: Optional task context for instruction-aware reranking.

        Returns:
            List of memory dicts with decay_adjusted_score.
        """
        from agent_utilities.knowledge_graph.memory.consolidation import (
            MEMORY_HALF_LIVES,
            ebbinghaus_decay,
        )

        # Search for memories
        results = self.search_hybrid(query, top_k=top_k * 3)

        # Filter to Memory nodes
        memories = []
        for r in results:
            r_type = str(r.get("type", "")).lower()
            r_category = str(r.get("category", "")).lower()
            if r_type != "memory" and r_category not in MEMORY_HALF_LIVES:
                continue
            if memory_type and r_category != memory_type.lower():
                continue
            # Trust filter (CONCEPT:KG-2.1 — ParamMem §6.2)
            trust = float(r.get("trust_score", 0.8))
            if not include_untrusted and trust < 0.3:
                continue
            memories.append(r)

        # Apply Ebbinghaus decay scoring
        now_ts = time.time()
        for mem in memories:
            base_score = float(mem.get("_score", 0.5))

            if apply_decay:
                # Parse timestamp
                ts_str = mem.get("last_accessed", mem.get("timestamp", ""))
                elapsed = 0.0
                if ts_str:
                    try:
                        from datetime import datetime

                        dt = datetime.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S")
                        dt = dt.replace(tzinfo=UTC)
                        elapsed = max(0.0, now_ts - dt.timestamp())
                    except (ValueError, TypeError):
                        elapsed = 0.0

                # Get half-life for this memory tier
                mem_type = str(
                    mem.get("memory_type", mem.get("category", "episodic"))
                ).lower()
                half_life = MEMORY_HALF_LIVES.get(mem_type, 14400)

                if half_life > 0:
                    mem["decay_adjusted_score"] = round(
                        ebbinghaus_decay(base_score, elapsed, half_life), 4
                    )
                else:
                    # Procedural — no decay
                    mem["decay_adjusted_score"] = base_score
            else:
                mem["decay_adjusted_score"] = base_score

        # Sort by decay-adjusted score
        memories.sort(key=lambda x: x.get("decay_adjusted_score", 0), reverse=True)

        # Instruction-aware reranking (CONCEPT:KG-2.1 — MemReranker)
        if task_context and self.hybrid_retriever.embed_model:
            try:
                task_emb = self.hybrid_retriever.embed_model.get_text_embedding(
                    task_context
                )
                for mem in memories:
                    mem_emb = mem.get("embedding")
                    if mem_emb and task_emb:
                        # Dot-product reranking — no LLM needed
                        dot = sum(
                            a * b
                            for a, b in zip(
                                task_emb, mem_emb[: len(task_emb)], strict=False
                            )
                        )
                        mem["task_relevance"] = round(dot, 4)
                memories.sort(
                    key=lambda x: x.get(
                        "task_relevance", x.get("decay_adjusted_score", 0)
                    ),
                    reverse=True,
                )
            except Exception as e:
                logger.warning(f"Task-context reranking failed: {e}")

        # Context budget compaction (CONCEPT:KG-2.1 — Research: 2604.20874v1)
        # Apply Root Theorem: compact results if they exceed budget
        budget = _context_optimizer.allocate_budget(
            task_complexity=0.5 if not task_context else 0.7,
        )
        if len(memories) > budget["max_items"]:
            memories, quality_metrics = _context_optimizer.compact_results(
                memories,
                budget_tokens=budget["budget_tokens"],
            )
            logger.debug(
                "ContextBudgetOptimizer: compacted %d→%d items (quality=%.3f)",
                quality_metrics["items_before"],
                quality_metrics["items_after"],
                quality_metrics["context_quality"],
            )

        # Update access counts
        for mem in memories[:top_k]:
            mem_id = mem.get("id", "")
            if mem_id and self.backend:
                try:
                    self.backend.execute(
                        "MATCH (m:Memory {id: $id}) SET m.access_count = m.access_count + 1, "
                        "m.last_accessed = $now",
                        {
                            "id": mem_id,
                            "now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        },
                    )
                except Exception as e:
                    logger.debug("Failed to update memory access count: %s", e)

        return memories[:top_k]

    def consolidate_memories(self, dry_run: bool = False) -> dict[str, Any]:
        """Trigger memory consolidation — runs all registered rules.

        CONCEPT:KG-2.1 — Research: ParamMem (2604.27707v1)

        Runs the ConsolidationEngine with all three rules:
        1. EpisodeToPreferenceRule
        2. DecisionToPrincipleRule
        3. TraceToSkillRule (NEW — from ParamMem research)

        Args:
            dry_run: If True, return proposals without persisting.

        Returns:
            Dict with proposal counts and details.
        """
        from agent_utilities.knowledge_graph.memory.consolidation import (
            ConsolidationEngine,
            DecisionToPrincipleRule,
            EpisodeToPreferenceRule,
            TraceToSkillRule,
        )

        consolidation = ConsolidationEngine(engine=self)  # type: ignore[arg-type]
        consolidation.register(EpisodeToPreferenceRule())
        consolidation.register(DecisionToPrincipleRule())
        consolidation.register(TraceToSkillRule())

        proposals = consolidation.run(dry_run=dry_run)
        deduped = consolidation.dedup_by_signature(proposals)

        return {
            "total_proposals": len(deduped),
            "by_rule": {
                rule.name: len([p for p in deduped if p.rule_name == rule.name])
                for rule in consolidation.rules
            },
            "proposals": [p.model_dump() for p in deduped[:20]],
            "dry_run": dry_run,
        }
