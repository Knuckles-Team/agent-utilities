"""CONCEPT:KG-2.1 — Unified Memory Manager.

Single entry point for the full memory lifecycle:
  startup → active context → compaction → consolidation → retrieval

Coordinates all memory subsystems without replacing them.  Each
subsystem remains the implementation; this manager provides the
lifecycle orchestration API on top.

Subsystems coordinated:
    - ``StartupContextBuilder``: Initial context assembly
    - ``ContextCompactor``: Active window management (3 strategies)
    - ``MementoCompressor``: LLM-based state compression
    - ``ConsolidationEngine``: Episode → Preference/Principle promotion
    - ``SemanticCompactor``: Trace compaction (prevents graph explosion)
    - ``MemoryRetriever``: KG-based semantic recall

Usage::

    mgr = MemoryLifecycleManager(engine=kg_engine)

    # Startup: assemble initial context within budget
    ctx = mgr.build_startup_context(query="debug auth flow", budget=8000)

    # Active session: compact when context grows too large
    result = mgr.compact_if_needed(messages, strategy="summarize_tools")

    # Long session: compress state to dense memento
    memento = mgr.compress_to_memento(messages, source="agent_runner")

    # Post-session: promote episodes to long-term knowledge
    await mgr.consolidate()

    # Recall: retrieve relevant memories for a new query
    memories = mgr.retrieve(query="auth flow", k=5)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    from .elastic_context_manager import CompactedResult, CompactionStrategy
    from .startup_context import StartupPayload

logger = logging.getLogger(__name__)


class MemoryLifecycleManager:
    """Coordinates memory subsystems through startup → compaction → consolidation → retrieval.

    CONCEPT:KG-2.1 — Memory Lifecycle Manager

    Provides a facade over the independent memory implementations,
    ensuring they are used in the correct lifecycle order and with
    consistent configuration.

    Args:
        engine: Optional ``IntelligenceGraphEngine`` for KG-backed
            persistence and retrieval.
        max_tokens: Default token budget for context compaction.
        auto_compaction_ratio: Fraction of max_tokens that triggers
            automatic compaction (default 0.8).
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        *,
        max_tokens: int = 8000,
        auto_compaction_ratio: float = 0.8,
    ) -> None:
        self.engine = engine
        self.max_tokens = max_tokens
        self.auto_compaction_ratio = auto_compaction_ratio

        # Lazy-initialized subsystem instances
        self._compactor: Any = None
        self._startup_builder: Any = None

    # ── Subsystem Access ─────────────────────────────────────────────

    @property
    def compactor(self) -> Any:
        """Lazy-initialized ContextCompactor."""
        if self._compactor is None:
            from .elastic_context_manager import ContextCompactor

            self._compactor = ContextCompactor(
                max_tokens=self.max_tokens,
                auto_compaction_ratio=self.auto_compaction_ratio,
            )
        return self._compactor

    @property
    def startup_builder(self) -> Any:
        """Lazy-initialized StartupContextBuilder."""
        if self._startup_builder is None and self.engine is not None:
            from .startup_context import StartupContextBuilder

            self._startup_builder = StartupContextBuilder(engine=self.engine)
        return self._startup_builder

    # ── Phase 1: Startup Context ─────────────────────────────────────

    def build_startup_context(
        self,
        query: str,
        budget: int = 8000,
    ) -> StartupPayload:
        """Assemble initial context within a token budget.

        Uses ``StartupContextBuilder`` to gather KG memories, recent
        mementos, and project context into a budgeted payload for
        the agent's first turn.

        Args:
            query: The user query to tailor context retrieval.
            budget: Maximum token budget for the startup payload.

        Returns:
            A ``StartupPayload`` with assembled context and metadata.
        """
        if self.engine is None:
            from .startup_context import StartupPayload

            return StartupPayload(
                text="",
                budget_chars=budget,
                included_handles=[],
                overflow=[],
            )

        from .startup_context import build_startup_payload

        return build_startup_payload(
            engine=self.engine,
            query=query,
            budget=budget,
        )

    # ── Phase 2: Active Context Management ───────────────────────────

    def compact_if_needed(
        self,
        messages: list[dict[str, Any]],
        strategy: CompactionStrategy | str = "summarize_tools",
    ) -> CompactedResult | None:
        """Compact active context if the auto-compaction threshold is exceeded.

        Args:
            messages: Current conversation messages.
            strategy: Compaction strategy to use (``summarize_tools``,
                ``drop_middle``, or ``progressive``).

        Returns:
            A ``CompactedResult`` if compaction was performed, else ``None``.
        """
        if self.compactor.should_compact(messages):
            result = self.compactor.compact(messages, strategy=strategy)
            logger.info(
                "[UnifiedMemory] Compacted context: %d → %d tokens (%s)",
                result.tokens_before,
                result.tokens_after,
                result.strategy_used,
            )
            return result
        return None

    def force_compact(
        self,
        messages: list[dict[str, Any]],
        strategy: CompactionStrategy | str = "summarize_tools",
    ) -> CompactedResult:
        """Force compaction regardless of threshold.

        Args:
            messages: Current conversation messages.
            strategy: Compaction strategy to use.

        Returns:
            A ``CompactedResult`` with compaction details.
        """
        return self.compactor.compact(messages, strategy=strategy)

    # ── Phase 3: State Compression ───────────────────────────────────

    def compress_to_memento(
        self,
        messages: list[dict[str, str]],
        *,
        source: str = "agent_runner",
        dry_run: bool = False,
    ) -> str | None:
        """Compress a block of messages into a dense memento.

        Uses LLM-based ``MementoCompressor`` to generate a state
        snapshot that can replace the raw message block for
        long-running sessions.

        Args:
            messages: The block of raw messages to compress.
            source: The source agent or component name.
            dry_run: If True, generate memento but don't persist to KG.

        Returns:
            The generated memento string, or None if compression failed.
        """
        if self.engine is None:
            return None

        from .memento_compressor import compress_to_memento

        return compress_to_memento(
            self.engine,
            messages,
            source=source,
            dry_run=dry_run,
        )

    def get_recent_mementos(
        self,
        source: str,
        limit: int = 5,
    ) -> list[str]:
        """Retrieve the most recent mementos for a given source.

        Args:
            source: The source agent name.
            limit: Maximum number of mementos to retrieve.

        Returns:
            List of memento content strings.
        """
        if self.engine is None:
            return []

        from .memento_compressor import get_recent_mementos

        return get_recent_mementos(self.engine, source=source, limit=limit)

    # ── Phase 4: Consolidation ───────────────────────────────────────

    def consolidate(
        self,
        *,
        dry_run: bool = True,
    ) -> list[Any]:
        """Promote episodes to preferences and principles.

        Runs the ``ConsolidationEngine`` to identify episode clusters
        that should be promoted to higher-tier knowledge structures
        in the Knowledge Graph.

        Args:
            dry_run: If True, returns proposals without persisting.

        Returns:
            List of ``ConsolidationProposal`` objects describing promotions.
        """
        if self.engine is None:
            return []

        from .consolidation import ConsolidationEngine

        ce = ConsolidationEngine(engine=self.engine)
        return ce.run(dry_run=dry_run)

    # ── Phase 5: Trace Compaction ────────────────────────────────────

    def compact_traces(
        self,
        agent_id: str,
        threshold: int = 10,
    ) -> int:
        """Compact semantic traces to prevent graph explosion.

        Uses ``SemanticCompactor`` to merge or prune trace nodes
        that exceed the threshold count for a given agent.

        Args:
            agent_id: The agent whose traces to compact.
            threshold: Maximum number of trace nodes before compaction.

        Returns:
            Number of compacted/removed nodes.
        """
        from .memory_compaction import SemanticCompactor

        sc = SemanticCompactor(engine=self.engine)
        return sc.compact_traces(agent_id=agent_id, threshold=threshold)

    # ── Phase 6: Retrieval ───────────────────────────────────────────

    def retrieve_self_model(self) -> dict[str, Any] | None:
        """Retrieve the current agent self-model from the Knowledge Graph.

        Uses ``MemoryRetriever.get_current()`` to load the latest
        versioned self-model snapshot.

        Returns:
            A dict with self-model data, or None if unavailable.
        """
        if not self.engine:
            return None

        try:
            from ..retrieval.memory_retriever import MemoryRetriever

            mr = MemoryRetriever(engine=self.engine)
            current = mr.get_current()
            if current:
                return current.model_dump()
            return None
        except ImportError:
            logger.debug("MemoryRetriever not available")
            return None
        except Exception as e:
            logger.debug("Self-model retrieval failed: %s", e)
            return None

    def query_capabilities(self, domain: str) -> dict[str, float]:
        """Query the self-model for capability scores in a domain.

        Args:
            domain: The domain to query (e.g., "gitlab", "servicenow").

        Returns:
            Dict with ``success_rate``, ``confidence``, ``proficiency``.
        """
        if not self.engine:
            return {"success_rate": 0.0, "confidence": 0.0, "proficiency": 0.0}

        try:
            from ..retrieval.memory_retriever import MemoryRetriever

            mr = MemoryRetriever(engine=self.engine)
            return mr.query_capabilities(domain)
        except Exception as e:
            logger.debug("Capability query failed: %s", e)
            return {"success_rate": 0.0, "confidence": 0.0, "proficiency": 0.0}
