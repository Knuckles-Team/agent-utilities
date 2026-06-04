#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:KG-2.1 — Persistent Self-Model with OWL Integration.

Provides a versioned metacognitive self-model that aggregates session
outcomes into a persistent KG representation of the agent's own
capabilities, strengths, and known failure modes.

Architecture:
    - **Versioned chain**: Each session creates a new ``MemoryRetrieverNode``
      linked to the previous via ``SUPERSEDES``.
    - **CURRENT pointer**: A ``CURRENT_SELF_MODEL`` edge enables O(1)
      lookup of the latest version.
    - **OWL integration**: Self-model triples are promoted into the
      OWL ontology for reasoner-driven metacognition (e.g., "What
      domains am I improving in?").

Integrates with:
    - CONCEPT:KG-2.0 (OGM): Declarative KG persistence
    - Existing OWL bridge: ``promote_to_owl()`` / ``reason_about_self()``
    - ``GraphState``: Session outcome aggregation

See docs/pillars/architecture_c4.md §CONCEPT:KG-2.1
"""


import logging
import time
import uuid
from typing import TYPE_CHECKING

from ...models.knowledge_graph import (
    MemoryRetrieverNode,
    RegistryEdgeType,
)
from ..core.ogm import KGMapper

if TYPE_CHECKING:
    from ...graph.state import GraphState
    from ..core.engine import IntelligenceGraphEngine
    from ..knowledge_graph.owl_bridge import OWLBridge

logger = logging.getLogger(__name__)

# Singleton anchor node ID for the self-model chain
SELF_MODEL_ANCHOR = "self:agent-model"


class MemoryRetriever:
    """Versioned metacognitive self-model with OWL reasoning.

    CONCEPT:KG-2.1 — Persistent Self-Model

    Maintains a linked chain of ``MemoryRetrieverNode`` snapshots, each
    representing the agent's self-assessed capabilities at a point in
    time. After each session, ``update_after_session()`` aggregates
    verification scores, tool usage, and failure patterns into the
    current version.

    The ``CURRENT_SELF_MODEL`` pointer always points to the latest
    version for fast retrieval. Historical versions are traversable
    via ``SUPERSEDES`` edges for trend analysis.

    Args:
        engine: The ``IntelligenceGraphEngine`` to operate on.
    """

    def __init__(self, engine: IntelligenceGraphEngine) -> None:
        self.engine = engine
        self.ogm = KGMapper(engine)

    # ── Core Lifecycle ────────────────────────────────────────────────

    def get_current(self) -> MemoryRetrieverNode | None:
        """Load the latest self-model version via the CURRENT pointer.

        Returns:
            The current ``MemoryRetrieverNode``, or ``None`` if no self-model exists.
        """
        if self.engine.backend:
            results = self.engine.backend.execute(
                "MATCH (anchor {id: $aid})-[:CURRENT_SELF_MODEL]->(sm:MemoryRetriever) "
                "RETURN sm",
                {"aid": SELF_MODEL_ANCHOR},
            )
            if results:
                data = results[0].get("sm", results[0])
                return self.ogm._deserialize(data, MemoryRetrieverNode)

        # graph compute fallback
        if SELF_MODEL_ANCHOR in self.engine.graph:
            for succ in self.engine.graph.successors(SELF_MODEL_ANCHOR):
                edge_data = self.engine.graph.get_edge_data(SELF_MODEL_ANCHOR, succ)
                if edge_data:
                    for _, edata in edge_data.items():
                        if edata.get("type") == RegistryEdgeType.CURRENT_SELF_MODEL:
                            ndata = dict(self.engine.graph.nodes[succ])
                            return self.ogm._deserialize(ndata, MemoryRetrieverNode)

        return None

    def get_or_create(self) -> MemoryRetrieverNode:
        """Load the current self-model, creating one if none exists.

        Returns:
            The current ``MemoryRetrieverNode``.
        """
        current = self.get_current()
        if current:
            return current

        # Create the initial self-model
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node = MemoryRetrieverNode(
            id=f"sm:{uuid.uuid4().hex[:8]}",
            name="Agent Self-Model v1",
            version=1,
            timestamp=ts,
            importance_score=1.0,
            is_permanent=True,
        )

        # Ensure anchor exists
        self.engine.graph.add_node(
            SELF_MODEL_ANCHOR,
            type="memory_retriever_anchor",
            name="Self-Model Anchor",
        )

        # Persist node + CURRENT pointer
        self.ogm.upsert(node)
        self.ogm.upsert_edge(
            SELF_MODEL_ANCHOR,
            node.id,
            RegistryEdgeType.CURRENT_SELF_MODEL,
        )

        logger.info("Created initial self-model: %s", node.id)
        return node

    def create_snapshot(self, session_id: str = "") -> MemoryRetrieverNode:
        """Create a new version of the self-model, linking to previous.

        Args:
            session_id: Optional session ID that triggered this snapshot.

        Returns:
            The newly created ``MemoryRetrieverNode``.
        """
        current = self.get_or_create()
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        new_node = MemoryRetrieverNode(
            id=f"sm:{uuid.uuid4().hex[:8]}",
            name=f"Agent Self-Model v{current.version + 1}",
            version=current.version + 1,
            timestamp=ts,
            importance_score=1.0,
            is_permanent=True,
            session_id=session_id,
            # Carry forward accumulated data
            domain_success_rates=dict(current.domain_success_rates),
            capability_confidence=dict(current.capability_confidence),
            tool_proficiency=dict(current.tool_proficiency),
            total_sessions=current.total_sessions,
            total_tasks_completed=current.total_tasks_completed,
            known_failure_patterns=list(current.known_failure_patterns),
            pheromone_trails={k: dict(v) for k, v in current.pheromone_trails.items()},
            model_synergies=dict(current.model_synergies),
        )

        # Persist
        self.ogm.upsert(new_node)

        # Link: new → old via SUPERSEDES
        self.ogm.upsert_edge(
            new_node.id,
            current.id,
            RegistryEdgeType.SUPERSEDES,
        )

        # Move CURRENT pointer
        # Remove old CURRENT edge
        if self.engine.backend:
            self.engine.backend.execute(
                "MATCH (a {id: $aid})-[r:CURRENT_SELF_MODEL]->() DELETE r",
                {"aid": SELF_MODEL_ANCHOR},
            )
        # Remove from graph compute
        edges_to_remove = []
        if SELF_MODEL_ANCHOR in self.engine.graph:
            for succ in self.engine.graph.successors(SELF_MODEL_ANCHOR):
                edge_data = self.engine.graph.get_edge_data(SELF_MODEL_ANCHOR, succ)
                if edge_data:
                    for key, edata in edge_data.items():
                        if edata.get("type") == RegistryEdgeType.CURRENT_SELF_MODEL:
                            edges_to_remove.append((SELF_MODEL_ANCHOR, succ, key))
        for src, tgt, key in edges_to_remove:
            self.engine.graph.remove_edge(src, tgt, key)

        # Add new CURRENT pointer
        self.ogm.upsert_edge(
            SELF_MODEL_ANCHOR,
            new_node.id,
            RegistryEdgeType.CURRENT_SELF_MODEL,
        )

        logger.info(
            "Self-model snapshot: v%d → v%d (%s)",
            current.version,
            new_node.version,
            new_node.id,
        )
        return new_node

    # ── Session Aggregation ───────────────────────────────────────────

    def update_after_session(self, session: GraphState) -> None:
        """Aggregate session outcomes into the self-model.

        Creates a new snapshot and updates:
            - ``domain_success_rates``: Running average from verification scores
            - ``capability_confidence``: From self-evaluation calibration
            - ``tool_proficiency``: Frequency × success from node history
            - ``known_failure_patterns``: From error messages
            - Session counters

        Args:
            session: The completed ``GraphState`` to aggregate from.
        """
        new_model = self.create_snapshot(session_id=session.session_id)

        # Update session counters
        new_model.total_sessions += 1
        tasks_completed = (
            sum(1 for t in session.task_list.tasks if t.status == "completed")
            if hasattr(session.task_list, "tasks")
            else 0
        )
        new_model.total_tasks_completed += tasks_completed

        # Update domain success rate (exponential moving average)
        if session.routed_domain:
            domain = session.routed_domain
            old_rate = new_model.domain_success_rates.get(domain, 0.5)
            # Simple: 1.0 if no error, 0.0 if error
            session_success = 0.0 if session.error else 1.0
            alpha = 0.3  # EMA smoothing factor
            new_model.domain_success_rates[domain] = (
                alpha * session_success + (1 - alpha) * old_rate
            )

        # Track tool usage from node history
        for node_name in session.node_history:
            old_prof = new_model.tool_proficiency.get(node_name, 0.0)
            # Simple increment — more sophisticated tracking would use
            # actual success/failure per node
            new_model.tool_proficiency[node_name] = min(1.0, old_prof + 0.05)

        # Track failure patterns
        if session.error and len(new_model.known_failure_patterns) < 50:
            pattern = session.error[:200]  # Truncate
            if pattern not in new_model.known_failure_patterns:
                new_model.known_failure_patterns.append(pattern)

        # Persist updated model
        self.ogm.upsert(new_model)

        # --- ACO: Pheromone trail decay & strengthening ---
        # Evaporate all trails by 10% (ant colony optimization)
        evaporation_rate = 0.10
        for specialist_id in list(new_model.pheromone_trails.keys()):
            trails = new_model.pheromone_trails[specialist_id]
            for pattern in list(trails.keys()):
                trails[pattern] *= 1.0 - evaporation_rate
                # Remove trails that have decayed below threshold
                if trails[pattern] < 0.01:
                    del trails[pattern]
            if not trails:
                del new_model.pheromone_trails[specialist_id]

        # Strengthen trails for successful specialist→domain combinations
        if session.routed_domain and not session.error:
            for node_name in session.node_history:
                if node_name not in new_model.pheromone_trails:
                    new_model.pheromone_trails[node_name] = {}
                domain = session.routed_domain
                old_strength = new_model.pheromone_trails[node_name].get(domain, 0.0)
                # Strengthen by 0.15, capped at 1.0
                new_model.pheromone_trails[node_name][domain] = min(
                    1.0, old_strength + 0.15
                )

        # Persist with pheromone updates
        self.ogm.upsert(new_model)

        # CONCEPT:AHE-3.3 — Model Synergy Tracker
        # Record the combination of models used in this session from the
        # routing confidence log.  Each entry in routing_confidence_log
        # records a specialist's routed tier; we collect the unique set
        # and key it as sorted pipe-delimited model tiers.
        models_used: list[str] = []
        for entry in session.routing_confidence_log:
            model_id = entry.get("routed_tier", "")
            if model_id and model_id not in models_used:
                models_used.append(model_id)
        if len(models_used) >= 2:
            synergy_key = "|".join(sorted(models_used))
            old_synergy = new_model.model_synergies.get(synergy_key, 0.5)
            session_success = 0.0 if session.error else 1.0
            alpha = 0.3
            new_model.model_synergies[synergy_key] = (
                alpha * session_success + (1 - alpha) * old_synergy
            )
            self.ogm.upsert(new_model)
            logger.info(
                "[CONCEPT:AHE-3.3] Model synergy updated: %s → %.2f",
                synergy_key,
                new_model.model_synergies[synergy_key],
            )

        # CONCEPT:ORCH-1.2 — Invalidate hot cache so routing reflects new self-knowledge
        from ...core.config import invalidate_registry_cache

        invalidate_registry_cache()

        logger.info(
            "Self-model updated: v%d (sessions=%d, domains=%d)",
            new_model.version,
            new_model.total_sessions,
            len(new_model.domain_success_rates),
        )

    # ── Query Interface ───────────────────────────────────────────────

    def query_capabilities(self, domain: str) -> dict[str, float]:
        """Return confidence scores for capabilities in a domain.

        Args:
            domain: The domain to query (e.g., "gitlab", "servicenow").

        Returns:
            Dict with ``success_rate``, ``confidence``, ``proficiency``.
        """
        current = self.get_current()
        if not current:
            return {"success_rate": 0.0, "confidence": 0.0, "proficiency": 0.0}

        return {
            "success_rate": current.domain_success_rates.get(domain, 0.0),
            "confidence": current.capability_confidence.get(domain, 0.0),
            "proficiency": current.tool_proficiency.get(domain, 0.0),
        }

    def get_best_synergies(
        self, available_models: list[str], top_k: int = 3
    ) -> list[tuple[str, float]]:
        """CONCEPT:AHE-3.3 — Find historically successful model combinations.

        Filters synergy records to only include combinations possible with
        the currently available models.

        Args:
            available_models: List of model IDs currently available.
            top_k: Number of top synergies to return.

        Returns:
            Sorted list of ``(synergy_key, success_rate)`` tuples,
            highest rate first.
        """
        current = self.get_current()
        if not current or not current.model_synergies:
            return []

        available_set = set(available_models)
        compatible: list[tuple[str, float]] = []
        for key, rate in current.model_synergies.items():
            models_in_combo = set(key.split("|"))
            if models_in_combo.issubset(available_set):
                compatible.append((key, rate))

        compatible.sort(key=lambda x: x[1], reverse=True)
        return compatible[:top_k]

    def temporal_trend(self, domain: str, lookback: int = 5) -> list[float]:
        """Traverse the SUPERSEDES chain to get historical performance.

        Args:
            domain: The domain to track.
            lookback: Number of versions to look back.

        Returns:
            List of success rates, oldest first.
        """
        trend: list[float] = []
        current = self.get_current()
        if not current:
            return trend

        node = current
        for _ in range(lookback):
            trend.append(node.domain_success_rates.get(domain, 0.0))
            # Find predecessor via SUPERSEDES
            prev = None
            if self.engine.backend:
                results = self.engine.backend.execute(
                    "MATCH (n {id: $nid})-[:SUPERSEDES]->(prev:MemoryRetriever) RETURN prev",
                    {"nid": node.id},
                )
                if results:
                    data = results[0].get("prev", results[0])
                    prev = self.ogm._deserialize(data, MemoryRetrieverNode)
            else:
                for succ in self.engine.graph.successors(node.id):
                    edge_data = self.engine.graph.get_edge_data(node.id, succ)
                    if edge_data:
                        for _, edata in edge_data.items():
                            if edata.get("type") == RegistryEdgeType.SUPERSEDES:
                                ndata = dict(self.engine.graph.nodes[succ])
                                prev = self.ogm._deserialize(ndata, MemoryRetrieverNode)
                                break

            if prev is None:
                break
            node = prev

        trend.reverse()  # Oldest first
        return trend

    def explain_self(self) -> str:
        """Generate a structured description of the agent's capabilities.

        Returns:
            A markdown-formatted capability summary.
        """
        current = self.get_current()
        if not current:
            return "No self-model available yet."

        lines = [
            f"# Agent Self-Model (v{current.version})",
            f"**Sessions**: {current.total_sessions} | "
            f"**Tasks completed**: {current.total_tasks_completed}",
            "",
            "## Domain Proficiency",
        ]

        for domain, rate in sorted(
            current.domain_success_rates.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            bar = "█" * int(rate * 10) + "░" * (10 - int(rate * 10))
            lines.append(f"- **{domain}**: {bar} {rate:.0%}")

        if current.known_failure_patterns:
            lines.append("")
            lines.append("## Known Failure Patterns")
            for pattern in current.known_failure_patterns[:10]:
                lines.append(f"- {pattern}")

        # CONCEPT:AHE-3.3 — Model Synergy section
        if current.model_synergies:
            lines.append("")
            lines.append("## Model Synergies")
            for combo, rate in sorted(
                current.model_synergies.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]:
                bar = "█" * int(rate * 10) + "░" * (10 - int(rate * 10))
                lines.append(f"- **{combo}**: {bar} {rate:.0%}")

        return "\n".join(lines)

    # ── OWL Integration ───────────────────────────────────────────────

    def promote_to_owl(self, owl_bridge: OWLBridge) -> int:
        """Push self-model triples into the OWL ontology.

        Creates triples for:
            - ``MemoryRetriever rdf:type Agent``
            - ``MemoryRetriever hasCapability Capability_X``
            - ``Capability_X confidenceScore 0.85``
            - ``MemoryRetriever knownWeakness FailurePattern_Y``

        This enables the OWL reasoner to infer routing decisions like
        "I am competent at GitLab tasks" or "I should delegate medical tasks."

        Args:
            owl_bridge: The ``OWLBridge`` instance to push triples into.

        Returns:
            Number of triples promoted.
        """
        current = self.get_current()
        if not current:
            return 0

        promoted = 0

        # Promote domain capabilities as graph nodes for OWL
        for domain, rate in current.domain_success_rates.items():
            cap_id = f"cap:{domain}"
            self.engine.graph.add_node(
                cap_id,
                type="capability",
                name=f"Capability: {domain}",
                confidence=rate,
                importance_score=rate,
            )
            self.engine.link_nodes(
                current.id,
                cap_id,
                "provides_capability",
                {"confidence_score": rate},
            )
            promoted += 1

        # Promote failure patterns as observations
        for i, pattern in enumerate(current.known_failure_patterns[:10]):
            obs_id = f"fail_pattern:{i}"
            self.engine.graph.add_node(
                obs_id,
                type="observation",
                name=f"Failure Pattern: {pattern[:50]}",
                content=pattern,
                importance_score=0.7,
            )
            self.engine.link_nodes(
                current.id,
                obs_id,
                "observes",
                {"is_weakness": True},
            )
            promoted += 1

        logger.info("Promoted %d self-model triples to OWL-ready graph", promoted)
        return promoted
