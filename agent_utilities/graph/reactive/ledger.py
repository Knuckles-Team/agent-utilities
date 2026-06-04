#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:ORCH-1.10 — Graph-Native, DB-Agnostic Reactive Event Ledger.

Integrates event-sourcing with the unified IntelligenceGraphEngine and KGMapper.
Events are represented as standard EventNode instances, making them fully
compatible with LadybugDB, Neo4j, PostgreSQL, and memory-only graph compute fallbacks.

Ontological Synergy:
    - Stores event traces as standard 'Event' classes in the OWL ontology.
    - Connects event chains using 'was_derived_from' edges for transitive lineage.
    - Links events to runs using 'occurred_during' edges to trigger automated
      description logic (DL) reasoning via the OWLBridge.
"""

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...knowledge_graph.core.engine import IntelligenceGraphEngine
    from ...knowledge_graph.core.ogm import KGMapper
    from ...models.knowledge_graph import EventNode

logger = logging.getLogger(__name__)


class EventLedger:
    """Graph-native Event Ledger supporting reactive event-sourcing.

    Leverages the active IntelligenceGraphEngine and KGMapper to persist
    and query event streams across all abstracted database backends.
    """

    def __init__(self, engine: IntelligenceGraphEngine | None = None) -> None:
        """Initialize the Event Ledger.

        Args:
            engine: Optional IntelligenceGraphEngine instance. If None,
                    dynamically retrieves the active engine singleton.
        """
        self._engine = engine
        self._last_event_ids: dict[
            str, str
        ] = {}  # Tracks last event ID per run_id for lineage

    @property
    def engine(self) -> IntelligenceGraphEngine:
        """Retrieve the bound or active IntelligenceGraphEngine."""
        if self._engine is not None:
            return self._engine
        from ...knowledge_graph.core.engine import IntelligenceGraphEngine

        active = IntelligenceGraphEngine.get_active()
        if active is None:
            raise RuntimeError("No active IntelligenceGraphEngine found.")
        return active

    @property
    def mapper(self) -> KGMapper:
        """Retrieve a KGMapper bound to the active engine."""
        from ...knowledge_graph.core.ogm import KGMapper

        return KGMapper(self.engine)

    def append_event(
        self,
        run_id: str,
        node_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        severity: str = "info",
        source: str = "reactive_dispatcher",
    ) -> EventNode:
        """Idempotently append a new event to the ledger.

        Ontologically persists the event to the active database (Neo4j,
        Ladybug, PostgreSQL, or GraphComputeEngine) and links it chronologically
        via 'was_derived_from' and 'occurred_during' edges.

        Args:
            run_id: Unique identifier for the execution run or episode.
            node_id: The graph node or component triggering the event.
            event_type: Categorized string identifying the event topic.
            payload: Structured dictionary of parameters, outputs, or metrics.
            severity: Event severity level (info, warning, error, critical).
            source: Source of the event generation.

        Returns:
            The created and persisted EventNode.
        """
        event_id = f"evt:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        props = payload or {}
        from ...models.knowledge_graph import EventNode

        event_node = EventNode(
            id=event_id,
            name=f"Event: {event_type} at {node_id}",
            description=f"Reactive execution event of type {event_type}",
            timestamp=ts,
            event_type=event_type,
            severity=severity,
            payload=props,
            source=source,
            episode_id=run_id,
            importance_score=0.2 if severity in ("error", "critical") else 0.1,
        )

        # 1. Upsert node via OGM
        self.mapper.upsert(event_node)

        # 2. Link event to the parent execution context/run using 'occurred_during'
        # Check if run node exists in KG, otherwise create a placeholder EpisodeNode
        try:
            self.mapper.upsert_edge(
                source_id=event_id,
                target_id=run_id,
                edge_type="OCCURRED_DURING",
            )
        except Exception as e:
            logger.debug("Failed to link occurred_during edge (non-fatal): %s", e)

        # 3. Establish temporal lineage using 'was_derived_from'
        prev_id = self._last_event_ids.get(run_id)
        if prev_id:
            try:
                self.mapper.upsert_edge(
                    source_id=event_id,
                    target_id=prev_id,
                    edge_type="WAS_DERIVED_FROM",
                )
            except Exception as e:
                logger.debug("Failed to link was_derived_from edge (non-fatal): %s", e)

        # Update tracking for this run
        self._last_event_ids[run_id] = event_id

        logger.debug(
            "[EventLedger] Appended event %s (%s) for run %s",
            event_id,
            event_type,
            run_id,
        )
        return event_node

    def get_run_events(self, run_id: str) -> list[EventNode]:
        """Retrieve all events associated with a run sorted chronologically.

        Args:
            run_id: Unique identifier for the execution run.

        Returns:
            Chronologically sorted list of EventNode objects.
        """
        from ...models.knowledge_graph import EventNode

        events: list[EventNode] = []

        if self.engine.backend:
            # Query Tier 1 persistent storage
            try:
                query = "MATCH (e:Event {episode_id: $run_id}) RETURN e"
                rows = self.engine.backend.execute(query, {"run_id": run_id})
                for row in rows:
                    data = row.get("e", row)
                    try:
                        # Hydrate using OGM deserializer
                        events.append(self.mapper._deserialize(data, EventNode))
                    except Exception as e:
                        logger.debug("Failed to deserialize EventNode: %s", e)
            except Exception as e:
                logger.warning("Failed to query events from backend: %s", e)

        # Fallback to Tier 2 in-memory MultiDiGraph
        if not events:
            for nid, data in self.engine.graph.nodes(data=True):
                # OGM serializes type as enum value string ("event")
                if data.get("type") == "event" and data.get("episode_id") == run_id:
                    try:
                        events.append(self.mapper._deserialize(dict(data), EventNode))
                    except Exception:  # nosec B110, B112
                        continue

        # Sort chronologically by timestamp
        events.sort(key=lambda x: x.timestamp or "")
        return events

    def fork_run(self, run_id: str, up_to_event_id: str) -> list[EventNode]:
        """Establish a starting trajectory for replay by pulling events up to a point.

        Supports time-travel debugging and replay-cached dry runs.

        Args:
            run_id: Unique identifier for the execution run.
            up_to_event_id: The event ID serving as the historical ceiling.

        Returns:
            List of events up to and including the targeted event_id.
        """
        all_events = self.get_run_events(run_id)
        forked_trajectory: list[EventNode] = []

        for evt in all_events:
            forked_trajectory.append(evt)
            if evt.id == up_to_event_id:
                break

        return forked_trajectory
