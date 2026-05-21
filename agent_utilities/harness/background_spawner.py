#!/usr/bin/python
from __future__ import annotations

"""Background Context Spawner (CONCEPT:AHE-3.4).

Monitors background context shifts via the Knowledge Graph and autonomously
spawns specialized sub-agents dynamically.
"""


import asyncio
import logging
from typing import TYPE_CHECKING

from ..graph.dynamic_graph_orchestrator import DynamicSubgraphOrchestrator

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class BackgroundContextSpawner:
    """Spawns sub-agents based on KG context shifts.

    CONCEPT:AHE-3.4
    """

    def __init__(self, engine: IntelligenceGraphEngine, poll_interval_sec: int = 60):
        self.engine = engine
        self.poll_interval_sec = poll_interval_sec
        self._running = False
        self._task: asyncio.Task | None = None
        self.orchestrator = DynamicSubgraphOrchestrator(engine=self.engine)

    def start(self):
        """Start the background spawner loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("[CONCEPT:AHE-3.4] Started BackgroundContextSpawner.")

    def stop(self):
        """Stop the background spawner loop."""
        self._running = False
        if self._task:
            self._task.cancel()

    async def _poll_loop(self):
        while self._running:
            try:
                await self._check_context_shifts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in BackgroundContextSpawner loop: %s", e)

            await asyncio.sleep(self.poll_interval_sec)

    async def _check_context_shifts(self):
        """Check the KG for significant context shifts requiring sub-agents."""
        if not self.engine.backend:
            return

        # Query the KG for unresolved events or high-impact state changes
        # This aligns with KG-2.21 Multi-Timescale Memory
        try:
            results = self.engine.backend.execute(
                "MATCH (e:Event) WHERE e.resolved = false AND e.impact_score > 0.8 "
                "RETURN e.id AS event_id, e.description AS descriptionription "
                "LIMIT 1"
            )

            for r in results:
                event_id = r.get("event_id")
                desc = r.get("description", "")

                if desc:
                    logger.info(
                        "[CONCEPT:AHE-3.4] High-impact event detected: %s. "
                        "Dynamically synthesizing response team.",
                        event_id,
                    )

                    # Dynamically synthesize a team to handle this background event
                    team = self.orchestrator.synthesize_team(
                        query=desc,
                        domain="background_operations",
                        complexity=4,
                    )

                    logger.info(
                        "Synthesized team %s for event %s", team.team_id, event_id
                    )

                    # Mark as resolved to avoid loop (in a real system, the spawned
                    # agents would mark it resolved when finished).
                    self.engine.backend.execute(
                        "MATCH (e:Event {id: $eid}) SET e.resolved = true",
                        {"eid": event_id},
                    )
        except Exception as e:
            logger.debug("Failed to poll context shifts: %s", e)
