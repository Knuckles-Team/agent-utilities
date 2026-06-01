import asyncio
import contextlib
import logging
from typing import Any

from agent_utilities.models.company_brain import ConflictStatus

logger = logging.getLogger(__name__)


class RuleSynthesizerDaemon:
    """
    Background daemon that monitors RESOLVED_HUMAN events in the Knowledge Graph
    and synthesizes new RuleNodes, enabling self-compounding automated rules.
    """

    def __init__(self, engine: Any):
        self.engine = engine
        self._running = False
        self._task: asyncio.Task[Any] | None = None

    async def start(self):
        """Start the background daemon loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("RuleSynthesizerDaemon started.")

    async def stop(self):
        """Stop the background daemon loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("RuleSynthesizerDaemon stopped.")

    async def _loop(self):
        while self._running:
            try:
                await self.process_resolved_conflicts()
            except Exception as e:
                logger.error(f"RuleSynthesizerDaemon error in loop: {e}")
            await asyncio.sleep(60)  # Check every minute

    async def process_resolved_conflicts(self):
        """
        Query the epistemic-graph for conflicts that were manually resolved by humans,
        penalize the failing source, and synthesize a new RuleNode.
        """
        # 1. Fetch newly resolved human conflicts
        cypher = (
            "MATCH (c:ConflictRecord)-[:AFFECTS]->(n) "
            "WHERE c.status = $status AND c.rule_synthesized IS NULL "
            "RETURN c, n"
        )
        params = {"status": ConflictStatus.RESOLVED_HUMAN.value}

        # Hypothetical method to query epistemic graph
        results = self.engine.query_cypher(cypher, params)
        if not results:
            return

        for record in results:
            conflict = record.get("c", {})
            record.get("n", {})

            # 2. Penalize failing source
            losing_source = conflict.get("losing_source_system")
            if losing_source:
                self._apply_conflict_penalty(losing_source)

            # 3. Synthesize Rule
            rule_id = f"rule_{conflict.get('id')}"
            human_resolver = conflict.get("resolved_by", "Unknown Human")
            resolved_value = conflict.get("resolved_value")

            conflict_id = conflict.get("id")
            node_id = record.get("n", {}).get("id")

            # Create a RuleNode ensuring future automated resolution
            rule_cypher = (
                "MATCH (c:ConflictRecord {id: $conflict_id}) "
                "MATCH (n {id: $node_id}) "
                "MERGE (r:RuleNode {id: $rule_id}) "
                "SET r.description = $desc, r.confidence = 1.0, r.created_by = $human "
                "MERGE (r)-[:GOVERNS]->(n) "
                "SET c.rule_synthesized = true"
            )
            rule_params = {
                "conflict_id": conflict_id,
                "node_id": node_id,
                "rule_id": rule_id,
                "desc": f"Automatically synthesize rule based on human arbitration by {human_resolver} setting value to {resolved_value}",
                "human": human_resolver,
            }
            self.engine.query_cypher(rule_cypher, rule_params)
            logger.info(
                f"Synthesized new RuleNode {rule_id} from human-resolved conflict."
            )

    def _apply_conflict_penalty(self, source_system: str):
        """
        Apply a dynamic trust penalty to a source system after it was overridden
        by a human in a conflict.
        """
        # Find the TrustHierarchyEntry for this source
        cypher = "MATCH (t:TrustHierarchyEntry {source_system: $source}) RETURN t"
        params = {"source": source_system}
        results = self.engine.query_cypher(cypher, params)
        if not results:
            return

        entry_data = results[0].get("t", {})
        current_authority = float(entry_data.get("authority_level", 0.5))
        penalty = float(entry_data.get("conflict_penalty", 0.05))

        new_authority = max(0.0, current_authority - penalty)

        update_cypher = "MATCH (t:TrustHierarchyEntry {source_system: $source}) SET t.authority_level = $new_auth"
        self.engine.query_cypher(
            update_cypher, {"source": source_system, "new_auth": new_authority}
        )
        logger.info(
            f"Applied conflict penalty to {source_system}: {current_authority} -> {new_authority}"
        )
