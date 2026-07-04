from __future__ import annotations

"""CONCEPT:AU-KG.research.research-pipeline-runner"""

import logging
import time
import typing
import uuid

if typing.TYPE_CHECKING:
    from .._engine_protocol import _EngineProtocol

    _Base = _EngineProtocol
else:
    _Base = object

from ...models.domains.ml_rlm import (
    OptimizationGoalNode,
    ParetoFrontierEntryNode,
    RLMActorNode,
)

logger = logging.getLogger(__name__)


class MachineLearningEngineMixin(_Base):
    """Machine Learning & RLM capabilities for the KG engine."""

    def register_rlm_actor(
        self, name: str, learning_rate: float, discount_factor: float
    ) -> str:
        """Register a new RLM actor for reinforcement learning tasks."""
        actor_id = f"rlm:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = RLMActorNode(
            id=actor_id,
            name=name,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            timestamp=ts,
        )
        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="RLMActor")
            self._upsert_node("RLMActor", actor_id, data)
        return actor_id

    def set_optimization_goal(
        self, actor_id: str, target_metric: str, threshold: float
    ) -> str:
        """Link an optimization goal to an RLM Actor."""
        goal_id = f"goal:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = OptimizationGoalNode(
            id=goal_id,
            name=f"Optimize {target_metric}",
            target_metric=target_metric,
            threshold=threshold,
            is_active=True,
            timestamp=ts,
        )
        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="OptimizationGoal")
            self._upsert_node("OptimizationGoal", goal_id, data)
            self.backend.execute(
                "MATCH (a:RLMActor {id: $aid}), (g:OptimizationGoal {id: $gid}) "
                "MERGE (a)-[:HAS_METRIC]->(g)",
                {"aid": actor_id, "gid": goal_id},
            )
        return goal_id

    def record_pareto_entry(
        self, obj1: float, obj2: float, dominates_id: str | None = None
    ) -> str:
        """Record an entry on the Pareto Frontier."""
        entry_id = f"pareto:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = ParetoFrontierEntryNode(
            id=entry_id,
            name=f"Pareto ({obj1}, {obj2})",
            objective_1_score=obj1,
            objective_2_score=obj2,
            is_dominated=False,
            timestamp=ts,
        )
        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="ParetoFrontierEntry")
            self._upsert_node("ParetoFrontierEntry", entry_id, data)

            if dominates_id:
                self.backend.execute(
                    "MATCH (p1:ParetoFrontierEntry {id: $pid1}), (p2:ParetoFrontierEntry {id: $pid2}) "
                    "MERGE (p1)-[:PARETO_DOMINATES]->(p2) "
                    "SET p2.is_dominated = true",
                    {"pid1": entry_id, "pid2": dominates_id},
                )
        return entry_id
