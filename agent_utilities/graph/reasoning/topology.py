#!/usr/bin/python
from __future__ import annotations

"""Versioned, graph-addressable reasoning-topology resource.

CONCEPT:AU-ORCH.planning.reasoning-graph-topologies

Each of the six topology modules in this package (``cot``, ``tot``, ``got``,
``react``, ``rap``) publishes exactly one :class:`TopologySpec`: a
content-addressed digest, the node contracts (``NodeKind`` values) it emits,
its shared-state schema, its declared loop/tool/token/cost/time budgets, the
classified termination conditions it can produce, and its checkpoint (memento)
semantics. This is deliberately the SAME resource shape this repo already uses
for skills/prompts/specs
(:class:`agent_utilities.models.knowledge_graph.ArtifactVersionNode` and its
``SkillVersionNode``/``SpecVersionNode`` subclasses) — a
:class:`~agent_utilities.models.knowledge_graph.ReasoningTopologyVersionNode`
is the same content-addressed-artifact contract applied to a reasoning
topology instead of a skill.

KG provenance is written best-effort and engine-optional: ``add_node`` at
registration, nothing after a run. WHICH topology runs is not decided here:
it is the ``au.reasoning.topology`` decision point
(:mod:`agent_utilities.decide.consumers.reasoning_topology`, EH-474), where EG
scores the runnable topologies with a calibrated head or abstains. The former
self-reported EMA outcome store on the topology node is deleted, as ST-7
deleted the swarm-topology one; ``scripts/check_topology_authority.py`` keeps
it deleted.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from ...models.knowledge_graph import ReasoningTopologyVersionNode

logger = logging.getLogger(__name__)

#: The ReasoningState fields every topology in this package reads/writes —
#: the "shared-state schema" part of the topology resource.
STATE_SCHEMA_FIELDS: tuple[str, ...] = (
    "nodes",
    "frontier",
    "visited",
    "candidates",
    "tool_calls",
    "rationale_summary",
)

#: Every topology resumes from a serialized ``ReasoningState`` snapshot after a
#: budget-halted run — one checkpoint contract for all six topologies.
CHECKPOINT_SEMANTICS = (
    "ReasoningState.to_memento()/from_memento() — a budget-halted run's state "
    "is fully recoverable; resuming re-enters the same topology with all "
    "nodes/frontier/candidates/tool_calls/rationale intact."
)


@dataclass(frozen=True)
class TopologySpec:
    """A versioned, content-addressed, graph-addressable topology resource."""

    name: str
    version: str
    node_contracts: tuple[str, ...]
    loop_budget: int
    tool_budget: int = 0
    token_budget: int | None = None
    cost_budget_usd: float | None = None
    time_budget_s: float | None = None
    termination_conditions: tuple[str, ...] = ()
    checkpoint_semantics: str = CHECKPOINT_SEMANTICS
    state_schema: tuple[str, ...] = STATE_SCHEMA_FIELDS

    @property
    def digest(self) -> str:
        """Stable content-address over the resource's identity-defining fields."""
        payload = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "node_contracts": list(self.node_contracts),
                "state_schema": list(self.state_schema),
                "loop_budget": self.loop_budget,
                "tool_budget": self.tool_budget,
                "termination_conditions": list(self.termination_conditions),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def topology_id(self) -> str:
        return f"topology:{self.name}:{self.digest}"

    def to_node(self) -> ReasoningTopologyVersionNode:
        """Render this spec as the KG-modeled versioned artifact resource."""
        return ReasoningTopologyVersionNode(
            id=self.topology_id,
            name=f"{self.name} v{self.version}",
            artifact_kind="reasoning_topology",
            artifact_id=self.name,
            version_hash=self.digest,
            node_contracts=list(self.node_contracts),
            loop_budget=self.loop_budget,
            tool_budget=self.tool_budget,
            token_budget=self.token_budget,
            cost_budget_usd=self.cost_budget_usd,
            time_budget_s=self.time_budget_s,
            termination_conditions=list(self.termination_conditions),
            checkpoint_semantics=self.checkpoint_semantics,
        )


def register_topology(engine: Any, spec: TopologySpec) -> None:
    """Best-effort KG registration of a topology resource (never raises).

    Mirrors :meth:`agent_utilities.graph.topology_engine.TopologyEngine.
    _record_materialization`'s pattern exactly: an engine-optional
    ``add_node`` call carrying the resource's typed properties.
    """
    if engine is None:
        return
    try:
        node = spec.to_node()
        engine.add_node(
            spec.topology_id,
            node.type.value,
            node.model_dump(),
        )
    except Exception as exc:  # noqa: BLE001 — registration is provenance, never fatal
        logger.debug("register_topology: failed to record %s: %s", spec.name, exc)
