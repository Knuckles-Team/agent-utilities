"""Unified skill+tool capability contract (CONCEPT:AU-KG.retrieval.unified-capability-contract).

A `skill://` resource and an MCP `Tool` are two different WIRE shapes, but the
SAME thing to a caller ranking or binding a capability: a named, describable,
rankable, bindable unit of work. Before this module, `find_tools` ranked only
fleet `Tool` entries and `Orchestrator.resolve_capability` special-cased
`Skill`/`WorkflowDefinition` node types with no notion of a `Tool` at all — two
disjoint, kind-branching paths a caller had to know about in advance.

`Capability` is the one contract both a `Tool` graph/catalog node and a
`Skill`/`CallableResource` graph/catalog node (in-loop OR served over
Skills-over-MCP `skill://` resources) satisfy: a `kind`, a stable `id`, a
display `name`, a ranking `score`, an optional owning `server`, and
`to_binding()` — the exact keyword arguments `graph_orchestrate` /
`Orchestrator.execute_agent` / `execute_capability` already accept
(`skill_name`, `tool_server`, `allowed_tools`, `agent_name`). A caller resolving
a capability from ranked intent search never branches on `kind` — it calls
`.to_binding()` and spreads the result into the SAME delegation call, whichever
kind won the ranking.

This module is intentionally dependency-free (no KG/engine imports) so both the
low-level fleet multiplexer (`mcp/multiplexer.py`, which stays decoupled from
the knowledge_graph layer) and the orchestration layer
(`orchestration/manager.py`) can share one classification+binding contract
without a new cross-layer dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

CapabilityKind = Literal["tool", "skill", "workflow", "agent"]

__all__ = ["Capability", "CapabilityKind", "capability_kind_from_node"]


@dataclass(frozen=True)
class Capability:
    """One entry in the unified skill+tool capability space.

    Attributes:
        kind: ``"tool"`` (a fleet MCP tool), ``"skill"`` (an in-loop or
            fleet-served ``skill://`` skill), ``"workflow"`` (an ingested
            ``WorkflowDefinition``), or ``"agent"`` (a named agent/expert,
            the fallback when nothing more specific resolved).
        id: Stable graph/catalog identity (e.g. ``tool_<server>_<name>`` or
            ``skill_<server>_<name>``, or a canonical ``skill:<slug>``).
        name: Display/dispatch name — what ``to_binding()`` passes downstream.
        description: Human-readable summary, for ranked-result display.
        score: Ranking score (higher is better); caller-comparable within one
            ranked result set, not across independent rankers.
        server: The owning MCP server for a ``tool`` or a fleet-served
            ``skill`` (``None`` for a purely local/in-loop skill or a named
            agent).
        source: Where this candidate came from (``"kg_hybrid"``,
            ``"fleet_probe"``, ``"caller"``, …) — provenance, not identity.
    """

    kind: CapabilityKind
    id: str
    name: str
    description: str = ""
    score: float = 0.0
    server: str | None = None
    source: str = ""

    def to_binding(self) -> dict[str, Any]:
        """Return the delegation kwargs that bind this capability.

        Spread the result into ``graph_orchestrate`` /
        ``Orchestrator.execute_agent`` / ``execute_capability`` — every kind
        lands on the SAME keyword surface those already accept
        (``skill_name`` / ``tool_server`` / ``allowed_tools`` / ``agent_name``),
        so binding a resolved capability never requires an if/elif on
        ``self.kind`` at the call site.
        """
        if self.kind == "tool":
            binding: dict[str, Any] = {"allowed_tools": [self.name]}
            if self.server:
                binding["tool_server"] = self.server
            return binding
        if self.kind in ("skill", "workflow"):
            binding = {"skill_name": self.name}
            if self.server:
                binding["tool_server"] = self.server
            return binding
        return {"agent_name": self.name}

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly projection for a ranked `find`/`find_tools` result."""
        payload: dict[str, Any] = {
            "kind": self.kind,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "score": round(float(self.score), 4),
            "bind": self.to_binding(),
        }
        if self.server:
            payload["server"] = self.server
        if self.source:
            payload["source"] = self.source
        return payload


def capability_kind_from_node(
    *, node_type: str = "", resource_type: str = "", node_id: str = ""
) -> CapabilityKind | Literal[""]:
    """Classify a graph/catalog node's capability kind from its typed shape.

    Table-driven so a new capability-bearing node type is one new predicate
    here, not a growing if/elif spread across every ranking call site — every
    caller (``Orchestrator.resolve_capability``, ``find``, ``find_tools``)
    calls this one function instead of re-deriving the classification.
    Returns ``""`` when the node is not a recognized capability (caller should
    skip it, not guess).
    """
    node_type_cf = (node_type or "").casefold()
    resource_type_cf = (resource_type or "").casefold()
    node_id_cf = (node_id or "").casefold()
    if node_type_cf == "workflowdefinition" or node_id_cf.startswith(
        ("workflow:", "skill_workflow:")
    ):
        return "workflow"
    if resource_type_cf == "agent_skill" or node_type_cf in {"skill", "agentskill"}:
        return "skill"
    if node_type_cf == "tool" or node_id_cf.startswith("tool_"):
        return "tool"
    return ""
