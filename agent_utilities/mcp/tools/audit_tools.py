"""graph_audit — Tamper-evident audit-ledger MCP tool (G23, audit-trail closure).

CONCEPT:AU-KG.audit.hash-chain-verify

Exposes the engine's mature Rust hash-chained audit log
(``epistemic-graph/src/audit.rs`` — every durable mutation already chains into a
per-graph SHA-256 hash chain, redb ``AUDIT`` table) to Python/MCP for the first
time (the ``epistemic_graph`` client package does not yet wrap
``Method::AuditVerify`` as a typed ``LedgerClient`` method, so
:meth:`~agent_utilities.knowledge_graph.core.graph_compute.GraphComputeEngine.
audit_verify` reaches it via the existing raw wire escape hatch — no Rust
rebuild required).

Two actions:

- ``verify`` — cryptographically walk the default graph's audit chain and report
  whether every entry's stored hash still matches its recomputed link (tamper
  evidence). Degrades cleanly with a clear message when the connected engine
  build/config doesn't support it (no ``security`` feature compiled in, or no
  durable redb persist dir configured) — the remaining dependency in that case
  is a corresponding change in the ``epistemic-graph`` repo, NOT this one.
- ``for_target`` — the entity-anchored reverse index (the KG-side half of G23):
  every ``:ToolCall`` that acted on a given entity id, in call order, plus a
  best-effort chain-verification snapshot alongside it. See
  :meth:`~agent_utilities.orchestration.manager.Orchestrator.get_tool_calls_for_target`.

Mirrors the ``graph_ops_causal`` action-router shape (single ``@mcp.tool``, an
``action`` enum, registered into ``kg_server.REGISTERED_TOOLS``) rather than
inventing a new tool convention.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server


def register_audit_tools(mcp: Any) -> None:
    """Register the ``graph_audit`` group on the given FastMCP server."""

    @mcp.tool(
        name="graph_audit",
        description=(
            "Tamper-evident audit ledger (G23): verifies the engine's hash-chained "
            "durable-mutation audit log (Rust `epistemic-graph/src/audit.rs`, SHA-256 "
            "per (graph, seq)) and reconstructs 'what happened to entity X' from the "
            "KG's own :ToolCall provenance. Actions: 'verify' (walk the target "
            "graph's audit chain; ok=true when every entry's stored hash matches its "
            "recomputed link — a broken chain reports first_broken_seq. Degrades "
            "cleanly with a clear error when the connected engine build/config "
            "doesn't expose it), 'for_target' (every :ToolCall -[:ACTED_ON]-> "
            "target_id, in call order, plus a best-effort verify() snapshot "
            "alongside it)."
        ),
        tags=["graph-os", "audit", "governance", "provenance"],
    )
    def graph_audit(
        action: str = Field(
            default="verify", description="verify | for_target"
        ),
        target_id: str = Field(
            default="",
            description="Entity id to reverse-index tool-call provenance for "
            "(required for for_target).",
        ),
    ) -> str:
        """Verify the tamper-evident audit chain, or reverse-index provenance for an entity."""
        action = (action or "verify").strip().lower()

        if action == "verify":
            return json.dumps(_verify())
        if action == "for_target":
            if not target_id:
                return json.dumps(
                    {
                        "surface": "audit",
                        "action": action,
                        "error": "target_id required for for_target",
                    }
                )
            return json.dumps(_for_target(target_id), default=str)
        return json.dumps(
            {"surface": "audit", "action": action, "error": f"unknown action {action!r}"}
        )

    kg_server.REGISTERED_TOOLS["graph_audit"] = graph_audit
    # No bespoke endpoint needed — the generic REST-twin factory in
    # kg_server._build_server (CONCEPT:AU-KG.coordination.engine-message-broker) mounts
    # POST /audit for every ACTION_TOOL_ROUTES entry without a bespoke handler,
    # dispatching through the SAME _execute_tool core.
    kg_server.ACTION_TOOL_ROUTES["graph_audit"] = "/audit"


def _verify() -> dict[str, Any]:
    """Cryptographically verify the default graph's hash-chained audit log."""
    engine = kg_server._get_engine()
    if engine is None or getattr(engine, "graph", None) is None:
        return {
            "surface": "audit",
            "action": "verify",
            "error": "IntelligenceGraphEngine not active",
        }
    try:
        report = engine.graph.audit_verify()
    except Exception as exc:  # noqa: BLE001 — surface engine errors as data, not 500
        return {
            "surface": "audit",
            "action": "verify",
            "available": False,
            "error": (
                "audit ledger not exposed by this engine build/config "
                f"({exc}). Requires the epistemic-graph `security` cargo "
                "feature (part of the default `full` build) AND a durable "
                "redb persist dir configured — otherwise this is a "
                "corresponding epistemic-graph-side gap, not an "
                "agent-utilities one."
            ),
        }
    return {"surface": "audit", "action": "verify", "available": True, **report}


def _for_target(target_id: str) -> dict[str, Any]:
    """Entity-anchored reverse index of :ToolCall provenance, plus a verify snapshot."""
    from agent_utilities.orchestration.manager import Orchestrator

    engine = kg_server._get_engine()
    if engine is None:
        return {
            "surface": "audit",
            "action": "for_target",
            "target_id": target_id,
            "error": "IntelligenceGraphEngine not active",
        }
    orch = Orchestrator(engine)
    result = orch.get_tool_calls_for_target(target_id)
    return {"surface": "audit", "action": "for_target", **result}
