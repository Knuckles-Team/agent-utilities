"""KG-query and MCP-tool-catalog validation stages for the committed validation harness.

Two stage callables, ``mcp_tool_catalog`` and ``kg_general``, prove — with real,
verified-working queries against the LIVE engine, not a synthetic fixture — that the
ecosystem's tool surface and general knowledge are actually reachable through the KG.
Both are read-only: neither writes to the graph, mints a session, restarts anything,
or calls ``sys.exit``. The driver (``scripts/full_validation_harness.py``, a parallel
lane) owns session/engine acquisition, the exit-code contract, and stage sequencing;
these two functions are handed an already-acquired engine handle and either return a
structured result or raise.

Reuses the SAME engine-acquisition convention as ``scripts/delegation_probe.py``
stage 3 (``IntelligenceGraphEngine.get_active()`` in preference to constructing a
second engine) — see that module's docstring for why: a private engine would not be
the one delegation/production actually uses, and could mask a live misbinding. This
module does not call ``get_active()`` itself; it is handed the engine the driver
already acquired, so ``kg_general``/``mcp_tool_catalog`` work against a fake engine in
tests too.

CYPHER-DIALECT CONSTRAINT (verified live against the graph-os pod on 2026-08-24,
kubectl -n platform exec into ``graph-os-5bd585f46b-vj882``)
-----------------------------------------------------------------------------------
``MATCH (n) RETURN labels(n) AS label, count(*)`` is NOT supported — it raises
``CypherEngineError`` (native Cypher authority rejected the request). Confirmed live,
not just from the schema-hint doc. This module never emits ``labels(``.

Labelled ``MATCH`` patterns (``MATCH (n:Tool) RETURN count(n) AS c``) DO work and are
used throughout below — also confirmed live (e.g. ``MATCH (s:MCPServer) RETURN
count(s) AS c`` returned a real, non-zero count in the same session).

For the label-cardinality sweep, the obvious SQL translation of the engine's own
schema hint (``agent_utilities/knowledge_graph/core/nl_query.py``) —
``SELECT type, COUNT(*) FROM nodes GROUP BY type`` — does NOT work here: it raised
``Schema error: No field named type`` live, because this fleet's ingestion writes the
property key ``node_type`` (never a bare ``type``), and the SQL surface is
schema-on-read from the properties nodes actually carry — a column only exists if
some node actually has that property. ``SELECT node_type, COUNT(*) FROM nodes GROUP
BY node_type ORDER BY c DESC LIMIT 25`` is the verified-working replacement (confirmed
live, returned 25 rows headed by ``RuntimeSignal``/``WorkItem``/``Concept``). A
``label`` property filter is avoided for the same reason the engine's own schema hint
warns about: an unrelated, usually-empty ``label`` property exists on many nodes and
would silently return 0 rows instead of erroring.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from agent_utilities.observability.trace_ontology import (
    OUTCOME_NODE_LABEL,
    TOOL_CALL_NODE_LABEL,
    TRACE_NODE_LABEL,
    TRACE_USED_TOOL_EDGE,
)

# --------------------------------------------------------------------------------
# Node/edge labels not exported as ontology constants.
# --------------------------------------------------------------------------------
# graph-os's own tool surface (agent_utilities/knowledge_graph/ingestion/engine.py
# :3880-3920), id pattern `tool_{server}_{name}`.
TOOL_NODE_LABEL = "Tool"

# The fleet-wide EXECUTABLE catalog written under (:Server)-[:PROVIDES]->
# (:CallableResource) by engine_mcp_discovery.py / engine_ingestion.py:267. This is
# the one delegation actually needs: agent_runner.py:2108-2133 fails closed with
# "has no runnable CallableResource (unmet precondition 'skill_body_served')" if a
# skill lacks one.
CALLABLE_RESOURCE_NODE_LABEL = "CallableResource"

# The `:Server` label engine_mcp_discovery.py / engine_ingestion.py:267 link
# CallableResource nodes FROM (id pattern `srv:{name}`) via PROVIDES. This is a
# DIFFERENT node type from the fleet-wide `:MCPServer` node type below — no shared
# constant exists for either in the codebase, so both are named literally here with
# this note as the cross-reference, rather than risk conflating them.
SERVER_NODE_LABEL = "Server"
PROVIDES_EDGE = "PROVIDES"

# The fleet-wide MCP server catalog entry node type, written by graph-os's own
# self-tool-surface ingestion (engine.py:3894, node_type="MCPServer") and by
# source_sync's fleet ingestion (`_write_fleet_nodes`, source_sync.py:502). Verified
# live: `MATCH (s:MCPServer) RETURN count(s) AS c` returned a real non-zero count.
MCP_SERVER_NODE_LABEL = "MCPServer"

CONCEPT_NODE_LABEL = "Concept"

# Cap on how many (:Server)-[:PROVIDES]->(:CallableResource) id rows are pulled back
# to dedupe client-side. Cypher DISTINCT is documented unreliable for this query
# shape (agent_utilities/knowledge_graph/core/nl_query.py:87-88: "the engine's
# DISTINCT does not reliably collapse duplicates for this query shape either"), so
# this module never relies on it — it counts distinct ids in Python instead.
_PROVIDES_REACHABLE_LIMIT = 20_000

# Top-N labels reported by the SQL cardinality sweep.
_LABEL_SWEEP_TOP_N = 25

_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_label(label: str) -> str:
    """Defence-in-depth: every label interpolated into a Cypher string here is a
    module-level constant, never caller input — but assert the shape anyway so a
    future edit cannot silently turn this into a query-injection path.
    """
    if not _LABEL_RE.fullmatch(label):
        raise ValueError(f"unsafe node/edge label for Cypher interpolation: {label!r}")
    return label


def _emit(stage: str, ok: bool, detail: str = "", elapsed: float | None = None) -> None:
    """Match ``scripts/delegation_probe.py``'s reporting convention exactly, for
    this module's own ``__main__`` self-test / manual-run mode. The driver script
    owns the real stage loop and is free to use this same convention or its own.
    """
    mark = "PASS" if ok else "FAIL"
    t = f" [{elapsed:6.2f}s]" if elapsed is not None else ""
    print(f"  {mark:4s} {stage:16s}{t} {detail}"[:600], flush=True)


async def _node_count(engine: Any, label: str) -> int:
    """``MATCH (n:<label>) RETURN count(n) AS c`` — a labelled MATCH pattern, never
    ``labels(n)``. Verified working live for Tool/CallableResource/RunTrace/
    ToolCall/Concept/MCPServer/OutcomeEvaluation/Server.
    """
    label = _safe_label(label)
    rows = await asyncio.to_thread(
        engine.backend.execute, f"MATCH (n:{label}) RETURN count(n) AS c", {}
    )
    if not rows:
        return 0
    return int(rows[0].get("c") or 0)


async def _total_node_count(engine: Any) -> int:
    rows = await asyncio.to_thread(
        engine.backend.execute, "MATCH (n) RETURN count(n) AS c", {}
    )
    if not rows:
        return 0
    return int(rows[0].get("c") or 0)


async def _provides_reachable_count(engine: Any) -> int:
    """Distinct ``:CallableResource`` ids reachable via
    ``(:Server)-[:PROVIDES]->(:CallableResource)``.

    A ``:CallableResource`` that exists but is unreachable via this exact edge is a
    known failure shape (an orphaned resource that delegation cannot bind to its
    owning server) — this is why the presence check below is a distinct, separate
    query from ``_node_count(engine, CALLABLE_RESOURCE_NODE_LABEL)`` rather than an
    assumption derived from it.
    """
    server = _safe_label(SERVER_NODE_LABEL)
    resource = _safe_label(CALLABLE_RESOURCE_NODE_LABEL)
    edge = _safe_label(PROVIDES_EDGE)
    rows = await asyncio.to_thread(
        engine.backend.execute,
        f"MATCH (s:{server})-[:{edge}]->(r:{resource}) "
        f"RETURN r.id AS id LIMIT {_PROVIDES_REACHABLE_LIMIT}",
        {},
    )
    ids = {row.get("id") for row in (rows or []) if row.get("id")}
    return len(ids)


async def _used_tool_edge_count(engine: Any) -> int:
    """``(:RunTrace)-[:USED_TOOL]->(:ToolCall)`` edge count, using the ontology
    constants so this can never drift from the writer's own schema.
    """
    trace = _safe_label(TRACE_NODE_LABEL)
    tool_call = _safe_label(TOOL_CALL_NODE_LABEL)
    edge = _safe_label(TRACE_USED_TOOL_EDGE)
    rows = await asyncio.to_thread(
        engine.backend.execute,
        f"MATCH (t:{trace})-[:{edge}]->(c:{tool_call}) RETURN count(*) AS c",
        {},
    )
    if not rows:
        return 0
    return int(rows[0].get("c") or 0)


async def _label_cardinality_sweep(engine: Any) -> list[dict[str, Any]]:
    """Top ~25 node labels by count, via the SQL catalog surface.

    Uses ``node_type`` (verified live), never ``type``/``label`` (both were tried
    live and rejected — see module docstring), and never ``MATCH (n) RETURN
    labels(n)`` (also verified live to raise ``CypherEngineError``).
    """
    rows = await asyncio.to_thread(
        engine.sql,
        "SELECT node_type, COUNT(*) AS c FROM nodes "
        f"GROUP BY node_type ORDER BY c DESC LIMIT {_LABEL_SWEEP_TOP_N}",
    )
    return [
        {"label": row.get("node_type"), "count": int(row.get("c") or 0)}
        for row in (rows or [])
        if row.get("node_type")
    ]


# ---------------------------------------------------------------------------------
# Stage: mcp_tool_catalog
# ---------------------------------------------------------------------------------
async def mcp_tool_catalog(engine: Any) -> dict[str, Any]:
    """Prove the MCP tool surface is actually ingested into the knowledge graph.

    Two distinct node types are checked, both required non-empty:

    * ``:Tool`` — graph-os's own tool surface.
    * ``:CallableResource`` — the fleet-wide EXECUTABLE catalog delegation binds to.

    Also asserts the ``PROVIDES`` edge actually links them: a ``:CallableResource``
    that exists but is unreachable via ``(:Server)-[:PROVIDES]->`` is orphaned from
    its ``Server`` and cannot be delegated to.

    Returns a structured dict with the three counts on success. Raises
    ``RuntimeError`` (never ``sys.exit``) describing every failing check when
    ``:Tool`` is empty, ``:CallableResource`` is empty, or ``:CallableResource`` is
    non-empty yet zero of them are reachable via ``PROVIDES``.
    """
    tool_count = await _node_count(engine, TOOL_NODE_LABEL)
    callable_resource_count = await _node_count(engine, CALLABLE_RESOURCE_NODE_LABEL)
    provides_reachable_count = await _provides_reachable_count(engine)

    failures: list[str] = []
    if tool_count == 0:
        failures.append(f":{TOOL_NODE_LABEL} count is 0")
    if callable_resource_count == 0:
        failures.append(f":{CALLABLE_RESOURCE_NODE_LABEL} count is 0")
    elif provides_reachable_count == 0:
        failures.append(
            f"0 :{CALLABLE_RESOURCE_NODE_LABEL} nodes reachable via "
            f"(:{SERVER_NODE_LABEL})-[:{PROVIDES_EDGE}]-> — every "
            f":{CALLABLE_RESOURCE_NODE_LABEL} is orphaned from its "
            f":{SERVER_NODE_LABEL}"
        )

    result = {
        "tool_count": tool_count,
        "callable_resource_count": callable_resource_count,
        "provides_reachable_count": provides_reachable_count,
    }
    detail = (
        f"tool={tool_count} callable_resource={callable_resource_count} "
        f"provides_reachable={provides_reachable_count}"
    )
    if failures:
        raise RuntimeError(f"mcp_tool_catalog failed: {'; '.join(failures)} ({detail})")
    return {**result, "detail": detail}


# ---------------------------------------------------------------------------------
# Stage: kg_general
# ---------------------------------------------------------------------------------
async def kg_general(engine: Any) -> dict[str, Any]:
    """Prove broad ingested-graph visibility, not just a single node count.

    * A label-cardinality sweep (top ~25 labels by count, via the SQL catalog
      surface) — must be non-empty.
    * Presence checks (count > 0) for ``:RunTrace``, ``:ToolCall``, ``:Concept``,
      the MCP server label (``:MCPServer``), and ``:OutcomeEvaluation`` — the labels
      that have an ontology constant are imported from
      ``agent_utilities.observability.trace_ontology`` rather than hardcoded, so
      this harness and the writer can never drift apart.
    * The ``(:RunTrace)-[:USED_TOOL]->(:ToolCall)`` edge must exist, not just the
      two node labels independently.

    Returns a structured dict with every count/sweep on success. Raises
    ``RuntimeError`` describing every failing check otherwise.
    """
    total_nodes = await _total_node_count(engine)
    label_sweep = await _label_cardinality_sweep(engine)
    run_trace_count = await _node_count(engine, TRACE_NODE_LABEL)
    tool_call_count = await _node_count(engine, TOOL_CALL_NODE_LABEL)
    concept_count = await _node_count(engine, CONCEPT_NODE_LABEL)
    mcp_server_count = await _node_count(engine, MCP_SERVER_NODE_LABEL)
    outcome_count = await _node_count(engine, OUTCOME_NODE_LABEL)
    used_tool_edge_count = await _used_tool_edge_count(engine)

    failures: list[str] = []
    if total_nodes == 0:
        failures.append("total node count is 0")
    if not label_sweep:
        failures.append("label-cardinality sweep returned 0 labels")
    if run_trace_count == 0:
        failures.append(f":{TRACE_NODE_LABEL} count is 0")
    if tool_call_count == 0:
        failures.append(f":{TOOL_CALL_NODE_LABEL} count is 0")
    if concept_count == 0:
        failures.append(f":{CONCEPT_NODE_LABEL} count is 0")
    if mcp_server_count == 0:
        failures.append(f":{MCP_SERVER_NODE_LABEL} count is 0")
    if outcome_count == 0:
        failures.append(f":{OUTCOME_NODE_LABEL} count is 0")
    if used_tool_edge_count == 0:
        failures.append(
            f"no (:{TRACE_NODE_LABEL})-[:{TRACE_USED_TOOL_EDGE}]->"
            f"(:{TOOL_CALL_NODE_LABEL}) edge found"
        )

    result = {
        "total_nodes": total_nodes,
        "label_sweep": label_sweep,
        "run_trace_count": run_trace_count,
        "tool_call_count": tool_call_count,
        "concept_count": concept_count,
        "mcp_server_count": mcp_server_count,
        "outcome_count": outcome_count,
        "used_tool_edge_count": used_tool_edge_count,
    }
    detail = (
        f"total_nodes={total_nodes} labels={len(label_sweep)} "
        f"run_trace={run_trace_count} tool_call={tool_call_count} "
        f"concept={concept_count} mcp_server={mcp_server_count} "
        f"outcome={outcome_count} used_tool_edges={used_tool_edge_count}"
    )
    if failures:
        raise RuntimeError(f"kg_general failed: {'; '.join(failures)} ({detail})")
    return {**result, "detail": detail}
