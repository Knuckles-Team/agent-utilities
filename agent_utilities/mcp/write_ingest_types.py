"""Typed request value object for ``write_ingest_tools``'s ``_ingest_action_*`` handlers.

Leaf module — imports NOTHING from ``agent_utilities`` (CX wD10-R-ARITY / BUG-CX-004 §9b,
same rationale as ``agent_utilities.mcp.bus_types`` / ``agent_utilities.mcp.ontology_types`` /
``agent_utilities.mcp.engine_surface_types``: ``agent_utilities.mcp.tools`` is the highest-
leverage node in au's import SCC because its ``__init__.py`` eagerly imports all 35
tool-registration modules).

``graph_ingest`` keeps its full wire signature unchanged — that is the published MCP tool
schema FastMCP derives from the function signature, so it is the wire contract. Every one of
its 36 ``_ingest_action_*`` handlers (dispatched through ``_INGEST_ACTION_DISPATCH``) shared
the SAME 13-parameter positional signature — ``(engine, action, target_path, max_depth,
agent_id, job_id, priority_bucket, corpus_name, base_path, description, content_type,
connection, graph)`` — which is the exact anti-pattern the review named: the union of every
action's parameters threaded through every handler whether or not that action reads them.
This dataclass replaces that tuple; ``engine`` stays a separate positional argument since it
is a live collaborator, not request data (same split as ``bus_types.BusExecContext``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IngestRequest:
    """One ``graph_ingest`` call, typed. Each of the 36 actions reads only the 2-6 fields
    relevant to it — see ``graph_ingest``'s tool description for the per-action field map."""

    action: str = "ingest"
    target_path: str = ""
    max_depth: int = 3
    agent_id: str = ""
    job_id: str = ""
    priority_bucket: int = 1
    corpus_name: str = ""
    base_path: str = ""
    description: str = ""
    content_type: str = ""
    connection: str = ""
    graph: str = ""
