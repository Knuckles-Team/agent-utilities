#!/usr/bin/python
"""External Graphs Pipeline Phase.

Phase — Registers external knowledge graph references (SPARQL, LPG) into the local knowledge graph.

CONCEPT:KG-2.1 — External Graph Federation
"""

import logging
import uuid
from typing import Any

from ....models.knowledge_graph import ExternalGraphReferenceNode
from ..types import PhaseResult, PipelineContext, PipelinePhase

logger = logging.getLogger(__name__)


async def execute_external_graphs(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Register external SPARQL and LPG graph references as nodes."""
    if not ctx.config.enable_external_graphs:
        return {"status": "skipped", "reason": "External graphs disabled"}

    nodes_added = 0

    # Register SPARQL endpoints
    for endpoint in ctx.config.external_sparql_endpoints:
        node_id = f"sparql_{uuid.uuid5(uuid.NAMESPACE_URL, endpoint).hex}"
        if not ctx.nx_graph.has_node(node_id):
            node = ExternalGraphReferenceNode(
                id=node_id,
                name=f"SPARQL Endpoint: {endpoint}",
                endpoint_url=endpoint,
                graph_type="sparql",
            )
            ctx.nx_graph.add_node(node_id, **node.model_dump())
            nodes_added += 1

    # Register LPG endpoints
    for name, endpoint in ctx.config.external_lpg_endpoints.items():
        node_id = f"lpg_{uuid.uuid5(uuid.NAMESPACE_URL, endpoint).hex}"
        if not ctx.nx_graph.has_node(node_id):
            node = ExternalGraphReferenceNode(
                id=node_id,
                name=f"LPG Endpoint: {name}",
                endpoint_url=endpoint,
                graph_type="lpg",
                properties={"lpg_name": name},
            )
            ctx.nx_graph.add_node(node_id, **node.model_dump())
            nodes_added += 1

    return {
        "status": "completed",
        "nodes_added": nodes_added,
    }


external_graphs_phase = PipelinePhase(
    name="external_graphs",
    deps=["sync"],
    execute_fn=execute_external_graphs,
)
