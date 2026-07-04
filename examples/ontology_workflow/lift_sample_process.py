#!/usr/bin/env python3
"""Lift examples/ontology_workflow/sample_process.bpmn into the KG (KG-2.53).

Runs the Camunda source extractor against the local BPMN fixture with an
injected XML-capable client (the same duck-typed surface camunda-mcp exposes:
``list_process_definitions`` + ``get_process_definition_xml``) and writes the
resulting BusinessProcess/BusinessTask/FLOWS_TO subgraph through the engine.

Usage (from the repository root, against a live engine):

    PYTHONPATH=. python3 examples/ontology_workflow/lift_sample_process.py

In a Camunda deployment this step is the harvest itself — the extractor is fed
the real camunda-mcp client instead of this file-backed one, and the lift is
identical (CONCEPT:AU-KG.ontology.descriptive-process-world-gains).
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_utilities.knowledge_graph.enrichment.extractors.camunda import extract
from agent_utilities.knowledge_graph.enrichment.registry import write_batch

FIXTURE = Path(__file__).parent / "sample_process.bpmn"


class FileBackedCamundaClient:
    """Duck-typed Camunda client serving the local BPMN fixture."""

    def list_process_definitions(self):
        return [
            {
                "id": "order_fulfillment:1:demo",
                "key": "order_fulfillment",
                "name": "Order Fulfillment",
            }
        ]

    def get_process_definition_xml(self, id=None):
        # Camunda 7 REST envelope shape ({"id": ..., "bpmn20Xml": "<xml...>"}).
        return {"id": id, "bpmn20Xml": FIXTURE.read_text(encoding="utf-8")}


class EngineWriter:
    """write_batch adapter over the IntelligenceGraphEngine node/edge API."""

    def __init__(self, engine):
        self.engine = engine

    def add_node(self, node_id, **props):
        node_type = props.pop("type", "Thing")
        self.engine.add_node(node_id, node_type, properties=props)

    def add_edge(self, src, tgt, **props):
        rel = props.pop("rel_type", "RELATES_TO")
        self.engine.link_nodes(src, tgt, rel, properties=props or None)


def main() -> None:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    batch = extract({"client": FileBackedCamundaClient()})
    engine = IntelligenceGraphEngine()
    nodes, edges = write_batch(EngineWriter(engine), batch)
    print(
        json.dumps(
            {
                "category": batch.category,
                "nodes_written": nodes,
                "edges_written": edges,
                "process_id": "bpmn_process:order_fulfillment:1:demo",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
