"""Live end-to-end test of the Camunda inbound path (CONCEPT:AU-KG.ingest.enterprise-source-extractor/KG-2.53).

Runs the real camunda-mcp client against a live Camunda 7 Engine REST endpoint
and asserts the extractor lifts BusinessProcess/BusinessTask/FLOWS_TO structure
from real BPMN XML. Deselected by default (``-m "not live"``); enable by
deploying a Camunda 7 engine with at least one process definition and pointing
``CAMUNDA7_URL`` at it, e.g.::

    CAMUNDA7_URL=http://camunda.arpa/engine-rest pytest -m live \\
        tests/integration/test_camunda_live.py

Then ``graph_ingest(action='materialize_source', corpus_name='camunda')``
persists the same batch into the KG and runs one reasoning cycle — verified
manually against the deployed engine.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live

CAMUNDA_URL = os.getenv("CAMUNDA7_URL")


@pytest.mark.skipif(
    not CAMUNDA_URL, reason="set CAMUNDA7_URL to a live Camunda 7 engine"
)
def test_camunda_extractor_lifts_live_bpmn():
    pytest.importorskip("camunda_mcp")
    from camunda_mcp.auth import get_client

    from agent_utilities.knowledge_graph.enrichment.extractors.camunda import extract
    from agent_utilities.knowledge_graph.enrichment.materialize import (
        materialize_source,
    )

    client = get_client().v7
    # at least one definition must exist on the engine for a meaningful assertion
    definitions = client.list_process_definitions() or []
    if not definitions:
        pytest.skip("live engine has no process definitions deployed")

    batch = extract({"client": client})
    processes = [n for n in batch.nodes if n.type == "BusinessProcess"]
    tasks = [n for n in batch.nodes if n.type == "BusinessTask"]
    flows = [e for e in batch.edges if e.rel_type == "FLOWS_TO"]

    assert processes, "expected at least one BusinessProcess from the live engine"
    # the step-level lift produces typed tasks from the real BPMN XML (runtime
    # user-task nodes legitimately carry no task_type, so require at least one
    # *lifted* task rather than all).
    lifted = [t for t in tasks if t.props.get("task_type")]
    assert lifted, "expected lifted BusinessTask nodes (task_type) from BPMN XML"
    assert flows, "expected FLOWS_TO sequence-flow edges"

    # the same batch persists through the generic writer (None backend = dry no-op)
    n, e = materialize_source(None, "camunda", client)
    assert (n, e) == (0, 0)  # dry run; real run targets engine.backend
