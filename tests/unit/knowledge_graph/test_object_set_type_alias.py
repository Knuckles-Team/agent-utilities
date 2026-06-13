"""Type-filtered object sets must resolve the canonical label the ingestion path
writes (``node_type``), not just ``type``/``label``.

Regression: ``graph_ingest agent_toolkit`` writes tool nodes with
``node_type="CallableResource"`` (via ``add_node(node_type=...)``) but left
``type``/``label`` unset, so ``object_set of_type('CallableResource')`` resolved
zero objects even though ``graph_query`` counted hundreds. The object-type
accessor now includes ``node_type`` in its alias chain. (CONCEPT:KG-2.45)
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.ontology.object_set import _prop


def test_prop_resolves_node_type_alias():
    # The exact shape the agent_toolkit ingestion writes.
    props = {"node_type": "CallableResource", "id": "res:egeria_harvest"}
    assert _prop(props, "type") == "CallableResource"


def test_prop_type_alias_precedence_unchanged():
    # Explicit ``type``/``_type``/``label`` still win ahead of ``node_type``.
    assert _prop({"type": "A", "node_type": "B"}, "type") == "A"
    assert _prop({"_type": "A", "node_type": "B"}, "type") == "A"


def test_prop_non_type_field_passthrough():
    assert _prop({"node_type": "X", "server": "egeria"}, "server") == "egeria"
    assert _prop({"node_type": "X"}, "missing") is None
