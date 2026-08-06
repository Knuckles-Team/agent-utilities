"""``_ControlPlaneWorkItemEngine.add_node`` must not diverge from
``IntelligenceGraphEngine.add_node``'s ``node_type`` casing.

CONCEPT:AU-KG.ontology.node-type-casing-convergence — live measurement
(2026-08-06) found ``WorkItem`` (PascalCase, 4,590 nodes) split against
``work_item`` (lowercase, 3,760 nodes) as two DIFFERENT ``node_type``
property values for the same logical class. Root cause: ``work_item.py``'s
single ``_authority(engine).add_node(item_id, "WorkItem", properties=node.
to_graph_properties())`` call site resolves to one of two adapters depending
on which graph a WorkItem is scheduled on. ``IntelligenceGraphEngine.
add_node`` (main graph) always stamps ``props["node_type"] = node_type`` (the
label parameter, ``"WorkItem"``), but ``_ControlPlaneWorkItemEngine.add_node``
(control-plane graph, used for ingestion-scheduled WorkItems) used to spread
``properties`` unreconciled — and ``RegistryNode.to_graph_properties()``
already writes its OWN ``node_type`` key (the lowercase ``RegistryNodeType``
enum value, e.g. ``"work_item"``), which silently won on this path only.

These tests pin the fix: the ``node_type`` PARAMETER (matching the schema
label every ``work_item.py`` Cypher query filters on) is the single source of
truth, converged on both adapters.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.core.engine_tasks import (
    _ControlPlaneWorkItemEngine,
)


class _FakeControlBackend:
    def __init__(self) -> None:
        self.add_node_calls: list[tuple[str, dict[str, Any]]] = []

    def add_node(self, node_id: str, **kwargs: Any) -> None:
        self.add_node_calls.append((node_id, kwargs))


class _FakeHost:
    def __init__(self, control: _FakeControlBackend) -> None:
        self._control = control


def test_a_caller_embedded_lowercase_node_type_does_not_survive_the_control_path():
    """The exact reproduction: ``properties`` carrying a lowercase
    ``RegistryNode.to_graph_properties()``-style ``node_type`` must not win
    over the adapter's own ``node_type`` (label) parameter."""
    control = _FakeControlBackend()
    engine = _ControlPlaneWorkItemEngine(_FakeHost(control))

    engine.add_node(
        "workitem:ingest_task:job-1",
        "WorkItem",
        properties={"node_type": "work_item", "kind": "ingest_task"},
    )

    node_id, kwargs = control.add_node_calls[0]
    assert node_id == "workitem:ingest_task:job-1"
    assert kwargs["label"] == "WorkItem"
    assert kwargs["node_type"] == "WorkItem", (
        "the control-plane adapter must converge on the SAME node_type "
        "casing IntelligenceGraphEngine.add_node produces, not whatever a "
        "RegistryNode subclass happened to fold into its own properties"
    )
    assert kwargs["kind"] == "ingest_task"  # every other property passes through


def test_properties_with_no_node_type_key_still_get_one():
    control = _FakeControlBackend()
    engine = _ControlPlaneWorkItemEngine(_FakeHost(control))

    engine.add_node("workitem:x", "WorkItem", properties={"status": "ready"})

    _node_id, kwargs = control.add_node_calls[0]
    assert kwargs["node_type"] == "WorkItem"
    assert kwargs["status"] == "ready"
