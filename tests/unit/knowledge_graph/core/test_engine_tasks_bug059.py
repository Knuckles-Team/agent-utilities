"""BUG-059 — _ControlPlaneWorkItemEngine is a JUSTIFIED chokepoint bypass,
evaluated and pinned (per BUG-LEDGER: "engine_tasks.py's is already
documented and may well stand as-is -- evaluate it, don't assume either
way").

``add_node``/``link_nodes`` write straight through ``self._host._control``
(a raw backend), never through ``IntelligenceGraphEngine._upsert_node``/
``GraphComputeEngine.add_node``, so they never reach ``stamp_ownership``.
Confirmed as deliberate, not an oversight: routing through the engine
wrapper is the ONE thing this adapter class exists to avoid (it would put
ingestion-scheduling WorkItem writes back on the ``__commons__`` graph's
write lock -- the class's own docstring). WorkItems are control-plane
scheduling bookkeeping, not owned content, and are minted from ingestion
scheduling code with no per-request actor to bind. This test pins that the
adapter keeps working with zero actor bound, so nobody "fixes" it into
routing through ``engine.add_node`` by accident and reintroduces the lock
contention the class exists to avoid.
"""

from __future__ import annotations

import contextvars

from agent_utilities.knowledge_graph.core.engine_tasks import (
    _ControlPlaneWorkItemEngine,
)


class _FakeControlBackend:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []

    def add_node(self, node_id, label="", **props):
        self.nodes[node_id] = {"label": label, **props}

    def add_edge(self, source, target, rel_type="", **props):
        self.edges.append((source, target, rel_type))


class _FakeHost:
    def __init__(self, control: _FakeControlBackend) -> None:
        self._control = control


def test_control_plane_work_item_add_node_works_with_no_actor_bound():
    control = _FakeControlBackend()
    engine = _ControlPlaneWorkItemEngine(_FakeHost(control))

    def isolated():
        # No actor bound anywhere in this fresh context — must not raise.
        engine.add_node("wi:1", "WorkItem", {"status": "pending"})

    contextvars.Context().run(isolated)

    props = control.nodes["wi:1"]
    assert props["node_type"] == "WorkItem"
    assert props["status"] == "pending"
    # No governance stamp was applied — this is the pinned, deliberate gap.
    assert "_owner_id" not in props
    assert "classification" not in props


def test_control_plane_work_item_link_nodes_works_with_no_actor_bound():
    control = _FakeControlBackend()
    engine = _ControlPlaneWorkItemEngine(_FakeHost(control))

    def isolated():
        engine.link_nodes("wi:1", "wi:2", "DEPENDS_ON")

    contextvars.Context().run(isolated)

    assert control.edges == [("wi:1", "wi:2", "DEPENDS_ON")]
