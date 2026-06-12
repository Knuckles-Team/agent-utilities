"""Periodic tenant GC for leaked community-detection tenants (CONCEPT:KG-2.8).

The per-job `{graph}__enrich_comm_{uuid}` tenant leaks when a process is killed
mid-ingest, and every leak is re-serialized on each checkpoint. `_tick_tenant_gc`
sweeps them when no bulk ingest is in flight, touching ONLY the comm pattern.
"""

from __future__ import annotations

import types

from agent_utilities.core.background_throttle import get_throttle
from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin


class _FakeTenants:
    def __init__(self, names):
        self._names = list(names)
        self.deleted = []

    def list(self):
        return [{"name": n, "type": "Agent"} for n in self._names]

    def delete(self, name):
        self.deleted.append(name)
        self._names.remove(name)


def _stub(names):
    tenants = _FakeTenants(names)
    client = types.SimpleNamespace(tenants=tenants)
    graph = types.SimpleNamespace(_client=client)
    backend = types.SimpleNamespace(graph=graph)
    return types.SimpleNamespace(backend=backend), tenants


def test_gc_drops_only_comm_tenants_when_idle():
    get_throttle().set_bulk_ingest(False)
    stub, tenants = _stub(
        [
            "__commons__",
            "agent:0",
            "x__enrich_comm_ab12",
            "y__enrich_comm_cd34",
            "test_deadbeef",
        ]
    )
    TaskManagerMixin._tick_tenant_gc(stub)
    assert sorted(tenants.deleted) == ["x__enrich_comm_ab12", "y__enrich_comm_cd34"]
    # real graphs untouched
    assert "__commons__" in tenants._names and "agent:0" in tenants._names
    assert "test_deadbeef" in tenants._names


def test_gc_skips_while_bulk_ingest_active():
    get_throttle().set_bulk_ingest(True)
    try:
        stub, tenants = _stub(["__commons__", "z__enrich_comm_ff00"])
        TaskManagerMixin._tick_tenant_gc(stub)
        assert tenants.deleted == []  # an active ingest may own its comm tenant
    finally:
        get_throttle().set_bulk_ingest(False)


def test_gc_tolerates_missing_client():
    get_throttle().set_bulk_ingest(False)
    stub = types.SimpleNamespace(backend=types.SimpleNamespace(graph=None))
    TaskManagerMixin._tick_tenant_gc(stub)  # must not raise
