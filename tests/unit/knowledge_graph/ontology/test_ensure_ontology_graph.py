"""U-96: `_ensure_ontology_graph` must provision the dedicated per-tenant
ontology graph as the engine's supported `Global` lifecycle type, not the
semantic content label `"Ontology"`.

`"Ontology"` is not a member of the engine's closed `GraphType` wire enum
(`crates/eg-types/src/protocol.rs`: `Agent | Team | Global | Commons`; its own
canonical ontology fixture creates `global:ontology` as `GraphType::Global`).
Sending it used to fail server-side deserialization, which surfaced as a
multi-minute connection timeout instead of an immediate error (the caller
fell back to non-durable process-local ontology state) -- fixed separately in
epistemic-graph's transport/decode-error path and in that client's
`tenants.create` allowlist. This test pins the ONE place in agent-utilities
that must never regress back to the unsupported value.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.ontology import lifecycle


class _RecordingTenants:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._graphs: set[str] = set()

    def list(self):
        return [{"name": g} for g in self._graphs]

    def create(self, name: str, graph_type: str) -> None:
        self.calls.append((name, graph_type))
        self._graphs.add(name)


class _FakeClient:
    def __init__(self, tenants: _RecordingTenants) -> None:
        self.tenants = tenants


class _FakeGraphComputeEngine:
    def __init__(self, client: _FakeClient) -> None:
        self.client = client


def setup_function() -> None:
    lifecycle.reset_registry()


def teardown_function() -> None:
    lifecycle.reset_registry()


def test_ensure_ontology_graph_provisions_the_supported_global_type():
    tenants = _RecordingTenants()
    gc = _FakeGraphComputeEngine(_FakeClient(tenants))

    lifecycle._ensure_ontology_graph(gc, "tenant__local__ontology")

    assert tenants.calls == [("tenant__local__ontology", "Global")]
    assert "tenant__local__ontology" in lifecycle._KNOWN_ONTOLOGY_GRAPHS


def test_ensure_ontology_graph_is_a_noop_when_already_listed():
    tenants = _RecordingTenants()
    tenants._graphs.add("tenant__local__ontology")
    gc = _FakeGraphComputeEngine(_FakeClient(tenants))

    lifecycle._ensure_ontology_graph(gc, "tenant__local__ontology")

    # Already present -> no create call at all (not even with the right type).
    assert tenants.calls == []
