"""OntologyLifecycle.delete → physical engine retract (CONCEPT:AU-KG.ingest.mirror-inbound).

KG-2.265 unloaded an ontology by dropping the registry record only; KG-2.266 wires it
to the engine's ``remove_triples`` retract op so the axioms physically leave the
engine RDF dataset. These tests attach a fake engine whose ``graph_compute`` records
``remove_triples`` calls.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ontology.lifecycle import (
    OntologyLifecycle,
    reset_registry,
)

pytestmark = pytest.mark.concept("AU-KG.ingest.mirror-inbound")

PETS_TTL = """
@prefix : <http://example.org/pets#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<http://example.org/pets> a owl:Ontology .
:Dog a owl:Class .
:Cat a owl:Class .
:Dog rdfs:subClassOf :Animal .
"""


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


class _FakeGraphCompute:
    def __init__(self):
        self.added = []
        self.removed = []

    def add_triples(self, turtle=None, ntriples=None):
        self.added.append(turtle)
        return {"triples": 3}

    def remove_triples(self, turtle=None, ntriples=None):
        self.removed.append(turtle)
        return {"removed": 3}


class _FakeEngine:
    def __init__(self):
        self.graph_compute = _FakeGraphCompute()


def test_delete_retracts_axioms_from_engine():
    engine = _FakeEngine()
    lc = OntologyLifecycle(engine=engine)
    lc.load(PETS_TTL, source_type="text")

    res = lc.delete("http://example.org/pets")
    assert res["status"] == "ok"
    assert res["axioms_retracted_from_engine"] is True
    # the engine actually received a remove_triples call with the stored turtle
    assert engine.graph_compute.removed, "remove_triples was never called"
    assert "Dog" in engine.graph_compute.removed[0]
    assert res["retractions"][0]["retracted_from_engine"] is True
    assert "retracted" in res["engine_note"]


def test_delete_reports_gap_when_retract_unavailable():
    class _NoRetract:
        graph_compute = type("GC", (), {"add_triples": lambda self, **k: {}})()

    lc = OntologyLifecycle(engine=_NoRetract())
    lc.load(PETS_TTL, source_type="text")
    res = lc.delete("http://example.org/pets")
    assert res["status"] == "ok"
    assert res["axioms_retracted_from_engine"] is False
    assert "no engine retract surface" in res["engine_note"]


def test_delete_no_engine_is_registry_only():
    lc = OntologyLifecycle(engine=None)
    lc.load(PETS_TTL, source_type="text")
    res = lc.delete("http://example.org/pets")
    assert res["axioms_retracted_from_engine"] is False
    assert res["engine_note"] == "no engine attached"


# ── CONCEPT:AU-KG.ontology.activation-fails-closed — a live engine's
# activation failure must NOT be reported as a successful activation ──


class _FailingGraphCompute:
    """A live engine attached, but ``add_triples`` always rejects the candidate
    (e.g. the engine's SHACL/ICV write guard rejecting malformed identifiers)."""

    def add_triples(self, turtle=None, ntriples=None):
        raise RuntimeError("engine rejected candidate: SHACL/ICV violation")


class _FailingEngine:
    def __init__(self):
        self.graph_compute = _FailingGraphCompute()


def test_load_reports_inactive_when_engine_activation_fails():
    lc = OntologyLifecycle(engine=_FailingEngine())
    result = lc.load(PETS_TTL, source_type="text")
    assert result["status"] == "ok"  # parse/validate succeeded
    onto = result["ontology"]
    # A live engine IS attached and DID reject the axioms -- "active" must
    # honestly reflect that, not the mere fact that activation was requested.
    assert onto["active"] is False
    assert onto["engine"]["loaded_to_engine"] is False
    assert onto["engine"]["engine_attached"] is True
    assert "SHACL/ICV" in onto["engine"]["reason"]


def test_set_active_reports_inactive_when_engine_activation_fails():
    lc = OntologyLifecycle(engine=_FailingEngine())
    # load() with activate=False so the initial record is inert, then flip it
    # on explicitly via set_active() to exercise that path's own fail-closed check.
    lc.load(PETS_TTL, source_type="text", activate=False)
    res = lc.set_active("http://example.org/pets", active=True)
    assert res["ontology"]["active"] is False
    assert res["ontology"]["engine"]["loaded_to_engine"] is False


def test_load_no_engine_still_reports_active_intent():
    """The pre-existing offline contract (no engine attached at all) is
    unchanged: 'active' stays the requested intent flag since there is
    nothing an engine could have rejected."""
    lc = OntologyLifecycle(engine=None)
    result = lc.load(PETS_TTL, source_type="text")
    assert result["ontology"]["active"] is True


# ── CONCEPT:AU-KG.ontology.dedicated-tbox-graph — a real engine's hosted-
# ontology registry is durable + per-tenant-graph-scoped, not a process dict ──


class _FakeNodesSubClient:
    """Stand-in for the eg client's ``.nodes`` sub-client (only ``.properties``
    is used by :class:`_EngineRegistryStore`)."""

    def __init__(self, node_data: dict[str, dict]):
        self._node_data = node_data

    def properties(self, node_id):
        return self._node_data.get(node_id)


class _FakeTenantsSubClient:
    """Stand-in for the eg client's ``.tenants`` sub-client."""

    def __init__(self, created_graphs: set[str]):
        self._created_graphs = created_graphs

    def list(self):
        return [{"name": g} for g in self._created_graphs]

    def create(self, name, graph_type="Global"):
        self._created_graphs.add(name)


class _FakeClient:
    def __init__(self, node_data: dict[str, dict], created_graphs: set[str]):
        self.nodes = _FakeNodesSubClient(node_data)
        self.tenants = _FakeTenantsSubClient(created_graphs)


class _FakeNodeStore:
    """A minimal in-memory stand-in for the engine's native typed-node surface
    (add_node/get_nodes_by_label/has_node/remove_node/client.nodes.properties),
    shared externally across separate OntologyLifecycle() instances -- exactly
    like a real durable engine backing store would be, so this proves the
    registry survives fresh-instance construction (every MCP tool call builds
    a NEW OntologyLifecycle) without relying on any Python-process-local state.
    """

    def __init__(self):
        self._node_data: dict[str, dict] = {}
        self.created_graphs: set[str] = set()
        self.client = _FakeClient(self._node_data, self.created_graphs)

    def add_node(self, node_id, node_type=None, **props):
        self._node_data[node_id] = {"node_type": node_type, **props}

    def has_node(self, node_id):
        return node_id in self._node_data

    def remove_node(self, node_id):
        self._node_data.pop(node_id, None)

    def get_nodes_by_label(self, label, limit=0):
        return [
            (nid, p)
            for nid, p in self._node_data.items()
            if p.get("node_type") == label
        ]

    def add_triples(self, turtle=None, ntriples=None):
        return {"triples": 3}

    def remove_triples(self, turtle=None, ntriples=None):
        return {"removed": 3}


class _DurableFakeEngine:
    """``graph_compute`` exposes the full native surface AND ``for_graph`` (a
    no-op returning self, since this fake has only one backing store) so
    ``OntologyLifecycle`` selects the durable engine-registry path, not the
    in-memory fallback."""

    def __init__(self, store: _FakeNodeStore):
        self._store = store
        self.graph_compute = self

    def for_graph(self, graph_name):
        return self

    def __getattr__(self, name):
        # Proxy add_node/has_node/remove_node/get_nodes_by_label/add_triples/
        # remove_triples straight through to the shared backing store.
        return getattr(self._store, name)

    @property
    def client(self):
        return self._store.client


def test_durable_registry_survives_fresh_instance_construction():
    store = _FakeNodeStore()

    lc1 = OntologyLifecycle(engine=_DurableFakeEngine(store))
    lc1.load(PETS_TTL, source_type="text")

    # A brand-new OntologyLifecycle over the SAME backing store (exactly what
    # every separate MCP `graph_ontology` tool call constructs) must still see
    # the record -- proving it isn't held in a Python-process-local dict.
    lc2 = OntologyLifecycle(engine=_DurableFakeEngine(store))
    listed = lc2.list_ontologies()
    assert listed["count"] == 1
    assert listed["ontologies"][0]["iri"] == "http://example.org/pets"

    # The dedicated ontology graph was actually provisioned on the engine.
    assert store.created_graphs, (
        "the dedicated per-tenant ontology graph was never created"
    )
