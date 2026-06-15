"""Phase 3 — ArchiMate bidirectional + Egeria sink (CONCEPT:KG-2.9)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors import archimate as ar_ext
from agent_utilities.knowledge_graph.enrichment.writeback import core, run_writeback


class FakeArchi:
    def __init__(self):
        self.elements = []
        self.rels = []

    def list_elements(self):
        return [
            {
                "id": "e1",
                "type": "BusinessProcess",
                "name": "Invoice",
                "documentation": "",
            },
            {"id": "e2", "type": "ApplicationComponent", "name": "Billing App"},
        ]

    def list_relationships(self):
        return [{"id": "r1", "type": "Realization", "source": "e2", "target": "e1"}]

    def add_element(self, type, name, documentation=""):
        self.elements.append((type, name))

    def add_relationship(self, source, target, type):
        self.rels.append((source, target, type))


def test_archimate_extract():
    batch = ar_ext.extract({"client": FakeArchi()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["archi:e1"].type == "BusinessProcess"
    assert by_id["archi:e2"].type == "ApplicationComponent"
    assert by_id["archi:e1"].props["domain"] == "archimate"
    triples = {(e.source, e.target, e.rel_type) for e in batch.edges}
    assert ("archi:e2", "archi:e1", "REALIZATION") in triples
    # element classes registered promotable
    from agent_utilities.knowledge_graph.core import owl_bridge

    assert "businessprocess" in owl_bridge.DYNAMIC_PROMOTABLE_NODE_TYPES


def test_archimate_sink_add_element_and_relation(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeArchi()

    class _Backend:
        def execute(self, q, p=None):
            if "externalToolId" in q:
                return [{"id": "x1", "guid": "e1"}, {"id": "x2", "guid": "e2"}]
            return []

    out = run_writeback(
        "archimate",
        backend=_Backend(),
        client=client,
        creations=[{"type": "Node", "name": "DB Server"}],
        inferences=[{"source": "x2", "rel_type": "REALIZES", "target": "x1"}],
        dry_run=False,
    )
    assert out["created"] == 1 and out["relations_written"] == 1
    assert client.elements == [("Node", "DB Server")]
    assert client.rels == [("e2", "e1", "Realization")]


def test_archimate_dry_run(monkeypatch):
    out = run_writeback(
        "archimate",
        client=FakeArchi(),
        creations=[{"type": "Node", "name": "x"}],
        dry_run=True,
    )
    assert out["proposals"][0]["op"] == "add_element"


class FakeEgeria:
    def __init__(self):
        self.assets = []

    def create_asset(self, type_name, qualified_name, display_name, **kw):
        self.assets.append((type_name, display_name))
        return {"guid": "g1"}


def test_egeria_sink_create_asset(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeEgeria()
    out = run_writeback(
        "egeria",
        client=client,
        creations=[{"type": "service", "name": "billing-svc"}],
        dry_run=False,
    )
    assert out["created"] == 1
    assert client.assets == [("SoftwareServer", "billing-svc")]


def test_egeria_refused(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: False)
    out = run_writeback(
        "egeria",
        client=FakeEgeria(),
        creations=[{"type": "service", "name": "x"}],
        dry_run=False,
    )
    assert out["status"] == "refused"
