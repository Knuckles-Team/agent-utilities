"""Stardog ontology catalog: overwrite-on-push + import-back-into-engine (WS3).

CONCEPT:AU-KG.ontology.stardog-catalog-overwrite / AU-KG.ontology.stardog-catalog-import — re-publishing an
updated ontology REPLACES the catalog slice (clear-then-add), and the ontologies already in
Stardog can be consumed back into epistemic-graph.
"""

from __future__ import annotations

import sys
import types

from agent_utilities.knowledge_graph.core.ontology_publisher import (
    OntologyPublisher,
    import_ontology_from_stardog,
)


class _FakeConn:
    def __init__(self, log):
        self._log = log

    def begin(self):
        self._log.append(("begin",))

    def clear(self, graph_uri=None):
        self._log.append(("clear", graph_uri))

    def update(self, sparql):
        self._log.append(("update", sparql))

    def add(self, content, graph_uri=None):
        self._log.append(("add", graph_uri))

    def commit(self):
        self._log.append(("commit",))

    def rollback(self):
        self._log.append(("rollback",))

    def close(self):
        self._log.append(("close",))

    def export(self, content_type=None, graph_uri=None):
        self._log.append(("export", graph_uri))
        return b"@prefix ex: <urn:ex:> . ex:A a ex:Class ."


def _install_fake_stardog(monkeypatch, log):
    mod = types.ModuleType("stardog")
    mod.Connection = lambda db, **kw: _FakeConn(log)  # noqa: ARG005
    content = types.SimpleNamespace(Raw=lambda data, content_type=None: ("raw", data))
    mod.content = content
    monkeypatch.setitem(sys.modules, "stardog", mod)


class _FakeGraph:
    def serialize(self, format="turtle"):  # noqa: A002
        return "@prefix ex: <urn:ex:> . ex:A a ex:Class ."

    def __len__(self):
        return 1


def test_push_overwrite_clears_named_graph_before_add(monkeypatch):
    log: list = []
    _install_fake_stardog(monkeypatch, log)
    res = OntologyPublisher().push_to_stardog(
        _FakeGraph(),
        named_graph="urn:source:ontology",
        overwrite=True,
    )
    assert res["status"] == "success"
    ops = [op[0] for op in log]
    # clear happens before add, inside the begin/commit txn.
    assert ops.index("clear") < ops.index("add")
    assert ("clear", "urn:source:ontology") in log
    assert ops[0] == "begin" and "commit" in ops


def test_push_without_overwrite_does_not_clear(monkeypatch):
    log: list = []
    _install_fake_stardog(monkeypatch, log)
    OntologyPublisher().push_to_stardog(_FakeGraph(), named_graph="g")
    assert "clear" not in [op[0] for op in log]
    assert "update" not in [op[0] for op in log]


def test_import_pulls_turtle_and_loads_into_engine(monkeypatch):
    log: list = []
    _install_fake_stardog(monkeypatch, log)

    loaded = {}

    class _FakeLifecycle:
        def __init__(self, engine):
            loaded["engine"] = engine

        def load(self, source, *, source_type, activate, force):
            loaded["source_type"] = source_type
            loaded["force"] = force
            return {"status": "ok", "activated": activate}

    import agent_utilities.knowledge_graph.ontology.lifecycle as lc

    monkeypatch.setattr(lc, "OntologyLifecycle", _FakeLifecycle)

    engine = object()
    res = import_ontology_from_stardog(named_graph="urn:source:ontology", engine=engine)
    assert res["status"] == "success"
    assert ("export", "urn:source:ontology") in log
    assert res["load"] == {"status": "ok", "activated": True}
    assert loaded["source_type"] == "turtle" and loaded["force"] is True
    assert loaded["engine"] is engine


def test_import_without_engine_returns_turtle(monkeypatch):
    log: list = []
    _install_fake_stardog(monkeypatch, log)
    res = import_ontology_from_stardog()
    assert res["status"] == "success"
    assert "ex:Class" in res["turtle"]
    assert "load" not in res
