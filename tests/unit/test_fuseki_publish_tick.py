"""Fuseki ontology-publish daemon tick (CONCEPT:AU-KG.ontology.authoritative-tbox).

The bundled ontology modules are pushed to an optional Apache Jena Fuseki
triplestore by a maintenance-scheduler tick gated on ``KG_FUSEKI_PUBLISH``
(default OFF — Fuseki is optional infrastructure). These tests cover the
tick registration gating, the publisher invocation with an injected
publisher, and the real Fuseki transport against a mocked ``requests.put``.

@pytest.mark.concept("AU-KG.ontology.authoritative-tbox")
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.ontology_publisher import (
    OntologyPublisher,
    collect_bundled_ontology_graph,
    publish_ontology_to_fuseki,
)

pytestmark = pytest.mark.concept("AU-KG.ontology.authoritative-tbox")


def _maint_specs():
    """Register maintenance :Schedule nodes for the current config (OS-5.44)."""
    from agent_utilities.core import schedule_engine as _se
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin

    inst = TaskManagerMixin.__new__(TaskManagerMixin)  # type: ignore[type-abstract]
    inst.backend = EpistemicGraphBackend()
    inst._register_maintenance_schedules()
    return {s.name: s for s in _se._load_all(inst)}


class TestDaemonRegistration:
    def _jobs(self, monkeypatch, enabled: bool):
        from agent_utilities.core.config import config as cfg

        monkeypatch.setattr(cfg, "kg_fuseki_publish", enabled)
        return {n for n, s in _maint_specs().items() if s.enabled}

    def test_flag_on_registers_tick(self, monkeypatch):
        assert "fuseki_publish" in self._jobs(monkeypatch, True)

    def test_flag_off_unregisters_tick(self, monkeypatch):
        assert "fuseki_publish" not in self._jobs(monkeypatch, False)

    def test_default_flag_is_off(self, monkeypatch):
        from agent_utilities.core.config import AgentConfig

        # Isolate from the deployment's semantic-plane wiring: a configured
        # Fuseki/Jena endpoint auto-enables publish (KG-2.52), so clear both
        # to assert the genuine field default with no endpoint present.
        monkeypatch.delenv("KG_FUSEKI_ENDPOINT", raising=False)
        monkeypatch.delenv("JENA_FUSEKI_URL", raising=False)
        assert AgentConfig().kg_fuseki_publish is False

    def test_tick_interval_comes_from_config(self, monkeypatch):
        from agent_utilities.core.config import config as cfg

        monkeypatch.setattr(cfg, "kg_fuseki_publish", True)
        monkeypatch.setattr(cfg, "kg_fuseki_publish_interval", 1234.0)
        spec = _maint_specs()["fuseki_publish"]
        assert spec.interval_s == 1234.0 and spec.payload["ref"] == "fuseki_publish"


class TestBundledOntologyCollection:
    def test_collects_all_bundled_modules(self):
        pytest.importorskip("rdflib")
        graph = collect_bundled_ontology_graph()
        # The platform ships ~33 ontology modules; the merged TBox is large.
        assert len(graph) > 100

    def test_contains_orchestration_workflow_class(self):
        rdflib = pytest.importorskip("rdflib")
        graph = collect_bundled_ontology_graph()
        kg = rdflib.Namespace("http://knuckles.team/kg#")
        owl = rdflib.Namespace("http://www.w3.org/2002/07/owl#")
        assert (kg.Workflow, rdflib.RDF.type, owl.Class) in graph


class _RecordingPublisher:
    """Publisher stub recording the push call (the mocked transport seam)."""

    def __init__(self):
        self.calls: list[dict] = []

    def push_to_jena_fuseki(
        self, rdf_graph, endpoint=None, dataset="agent_kg", named_graph=None
    ):
        self.calls.append(
            {
                "triples": len(rdf_graph),
                "endpoint": endpoint,
                "dataset": dataset,
                "named_graph": named_graph,
            }
        )
        return {"status": "success", "triple_count": len(rdf_graph)}


class TestPublishInvocation:
    def test_publisher_invoked_with_merged_graph_and_endpoint(self):
        pytest.importorskip("rdflib")
        pub = _RecordingPublisher()
        report = publish_ontology_to_fuseki(
            endpoint="http://fuseki.test:3030", publisher=pub
        )
        assert report["status"] == "success"
        (call,) = pub.calls
        assert call["endpoint"] == "http://fuseki.test:3030"
        assert call["dataset"] == "agent_kg"
        assert call["triples"] > 100

    def test_tick_routes_config_endpoint_to_publisher(self, monkeypatch):
        pytest.importorskip("rdflib")
        from agent_utilities.core.config import config as cfg
        from agent_utilities.knowledge_graph.core import (
            engine_tasks,
            ontology_publisher,
        )
        from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin

        monkeypatch.setattr(cfg, "kg_fuseki_endpoint", "http://fuseki.cfg:3030")
        seen = {}

        def _fake_publish(endpoint=None, **kwargs):
            seen["endpoint"] = endpoint
            return {"status": "success", "triple_count": 1}

        monkeypatch.setattr(
            ontology_publisher, "publish_ontology_to_fuseki", _fake_publish
        )
        inst = object.__new__(TaskManagerMixin)
        engine_tasks.TaskManagerMixin._tick_fuseki_publish(inst)
        assert seen["endpoint"] == "http://fuseki.cfg:3030"

    def test_fuseki_transport_put_with_mocked_requests(self, monkeypatch):
        """The real ``push_to_jena_fuseki`` issues a Graph Store Protocol PUT."""
        rdflib = pytest.importorskip("rdflib")
        import requests

        calls = {}

        class _Resp:
            def raise_for_status(self):
                return None

        def _fake_put(url, data=None, params=None, headers=None, timeout=None):
            calls.update(
                {"url": url, "params": params, "headers": headers, "data": data}
            )
            return _Resp()

        monkeypatch.setattr(requests, "put", _fake_put)
        graph = rdflib.Graph()
        graph.add(
            (
                rdflib.URIRef("http://knuckles.team/kg#A"),
                rdflib.RDFS.label,
                rdflib.Literal("a"),
            )
        )
        report = OntologyPublisher().push_to_jena_fuseki(
            graph, endpoint="http://fuseki.test:3030", dataset="agent_kg"
        )
        assert report["status"] == "success"
        assert calls["url"] == "http://fuseki.test:3030/agent_kg/data"
        assert calls["headers"]["Content-Type"] == "text/turtle"
        assert b"knuckles.team" in calls["data"]
