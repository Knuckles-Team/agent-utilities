#!/usr/bin/python
"""Engine-native SPARQL / OWL / SHACL demotion tests (CONCEPT:KG-2.242).

Proves the semantic-web stack routes to the engine's native RDF/SPARQL/OWL surface
(`client.rdf.*`) and that the Python rdflib/owlready2/pyshacl path is a true
last-resort fallback:

* SPARQL via the engine == via rdflib on the same fixture graph.
* OWL reasoning via the engine derives the expected class-membership entailments.
* The SHACL gate still catches a known violation when its data graph is sourced
  from the engine's RDF projection.
* The no-engine fallback (`_sparql_via_rdflib`, `_python_reasoning`) still works.

These engine-mode tests run against the **REAL ephemeral epistemic-graph engine**
the session fixture deploys (CONCEPT:KG-2.238): they request the conftest
``engine_graph`` (a fresh per-test tenant on the one running engine, source-built
so its client carries the ``.rdf`` namespace), seed a tiny graph, and assert the
engine-native semantic surface. There is NO SQLite and no per-module engine spawn;
when no real engine is reachable the conftest hermetic-skip turns the connection
error into a clean skip. The ``rdf``/``sparql``/``owl`` features are part of the
lean ``pi``-tier the fixture builds, so the engine path is exercised on any dev/CI
box that has (or can build) the engine.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge


@pytest.fixture()
def owl_graph(engine_graph):
    """A tiny typed graph on the REAL ephemeral engine tenant (CONCEPT:KG-2.238).

    Layers a small fixture topology (two Agents + one ``knows`` edge) onto the
    fresh per-test tenant ``engine_graph`` yields, so the engine-native
    SPARQL/OWL/SHACL assertions have content to reason over. The conftest fixture
    owns the engine lifecycle + tenant purge — this only seeds nodes.
    """
    engine_graph.add_node("alice", type="Agent", name="Alice")
    engine_graph.add_node("bob", type="Agent", name="Bob")
    engine_graph.add_edge("alice", "bob", type="knows")
    return engine_graph


def _localname(term: str | None) -> str | None:
    """Strip an RDF term to its bare local name so the engine's `au:`-bracketed IRIs
    (`<http://…#alice>`) and rdflib's `str(URIRef)` (`http://…#alice`) compare equal."""
    if term is None:
        return None
    t = term
    if t.startswith("<") and t.endswith(">"):
        t = t[1:-1]
    return t.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _triple_set(rows: list[dict[str, str | None]]) -> set[tuple]:
    """Normalize ``SELECT ?s ?p ?o`` rows to a localname triple set."""
    return {
        (_localname(r.get("s")), _localname(r.get("p")), _localname(r.get("o")))
        for r in rows
    }


def test_sparql_engine_equals_rdflib(owl_graph):
    """query_sparql resolves over the ENGINE'S LPG→RDF projection (engine concept KG-2.240)
    and returns the SAME triples rdflib materializes from the same LPG — proving the
    engine projection now speaks AU's vocabulary, so rdflib leaves the runtime path."""
    pytest.importorskip("rdflib")
    bridge = OWLBridge(graph=owl_graph, owl_backend=None, backend=None)
    query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"

    # query_sparql must answer via the ENGINE, NOT the rdflib fallback: spy that
    # _sparql_via_rdflib is never entered while the engine path runs.
    with patch.object(
        bridge,
        "_sparql_via_rdflib",
        side_effect=AssertionError("rdflib fallback ran despite a live engine"),
    ):
        engine_rows = bridge.query_sparql(query)
    assert engine_rows, "engine SPARQL returned no rows over a non-empty LPG"

    # The rdflib materialization on the SAME graph is the reference.
    rdflib_rows = bridge._sparql_via_rdflib(query)

    eng = _triple_set(engine_rows)
    rdf = _triple_set(rdflib_rows)
    assert eng == rdf, (
        f"engine projection != rdflib materialization\n eng={eng}\n rdf={rdf}"
    )

    # The shared set carries the SYNTHESIZED rdf:type (au:Agent) + the typed edge — the
    # vocabulary the engine raw projection used to miss entirely.
    assert ("alice", "type", "Agent") in eng
    assert ("bob", "type", "Agent") in eng
    assert ("alice", "knows", "bob") in eng


def test_sparql_by_type_via_engine(owl_graph):
    """The by-class query that was the original failure (`?s a au:Agent`) now resolves
    ENGINE-native over the LPG — both typed nodes come back, with NO rdflib fallback."""
    bridge = OWLBridge(graph=owl_graph, owl_backend=None, backend=None)
    query = "SELECT ?s WHERE { ?s a au:Agent }"

    with patch.object(
        bridge,
        "_sparql_via_rdflib",
        side_effect=AssertionError("rdflib fallback ran despite a live engine"),
    ):
        rows = bridge.query_sparql(query)

    subjects = {_localname(r.get("s")) for r in rows}
    assert {"alice", "bob"} <= subjects, f"by-class query missed agents; got {subjects}"


def test_owl_reasoning_engine_derives_entailments(owl_graph):
    """OWL reasoning via the engine derives the expected class memberships."""
    res = owl_graph.owl_reason()
    assert res.get("consistent") is True
    instances = {tuple(p) for p in res.get("instances", []) if len(p) == 2}
    # Both typed nodes are inferred members of Agent.
    assert ("alice", "Agent") in instances
    assert ("bob", "Agent") in instances


def test_bridge_lightweight_reasoning_uses_engine(owl_graph):
    """OWLBridge lightweight reasoning runs engine-native with NO owl backend and
    materializes inferred rdf:type edges back into the graph."""
    bridge = OWLBridge(graph=owl_graph, owl_backend=None, backend=None)
    inferences = bridge._engine_reasoning()
    types = {
        (i["subject"], i["object"]) for i in inferences if i["predicate"] == "rdf:type"
    }
    assert ("alice", "Agent") in types
    assert ("bob", "Agent") in types


def test_shacl_gate_data_graph_from_engine(owl_graph):
    """The SHACL gate sources its data graph from the engine's RDF projection and
    still flags a node missing a required property."""
    pytest.importorskip("pyshacl")
    pytest.importorskip("rdflib")
    from agent_utilities.knowledge_graph.pipeline.phases import shacl_gate

    # Build the data graph via the engine triples path (CONCEPT:KG-2.242).
    data = shacl_gate._data_graph_from_engine_triples(owl_graph)
    assert data is not None, "engine triple path returned no data graph"

    # An inline shape: every Agent must have a name. alice/bob HAVE names, so add a
    # nameless Agent and prove the engine-sourced data graph carries the violation.
    owl_graph.add_node("ghost", type="Agent")
    data2 = shacl_gate._data_graph_from_engine_triples(owl_graph)
    import rdflib

    shapes = rdflib.Graph()
    shapes.parse(
        data="""
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix : <http://knuckles.team/kg#> .
:AgentShape a sh:NodeShape ;
    sh:targetClass :Agent ;
    sh:property [ sh:path :name ; sh:minCount 1 ;
                  sh:message "Agent requires a name" ] .
""",
        format="turtle",
    )
    import pyshacl

    conforms, _g, text = pyshacl.validate(
        data2, shacl_graph=shapes, inference="none", abort_on_first=False
    )
    assert conforms is False
    assert "ghost" in str(text) or "name" in str(text).lower()


def test_no_engine_fallback_still_works():
    """With no engine attached, the bridge's rdflib + python-reasoning fallbacks run.

    Uses a networkx-shaped stub graph (no engine client) so query_sparql / lightweight
    reasoning exercise the pure-Python last-resort path. This is the NO-ENGINE
    last-resort (rdflib), NOT a SQLite path — it stays as a graceful degradation.
    """
    pytest.importorskip("rdflib")
    import networkx as nx

    g = nx.MultiDiGraph()
    g.add_node("a", type="symbol")
    g.add_node("b", type="symbol")
    g.add_node("c", type="symbol")
    g.add_edge("a", "b", type="depends_on")
    g.add_edge("b", "c", type="depends_on")

    bridge = OWLBridge(graph=g, owl_backend=None, backend=None)

    # No engine (g has no .sparql) -> rdflib materialization fallback.
    rows = bridge.query_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
    assert isinstance(rows, list)

    # No engine (g has no .owl_reason) -> python RDFS+ transitive closure.
    inferences = bridge._lightweight_reasoning()
    transitive = {
        (i["subject"], i["object"])
        for i in inferences
        if i["predicate"] == "depends_on"
    }
    assert ("a", "c") in transitive  # a depends_on b depends_on c => a depends_on c
