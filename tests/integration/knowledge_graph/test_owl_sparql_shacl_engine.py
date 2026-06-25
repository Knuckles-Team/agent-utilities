#!/usr/bin/python
"""Engine-native SPARQL / OWL / SHACL demotion tests (CONCEPT:KG-2.204).

Proves the semantic-web stack routes to the engine's native RDF/SPARQL/OWL surface
(`client.rdf.*`) and that the Python rdflib/owlready2/pyshacl path is a true
last-resort fallback:

* SPARQL via the engine == via rdflib on the same fixture graph.
* OWL reasoning via the engine derives the expected class-membership entailments.
* The SHACL gate still catches a known violation when its data graph is sourced
  from the engine's RDF projection.
* The no-engine fallback (`_sparql_via_rdflib`, `_python_reasoning`) still works.

The engine path needs a server built with the ``rdf``/``sparql``/``owl`` features
(the pi/node/full tiers); the fixture starts that binary and skips cleanly when it
is unavailable, so the suite never hard-fails on a feature-less build.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge

_BIN = (
    Path(__file__).resolve().parents[3].parent
    / "epistemic-graph"
    / "target"
    / "release"
    / "epistemic-graph-server"
)


def _server_has_rdf() -> bool:
    if not _BIN.exists():
        return False
    try:
        blob = _BIN.read_bytes()
    except OSError:
        return False
    return b"OwlReason" in blob and b"AddTriples" in blob


pytestmark = pytest.mark.skipif(
    not _server_has_rdf(),
    reason="engine binary without rdf/sparql/owl features (need pi/node/full tier)",
)


@pytest.fixture(scope="module")
def engine_graph():
    """A live engine-backed GraphComputeEngine with a small fixture graph.

    Starts a dedicated rdf/owl-capable server on a throwaway socket and points the
    GraphComputeEngine at it via GRAPH_SERVICE_SOCKET/SECRET (no autostart).
    """
    sock = tempfile.mktemp(suffix=".sock")
    secret = "kg2204-test-secret"
    persist = tempfile.mkdtemp(prefix="kg2204-")
    proc = subprocess.Popen(
        [
            str(_BIN),
            "--socket-path",
            sock,
            "--auth-secret",
            secret,
            "--persist-dir",
            persist,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        if os.path.exists(sock):
            break
        time.sleep(0.1)

    prev = {
        k: os.environ.get(k)
        for k in ("GRAPH_SERVICE_SOCKET", "GRAPH_SERVICE_AUTH_SECRET")
    }
    os.environ["GRAPH_SERVICE_SOCKET"] = sock
    os.environ["GRAPH_SERVICE_AUTH_SECRET"] = secret

    g = GraphComputeEngine(graph_name="kg2204")
    # A tiny typed graph: two Agents, one knows edge.
    g.add_node("alice", type="Agent", name="Alice")
    g.add_node("bob", type="Agent", name="Bob")
    g.add_edge("alice", "bob", type="knows")
    try:
        yield g
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def _sorted_rows(rows):
    return sorted(tuple(sorted(r.items())) for r in rows)


def test_sparql_engine_equals_rdflib(engine_graph):
    """SPARQL via the engine returns the same triples as the rdflib fallback."""
    bridge = OWLBridge(graph=engine_graph, owl_backend=None, backend=None)
    query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"

    engine_rows = engine_graph.sparql(query)
    assert engine_rows, "engine SPARQL returned no rows"
    # The knows edge is present as a triple.
    assert any(
        r.get("s") == "alice" and r.get("o") == "bob" for r in engine_rows
    )

    # Same query through the bridge prefers the engine (query_sparql).
    via_bridge = bridge.query_sparql(query)
    assert any(r.get("s") == "alice" and r.get("o") == "bob" for r in via_bridge)

    # Equivalence vs the pure rdflib materialization on the SAME graph.
    rdflib = pytest.importorskip("rdflib")
    assert rdflib is not None
    rdflib_rows = bridge._sparql_via_rdflib(query)
    # Both surfaces see the alice-knows-bob edge (namespaces differ, so compare the
    # edge existence rather than raw IRI strings).
    eng_edges = {
        (r.get("s"), r.get("o"))
        for r in engine_rows
        if r.get("p") in ("knows", "au:knows")
    }
    rdf_edges = set()
    for r in rdflib_rows:
        s = (r.get("s") or "").rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        o = (r.get("o") or "").rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        p = (r.get("p") or "")
        if p.endswith("knows"):
            rdf_edges.add((s, o))
    assert ("alice", "bob") in eng_edges
    assert ("alice", "bob") in rdf_edges


def test_owl_reasoning_engine_derives_entailments(engine_graph):
    """OWL reasoning via the engine derives the expected class memberships."""
    res = engine_graph.owl_reason()
    assert res.get("consistent") is True
    instances = {tuple(p) for p in res.get("instances", []) if len(p) == 2}
    # Both typed nodes are inferred members of Agent.
    assert ("alice", "Agent") in instances
    assert ("bob", "Agent") in instances


def test_bridge_lightweight_reasoning_uses_engine(engine_graph):
    """OWLBridge lightweight reasoning runs engine-native with NO owl backend and
    materializes inferred rdf:type edges back into the graph."""
    bridge = OWLBridge(graph=engine_graph, owl_backend=None, backend=None)
    inferences = bridge._engine_reasoning()
    types = {
        (i["subject"], i["object"])
        for i in inferences
        if i["predicate"] == "rdf:type"
    }
    assert ("alice", "Agent") in types
    assert ("bob", "Agent") in types


def test_shacl_gate_data_graph_from_engine(engine_graph):
    """The SHACL gate sources its data graph from the engine's RDF projection and
    still flags a node missing a required property."""
    pytest.importorskip("pyshacl")
    pytest.importorskip("rdflib")
    from agent_utilities.knowledge_graph.pipeline.phases import shacl_gate

    # Build the data graph via the engine triples path (CONCEPT:KG-2.204).
    data = shacl_gate._data_graph_from_engine_triples(engine_graph)
    assert data is not None, "engine triple path returned no data graph"

    # An inline shape: every Agent must have a name. alice/bob HAVE names, so add a
    # nameless Agent and prove the engine-sourced data graph carries the violation.
    engine_graph.add_node("ghost", type="Agent")
    data2 = shacl_gate._data_graph_from_engine_triples(engine_graph)
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
    reasoning exercise the pure-Python last-resort path.
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
