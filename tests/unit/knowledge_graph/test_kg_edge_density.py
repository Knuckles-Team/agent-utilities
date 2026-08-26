"""Edge density and inference, proven against the REAL engine.

Three things the graph could not do before, each measured here rather than
asserted structurally:

1. A real ``ingest_concepts`` run leaves a connected taxonomy, not 2,438
   isolated string rows (CONCEPT:AU-KG.enrichment.relation-projection).
2. The bundled ``ontology*.ttl`` library reaches a reasoner at all
   (CONCEPT:AU-KG.ontology.ontology-driven-reasoning).
3. Reasoning produces at least one edge carrying ``inferred = true`` — the
   single assertion whose absence hid the entire failure.
"""

from __future__ import annotations

import pytest


def _concepts(count: int) -> list[dict[str, str]]:
    """A representative slice of the real corpus: dotted OKF-CIS ids + pillars."""
    areas = ("query", "ingest", "ontology", "storage")
    return [
        {
            "id": f"AU-KG.{areas[index % len(areas)]}.capability-{index}",
            "name": f"Capability {index}",
            "pillar": "AU-KG.platform",
        }
        for index in range(count)
    ]


def _live_engine(engine_graph):
    """A REAL engine bound to the same isolated tenant graph as the fixture."""
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    return IntelligenceGraphEngine(
        backend=EpistemicGraphBackend(graph_name=engine_graph.graph_name),
        defer_background_start=True,
    )


def _edges(engine) -> list[dict]:
    return engine.query_cypher(
        "MATCH (a)-[r]->(b) RETURN a.id AS src, type(r) AS rel, b.id AS dst"
    )


def _inferred_edges(engine) -> list[dict]:
    """Inferred edges — projected in the RETURN clause and filtered in Python.

    There is no Cypher predicate on this engine that filters by a RELATIONSHIP
    property; see
    :func:`test_relationship_property_predicates_never_filter_relationships`.
    Reading the property back in ``RETURN`` is the only form that works.
    """
    return [
        row
        for row in engine.query_cypher(
            "MATCH (a)-[r]->(b) "
            "RETURN a.id AS src, type(r) AS rel, b.id AS dst, r.inferred AS inferred"
        )
        if row.get("inferred")
    ]


def test_relationship_property_predicates_never_filter_relationships(engine_graph):
    """A measurement trap, pinned so nobody re-derives a false negative from it.

    Neither ``WHERE r.<prop> = <value>`` nor the inline ``[r {prop: value}]``
    map filters by a relationship property on this engine: the ``WHERE`` form
    matches NOTHING and the inline form matches EVERYTHING, for booleans and
    strings alike — while the identical predicate over a NODE property behaves
    correctly and ``RETURN r.<prop>`` reads the value back fine.

    Any audit concluding "the graph contains no edge with X" from either form
    has measured the query language, not the data. The only sound way to count
    edges by a relationship property is to project it in ``RETURN`` and filter
    client-side, which is what :func:`_inferred_edges` does.
    """
    engine_graph.batch_update(
        [
            {
                "op": "upsert_node",
                "id": "trap:a",
                "properties": {"id": "trap:a", "node_type": "Concept", "flag": True},
            },
            {
                "op": "upsert_node",
                "id": "trap:b",
                "properties": {"id": "trap:b", "node_type": "Concept"},
            },
            {
                "op": "upsert_node",
                "id": "trap:c",
                "properties": {"id": "trap:c", "node_type": "Concept"},
            },
            {
                "op": "upsert_edge",
                "source": "trap:a",
                "target": "trap:b",
                "properties": {
                    "relationship": "PART_OF",
                    "inferred": True,
                    "inferred_from": "trap",
                },
            },
            # A second edge with NO ``inferred`` property, to expose an
            # over-matching filter as well as an under-matching one.
            {
                "op": "upsert_edge",
                "source": "trap:b",
                "target": "trap:c",
                "properties": {"relationship": "PART_OF"},
            },
        ]
    )

    # The value is stored and readable...
    returned = engine_graph.query_cypher(
        "MATCH (a)-[r]->(b) WHERE a.id = 'trap:a' RETURN r.inferred AS inferred"
    )
    assert returned and returned[0]["inferred"] is True

    # ...and the same predicate shape over a NODE property filters correctly...
    assert [
        row["id"]
        for row in engine_graph.query_cypher(
            "MATCH (n) WHERE n.flag = true RETURN n.id AS id"
        )
    ] == ["trap:a"]

    # ...but over a RELATIONSHIP property, WHERE matches nothing,
    assert (
        engine_graph.query_cypher(
            "MATCH (a)-[r]->(b) WHERE r.inferred = true RETURN a.id AS src"
        )
        == []
    )
    assert (
        engine_graph.query_cypher(
            "MATCH (a)-[r]->(b) WHERE r.inferred_from = 'trap' RETURN a.id AS src"
        )
        == []
    )
    # ...and the inline relationship-property map is ignored outright — it
    # matches the edge that has no such property at all.
    inline = engine_graph.query_cypher(
        "MATCH (a)-[r {inferred: true}]->(b) RETURN a.id AS src"
    )
    assert {row["src"] for row in inline} == {"trap:a", "trap:b"}, inline

    # The sound measurement returns exactly one edge.
    assert [row["src"] for row in _inferred_edges(engine_graph)] == ["trap:a"]


# ── 1. reachability: a real ingest now produces edges ─────────────────────


def test_ingest_concepts_builds_a_connected_taxonomy(engine_graph):
    """``ingest_concepts`` → ``engine.add_node`` → the write chokepoint → edges.

    This is the reachability proof for the projection: nothing in this test
    calls the projection, ``link_nodes`` or ``add_edge``. It calls the same
    ``ingest_concepts`` the ecosystem concept bridge calls, and the edges appear
    because the chokepoint every durable node write passes through now emits
    them.
    """
    from agent_utilities.knowledge_graph.assimilation import ingest_concepts

    engine = _live_engine(engine_graph)
    corpus = _concepts(24)

    before = len(_edges(engine_graph))
    report = ingest_concepts(engine, corpus)
    assert report.ingested == len(corpus)
    after = _edges(engine_graph)

    assert len(after) > before, "a concept ingest still produces no edges"
    broader = {(row["src"], row["dst"]) for row in after if row["rel"] == "BROADER"}
    part_of = {(row["src"], row["dst"]) for row in after if row["rel"] == "PART_OF"}

    # Every concept reaches its dotted parent...
    assert (
        "concept:AU-KG.QUERY.CAPABILITY-0",
        "concept:AU-KG.QUERY",
    ) in broader
    # ...and the intermediate ancestor reaches the root.
    assert ("concept:AU-KG.QUERY", "concept:AU-KG") in broader
    # ...and the declared pillar is an edge, not just a string property.
    assert ("concept:AU-KG.QUERY.CAPABILITY-0", "concept:AU-KG.PLATFORM") in part_of

    # The property is KEPT — the edge is an addition, not a migration.
    rows = engine_graph.query_cypher(
        "MATCH (n:Concept {id: 'concept:AU-KG.QUERY.CAPABILITY-0'}) "
        "RETURN n.pillar AS pillar, n.concept_id AS concept_id"
    )
    assert rows and rows[0]["pillar"] == "AU-KG.platform"
    assert rows[0]["concept_id"] == "AU-KG.QUERY.CAPABILITY-0"

    # Idempotent: re-ingesting the same corpus adds no duplicate edges.
    ingest_concepts(engine, corpus)
    assert len(_edges(engine_graph)) == len(after)


def test_non_projecting_writes_are_untouched(engine_graph):
    """A ``RuntimeSignal`` has no reference-bearing property and gains no edge.

    The projection must not invent relationships for the 15,327-node telemetry
    population; ``subject`` is an opaque label, not a node identity.
    """
    from agent_utilities.observability.runtime_signals import persist_runtime_signals

    engine = _live_engine(engine_graph)
    written = persist_runtime_signals(
        engine,
        [
            {"kind": "engine_latency", "subject": "AddNode", "detail": {"ms": 3}}
            for _ in range(5)
        ],
    )
    assert written == 5
    assert _edges(engine_graph) == []


# ── 2. the ontology reaches a reasoner ────────────────────────────────────


def test_bundled_ontology_projects_real_axioms():
    """The 29 ``.ttl`` modules become reasoner input, not one hardcoded axiom."""
    from agent_utilities.knowledge_graph.ontology.axioms import (
        closure_sets,
        ontology_axioms,
    )

    axioms = ontology_axioms()
    if not axioms:
        pytest.skip("rdflib unavailable — the ontology cannot be parsed here")

    assert axioms.counts["triples"] > 5000
    # The measured shape of the library: these are the axiom families that
    # produce EDGES under a materialising reasoner.
    assert len(axioms.transitive_properties) >= 15
    assert len(axioms.symmetric_properties) >= 5
    assert len(axioms.inverse_properties) >= 20
    assert axioms.property_chains
    # camelCase OWL local names are translated to the graph's edge convention.
    assert "DEPENDS_ON" in axioms.transitive_properties
    assert "PART_OF" in axioms.transitive_properties

    transitive, symmetric, inverse = closure_sets()
    # Casefolded for the closure's own comparison against edge relationships.
    assert "depends_on" in transitive
    assert "part_of" in transitive
    assert symmetric and inverse


def test_owl_bridge_is_seeded_from_the_bundled_ontology():
    """``OWLBridge`` no longer reasons over a single hardcoded axiom."""
    from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge
    from agent_utilities.knowledge_graph.ontology.axioms import ontology_axioms

    if not ontology_axioms():
        pytest.skip("rdflib unavailable — the ontology cannot be parsed here")

    bridge = OWLBridge(graph=None, owl_backend=None, backend=None)
    assert len(bridge._pack_transitive) > 10, bridge._pack_transitive
    assert "depends_on" in bridge._pack_transitive
    assert bridge._pack_symmetric
    # ...and the Turtle it seeds the engine reasoner with says so.
    turtle = bridge._pack_axioms_turtle()
    assert "au:depends_on a owl:TransitiveProperty ." in turtle
    assert turtle.count("owl:TransitiveProperty") > 10


# ── 3. an inference actually lands as inferred = true ─────────────────────


def test_reasoning_produces_an_inferred_edge(engine_graph):
    """The assertion the forensics said would have caught the whole failure.

    A three-node ``PART_OF`` chain (``part_of`` is transitive both in the
    bundled ontology and in the lightweight closure's own set) must close to
    ``a -> c`` marked ``inferred = true``. Before this change the graph
    contained ZERO such edges and no test asserted one could exist.
    """
    engine = _live_engine(engine_graph)
    for node_id in ("part:a", "part:b", "part:c"):
        engine.add_node(node_id, "Concept", properties={"name": node_id})
    engine.link_nodes("part:a", "part:b", "PART_OF")
    engine.link_nodes("part:b", "part:c", "PART_OF")

    assert _inferred_edges(engine_graph) == [], "precondition: nothing inferred yet"

    engine._tick_reasoning()

    inferred = _inferred_edges(engine_graph)
    assert inferred, (
        "reasoning produced no edge carrying inferred = true; all edges: "
        f"{engine_graph.query_cypher('MATCH (a)-[r]->(b) RETURN a.id AS s, type(r) AS t, b.id AS d, r.inferred AS i')}"
    )
    assert any(row["src"] == "part:a" and row["dst"] == "part:c" for row in inferred), (
        inferred
    )


def test_inference_never_overwrites_an_asserted_edge(engine_graph):
    """Reasoning may CONNECT, never RELABEL.

    The native graph holds at most one edge per ordered node pair, so writing a
    derived relation onto an already-connected pair replaces the asserted one.
    Both asserted ``PART_OF`` edges must survive a reasoning pass intact.
    """
    engine = _live_engine(engine_graph)
    for node_id in ("keep:a", "keep:b", "keep:c"):
        engine.add_node(node_id, "Concept", properties={"name": node_id})
    engine.link_nodes("keep:a", "keep:b", "PART_OF")
    engine.link_nodes("keep:b", "keep:c", "PART_OF")

    engine._tick_reasoning()

    surviving = {
        (row["src"], row["rel"], row["dst"])
        for row in _edges(engine_graph)
        if row["src"].startswith("keep:")
    }
    assert ("keep:a", "PART_OF", "keep:b") in surviving, surviving
    assert ("keep:b", "PART_OF", "keep:c") in surviving, surviving
    # ...and neither asserted edge was restamped as inferred.
    asserted = engine_graph.query_cypher(
        "MATCH (a)-[r]->(b) WHERE a.id = 'keep:a' AND b.id = 'keep:b' "
        "RETURN r.inferred AS inferred"
    )
    assert asserted and not asserted[0]["inferred"]


def test_run_inference_has_a_production_caller(engine_graph):
    """``InferenceEngine`` is reachable from a scheduled tick, not only tests."""
    import inspect

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    source = inspect.getsource(IntelligenceGraphEngine._tick_reasoning)
    assert "run_inference" in source
    assert "run_closure" in source

    from agent_utilities.core.schedule_engine import _MAINTENANCE_REF_ALLOWLIST

    assert "reasoning" in _MAINTENANCE_REF_ALLOWLIST
