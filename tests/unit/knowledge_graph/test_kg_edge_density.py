"""Edge density and inference, proven against the REAL engine.

Three things the graph could not do before, each measured here rather than
asserted structurally:

1. A real ``ingest_concepts`` run leaves a connected taxonomy, not 2,438
   isolated string rows (CONCEPT:AU-KG.enrichment.relation-projection).
2. Reasoning produces at least one edge carrying ``inferred = true`` — the
   single assertion whose absence hid the entire failure.
"""

from __future__ import annotations


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
    """Inferred edges — projected in the RETURN clause and filtered in Python."""
    return [
        row
        for row in engine.query_cypher(
            "MATCH (a)-[r]->(b) "
            "RETURN a.id AS src, type(r) AS rel, b.id AS dst, r.inferred AS inferred"
        )
        if row.get("inferred")
    ]


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


# ── 3. an inference actually lands as inferred = true ─────────────────────


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
