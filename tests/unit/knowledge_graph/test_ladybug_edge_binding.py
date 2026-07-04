"""LadybugDB (Kuzu) label-less edge binding (CONCEPT:AU-KG.backend.mirror-health-repair).

Kuzu rel creation must bind to typed node tables, but ingest emits label-less
edge writes (``MATCH (s {id:$x}) MATCH (t {id:$y}) MERGE (s)-[:REL]->(t)``).
LadybugBackend resolves each endpoint's table by id, ensures the rel-pair table,
and binds the MERGE — so typed edges persist and traverse (previously they were
silently dropped: "Create rel bound by multiple node labels is not supported").
Skipped when the Kuzu/ladybug driver isn't installed.
"""

import pytest


def _backend(tmp_path):
    try:
        from agent_utilities.knowledge_graph.backends import create_backend

        b = create_backend(backend_type="ladybug", db_path=str(tmp_path / "lb.db"))
    except Exception as exc:  # noqa: BLE001 - driver optional
        pytest.skip(f"ladybug/kuzu unavailable: {exc}")
    if b is None or type(b).__name__ != "LadybugBackend":
        pytest.skip("ladybug backend not available")
    return b


def _seed_chain(b):
    for r in [
        {"id": "top", "name": "top"},
        {"id": "mid", "name": "mid"},
        {"id": "leaf", "name": "leaf"},
    ]:
        b.execute("MERGE (n:Code {id:$id}) SET n.name=$name", r)
    # The exact label-LESS edge shape ingest_external_batch emits.
    for s, t in [("top", "mid"), ("mid", "leaf")]:
        b.execute(
            "MATCH (s {id:$source}) MATCH (t {id:$target}) MERGE (s)-[r:calls]->(t)",
            {"source": s, "target": t},
        )


def test_label_less_edge_binds_and_traverses(tmp_path):
    b = _backend(tmp_path)
    _seed_chain(b)
    edges = b.execute(
        "MATCH (a:Code)-[:calls]->(c:Code) RETURN a.name AS f, c.name AS t"
    )
    assert {(e["f"], e["t"]) for e in edges} == {("top", "mid"), ("mid", "leaf")}
    # find_references (callers of mid)
    refs = b.execute(
        "MATCH (caller:Code)-[:calls]->(def:Code) WHERE def.name=$n RETURN caller.name AS name",
        {"n": "mid"},
    )
    assert [r["name"] for r in refs] == ["top"]
    # trace_call_graph (transitive callees of top) — Kuzu-native var-length
    trace = b.execute(
        "MATCH (s:Code)-[:calls*1..3]->(x:Code) WHERE s.name=$n RETURN DISTINCT x.name AS name",
        {"n": "top"},
    )
    assert {r["name"] for r in trace} == {"mid", "leaf"}


def test_execute_batch_unwind_persists(tmp_path):
    b = _backend(tmp_path)
    b.execute_batch(
        "UNWIND $batch AS row MERGE (n:Code {id: row.id}) SET n.name = row.`name`",
        [{"id": "a", "name": "alpha"}, {"id": "b", "name": "beta"}],
    )
    assert b.execute(
        "MATCH (c:Code) WHERE c.name=$n RETURN c.id AS id", {"n": "beta"}
    ) == [{"id": "b"}]
