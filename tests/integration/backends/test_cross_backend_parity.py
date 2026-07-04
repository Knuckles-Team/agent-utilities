"""Live cross-backend parity for native data migration (CONCEPT:AU-KG.backend.mirror-health-repair).

Proves the abstraction: the SAME source graph, copied via :func:`copy_graph` into
every durable backend (Postgres-transpiler, Neo4j, FalkorDB, LadybugDB), lands as
identical node/edge counts — no dialect breaks, no malformed cypher, no dropped
edges. Plus a durable→durable round-trip (Neo4j → FalkorDB) converges.

`pytest -m live` (containers for neo4j/falkordb/postgres; ladybug is embedded).
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.backends import create_backend
from agent_utilities.knowledge_graph.backends.base import GraphBackend
from agent_utilities.knowledge_graph.migration import copy_graph

pytestmark = pytest.mark.live

# A deterministic source graph: declared + ad-hoc + nested props, plus edges.
_N_NODES = 40
_N_EDGES = 25


class _SourceGraph:
    """A fake L1 compute graph (the `.graph` copy_graph reads from).

    Uses the declared ``Agent`` node + ``DEPENDS_ON`` (Agent→Agent) edge so the
    strict-schema LadybugDB target accepts the workload (both must be declared);
    full-cypher backends accept anything. Props mix declared (name/description/
    importance_score), nested (tags), and ad-hoc (adhoc_field → metadata JSON).
    """

    def _get_all_nodes(self):
        return [f"agent:{i}" for i in range(_N_NODES)]

    def _get_node_properties(self, nid):
        i = int(nid.split(":")[1])
        return {
            "id": nid,
            "type": "Agent",
            "name": f"Agent {i}",
            "description": f"desc {i}",
            "importance_score": 0.1 + (i % 10) / 100.0,
            "tags": ["a", "b"],  # nested → JSON-encoded on map-unsafe drivers
            "adhoc_field": f"extra-{i}",  # folds into metadata on strict-schema
        }

    def _get_all_edges(self):
        return [
            (
                f"agent:{i}",
                f"agent:{(i + 1) % _N_NODES}",
                {"type": "DEPENDS_ON", "confidence": 0.9},
            )
            for i in range(_N_EDGES)
        ]


class _Source:
    graph = _SourceGraph()


def _count_nodes(b: GraphBackend, label: str = "Agent") -> int:
    rows = b.execute(f"MATCH (n:{label}) RETURN count(n) AS c")
    if rows and isinstance(rows[0], dict):
        for k in ("c", "count(n)", "count"):
            if k in rows[0]:
                return int(rows[0][k])
        return int(next(iter(rows[0].values())))
    return 0


@pytest.fixture
def four_backends(
    tmp_path,
    ephemeral_neo4j: dict[str, Any],
    ephemeral_falkordb: dict[str, Any],
    ephemeral_pg_age: dict[str, Any],
):
    """neo4j + falkordb + postgres(transpiler) + ladybug(embedded), all fresh."""
    backends = {
        "neo4j": create_backend(
            "neo4j",
            uri=ephemeral_neo4j["uri"],
            user=ephemeral_neo4j["user"],
            password=ephemeral_neo4j["password"],
        ),
        "falkordb": create_backend(
            "falkordb",
            host=ephemeral_falkordb["host"],
            port=ephemeral_falkordb["port"],
            db_name=ephemeral_falkordb["db_name"],
        ),
        "postgresql": create_backend(
            "postgresql",
            uri=ephemeral_pg_age["uri"],
            db_name=ephemeral_pg_age["db_name"],
        ),
        "ladybug": create_backend("ladybug", db_path=str(tmp_path / "parity.db")),
    }
    missing = [k for k, v in backends.items() if v is None]
    if missing:
        pytest.skip(f"backends unavailable: {missing}")
    yield backends
    for b in backends.values():
        try:
            b.close()
        except Exception:
            pass


def test_copy_graph_converges_across_all_backends(four_backends):
    """The same source graph migrated into all 4 backends → identical node counts."""
    for name, b in four_backends.items():
        summary = copy_graph(_Source(), b, copy_embeddings=False)
        assert summary["errors"] == 0, f"{name}: {summary['errors']} write errors"
        assert summary["nodes"] == _N_NODES, f"{name}: wrote {summary['nodes']} nodes"
        assert summary["edges"] == _N_EDGES, f"{name}: wrote {summary['edges']} edges"

    # Every backend holds the full node set (the interchangeable-storage assertion).
    for name, b in four_backends.items():
        assert _count_nodes(b) == _N_NODES, f"{name}: node count != {_N_NODES}"


def test_copy_graph_durable_round_trip_neo4j_to_falkordb(four_backends):
    """Seed Neo4j from source, then migrate Neo4j → FalkorDB and converge."""
    neo, fk = four_backends["neo4j"], four_backends["falkordb"]
    copy_graph(_Source(), neo, copy_embeddings=False)
    # wipe falkordb's earlier copy and re-fill it FROM neo4j (durable source read)
    fk.execute("MATCH (n) DETACH DELETE n")
    summary = copy_graph(neo, fk, copy_embeddings=False)
    assert summary["nodes"] == _N_NODES
    assert _count_nodes(fk) == _count_nodes(neo) == _N_NODES
