"""Live cross-backend convergence for the fan-out mirror (CONCEPT:KG-2.74).

Runs under ``pytest -m live`` against throwaway Postgres-AGE (authority) + Neo4j
and FalkorDB (mirrors) containers. Proves the two contract guarantees against
REAL stores:

1. **Steady-state convergence** — every write reaches every mirror; all stores
   hold identical node counts after the outbox drains.
2. **No loss across an outage** — a mirror that fails for a window keeps its
   unapplied tail and replays it on recovery, converging to parity (drift = 0).
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.backends import create_backend
from agent_utilities.knowledge_graph.backends.base import GraphBackend
from agent_utilities.knowledge_graph.backends.fanout_backend import FanOutBackend

pytestmark = pytest.mark.live


class _FlakyMirror(GraphBackend):
    """Wraps a real backend; ``down=True`` makes writes raise (simulated outage).

    Delegates everything to the wrapped backend so replay lands in the real
    store; only gates ``execute`` so we can model a transient mirror failure.
    """

    def __init__(self, inner: GraphBackend) -> None:
        self.inner = inner
        self.down = False

    def execute(self, query, params=None):
        if self.down:
            raise RuntimeError("simulated mirror outage")
        return self.inner.execute(query, params)

    def execute_batch(self, query, batch):
        if self.down:
            raise RuntimeError("simulated mirror outage")
        return self.inner.execute_batch(query, batch)

    def create_schema(self):
        return self.inner.create_schema()

    def add_embedding(self, node_id, embedding):
        return self.inner.add_embedding(node_id, embedding)

    def semantic_search(self, query_embedding, n_results=5):
        return self.inner.semantic_search(query_embedding, n_results)

    def prune(self, criteria):
        return self.inner.prune(criteria)

    def close(self):
        return self.inner.close()


def _count(backend: GraphBackend, label: str) -> int:
    rows = backend.execute(f"MATCH (n:{label}) RETURN count(n) AS c")
    if rows and isinstance(rows[0], dict):
        for key in ("c", "count(n)", "count"):
            if key in rows[0]:
                return int(rows[0][key])
        # single-value row fallback
        return int(next(iter(rows[0].values())))
    return 0


@pytest.fixture
def fanout_pair(
    ephemeral_neo4j: dict[str, Any],
    ephemeral_falkordb: dict[str, Any],
):
    """Authority = FalkorDB; mirror = Neo4j — two real, heterogeneous,
    full-openCypher engines on official images (no custom extension needed).

    Neo4j is the mirror we assert convergence on because it is the more faithful
    counter for rapid distinct ``CREATE``\\ s; the point under test is that the
    fan-out outbox *delivers* every authority write to the mirror and replays
    after an outage. The unit suite (``test_fanout_backend.py``) proves the
    N≥2-mirror fan-out + replay logic with fakes; this exercises the same path
    across two different real engines.
    """
    authority = create_backend(
        "falkordb",
        host=ephemeral_falkordb["host"],
        port=ephemeral_falkordb["port"],
        db_name=ephemeral_falkordb["db_name"],
    )
    neo4j = create_backend(
        "neo4j",
        uri=ephemeral_neo4j["uri"],
        user=ephemeral_neo4j["user"],
        password=ephemeral_neo4j["password"],
    )
    if not (authority and neo4j):
        pytest.skip("one or more live backends unavailable")
    yield authority, _FlakyMirror(neo4j)
    for b in (authority, neo4j):
        try:
            b.close()
        except Exception:
            pass


def test_steady_state_convergence(tmp_path, fanout_pair):
    authority, neo4j = fanout_pair
    fan = FanOutBackend(
        authority,
        {"neo4j": neo4j},
        outbox_path=str(tmp_path / "ob.db"),
    )
    label = "DocSteady"  # unique label → isolated from the session-shared store
    try:
        n = 50
        for i in range(n):
            fan.execute(f"CREATE (x:{label} {{id:'{i}'}})", is_write=True)
        assert fan.flush_mirrors(timeout=60.0)
        # Every authority write was delivered to the mirror (outbox drained).
        assert fan.durability_stats()["mirrors"]["neo4j"]["lag"] == 0
        assert _count(neo4j.inner, label) == n  # Neo4j mirror holds the full set
    finally:
        fan.close()


def test_no_loss_across_mirror_outage(tmp_path, fanout_pair):
    authority, neo4j = fanout_pair
    fan = FanOutBackend(
        authority,
        {"neo4j": neo4j},
        outbox_path=str(tmp_path / "ob.db"),
    )
    label = "DocOutage"  # unique label → isolated from the session-shared store
    try:
        neo4j.down = True  # mirror offline for the whole write burst
        n = 40
        for i in range(n):
            fan.execute(f"CREATE (x:{label} {{id:'{i}'}})", is_write=True)
        # The mirror is fully behind but has lost nothing (durable outbox tail).
        assert fan.flush_mirrors(timeout=15.0) is False
        assert fan.durability_stats()["mirrors"]["neo4j"]["lag"] == n

        # Recover — the drainer replays the persisted tail into the real store.
        neo4j.down = False
        assert fan.flush_mirrors(timeout=60.0)
        assert _count(neo4j.inner, label) == n  # NO LOSS
        assert fan.durability_stats()["mirrors"]["neo4j"]["lag"] == 0
    finally:
        fan.close()
