"""Unit tests for the fan-out (N-way mirror) backend (CONCEPT:KG-2.74).

Verifies the lossless-mirroring contract with recording fakes (no real
Postgres/Neo4j/FalkorDB server required):

* every write reaches every mirror (eventual convergence via the drainer);
* a mirror that is DOWN for a while loses nothing — it replays its outbox tail
  from the persisted cursor once it recovers;
* the durable outbox survives a "process restart" (close + reopen) and resumes
  from the cursor;
* reads are served from the authority and never block on a mirror.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from agent_utilities.knowledge_graph.backends.base import GraphBackend
from agent_utilities.knowledge_graph.backends.fanout_backend import FanOutBackend
from agent_utilities.knowledge_graph.backends.outbox import GraphOutbox


class RecordingBackend(GraphBackend):
    """Fake GraphBackend recording applied writes; can be toggled down."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.down = False
        self._lock = threading.Lock()
        self.writes: list[tuple[str, Any]] = []

    def _check(self) -> None:
        if self.down:
            raise RuntimeError(f"{self.name} unreachable")

    def execute(self, query, params=None):
        self._check()
        with self._lock:
            self.writes.append(("execute", query))
        return [{"backend": self.name}]

    def execute_batch(self, query, batch):
        self._check()
        with self._lock:
            self.writes.append(("execute_batch", len(batch)))
        return [{"backend": self.name}]

    def create_schema(self):
        with self._lock:
            self.writes.append(("create_schema", None))

    def add_embedding(self, node_id, embedding):
        self._check()
        with self._lock:
            self.writes.append(("add_embedding", node_id))

    def semantic_search(self, query_embedding, n_results=5):
        return [{"backend": self.name}]

    def prune(self, criteria):
        self._check()

    def close(self):
        pass

    def n_execute(self) -> int:
        with self._lock:
            return sum(1 for op, _ in self.writes if op == "execute")


def _make(tmp_path: Path, mirrors: dict[str, RecordingBackend]) -> FanOutBackend:
    return FanOutBackend(
        RecordingBackend("authority"),
        mirrors,
        outbox_path=str(tmp_path / "outbox.db"),
    )


def test_write_fans_out_to_every_mirror(tmp_path):
    a, b = RecordingBackend("a"), RecordingBackend("b")
    fan = _make(tmp_path, {"a": a, "b": b})
    try:
        for i in range(20):
            fan.execute(f"CREATE (n:Doc {{id:'{i}'}})", is_write=True)
        assert fan.flush_mirrors(timeout=10.0)
        assert a.n_execute() == 20
        assert b.n_execute() == 20
    finally:
        fan.close()


def test_reads_go_to_authority_not_mirrors(tmp_path):
    a = RecordingBackend("a")
    fan = _make(tmp_path, {"a": a})
    try:
        rows = fan.execute("MATCH (n) RETURN n", is_write=False)
        assert rows == [{"backend": "authority"}]
        # A pure read must never be enqueued to a mirror.
        assert fan.flush_mirrors(timeout=5.0)
        assert a.n_execute() == 0
    finally:
        fan.close()


def test_down_mirror_loses_nothing_and_replays(tmp_path):
    """A mirror that is offline during writes catches up fully on recovery."""
    up, down = RecordingBackend("up"), RecordingBackend("down")
    down.down = True  # offline before any write
    fan = _make(tmp_path, {"up": up, "down": down})
    try:
        for i in range(15):
            fan.execute(f"CREATE (n:Doc {{id:'{i}'}})", is_write=True)
        # The healthy mirror converges; the down one is fully behind.
        assert fan.flush_mirrors(timeout=2.0) is False
        assert up.n_execute() == 15
        stats = fan.durability_stats()
        assert stats["mirrors"]["down"]["lag"] == 15
        # Recover the mirror — the drainer replays the persisted tail.
        down.down = False
        assert fan.flush_mirrors(timeout=10.0)
        assert down.n_execute() == 15  # NO LOSS
        assert fan.durability_stats()["mirrors"]["down"]["lag"] == 0
    finally:
        fan.close()


def test_outbox_survives_restart(tmp_path):
    """The durable log resumes from its cursor after a process restart."""
    path = tmp_path / "ob.db"
    ob = GraphOutbox(path, ["m"])
    ob.append("execute", {"query": "CREATE (n)", "params": None})
    ob.append("execute", {"query": "CREATE (m)", "params": None})
    assert ob.lag("m") == 2
    ob.ack("m", 1)  # applied the first only
    ob.close()

    # Reopen — the cursor persisted; only the unapplied tail remains pending.
    ob2 = GraphOutbox(path, ["m"])
    try:
        pending = ob2.pending("m")
        assert [e.seq for e in pending] == [2]
        assert ob2.applied_seq("m") == 1
    finally:
        ob2.close()


class _GraphStub:
    def _get_all_nodes(self):
        return ["a", "b"]

    def _get_node_properties(self, nid):
        return {"type": "Concept", "name": nid}

    def _get_all_edges(self):
        return [("a", "b", {"type": "RELATED_TO"})]


class _AuthorityWithGraph(RecordingBackend):
    @property
    def graph(self):
        return _GraphStub()


def test_reconcile_repairs_each_mirror(tmp_path):
    """reconcile() delegates to the tiered re-sync for every mirror, fast and
    without spinning — proves the authority→mirror drift-repair wrapper."""
    mirror = RecordingBackend("m")
    fan = FanOutBackend(
        _AuthorityWithGraph("auth"),
        {"m": mirror},
        outbox_path=str(tmp_path / "ob.db"),
    )
    try:
        report = fan.reconcile()  # all mirrors
        assert "m" in report
        assert report["m"]["nodes"] == 2  # both authority nodes re-synced
        assert report["m"]["edges"] == 1
        assert report["m"]["errors"] == 0
        # The mirror received node CREATE + edge MERGE writes.
        writes = [
            c for c in mirror.writes
            if c[0] == "execute"
        ]
        assert writes  # drift repair actually wrote to the mirror
    finally:
        fan.close()


def test_durability_stats_shape(tmp_path):
    fan = _make(tmp_path, {"a": RecordingBackend("a")})
    try:
        fan.execute("CREATE (n:Doc {id:'1'})", is_write=True)
        assert fan.flush_mirrors(timeout=5.0)
        stats = fan.durability_stats()
        assert stats["authority_writes"] == 1
        assert "a" in stats["mirrors"]
        assert stats["mirrors"]["a"]["writes"] == 1
        assert stats["mirrors"]["a"]["lag"] == 0
    finally:
        fan.close()
