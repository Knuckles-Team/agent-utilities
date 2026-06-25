"""Dual-mode round-trip tests for the consolidated local-store migrations.

CONCEPT:KG-2.204 (card cache), KG-2.205 (registry predicate), KG-2.206 (timeseries
engine arm), KG-2.208 (writeback proposals), KG-2.209 (code-health baselines).

Each migrated store routes to the durable engine authority when one is present and
falls back to its zero-infra local store (SQLite / JSON) for the ``tiny`` profile.
These tests prove BOTH arms: a ``_FakeDurableBackend`` (an ``execute()``-bearing
graph backend whose class name is NOT in the non-durable set) drives engine mode,
and ``backend=None`` drives the local fallback.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from agent_utilities.knowledge_graph.backends.base import is_durable_backend


class _FakeDurableBackend:
    """A tiny in-memory Cypher backend: supports the MERGE/MATCH/SET/count shapes
    the consolidated stores emit. Its class name is durable (not in the
    non-durable set), so ``is_durable_backend`` selects the engine arm."""

    def __init__(self) -> None:
        # label -> { node_key -> props }
        self.store: dict[str, dict[str, dict]] = {}

    def execute(self, query: str, params: dict | None = None):
        params = params or {}
        q = query.strip()
        label_m = re.search(r"\((\w+):(\w+)", q)
        label = label_m.group(2) if label_m else ""
        bag = self.store.setdefault(label, {})

        # count(p)
        if "count(p)" in q:
            target = params.get("target")
            n = sum(1 for p in bag.values() if p.get("target") == target)
            return [{"c": n}]

        if q.startswith("MERGE"):
            key_m = re.search(r"\{(\w+):\s*\$(\w+)\}", q)
            keyfield = key_m.group(1)
            keyparam = key_m.group(2)
            node_key = params[keyparam]
            node = bag.setdefault(node_key, {keyfield: node_key})
            # apply SET assignments: <var>.<field> = $param
            for field, pname in re.findall(r"\.(\w+)\s*=\s*\$(\w+)", q):
                node[field] = params.get(pname)
            # literal assignments: <var>.<field> = 'literal'
            for field, lit in re.findall(r"\.(\w+)\s*=\s*'([^']*)'", q):
                node[field] = lit
            return []

        if q.startswith("MATCH"):
            var_m = re.search(r"\((\w+):", q)
            var = var_m.group(1) if var_m else "n"
            # SET on match: MATCH (p:L {id:$id}) SET p.status = $status
            if " SET " in q:
                key_m = re.search(r"\{(\w+):\s*\$(\w+)\}", q)
                node_key = params[key_m.group(2)]
                node = bag.get(node_key)
                if node is not None:
                    for field, pname in re.findall(r"\.(\w+)\s*=\s*\$(\w+)", q):
                        node[field] = params.get(pname)
                return []
            # IN $hashes
            in_m = re.search(r"\.(\w+)\s+IN\s+\$(\w+)", q)
            if in_m:
                field, pname = in_m.group(1), in_m.group(2)
                wanted = set(params.get(pname) or [])
                return [
                    {var: p} for p in bag.values() if p.get(field) in wanted
                ]
            # {keyfield:$id}
            key_m = re.search(r"\{(\w+):\s*\$(\w+)\}", q)
            if key_m:
                node = bag.get(params[key_m.group(2)])
                return [{var: node}] if node is not None else []
            # WHERE field = $param
            where_m = re.search(r"WHERE\s+\w+\.(\w+)\s*=\s*\$(\w+)", q)
            if where_m:
                field, pname = where_m.group(1), where_m.group(2)
                val = params.get(pname)
                return [{var: p} for p in bag.values() if p.get(field) == val]
            # plain MATCH (p:L) RETURN p
            return [{var: p} for p in bag.values()]
        return []


def test_fake_backend_is_durable() -> None:
    assert is_durable_backend(_FakeDurableBackend()) is True
    assert is_durable_backend(None) is False


# ── KG-2.204 card cache ────────────────────────────────────────────────
def test_card_store_sqlite_roundtrip(tmp_path) -> None:
    from agent_utilities.knowledge_graph.enrichment.cards import CardStore

    s = CardStore(path=str(tmp_path / "cards.db"))
    assert s.mode == "sqlite"
    s.put_many([("h1", "does X", ["r1", "r2"]), ("h2", "does Y", [])])
    got = s.get_many(["h1", "h2", "h3"])
    assert got["h1"] == ("does X", ["r1", "r2"])
    assert got["h2"] == ("does Y", [])
    assert "h3" not in got
    # durable across "restart"
    s2 = CardStore(path=str(tmp_path / "cards.db"))
    assert s2.get_many(["h1"])["h1"][0] == "does X"


def test_card_store_engine_roundtrip() -> None:
    from agent_utilities.knowledge_graph.enrichment.cards import CardStore

    be = _FakeDurableBackend()
    s = CardStore(backend=be)
    assert s.mode == "graph"
    s.put_many([("h1", "does X", ["r1"]), ("h2", "does Y", [])])
    got = s.get_many(["h1", "h2"])
    assert got["h1"] == ("does X", ["r1"])
    assert got["h2"] == ("does Y", [])
    # stored on the engine, not a local DB
    assert "CardCache" in be.store
    assert be.store["CardCache"]["h1"]["summary"] == "does X"


# ── KG-2.208 writeback proposals ───────────────────────────────────────
def test_proposal_queue_json_roundtrip(tmp_path) -> None:
    from agent_utilities.knowledge_graph.enrichment.writeback.approval import (
        ProposalQueue,
    )

    q = ProposalQueue(path=str(tmp_path / "wb.json"))
    q._backend = None
    q.mode = "json"  # tiny profile: JSON fallback
    pid = q.enqueue("sink:x", {"a": 1, "_secret": 2}, [{"op": "set"}])
    p = q.get(pid)
    assert p["target"] == "sink:x"
    assert p["ops"] == {"a": 1}  # underscore-prefixed dropped
    assert p["status"] == "pending"
    assert q.list("pending")[0]["id"] == pid
    q.mark(pid, "approved")
    assert q.get(pid)["status"] == "approved"
    assert q.list("pending") == []


def test_proposal_queue_engine_roundtrip() -> None:
    from agent_utilities.knowledge_graph.enrichment.writeback.approval import (
        ProposalQueue,
    )

    be = _FakeDurableBackend()
    q = ProposalQueue(backend=be)
    assert q.mode == "graph"
    pid = q.enqueue("sink:y", {"a": 1, "_secret": 2}, [{"op": "set"}])
    p = q.get(pid)
    assert p["target"] == "sink:y"
    assert p["ops"] == {"a": 1}
    assert p["proposals"] == [{"op": "set"}]
    assert p["status"] == "pending"
    assert q.list("pending")[0]["id"] == pid
    q.mark(pid, "approved")
    assert q.get(pid)["status"] == "approved"
    assert q.list("pending") == []
    assert "WritebackProposal" in be.store


# ── KG-2.209 code-health baselines ─────────────────────────────────────
def test_code_health_baseline_file_roundtrip(tmp_path, monkeypatch) -> None:
    from agent_utilities.knowledge_graph.adaptation import code_health

    monkeypatch.setattr(code_health, "_BASELINE_DIR", tmp_path)
    snap = {"findings": [{"id": "f1"}]}
    assert code_health._load_baseline_snapshot(None, "repoA") is None
    code_health._save_baseline_snapshot(None, "repoA", snap)
    assert code_health._load_baseline_snapshot(None, "repoA") == snap


def test_code_health_baseline_engine_roundtrip() -> None:
    from agent_utilities.knowledge_graph.adaptation import code_health

    be = _FakeDurableBackend()
    assert code_health._load_baseline_snapshot(be, "repoB") is None
    snap = {"findings": [{"id": "f2"}]}
    code_health._save_baseline_snapshot(be, "repoB", snap)
    assert code_health._load_baseline_snapshot(be, "repoB") == snap
    assert "CodeHealthBaseline" in be.store


def test_code_health_baseline_backend_selection() -> None:
    from agent_utilities.knowledge_graph.adaptation import code_health

    class _Eng:
        backend = _FakeDurableBackend()

    assert code_health._baseline_backend(_Eng()) is not None
    assert code_health._baseline_backend(object()) is None


# ── KG-2.206 timeseries factory dual-mode ──────────────────────────────
def test_timeseries_factory_sqlite(tmp_path) -> None:
    from agent_utilities.knowledge_graph.memory.timeseries import (
        get_timeseries_backend,
    )
    from agent_utilities.knowledge_graph.memory.timeseries.sqlite_backend import (
        SQLiteTimeSeriesBackend,
    )

    be = get_timeseries_backend("sqlite", db_path=str(tmp_path / "ts.db"))
    assert isinstance(be, SQLiteTimeSeriesBackend)


def test_timeseries_factory_auto_degrades(monkeypatch) -> None:
    """auto → SQLite when the engine is unreachable (zero-infra tiny)."""
    import agent_utilities.knowledge_graph.memory.timeseries as ts_pkg
    from agent_utilities.knowledge_graph.memory.timeseries.sqlite_backend import (
        SQLiteTimeSeriesBackend,
    )

    def _boom(self):
        raise RuntimeError("engine down")

    monkeypatch.setattr(ts_pkg.EngineTimeSeriesBackend, "initialize", _boom)
    be = ts_pkg.get_timeseries_backend("auto")
    assert isinstance(be, SQLiteTimeSeriesBackend)


class _FakeTsClient:
    """Minimal stand-in for SyncEpistemicGraphClient.timeseries + nodes/query."""

    def __init__(self) -> None:
        self.series: dict[str, list[tuple[int, list[float]]]] = {}
        self.nodes_props: dict[str, dict] = {}

        outer = self

        class _TS:
            def register_series(self, sid, *, entity_id=None, field_names=None,
                                metadata=None):
                props = {"series_id": sid, "field_names": list(field_names or [])}
                if metadata:
                    props.update(metadata)
                outer.nodes_props[f"series:{sid}"] = props

            def append(self, sid, points, *, field_names=None, **kw):
                outer.series.setdefault(sid, []).extend(points)
                return len(points)

            def range(self, sid, frm, to):
                return [
                    (ts, v) for ts, v in outer.series.get(sid, [])
                    if frm <= ts < to
                ]

        class _Nodes:
            def properties(self, node_id):
                return outer.nodes_props.get(node_id)

        class _Query:
            def cypher(self, q):
                out = []
                for nid, props in outer.nodes_props.items():
                    if props.get("symbol") and f"'{props['symbol']}'" in q:
                        out.append({"series_id": props["series_id"]})
                return out

        self.timeseries = _TS()
        self.nodes = _Nodes()
        self.query = _Query()

    def close(self):
        pass


def test_timeseries_engine_roundtrip() -> None:
    from agent_utilities.knowledge_graph.memory.timeseries.base import (
        TimeSeriesDataPoint,
    )
    from agent_utilities.knowledge_graph.memory.timeseries.engine_backend import (
        EngineTimeSeriesBackend,
    )

    be = EngineTimeSeriesBackend(client=_FakeTsClient())
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    be.insert([
        TimeSeriesDataPoint(symbol="AAPL", timestamp=t0,
                            metrics={"px": 100.0, "vol": 5.0}),
        TimeSeriesDataPoint(symbol="AAPL", timestamp=t1,
                            metrics={"px": 101.0, "vol": 6.0}),
    ])
    got = be.query("AAPL", t0, t1)
    assert len(got) == 2
    assert got[0].metrics["px"] == 100.0
    assert got[0].metrics["vol"] == 5.0
    assert got[1].metrics["px"] == 101.0


def test_timeseries_engine_tag_isolation() -> None:
    from agent_utilities.knowledge_graph.memory.timeseries.base import (
        TimeSeriesDataPoint,
    )
    from agent_utilities.knowledge_graph.memory.timeseries.engine_backend import (
        EngineTimeSeriesBackend,
    )

    be = EngineTimeSeriesBackend(client=_FakeTsClient())
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    be.insert([
        TimeSeriesDataPoint(symbol="X", timestamp=t0, metrics={"px": 1.0},
                            tags={"venue": "a"}),
        TimeSeriesDataPoint(symbol="X", timestamp=t0, metrics={"px": 2.0},
                            tags={"venue": "b"}),
    ])
    only_a = be.query("X", t0, t1, tags={"venue": "a"})
    assert len(only_a) == 1
    assert only_a[0].metrics["px"] == 1.0
    assert only_a[0].tags == {"venue": "a"}
