"""Engine-only round-trip tests for the consolidated store migrations.

CONCEPT:AU-KG.backend.cache-lives-as (card cache), KG-2.245 (registry predicate), KG-2.246 (timeseries
engine arm), KG-2.247 (writeback proposals), KG-2.248 (code-health baselines).

USER DIRECTIVE: there is NO SQLite/JSON/file fallback — the consolidated stores
route to the ONE epistemic-graph engine authority unconditionally. So these tests
validate against the **REAL ephemeral engine** the session fixture deploys
(CONCEPT:AU-KG.memory.provides-real-ephemeral-one): each store test requests the conftest ``engine_graph`` (a fresh
per-test tenant on the running engine), binds an ``EpistemicGraphBackend`` to that
tenant, and asserts the store round-trips through the engine's ``execute()`` Cypher
surface. The previously-SQLite/JSON tests are deleted with the fallback code.

The time-series field-vector mapping is exercised with a tiny in-memory tsdb client
stand-in (``_FakeTsClient``) because it tests the *mapping* logic (metrics ->
ordered field vector, tag-set -> distinct series), not a storage fallback — there
is no SQLite arm to test.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent_utilities.knowledge_graph.backends.base import (
    is_engine_authority_backend,
)


@pytest.fixture()
def engine_backend(engine_graph):
    """An ``EpistemicGraphBackend`` bound to the per-test REAL engine tenant.

    CONCEPT:AU-KG.memory.provides-real-ephemeral-one — gives the consolidated stores a live engine-authority
    ``backend.execute()`` over the fresh ``engine_graph`` tenant, so a round-trip
    actually exercises the shipped database (no mock, no SQLite).
    """
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )

    be = EpistemicGraphBackend(graph_name=engine_graph.graph_name)
    assert is_engine_authority_backend(be)
    return be


# ── KG-2.244 card cache ────────────────────────────────────────────────
def test_card_store_engine_roundtrip(engine_backend) -> None:
    from agent_utilities.knowledge_graph.enrichment.cards import CardStore

    s = CardStore(backend=engine_backend)
    s.put_many([("h1", "does X", ["r1"]), ("h2", "does Y", [])])
    got = s.get_many(["h1", "h2", "h3"])
    assert got["h1"] == ("does X", ["r1"])
    assert got["h2"] == ("does Y", [])
    assert "h3" not in got


# ── KG-2.247 writeback proposals ───────────────────────────────────────
def test_proposal_queue_engine_roundtrip(engine_backend) -> None:
    from agent_utilities.knowledge_graph.enrichment.writeback.approval import (
        ProposalQueue,
    )

    q = ProposalQueue(backend=engine_backend)
    pid = q.enqueue("sink:y", {"a": 1, "_secret": 2}, [{"op": "set"}])
    p = q.get(pid)
    assert p["target"] == "sink:y"
    assert p["ops"] == {"a": 1}  # underscore-prefixed dropped
    assert p["proposals"] == [{"op": "set"}]
    assert p["status"] == "pending"
    assert q.list("pending")[0]["id"] == pid
    q.mark(pid, "approved")
    assert q.get(pid)["status"] == "approved"
    assert q.list("pending") == []


# ── KG-2.248 code-health baselines ─────────────────────────────────────
def test_code_health_baseline_engine_roundtrip(engine_backend) -> None:
    from agent_utilities.knowledge_graph.adaptation import code_health

    assert code_health._load_baseline_snapshot(engine_backend, "repoB") is None
    snap = {"findings": [{"id": "f2"}]}
    code_health._save_baseline_snapshot(engine_backend, "repoB", snap)
    assert code_health._load_baseline_snapshot(engine_backend, "repoB") == snap


def test_code_health_baseline_backend_selection(engine_backend) -> None:
    """``_baseline_backend`` accepts an engine-bound engine and resolves it."""
    from agent_utilities.knowledge_graph.adaptation import code_health

    class _Eng:
        backend = engine_backend

    resolved = code_health._baseline_backend(_Eng())
    assert is_engine_authority_backend(resolved)


# ── KG-2.246 timeseries factory is engine-only (no SQLite) ─────────────
def test_timeseries_factory_rejects_non_engine() -> None:
    """The only backend is the engine tsdb; an unknown type is rejected (no SQLite)."""
    from agent_utilities.knowledge_graph.memory.timeseries import (
        get_timeseries_backend,
    )

    with pytest.raises(ValueError, match="engine"):
        get_timeseries_backend("sqlite")


class _FakeTsClient:
    """Minimal stand-in for SyncEpistemicGraphClient.timeseries + nodes/query.

    Exercises the field-vector MAPPING (metrics -> ordered fields, tag-set ->
    distinct series), not a storage fallback — there is no SQLite arm.
    """

    def __init__(self) -> None:
        self.series: dict[str, list[tuple[int, list[float]]]] = {}
        self.nodes_props: dict[str, dict] = {}

        outer = self

        class _TS:
            def register_series(
                self, sid, *, entity_id=None, field_names=None, metadata=None
            ):
                props = {"series_id": sid, "field_names": list(field_names or [])}
                if metadata:
                    props.update(metadata)
                outer.nodes_props[f"series:{sid}"] = props

            def append(self, sid, points, *, field_names=None, **kw):
                outer.series.setdefault(sid, []).extend(points)
                return len(points)

            def range(self, sid, frm, to):
                return [
                    (ts, v) for ts, v in outer.series.get(sid, []) if frm <= ts < to
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
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 1, 2, tzinfo=UTC)
    be.insert(
        [
            TimeSeriesDataPoint(
                symbol="AAPL", timestamp=t0, metrics={"px": 100.0, "vol": 5.0}
            ),
            TimeSeriesDataPoint(
                symbol="AAPL", timestamp=t1, metrics={"px": 101.0, "vol": 6.0}
            ),
        ]
    )
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
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 1, 2, tzinfo=UTC)
    be.insert(
        [
            TimeSeriesDataPoint(
                symbol="X", timestamp=t0, metrics={"px": 1.0}, tags={"venue": "a"}
            ),
            TimeSeriesDataPoint(
                symbol="X", timestamp=t0, metrics={"px": 2.0}, tags={"venue": "b"}
            ),
        ]
    )
    only_a = be.query("X", t0, t1, tags={"venue": "a"})
    assert len(only_a) == 1
    assert only_a[0].metrics["px"] == 1.0
    assert only_a[0].tags == {"venue": "a"}
