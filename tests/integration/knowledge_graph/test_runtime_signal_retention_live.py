from __future__ import annotations

"""``:RuntimeSignal`` retention against the REAL engine (CONCEPT:AU-KG.memory.provides-real-ephemeral-one).

CONCEPT:AU-AHE.harness.runtime-reliability-loop.

These tests exist because the unit-level doubles could not have caught the defect that
let 15,327 ``:RuntimeSignal`` nodes accumulate under a nominal 2-hour TTL. The retention
statement was a ``DETACH DELETE`` issued through ``engine.query_cypher`` — the READ
chokepoint, which always reaches the engine with wire ``mode="read"``. The engine
re-parses every statement and rejects a declared-mode/parsed-statement mismatch BEFORE
execution (``src/server/handlers/query.rs::validate_cypher_mode`` against
``eg_query::classify_cypher``, which classifies ``DETACH DELETE`` as ``Write``), and the
caller wrapped the whole thing in ``contextlib.suppress(Exception)``. A mock that accepts
any string happily "deleted" the rows; only the real engine refuses.

So: one test pins the REFUSAL (the root cause, so a regression that routes retention back
through the read surface fails here), and one pins that the shipped sweep actually removes
expired rows through the write surface.
"""

import pytest

from agent_utilities.observability import runtime_signals

pytestmark = pytest.mark.concept("AU-AHE.harness.runtime-reliability-loop")


@pytest.fixture()
def signal_backend(engine_graph):
    """An ``EpistemicGraphBackend`` on the REAL ephemeral engine tenant."""
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )

    return EpistemicGraphBackend(graph_name=engine_graph.graph_name)


class _Engine:
    """The minimal ``engine`` shape the retention path uses: ``.backend`` + a read."""

    def __init__(self, backend) -> None:
        self.backend = backend

    def query_cypher(self, query: str, params: dict | None = None):
        """The READ chokepoint's contract: reads only, wire ``mode="read"``."""
        return self.backend.execute_read(query, params or {})


def _seed(backend, count: int, *, ts: float) -> None:
    label = runtime_signals.RUNTIME_SIGNAL_LABEL
    for i in range(count):
        sid = f"runtime:signal:test:{ts:.0f}-{i}"
        backend.add_node(sid, label, node_type=label, id=sid, ts=ts, subject=f"op-{i}")


def test_delete_through_the_read_surface_is_refused_by_the_engine(signal_backend):
    """THE ROOT CAUSE. The old retention statement cannot delete anything, ever."""
    label = runtime_signals.RUNTIME_SIGNAL_LABEL
    _seed(signal_backend, 2, ts=1.0)

    with pytest.raises(Exception) as excinfo:  # noqa: B017 - backend wraps the wire error
        signal_backend.execute_read(
            f"MATCH (n:{label}) WHERE n.ts < $cutoff DETACH DELETE n", {"cutoff": 2.0}
        )
    assert "mode" in str(excinfo.value).lower() or "write" in str(excinfo.value).lower()

    # And the rows are still there — which is exactly what `suppress(Exception)` hid.
    rows = signal_backend.execute_read(f"MATCH (n:{label}) RETURN n.id AS id LIMIT 10")
    assert len(rows) == 2


def test_retention_sweep_removes_expired_rows_on_the_real_engine(signal_backend):
    """THE FIX. Through the write-capable surface, expired rows actually go away."""
    label = runtime_signals.RUNTIME_SIGNAL_LABEL
    now = runtime_signals._now()
    _seed(signal_backend, 3, ts=now - 100_000.0)  # expired
    _seed(signal_backend, 2, ts=now)  # fresh

    engine = _Engine(signal_backend)
    assert (
        runtime_signals.count_expired_runtime_signals(engine, retention_s=7200.0) == 3
    )

    report = runtime_signals.prune_old_runtime_signals(
        engine, retention_s=7200.0, delete=True
    )

    assert report.deleted == 3
    assert report.truncated is False
    remaining = signal_backend.execute_read(
        f"MATCH (n:{label}) RETURN n.id AS id LIMIT 100"
    )
    assert len(remaining) == 2
    assert (
        runtime_signals.count_expired_runtime_signals(engine, retention_s=7200.0) == 0
    )


def test_window_read_is_bounded_by_the_cutoff_on_the_real_engine(signal_backend):
    """The window predicate must be evaluated BY THE ENGINE, not only in Python."""
    now = runtime_signals._now()
    _seed(signal_backend, 4, ts=now - 100_000.0)  # outside the window
    _seed(signal_backend, 1, ts=now)  # inside

    engine = _Engine(signal_backend)
    recent = runtime_signals.read_recent_runtime_signals(
        engine, window_s=900.0, limit=100
    )
    assert len(recent) == 1
