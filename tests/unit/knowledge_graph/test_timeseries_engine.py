#!/usr/bin/python
"""Real-engine proof for native time-series (CONCEPT:KG-2.252).

Against the ACTUAL ephemeral engine (KG-2.238, pi-max tier so tsdb is served):

* the engine tsdb append/range/window/asof primitives return correct results;
* the EngineTimeSeriesBackend (KG-2.246, now LIVE) round-trips telemetry-shaped
  points;
* telemetry written through TokenUsageTracker.record is queryable via the native
  range/window path (the dead-layer → live wiring);
* finance gap-fill via engine_series matches the pandas LOCF result within tolerance.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.engine, pytest.mark.concept("KG-2.252")]


def test_engine_tsdb_range_window_asof(engine_graph):
    """append → range / window / asof on the raw engine tsdb client."""
    client = engine_graph._client
    sid = "tstest:px"
    # 4 points: t=0,1,2,3 seconds (ns), values 10,20,30,40.
    pts = [
        (0, [10.0]),
        (1_000_000_000, [20.0]),
        (2_000_000_000, [30.0]),
        (3_000_000_000, [40.0]),
    ]
    client.timeseries.append(sid, pts)

    # range [0, 4s) returns all 4 in order.
    got = client.timeseries.range(sid, 0, 4_000_000_000)
    assert [v[0] for _ts, v in got] == [10.0, 20.0, 30.0, 40.0]

    # window: 2-second buckets, mean → bucket[0..2s)=mean(10,20)=15, [2..4s)=mean(30,40)=35.
    bars = client.timeseries.window(
        sid, 0, 4_000_000_000, 2_000_000_000, "mean"
    )
    means = [v for _b, v, _c in bars]
    assert means == [15.0, 35.0]

    # asof at t=1.5s → the value as of (at-or-before) = 20.0.
    vals = client.timeseries.asof_join(sid, [1_500_000_000])
    assert vals == [20.0]


def test_engine_timeseries_backend_roundtrip(engine_graph):
    """The (now-live) EngineTimeSeriesBackend inserts + queries telemetry-shaped points."""
    from datetime import UTC, datetime

    from agent_utilities.knowledge_graph.memory.timeseries.base import (
        TimeSeriesDataPoint,
    )
    from agent_utilities.knowledge_graph.memory.timeseries.engine_backend import (
        EngineTimeSeriesBackend,
    )

    backend = EngineTimeSeriesBackend(client=engine_graph._client)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 1, 2, tzinfo=UTC)
    backend.insert(
        [
            TimeSeriesDataPoint("AGENT_A", t0, {"total_tokens": 100.0}),
            TimeSeriesDataPoint("AGENT_A", t1, {"total_tokens": 250.0}),
        ]
    )
    out = backend.query("AGENT_A", t0, t1)
    totals = sorted(p.metrics["total_tokens"] for p in out)
    assert totals == [100.0, 250.0]


def test_telemetry_record_to_tsdb_live(engine_graph):
    """TokenUsageTracker.record → engine tsdb → query_token_series reads it back."""
    from datetime import UTC, datetime

    from agent_utilities.knowledge_graph.memory.timeseries.engine_backend import (
        EngineTimeSeriesBackend,
    )
    from agent_utilities.observability.token_tracker import (
        TokenUsageRecord,
        TokenUsageTracker,
    )

    # Bind the tracker's tsdb backend to THIS test's engine tenant for isolation.
    tracker = TokenUsageTracker()
    tracker._ts_backend = EngineTimeSeriesBackend(client=engine_graph._client)

    base = 1_900_000_000.0  # fixed epoch seconds so the window math is deterministic
    tracker.record(
        TokenUsageRecord(
            agent_name="planner", prompt_tokens=10, response_tokens=5, timestamp=base
        )
    )
    tracker.record(
        TokenUsageRecord(
            agent_name="planner",
            prompt_tokens=20,
            response_tokens=10,
            timestamp=base + 1.0,
        )
    )

    # Read back through the SAME tenant-scoped backend (the record() side-effect wrote
    # the telemetry series into this graph). Proves the dead-layer → live wiring.
    out = tracker._ts_backend.query(
        tracker._series_id("planner"),
        datetime.fromtimestamp(base - 1, tz=UTC),
        datetime.fromtimestamp(base + 10, tz=UTC),
    )
    totals = sorted(p.metrics["total_tokens"] for p in out)
    assert totals == [15.0, 30.0]


def test_finance_gap_fill_parity(engine_graph):
    """engine gap-fill matches the pandas LOCF result within tolerance."""
    pd = pytest.importorskip("pandas")
    from agent_utilities.domains.finance.engine_series import gap_fill_series

    # Irregular daily series with a GAP (missing 2026-01-03): LOCF must carry 20 forward.
    idx = pd.to_datetime(
        ["2026-01-01", "2026-01-02", "2026-01-04"], utc=True
    )
    s = pd.Series([10.0, 20.0, 40.0], index=idx, name="close")

    filled = gap_fill_series(s, "1D", client=engine_graph._client)
    # Grid 01-01..01-04 → [10, 20, 20(carried), 40].
    vals = list(filled.to_numpy())
    assert vals[0] == pytest.approx(10.0)
    assert vals[1] == pytest.approx(20.0)
    assert vals[2] == pytest.approx(20.0)  # gap-filled (LOCF)
    assert vals[3] == pytest.approx(40.0)

    # Parity vs the pure-pandas LOCF reindex on the same grid.
    grid = pd.date_range(idx.min(), idx.max(), freq="1D", tz="UTC")
    pandas_locf = s.reindex(s.index.union(grid)).ffill().reindex(grid)
    assert list(filled.to_numpy()) == pytest.approx(list(pandas_locf.to_numpy()))
