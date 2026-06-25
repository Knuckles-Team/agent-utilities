"""Time-Series Memory Abstraction Layer.

Provides a unified interface for storing and querying high-frequency time-series data
(OHLCV, tick data, streaming signals) natively within the memory layer.

Engine-only (CONCEPT:KG-2.246): time-series points live on the **one
epistemic-graph engine authority** via its native ``client.timeseries.*`` tsdb
(eg-tsdb, CONCEPT:KG-2.210/211) — each series stored as ``(ts_ns, field-vector)``
points in the engine's durable ``series.redb``, beside the graph. There is NO
local SQLite fallback: the OS-5.63 resolver auto-starts the pi-tier engine in
prod and the test fixture (CONCEPT:KG-2.238) provides a real ephemeral one, so an
unreachable engine is a hard error, not a silent degrade to a straggler local DB.
"""

from .base import TimeSeriesBackend
from .engine_backend import EngineTimeSeriesBackend


def get_timeseries_backend(backend_type: str = "engine", **kwargs) -> TimeSeriesBackend:
    """Return the engine-backed time-series backend.

    CONCEPT:KG-2.246 — there is one backend: the epistemic-graph engine tsdb. It
    ``initialize()``s eagerly, raising a clear error when the engine is genuinely
    unreachable (no SQLite fallback). ``backend_type`` is accepted only as
    ``"engine"`` (the default) for call-site clarity; any other value is rejected.
    """
    if backend_type != "engine":
        raise ValueError(
            f"Unknown timeseries backend {backend_type!r}: the only backend is the "
            "engine-backed tsdb ('engine'). The local SQLite fallback was removed "
            "(CONCEPT:KG-2.246) — time-series lives on the one engine authority."
        )
    be = EngineTimeSeriesBackend(**kwargs)
    be.initialize()
    return be
