"""Time-Series Memory Abstraction Layer.

Provides a unified interface for storing and querying high-frequency time-series data
(OHLCV, tick data, streaming signals) natively within the memory layer.

Supports pluggable backends (dual-mode like ``DeltaManifest`` — CONCEPT:KG-2.206):
- ``engine`` — the epistemic-graph engine's native ``client.timeseries.*`` tsdb
  (the one durable authority; points live beside the graph in ``series.redb``).
- ``sqlite`` — the zero-infra local embedded fallback (the ``tiny`` profile).
- ``auto`` (default) — engine when reachable, else the SQLite fallback.
"""

import logging

from .base import TimeSeriesBackend
from .engine_backend import EngineTimeSeriesBackend
from .sqlite_backend import SQLiteTimeSeriesBackend

logger = logging.getLogger(__name__)


def get_timeseries_backend(backend_type: str = "auto", **kwargs) -> TimeSeriesBackend:
    """Return a time-series backend.

    ``backend_type``:
        * ``"engine"`` — force the epistemic-graph tsdb backend (raises if the
          engine is unreachable).
        * ``"sqlite"`` — force the local embedded SQLite fallback.
        * ``"auto"`` (default) — try the engine; on any failure, transparently
          degrade to SQLite so the ``tiny`` zero-infra profile still works.
    """
    if backend_type == "engine":
        be = EngineTimeSeriesBackend(**kwargs)
        be.initialize()
        return be
    if backend_type == "sqlite":
        return SQLiteTimeSeriesBackend(**kwargs)
    if backend_type == "auto":
        try:
            be = EngineTimeSeriesBackend(**kwargs)
            be.initialize()
            return be
        except Exception as e:  # noqa: BLE001 - degrade to zero-infra SQLite
            logger.debug(
                "timeseries: engine backend unavailable (%s); using SQLite fallback",
                e,
            )
            return SQLiteTimeSeriesBackend(**kwargs)
    raise ValueError(f"Unknown timeseries backend: {backend_type}")
