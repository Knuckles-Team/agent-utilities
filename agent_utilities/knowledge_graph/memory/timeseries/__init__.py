"""Time-Series Memory Abstraction Layer.

Provides a unified interface for storing and querying high-frequency time-series data
(OHLCV, tick data, streaming signals) natively within the memory layer.

Supports pluggable backends:
- sqlite (default local embedded backend)
- timescale / influx (optional universal backends)
"""

from .base import TimeSeriesBackend
from .sqlite_backend import SQLiteTimeSeriesBackend


def get_timeseries_backend(backend_type: str = "sqlite", **kwargs) -> TimeSeriesBackend:
    if backend_type == "sqlite":
        return SQLiteTimeSeriesBackend(**kwargs)
    raise ValueError(f"Unknown timeseries backend: {backend_type}")
