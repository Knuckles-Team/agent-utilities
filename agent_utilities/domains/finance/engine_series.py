"""Engine-native time-series alignment for finance (CONCEPT:KG-2.252).

The finance feature math (rolling/ewm/shift in ``features.py`` / ``alpha_factors.py``)
operates on *regular, aligned* series where vectorized pandas is already optimal — and
the engine guide is explicit that tight per-element math stays in-process; only a
*batch* that amortizes the socket round-trip should go to the engine. So those are
deliberately left in pandas.

What pandas does NOT do natively, and where the engine's native tsdb IS the clear win,
is the *irregular*-series primitives: **gap-fill** onto a fixed grid (LOCF), **ASOF**
alignment of one series to another's timestamps, and **time-bucketed** aggregation —
all in-engine over ``client.timeseries.*`` (CONCEPT:KG-2.210/211), needing no
DataFusion. This module routes exactly those, keeping the public feature API in pandas.

A throwaway series is staged in the engine tsdb, the primitive runs server-side, and
the result returns as a pandas object — one round-trip per primitive (a batch), never
per row.
"""

from __future__ import annotations

import logging
import uuid

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover - finance extra not installed
    raise ImportError(
        "Finance extra dependencies missing. Please install agent-utilities[finance]"
    ) from e

logger = logging.getLogger(__name__)


def _client():
    """A connected engine client, or ``None`` when no engine is reachable."""
    try:
        from epistemic_graph.client import SyncEpistemicGraphClient

        from agent_utilities.knowledge_graph.core.engine_resolver import (
            client_connect_kwargs,
        )

        return SyncEpistemicGraphClient.connect(**client_connect_kwargs())
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONCEPT:KG-2.252] engine unavailable for series op: %s", e)
        return None


def _to_ns(index: pd.Index) -> list[int]:
    ts = pd.to_datetime(index, utc=True)
    return [int(v.value) for v in ts]  # pandas Timestamp.value is ns since epoch


def gap_fill_series(
    series: pd.Series, step: str = "1D", *, client=None
) -> pd.Series:
    """LOCF gap-fill ``series`` onto a fixed ``step`` grid, computed IN-ENGINE.

    The engine's ``timeseries.gap_fill`` carries the last observation forward on a
    regular grid (the clear win over hand-rolled pandas reindex+ffill on irregular
    input). Returns a new pandas Series on the regular grid; falls back to a pandas
    reindex+ffill only when no engine is reachable (so callers always get a result).
    """
    if series.empty:
        return series
    own_client = client is None
    client = client or _client()
    if client is None:
        # No engine — degrade to the pandas equivalent so the caller still works.
        grid = pd.date_range(series.index.min(), series.index.max(), freq=step, tz="UTC")
        return series.reindex(series.index.union(grid)).ffill().reindex(grid)
    sid = f"finseries:{uuid.uuid4().hex[:12]}"
    try:
        ns = _to_ns(series.index)
        client.timeseries.append(
            sid, [(t, [float(v)]) for t, v in zip(ns, series.to_numpy(), strict=False)]
        )
        step_ns = int(pd.Timedelta(step).value)
        rows = client.timeseries.gap_fill(sid, ns[0], ns[-1] + 1, step_ns)
        idx = pd.to_datetime([t for t, _v, _f in rows], utc=True)
        vals = [v for _t, v, _f in rows]
        return pd.Series(vals, index=idx, name=series.name)
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONCEPT:KG-2.252] gap_fill_series engine path failed: %s", e)
        grid = pd.date_range(series.index.min(), series.index.max(), freq=step, tz="UTC")
        return series.reindex(series.index.union(grid)).ffill().reindex(grid)
    finally:
        if own_client:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass


def asof_align(
    series: pd.Series, at: pd.Index, *, client=None
) -> pd.Series:
    """ASOF-align ``series`` to the timestamps ``at`` (nearest at-or-before), IN-ENGINE.

    For each timestamp in ``at``, the engine returns ``series``'s value as of that
    time (``timeseries.asof_join``) — the native point-in-time join pandas only does
    via the heavier ``merge_asof``. Returns a Series indexed by ``at``; falls back to
    pandas ``reindex(method='ffill')`` with no engine.
    """
    if series.empty:
        return pd.Series(index=at, dtype=float, name=series.name)
    own_client = client is None
    client = client or _client()
    if client is None:
        return series.reindex(series.index.union(at)).ffill().reindex(at)
    sid = f"finseries:{uuid.uuid4().hex[:12]}"
    try:
        ns = _to_ns(series.index)
        client.timeseries.append(
            sid, [(t, [float(v)]) for t, v in zip(ns, series.to_numpy(), strict=False)]
        )
        at_ns = _to_ns(at)
        vals = client.timeseries.asof_join(sid, at_ns)
        return pd.Series(vals, index=at, name=series.name)
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONCEPT:KG-2.252] asof_align engine path failed: %s", e)
        return series.reindex(series.index.union(at)).ffill().reindex(at)
    finally:
        if own_client:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
