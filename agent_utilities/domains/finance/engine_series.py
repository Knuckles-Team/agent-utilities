"""Engine-native time-series alignment for finance (CONCEPT:AU-KG.domains.ohlcv-gap-fill).

The finance feature math (rolling/ewm/shift in ``features.py`` / ``alpha_factors.py``)
operates on *regular, aligned* series where vectorized pandas is already optimal — and
the engine guide is explicit that tight per-element math stays in-process; only a
*batch* that amortizes the socket round-trip should go to the engine. So those are
deliberately left in pandas.

What pandas does NOT do natively, and where the engine's native tsdb IS the clear win,
is the *irregular*-series primitives: **gap-fill** onto a fixed grid (LOCF), **ASOF**
alignment of one series to another's timestamps, and **time-bucketed** aggregation —
all in-engine over ``client.timeseries.*`` (CONCEPT:AU-KG.retrieval.god-nodes-communities/211), needing no
DataFusion. This module routes exactly those, keeping the public feature API in pandas.

A throwaway series is staged in the engine tsdb, the primitive runs server-side, and
the result returns as a pandas object — one round-trip per primitive (a batch), never
per row.

``gap_fill_series`` treats naive input timestamps as UTC and converts timezone-aware
timestamps to UTC instants before either the engine or pandas route runs. This keeps
DST gaps and folds unambiguous and makes the returned regular grid identical between
routes. Legacy pandas interval suffixes ``H`` and ``T`` are normalized to ``h`` and
``min`` before pandas parses them.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import UTC
from numbers import Integral

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover - finance extra not installed
    raise ImportError(
        "Finance extra dependencies missing. Please install agent-utilities[finance]"
    ) from e

from agent_utilities.domains.finance.errors import InvalidIntervalError

logger = logging.getLogger(__name__)
_LEGACY_INTERVAL_ALIAS = re.compile(r"(?P<count>\d*)(?P<unit>[HT])\Z")


def _client():
    """A non-owning process-engine view, or ``None`` when unreachable."""
    try:
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )

        return GraphComputeEngine.get_or_create().client
    except Exception as e:  # noqa: BLE001 — engine client construction, caller treats None as "use local fallback"
        logger.debug(
            "[CONCEPT:AU-KG.domains.ohlcv-gap-fill] engine unavailable for series op: %s",
            e,
        )
        return None


def _to_ns(index: pd.Index) -> list[int]:
    ts = pd.to_datetime(index, utc=True)
    return [int(v.value) for v in ts]  # pandas Timestamp.value is ns since epoch


def _normalize_step(step: str | int | pd.Timedelta) -> pd.Timedelta:
    """Return a canonical, strictly positive interval.

    Raises:
        InvalidIntervalError: ``step`` is empty, malformed, zero, or negative.
        Checked here — before ``gap_fill_series``/``asof_align`` touch either
        the engine or the pandas fallback — so every caller sees the same
        typed, documented error at the API boundary instead of a downstream
        ``ZeroDivisionError`` from a zero-frequency grid or a bare pandas
        parser ``ValueError`` (D-CDX-96).
    """
    original = step
    if isinstance(step, str):
        legacy_alias = _LEGACY_INTERVAL_ALIAS.fullmatch(step)
        if legacy_alias is not None:
            count = legacy_alias.group("count") or "1"
            unit = "h" if legacy_alias.group("unit") == "H" else "min"
            step = f"{count}{unit}"
        # The vector parser preserves the unit encoded in a frequency string and
        # avoids pandas' scalar path through NumPy's generic timedelta unit.
        try:
            normalized = pd.to_timedelta([step])[0]
        except ValueError as e:
            raise InvalidIntervalError(
                f"Invalid gap-fill interval {original!r}: not a recognized "
                "pandas frequency/interval string"
            ) from e
    elif isinstance(step, Integral) and not isinstance(step, bool):
        # ``pd.Timedelta(int)`` historically means nanoseconds; keep that contract
        # explicit so newer NumPy versions never infer the deprecated generic unit.
        normalized = pd.Timedelta(int(step), unit="ns")
    else:
        normalized = pd.Timedelta(step)
    if pd.isna(normalized) or normalized <= pd.Timedelta(0):
        raise InvalidIntervalError(
            f"Invalid gap-fill interval {original!r}: must resolve to a "
            f"positive duration, got {normalized!r}"
        )
    return normalized


def _utc_series(series: pd.Series) -> pd.Series:
    """Return a non-mutating UTC-indexed view for both gap-fill implementations."""
    # ``pd.to_datetime(..., utc=True)`` rebuilds even an already canonical UTC index.
    # Neither gap-fill route mutates its input, so retaining that view avoids an
    # otherwise dominant O(n) conversion on the common UTC market-data path.
    if isinstance(series.index, pd.DatetimeIndex) and series.index.tz == UTC:
        return series
    normalized = series.copy(deep=False)
    normalized.index = pd.to_datetime(series.index, utc=True)
    return normalized


def _pandas_gap_fill(series: pd.Series, step: pd.Timedelta) -> pd.Series:
    """Run the UTC LOCF fallback shared by unavailable and failed engine paths."""
    grid = pd.date_range(series.index.min(), series.index.max(), freq=step, tz="UTC")
    return series.reindex(series.index.union(grid)).ffill().reindex(grid)


def gap_fill_series(
    series: pd.Series, step: str | int | pd.Timedelta = "1D", *, client=None
) -> pd.Series:
    """LOCF gap-fill ``series`` onto a fixed ``step`` grid, computed IN-ENGINE.

    The engine's ``timeseries.gap_fill`` carries the last observation forward on a
    regular grid (the clear win over hand-rolled pandas reindex+ffill on irregular
    input). ``step`` accepts a pandas frequency string, an integer nanosecond interval,
    or a pandas ``Timedelta``. Legacy ``H`` and ``T`` suffixes are normalized to
    ``h`` and ``min`` before pandas parses them. Naive timestamps are interpreted as
    UTC; timezone-aware inputs are converted to UTC instants, and both routes return
    a UTC-indexed Series. The function falls back to pandas reindex+ffill only when
    no engine is reachable (so callers always get a result).
    """
    normalized_step = _normalize_step(step)
    series = _utc_series(series)
    if series.empty:
        return series
    client = client or _client()
    if client is None:
        # No engine — degrade to the pandas equivalent so the caller still works.
        return _pandas_gap_fill(series, normalized_step)
    sid = f"finseries:{uuid.uuid4().hex}"
    try:
        ns = _to_ns(series.index)
        client.timeseries.append(
            sid, [(t, [float(v)]) for t, v in zip(ns, series.to_numpy(), strict=False)]
        )
        step_ns = int(normalized_step.value)
        rows = client.timeseries.gap_fill(sid, ns[0], ns[-1] + 1, step_ns)
        idx = pd.to_datetime([t for t, _v, _f in rows], utc=True)
        vals = [v for _t, v, _f in rows]
        return pd.Series(vals, index=idx, name=series.name)
    except Exception as e:  # noqa: BLE001 — engine-accelerated path, equivalent pandas-only fallback follows
        logger.debug(
            "[CONCEPT:AU-KG.domains.ohlcv-gap-fill] gap_fill_series engine path failed: %s",
            e,
        )
        return _pandas_gap_fill(series, normalized_step)


def asof_align(series: pd.Series, at: pd.Index, *, client=None) -> pd.Series:
    """ASOF-align ``series`` to the timestamps ``at`` (nearest at-or-before), IN-ENGINE.

    For each timestamp in ``at``, the engine returns ``series``'s value as of that
    time (``timeseries.asof_join``) — the native point-in-time join pandas only does
    via the heavier ``merge_asof``. Returns a Series indexed by ``at``; falls back to
    pandas ``reindex(method='ffill')`` with no engine.
    """
    if series.empty:
        return pd.Series(index=at, dtype=float, name=series.name)
    client = client or _client()
    if client is None:
        return series.reindex(series.index.union(at)).ffill().reindex(at)
    sid = f"finseries:{uuid.uuid4().hex}"
    try:
        ns = _to_ns(series.index)
        client.timeseries.append(
            sid, [(t, [float(v)]) for t, v in zip(ns, series.to_numpy(), strict=False)]
        )
        at_ns = _to_ns(at)
        vals = client.timeseries.asof_join(sid, at_ns)
        return pd.Series(vals, index=at, name=series.name)
    except Exception as e:  # noqa: BLE001 — engine-accelerated path, equivalent pandas-only fallback follows
        logger.debug(
            "[CONCEPT:AU-KG.domains.ohlcv-gap-fill] asof_align engine path failed: %s",
            e,
        )
        return series.reindex(series.index.union(at)).ffill().reindex(at)
