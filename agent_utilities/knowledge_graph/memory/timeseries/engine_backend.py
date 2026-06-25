"""Engine-backed time-series memory backend (CONCEPT:KG-2.206).

Routes the time-series memory abstraction onto the **one epistemic-graph engine
authority** via its native ``client.timeseries.*`` namespace (eg-tsdb, CONCEPT:
KG-2.210/211) instead of a local ``timeseries.db`` SQLite file. The engine stores
each series as ``(ts_ns, [field0, field1, ...])`` points in its own durable
``series.redb``, so high-frequency points live beside the graph, not in a
straggler local DB.

Dual-mode like ``DeltaManifest``: the factory selects this engine arm when a
durable engine is reachable; otherwise the zero-infra ``SQLiteTimeSeriesBackend``
remains the ``tiny`` fallback.

Mapping from the abstraction's ``TimeSeriesDataPoint`` (``symbol`` + ``datetime``
+ ``metrics: {name: float}`` + optional string ``tags``) onto the engine's
field-vector model:

* A point's ``metrics`` become the ordered field vector. Field order is fixed per
  series at registration (``field_names``) so reads decode back to the same names.
* ``tags`` (string key/values) cannot live in a numeric field vector, so each
  distinct ``(symbol, frozenset(tags))`` maps to its own ``series_id``; the tag map
  + field names are recorded on the ``:Series`` registry node. A tag-filtered query
  resolves the matching series ids and unions their points.
* ``datetime`` <-> integer nanoseconds since epoch.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from .base import TimeSeriesBackend, TimeSeriesDataPoint

logger = logging.getLogger(__name__)


def _to_ns(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _from_ns(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc)


def _series_id(symbol: str, tags: dict[str, str] | None) -> str:
    """Stable series id for a ``(symbol, tags)`` pair.

    Tags are folded into the id (sorted, hashed) so distinct tag-sets get distinct
    engine series while the same tag-set always resolves to the same id.
    """
    if not tags:
        return f"ts:{symbol}"
    canon = json.dumps(tags, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]  # noqa: S324
    return f"ts:{symbol}:{digest}"


class EngineTimeSeriesBackend(TimeSeriesBackend):
    """Time-series backend served by the epistemic-graph engine's tsdb namespace.

    CONCEPT:KG-2.206. Best-effort acquisition of a ``SyncEpistemicGraphClient`` at
    ``initialize()``; the factory only selects this arm when an engine is reachable,
    so this raises if the client cannot be built (the factory degrades to SQLite).
    """

    def __init__(self, client: Any = None):
        self._client = client
        # series_id -> ordered metric field names (decode key for reads)
        self._fields: dict[str, list[str]] = {}

    def initialize(self) -> None:
        if self._client is not None:
            return
        from epistemic_graph.client import SyncEpistemicGraphClient

        from agent_utilities.knowledge_graph.core.engine_resolver import (
            client_connect_kwargs,
        )

        self._client = SyncEpistemicGraphClient.connect(**client_connect_kwargs())
        logger.debug("EngineTimeSeriesBackend initialized via engine tsdb")

    def _ensure_client(self) -> Any:
        if self._client is None:
            self.initialize()
        return self._client

    def _register(
        self,
        series_id: str,
        symbol: str,
        fields: list[str],
        tags: dict[str, str] | None,
    ) -> None:
        """Register the series + its field-name decode key (idempotent)."""
        client = self._ensure_client()
        self._fields[series_id] = fields
        client.timeseries.register_series(
            series_id,
            entity_id=f"symbol:{symbol}",
            field_names=fields,
            metadata={
                "symbol": symbol,
                "tags_json": json.dumps(tags) if tags else "",
            },
        )

    def _field_names_for(self, series_id: str) -> list[str]:
        """Resolve the ordered field names for a series (cache, else registry node)."""
        if series_id in self._fields:
            return self._fields[series_id]
        client = self._ensure_client()
        props = client.nodes.properties(f"series:{series_id}") or {}
        fields = list(props.get("field_names") or [])
        self._fields[series_id] = fields
        return fields

    def insert(self, points: list[TimeSeriesDataPoint]) -> None:
        if not points:
            return
        client = self._ensure_client()
        # Group points by (symbol, tags) -> series, with a stable metric-field order.
        by_series: dict[str, list[tuple[int, list[float]]]] = {}
        meta: dict[str, tuple[str, list[str], dict[str, str] | None]] = {}
        for p in points:
            fields = sorted(p.metrics.keys())
            sid = _series_id(p.symbol, p.tags)
            known = self._fields.get(sid)
            if known is not None:
                fields = known + [f for f in fields if f not in known]
            meta[sid] = (p.symbol, fields, p.tags)
            vec = [float(p.metrics.get(f, 0.0)) for f in fields]
            by_series.setdefault(sid, []).append((_to_ns(p.timestamp), vec))
        for sid, (symbol, fields, tags) in meta.items():
            self._register(sid, symbol, fields, tags)
        for sid, batch in by_series.items():
            client.timeseries.append(sid, batch, field_names=self._fields.get(sid))

    def query(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        tags: dict[str, str] | None = None,
    ) -> list[TimeSeriesDataPoint]:
        client = self._ensure_client()
        from_ns, to_ns = _to_ns(start_time), _to_ns(end_time) + 1
        if tags:
            series_ids = [_series_id(symbol, tags)]
        else:
            series_ids = self._series_for_symbol(symbol)
        out: list[TimeSeriesDataPoint] = []
        for sid in series_ids:
            fields = self._field_names_for(sid)
            sid_tags = self._tags_for(sid)
            rows = client.timeseries.range(sid, from_ns, to_ns)
            for ts, vals in rows:
                metrics = {
                    name: vals[i] for i, name in enumerate(fields) if i < len(vals)
                }
                out.append(
                    TimeSeriesDataPoint(
                        symbol=symbol,
                        timestamp=_from_ns(ts),
                        metrics=metrics,
                        tags=sid_tags,
                    )
                )
        out.sort(key=lambda p: p.timestamp)
        return out

    def _series_for_symbol(self, symbol: str) -> list[str]:
        """All registered series ids for a symbol (engine registry query)."""
        client = self._ensure_client()
        try:
            rows = client.query.cypher(
                f"MATCH (s:Series) WHERE s.symbol = '{symbol}' "
                "RETURN s.series_id AS series_id"
            )
            ids = [r["series_id"] for r in rows if r.get("series_id")]
            if ids:
                return ids
        except Exception as e:  # noqa: BLE001 - registry query best-effort
            logger.debug("series-for-symbol query failed: %s", e)
        local = [sid for sid in self._fields if sid.startswith(f"ts:{symbol}")]
        return local or [_series_id(symbol, None)]

    def _tags_for(self, series_id: str) -> dict[str, str] | None:
        client = self._ensure_client()
        props = client.nodes.properties(f"series:{series_id}") or {}
        raw = props.get("tags_json")
        if raw:
            try:
                return json.loads(raw)
            except (ValueError, json.JSONDecodeError):
                return None
        return None

    def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
