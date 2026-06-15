"""Source-agnostic KG synchronization (CONCEPT:KG-2.9).

One sync mechanism for every external source registered in the hydration
``CAPABILITY_REGISTRY`` (LeanIX, Camunda, ARIS, ServiceNow, …), so they share a
single entrypoint, scheduler dispatch, and operational model instead of each
re-hydrating ad hoc:

* **Watermark poll (delta)** — the max ``updatedAt`` seen for a source is persisted
  on a per-source ``SourceSyncState`` node; the next run fetches only what changed.
  "Grab the delta on the next ingest" falls out for free.
* **Reconcile** — watermark deltas never surface deletions, so a reconcile compares
  the live id set with the KG's ``domain=<source>`` nodes and tombstones the gone.
* **Webhook narrowing** — a source's webhook can drive a sync for a specific set of
  ``ids`` (near-real-time).

Sources opt into **delta** by registering a handler in :data:`_DELTA_HANDLERS`
(LeanIX is the first). Any other registered source still syncs through this one
entrypoint — it just falls back to a **full hydrate** via the capability registry
until it grows a delta handler. This keeps the surface uniform while being honest
about which sources are incremental today.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

# Actions that route to source sync (used by the scheduler dispatch).
SYNC_ACTIONS = {"delta", "full", "reconcile"}


# ── Per-source watermark store (keyed by source) ─────────────────────────────


def _watermark_id(source: str) -> str:
    return f"sync:{source}"


def _read_watermark(backend: Any, source: str) -> str | None:
    if backend is None:
        return None
    try:
        rows = backend.execute(
            "MATCH (n:SourceSyncState {id: $id}) RETURN n.watermark AS w",
            {"id": _watermark_id(source)},
        )
        for r in rows or []:
            if isinstance(r, dict) and r.get("w"):
                return str(r["w"])
    except Exception:  # noqa: BLE001 - watermark is best-effort; full pull is safe
        logger.debug("%s watermark read failed", source, exc_info=True)
    return None


def _write_watermark(backend: Any, source: str, watermark: str | None) -> None:
    if backend is None or not watermark:
        return
    try:
        backend.execute(
            "MERGE (n:SourceSyncState {id: $id}) SET n.watermark = $wm, n.source = $src",
            {"id": _watermark_id(source), "wm": watermark, "src": source},
        )
    except Exception:  # noqa: BLE001
        logger.debug("%s watermark write failed", source, exc_info=True)


def _reconcile(engine: Any, source: str, live_ids: set[str]) -> dict[str, Any]:
    """Tombstone ``domain=<source>`` KG nodes whose external id is no longer live."""
    backend = getattr(engine, "backend", None)
    if not live_ids:
        return {"status": "skipped", "reason": "no live ids returned"}
    tombstoned = 0
    if backend is not None:
        try:
            rows = backend.execute(
                "MATCH (n) WHERE n.domain = $src AND n.externalToolId IS NOT NULL "
                "RETURN n.id AS id, n.externalToolId AS guid",
                {"src": source},
            )
            for r in rows or []:
                guid = r.get("guid") if isinstance(r, dict) else None
                if guid and guid not in live_ids:
                    backend.execute(
                        "MATCH (n {id: $id}) SET n.archived = true, "
                        "n.archivedReason = $reason",
                        {"id": r["id"], "reason": f"absent-from-{source}"},
                    )
                    tombstoned += 1
        except Exception:  # noqa: BLE001
            logger.debug("%s reconcile query failed", source, exc_info=True)
    return {"status": "completed", "live": len(live_ids), "tombstoned": tombstoned}


# ── LeanIX delta handler (the first delta-capable source) ────────────────────


def _sync_leanix(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    if client is None:
        from ...ecosystem.ea_clients import get_leanix_client

        client = get_leanix_client()
    if client is None:
        return {"status": "skipped", "reason": "no LeanIX client configured"}

    if mode == "reconcile":
        live: set[str] = set()
        getter = getattr(client, "fact_sheet_ids", None)
        if callable(getter):
            try:
                live = getter() or set()
            except Exception:  # noqa: BLE001
                live = set()
        return _reconcile(engine, "leanix", live)

    backend = getattr(engine, "backend", None)
    since = None if mode == "full" else _read_watermark(backend, "leanix")

    from ..enrichment.extractors.leanix import extract as leanix_extract

    batch = leanix_extract(SimpleNamespace(client=client, since=since, ids=ids))
    entities = [{"id": n.id, "type": n.type, **n.props} for n in batch.nodes]
    relationships = [
        {"source": e.source, "target": e.target, "type": e.rel_type, **e.props}
        for e in batch.edges
    ]
    if entities:
        engine.ingest_external_batch("leanix", entities, relationships)

    seen = [e["updatedAt"] for e in entities if e.get("updatedAt")]
    new_watermark = max(seen) if seen else None
    if new_watermark and (since is None or new_watermark > since):
        _write_watermark(backend, "leanix", new_watermark)

    return {
        "status": "ok",
        "source": "leanix",
        "mode": mode,
        "delta_capable": True,
        "nodes_hydrated": len(entities),
        "relations_hydrated": len(relationships),
        "since": since,
        "watermark": new_watermark or since,
    }


# Sources with a native delta (watermark/reconcile) handler. Add an entry here to
# make another source incremental (e.g. Camunda once its extractor takes `since`).
_DELTA_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "leanix": _sync_leanix,
}


def sync_source(
    engine: Any,
    source: str,
    *,
    mode: str = "delta",
    ids: list[str] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Sync one external source into the KG (the single entrypoint).

    ``mode`` ∈ {delta, full, reconcile}. Delta-capable sources (``_DELTA_HANDLERS``)
    do incremental watermark/reconcile; any other registered source falls back to a
    full hydrate via the capability registry.
    """
    source = (source or "").lower().strip()
    handler = _DELTA_HANDLERS.get(source)
    if handler is not None:
        return handler(engine, mode=mode, ids=ids, client=client)

    if mode == "reconcile":
        return {
            "status": "skipped",
            "reason": f"reconcile not supported for '{source}' (no delta handler)",
        }
    # Generic full hydrate via the standard capability registry.
    from .hydration import HydrationManager

    res = HydrationManager().hydrate_source(engine, source)
    if isinstance(res, dict):
        res.setdefault("source", source)
        res.setdefault("mode", "full")
        res.setdefault("delta_capable", False)
    return res
