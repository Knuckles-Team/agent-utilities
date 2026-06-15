"""LeanIX delta synchronization (CONCEPT:KG-2.9).

Keeps the KG mirror in step with LeanIX without a full re-ingest each time:

* **Watermark poll** — the max ``updatedAt`` seen is persisted on a singleton
  ``LeanixSyncState`` node; the next run (scheduled, or the operator's next
  manual sync) fetches only fact sheets modified since. "Grab the delta on the
  next ingest" falls out for free.
* **Webhook narrowing** — a LeanIX webhook can drive a sync for a specific set of
  ``ids`` for near-real-time updates.
* **Reconcile** — watermark deltas never surface deletions, so a periodic full
  reconcile compares the live fact sheet id set with the KG's ``domain="leanix"``
  nodes and tombstones the ones that disappeared.

All three run through the one extractor + ``ingest_external_batch`` path and the
one :class:`ea_clients.LeanixEAClient`.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

_WATERMARK_NODE_ID = "leanix:sync:state"


def _read_watermark(backend: Any) -> str | None:
    if backend is None:
        return None
    try:
        rows = backend.execute(
            "MATCH (n:LeanixSyncState {id: $id}) RETURN n.watermark AS w",
            {"id": _WATERMARK_NODE_ID},
        )
        for r in rows or []:
            if isinstance(r, dict) and r.get("w"):
                return str(r["w"])
    except Exception:  # noqa: BLE001 - watermark is best-effort; full pull is safe
        logger.debug("LeanIX watermark read failed", exc_info=True)
    return None


def _write_watermark(backend: Any, watermark: str | None) -> None:
    if backend is None or not watermark:
        return
    try:
        backend.execute(
            "MERGE (n:LeanixSyncState {id: $id}) SET n.watermark = $wm",
            {"id": _WATERMARK_NODE_ID, "wm": watermark},
        )
    except Exception:  # noqa: BLE001
        logger.debug("LeanIX watermark write failed", exc_info=True)


def reconcile_leanix(engine: Any, client: Any) -> dict[str, Any]:
    """Tombstone KG fact-sheet nodes that no longer exist in LeanIX."""
    backend = getattr(engine, "backend", None)
    live: set[str] = set()
    getter = getattr(client, "fact_sheet_ids", None)
    if callable(getter):
        try:
            live = getter() or set()
        except Exception:  # noqa: BLE001
            live = set()
    if not live:
        return {"status": "skipped", "reason": "no live fact sheet ids returned"}

    tombstoned = 0
    if backend is not None:
        try:
            rows = backend.execute(
                "MATCH (n) WHERE n.domain = 'leanix' AND n.externalToolId IS NOT NULL "
                "RETURN n.id AS id, n.externalToolId AS guid"
            )
            for r in rows or []:
                guid = r.get("guid") if isinstance(r, dict) else None
                if guid and guid not in live:
                    backend.execute(
                        "MATCH (n {id: $id}) SET n.archived = true, "
                        "n.archivedReason = 'absent-from-leanix'",
                        {"id": r["id"]},
                    )
                    tombstoned += 1
        except Exception:  # noqa: BLE001
            logger.debug("LeanIX reconcile query failed", exc_info=True)
    return {"status": "completed", "live": len(live), "tombstoned": tombstoned}


def sync_leanix(
    engine: Any,
    *,
    mode: str = "delta",
    ids: list[str] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Run a LeanIX sync. ``mode`` ∈ {delta, full, reconcile}; ``ids`` narrows (webhook)."""
    if client is None:
        from ...ecosystem.ea_clients import get_leanix_client

        client = get_leanix_client()
    if client is None:
        return {"status": "skipped", "reason": "no LeanIX client configured"}

    if mode == "reconcile":
        return reconcile_leanix(engine, client)

    backend = getattr(engine, "backend", None)
    since = None if mode == "full" else _read_watermark(backend)

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
        _write_watermark(backend, new_watermark)

    return {
        "status": "ok",
        "mode": mode,
        "nodes_hydrated": len(entities),
        "relations_hydrated": len(relationships),
        "since": since,
        "watermark": new_watermark or since,
    }
