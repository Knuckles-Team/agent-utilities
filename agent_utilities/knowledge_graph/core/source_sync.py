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


def _sync_archivebox(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest preserved ArchiveBox snapshots into the KG (CONCEPT:KG-2.7).

    Enumerates snapshots via the ``archivebox`` mcp_tool source preset (delta =
    ``created_at__gte`` watermark; "pull all" = ``mode='full'``; ``ids`` selects
    specific snapshots), then ingests each archived URL through the unified
    ``DOCUMENT`` path — so the body is retrieved robustly via
    ``web_fetch.resolve_web_fetch`` (ArchiveBox-preferred when configured) and a
    research-roundup snapshot also auto-acquires the papers it cites (Phase 2).
    """
    from ...core.config import setting

    if not (setting("ARCHIVEBOX_URL", default="") or "").strip():
        return {"status": "skipped", "reason": "ARCHIVEBOX_URL not configured"}

    backend = getattr(engine, "backend", None)
    since = None if mode == "full" else _read_watermark(backend, "archivebox")

    from ...protocols.source_connectors.connectors.mcp_package import _run_async
    from ...protocols.source_connectors.registry import build_connector
    from ..ingestion.engine import ContentType, IngestionEngine, IngestionManifest

    params: dict[str, Any] = {}
    if since:
        params["created_at__gte"] = since
    if ids:
        params["id"] = ",".join(ids)
    config: dict[str, Any] = {"preset": "archivebox"}
    if params:
        config["params"] = params
    conn = build_connector("mcp_tool", config)
    docs = list(conn.poll_all()) if hasattr(conn, "poll_all") else list(conn.load())  # type: ignore[attr-defined]

    urls = [
        u for d in docs if (u := (d.metadata or {}).get("url") or "").startswith("http")
    ]
    manifests = [
        IngestionManifest(
            content_type=ContentType.DOCUMENT,
            source_uri=u,
            metadata={"source_system": "archivebox"},
        )
        for u in urls
    ]
    ingested = 0
    if manifests:
        engine_ie = IngestionEngine(kg_engine=engine)
        results = _run_async(engine_ie.ingest_batch(manifests))
        ingested = sum(1 for r in results if r and r.status == "success")

    seen = [d.updated_at for d in docs if d.updated_at]
    new_watermark = max(seen) if seen else None
    if new_watermark and (since is None or new_watermark > since):
        _write_watermark(backend, "archivebox", new_watermark)

    return {
        "status": "ok",
        "source": "archivebox",
        "mode": mode,
        "delta_capable": True,
        "snapshots_seen": len(docs),
        "documents_ingested": ingested,
        "since": since,
        "watermark": new_watermark or since,
    }


def _as_epoch(value: Any) -> int | None:
    """Best-effort parse of a watermark value to int unix-seconds."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _sync_freshrss(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Relevance-gated ingestion of curated FreshRSS items (CONCEPT:KG-2.115).

    Enumerates items via the ``freshrss`` mcp_tool preset over the Google-Reader API
    (delta = ``newer_than`` → GReader ``ot`` **unix-seconds** watermark on
    ``published``; ``mode='full'`` drains all). Unlike a mirror connector this source
    is INTENTIONALLY GATED: each item passes the world-model relevance gate
    (:class:`WorldModelPipelineRunner`, CONCEPT:KG-2.116) — only items relevant to the
    existing KG (taxonomy score OR concept-novelty) or agent-force flagged are fully
    ingested as ``news_article`` Documents; the rest get a marginal footprint or are
    skipped. Research/arXiv-feed items route to the research path (CONCEPT:KG-2.117),
    unifying RSS intake. ``skipped_unchanged`` plus the watermark prove the delta on a
    re-run; the write-layer content-hash delta (KG_WRITE_DELTA) is the second guard.
    """
    from ...core.config import setting

    if not (setting("FRESHRSS_URL", default="") or "").strip():
        return {"status": "skipped", "reason": "FRESHRSS_URL not configured"}

    backend = getattr(engine, "backend", None)
    since = None if mode == "full" else _read_watermark(backend, "freshrss")

    from ...automation.worldmodel_pipeline import WorldModelPipelineRunner
    from ...protocols.source_connectors.registry import build_connector

    params: dict[str, Any] = {}
    if since and (since_epoch := _as_epoch(since)) is not None:
        params["newer_than"] = since_epoch  # GReader ``ot`` — unix seconds
    config: dict[str, Any] = {"preset": "freshrss"}
    if params:
        config["params"] = params
    conn = build_connector("mcp_tool", config)
    docs = list(conn.poll_all()) if hasattr(conn, "poll_all") else list(conn.load())  # type: ignore[attr-defined]

    report = WorldModelPipelineRunner(engine=engine).run_gated_ingest(docs)

    seen = [e for d in docs if (e := _as_epoch(d.updated_at)) is not None]
    new_watermark = str(max(seen)) if seen else None
    since_epoch = _as_epoch(since) if since else None
    if new_watermark and (since_epoch is None or int(new_watermark) > since_epoch):
        _write_watermark(backend, "freshrss", new_watermark)

    return {
        "status": "ok",
        "source": "freshrss",
        "mode": mode,
        "delta_capable": True,
        "items_seen": len(docs),
        "ingested": report.ingested,
        "relevant": report.relevant,
        "marginal": report.marginal,
        "research": report.research,
        "skipped_unchanged": report.skipped,
        "since": since,
        "watermark": new_watermark or since,
    }


def _sync_gitlab(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Index whole GitLab instances as a resolved code graph (CONCEPT:KG-2.9g).

    For every configured instance (``GITLAB_INSTANCES`` JSON, else single
    ``GITLAB_URL``/``GITLAB_TOKEN``) this enumerates projects → default-branch code
    files and ships each project to the engine's ``index_repository`` resolver
    (CONCEPT:KG-2.8r), writing ``:Code`` symbols + resolved ``calls``/``depends_on``
    + ``Repository``/``File`` structure under ``source_system = gitlab:<instance>``.
    ``mode='full'`` re-indexes all; delta uses a per-instance ``last_activity_at``
    watermark; ``ids`` narrows to specific projects (webhook delta).
    """
    from .gitlab_indexer import (
        GitLabRestSource,
        GitLabSource,
        index_instance,
        instances_from_config,
    )

    graph_compute = getattr(engine, "graph_compute", None)
    if graph_compute is None or not getattr(
        graph_compute, "supports_index_repository", False
    ):
        return {
            "status": "skipped",
            "reason": "engine does not advertise IndexRepository (rebuild with the resolver)",
        }

    # An injected `client` is an explicit single-source override (tests / a caller
    # supplying its own GitLabSource): use one sentinel instance, ignore config.
    instances = [None] if client is not None else instances_from_config()  # type: ignore[list-item]
    if not instances:
        return {"status": "skipped", "reason": "no GitLab instance configured"}

    backend = getattr(engine, "backend", None)
    project_ids = {str(i) for i in ids} if ids else None

    results: list[dict[str, Any]] = []
    for inst in instances:
        name = inst.name if inst is not None else "gitlab"
        wm_key = f"gitlab:{name}"
        since = None if mode == "full" else _read_watermark(backend, wm_key)
        # `inst is None` only occurs on the injected-client override path (above),
        # so a real instance always pairs with the REST source.
        if client is not None:
            source: GitLabSource = client
        else:
            assert inst is not None
            source = GitLabRestSource(inst)
        summary = index_instance(
            instance=name,
            source=source,
            index_fn=graph_compute.index_repository,
            ingest=engine.ingest_external_batch,
            project_ids=project_ids,
            since=since,
        )
        if summary.watermark and (since is None or summary.watermark > since):
            _write_watermark(backend, wm_key, summary.watermark)
        results.append(summary.as_dict())

    return {
        "status": "ok",
        "source": "gitlab",
        "mode": mode,
        "delta_capable": True,
        "instances": results,
        "projects_indexed": sum(r["projects_indexed"] for r in results),
        "symbols": sum(r["symbols"] for r in results),
        "calls_resolved": sum(r["calls_resolved"] for r in results),
    }


# Sources with a native delta (watermark/reconcile) handler. Add an entry here to
# make another source incremental (e.g. Camunda once its extractor takes `since`).
_DELTA_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "leanix": _sync_leanix,
    "archivebox": _sync_archivebox,
    "gitlab": _sync_gitlab,
    "freshrss": _sync_freshrss,
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

    # "all"/"*"/"sweep" → fan out across every configured connector in one pass
    # so the one entrypoint also covers "ingest everything" (CONCEPT:KG-2.9).
    if source in {"all", "*", "sweep"}:
        return sweep_all_sources(engine, mode=mode if mode in SYNC_ACTIONS else "delta")

    handler = _DELTA_HANDLERS.get(source)
    if handler is not None:
        return handler(engine, mode=mode, ids=ids, client=client)

    if mode == "reconcile":
        return {
            "status": "skipped",
            "reason": f"reconcile not supported for '{source}' (no delta handler)",
        }

    # Extractor/materialize-substrate sources (camunda/aris/egeria) route through the
    # shared materialize core so this stays the one entrypoint for every source.
    from ..enrichment.materialize import MATERIALIZE_SOURCES, run_materialize_source

    if source in MATERIALIZE_SOURCES:
        res = run_materialize_source(engine, source)
        res.setdefault("mode", "full")
        res.setdefault("delta_capable", False)
        return res

    # Otherwise: generic full hydrate via the CAPABILITY_REGISTRY.
    from .hydration import HydrationManager

    res = HydrationManager().hydrate_source(engine, source)
    if isinstance(res, dict):
        res.setdefault("source", source)
        res.setdefault("mode", "full")
        res.setdefault("delta_capable", False)
    return res


def sweep_all_sources(
    engine: Any, *, mode: str = "delta", include_materialize: bool = True
) -> dict[str, Any]:
    """Ingest every *configured* connector in one background sweep (CONCEPT:KG-2.9).

    The fleet-wide counterpart to :func:`sync_source` — it enumerates the union of

    * the delta-capable handlers (:data:`_DELTA_HANDLERS`),
    * the capability-registry sources that env-detect as *configured*, and
    * (optionally) the materialize extractor sources,

    and dispatches each through :func:`sync_source` so they share the one
    watermark/delta/full machinery. ``mode`` defaults to ``"delta"`` so a
    scheduled sweep only pulls (and, via the write-layer content-hash delta, only
    writes) what changed. Per-source failures are isolated and recorded — a
    background sweep never aborts on one bad connector, and unconfigured sources
    are reported as *skipped*, not *errored*.
    """
    candidates: set[str] = set(_DELTA_HANDLERS)

    from .hydration import HydrationManager

    try:
        for src, conf in HydrationManager().get_status().items():
            if isinstance(conf, dict) and conf.get("configured"):
                candidates.add(src)
    except Exception:  # noqa: BLE001 — status probe is best-effort
        logger.debug("capability status probe failed", exc_info=True)

    if include_materialize:
        try:
            from ..enrichment.materialize import MATERIALIZE_SOURCES

            candidates |= set(MATERIALIZE_SOURCES)
        except Exception:  # noqa: BLE001
            logger.debug("materialize source list unavailable", exc_info=True)

    synced: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    errors: dict[str, str] = {}
    _UNCONFIGURED = (
        "not configured",
        "no client",
        "missing",
        "unconfigured",
        "credential",
    )

    for src in sorted(candidates):
        try:
            res = sync_source(engine, src, mode=mode)
            status = res.get("status") if isinstance(res, dict) else "ok"
            if status in {"skipped", "noop"}:
                reason = res.get("reason") if isinstance(res, dict) else None
                skipped[src] = str(reason or "skipped")
            elif status in {"error", "failed"}:
                errors[src] = str(
                    (isinstance(res, dict) and (res.get("error") or res.get("reason")))
                    or "error"
                )
            else:
                synced[src] = {
                    k: res[k]
                    for k in ("nodes", "edges", "ingested", "skipped_unchanged")
                    if isinstance(res, dict) and k in res
                } or status
        except Exception as e:  # noqa: BLE001 — isolate one bad connector
            msg = str(e)
            if any(t in msg.lower() for t in _UNCONFIGURED):
                skipped[src] = f"unconfigured: {msg[:120]}"
            else:
                errors[src] = msg[:200]
                logger.warning("sweep: source '%s' failed: %s", src, e)

    return {
        "status": "ok",
        "mode": mode,
        "swept": len(candidates),
        "synced": synced,
        "skipped": skipped,
        "errors": errors,
        "counts": {
            "synced": len(synced),
            "skipped": len(skipped),
            "errors": len(errors),
        },
    }
