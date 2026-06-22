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


# ── Fleet capability elevation (CONCEPT:KG-2.133) ────────────────────────────
#
# The ~62 fleet MCP servers' tools were AST-ingested as generic ``Code`` symbols
# and never elevated to capability nodes, so the KG lacked the fleet capability
# vocabulary: a query naming "portainer"/"github" matched no ``Tool`` node, so
# neither the ontology classification gate nor the dispatcher's specialist
# routing (``config._fetch_tools`` → ``MATCH (t:Tool)``) could act on it. This
# handler enumerates the **served multiplexer catalog** (the source of truth that
# already lists every fleet tool with name+description+owning server) and writes
# each tool as a ``Tool`` capability node linked to its ``MCPServer`` — fixing the
# classification gate AND the "no fleet specialist registered" hole with one pass.

# Suffix tokens stripped to recover a product/brand synonym from a server name
# (``portainer-agent`` → ``portainer``), mirroring how ``config.py``'s
# ``_synthesize_partition_agents`` derives its ``server_tag``.
_CAPABILITY_GENERIC_TOKENS = frozenset(
    {
        "mcp",
        "agent",
        "api",
        "server",
        "service",
        "manager",
        "tool",
        "tools",
        "client",
        "connector",
        "package",
    }
)
_CAPABILITY_NAME_SUFFIXES = (
    "-mcp",
    "_mcp",
    "-agent",
    "_agent",
    "-api",
    "_api",
    "-server",
    "_server",
    "-manager",
    "_manager",
    "-service",
    "_service",
)


def derive_capability_synonyms(server_name: str) -> list[str]:
    """Matchable terms for a fleet server, for the ontology lexical gate.

    Returns the full server name, its de-suffixed product name, and each
    non-generic token — so a chat turn naming the product ("portainer") matches
    the capability node even though the server is registered as
    ``portainer-agent``. Deterministic and embedding-free.
    """
    import re

    base = (server_name or "").lower().strip()
    if not base:
        return []
    product = base
    for suf in _CAPABILITY_NAME_SUFFIXES:
        if product.endswith(suf):
            product = product[: -len(suf)]
            break
    syns = {base}
    if product:
        syns.add(product)
    for tok in re.split(r"[-_\s]+", base):
        if len(tok) > 1 and tok not in _CAPABILITY_GENERIC_TOKENS:
            syns.add(tok)
    return sorted(syns)


def _capability_product(server_name: str) -> str:
    """The de-suffixed product tag (``portainer-agent`` → ``portainer``)."""
    syns = derive_capability_synonyms(server_name)
    base = (server_name or "").lower().strip()
    # prefer the de-suffixed product (the shortest synonym that is a prefix of base)
    for s in syns:
        if s != base and base.startswith(s):
            return s
    return base


def _existing_disabled(engine: Any, node_id: str) -> bool:
    """Best-effort read of a node's ``disabled`` flag so a re-sync preserves an
    operator's manual disable (mirrors ``kg_server.get_existing_disabled`` without
    creating a knowledge_graph → mcp import inversion)."""
    try:
        gc = getattr(engine, "graph_compute", None)
        graph = getattr(gc, "graph", None)
        if graph is not None and node_id in graph:
            return bool(graph.nodes[node_id].get("disabled", False))
        rows = engine.query_cypher(
            "MATCH (n) WHERE n.id = $id RETURN n.disabled AS disabled", {"id": node_id}
        )
        if rows and isinstance(rows, list):
            return bool(rows[0].get("disabled", False))
    except Exception:  # noqa: BLE001 — disabled is best-effort; default enabled
        pass
    return False


def _derive_tool_mode(input_schema: dict | None) -> str:
    """Classify a served tool as ``condensed`` or ``verbose`` (CONCEPT:KG-2.133).

    The fleet exposes two surfaces per server (MCP_TOOL_MODE=both, ECO-4.82): a *condensed*
    action-routed tool (one tool with ``action`` + ``params_json``) and *verbose* 1:1 tools
    (one typed tool per operation). Both are ingested as distinct ``Tool`` nodes; tagging the
    variant lets selection/analytics prefer the right altitude (condensed for broad, verbose
    for a specific operation) instead of guessing from the name.
    """
    props = (input_schema or {}).get("properties")
    if isinstance(props, dict) and "action" in props and "params_json" in props:
        return "condensed"
    return "verbose"


def _write_fleet_nodes(engine: Any, catalog: dict[str, dict]) -> dict[str, Any]:
    """Write a probed multiplexer catalog into the KG as capability nodes.

    ``catalog`` is the ``{server: {"tools": [{name, description, ...}], "error":
    str|None}}`` map returned by :meth:`MCPMultiplexer.probe_catalog`. For each
    reachable server every tool becomes a ``Tool`` node carrying the schema the
    dispatcher reads (``name``, ``description``, ``mcp_server``, ``tags``,
    ``relevance_score``, ``requires_approval``) plus ``synonyms`` for the lexical
    gate, linked to its (defensively upserted) ``MCPServer`` via ``SERVES``.
    Idempotent: stable node ids + the write-layer content-hash delta skip
    unchanged tools on re-sync. Factored out of :func:`_sync_fleet` so it is
    testable without spawning any servers.
    """
    servers_written = 0
    tools_written = 0
    unreachable: dict[str, str] = {}

    for server_name, info in (catalog or {}).items():
        if not isinstance(info, dict):
            continue
        err = info.get("error")
        if err:
            unreachable[server_name] = str(err)
            continue
        tools = info.get("tools") or []
        if not tools:
            continue

        synonyms = derive_capability_synonyms(server_name)
        product = _capability_product(server_name)
        server_node_id = f"mcp_server_{server_name}"
        # Defensively upsert the server node so the SERVES edge always resolves
        # even when this runs via the MCP/REST surface (not the boot ingest that
        # writes command/args). MERGE+SET only touches the keys we pass, so a
        # prior richer write (command/args) is preserved.
        try:
            engine.add_node(
                server_node_id,
                "MCPServer",
                {
                    "name": server_name,
                    "synonyms": synonyms,
                    "disabled": _existing_disabled(engine, server_node_id),
                },
            )
            servers_written += 1
        except Exception:  # noqa: BLE001 — one bad server never aborts the sweep
            logger.debug(
                "fleet: server upsert failed for %s", server_name, exc_info=True
            )

        for entry in tools:
            if not isinstance(entry, dict):
                continue
            tool_name = entry.get("name")
            if not tool_name:
                continue
            tool_node_id = f"tool_{server_name}_{tool_name}"
            try:
                engine.add_node(
                    tool_node_id,
                    "Tool",
                    {
                        "name": tool_name,
                        "description": entry.get("description", "") or "",
                        "mcp_server": server_name,
                        "tags": [product] if product else [],
                        "relevance_score": 0.5,
                        "requires_approval": False,
                        "synonyms": synonyms,
                        "kind": "mcp_tool",
                        "tool_mode": _derive_tool_mode(entry.get("inputSchema")),
                        "disabled": _existing_disabled(engine, tool_node_id),
                    },
                )
                engine.link_nodes(server_node_id, tool_node_id, "SERVES", {})
                tools_written += 1
            except Exception:  # noqa: BLE001 — isolate per-tool failures
                logger.debug(
                    "fleet: tool write failed for %s/%s",
                    server_name,
                    tool_name,
                    exc_info=True,
                )

    return {
        "servers_written": servers_written,
        "tools_written": tools_written,
        "unreachable": unreachable,
    }


def _resolve_fleet_config():
    """Resolve the fleet ``mcp_config.json`` — the one the multiplexer serves.

    Returns the first candidate that actually parses to ≥1 ``mcpServers`` entry,
    so an empty/placeholder file (e.g. a 0-byte ``~/.gemini/antigravity/
    mcp_config.json``) is skipped rather than silently yielding a 0-server probe.
    Order follows the connector convention (``MCP_CONFIG_PATH``/``MCP_CONFIG`` env
    → ``WORKSPACE_PATH/mcp_config.json``) before the multiplexer's own default
    search, so it stays deployment-agnostic (genesis sets the env).
    """
    import json
    from pathlib import Path

    from ...core.config import setting

    candidates: list[Path] = []
    for key in ("MCP_CONFIG_PATH", "MCP_CONFIG"):
        val = (setting(key, default="") or "").strip()
        if val:
            candidates.append(Path(val))
    ws = (setting("WORKSPACE_PATH", default="/home/apps/workspace") or "").strip()
    if ws:
        candidates.append(Path(ws) / "mcp_config.json")
    try:
        from ...mcp.multiplexer import _resolve_config_path

        rp = _resolve_config_path(None)
        if rp is not None:
            candidates.append(rp)
    except Exception:  # noqa: BLE001 — multiplexer default search is a fallback
        pass

    for path in candidates:
        try:
            if path.exists() and path.stat().st_size > 0:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("mcpServers"):
                    return path
        except Exception:  # noqa: BLE001 — skip unreadable/invalid candidates
            continue
    return None


def _sync_fleet(
    engine: Any, *, mode: str = "full", ids: list[str] | None = None, client: Any = None
) -> dict[str, Any]:
    """Elevate fleet MCP-server tools to KG capability nodes (CONCEPT:KG-2.133).

    Probes the served multiplexer catalog (each fleet server's real tools, via a
    bounded connect→list_tools→release sweep) and writes them as ``Tool``
    capability nodes. ``client`` may inject a pre-probed catalog dict (tests /
    callers that already hold one); otherwise the multiplexer is built from the
    fleet ``mcp_config.json`` and probed. Unreachable servers are recorded, never
    fatal — coverage is "the currently registered + reachable fleet".
    """
    catalog = client if isinstance(client, dict) else None
    if catalog is None:
        try:
            from ...mcp.multiplexer import MCPMultiplexer
            from ...protocols.source_connectors.connectors.mcp_package import _run_async
        except Exception as exc:  # noqa: BLE001 — multiplexer optional at import
            return {
                "status": "skipped",
                "source": "fleet",
                "reason": f"multiplexer unavailable: {exc}",
            }

        config_path = _resolve_fleet_config()
        if config_path is None:
            return {
                "status": "skipped",
                "source": "fleet",
                "reason": "no mcp_config.json with servers found",
            }
        try:
            mux = MCPMultiplexer(config_path)
            catalog = _run_async(mux.probe_catalog())
        except Exception as exc:  # noqa: BLE001 — probe is best-effort
            return {"status": "error", "source": "fleet", "reason": str(exc)}

    counts = _write_fleet_nodes(engine, catalog)
    return {
        "status": "ok",
        "source": "fleet",
        "mode": mode,
        "delta_capable": True,
        "servers_seen": len(catalog or {}),
        **counts,
    }


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


def _sync_fleet_connectors(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Drain EVERY configured ``agent-packages/agents/*`` connector in one pass (CONCEPT:KG-2.151).

    The fleet ships ~50 sibling packages, each a FastMCP server with a declared
    document-yielding tool (the :data:`package_manifest.PACKAGE_PRESETS` catalog —
    scholarx/github-agent/gitlab-api/servicenow-api/mattermost/nextcloud/microsoft/
    atlassian/plane/erpnext/mealie/langfuse/…). Rather than a per-package handler,
    this ONE declarative handler iterates the preset catalog, reaches each package
    through the generic ``mcp`` connector (:class:`MCPPackageConnector`, ECO-4.29),
    and ingests every yielded record through the unified ``DOCUMENT`` path — so the
    "every agents/* connector" leg of a FULL ingest is a single registered source
    (``fleet_connectors``) that the ``source="all"`` sweep fans out as its own laned
    ``connector_sync`` task.

    A package is only attempted when its MCP server is registered in the workspace
    ``mcp_config.json`` (the same source the multiplexer/connector use), so
    unconfigured packages are reported *skipped* — never errored — and one bad
    package never aborts the rest. Delta = a per-package ISO ``updated_at`` watermark
    (``fleet:<package>``); the write-layer content-hash is the second guard, so a
    re-run is a no-op for unchanged records. ``mode='full'`` drains from scratch.
    """
    from ...protocols.source_connectors.connectors.mcp_package import _load_mcp_config
    from ...protocols.source_connectors.connectors.package_manifest import (
        PACKAGE_PRESETS,
        get_preset,
    )
    from ...protocols.source_connectors.registry import build_connector

    servers = _load_mcp_config() or {}
    backend = getattr(engine, "backend", None)

    synced: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    errors: dict[str, str] = {}
    proc: Any = None

    for package in sorted(PACKAGE_PRESETS):
        preset = get_preset(package)
        server = str(preset.get("server") or f"{package}-mcp")
        # Configured = the package's MCP server is registered with the multiplexer.
        if server not in servers and package not in servers:
            skipped[package] = f"{server} not in mcp_config"
            continue
        wm_key = f"fleet:{package}"
        since = None if mode == "full" else _read_watermark(backend, wm_key)
        try:
            conn = build_connector("mcp", {"package": package})
            docs = _drain_incremental(conn, since)
            doc_type = str(preset.get("doc_type") or "document")
            ingested = 0
            for doc in docs:
                text = getattr(doc, "text", "") or ""
                if not text.strip():
                    continue
                if proc is None:
                    proc = _confluence_processor(engine)
                doc_id = f"fleet:{package}:{getattr(doc, 'id', '')}"
                proc.process(
                    text,
                    document_id=doc_id,
                    title=getattr(doc, "title", "") or str(getattr(doc, "id", "")),
                    doc_type=doc_type,
                    source=getattr(doc, "source_uri", "") or "",
                    metadata={
                        "source_system": f"fleet:{package}",
                        "package": package,
                        "updated_at": getattr(doc, "updated_at", None),
                    },
                )
                ingested += 1
            watermark = _max_updated(docs)
            if watermark and (since is None or str(watermark) > str(since)):
                _write_watermark(backend, wm_key, watermark)
            synced[package] = {
                "records_seen": len(docs),
                "documents_ingested": ingested,
                "watermark": watermark,
            }
        except Exception as exc:  # noqa: BLE001 — isolate one bad package
            msg = str(exc)
            if any(
                t in msg.lower()
                for t in ("not configured", "no client", "missing", "credential")
            ):
                skipped[package] = f"unconfigured: {msg[:120]}"
            else:
                errors[package] = msg[:200]
                logger.warning("fleet_connectors: %s failed: %s", package, exc)

    return {
        "status": "ok",
        "source": "fleet_connectors",
        "mode": mode,
        "delta_capable": True,
        "synced": synced,
        "skipped": skipped,
        "errors": errors,
        "counts": {
            "synced": len(synced),
            "skipped": len(skipped),
            "errors": len(errors),
        },
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

    # Configured if FRESHRSS_URL is set OR the freshrss-mcp server is registered in
    # mcp_config — the connector reaches FreshRSS through that server (which holds the
    # GReader credentials), so graph-os itself needs no direct FreshRSS env.
    configured = bool((setting("FRESHRSS_URL", default="") or "").strip())
    if not configured:
        try:
            from ...protocols.source_connectors.connectors.mcp_tool import (
                _load_mcp_config,
            )

            servers = _load_mcp_config() or {}
            configured = "freshrss-mcp" in servers or "freshrss" in servers
        except Exception:  # noqa: BLE001 — best-effort discovery
            configured = False
    if not configured:
        return {
            "status": "skipped",
            "reason": "FreshRSS not configured (set FRESHRSS_URL or add the "
            "freshrss-mcp server to mcp_config)",
        }

    # Register FreshRSS as a first-class :FeedSource node (CONCEPT:KG-2.122) for
    # symmetry with the RSS/scholarx feeds — idempotent, best-effort upsert.
    try:
        from ...automation.feed_sources import upsert_feed_source

        upsert_feed_source(
            engine,
            key="freshrss",
            source_system="freshrss",
            feed_url=(setting("FRESHRSS_URL", default="") or ""),
            kind="FeedSource",
            name="FreshRSS",
        )
    except Exception:  # noqa: BLE001 — registry write is best-effort
        pass

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
    # Bound each run (the */20min sweep drains incrementally) so a cold first run —
    # thousands of backlog articles before any watermark — can't run unbounded. Each
    # cursor batch is ~100 items; default 3 pages ≈ 300 items/run. Override with
    # FRESHRSS_MAX_BATCHES.
    try:
        max_batches = int(setting("FRESHRSS_MAX_BATCHES", default="3") or 3)
    except (TypeError, ValueError):
        max_batches = 3
    if hasattr(conn, "poll_all"):
        docs = list(conn.poll_all(max_batches=max_batches))  # type: ignore[attr-defined]
    else:
        docs = list(conn.load())  # type: ignore[attr-defined]

    from ...automation.worldmodel_pipeline import WorldModelConfig
    from ...base_utilities import to_boolean

    wm_config = WorldModelConfig(
        use_novelty=to_boolean(setting("FRESHRSS_USE_NOVELTY", default="False"))
    )
    report = WorldModelPipelineRunner(engine=engine, config=wm_config).run_gated_ingest(
        docs
    )

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


def _sync_rss(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Native RSS/Atom feeds + ScholarX arXiv through the ONE world-model gate (KG-2.121).

    The unified feed handler: native feed URLs (``KG_RSS_FEEDS``) are drained by the
    zero-infra ``rss`` connector and ScholarX arXiv items by the scholarx feed bridge;
    both emit the same ``SourceDocument`` shape and flow through
    :meth:`WorldModelPipelineRunner.run_gated_ingest` — research/arXiv items take the
    prioritized ``research_paper_fetch`` path, news items the relevance+novelty gate.
    Each configured feed is materialized as a first-class ``:FeedSource`` node on this
    live path. Delta = an ISO publish-date watermark; node-existence (``_is_known``)
    is the cross-run dedup.
    """
    from ...automation.feed_sources import (
        list_feed_sources,
        register_feed_nodes,
        scholarx_feed_documents,
    )
    from ...automation.worldmodel_pipeline import WorldModelPipelineRunner
    from ...core.config import config as _cfg
    from ...protocols.source_connectors.base import ConnectorCheckpoint
    from ...protocols.source_connectors.registry import build_connector

    # Native feed URLs = the comma-separated config seed UNION the runtime-added
    # :FeedSource registry (so graph_feeds add → next sweep ingests it).
    seed = (getattr(_cfg, "kg_rss_feeds", "") or "").split(",")
    native_url_set = {u.strip() for u in seed if u.strip()}
    for node in list_feed_sources(engine):
        if (
            node.get("source_system") == "rss"
            and node.get("enabled", True)
            and node.get("feed_url")
        ):
            native_url_set.add(str(node["feed_url"]))
    native_urls = sorted(native_url_set)
    try:
        import scholarx  # noqa: F401

        scholarx_ok = True
    except Exception:  # noqa: BLE001
        scholarx_ok = False
    if not native_urls and not scholarx_ok:
        return {
            "status": "skipped",
            "reason": "no native RSS feeds (set KG_RSS_FEEDS) and scholarx not installed",
        }

    backend = getattr(engine, "backend", None)
    since = None if mode == "full" else _read_watermark(backend, "rss")

    # Materialize the feed registry on the live sweep path (Wire-First, KG-2.122).
    register_feed_nodes(
        engine,
        native_urls=native_urls,
        scholarx_categories=(["arxiv"] if scholarx_ok else []),
    )

    docs: list[Any] = []
    if native_urls:
        conn = build_connector("rss", {"feed_urls": native_urls})
        cp = ConnectorCheckpoint(watermark=since) if since else None
        if hasattr(conn, "poll_all"):
            docs.extend(list(conn.poll_all(cp)))  # type: ignore[attr-defined]
        else:
            docs.extend(list(conn.load()))  # type: ignore[attr-defined]
    if scholarx_ok:
        docs.extend(scholarx_feed_documents())

    report = WorldModelPipelineRunner(engine=engine).run_gated_ingest(docs)

    iso_dates = [d.updated_at for d in docs if getattr(d, "updated_at", None)]
    new_watermark = max(iso_dates) if iso_dates else None
    if new_watermark and (since is None or new_watermark > since):
        _write_watermark(backend, "rss", new_watermark)

    return {
        "status": "ok",
        "source": "rss",
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


# ── Atlassian + Plane issue trackers / wiki (KG-2.123/2.124/2.125) ────────────
#
# Three first-class delta connectors that reach Jira / Confluence / Plane through
# their fleet MCP servers (``atlassian-mcp`` / ``plane-mcp``) via the declarative
# mcp_tool presets — never a direct vendor client. Each is **multi-instance** (the
# GitLab pattern): a second Atlassian site or Plane workspace is a second ``*-mcp``
# server entry + a typed ``*_instances`` config row, so the same logic ingests both.
#
# CONCEPT:KG-2.123 — Confluence first-class delta connector
# CONCEPT:KG-2.124 — Jira first-class delta connector
# CONCEPT:KG-2.125 — Plane first-class delta connector


def _resolve_tracker_instances(
    field: str,
    *,
    default_name: str,
    default_server: str,
    scope_key: str,
    scope_setting: str,
) -> list[dict[str, Any]]:
    """Configured ``*_instances`` rows, or one synthetic instance from the single-host
    settings (mirrors ``gitlab_indexer.instances_from_config``)."""
    from ...core.config import config as cfg
    from ...core.config import setting

    rows = [r for r in (getattr(cfg, field, None) or []) if isinstance(r, dict)]
    if rows:
        return rows
    scope = [
        s.strip()
        for s in (setting(scope_setting, default="") or "").split(",")
        if s.strip()
    ]
    return [{"name": default_name, "server": default_server, scope_key: scope}]


def _build_preset_conn(preset: str, server: str, params: dict[str, Any]) -> Any:
    """Build the mcp_tool connector for one tracker instance (preset + per-instance
    server override + per-run params)."""
    from ...protocols.source_connectors.registry import build_connector

    config: dict[str, Any] = {"preset": preset, "server": server}
    if params:
        config["params"] = params
    return build_connector("mcp_tool", config)


def _drain_incremental(
    conn: Any, since: str | None, *, max_batches: int = 25
) -> list[Any]:
    """Drain a connector incrementally via ``poll()`` — binds the ``since`` watermark
    (client-side ``updated_field`` filter), resumes the cursor across batches, and is
    bounded so a cold first run can't run unbounded."""
    from ...protocols.source_connectors.base import ConnectorCheckpoint

    docs: list[Any] = []
    cp = ConnectorCheckpoint(watermark=since) if since else None
    for _ in range(max(1, max_batches)):
        batch = conn.poll(cp)
        docs.extend(batch.documents)
        cp = batch.checkpoint
        if not getattr(cp, "has_more", False):
            break
    return docs


def _record_of(doc: Any) -> dict[str, Any]:
    """The raw source record the connector preserved in ``metadata.record``."""
    rec = (getattr(doc, "metadata", None) or {}).get("record")
    return rec if isinstance(rec, dict) else {}


def _max_updated(docs: list[Any]) -> str | None:
    seen = [u for d in docs if (u := getattr(d, "updated_at", None))]
    return max(seen, key=str) if seen else None


def _jira_jql_date(value: Any) -> str | None:
    """Render an ISO8601 watermark as a Jira JQL datetime (``yyyy-MM-dd HH:mm``)."""
    import re

    m = re.match(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})", str(value))
    return (
        f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}"
        if m
        else None
    )


def _jira_jql(inst: dict[str, Any], since: str | None, ids: list[str] | None) -> str:
    clauses: list[str] = []
    keys = [str(k) for k in (inst.get("project_keys") or []) if k]
    if keys:
        clauses.append(f"project in ({','.join(keys)})")
    if ids:
        clauses.append(f"key in ({','.join(str(i) for i in ids)})")
    if since and (d := _jira_jql_date(since)):
        clauses.append(f'updated >= "{d}"')
    if extra := str(inst.get("jql") or "").strip():
        clauses.append(f"({extra})")
    where = " AND ".join(clauses)
    return f"{where} ORDER BY updated DESC" if where else "ORDER BY updated DESC"


def _jira_entities(
    docs: list[Any], instance: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map drained Jira records → issue/person/epic entities + relationships
    (the mapping inherited from the removed ``_hydrate_jira``)."""
    entities: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []
    src = f"jira:{instance}"
    for doc in docs:
        key = getattr(doc, "id", None)
        if not key:
            continue
        fields = _record_of(doc).get("fields") or {}
        node_id = f"jira:{instance}:issue:{key}"
        entities.append(
            {
                "id": node_id,
                "type": "issue",
                "name": fields.get("summary") or f"Issue {key}",
                "status": (fields.get("status") or {}).get("name", ""),
                "priority": (fields.get("priority") or {}).get("name", ""),
                "issueKey": str(key),
                "domain": "jira",
                "source_system": src,
                "externalToolId": str(key),
                "updatedAt": fields.get("updated"),
            }
        )
        assignee = fields.get("assignee")
        if isinstance(assignee, dict):
            uid = assignee.get("accountId") or assignee.get("name")
            if uid:
                user_node = f"jira:{instance}:user:{uid}"
                entities.append(
                    {
                        "id": user_node,
                        "type": "person",
                        "name": assignee.get("displayName") or f"User {uid}",
                        "domain": "jira",
                        "source_system": src,
                    }
                )
                rels.append(
                    {
                        "source": node_id,
                        "target": user_node,
                        "type": "has_role",
                        "domain": "jira",
                    }
                )
        parent = fields.get("parent")
        epic = (
            parent.get("key")
            if isinstance(parent, dict)
            else fields.get("customfield_10014")
        )
        if epic:
            epic_node = f"jira:{instance}:epic:{epic}"
            entities.append(
                {
                    "id": epic_node,
                    "type": "goal",
                    "name": f"Epic {epic}",
                    "domain": "jira",
                    "source_system": src,
                }
            )
            rels.append(
                {
                    "source": node_id,
                    "target": epic_node,
                    "type": "part_of",
                    "domain": "jira",
                }
            )
    return entities, rels


def _sync_jira(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Jira issues as typed issue/person/epic entities (CONCEPT:KG-2.124).

    Per configured instance, drains the ``jira`` mcp_tool preset over its
    ``atlassian-mcp`` server with a JQL ``updated >= <watermark>`` server-side delta
    (the write-layer content-hash is the second guard); rebuilds the issue graph from
    each record and ``ingest_external_batch``-es it. ``ids`` narrows to specific keys
    (webhook). Replaces the removed single-shot ``_hydrate_jira``.
    """
    backend = getattr(engine, "backend", None)
    instances = _resolve_tracker_instances(
        "jira_instances",
        default_name="jira",
        default_server="atlassian-mcp",
        scope_key="project_keys",
        scope_setting="JIRA_PROJECT_KEYS",
    )
    results: list[dict[str, Any]] = []
    total_e = total_r = 0
    for inst in instances:
        name = str(inst.get("name") or "jira")
        server = str(inst.get("server") or "atlassian-mcp")
        wm_key = f"jira:{name}"
        since = None if mode == "full" else _read_watermark(backend, wm_key)
        if mode == "reconcile":
            conn = _build_preset_conn(
                "jira", server, {"jql": _jira_jql(inst, None, None)}
            )
            live = {str(getattr(d, "id", "")) for d in _drain_incremental(conn, None)}
            results.append(_reconcile(engine, "jira", live) | {"instance": name})
            continue
        conn = _build_preset_conn("jira", server, {"jql": _jira_jql(inst, since, ids)})
        docs = _drain_incremental(conn, since)
        entities, rels = _jira_entities(docs, name)
        if entities:
            engine.ingest_external_batch("jira", entities, rels)
        watermark = _max_updated(docs)
        if watermark and (since is None or str(watermark) > str(since)):
            _write_watermark(backend, wm_key, watermark)
        total_e += len(entities)
        total_r += len(rels)
        results.append({"instance": name, "issues": len(docs), "watermark": watermark})
    return {
        "status": "ok",
        "source": "jira",
        "mode": mode,
        "delta_capable": True,
        "instances": results,
        "nodes_hydrated": total_e,
        "relations_hydrated": total_r,
    }


def _plane_entities(
    docs: list[Any], instance: str, project_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map drained Plane work items → issue + project entities (inherited from the
    removed ``_hydrate_plane``)."""
    entities: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []
    src = f"plane:{instance}"
    proj_node = f"plane:{instance}:proj:{project_id}"
    proj_emitted = False
    for doc in docs:
        iid = getattr(doc, "id", None)
        if not iid:
            continue
        rec = _record_of(doc)
        state = rec.get("state")
        state_name = state.get("name", "") if isinstance(state, dict) else (state or "")
        node_id = f"plane:{instance}:issue:{iid}"
        entities.append(
            {
                "id": node_id,
                "type": "issue",
                "name": rec.get("name") or f"Plane Issue {iid}",
                "state": state_name,
                "priority": rec.get("priority") or "",
                "domain": "plane",
                "source_system": src,
                "externalToolId": str(iid),
                "updatedAt": rec.get("updated_at"),
            }
        )
        if not proj_emitted:
            entities.append(
                {
                    "id": proj_node,
                    "type": "software_project",
                    "name": f"Plane Project {project_id}",
                    "domain": "plane",
                    "source_system": src,
                }
            )
            proj_emitted = True
        rels.append(
            {
                "source": node_id,
                "target": proj_node,
                "type": "part_of",
                "domain": "plane",
            }
        )
    return entities, rels


def _sync_plane(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Plane work items as typed issue/project entities (CONCEPT:KG-2.125).

    Per configured instance × project, drains the ``plane`` mcp_tool preset over its
    ``plane-mcp`` server (a SECOND Plane workspace is a second instance row pointing at
    a second server). Delta = the ``updated_at`` watermark + content-hash. Replaces the
    removed ``_hydrate_plane``.
    """
    backend = getattr(engine, "backend", None)
    instances = _resolve_tracker_instances(
        "plane_instances",
        default_name="plane",
        default_server="plane-mcp",
        scope_key="projects",
        scope_setting="PLANE_PROJECT_IDS",
    )
    results: list[dict[str, Any]] = []
    total_e = total_r = 0
    for inst in instances:
        name = str(inst.get("name") or "plane")
        server = str(inst.get("server") or "plane-mcp")
        projects = [str(p) for p in (inst.get("projects") or []) if p]
        if not projects:
            results.append(
                {
                    "instance": name,
                    "status": "skipped",
                    "reason": "no projects configured",
                }
            )
            continue
        inst_e: list[dict[str, Any]] = []
        inst_r: list[dict[str, Any]] = []
        for pid in projects:
            wm_key = f"plane:{name}:{pid}"
            since = None if mode == "full" else _read_watermark(backend, wm_key)
            params: dict[str, Any] = {"project_id": pid}
            if ids:
                params["filters"] = {"id": ids}
            conn = _build_preset_conn("plane", server, params)
            docs = _drain_incremental(conn, since)
            e, r = _plane_entities(docs, name, pid)
            inst_e += e
            inst_r += r
            watermark = _max_updated(docs)
            if watermark and (since is None or str(watermark) > str(since)):
                _write_watermark(backend, wm_key, watermark)
        if inst_e:
            engine.ingest_external_batch("plane", inst_e, inst_r)
        total_e += len(inst_e)
        total_r += len(inst_r)
        results.append(
            {"instance": name, "issues": sum(1 for x in inst_e if x["type"] == "issue")}
        )
    return {
        "status": "ok",
        "source": "plane",
        "mode": mode,
        "delta_capable": True,
        "instances": results,
        "nodes_hydrated": total_e,
        "relations_hydrated": total_r,
    }


def _confluence_processor(engine: Any) -> Any:
    from ..ontology.document_processing import ChunkingConfig, DocumentProcessor

    return DocumentProcessor(
        getattr(engine, "backend", None), chunking=ChunkingConfig(), contextual=True
    )


def _sync_confluence(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Full-mirror Confluence pages as ``:ConfluencePage`` Documents (CONCEPT:KG-2.123).

    Per configured instance × space, drains the ``confluence`` mcp_tool preset
    (Cloud-v2 ``get_pages``, recency-sorted, body inline) over its ``atlassian-mcp``
    server and ingests each page through the KG-2.48 ``DocumentProcessor`` (chunk +
    embed) so the wiki is fully searchable. Delta = the ``version.createdAt`` since
    filter + the write-layer content-hash. ``ids`` narrows to specific pages (webhook).
    NOT relevance-gated — internal wiki is curated knowledge.
    """
    backend = getattr(engine, "backend", None)
    instances = _resolve_tracker_instances(
        "confluence_instances",
        default_name="confluence",
        default_server="atlassian-mcp",
        scope_key="spaces",
        scope_setting="CONFLUENCE_SPACE_IDS",
    )
    proc: Any = None
    results: list[dict[str, Any]] = []
    total = 0
    for inst in instances:
        name = str(inst.get("name") or "confluence")
        server = str(inst.get("server") or "atlassian-mcp")
        spaces: list[str | None] = [
            str(s) for s in (inst.get("spaces") or []) if s
        ] or [None]
        pages = 0
        for space in spaces:
            wm_key = f"confluence:{name}:{space or 'all'}"
            since = None if mode == "full" else _read_watermark(backend, wm_key)
            params: dict[str, Any] = {}
            if space:
                params["space_id"] = [space]
            if ids:
                params["id_"] = ids
            conn = _build_preset_conn("confluence", server, params)
            docs = _drain_incremental(conn, since)
            if docs and proc is None:
                proc = _confluence_processor(engine)
            for doc in docs:
                doc_id = f"confluence:{name}:{getattr(doc, 'id', '')}"
                rec = _record_of(doc)
                try:
                    proc.process(
                        getattr(doc, "text", "") or "",
                        document_id=doc_id,
                        title=getattr(doc, "title", "") or str(getattr(doc, "id", "")),
                        doc_type="wiki",
                        source=getattr(doc, "source_uri", ""),
                        metadata={
                            "source_system": f"confluence:{name}",
                            "space_id": rec.get("spaceId"),
                            "version": (rec.get("version") or {}).get("number"),
                            "confluence_page_id": str(getattr(doc, "id", "")),
                            "updated_at": getattr(doc, "updated_at", None),
                        },
                    )
                    pages += 1
                except Exception as exc:  # noqa: BLE001 — one bad page must not abort
                    logger.warning(
                        "[KG-2.123] confluence page ingest failed for %s: %s",
                        getattr(doc, "id", "?"),
                        exc,
                    )
            watermark = _max_updated(docs)
            if watermark and (since is None or str(watermark) > str(since)):
                _write_watermark(backend, wm_key, watermark)
        total += pages
        results.append({"instance": name, "pages": pages})
    return {
        "status": "ok",
        "source": "confluence",
        "mode": mode,
        "delta_capable": True,
        "instances": results,
        "pages_ingested": total,
    }


# Sources with a native delta (watermark/reconcile) handler. Add an entry here to
# make another source incremental (e.g. Camunda once its extractor takes `since`).
_DELTA_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "leanix": _sync_leanix,
    "archivebox": _sync_archivebox,
    "gitlab": _sync_gitlab,
    "freshrss": _sync_freshrss,
    "rss": _sync_rss,
    "jira": _sync_jira,
    "confluence": _sync_confluence,
    "plane": _sync_plane,
    "fleet": _sync_fleet,
    "fleet_connectors": _sync_fleet_connectors,
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
    engine: Any,
    *,
    mode: str = "delta",
    include_materialize: bool = True,
    enqueue: bool = True,
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
    # ``fleet`` capability elevation re-probes ~62 MCP servers; the capability
    # vocabulary is slow-changing, so it runs at boot + on explicit refresh
    # (``source_sync source=fleet``), not on every */20m document sweep.
    candidates.discard("fleet")

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

    # CONCEPT:ORCH-1.77 — fan the sweep out as LANED ``connector_sync`` tasks (the 'connectors'
    # lane) so every connector syncs in PARALLEL instead of one slow connector (gitlab/
    # servicenow) head-of-line-blocking the rest in the sequential inline loop below. Each task
    # runs ``sync_source(src, mode)`` → the same watermark/delta machinery + content-hash delta.
    if enqueue and hasattr(engine, "submit_task"):
        jobs: list[str] = []
        for src in sorted(candidates):
            try:
                jobs.append(
                    engine.submit_task(
                        target_path=src,
                        is_codebase=False,
                        provenance={"sync_mode": mode},
                        task_type="connector_sync",
                    )
                )
            except Exception:  # noqa: BLE001 — one bad enqueue never aborts the sweep
                logger.debug("enqueue connector_sync failed for %s", src, exc_info=True)
        return {
            "status": "enqueued",
            "enqueued": len(jobs),
            "candidates": len(candidates),
            "mode": mode,
            "jobs": jobs,
        }

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
