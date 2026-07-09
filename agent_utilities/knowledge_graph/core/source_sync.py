"""Source-agnostic KG synchronization (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

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


# ── Fleet capability elevation (CONCEPT:AU-KG.ontology.capability-node-aliases-lexical) ────────────────────────────
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
    """Classify a served tool as ``condensed`` or ``verbose`` (CONCEPT:AU-KG.ontology.capability-node-aliases-lexical).

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
    """Elevate fleet MCP-server tools to KG capability nodes (CONCEPT:AU-KG.ontology.capability-node-aliases-lexical).

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
    """Ingest preserved ArchiveBox snapshots into the KG (CONCEPT:AU-KG.query.vendor-agnostic-traversal).

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


# PACKAGE_PRESETS packages whose upstream is ALREADY ingested by a dedicated
# _DELTA_HANDLERS source — excluded from the fleet sweep to avoid double-ingestion
# (CONCEPT:AU-KG.compute.gitlab-api-gitlab-atlassian). gitlab-api→gitlab, atlassian-agent→jira/confluence,
# plane-agent→plane, scholarx→the research feed. Keep this in sync with
# _DELTA_HANDLERS: a package gains a dedicated handler ⇒ add it here.
_FLEET_DEDICATED_PACKAGES: frozenset[str] = frozenset(
    {"gitlab-api", "atlassian-agent", "plane-agent", "scholarx"}
)


def _sync_fleet_connectors(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Drain EVERY configured ``agent-packages/agents/*`` connector in one pass (CONCEPT:AU-KG.compute.gitlab-api-gitlab-atlassian).

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
        # Skip packages already covered by a DEDICATED delta handler — otherwise
        # source="all" enqueues BOTH this fleet leg AND the dedicated source for the
        # same upstream, writing under non-matching doc-id namespaces so the
        # content-hash delta can't dedup → duplicate Document/Chunk nodes every run
        # (CONCEPT:AU-KG.compute.gitlab-api-gitlab-atlassian). The dedicated handler owns these upstreams.
        if package in _FLEET_DEDICATED_PACKAGES:
            skipped[package] = "covered by a dedicated delta handler"
            continue
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
    """Relevance-gated ingestion of curated FreshRSS items (CONCEPT:AU-KG.compute.homelab-rss-reader-as).

    Enumerates items via the ``freshrss`` mcp_tool preset over the Google-Reader API
    (delta = ``newer_than`` → GReader ``ot`` **unix-seconds** watermark on
    ``published``; ``mode='full'`` drains all). Unlike a mirror connector this source
    is INTENTIONALLY GATED: each item passes the world-model relevance gate
    (:class:`WorldModelPipelineRunner`, CONCEPT:AU-KG.ingest.news-finance-tech-sibling) — only items relevant to the
    existing KG (taxonomy score OR concept-novelty) or agent-force flagged are fully
    ingested as ``news_article`` Documents; the rest get a marginal footprint or are
    skipped. Research/arXiv-feed items route to the research path (CONCEPT:AU-KG.ingest.worldmodel-gated-ingestion),
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

    # Register FreshRSS as a first-class :FeedSource node (CONCEPT:AU-KG.compute.first-class-rss-atom) for
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
    """Index whole GitLab instances as a resolved code graph (CONCEPT:AU-KG.backend.declared-columns-so-schema).

    For every configured instance (``GITLAB_INSTANCES`` JSON, else single
    ``GITLAB_URL``/``GITLAB_TOKEN``) this enumerates projects → default-branch code
    files and ships each project to the engine's ``index_repository`` resolver
    (CONCEPT:EG-KG.compute.turn-each-project), writing ``:Code`` symbols + resolved ``calls``/``depends_on``
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
# CONCEPT:AU-KG.compute.confluence-first-class-delta — Confluence first-class delta connector
# CONCEPT:AU-KG.compute.jira-first-class-delta — Jira first-class delta connector
# CONCEPT:AU-KG.compute.plane-first-class-delta — Plane first-class delta connector


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
    return (
        f"{where} ORDER BY updated DESC"
        if where
        # Jira Cloud /search/jql (search-and-reconcile) rejects an UNBOUNDED query
        # (400); a wide created-bound keeps "all issues" valid (CONCEPT:AU-KG.compute.jira-first-class-delta).
        else 'created >= "1970-01-01" ORDER BY updated DESC'
    )


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
    """Ingest Jira issues as typed issue/person/epic entities (CONCEPT:AU-KG.compute.jira-first-class-delta).

    Per configured instance, drains the ``jira`` mcp_tool preset over its
    ``atlassian-mcp`` server with a JQL ``updated >= <watermark>`` server-side delta
    (the write-layer content-hash is the second guard); rebuilds the issue graph from
    each record and ``ingest_external_batch``-es it. ``ids`` narrows to specific keys
    (webhook). Replaces the removed single-shot ``_hydrate_jira``.

    Deployment note (CONCEPT:AU-KG.compute.jira-first-class-delta): wire ``atlassian-mcp`` in the source
    ``mcp_config`` over streamable-http (``transport``/``url`` →
    ``http://atlassian-mcp.arpa/mcp``), mirroring freshrss-mcp / plane-mcp — never a
    local ``command`` venv binary, which would (mis)spawn a stdio server on the host.
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
    """Ingest Plane work items as typed issue/project entities (CONCEPT:AU-KG.compute.plane-first-class-delta).

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
    """Full-mirror Confluence pages as ``:ConfluencePage`` Documents (CONCEPT:AU-KG.compute.confluence-first-class-delta).

    Per configured instance × space, drains the ``confluence`` mcp_tool preset
    (Cloud-v2 ``get_pages``, recency-sorted, body inline) over its ``atlassian-mcp``
    server and ingests each page through the KG-2.48 ``DocumentProcessor`` (chunk +
    embed) so the wiki is fully searchable. Delta = the ``version.createdAt`` since
    filter + the write-layer content-hash. ``ids`` narrows to specific pages (webhook).
    NOT relevance-gated — internal wiki is curated knowledge.

    Deployment note (CONCEPT:AU-KG.compute.confluence-first-class-delta): the ``atlassian-mcp`` server is reached over
    streamable-http at its fleet URL (``http://atlassian-mcp.arpa/mcp``) — wire it in
    the source ``mcp_config`` with ``transport``/``url`` (mirroring freshrss-mcp /
    plane-mcp), never a local ``command`` venv binary. Confluence Cloud v2 paths are
    bare (``/spaces``, ``/pages``), so the **service** must set
    ``ATLASSIAN_CONFLUENCE_CLOUD_URL=https://<site>.atlassian.net/wiki/api/v2`` — the
    per-suite override in ``atlassian_agent.auth.get_confluence_cloud_client``;
    otherwise the client falls back to the Jira base URL and every call 404s.
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


# ── Ops / platform connectors as typed OWL entities (CONCEPT:AU-KG.compute.dockerhub-repositories–2.161) ──────
#
# Seven first-class delta connectors that reach their upstream ONLY through a fleet
# ``*-mcp`` server (like jira/confluence/plane) and rebuild **typed** entities mapped to
# OWL classes — not generic Documents. Each is MCP-configured: its "configured" signal is
# *"the server is registered in mcp_config.json"*. Delta = a per-source ISO ``updated_at``
# watermark + the write-layer content-hash; the server it reaches is in ``_MCP_TRACKER_SERVERS``.
#
# CONCEPT:AU-KG.compute.dockerhub-repositories — DockerHub repositories → :Repository / :ContainerImage
# CONCEPT:AU-KG.compute.langfuse-traces-observations — Langfuse traces/observations → :Trace / :Observation / :Generation
# CONCEPT:AU-KG.compute.technitium-dns-zones-records — Technitium DNS zones+records → :DnsZone / :DnsRecord
# CONCEPT:AU-KG.compute.tunnel-manager-hosts — tunnel-manager hosts → :Host / :Tunnel
# CONCEPT:AU-KG.compute.uptime-kuma-monitors — Uptime Kuma monitors → :Monitor / :HeartbeatStat
# CONCEPT:AU-KG.compute.home-assistant-states — Home Assistant states → :Device / :Entity
# CONCEPT:AU-KG.compute.twenty-crm-people-companies — Twenty CRM people/companies/opportunities → :Person / :Company / :Opportunity


def _configured_server(server_candidates: tuple[str, ...]) -> str | None:
    """The first candidate ``*-mcp`` server actually registered in ``mcp_config.json``
    (or its ``<name>-mcp`` alias), or ``None`` when none is — so a handler reaches the
    upstream through the server the operator really configured (the catalog name and the
    local config key can differ, e.g. ``uptime-mcp`` vs ``uptime-kuma-mcp``)."""
    try:
        from ...protocols.source_connectors.connectors.mcp_package import (
            _load_mcp_config,
        )

        servers = _load_mcp_config() or {}
    except Exception:  # noqa: BLE001 — no readable config → not configured here
        return None
    for cand in server_candidates:
        if cand in servers:
            return cand
        if f"{cand}-mcp" in servers:
            return f"{cand}-mcp"
    return None


def _server_configured(server_candidates: tuple[str, ...]) -> bool:
    """True when any candidate ``*-mcp`` server (or its de-suffixed alias) is in
    ``mcp_config.json`` — the connector reaches the upstream only through that server."""
    return _configured_server(server_candidates) is not None


def _drain_preset(
    preset: str, *, server: str = "", params: dict[str, Any] | None = None
) -> list[Any]:
    """Build the ``mcp_tool`` connector for a preset and drain ONE full sweep.

    Used by the typed handlers below: the preset does the list/page/cursor drain, the
    handler maps each ``metadata.record`` to a typed entity. Bounded by the connector's
    own ``max_pages`` so a cold run can't loop unbounded.
    """
    from ...protocols.source_connectors.registry import build_connector

    config: dict[str, Any] = {"preset": preset}
    if server:
        config["server"] = server
    if params:
        config["params"] = params
    conn = build_connector("mcp_tool", config)
    if hasattr(conn, "poll_all"):
        return list(conn.poll_all())  # type: ignore[attr-defined]
    return list(conn.load())  # type: ignore[attr-defined]


def _ingest_typed(
    engine: Any,
    source: str,
    entities: list[dict[str, Any]],
    rels: list[dict[str, Any]],
    *,
    wm_key: str,
    since: str | None,
    watermark: str | None,
) -> None:
    """Ingest a typed entity/relationship batch + advance the watermark (shared tail)."""
    if entities:
        engine.ingest_external_batch(source, entities, rels)
    backend = getattr(engine, "backend", None)
    if watermark and (since is None or str(watermark) > str(since)):
        _write_watermark(backend, wm_key, watermark)


def _sync_dockerhub(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest DockerHub repositories as :Repository + :ContainerImage (CONCEPT:AU-KG.compute.dockerhub-repositories).

    Per configured namespace (``DOCKERHUB_NAMESPACES`` CSV, else ``DOCKERHUB_NAMESPACE``,
    else ``ids`` as namespaces) drains the ``dockerhub-repos`` preset over ``dockerhub-mcp``
    and rebuilds each repo as a :ContainerImage (image coordinates + pull/star counts) that
    ``contains`` the namespace's :Repository. Delta = the ``last_updated`` watermark.

    Uses the shared transform primitives (CONCEPT:AU-KG.etl.transform-primitives) —
    :func:`~..etl.transforms.coalesce` for the image-name fallback and
    :func:`~..etl.transforms.stable_id` for the ``dockerhub:<ns>[/<name>]`` node ids
    — as the first migrated handler proving the pattern.
    """
    if not _server_configured(("dockerhub-mcp", "dockerhub-api")):
        return {"status": "skipped", "reason": "dockerhub-mcp not in mcp_config"}
    from ..etl.transforms import coalesce, stable_id
    from ...core.config import setting

    namespaces = [
        n.strip()
        for n in (
            setting("DOCKERHUB_NAMESPACES", default="")
            or setting("DOCKERHUB_NAMESPACE", default="")
        ).split(",")
        if n.strip()
    ] or [str(i) for i in (ids or [])]
    if not namespaces:
        return {"status": "skipped", "reason": "no DockerHub namespace configured"}

    backend = getattr(engine, "backend", None)
    total = 0
    results: list[dict[str, Any]] = []
    for ns in namespaces:
        wm_key = f"dockerhub:{ns}"
        since = None if mode == "full" else _read_watermark(backend, wm_key)
        docs = _drain_preset("dockerhub-repos", params={"namespace": ns})
        repo_node = stable_id(ns, prefix="dockerhub")
        entities: list[dict[str, Any]] = [
            {
                "id": repo_node,
                "type": "repository",
                "name": ns,
                "domain": "dockerhub",
                "source_system": f"dockerhub:{ns}",
            }
        ]
        rels: list[dict[str, Any]] = []
        for doc in docs:
            rec = _record_of(doc)
            name = coalesce(rec, "name") or getattr(doc, "id", None)
            if not name:
                continue
            img_id = f"dockerhub:{ns}/{name}"
            entities.append(
                {
                    "id": img_id,
                    "type": "container_image",
                    "name": f"{ns}/{name}",
                    "description": rec.get("description") or "",
                    "pull_count": rec.get("pull_count"),
                    "star_count": rec.get("star_count"),
                    "is_private": rec.get("is_private"),
                    "domain": "dockerhub",
                    "source_system": f"dockerhub:{ns}",
                    "externalToolId": f"{ns}/{name}",
                    "updatedAt": rec.get("last_updated"),
                }
            )
            rels.append(
                {
                    "source": repo_node,
                    "target": img_id,
                    "type": "contains",
                    "domain": "dockerhub",
                }
            )
        _ingest_typed(
            engine,
            "dockerhub",
            entities,
            rels,
            wm_key=wm_key,
            since=since,
            watermark=_max_updated(docs),
        )
        total += len(docs)
        results.append({"namespace": ns, "images": len(docs)})
    return {
        "status": "ok",
        "source": "dockerhub",
        "mode": mode,
        "delta_capable": True,
        "namespaces": results,
        "images_ingested": total,
    }


def _sync_langfuse(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Langfuse traces + observations as :Trace / :Observation / :Generation
    (CONCEPT:AU-KG.compute.langfuse-traces-observations).

    Drains the ``langfuse-traces`` and ``langfuse-observations`` presets over
    ``langfuse-mcp``; each trace is a :Trace, each observation a :Observation (LLM-call
    observations — ``type == 'GENERATION'`` — are :Generation), linked ``part_of`` their
    trace via ``traceId``. Delta = the ``timestamp`` / ``startTime`` watermark.
    """
    if not _server_configured(("langfuse-mcp", "langfuse-agent")):
        return {"status": "skipped", "reason": "langfuse-mcp not in mcp_config"}
    backend = getattr(engine, "backend", None)
    wm_key = "langfuse"
    since = None if mode == "full" else _read_watermark(backend, wm_key)
    src = "langfuse"

    trace_docs = _drain_preset("langfuse-traces")
    obs_docs = _drain_preset("langfuse-observations")
    entities: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []
    for doc in trace_docs:
        tid = getattr(doc, "id", None)
        if not tid:
            continue
        rec = _record_of(doc)
        entities.append(
            {
                "id": f"langfuse:trace:{tid}",
                "type": "trace",
                "name": rec.get("name") or f"Trace {tid}",
                "user_id": rec.get("userId"),
                "session_id": rec.get("sessionId"),
                "domain": "langfuse",
                "source_system": src,
                "externalToolId": str(tid),
                "updatedAt": rec.get("timestamp"),
            }
        )
    for doc in obs_docs:
        oid = getattr(doc, "id", None)
        if not oid:
            continue
        rec = _record_of(doc)
        is_gen = str(rec.get("type") or "").upper() == "GENERATION"
        node_id = f"langfuse:obs:{oid}"
        entities.append(
            {
                "id": node_id,
                "type": "generation" if is_gen else "observation",
                "name": rec.get("name") or f"Observation {oid}",
                "model": rec.get("model"),
                "domain": "langfuse",
                "source_system": src,
                "externalToolId": str(oid),
                "updatedAt": rec.get("startTime"),
            }
        )
        if tid := rec.get("traceId"):
            rels.append(
                {
                    "source": node_id,
                    "target": f"langfuse:trace:{tid}",
                    "type": "part_of",
                    "domain": "langfuse",
                }
            )
    _ingest_typed(
        engine,
        src,
        entities,
        rels,
        wm_key=wm_key,
        since=since,
        watermark=_max_updated(trace_docs + obs_docs),
    )
    return {
        "status": "ok",
        "source": "langfuse",
        "mode": mode,
        "delta_capable": True,
        "traces": len(trace_docs),
        "observations": len(obs_docs),
        "nodes_hydrated": len(entities),
        "relations_hydrated": len(rels),
    }


def _sync_technitium(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Technitium DNS zones + records as :DnsZone / :DnsRecord (CONCEPT:AU-KG.compute.technitium-dns-zones-records).

    Lists zones via ``technitium_dns_zones`` (action=list_zones), then per zone lists its
    records (action=get_records, list_zone=true). Each zone → a :DnsZone; each record →
    a :DnsRecord ``part_of`` its zone. Dict-shaped Technitium envelope (``response.zones`` /
    ``response.records``) → calls the tool directly via ``call_tool_once``. Full snapshot
    each run (DNS is small); the write-layer content-hash makes a re-run a no-op.
    """
    server = _configured_server(("technitium-dns-mcp", "technitium-dns"))
    if server is None:
        return {"status": "skipped", "reason": "technitium-dns-mcp not in mcp_config"}
    from ...protocols.source_connectors.connectors.mcp_package import _run_async
    from ...protocols.source_connectors.connectors.mcp_tool import call_tool_once
    from ...protocols.source_connectors.connectors.rest import _dig

    def _call(action: str, params: dict[str, Any]) -> Any:
        return _run_async(
            call_tool_once(
                server=server,
                tool="technitium_dns_zones",
                action=action,
                params=params,
            )
        )

    zones_res = _call("list_zones", {})
    zones = (
        (_dig(zones_res, "response.zones") or []) if isinstance(zones_res, dict) else []
    )
    entities: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []
    records_total = 0
    for zone in zones:
        if not isinstance(zone, dict):
            continue
        zname = zone.get("name")
        if not zname:
            continue
        zone_node = f"technitium:zone:{zname}"
        entities.append(
            {
                "id": zone_node,
                "type": "dns_zone",
                "name": zname,
                "zone_type": zone.get("type"),
                "disabled": zone.get("disabled"),
                "domain": "technitium",
                "source_system": "technitium",
                "externalToolId": zname,
            }
        )
        try:
            rec_res = _call(
                "get_records", {"domain": zname, "zone": zname, "list_zone": True}
            )
        except Exception as exc:  # noqa: BLE001 — one bad zone never aborts the rest
            logger.warning(
                "[KG-2.157] technitium records fetch failed for %s: %s", zname, exc
            )
            continue
        records = (
            (_dig(rec_res, "response.records") or [])
            if isinstance(rec_res, dict)
            else []
        )
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rname = rec.get("name")
            rtype = rec.get("type")
            rdata = rec.get("rData")
            value = ""
            if isinstance(rdata, dict):
                value = str(
                    rdata.get("ipAddress")
                    or rdata.get("value")
                    or rdata.get("text")
                    or ""
                )
            rec_node = f"technitium:rec:{zname}:{rname}:{rtype}:{value}"
            entities.append(
                {
                    "id": rec_node,
                    "type": "dns_record",
                    "name": f"{rname} {rtype}".strip(),
                    "record_type": rtype,
                    "ttl": rec.get("ttl"),
                    "value": value,
                    "disabled": rec.get("disabled"),
                    "domain": "technitium",
                    "source_system": "technitium",
                }
            )
            rels.append(
                {
                    "source": rec_node,
                    "target": zone_node,
                    "type": "part_of",
                    "domain": "technitium",
                }
            )
            records_total += 1
    if entities:
        engine.ingest_external_batch("technitium", entities, rels)
    return {
        "status": "ok",
        "source": "technitium",
        "mode": mode,
        "delta_capable": False,
        "zones": len(zones),
        "records": records_total,
        "nodes_hydrated": len(entities),
        "relations_hydrated": len(rels),
    }


def _sync_tunnel_manager(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest tunnel-manager host inventory as :Host / :Tunnel (CONCEPT:AU-KG.compute.tunnel-manager-hosts).

    Calls ``tm_hosts`` (action=list) — a dict ``{"hosts": {alias: HostConfig}}`` (args-style,
    not a record list) — so it goes through ``call_tool_once`` directly. Each alias → a
    :Host (hostname/user/port + any ``extra_config`` inventory keys); a configured
    ``proxy_command`` (a jump/tunnel) → a :Tunnel the host ``connects_via``.
    """
    server = _configured_server(("tunnel-manager-mcp", "tunnel-manager"))
    if server is None:
        return {"status": "skipped", "reason": "tunnel-manager-mcp not in mcp_config"}
    from ...protocols.source_connectors.connectors.mcp_package import _run_async
    from ...protocols.source_connectors.connectors.mcp_tool import call_tool_once

    res = _run_async(
        call_tool_once(
            server=server,
            tool="tm_hosts",
            params={"action": "list"},
            params_style="args",
            action="",
        )
    )
    hosts = (res.get("hosts") if isinstance(res, dict) else None) or {}
    entities: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []
    for alias, cfg in hosts.items():
        if not isinstance(cfg, dict):
            continue
        extra = ec if isinstance((ec := cfg.get("extra_config")), dict) else {}
        host_node = f"tunnel:host:{alias}"
        entities.append(
            {
                "id": host_node,
                "type": "host",
                "name": str(alias),
                "hostname": cfg.get("hostname"),
                "ssh_user": cfg.get("user"),
                "ssh_port": cfg.get("port"),
                "group": extra.get("group") or extra.get("ansible_group"),
                "ip_address": extra.get("ansible_host") or cfg.get("hostname"),
                "domain": "tunnel_manager",
                "source_system": "tunnel_manager",
                "externalToolId": str(alias),
            }
        )
        if proxy := cfg.get("proxy_command"):
            tun_node = f"tunnel:link:{alias}"
            entities.append(
                {
                    "id": tun_node,
                    "type": "tunnel",
                    "name": f"tunnel:{alias}",
                    "proxy_command": str(proxy),
                    "domain": "tunnel_manager",
                    "source_system": "tunnel_manager",
                }
            )
            rels.append(
                {
                    "source": host_node,
                    "target": tun_node,
                    "type": "connects_via",
                    "domain": "tunnel_manager",
                }
            )
    if entities:
        engine.ingest_external_batch("tunnel_manager", entities, rels)
    return {
        "status": "ok",
        "source": "tunnel_manager",
        "mode": mode,
        "delta_capable": False,
        "hosts": sum(1 for e in entities if e["type"] == "host"),
        "tunnels": sum(1 for e in entities if e["type"] == "tunnel"),
        "nodes_hydrated": len(entities),
        "relations_hydrated": len(rels),
    }


def _sync_uptime_kuma(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Uptime Kuma monitors + heartbeat stats as :Monitor / :HeartbeatStat
    (CONCEPT:AU-KG.compute.uptime-kuma-monitors).

    Calls ``uptime_kuma_monitors`` (action=get_monitors → bare list) and
    ``uptime_kuma_status`` (action=get_heartbeats → dict keyed by monitor id), both
    args-shaped, via ``call_tool_once``. Each monitor → a :Monitor; the latest heartbeat
    per monitor → a :HeartbeatStat ``part_of`` it (status/ping). Full snapshot each run;
    the write-layer content-hash makes unchanged monitors a no-op — for service-health and
    failure-pattern analysis over the KG.
    """
    server = _configured_server(("uptime-mcp", "uptime-kuma-mcp", "uptime-kuma-agent"))
    if server is None:
        return {"status": "skipped", "reason": "uptime-kuma server not in mcp_config"}
    from ...protocols.source_connectors.connectors.mcp_package import _run_async
    from ...protocols.source_connectors.connectors.mcp_tool import call_tool_once

    monitors = _run_async(
        call_tool_once(
            server=server,
            tool="uptime_kuma_monitors",
            params={"action": "get_monitors"},
            params_style="json",
            action="",
        )
    )
    # get_monitors may return a bare list OR a dict keyed by id, depending on the
    # uptime_kuma_api version — normalize both to a list of monitor dicts.
    if isinstance(monitors, dict):
        mon_list = [m for m in monitors.values() if isinstance(m, dict)]
    elif isinstance(monitors, list):
        mon_list = [m for m in monitors if isinstance(m, dict)]
    else:
        mon_list = []
    try:
        heartbeats = _run_async(
            call_tool_once(
                server=server,
                tool="uptime_kuma_status",
                params={"action": "get_heartbeats"},
                params_style="json",
                action="",
            )
        )
    except Exception:  # noqa: BLE001 — heartbeats are best-effort enrichment
        heartbeats = {}
    hb_map = heartbeats if isinstance(heartbeats, dict) else {}

    entities: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []
    for mon in mon_list:
        mid = mon.get("id")
        if mid is None:
            continue
        mon_node = f"uptime:monitor:{mid}"
        entities.append(
            {
                "id": mon_node,
                "type": "uptime_monitor",
                "name": mon.get("name") or f"Monitor {mid}",
                "url": mon.get("url"),
                "monitor_type": mon.get("type"),
                "active": mon.get("active"),
                "domain": "uptime_kuma",
                "source_system": "uptime_kuma",
                "externalToolId": str(mid),
            }
        )
        beats = hb_map.get(str(mid)) or hb_map.get(mid) or []
        last = beats[-1] if isinstance(beats, list) and beats else None
        if isinstance(last, dict):
            hb_node = f"uptime:hb:{mid}"
            entities.append(
                {
                    "id": hb_node,
                    "type": "heartbeat_stat",
                    "name": f"heartbeat:{mid}",
                    "up": last.get("status") == 1,
                    "ping": last.get("ping"),
                    "msg": last.get("msg"),
                    "domain": "uptime_kuma",
                    "source_system": "uptime_kuma",
                    "updatedAt": last.get("time"),
                }
            )
            rels.append(
                {
                    "source": hb_node,
                    "target": mon_node,
                    "type": "part_of",
                    "domain": "uptime_kuma",
                }
            )
    if entities:
        engine.ingest_external_batch("uptime_kuma", entities, rels)
    return {
        "status": "ok",
        "source": "uptime_kuma",
        "mode": mode,
        "delta_capable": False,
        "monitors": len(mon_list),
        "nodes_hydrated": len(entities),
        "relations_hydrated": len(rels),
    }


def _sync_home_assistant(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Home Assistant entities/states as :Device / :Entity (CONCEPT:AU-KG.compute.home-assistant-states).

    Drains the ``home-assistant-states`` preset (action=list_states → bare list) over
    ``home-assistant-mcp``. Each ``entity_id`` → an :Entity (state + attributes); its
    domain prefix (``light``/``sensor``/…) rolls up to a :Device the entity is ``part_of``.
    Delta = the ``last_updated`` watermark.
    """
    if not _server_configured(("home-assistant-mcp", "home-assistant-agent")):
        return {"status": "skipped", "reason": "home-assistant-mcp not in mcp_config"}
    backend = getattr(engine, "backend", None)
    wm_key = "home_assistant"
    since = None if mode == "full" else _read_watermark(backend, wm_key)
    docs = _drain_preset("home-assistant-states")
    entities: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []
    devices: set[str] = set()
    for doc in docs:
        eid = getattr(doc, "id", None)
        if not eid:
            continue
        rec = rd if isinstance((rd := _record_of(doc)), dict) else {}
        attrs = a if isinstance((a := rec.get("attributes")), dict) else {}
        device_class = str(eid).split(".", 1)[0]  # light / sensor / switch / …
        ent_node = f"hass:entity:{eid}"
        entities.append(
            {
                "id": ent_node,
                "type": "entity",
                "name": attrs.get("friendly_name") or str(eid),
                "entity_id": str(eid),
                "state": rec.get("state"),
                "device_class": device_class,
                "domain": "home_assistant",
                "source_system": "home_assistant",
                "externalToolId": str(eid),
                "updatedAt": rec.get("last_updated"),
            }
        )
        dev_node = f"hass:device:{device_class}"
        if device_class not in devices:
            entities.append(
                {
                    "id": dev_node,
                    "type": "device",
                    "name": f"HA {device_class}",
                    "domain": "home_assistant",
                    "source_system": "home_assistant",
                }
            )
            devices.add(device_class)
        rels.append(
            {
                "source": ent_node,
                "target": dev_node,
                "type": "part_of",
                "domain": "home_assistant",
            }
        )
    _ingest_typed(
        engine,
        "home_assistant",
        entities,
        rels,
        wm_key=wm_key,
        since=since,
        watermark=_max_updated(docs),
    )
    return {
        "status": "ok",
        "source": "home_assistant",
        "mode": mode,
        "delta_capable": True,
        "entities": sum(1 for e in entities if e["type"] == "entity"),
        "devices": len(devices),
        "nodes_hydrated": len(entities),
        "relations_hydrated": len(rels),
    }


def _sync_twenty(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Twenty CRM people/companies/opportunities as :Person / :Company /
    :Opportunity (CONCEPT:AU-KG.compute.twenty-crm-people-companies).

    Drains the ``twenty-people`` / ``twenty-companies`` / ``twenty-opportunities`` presets
    over ``twenty-mcp``. People with a ``companyId`` are linked ``member_of`` their company;
    opportunities with a ``companyId`` are linked ``part_of`` it. Delta = the ``updatedAt``
    watermark across the three object types.
    """
    if not _server_configured(("twenty-mcp", "twenty")):
        return {"status": "skipped", "reason": "twenty-mcp not in mcp_config"}
    backend = getattr(engine, "backend", None)
    wm_key = "twenty"
    since = None if mode == "full" else _read_watermark(backend, wm_key)
    src = "twenty"

    people = _drain_preset("twenty-people")
    companies = _drain_preset("twenty-companies")
    opps = _drain_preset("twenty-opportunities")
    entities: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []

    def _company_id(rec: dict[str, Any]) -> str | None:
        cid = rec.get("companyId")
        if cid:
            return str(cid)
        company = rec.get("company")
        return (
            str(company["id"])
            if isinstance(company, dict) and company.get("id")
            else None
        )

    for doc in companies:
        cid = getattr(doc, "id", None)
        if not cid:
            continue
        rec = _record_of(doc)
        entities.append(
            {
                "id": f"twenty:company:{cid}",
                "type": "company",
                "name": rec.get("name") or f"Company {cid}",
                "domain": "twenty",
                "source_system": src,
                "externalToolId": str(cid),
                "updatedAt": rec.get("updatedAt"),
            }
        )
    for doc in people:
        pid = getattr(doc, "id", None)
        if not pid:
            continue
        rec = _record_of(doc)
        name = rec.get("name") or {}
        full = (
            f"{name.get('firstName', '')} {name.get('lastName', '')}".strip()
            if isinstance(name, dict)
            else str(name)
        )
        node_id = f"twenty:person:{pid}"
        entities.append(
            {
                "id": node_id,
                "type": "person",
                "name": full or f"Person {pid}",
                "job_title": rec.get("jobTitle"),
                "domain": "twenty",
                "source_system": src,
                "externalToolId": str(pid),
                "updatedAt": rec.get("updatedAt"),
            }
        )
        if cid := _company_id(rec):
            rels.append(
                {
                    "source": node_id,
                    "target": f"twenty:company:{cid}",
                    "type": "member_of",
                    "domain": "twenty",
                }
            )
    for doc in opps:
        oid = getattr(doc, "id", None)
        if not oid:
            continue
        rec = _record_of(doc)
        node_id = f"twenty:opportunity:{oid}"
        entities.append(
            {
                "id": node_id,
                "type": "opportunity",
                "name": rec.get("name") or f"Opportunity {oid}",
                "stage": rec.get("stage"),
                "domain": "twenty",
                "source_system": src,
                "externalToolId": str(oid),
                "updatedAt": rec.get("updatedAt"),
            }
        )
        if cid := _company_id(rec):
            rels.append(
                {
                    "source": node_id,
                    "target": f"twenty:company:{cid}",
                    "type": "part_of",
                    "domain": "twenty",
                }
            )
    _ingest_typed(
        engine,
        src,
        entities,
        rels,
        wm_key=wm_key,
        since=since,
        watermark=_max_updated(people + companies + opps),
    )
    return {
        "status": "ok",
        "source": "twenty",
        "mode": mode,
        "delta_capable": True,
        "people": len(people),
        "companies": len(companies),
        "opportunities": len(opps),
        "nodes_hydrated": len(entities),
        "relations_hydrated": len(rels),
    }


# ── Media / finance / document / genealogy connectors as typed OWL entities ──────
# (CONCEPT:AU-KG.compute.audiobookshelf-libraries-books-authors–2.166)
#
# Four more first-class delta connectors reaching their upstream ONLY through a fleet
# ``*-mcp`` server (same contract as jira/dockerhub/twenty) and rebuilding **typed**
# entities mapped to OWL classes — not generic Documents.
#
# CONCEPT:AU-KG.compute.audiobookshelf-libraries-books-authors — Audiobookshelf libraries/books/authors → :Library / :Book / :Author
# CONCEPT:AU-KG.compute.firefly-iii-accounts-transactions — Firefly III accounts/transactions/budgets → :Account / :Transaction / :Budget
# CONCEPT:AU-KG.compute.paperless-ngx-documents-correspondents — Paperless-ngx documents/correspondents/tags → :Document / :Correspondent / :Tag
# CONCEPT:AU-KG.compute.gramps-web-people-families — Gramps Web people/families/events → :Person / :Family / :Event


def _sync_audiobookshelf(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Audiobookshelf libraries/books/authors as :Library / :Book / :Author
    (CONCEPT:AU-KG.compute.audiobookshelf-libraries-books-authors).

    Multi-step over ``audiobookshelf-mcp``: ``library_operations(action=list)`` →
    ``{"libraries": [...]}``; per library ``action=items`` → ``{"results": [...]}`` (each
    library item is a :Book ``part_of`` its :Library) and ``action=authors`` →
    ``{"authors": [...]}`` (each :Author, with books linked ``authored_by``). Dict-shaped /
    multi-step → calls the tool directly via ``call_tool_once``. Full snapshot each run; the
    write-layer content-hash makes a re-run a no-op.
    """
    server = _configured_server(("audiobookshelf-mcp", "audiobookshelf-agent"))
    if server is None:
        return {"status": "skipped", "reason": "audiobookshelf-mcp not in mcp_config"}
    from ...protocols.source_connectors.connectors.mcp_package import _run_async
    from ...protocols.source_connectors.connectors.mcp_tool import call_tool_once

    def _call(action: str, params: dict[str, Any]) -> Any:
        return _run_async(
            call_tool_once(
                server=server,
                tool="library_operations",
                action=action,
                params=params,
            )
        )

    libs_res = _call("list", {})
    libraries = (
        libs_res.get("libraries") if isinstance(libs_res, dict) else None
    ) or []
    if isinstance(libs_res, list):  # some ABS builds return a bare list
        libraries = libs_res
    entities: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []
    books_total = 0
    authors_total = 0
    for lib in libraries:
        if not isinstance(lib, dict):
            continue
        lib_id = lib.get("id")
        if not lib_id:
            continue
        lib_node = f"abs:library:{lib_id}"
        entities.append(
            {
                "id": lib_node,
                "type": "library",
                "name": lib.get("name") or f"Library {lib_id}",
                "media_type": lib.get("mediaType"),
                "domain": "audiobookshelf",
                "source_system": "audiobookshelf",
                "externalToolId": str(lib_id),
            }
        )
        try:
            items_res = _call("items", {"id": lib_id, "limit": 500})
        except Exception as exc:  # noqa: BLE001 — one bad library never aborts the rest
            logger.warning(
                "[KG-2.163] audiobookshelf items fetch failed for %s: %s", lib_id, exc
            )
            items_res = {}
        items = (
            items_res.get("results") if isinstance(items_res, dict) else None
        ) or []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if not item_id:
                continue
            media = m if isinstance((m := item.get("media")), dict) else {}
            meta = mm if isinstance((mm := media.get("metadata")), dict) else {}
            title = meta.get("title") or item.get("title") or f"Book {item_id}"
            book_node = f"abs:book:{item_id}"
            entities.append(
                {
                    "id": book_node,
                    "type": "book",
                    "name": str(title),
                    "subtitle": meta.get("subtitle"),
                    "isbn": meta.get("isbn"),
                    "asin": meta.get("asin"),
                    "publisher": meta.get("publisher"),
                    "published_year": meta.get("publishedYear"),
                    "duration": media.get("duration"),
                    "domain": "audiobookshelf",
                    "source_system": "audiobookshelf",
                    "externalToolId": str(item_id),
                    "updatedAt": item.get("updatedAt"),
                }
            )
            rels.append(
                {
                    "source": book_node,
                    "target": lib_node,
                    "type": "part_of",
                    "domain": "audiobookshelf",
                }
            )
            for author in meta.get("authors") or []:
                if not isinstance(author, dict):
                    continue
                aid = author.get("id")
                aname = author.get("name")
                if not (aid or aname):
                    continue
                author_node = f"abs:author:{aid or aname}"
                rels.append(
                    {
                        "source": book_node,
                        "target": author_node,
                        "type": "authored_by",
                        "domain": "audiobookshelf",
                    }
                )
            books_total += 1
        try:
            authors_res = _call("authors", {"id": lib_id})
        except Exception:  # noqa: BLE001 — authors are best-effort enrichment
            authors_res = {}
        authors = (
            authors_res.get("authors") if isinstance(authors_res, dict) else None
        ) or []
        for author in authors:
            if not isinstance(author, dict):
                continue
            aid = author.get("id")
            aname = author.get("name")
            if not (aid or aname):
                continue
            author_node = f"abs:author:{aid or aname}"
            entities.append(
                {
                    "id": author_node,
                    "type": "author",
                    "name": aname or f"Author {aid}",
                    "num_books": author.get("numBooks"),
                    "domain": "audiobookshelf",
                    "source_system": "audiobookshelf",
                    "externalToolId": str(aid or aname),
                }
            )
            authors_total += 1
    if entities:
        engine.ingest_external_batch("audiobookshelf", entities, rels)
    return {
        "status": "ok",
        "source": "audiobookshelf",
        "mode": mode,
        "delta_capable": False,
        "libraries": sum(1 for e in entities if e["type"] == "library"),
        "books": books_total,
        "authors": authors_total,
        "nodes_hydrated": len(entities),
        "relations_hydrated": len(rels),
    }


def _sync_firefly_iii(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Firefly III accounts/transactions/budgets as :Account / :Transaction /
    :Budget (CONCEPT:AU-KG.compute.firefly-iii-accounts-transactions).

    Drains the ``firefly-accounts`` / ``firefly-transactions`` / ``firefly-budgets``
    presets over ``firefly-iii-mcp``. Each JSON:API record's ``attributes`` block carries
    the real fields. A transaction's first split is linked ``part_of`` its source :Account.
    Delta = the ``updated_at`` watermark across the three object types.

    Uses the shared transform primitives (CONCEPT:AU-KG.etl.transform-primitives) —
    :func:`~..etl.transforms.dig` for the JSON:API ``attributes`` envelope unwrap
    (replacing the handler-local ``_attrs`` helper), :func:`~..etl.transforms.coalesce`
    for name fallbacks, and :func:`~..etl.transforms.stable_id` for node ids.
    """
    if not _server_configured(("firefly-iii-mcp", "firefly-iii-agent")):
        return {"status": "skipped", "reason": "firefly-iii-mcp not in mcp_config"}
    from ..etl.transforms import coalesce, dig, stable_id

    backend = getattr(engine, "backend", None)
    wm_key = "firefly_iii"
    since = None if mode == "full" else _read_watermark(backend, wm_key)
    src = "firefly_iii"

    accounts = _drain_preset("firefly-accounts")
    transactions = _drain_preset("firefly-transactions")
    budgets = _drain_preset("firefly-budgets")
    entities: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []

    for doc in accounts:
        aid = getattr(doc, "id", None)
        if not aid:
            continue
        attrs = dig(_record_of(doc), "attributes", default={})
        entities.append(
            {
                "id": stable_id(aid, prefix="firefly:account"),
                "type": "account",
                "name": coalesce(attrs, "name", default=f"Account {aid}"),
                "account_type": attrs.get("type"),
                "account_role": attrs.get("account_role"),
                "currency_code": attrs.get("currency_code"),
                "current_balance": attrs.get("current_balance"),
                "domain": "firefly_iii",
                "source_system": src,
                "externalToolId": str(aid),
                "updatedAt": attrs.get("updated_at"),
            }
        )
    for doc in budgets:
        bid = getattr(doc, "id", None)
        if not bid:
            continue
        attrs = dig(_record_of(doc), "attributes", default={})
        entities.append(
            {
                "id": stable_id(bid, prefix="firefly:budget"),
                "type": "budget",
                "name": coalesce(attrs, "name", default=f"Budget {bid}"),
                "active": attrs.get("active"),
                "domain": "firefly_iii",
                "source_system": src,
                "externalToolId": str(bid),
                "updatedAt": attrs.get("updated_at"),
            }
        )
    for doc in transactions:
        tid = getattr(doc, "id", None)
        if not tid:
            continue
        attrs = dig(_record_of(doc), "attributes", default={})
        splits = attrs.get("transactions")
        first = splits[0] if isinstance(splits, list) and splits else {}
        first = first if isinstance(first, dict) else {}
        node_id = stable_id(tid, prefix="firefly:transaction")
        entities.append(
            {
                "id": node_id,
                "type": "transaction",
                "name": coalesce(attrs, "group_title")
                or coalesce(first, "description", default=f"Transaction {tid}"),
                "transaction_type": first.get("type"),
                "amount": first.get("amount"),
                "currency_code": first.get("currency_code"),
                "transaction_date": first.get("date"),
                "category_name": first.get("category_name"),
                "domain": "firefly_iii",
                "source_system": src,
                "externalToolId": str(tid),
                "updatedAt": attrs.get("updated_at"),
            }
        )
        if src_acct := first.get("source_id"):
            rels.append(
                {
                    "source": node_id,
                    "target": stable_id(src_acct, prefix="firefly:account"),
                    "type": "part_of",
                    "domain": "firefly_iii",
                }
            )
        if budget_id := first.get("budget_id"):
            rels.append(
                {
                    "source": node_id,
                    "target": stable_id(budget_id, prefix="firefly:budget"),
                    "type": "member_of",
                    "domain": "firefly_iii",
                }
            )
    _ingest_typed(
        engine,
        src,
        entities,
        rels,
        wm_key=wm_key,
        since=since,
        watermark=_max_updated(accounts + transactions + budgets),
    )
    return {
        "status": "ok",
        "source": "firefly_iii",
        "mode": mode,
        "delta_capable": True,
        "accounts": len(accounts),
        "transactions": len(transactions),
        "budgets": len(budgets),
        "nodes_hydrated": len(entities),
        "relations_hydrated": len(rels),
    }


def _sync_paperless_ngx(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Paperless-ngx documents/correspondents/tags as :Document / :Correspondent /
    :Tag (CONCEPT:AU-KG.compute.paperless-ngx-documents-correspondents).

    Drains the ``paperless-documents`` / ``paperless-correspondents`` / ``paperless-tags``
    presets over ``paperless-ngx-mcp`` (each tool paginates internally → a flat list). Each
    document → a :Document linked ``member_of`` its :Correspondent and ``tagged_with`` each
    :Tag. Delta = the ``modified`` watermark on documents.

    Uses the shared transform primitives (CONCEPT:AU-KG.etl.transform-primitives) —
    :func:`~..etl.transforms.coalesce` for name fallbacks and
    :func:`~..etl.transforms.stable_id` for the ``paperless:<type>:<id>`` node ids.
    """
    if not _server_configured(("paperless-ngx-mcp", "paperless-ngx-agent")):
        return {"status": "skipped", "reason": "paperless-ngx-mcp not in mcp_config"}
    from ..etl.transforms import coalesce, stable_id

    backend = getattr(engine, "backend", None)
    wm_key = "paperless_ngx"
    since = None if mode == "full" else _read_watermark(backend, wm_key)
    src = "paperless_ngx"

    correspondents = _drain_preset("paperless-correspondents")
    tags = _drain_preset("paperless-tags")
    documents = _drain_preset("paperless-documents")
    entities: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []

    for doc in correspondents:
        cid = getattr(doc, "id", None)
        if cid is None:
            continue
        rec = _record_of(doc)
        entities.append(
            {
                "id": stable_id(cid, prefix="paperless:correspondent"),
                "type": "correspondent",
                "name": coalesce(rec, "name", default=f"Correspondent {cid}"),
                "document_count": rec.get("document_count"),
                "domain": "paperless_ngx",
                "source_system": src,
                "externalToolId": str(cid),
            }
        )
    for doc in tags:
        tid = getattr(doc, "id", None)
        if tid is None:
            continue
        rec = _record_of(doc)
        entities.append(
            {
                "id": stable_id(tid, prefix="paperless:tag"),
                "type": "tag",
                "name": coalesce(rec, "name", default=f"Tag {tid}"),
                "color": rec.get("color"),
                "domain": "paperless_ngx",
                "source_system": src,
                "externalToolId": str(tid),
            }
        )
    for doc in documents:
        did = getattr(doc, "id", None)
        if did is None:
            continue
        rec = _record_of(doc)
        node_id = stable_id(did, prefix="paperless:document")
        entities.append(
            {
                "id": node_id,
                "type": "document",
                "name": coalesce(rec, "title", default=f"Document {did}"),
                "created": rec.get("created"),
                "added": rec.get("added"),
                "archive_serial_number": rec.get("archive_serial_number"),
                "domain": "paperless_ngx",
                "source_system": src,
                "externalToolId": str(did),
                "updatedAt": rec.get("modified"),
            }
        )
        if (corr := rec.get("correspondent")) is not None:
            rels.append(
                {
                    "source": node_id,
                    "target": stable_id(corr, prefix="paperless:correspondent"),
                    "type": "member_of",
                    "domain": "paperless_ngx",
                }
            )
        for tag_id in rec.get("tags") or []:
            rels.append(
                {
                    "source": node_id,
                    "target": stable_id(tag_id, prefix="paperless:tag"),
                    "type": "tagged_with",
                    "domain": "paperless_ngx",
                }
            )
    _ingest_typed(
        engine,
        src,
        entities,
        rels,
        wm_key=wm_key,
        since=since,
        watermark=_max_updated(documents),
    )
    return {
        "status": "ok",
        "source": "paperless_ngx",
        "mode": mode,
        "delta_capable": True,
        "documents": len(documents),
        "correspondents": len(correspondents),
        "tags": len(tags),
        "nodes_hydrated": len(entities),
        "relations_hydrated": len(rels),
    }


def _sync_gramps(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Gramps Web people/families/events as :Person / :Family / :Event
    (CONCEPT:AU-KG.compute.gramps-web-people-families).

    Calls ``gramps_people`` / ``gramps_families`` / ``gramps_events`` (action
    ``get_*``) directly via ``call_tool_once`` — each returns the ``Response`` envelope whose
    ``data`` is the decoded collection. Each person → a :Person; each family → a :Family the
    person is ``member_of`` (father/mother/children handles); each event → an :Event a person
    ``part_of`` (via the person's ``event_ref_list``). Full snapshot each run; the write-layer
    content-hash makes a re-run a no-op. The genealogy graph is the substrate for relationship
    reasoning over the KG.
    """
    server = _configured_server(("gramps-mcp", "gramps-agent"))
    if server is None:
        return {"status": "skipped", "reason": "gramps-mcp not in mcp_config"}
    from ...protocols.source_connectors.connectors.mcp_package import _run_async
    from ...protocols.source_connectors.connectors.mcp_tool import call_tool_once
    from ...protocols.source_connectors.connectors.rest import _dig

    def _collection(tool: str, action: str) -> list[dict[str, Any]]:
        # Fail-soft per collection: a connector that doesn't expose one list action
        # (if a connector doesn't expose one list action) must not sink the
        # whole sync — skip that collection and ingest the others.
        try:
            res = _run_async(
                call_tool_once(
                    server=server,
                    tool=tool,
                    action=action,
                    params={"pagesize": 500},
                )
            )
        except Exception as exc:
            logger.warning("gramps: %s/%s unavailable, skipping: %s", tool, action, exc)
            return []
        data = _dig(res, "data") if isinstance(res, dict) else res
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []

    people = _collection("gramps_people", "get_people")
    families = _collection("gramps_families", "get_families")
    events = _collection("gramps_events", "get_events")
    entities: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []

    def _person_name(rec: dict[str, Any]) -> str:
        name = rec.get("primary_name")
        if isinstance(name, dict):
            first = name.get("first_name") or ""
            surnames = name.get("surname_list") or []
            last = ""
            if isinstance(surnames, list) and surnames:
                s0 = surnames[0]
                last = s0.get("surname", "") if isinstance(s0, dict) else ""
            full = f"{first} {last}".strip()
            if full:
                return full
        return rec.get("gramps_id") or rec.get("handle") or "Person"

    person_handles: set[str] = set()
    for rec in people:
        handle = rec.get("handle")
        if not handle:
            continue
        person_handles.add(str(handle))
        node_id = f"gramps:person:{handle}"
        entities.append(
            {
                "id": node_id,
                "type": "person",
                "name": _person_name(rec),
                "gramps_id": rec.get("gramps_id"),
                "gender": rec.get("gender"),
                "domain": "gramps",
                "source_system": "gramps",
                "externalToolId": str(handle),
                "updatedAt": rec.get("change"),
            }
        )
        for eref in rec.get("event_ref_list") or []:
            if not isinstance(eref, dict):
                continue
            if ev := eref.get("ref"):
                rels.append(
                    {
                        "source": node_id,
                        "target": f"gramps:event:{ev}",
                        "type": "part_of",
                        "domain": "gramps",
                    }
                )
    for rec in families:
        handle = rec.get("handle")
        if not handle:
            continue
        fam_node = f"gramps:family:{handle}"
        entities.append(
            {
                "id": fam_node,
                "type": "family",
                "name": rec.get("gramps_id") or f"Family {handle}",
                "gramps_id": rec.get("gramps_id"),
                "relationship": (
                    (rec.get("type") or {}).get("string")
                    if isinstance(rec.get("type"), dict)
                    else None
                ),
                "domain": "gramps",
                "source_system": "gramps",
                "externalToolId": str(handle),
                "updatedAt": rec.get("change"),
            }
        )
        members: list[Any] = [rec.get("father_handle"), rec.get("mother_handle")]
        for child in rec.get("child_ref_list") or []:
            if isinstance(child, dict) and child.get("ref"):
                members.append(child["ref"])
        for member in members:
            if member:
                rels.append(
                    {
                        "source": f"gramps:person:{member}",
                        "target": fam_node,
                        "type": "member_of",
                        "domain": "gramps",
                    }
                )
    for rec in events:
        handle = rec.get("handle")
        if not handle:
            continue
        entities.append(
            {
                "id": f"gramps:event:{handle}",
                "type": "event",
                "name": (
                    (rec.get("type") or {}).get("string")
                    if isinstance(rec.get("type"), dict)
                    else rec.get("gramps_id")
                )
                or f"Event {handle}",
                "gramps_id": rec.get("gramps_id"),
                "description": rec.get("description"),
                "domain": "gramps",
                "source_system": "gramps",
                "externalToolId": str(handle),
                "updatedAt": rec.get("change"),
            }
        )
    if entities:
        engine.ingest_external_batch("gramps", entities, rels)
    return {
        "status": "ok",
        "source": "gramps",
        "mode": mode,
        "delta_capable": False,
        "people": len(people),
        "families": len(families),
        "events": len(events),
        "nodes_hydrated": len(entities),
        "relations_hydrated": len(rels),
    }


# MCP-backed dedicated trackers (CONCEPT:AU-KG.compute.mcp-backed-dedicated-trackers) — each reaches its upstream ONLY
# through a fleet ``*-mcp`` server (never a direct vendor client / env token), so unlike
# the capability-registry sources (env-token configured) and the always-local feed/fleet
# handlers, their "configured" signal is *"the server is registered in mcp_config.json"*.
# Maps the delta source → the candidate ``server`` keys to probe (the handler's
# ``default_server``; per-instance overrides are unioned in at sweep time). Keep in sync
# with the ``default_server`` of each ``_resolve_tracker_instances`` call.
_MCP_TRACKER_SERVERS: dict[str, tuple[str, ...]] = {
    "jira": ("atlassian-mcp",),
    "confluence": ("atlassian-mcp",),
    "plane": ("plane-mcp",),
    # Ops / platform typed connectors (CONCEPT:AU-KG.compute.dockerhub-repositories–2.161) — server-configured, so the
    # sweep keeps each candidate only when its ``*-mcp`` server is in mcp_config (else drops
    # it, never mis-reporting an unconfigured connector as failed work).
    "dockerhub": ("dockerhub-mcp", "dockerhub-api"),
    "langfuse": ("langfuse-mcp", "langfuse-agent"),
    "technitium": ("technitium-dns-mcp", "technitium-dns"),
    "tunnel_manager": ("tunnel-manager-mcp", "tunnel-manager"),
    "uptime_kuma": ("uptime-mcp", "uptime-kuma-agent", "uptime-kuma-mcp"),
    "home_assistant": ("home-assistant-mcp", "home-assistant-agent"),
    "twenty": ("twenty-mcp", "twenty"),
    # Media / finance / document / genealogy connectors (CONCEPT:AU-KG.compute.audiobookshelf-libraries-books-authors–2.166)
    "audiobookshelf": ("audiobookshelf-mcp", "audiobookshelf-agent"),
    "firefly_iii": ("firefly-iii-mcp", "firefly-iii-agent"),
    "paperless_ngx": ("paperless-ngx-mcp", "paperless-ngx-agent"),
    "gramps": ("gramps-mcp", "gramps-agent"),
}


def _mcp_server_configured(servers: dict[str, Any], name: str) -> bool:
    """True when ``name`` (or ``<name>-mcp``) is registered in the loaded mcp_config
    ``mcpServers`` map — mirrors the connector's own transport resolution
    (:meth:`McpToolSourceConnector` server lookup), so "candidate" and "reachable"
    agree on what counts as configured."""
    if not name:
        return False
    return name in servers or f"{name}-mcp" in servers


def _tracker_instance_servers(field: str, default_server: str) -> tuple[str, ...]:
    """Servers a tracker delta source will actually reach: the per-instance ``server``
    overrides from a configured ``*_instances`` config row, else the ``default_server``.
    Lets a sweep recognise a second Atlassian site / Plane workspace as configured."""
    try:
        from ...core.config import config as cfg

        rows = [r for r in (getattr(cfg, field, None) or []) if isinstance(r, dict)]
        servers = tuple(str(r.get("server") or default_server) for r in rows)
        if servers:
            return servers
    except Exception:  # noqa: BLE001 — config probe is best-effort
        pass
    return (default_server,)


def _mcp_tracker_configured(source: str) -> bool:
    """True when an MCP-backed dedicated tracker (jira/confluence/plane) is configured
    for the sweep — i.e. at least one server it would reach is registered in
    ``mcp_config.json``. Unknown sources default to *configured* (no extra gate)."""
    default_servers = _MCP_TRACKER_SERVERS.get(source)
    if default_servers is None:
        return True
    try:
        from ...protocols.source_connectors.connectors.mcp_package import (
            _load_mcp_config,
        )

        servers = _load_mcp_config() or {}
    except Exception:  # noqa: BLE001 — no config readable → not configured here
        return False
    _INST_FIELD = {
        "jira": "jira_instances",
        "confluence": "confluence_instances",
        "plane": "plane_instances",
    }
    candidate_servers: set[str] = set(default_servers)
    inst_field = _INST_FIELD.get(source)
    if inst_field:
        # Multi-instance trackers union in per-instance ``server`` overrides so a second
        # Atlassian site / Plane workspace counts as configured; the ops/platform connectors
        # (KG-2.155+) are single-server, so their default candidate tuple is authoritative.
        for default_server in default_servers:
            candidate_servers.update(
                _tracker_instance_servers(inst_field, default_server)
            )
    return any(_mcp_server_configured(servers, s) for s in candidate_servers)


# ── ARD registry delta handler (CONCEPT:AU-KG.ingest.source-sync-canonical) ────────────────────────────


def _resolve_ard_registries() -> list[dict[str, Any]]:
    """Resolve configured external ARD registries from ``ARD_REGISTRIES``.

    The value is a JSON list of ``{name, preset|catalog_url, search_url?, media_types?}``
    objects (a bare string item is treated as a preset name), so an operator points the
    consume side at HF + any peer registry with one config key.
    """
    import json as _json

    from ...core.config import setting

    raw = (setting("ARD_REGISTRIES", default="") or "").strip()
    if not raw:
        return []
    try:
        data = _json.loads(raw)
    except Exception:  # noqa: BLE001 — malformed config ⇒ no registries
        return []
    items = data if isinstance(data, list) else [data]
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            out.append({"name": item, "preset": item})
        elif isinstance(item, dict):
            out.append(dict(item))
    return out


def _ard_entities(
    docs: list[Any], registry_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map drained ARD resource docs → typed KG entities + relationships (KG-2.188).

    ``application/mcp-server*`` → ``:MCPServer``; ``application/ai-skill`` → ``:Skill``;
    every resource links ``registeredIn`` its ``:ResourceRegistry`` and ``providesCapability``
    a ``:ServiceCapability`` per tag — reusing the a2a/capability ontology terms so an
    ingested external capability is queryable exactly like a native one.
    """
    import re as _re

    def _slug(value: str) -> str:
        return _re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "x"

    entities: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []
    src = f"ard:{registry_name}"
    registry_node = f"ard:registry:{_slug(registry_name)}"
    entities.append(
        {
            "id": registry_node,
            "type": "ResourceRegistry",
            "name": registry_name,
            "domain": "ard",
            "source_system": src,
        }
    )
    for doc in docs:
        eid = getattr(doc, "id", None)
        if not eid:
            continue
        meta = getattr(doc, "metadata", None) or {}
        record = r if isinstance((r := meta.get("record")), dict) else {}
        media = str((meta or {}).get("ard_media_type") or "")
        node_type = "Skill" if media == "application/ai-skill" else "MCPServer"
        node_id = f"ard:{registry_name}:{_slug(eid)}"
        publisher_domain = str((record.get("publisher") or {}).get("domain", ""))
        entities.append(
            {
                "id": node_id,
                "type": node_type,
                "name": getattr(doc, "title", None) or str(eid),
                "description": getattr(doc, "text", "") or "",
                "domain": "ard",
                "source_system": src,
                "externalToolId": str(eid),
                "ardMediaType": media,
                "publisherDomain": publisher_domain,
                "updatedAt": getattr(doc, "updated_at", None),
            }
        )
        rels.append(
            {
                "source": node_id,
                "target": registry_node,
                "type": "registeredIn",
                "domain": "ard",
            }
        )
        for tag in record.get("tags") or []:
            cap = str(tag).strip().lower()
            if not cap:
                continue
            cap_node = f"capability:{_slug(cap)}"
            entities.append(
                {
                    "id": cap_node,
                    "type": "ServiceCapability",
                    "name": cap,
                    "domain": "ard",
                    "source_system": src,
                }
            )
            rels.append(
                {
                    "source": node_id,
                    "target": cap_node,
                    "type": "providesCapability",
                    "domain": "ard",
                }
            )
    return entities, rels


def _sync_ard(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest external ARD registries as typed discoverable resources (CONCEPT:AU-KG.ingest.source-sync-canonical).

    For every registry in ``ARD_REGISTRIES`` (e.g. ``[{"name":"hf","preset":"huggingface"}]``)
    this drains the ``ard`` connector (signature-verified), maps each resource to a typed
    ``:MCPServer``/``:Skill`` node linked to its ``:ResourceRegistry`` + capabilities, and
    ``ingest_external_batch``-es it under ``domain="ard"``. ``mode='reconcile'`` tombstones
    resources no longer present. ``client`` may inject a fetch function for offline tests.
    """
    from ...protocols.source_connectors.registry import build_connector

    registries = _resolve_ard_registries()
    if not registries:
        return {"status": "skipped", "reason": "no ARD_REGISTRIES configured"}

    backend = getattr(engine, "backend", None)
    results: list[dict[str, Any]] = []
    total_e = total_r = total_fail = 0
    all_live: set[str] = set()
    for reg in registries:
        name = str(reg.get("name") or reg.get("preset") or "ard")
        conf = {k: v for k, v in reg.items() if k != "name"}
        if callable(client):
            conf["fetch_fn"] = client
        try:
            conn = build_connector("ard", conf)
        except Exception as exc:  # noqa: BLE001 — a misconfigured registry is a skip
            results.append(
                {"registry": name, "status": "skipped", "reason": str(exc)[:160]}
            )
            continue
        wm_key = f"ard:{name}"
        since = None if mode == "full" else _read_watermark(backend, wm_key)
        docs = _drain_incremental(conn, since)
        live = {str(getattr(d, "id", "")) for d in docs if getattr(d, "id", None)}
        all_live |= live
        if mode == "reconcile":
            results.append(_reconcile(engine, "ard", live) | {"registry": name})
            continue
        entities, rels = _ard_entities(docs, name)
        if entities:
            engine.ingest_external_batch("ard", entities, rels)
        watermark = _max_updated(docs)
        if watermark and (since is None or str(watermark) > str(since)):
            _write_watermark(backend, wm_key, watermark)
        fails = int(getattr(conn, "verify_failures", 0) or 0)
        total_e += len(entities)
        total_r += len(rels)
        total_fail += fails
        results.append(
            {
                "registry": name,
                "resources": len(docs),
                "verify_failures": fails,
                "watermark": watermark,
            }
        )
    return {
        "status": "ok",
        "source": "ard",
        "mode": mode,
        "delta_capable": True,
        "registries": results,
        "nodes_hydrated": total_e,
        "relations_hydrated": total_r,
        "verify_failures": total_fail,
    }


def _parse_memory_file(path: Any) -> tuple[str, str, str, str, str, list[str]]:
    """Parse a Claude Code memory markdown file into
    ``(slug, name, description, memory_type, body, links)``.

    Reads the ``name`` / ``description`` / ``metadata.type`` YAML frontmatter (dependency-
    free — a tiny line scan, no yaml import) and the ``[[other-slug]]`` wiki-links in the
    body. Anything missing falls back to the filename stem / sensible defaults.
    """
    import re

    text = path.read_text(encoding="utf-8", errors="replace")
    slug = path.stem
    name, description, mtype, body = slug, "", "memory", text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if m:
        fm, body = m.group(1), m.group(2)
        for line in fm.splitlines():
            key, _, val = line.partition(":")
            k, v = key.strip(), val.strip()
            if k == "name" and v:
                name = v
            elif k == "description":
                description = v
            elif k == "type" and v:  # ``metadata.type`` (indented) or a top-level type
                mtype = v
    links = re.findall(r"\[\[([a-z0-9][a-z0-9-]*)\]\]", body)
    return slug, name, description, mtype, body.strip(), links


def _sync_claude_memory(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest the Claude Code file-based memory (the ``MEMORY.md`` topic files) into the KG
    as typed ``:AgentMemory`` nodes (CONCEPT:AU-KG.ingest.claude-memory-connector).

    The harness keeps its cross-session memory as flat markdown outside the graph; this
    dogfoods our OWN memory substrate — each topic file becomes a semantically-searchable
    ``:AgentMemory`` node (name/type/description/body embedded, findable via ``graph_search``)
    and its ``[[other-slug]]`` wiki-links become ``RELATED_TO`` edges, so the session
    knowledge is connected to the rest of the ecosystem graph instead of stranded on disk.

    Zero-infra + offline (reads local markdown, no network). The memory dir is
    ``CLAUDE_MEMORY_DIR`` when set, else every ``~/.claude/projects/*/memory`` is swept.
    Delta is the content-hash write-delta in ``ingest_external_batch`` (unchanged topic
    files are skipped even on a full sweep); ``ids`` narrows to specific slugs. The
    ``MEMORY.md`` / ``MEMORY-ARCHIVE.md`` indexes themselves are skipped — only the
    per-memory topic files are ingested.
    """
    import glob
    import os
    from pathlib import Path

    from ...core.config import setting

    explicit = (setting("CLAUDE_MEMORY_DIR", default="") or "").strip()
    dirs = (
        [explicit]
        if explicit
        else sorted(glob.glob(os.path.expanduser("~/.claude/projects/*/memory")))
    )
    files: list[Any] = []
    for d in dirs:
        p = Path(d)
        if p.is_dir():
            files.extend(
                f
                for f in sorted(p.glob("*.md"))
                if f.name not in ("MEMORY.md", "MEMORY-ARCHIVE.md")
            )
    if not files:
        return {
            "status": "skipped",
            "reason": "no Claude memory dir (set CLAUDE_MEMORY_DIR) or no *.md topic files",
        }

    id_filter = set(ids or [])
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for f in files:
        slug, name, description, mtype, body, links = _parse_memory_file(f)
        if id_filter and slug not in id_filter:
            continue
        eid = f"claude_memory:{slug}"
        entities.append(
            {
                "id": eid,
                "type": "AgentMemory",
                "name": name,
                "slug": slug,
                "memory_type": mtype,
                "description": description,
                "text": (f"{description}\n\n{body}").strip(),
                "source_uri": str(f),
            }
        )
        for tgt in dict.fromkeys(links):  # de-dup, preserve order
            if tgt != slug:
                relationships.append(
                    {
                        "source": eid,
                        "target": f"claude_memory:{tgt}",
                        "type": "RELATED_TO",
                    }
                )

    result = (
        engine.ingest_external_batch("claude_memory", entities, relationships)
        if entities
        else {}
    )
    return {
        "status": "ok",
        "source": "claude_memory",
        "mode": mode,
        "delta_capable": True,
        "memories_seen": len(files),
        "nodes": result.get("nodes", 0),
        "edges": result.get("edges", 0),
        "skipped_unchanged": result.get("skipped_unchanged", 0),
    }


# Sources with a native delta (watermark/reconcile) handler. Add an entry here to
# make another source incremental (e.g. Camunda once its extractor takes `since`).
_DELTA_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "claude_memory": _sync_claude_memory,
    "leanix": _sync_leanix,
    "archivebox": _sync_archivebox,
    "gitlab": _sync_gitlab,
    "freshrss": _sync_freshrss,
    "rss": _sync_rss,
    "jira": _sync_jira,
    "confluence": _sync_confluence,
    "plane": _sync_plane,
    # Ops / platform connectors as typed OWL entities (CONCEPT:AU-KG.compute.dockerhub-repositories–2.161)
    "dockerhub": _sync_dockerhub,
    "langfuse": _sync_langfuse,
    "technitium": _sync_technitium,
    "tunnel_manager": _sync_tunnel_manager,
    "uptime_kuma": _sync_uptime_kuma,
    "home_assistant": _sync_home_assistant,
    "twenty": _sync_twenty,
    # Media / finance / document / genealogy connectors (CONCEPT:AU-KG.compute.audiobookshelf-libraries-books-authors–2.166)
    "audiobookshelf": _sync_audiobookshelf,
    "firefly_iii": _sync_firefly_iii,
    "paperless_ngx": _sync_paperless_ngx,
    "gramps": _sync_gramps,
    # External ARD registries (HF + peers) as typed discoverable resources (KG-2.188).
    "ard": _sync_ard,
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

    The heterogeneous result each of the ~20 ``_sync_*`` handlers / the
    materialize core / the capability-registry hydrate / the fleet sweep returns
    is coerced through :class:`..etl.result.EtlResult` (CONCEPT:AU-KG.etl.result-contract)
    at this single choke point — every dispatch path gets a validated ``status``/
    ``counts`` contract without any handler needing to be rewritten (their extra,
    handler-specific fields pass through unedited).
    """
    from ..etl.result import EtlResult

    norm_source = (source or "").lower().strip()
    res = _dispatch_sync_source(engine, norm_source, mode=mode, ids=ids, client=client)
    return EtlResult.coerce(res, source=norm_source or None, mode=mode).model_dump()


def _dispatch_sync_source(
    engine: Any,
    source: str,
    *,
    mode: str = "delta",
    ids: list[str] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """The raw dispatch logic for :func:`sync_source` (pre-``EtlResult`` coercion)."""

    # "all"/"*"/"sweep" → fan out across every configured connector in one pass
    # so the one entrypoint also covers "ingest everything" (CONCEPT:AU-KG.ingest.enterprise-source-extractor).
    if source in {"all", "*", "sweep"}:
        return sweep_all_sources(engine, mode=mode if mode in SYNC_ACTIONS else "delta")

    # CONCEPT:AU-KG.ontology.single-source-full-drain — a single-source FULL drain of a LARGE corpus must NOT run inline:
    # that would block the MCP/REST request until the whole backlog is drained (timeout) or
    # force a human/agent to hand-repeat delta waves. Normalize that ONE call into a stream of
    # capacity-guarded, paginated ``connector_drain`` batch-tasks and return a handle IMMEDIATELY
    # — the "controlled waves" are baked in, not hand-driven. Small/delta syncs stay inline (fast).
    if mode == "full" and hasattr(engine, "submit_task"):
        from .chunked_drain import (
            chunked_drain_enabled,
            start_chunked_drain,
            supports_chunked_drain,
        )

        if chunked_drain_enabled() and supports_chunked_drain(source):
            return start_chunked_drain(engine, source, mode="full")

    handler = _DELTA_HANDLERS.get(source)
    if handler is not None:
        try:
            return handler(engine, mode=mode, ids=ids, client=client)
        except Exception as exc:  # noqa: BLE001
            # An UNCONFIGURED upstream (its MCP server isn't in mcp_config, no
            # creds, etc.) is a *skip*, never a task failure — the fleet sweep
            # routinely runs with only a subset of connectors provisioned
            # (CONCEPT:AU-KG.ingest.enterprise-source-extractor). Real errors still propagate.
            msg = str(exc).lower()
            if any(
                t in msg
                for t in (
                    "not found in mcp_config",
                    "not configured",
                    "no client",
                    "missing",
                    "credential",
                    "unconfigured",
                )
            ):
                return {"status": "skipped", "source": source, "reason": str(exc)[:160]}
            raise

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
    """Ingest every *configured* connector in one background sweep (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

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

    # CONCEPT:AU-KG.compute.mcp-backed-dedicated-trackers — the MCP-backed dedicated trackers (jira/confluence/plane) reach
    # their upstream ONLY through a fleet ``*-mcp`` server, so their "configured" signal is
    # *"the server is registered in mcp_config.json"* — NOT an env token (capability-registry)
    # nor always-on (feed/fleet handlers). Keep one as a candidate when its server is in
    # mcp_config (the live remote-routed atlassian/plane case the operator runs), and DROP it
    # when truly unconfigured so the sweep neither wastes a connector_sync task nor misreports
    # a reachable tracker as missing. (Before this gate they were enqueued unconditionally,
    # so a tracker whose ``*-mcp`` server was absent under the expected key still spawned a
    # task that the connector then aborted with "not found in mcp_config" → 0 nodes, never
    # surfacing as configured work.)
    for _tracker in _MCP_TRACKER_SERVERS:
        if _tracker in candidates and not _mcp_tracker_configured(_tracker):
            candidates.discard(_tracker)

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

    # CONCEPT:AU-ORCH.dispatch.laned-sweep-fanout — fan the sweep out as LANED ``connector_sync`` tasks (the 'connectors'
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
