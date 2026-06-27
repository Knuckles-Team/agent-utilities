"""Local session collector (CONCEPT:ECO-4.38 / ECO-4.42).

The client-side half of remote ingest: discover installed agents, parse their
local logs into normalized bundles, then **sink** them — either writing to the
local usage store (local engine) or **pushing** to a central gateway/MCP when
the engine is remote. The parsing always happens where the files are, so the
server never needs filesystem access to the client.

Zero-config: agents and their log dirs are auto-detected; nothing to list.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from agent_utilities.core.config import setting
from agent_utilities.usage.models import ParsedSessionBundle

from .agent_sources import detect_installed, ensure_parsers_loaded
from .agent_sources.parsers import NONJSONL_PENDING

logger = logging.getLogger(__name__)


def iter_local_bundles(
    *, only_changed: bool = True, backend=None
) -> Iterator[tuple[str, int, int, ParsedSessionBundle]]:
    """Yield ``(path, mtime, size, bundle)`` for every parseable local session.

    When ``only_changed`` and a backend is given, the mtime/size skip-cache is
    consulted so unchanged files are skipped (O(1) re-sync).
    """
    ensure_parsers_loaded()
    for source in detect_installed():
        for path in source.discover():
            try:
                st = path.stat()
            except OSError:
                continue
            mtime, size = int(st.st_mtime), int(st.st_size)
            if (
                only_changed
                and backend is not None
                and not backend.should_sync(str(path), mtime, size)
            ):
                continue
            try:
                for bundle in source.parse(path):
                    yield str(path), mtime, size, bundle
            except Exception as exc:  # noqa: BLE001 — one bad file never aborts the sweep
                logger.debug("parse failed for %s: %s", path, exc)


def _engine_is_remote() -> bool:
    """True when the KG engine is a remote/client process (push instead of write).

    Detected via the daemon role and a configured remote engine endpoint; a
    local UDS/in-process engine returns False (fast in-process write path).
    """
    from agent_utilities.core.config import setting

    role = setting("KG_DAEMON_ROLE", "auto")
    if role == "client":
        return True
    endpoint = setting("GRAPH_ENGINE_ENDPOINT") or setting("EPISTEMIC_GRAPH_ENDPOINT")
    return bool(endpoint and ("://" in endpoint or ":" in endpoint))


def collect_local_sessions(*, only_changed: bool = True) -> dict:
    """Discover + parse + sink all local sessions. Returns a summary dict.

    Local engine → write straight to the usage store. Remote engine → push the
    bundles to the central gateway (HTTP) so the server stays filesystem-blind.
    """
    if _engine_is_remote():
        return push_local_sessions()

    from agent_utilities.usage import get_usage_backend
    from agent_utilities.usage.recorder import get_usage_recorder

    backend = get_usage_backend()
    recorder = get_usage_recorder()
    ingested = 0
    files = 0
    seen_files: set[str] = set()
    for path, mtime, size, bundle in iter_local_bundles(
        only_changed=only_changed, backend=backend
    ):
        if recorder.record_bundle(bundle):
            ingested += 1
        if path not in seen_files:
            files += 1
            seen_files.add(path)
        backend.mark_synced(path, mtime, size)
    return {
        "mode": "local",
        "files": files,
        "ingested": ingested,
        "agents": [s.agent_type for s in detect_installed()],
        "nonjsonl_pending": sorted(NONJSONL_PENDING),
    }


def push_local_sessions(
    *, gateway_url: str | None = None, tenant_id: str = "", batch: int = 50
) -> dict:
    """Push locally-parsed bundles to a central gateway upload endpoint.

    ``gateway_url`` defaults to ``USAGE_GATEWAY_URL`` (e.g.
    ``https://graph-os.arpa``). Bundles are sent in batches to
    ``/api/observability/sessions/upload``.
    """
    import httpx

    gateway_url = gateway_url or setting("USAGE_GATEWAY_URL", "")
    tenant_id = tenant_id or setting("USAGE_TENANT_ID", "")
    if not gateway_url:
        logger.warning("push_local_sessions: USAGE_GATEWAY_URL unset; nothing pushed")
        return {"mode": "push", "pushed": 0, "error": "no gateway url"}

    url = gateway_url.rstrip("/") + "/api/observability/sessions/upload"
    params = {"tenant_id": tenant_id} if tenant_id else {}
    pushed = 0
    files: set[str] = set()
    pending: list[dict] = []

    def _flush() -> None:
        nonlocal pushed
        if not pending:
            return
        try:
            resp = httpx.post(url, params=params, json=pending, timeout=60.0)
            resp.raise_for_status()
            pushed += int(resp.json().get("ingested", 0))
        except Exception as exc:  # noqa: BLE001
            logger.warning("upload push failed (%d bundles): %s", len(pending), exc)
        pending.clear()

    for path, _mtime, _size, bundle in iter_local_bundles(only_changed=False):
        files.add(path)
        pending.append(bundle.model_dump())
        if len(pending) >= batch:
            _flush()
    _flush()
    return {"mode": "push", "files": len(files), "pushed": pushed, "gateway": url}


def upload_local_sessions(
    *,
    server: str = "graph-os",
    url: str = "",
    tenant_id: str = "",
    batch: int = 50,
    only_changed: bool = False,
) -> dict:
    """Client-side parse → push to a REMOTE engine via the MCP ``ingest_sessions`` tool.

    This is the path for a **remote engine** (CONCEPT:ECO-4.42): the engine runs on
    another host and cannot read THIS client's local agent logs, so the *client*
    parses its own ``~/.claude/projects/**/*.jsonl`` and Antigravity
    (``~/.gemini/antigravity``) sessions into :class:`ParsedSessionBundle`\\s and
    pushes them, batched, to the remote graph-os ``ingest_sessions`` tool with
    ``action="upload"`` (which sinks them into the usage store + KG). Transport
    reuses the KG-2.59 fleet client (``server`` resolved through ``mcp_config.json``,
    or an explicit ``url``) so no gateway URL / bespoke HTTP client is required —
    fixing the ``collect``-on-remote ``"no gateway url"`` gap.

    Unlike ``collect_local_sessions`` (which only sinks remotely through the REST
    ``USAGE_GATEWAY_URL`` path), this drives the MCP surface directly and covers
    EVERY auto-detected agent, Claude and Antigravity included.
    """
    import json as _json

    from ..protocols.source_connectors.connectors.mcp_package import _run_async
    from ..protocols.source_connectors.connectors.mcp_tool import call_tool_once

    tenant_id = tenant_id or setting("USAGE_TENANT_ID", "")
    pending: list[dict] = []
    files: set[str] = set()
    received = 0
    ingested = 0
    agents: set[str] = set()

    def _flush() -> None:
        nonlocal received, ingested
        if not pending:
            return
        params: dict = {"bundles_json": _json.dumps(pending, default=str)}
        if tenant_id:
            params["tenant_id"] = tenant_id
        result = _run_async(
            call_tool_once(
                tool="ingest_sessions",
                server=server,
                url=url,
                action="upload",
                params=params,
                params_style="args",
            )
        )
        if isinstance(result, dict):
            received += int(result.get("received", 0) or 0)
            ingested += int(result.get("ingested", 0) or 0)
        pending.clear()

    for path, _mtime, _size, bundle in iter_local_bundles(only_changed=only_changed):
        files.add(path)
        agents.add(bundle.session.agent)
        pending.append(bundle.model_dump())
        if len(pending) >= batch:
            _flush()
    _flush()
    return {
        "mode": "upload",
        "transport": "mcp",
        "server": server or url,
        "files": len(files),
        "received": received,
        "ingested": ingested,
        "agents": sorted(agents),
    }


def collect_paths(paths: list[str | Path]) -> dict:
    """Parse explicit files/dirs (CLI/manual ingest) into the local store."""
    ensure_parsers_loaded()
    from agent_utilities.usage import get_usage_backend
    from agent_utilities.usage.recorder import get_usage_recorder

    get_usage_backend()  # ensure schema before the recorder writes
    recorder = get_usage_recorder()
    from .agent_sources import all_sources

    ingested = 0
    for raw in paths:
        p = Path(raw).expanduser()
        candidates = [p] if p.is_file() else sorted(p.rglob("*.jsonl"))
        for fp in candidates:
            # First source (registry order) that yields a real session wins, so
            # a file is ingested once under its best-matching agent.
            for source in all_sources():
                matched = False
                try:
                    for bundle in source.parse(fp):
                        if bundle.messages and recorder.record_bundle(bundle):
                            ingested += 1
                            matched = True
                except Exception:  # noqa: BLE001
                    continue
                if matched:
                    break
    return {"mode": "explicit", "ingested": ingested}
