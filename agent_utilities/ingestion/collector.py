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
import os
from collections.abc import Iterator
from pathlib import Path

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
    role = os.environ.get("KG_DAEMON_ROLE", "auto")
    if role == "client":
        return True
    endpoint = os.environ.get("GRAPH_ENGINE_ENDPOINT") or os.environ.get(
        "EPISTEMIC_GRAPH_ENDPOINT"
    )
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

    gateway_url = gateway_url or os.environ.get("USAGE_GATEWAY_URL", "")
    tenant_id = tenant_id or os.environ.get("USAGE_TENANT_ID", "")
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
