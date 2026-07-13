"""Unified web-fetch resolver — fetch a URL as markdown via the best backend.

CONCEPT:AU-KG.ingest.web-fetch-front-door — Pluggable web fetching. One front door that both the ingestion
engine (the ``DOCUMENT`` URL path) and the skill-graph distillation pipeline call,
so the acquisition backend is chosen once, consistently, in this precedence:

  1. **ArchiveBox** (when ``config.archivebox_url`` is set) — serve the *preserved*
     snapshot through the ``archivebox-api`` MCP server; archive-on-miss. Fast, no
     live crawl, immune to a site going down or rate-limiting us.
  2. **crawl4ai** (when the crawler is installed) — render JS, recursive-capable.
  3. **requests + markitdown** — the zero-dependency floor.

A single page is the unit here; bulk/recursive crawling stays in the skill-graph
pipeline. ArchiveBox + crawl4ai are thus first-class without each caller
re-implementing backend selection.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

# Browser-like UA — the bare ``python-requests`` agent is 403'd by many sites.
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


@dataclass
class FetchedPage:
    """One fetched web page normalized to markdown/text, with provenance."""

    url: str
    markdown: str
    title: str = ""
    backend: str = ""  # "archivebox" | "crawl4ai" | "requests"


def archivebox_configured() -> bool:
    """True when an ArchiveBox instance is wired (the web-fetch on-signal)."""
    return bool((setting("ARCHIVEBOX_URL", default="") or "").strip())


def resolve_web_fetch(
    url: str, *, prefer: str | None = None, timeout: float = 90.0
) -> FetchedPage | None:
    """Fetch ``url`` as markdown via the best available backend (see module doc).

    ``prefer`` forces a specific backend ("archivebox" | "crawl4ai" | "requests");
    otherwise precedence applies. Returns ``None`` only if every available backend
    fails — callers treat that as an unreachable source.
    """
    order = (
        [prefer]
        if prefer
        else (
            (["archivebox"] if archivebox_configured() else [])
            + ["crawl4ai", "requests"]
        )
    )
    backends = {
        "archivebox": _fetch_via_archivebox,
        "crawl4ai": _fetch_via_crawl4ai,
        "requests": _fetch_via_requests,
    }
    for name in order:
        fn = backends.get(name)
        if fn is None:
            continue
        try:
            page = fn(url, timeout)
        except Exception as exc:  # noqa: BLE001 — try the next backend
            logger.debug("web-fetch backend %s failed for %s: %s", name, url, exc)
            page = None
        if page is not None and page.markdown.strip():
            return page
    return None


# ── ArchiveBox (preserved snapshot via the archivebox-api MCP server) ───────────


def _fetch_via_archivebox(url: str, timeout: float) -> FetchedPage | None:
    """Serve ``url`` from ArchiveBox; archive-on-miss, then retrieve the text.

    Reuses the KG-2.59 fleet-tool transport (``call_tool_once``) against the
    ``archivebox-api`` server — no bespoke HTTP client. Returns ``None`` (fall
    through to crawl4ai/requests) when no usable preserved text is available.
    """
    from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
        call_tool_once,
    )

    def _call(tool: str, action: str, params: dict[str, Any]) -> Any:
        from agent_utilities.protocols.source_connectors.connectors.mcp_package import (
            _run_async,
        )

        return _run_async(
            call_tool_once(
                tool=tool,
                server="archivebox-api",
                action=action,
                params=params,
                timeout=timeout,
            )
        )

    snap = _archivebox_snapshot(_call, url)
    if snap is None:
        # Archive-on-miss, then re-resolve the freshly created snapshot.
        _call("archivebox_cli", "cli_add", {"urls": [url], "extractors": "text"})
        snap = _archivebox_snapshot(_call, url)
    if snap is None:
        return None
    text, title = _archivebox_text(_call, snap, url)
    if not text.strip():
        return None
    return FetchedPage(url=url, markdown=text, title=title, backend="archivebox")


def _archivebox_snapshot(call: Any, url: str) -> dict[str, Any] | None:
    """Return the most relevant ArchiveBox snapshot record for ``url`` (or None)."""
    res = call("archivebox_core", "get_snapshots", {"url": url, "limit": 1})
    rows = _records(res)
    return rows[0] if rows else None


def _archivebox_text(call: Any, snap: dict[str, Any], url: str) -> tuple[str, str]:
    """Best-effort extracted text + title for a snapshot.

    Prefers an inline ``markdown``/``text`` artifact on the snapshot or its
    archiveresults; ArchiveBox stores some extractor outputs inline.
    """
    title = str(snap.get("title") or "")
    sid = str(snap.get("abid") or snap.get("id") or snap.get("timestamp") or "")
    # 1. inline body already on the snapshot record.
    for key in ("markdown", "text", "extracted_text", "content"):
        val = snap.get(key)
        if isinstance(val, str) and val.strip():
            return val, title
    # 2. ask for the text/markdown archiveresult of this snapshot.
    if sid:
        for extractor in ("markdown", "text", "htmltotext"):
            res = call(
                "archivebox_core",
                "get_archiveresults",
                {"snapshot_id": sid, "extractor": extractor},
            )
            for row in _records(res):
                for key in ("output", "text", "content", "markdown"):
                    val = row.get(key)
                    if (
                        isinstance(val, str)
                        and val.strip()
                        and not _looks_like_path(val)
                    ):
                        return val, title
    return "", title


def _looks_like_path(val: str) -> bool:
    """An archiveresult ``output`` is often a filename, not the content itself."""
    v = val.strip()
    return (
        "\n" not in v
        and len(v) < 256
        and ("/" in v or v.endswith((".txt", ".md", ".html", ".pdf", ".json")))
    )


def _records(result: Any) -> list[dict[str, Any]]:
    """Normalize an MCP/archivebox response to a list of record dicts."""
    if result is None:
        return []
    data = result
    if isinstance(data, dict):
        for key in ("results", "snapshots", "items", "archiveresults", "data"):
            if isinstance(data.get(key), list):
                return [r for r in data[key] if isinstance(r, dict)]
        return [data]
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []


# ── crawl4ai (single-page render via the web-crawler subprocess) ────────────────


def _fetch_via_crawl4ai(url: str, timeout: float) -> FetchedPage | None:
    """Fetch one page with the crawl4ai web-crawler (single page, no recursion)."""
    from agent_utilities.knowledge_graph.distillation.skill_graph_pipeline import (
        SourceSpec,
        _crawl_via_script,
        _resolve_crawler,
    )

    crawler = _resolve_crawler()
    if crawler is None:
        return None
    crawler_py, script = crawler
    # ``max_depth: 1`` (not 0) — the recursive strategy's ``for depth in
    # range(max_depth)`` loop body is what actually fetches+saves a page, so
    # ``max_depth=0`` crawled ZERO pages and this backend always returned None.
    # ``no_sitemap: True`` keeps this a genuine single-page fetch — without it,
    # a seed URL whose domain publishes a sitemap.xml auto-upgrades to a
    # whole-site ``sitemap-parallel`` crawl, defeating "one page" semantics.
    spec = SourceSpec(
        "web",
        url,
        {"max_depth": 1, "max_pages": 1, "crawl_timeout": timeout, "no_sitemap": True},
    )
    docs = _crawl_via_script(spec, crawler_py, script)
    if not docs:
        return None
    text = "\n\n".join(d.text for d in docs if d.text.strip())
    title = docs[0].title or ""
    if not text.strip():
        return None
    return FetchedPage(url=url, markdown=text, title=title, backend="crawl4ai")


# ── requests + markitdown (zero-dependency floor) ───────────────────────────────


def _fetch_via_requests(url: str, timeout: float) -> FetchedPage | None:
    """Plain HTTP GET → markdown via markitdown, falling back to a light tag strip."""
    import os
    import tempfile

    import requests

    # A browser-like UA: many sites (e.g. turingpost) 403 the default
    # ``python-requests`` agent. crawl4ai/ArchiveBox sidestep this when present;
    # the floor must too.
    to = min(timeout, 60.0)
    resp = requests.get(url, timeout=to, headers={"User-Agent": _UA})
    resp.raise_for_status()
    raw = resp.text

    text = raw
    try:
        from markitdown import MarkItDown

        with tempfile.NamedTemporaryFile(
            "w", suffix=".html", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        text = MarkItDown().convert(tmp_path).text_content
        os.unlink(tmp_path)
    except Exception:  # noqa: BLE001 — degrade to a light tag strip
        text = re.sub(r"<[^>]+>", " ", raw)

    title_m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
    title = (title_m.group(1).strip() if title_m else "")[:200]
    if not text.strip():
        return None
    return FetchedPage(url=url, markdown=text, title=title, backend="requests")
