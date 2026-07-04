"""Download research papers referenced by a page into the scholarx library.

CONCEPT:AU-KG.ingest.content-acquisition — the acquisition half of content-aware ingestion. Given
:class:`PaperRef`s extracted from a research roundup, fetch the PDFs into the
shared scholarx store (``paths.research_dir()/papers``) so the ingestion engine
can then ingest them as documents/papers.

Prefers the deployed ``scholarx-mcp`` server (dedup, metadata, queue) and falls
back to a direct arXiv/PDF download so acquisition still works when the MCP is
unreachable — both write to the same directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent_utilities.core import paths

from .paper_links import PaperRef

logger = logging.getLogger(__name__)


def papers_dir() -> Path:
    """The shared scholarx paper store (mirrors scholarx.paper_storage)."""
    d = paths.research_dir() / "papers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def acquire_papers(refs: list[PaperRef], *, timeout: float = 300.0) -> list[Path]:
    """Download the given paper refs; return the local PDF paths obtained.

    arXiv refs go through scholarx (MCP, else direct); explicit PDF urls are
    fetched directly. DOIs without a resolvable PDF are skipped (logged).
    """
    arxiv_ids = [r.ident for r in refs if r.kind == "arxiv"]
    pdf_urls = [r.ident for r in refs if r.kind == "pdf"]

    out: list[Path] = []
    if arxiv_ids:
        out.extend(_download_arxiv(arxiv_ids, timeout))
    for url in pdf_urls:
        p = _download_pdf_url(url, timeout)
        if p is not None:
            out.append(p)
    # de-dup by resolved path
    uniq: dict[str, Path] = {str(p): p for p in out if p is not None}
    return list(uniq.values())


def _download_arxiv(ids: list[str], timeout: float) -> list[Path]:
    """Download arXiv papers via the scholarx MCP, falling back to direct fetch."""
    paths_found = _scholarx_download_url(ids, timeout)
    if paths_found:
        return paths_found
    # Fallback: direct arXiv PDF fetch into the same store.
    out: list[Path] = []
    for pid in ids:
        p = _download_pdf_url(f"https://arxiv.org/pdf/{pid}", timeout, name=pid)
        if p is not None:
            out.append(p)
    return out


def _scholarx_download_url(ids: list[str], timeout: float) -> list[Path]:
    """Call ``scholarx-mcp`` ``sx_storage download_url``; return downloaded paths."""
    try:
        from agent_utilities.protocols.source_connectors.connectors.mcp_package import (
            _run_async,
        )
        from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
            call_tool_once,
        )

        result = _run_async(
            call_tool_once(
                tool="sx_storage",
                server="scholarx-mcp",
                action="download_url",
                params={"paper_ids": ",".join(ids)},
                timeout=timeout,
            )
        )
    except Exception as exc:  # noqa: BLE001 — MCP unreachable → caller falls back
        logger.debug("scholarx MCP download_url unavailable: %s", exc)
        return []
    rows = result.get("results", []) if isinstance(result, dict) else []
    out: list[Path] = []
    for row in rows:
        lp = row.get("local_path") if isinstance(row, dict) else None
        if lp and Path(lp).exists():
            out.append(Path(lp))
    return out


def _download_pdf_url(
    url: str, timeout: float, *, name: str | None = None
) -> Path | None:
    """Fetch a single PDF into the scholarx store (idempotent on filename)."""
    import re

    stem = name or re.sub(r"[^A-Za-z0-9._-]+", "_", url.rsplit("/", 1)[-1]) or "paper"
    if not stem.endswith(".pdf"):
        stem += ".pdf"
    dest = papers_dir() / stem
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        import requests

        to = min(timeout, 120.0)
        resp = requests.get(url, timeout=to, headers={"User-Agent": "agent-utilities"})
        resp.raise_for_status()
        if not resp.content:
            return None
        dest.write_bytes(resp.content)
        return dest
    except Exception as exc:  # noqa: BLE001 — a single paper failing is non-fatal
        logger.debug("direct PDF download failed for %s: %s", url, exc)
        return None
