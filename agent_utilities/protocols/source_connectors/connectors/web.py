from __future__ import annotations

"""Recursive web-crawler document-source connector.

CONCEPT:ECO-4.25 — reference ``Load`` + ``Poll`` connector.
CONCEPT:ECO-4.26 — incremental poll keyed on a crawl-time watermark + seen ids.

A same-domain breadth-first crawler. Pages are fetched (lazy ``httpx``; the caller
may inject a ``fetch_fn`` for offline tests), converted HTML→text via the
``markitdown`` soft-dep (falling back to a light tag strip — the same degradation
the ingestion engine's URL path uses), and yielded as :class:`SourceDocument`s.
No new hard dependency is added: ``httpx`` is imported lazily and a clear
RuntimeError is raised only when an actual network fetch is attempted without it.
"""

import re
from collections.abc import Callable, Iterator
from urllib.parse import urldefrag, urljoin, urlparse

from ..base import (
    CheckpointedBatch,
    ConnectorCheckpoint,
    ExternalAccess,
    LoadConnector,
    PollConnector,
    SourceDocument,
)
from ..registry import register_source

FetchFn = Callable[[str], str]

_LINK_RE = re.compile(r'href=["\']([^"\'#]+)["\']', re.IGNORECASE)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _html_to_text(html: str) -> str:
    """HTML → text via markitdown, falling back to a regex tag strip."""
    import tempfile

    try:
        from markitdown import MarkItDown

        with tempfile.NamedTemporaryFile(
            "w", suffix=".html", delete=True, encoding="utf-8"
        ) as tmp:
            tmp.write(html)
            tmp.flush()
            return MarkItDown().convert(tmp.name).text_content
    except Exception:  # noqa: BLE001 — fall back to a light strip (no hard dep)
        text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()


def _default_fetch(url: str) -> str:
    """Fetch a URL with lazy ``httpx`` (raises a clear error if unavailable)."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - environment without httpx
        raise RuntimeError(
            "WebCrawlerConnector needs 'httpx' to fetch URLs. "
            "Install it, or pass a fetch_fn for offline use."
        ) from exc
    resp = httpx.get(url, timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


@register_source("web")
class WebCrawlerConnector(LoadConnector, PollConnector):
    """Recursively crawl a website, same-domain, into documents.

    CONCEPT:ECO-4.25.

    Config:
        base_url: Seed URL (required).
        max_depth: Link-following depth from the seed (default 1).
        max_pages: Hard cap on pages crawled (default 50).
        same_domain: Restrict to the seed's host (default True).
        fetch_fn: Optional ``(url) -> html`` injectable for offline tests.
    """

    provider = "Web Crawler"
    priority = 60

    def configure(
        self,
        *,
        base_url: str = "",
        max_depth: int = 1,
        max_pages: int = 50,
        same_domain: bool = True,
        fetch_fn: FetchFn | None = None,
        **_: object,
    ) -> None:
        if not base_url:
            raise ValueError("WebCrawlerConnector requires a 'base_url'")
        self.base_url = base_url
        self.max_depth = max(0, int(max_depth))
        self.max_pages = max(1, int(max_pages))
        self.same_domain = same_domain
        self._fetch: FetchFn = fetch_fn or _default_fetch
        self._host = urlparse(base_url).netloc

    def health_check(self) -> bool:
        return bool(self.base_url)

    def _same_domain(self, url: str) -> bool:
        return (not self.same_domain) or urlparse(url).netloc == self._host

    def _extract_links(self, base: str, html: str) -> list[str]:
        out: list[str] = []
        for href in _LINK_RE.findall(html):
            absolute, _ = urldefrag(urljoin(base, href))
            if absolute.startswith(("http://", "https://")) and self._same_domain(
                absolute
            ):
                out.append(absolute)
        return out

    def _crawl(self, skip_ids: set[str] | None = None) -> Iterator[SourceDocument]:
        """BFS from the seed up to ``max_depth`` / ``max_pages``.

        ``skip_ids`` lets the incremental poll avoid re-emitting already-seen URLs.
        """
        skip = skip_ids or set()
        seen: set[str] = set()
        queue: list[tuple[str, int]] = [(self.base_url, 0)]
        emitted = 0
        while queue and emitted < self.max_pages:
            url, depth = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)
            try:
                html = self._fetch(url)
            except Exception:  # noqa: BLE001 — a dead link must not abort the crawl
                continue
            text = _html_to_text(html)
            if text.strip() and url not in skip:
                title_match = _TITLE_RE.search(html)
                title = (title_match.group(1).strip() if title_match else url)[:200]
                emitted += 1
                yield SourceDocument(
                    id=url,
                    source_uri=url,
                    title=title,
                    text=text,
                    doc_type="webpage",
                    metadata={"depth": depth},
                    external_access=ExternalAccess.public(),
                )
            if depth < self.max_depth:
                for link in self._extract_links(url, html):
                    if link not in seen:
                        queue.append((link, depth + 1))

    # -- LoadConnector -----------------------------------------------------

    def load(self) -> Iterator[SourceDocument]:
        yield from self._crawl()

    # -- PollConnector -----------------------------------------------------

    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        """Crawl, skipping URLs already emitted in a prior poll (by ``seen_ids``).

        CONCEPT:ECO-4.26 — a re-poll re-walks the site but only emits pages not in
        the prior ``seen_ids`` set, and records the union so the next poll skips
        them too.
        """
        prior_ids = set(checkpoint.seen_ids) if checkpoint else set()
        docs = list(self._crawl(skip_ids=prior_ids))
        new_ids = prior_ids | {d.id for d in docs}
        cp = ConnectorCheckpoint(
            has_more=False,
            watermark=None,
            seen_ids=sorted(new_ids),
        )
        return CheckpointedBatch(documents=docs, checkpoint=cp)
