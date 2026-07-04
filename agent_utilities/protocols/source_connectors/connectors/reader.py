from __future__ import annotations

"""Readability reader document-source connector (CONCEPT:AU-KG.enrichment.multimodal-readers).

Turns a single URL into clean, boilerplate-stripped markdown — the content a
fact extractor actually wants, versus the raw HTML the recursive ``web`` crawler
yields. Three tiers, best-to-worst, chosen automatically (no per-call knob):

1. **Jina Reader** (``https://r.jina.ai/{url}``) when a ``JINA_API_KEY`` is
   configured — server-side readability → markdown.
2. **Local readability** via the ``trafilatura`` soft-dep (no key, no network
   round-trip beyond the page itself).
3. **Light tag strip** — the same final degradation the web crawler uses, so the
   connector always returns *something* rather than raising.

Assimilated from ``knowledge-graph-extractor``'s Jina Reader URL ingestion;
generalized with a local fallback so it works with zero external credentials.
"""

from collections.abc import Callable, Iterator

from agent_utilities.core.config import setting

from ..base import (
    ExternalAccess,
    LoadConnector,
    SourceDocument,
)
from ..registry import register_source

FetchFn = Callable[[str], str]


def _jina_read(url: str, api_key: str) -> tuple[str, str] | None:
    """Fetch clean markdown via Jina Reader; ``None`` on any failure."""
    try:
        import httpx
    except ImportError:  # pragma: no cover - environment without httpx
        return None
    try:
        headers = {"Accept": "text/markdown"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = httpx.get(f"https://r.jina.ai/{url}", headers=headers, timeout=60.0)
        resp.raise_for_status()
        text = resp.text
        title = ""
        for line in text.strip().split("\n")[:5]:
            if line.startswith("Title:"):
                title = line[6:].strip()
                break
            if line.startswith("# "):
                title = line[2:].strip()
                break
        return text, title
    except Exception:  # noqa: BLE001 — fall through to the local tier
        return None


def _local_read(url: str, fetch: FetchFn) -> tuple[str, str]:
    """Local readability: trafilatura if present, else a light tag strip."""
    html = fetch(url)
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html, output_format="markdown", include_links=False
        )
        if extracted:
            meta = trafilatura.extract_metadata(html)
            title = (getattr(meta, "title", "") or "") if meta else ""
            return extracted, title
    except Exception:  # noqa: BLE001 — degrade to a tag strip
        pass
    import re

    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = (title_m.group(1).strip() if title_m else "")[:200]
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip(), title


def _default_fetch(url: str) -> str:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - environment without httpx
        raise RuntimeError(
            "ReaderConnector needs 'httpx' to fetch URLs. "
            "Install it, or pass a fetch_fn for offline use."
        ) from exc
    resp = httpx.get(url, timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


@register_source("reader")
class ReaderConnector(LoadConnector):
    """Read a single URL into clean markdown (readability), one document.

    CONCEPT:AU-KG.enrichment.multimodal-readers.

    Config:
        url: The page to read (required).
        api_key: Override the configured ``JINA_API_KEY`` (optional).
        fetch_fn: Optional ``(url) -> html`` injectable for offline tests / the
            local tier (Jina Reader is bypassed when ``fetch_fn`` is given).
    """

    provider = "Readability Reader"

    def configure(
        self,
        *,
        url: str = "",
        api_key: str | None = None,
        fetch_fn: FetchFn | None = None,
        **_: object,
    ) -> None:
        if not url:
            raise ValueError("ReaderConnector requires a 'url'")
        self.url = url
        # config-discipline: the key is a deployment secret read through the
        # sanctioned accessor, never bare os.environ.
        self.api_key = api_key if api_key is not None else setting("JINA_API_KEY", "")
        self._fetch: FetchFn = fetch_fn or _default_fetch
        self._offline = fetch_fn is not None

    def health_check(self) -> bool:
        return bool(self.url)

    def _read(self) -> tuple[str, str]:
        # Skip the network Jina tier when a fetch_fn is injected (offline tests)
        # or no key is configured; otherwise prefer it for best readability.
        if not self._offline and self.api_key:
            result = _jina_read(self.url, self.api_key)
            if result is not None:
                return result
        return _local_read(self.url, self._fetch)

    def load(self) -> Iterator[SourceDocument]:
        text, title = self._read()
        if not text.strip():
            return
        yield SourceDocument(
            id=self.url,
            source_uri=self.url,
            title=(title or self.url)[:200],
            text=text,
            doc_type="article",
            metadata={
                "reader": "jina" if (self.api_key and not self._offline) else "local"
            },
            external_access=ExternalAccess.public(),
        )
