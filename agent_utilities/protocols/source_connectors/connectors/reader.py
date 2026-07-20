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
    LoadConnector,
    SourceDocument,
    default_external_access,
)
from ..http_safety import (
    normalize_allowed_hosts,
    require_safe_source_url,
    safe_get_text,
)
from ..registry import register_source

FetchFn = Callable[[str], str]


def _jina_read(
    url: str, api_key: str, *, max_response_bytes: int
) -> tuple[str, str] | None:
    """Fetch clean markdown via Jina Reader; ``None`` on any failure."""
    try:
        # Never ask an external reader proxy to dereference an unvalidated
        # destination. Private sources are handled only by the local tier.
        require_safe_source_url(url, allowed_private_hosts=(), resolve_dns=True)
        headers = {"Accept": "text/markdown"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        text = safe_get_text(
            f"https://r.jina.ai/{url}",
            headers=headers,
            timeout=60.0,
            max_bytes=max_response_bytes,
        )
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
        allowed_private_hosts: list[str] | None = None,
        allowed_redirect_hosts: list[str] | None = None,
        max_response_bytes: int = 10 * 1024 * 1024,
        **_: object,
    ) -> None:
        if not url:
            raise ValueError("ReaderConnector requires a 'url'")
        self._allowed_private_hosts = normalize_allowed_hosts(allowed_private_hosts)
        self._url_host = require_safe_source_url(
            url,
            allowed_private_hosts=self._allowed_private_hosts,
            resolve_dns=False,
        )
        self.url = url
        # config-discipline: the key is a deployment secret read through the
        # sanctioned accessor, never bare os.environ.
        self.api_key = api_key if api_key is not None else setting("JINA_API_KEY", "")
        if fetch_fn is not None:
            self._fetch = fetch_fn
        else:
            redirect_hosts = list(allowed_redirect_hosts or [])

            def _safe_fetch(target: str) -> str:
                return safe_get_text(
                    target,
                    timeout=60.0,
                    max_bytes=max_response_bytes,
                    allowed_private_hosts=self._allowed_private_hosts,
                    allowed_redirect_hosts=redirect_hosts,
                )

            self._fetch = _safe_fetch
        self._offline = fetch_fn is not None
        self._max_response_bytes = max_response_bytes
        self.external_access = default_external_access()

    def health_check(self) -> bool:
        return bool(self.url)

    def _read(self) -> tuple[str, str]:
        # Skip the network Jina tier when a fetch_fn is injected (offline tests)
        # or no key is configured; otherwise prefer it for best readability.
        if (
            not self._offline
            and self.api_key
            and self._url_host not in self._allowed_private_hosts
        ):
            result = _jina_read(
                self.url,
                self.api_key,
                max_response_bytes=self._max_response_bytes,
            )
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
            external_access=self.external_access.model_copy(deep=True),
        )
