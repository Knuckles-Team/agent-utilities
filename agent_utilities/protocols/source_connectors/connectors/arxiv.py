from __future__ import annotations

"""Native arXiv document-source connector (zero-infra, CONCEPT:AU-KG.ingest.arxiv-feed-connector).

Drives the public ``export.arxiv.org`` Atom query API directly — no external MCP
server or account is required, mirroring the zero-infra ``rss``/``web`` connectors
(CONCEPT:AU-ECO.connector.document-source-framework). This is deliberately a THIRD, narrower research feed
alongside ``freshrss`` (curated world-model RSS, gated by ``WorldModelPipelineRunner``)
and ScholarX (paper search/dedup service reached via ``scholarx-mcp``): it is the raw
per-category arXiv listing, useful when neither of those is deployed.

Each entry is emitted with the SAME ``metadata["record"]`` envelope shape as
:func:`agent_utilities.automation.feed_sources.scholarx_feed_documents` and the
native ``rss`` connector (``canonical`` + ``origin.streamId``), so it converges on
the identical canonical ``arxiv:<id>`` node the moment ``WorldModelPipelineRunner``
(CONCEPT:AU-KG.ingest.worldmodel-gated-ingestion) routes it to the research path — one paper arriving via
FreshRSS-arXiv, ScholarX, AND this connector collapses to one KG node, never three.

**Budget-bounded by construction, never a firehose (the ★ critical constraint of
CONCEPT:AU-KG.ingest.arxiv-feed-connector):** ``categories`` has NO default — an operator must opt in
explicitly (``KG_ARXIV_CATEGORIES``) — and ``max_results`` caps each category's
per-poll page size. The connector itself does no relevance scoring; every entry it
yields still passes through the SAME downstream gate as every other research feed
(``grade_and_enqueue_paper`` — keyword score + novelty dedup + KG-stored watchlists),
so this connector only ever WIDENS the funnel's mouth, never bypasses its throat.
"""

import time
from collections.abc import Callable, Iterator
from typing import Any

from ..base import (
    CheckpointedBatch,
    ConnectorCheckpoint,
    LoadConnector,
    PollConnector,
    SourceDocument,
    default_external_access,
)
from ..http_safety import require_safe_source_url, safe_get_text
from ..registry import register_source

FetchFn = Callable[[str, dict[str, Any]], str]

#: The public arXiv Atom query API (no auth, no key — export.arxiv.org's documented
#: rate-limit courtesy window is ~1 request/3s per the arXiv API terms of use; the
#: per-category cap plus the sweep cadence keeps this connector well under that).
_ARXIV_API_URL = "https://export.arxiv.org/api/query"
#: Cap the persisted seen-id belt so the checkpoint can't grow unbounded.
_SEEN_CAP = 5000
#: Hard ceiling on ``max_results`` — arXiv's own API caps a single page at 2000, but
#: nothing here needs to approach that; the operator-facing default is far lower.
_MAX_RESULTS_CEILING = 500
_FETCH_TIMEOUT_S = 20.0


def _entry_date(entry: dict[str, Any]) -> str | None:
    """Normalize an Atom entry's submitted/updated date to ISO-8601 UTC."""
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            try:
                return time.strftime("%Y-%m-%dT%H:%M:%SZ", st)
            except (TypeError, ValueError):
                continue
    return None


def _arxiv_bare_id(raw_id: str) -> str:
    """Strip the ``https://arxiv.org/abs/`` prefix and version suffix from an entry id."""
    tail = str(raw_id or "").rsplit("/", 1)[-1]
    if "v" in tail:
        head, _, version = tail.rpartition("v")
        if version.isdigit():
            tail = head
    return tail


def _pdf_url(entry: dict[str, Any]) -> str:
    for link in entry.get("links") or []:
        if not isinstance(link, dict):
            continue
        if link.get("type") == "application/pdf" or link.get("title") == "pdf":
            return str(link.get("href") or "")
    return ""


def _authors(entry: dict[str, Any]) -> list[str]:
    out = []
    for author in entry.get("authors") or []:
        name = author.get("name") if isinstance(author, dict) else None
        if name:
            out.append(str(name))
    return out


def _categories(entry: dict[str, Any]) -> list[str]:
    return [
        str(t.get("term"))
        for t in (entry.get("tags") or [])
        if isinstance(t, dict) and t.get("term")
    ]


@register_source("arxiv")
class ArxivConnector(LoadConnector, PollConnector):
    """Fetch + parse the arXiv Atom query API into research-paper documents.

    Config:
        categories: One or more arXiv category codes (e.g. ``cs.AI``), REQUIRED —
            there is no default; an unscoped query is not a valid arXiv listing and
            would be an unbounded firehose (CONCEPT:AU-KG.ingest.arxiv-feed-connector).
        max_results: Per-category, per-poll page cap (default 50, capped at 500).
        doc_type: Document-type hint stamped on each item (default ``paper``).
        source_name: ``source_system`` provenance label (default ``arxiv``).
        fetch_fn: Optional ``(url, params) -> atom_xml`` injectable for offline tests.
    """

    provider = "arXiv API"

    def configure(
        self,
        *,
        categories: list[str] | str | tuple[str, ...] = (),
        max_results: int = 50,
        doc_type: str = "paper",
        source_name: str = "arxiv",
        fetch_fn: FetchFn | None = None,
        allowed_private_hosts: list[str] | None = None,
        max_response_bytes: int = 10 * 1024 * 1024,
        **_: object,
    ) -> None:
        cats = [categories] if isinstance(categories, str) else list(categories or [])
        self.categories = [c for c in (s.strip() for s in cats) if c]
        if not self.categories:
            raise ValueError(
                "ArxivConnector requires one or more 'categories' — an unscoped "
                "query is not supported (would be an unbounded firehose)"
            )
        if len(self.categories) > 50:
            raise ValueError("ArxivConnector accepts at most 50 categories")
        self.max_results = min(_MAX_RESULTS_CEILING, max(1, int(max_results)))
        self.doc_type = doc_type
        self.source_name = source_name
        self.external_access = default_external_access()
        require_safe_source_url(
            _ARXIV_API_URL,
            allowed_private_hosts=list(allowed_private_hosts or []),
            resolve_dns=False,
        )
        if fetch_fn is not None:
            self._fetch = fetch_fn
        else:

            def _safe_fetch(url: str, params: dict[str, Any]) -> str:
                return safe_get_text(
                    url,
                    params=params,
                    timeout=_FETCH_TIMEOUT_S,
                    headers={"User-Agent": "agent-utilities-arxiv/1.0"},
                    max_bytes=max_response_bytes,
                    allowed_private_hosts=list(allowed_private_hosts or []),
                )

            self._fetch = _safe_fetch

    def health_check(self) -> bool:
        return bool(self.categories)

    # -- parsing -----------------------------------------------------------

    def _parse(self, content: str):
        try:
            import feedparser
        except (
            ImportError
        ) as exc:  # pragma: no cover - dependency declared in pyproject
            raise RuntimeError(
                "ArxivConnector needs 'feedparser' to parse the Atom response "
                "(declared in pyproject dependencies)."
            ) from exc
        return feedparser.parse(content)

    def _category_entries(self, category: str) -> list[SourceDocument]:
        """Fetch + parse one category's most recent entries (a dead category → [])."""
        params = {
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": self.max_results,
        }
        try:
            content = self._fetch(_ARXIV_API_URL, params)
        except Exception:  # noqa: BLE001 — one dead category must not abort the sweep
            return []
        parsed = self._parse(content)
        out: list[SourceDocument] = []
        for entry in parsed.entries or []:
            raw_id = str(entry.get("id") or "")
            aid = _arxiv_bare_id(raw_id)
            if not aid:
                continue
            published = _entry_date(entry)
            title = str(entry.get("title") or "").strip()
            abstract = str(entry.get("summary") or "").strip()
            pdf_url = _pdf_url(entry)
            abs_url = raw_id or f"https://arxiv.org/abs/{aid}"
            record = {
                "id": aid,
                "title": title,
                "published": published,
                "categories": _categories(entry) or [category],
                "authors": _authors(entry),
                "pdf_url": pdf_url,
                "canonical": [{"href": abs_url}],
                "origin": {
                    "htmlUrl": abs_url,
                    "streamId": "arxiv:api",
                    "title": f"arXiv {category}",
                },
            }
            out.append(
                SourceDocument(
                    id=f"arxiv:{aid}",
                    source_uri=abs_url,
                    title=title[:300],
                    text=abstract,
                    doc_type=self.doc_type,
                    updated_at=published,
                    metadata={"record": record, "source_system": self.source_name},
                    external_access=self.external_access.model_copy(deep=True),
                )
            )
        return out

    def _all_documents(self) -> list[SourceDocument]:
        """Fetch every configured category, deduping papers cross-listed in >1."""
        seen: set[str] = set()
        docs: list[SourceDocument] = []
        for category in self.categories:
            for doc in self._category_entries(category):
                if doc.id in seen:
                    continue
                seen.add(doc.id)
                docs.append(doc)
        return docs

    # -- LoadConnector -------------------------------------------------------

    def load(self) -> Iterator[SourceDocument]:
        yield from self._all_documents()

    # -- PollConnector ---------------------------------------------------------

    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        """Emit only entries newer than the prior watermark AND not already seen.

        Same dual-guard shape as the native ``rss`` connector: the submitted-date
        watermark is the primary delta, the seen-id belt catches same-date dupes.
        arXiv's API has no server-side "since" filter, so each poll re-fetches the
        (bounded) most-recent page per category and filters client-side — the
        ``max_results`` cap is what keeps this a bounded read, not a full re-scan.
        """
        prior_ids = set(checkpoint.seen_ids) if checkpoint else set()
        wm = checkpoint.watermark if checkpoint else None
        all_docs = self._all_documents()
        fresh = [
            d
            for d in all_docs
            if d.id not in prior_ids
            and (wm is None or not d.updated_at or d.updated_at >= wm)
        ]
        dates = [d.updated_at for d in all_docs if d.updated_at]
        if wm:
            dates.append(wm)
        new_wm = max(dates) if dates else wm
        new_ids = sorted(prior_ids | {d.id for d in fresh})[-_SEEN_CAP:]
        cp = ConnectorCheckpoint(has_more=False, watermark=new_wm, seen_ids=new_ids)
        return CheckpointedBatch(documents=fresh, checkpoint=cp)
