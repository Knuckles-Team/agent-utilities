from __future__ import annotations

"""Native RSS/Atom feed document-source connector (zero-infra).

CONCEPT:KG-2.121 — native zero-infra RSS/Atom feed extractor and unified feed source.
Point it at one or more feed URLs and it yields each entry as a
:class:`SourceDocument`, with NO
external service required (unlike the FreshRSS feeder, which aggregates many feeds
behind its own server). It is the RSS analog of the zero-infra ``filesystem`` /
``web`` connectors.

Parsing uses ``feedparser`` (imported lazily so the module imports without it;
a clear error is raised only when an actual parse is attempted), which robustly
handles RSS 0.9x/1.0/2.0 + Atom + malformed feeds. Each entry is emitted with a
``metadata["record"]`` envelope (``categories`` + ``origin``) so it flows through
the SAME ``WorldModelPipelineRunner`` gate as the FreshRSS and ScholarX sources —
arXiv/research entries route to the research path, news entries to the world-model
gate. The incremental ``poll`` advances on the entry publish date plus a seen-id
belt for feeds with missing/equal dates.
"""

import time
from collections.abc import Callable, Iterator

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
#: Cap the persisted seen-id belt so the checkpoint can't grow unbounded; the
#: publish-date watermark is the primary delta, seen_ids only guards same-date dupes.
_SEEN_CAP = 5000
#: Per-feed HTTP fetch timeout (seconds). A feed slower than this in a sweep is
#: skipped rather than stalling the whole pass; with concurrent fetch the sweep's
#: wall-clock is bounded by this, not by N×timeout. (CONCEPT:KG-2.121)
_FEED_FETCH_TIMEOUT_S = 20.0
#: Max feeds fetched concurrently per sweep (bounds sockets/threads; the sweep
#: scales to many feeds without a serial stall — the 2000-reviews/hr path).
_FEED_FETCH_CONCURRENCY = 12


def _default_fetch(url: str) -> str:
    """Fetch a feed URL with lazy ``httpx`` (clear error if unavailable)."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - environment without httpx
        raise RuntimeError(
            "RssConnector needs 'httpx' to fetch feeds. "
            "Install it, or pass a fetch_fn for offline use."
        ) from exc
    resp = httpx.get(
        url,
        # Bounded per-feed so one slow feed can't stall a multi-feed sweep; feeds
        # are fetched concurrently (see RssConnector._all_documents), so the sweep's
        # wall-clock is ~the slowest single feed, not the sum. (CONCEPT:KG-2.121)
        timeout=_FEED_FETCH_TIMEOUT_S,
        follow_redirects=True,
        headers={"User-Agent": "agent-utilities-rss/1.0"},
    )
    resp.raise_for_status()
    return resp.text


def _parse(content: str):
    """Parse feed content with lazy ``feedparser``."""
    try:
        import feedparser
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise RuntimeError(
            "RssConnector needs 'feedparser' to parse feeds (declared in "
            "pyproject dependencies)."
        ) from exc
    return feedparser.parse(content)


def _entry_date(entry) -> str | None:
    """Normalize an entry's publish/updated date to an ISO-8601 UTC string.

    ISO-8601 sorts lexicographically == chronologically, so the connector's
    watermark comparison is a plain string compare.
    """
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            try:
                return time.strftime("%Y-%m-%dT%H:%M:%SZ", st)
            except (TypeError, ValueError):
                continue
    return None


def _entry_text(entry) -> str:
    content = entry.get("content")
    if isinstance(content, list) and content:
        val = content[0].get("value") if isinstance(content[0], dict) else None
        if val:
            return str(val)
    return str(entry.get("summary") or entry.get("description") or "")


@register_source("rss")
class RssConnector(LoadConnector, PollConnector):
    """Fetch + parse arbitrary RSS/Atom feeds into documents (CONCEPT:KG-2.121).

    Config:
        feed_urls: One URL or a list of feed URLs (required).
        max_items: Per-feed entry cap (default 200).
        doc_type: Document-type hint stamped on each item (default ``feed_item``).
        source_name: ``source_system`` provenance label (default ``rss``).
        fetch_fn: Optional ``(url) -> feed_content`` injectable for offline tests.
    """

    provider = "RSS/Atom Feed"

    def configure(
        self,
        *,
        feed_urls: list[str] | str | tuple[str, ...] = (),
        max_items: int = 200,
        doc_type: str = "feed_item",
        source_name: str = "rss",
        fetch_fn: FetchFn | None = None,
        **_: object,
    ) -> None:
        urls = [feed_urls] if isinstance(feed_urls, str) else list(feed_urls or [])
        self.feed_urls = [u for u in (s.strip() for s in urls) if u]
        if not self.feed_urls:
            raise ValueError("RssConnector requires one or more 'feed_urls'")
        self.max_items = max(1, int(max_items))
        self.doc_type = doc_type
        self.source_name = source_name
        self._fetch: FetchFn = fetch_fn or _default_fetch

    def health_check(self) -> bool:
        return bool(self.feed_urls)

    # -- parsing -----------------------------------------------------------

    def _entries(self, url: str) -> list[SourceDocument]:
        """Fetch + parse one feed into SourceDocuments (a dead feed → ``[]``)."""
        try:
            content = self._fetch(url)
        except Exception:  # noqa: BLE001 — a dead feed must not abort the sweep
            return []
        parsed = _parse(content)
        feed_title = str((parsed.feed or {}).get("title") or url)
        out: list[SourceDocument] = []
        for entry in (parsed.entries or [])[: self.max_items]:
            eid = entry.get("id") or entry.get("link") or entry.get("title") or ""
            if not eid:
                continue
            published = _entry_date(entry)
            link = str(entry.get("link") or "")
            cats = [
                str(t.get("term"))
                for t in (entry.get("tags") or [])
                if isinstance(t, dict) and t.get("term")
            ]
            text = _entry_text(entry)
            record = {
                "id": str(eid),
                "title": str(entry.get("title") or ""),
                "published": published,
                "categories": cats,
                # ``canonical`` + ``origin`` mirror the GReader shape the gate reads,
                # so RssConnector items route through the SAME router as FreshRSS.
                "canonical": [{"href": link}] if link else [],
                "origin": {
                    "htmlUrl": link,
                    "streamId": url,
                    "title": feed_title,
                },
            }
            out.append(
                SourceDocument(
                    id=str(eid),
                    source_uri=url,
                    title=str(entry.get("title") or "")[:300],
                    text=text,
                    doc_type=self.doc_type,
                    updated_at=published,
                    metadata={"record": record, "source_system": self.source_name},
                    external_access=ExternalAccess.public(),
                )
            )
        return out

    def _all_documents(self) -> list[SourceDocument]:
        """Fetch + parse every feed CONCURRENTLY so one slow feed can't stall the
        sweep. Each ``_entries`` call is bounded by the per-feed fetch timeout and
        degrades to ``[]`` on error, so the whole pass costs ~the slowest single
        feed (× ceil(N / concurrency)) instead of the serial sum — the throughput
        unlock for many-feed sweeps (the 2000-reviews/hr path). (CONCEPT:KG-2.121)"""
        urls = self.feed_urls
        if len(urls) <= 1:
            return self._entries(urls[0]) if urls else []
        from concurrent.futures import ThreadPoolExecutor

        docs: list[SourceDocument] = []
        workers = min(len(urls), _FEED_FETCH_CONCURRENCY)
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="rss-fetch"
        ) as ex:
            # ex.map preserves feed order; _entries never raises (dead feed → []).
            for batch in ex.map(self._entries, urls):
                docs.extend(batch)
        return docs

    # -- LoadConnector -----------------------------------------------------

    def load(self) -> Iterator[SourceDocument]:
        yield from self._all_documents()

    # -- PollConnector -----------------------------------------------------

    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        """Emit only entries newer than the prior watermark AND not already seen.

        Dual guard (CONCEPT:KG-2.121): the publish-date watermark is the primary
        delta; the seen-id belt catches feeds with missing or identical dates so a
        re-poll never re-emits an unchanged entry.
        """
        prior_ids = set(checkpoint.seen_ids) if checkpoint else set()
        wm = checkpoint.watermark if checkpoint else None
        all_docs = self._all_documents()
        # ``>= wm`` (not ``>``) so a new entry sharing the watermark second is still
        # emitted; the seen-id belt is what prevents re-emitting one already sent.
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
