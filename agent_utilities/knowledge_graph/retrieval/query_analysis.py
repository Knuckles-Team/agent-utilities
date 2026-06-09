from __future__ import annotations

"""Query analysis — source/time auto-filters + citation processing.

CONCEPT:ECO-4.32 — Query Analysis

Onyx derives filters from the natural-language query before retrieval (a secondary
LLM flow that detects a *time window* and *source-type* restriction) and attaches
*citations* to answers. This module ports both onto agent-utilities' retrieval:

  * :func:`analyze_query` returns :class:`QueryFilters` — a detected ``as_of``
    time reference / ``time_range`` and a set of ``source_types`` the query is
    asking about. It uses an ``llm_fn`` when supplied and a **deterministic
    regex/keyword fallback** otherwise, so it works offline and in tests.
  * :class:`CitationProcessor` turns retrieved nodes into ``[n] source`` citations
    and rewrites bare ``[n]`` markers in an answer into linked references — using
    the provenance already on ``Document`` / ``Chunk`` nodes (``source`` /
    ``source_url`` / ``file_path``), so no new provenance is needed.

The retriever wires :func:`analyze_query` as an **opt-in pre-filter**
(``HybridRetriever.retrieve_hybrid(query_analysis=True)``) so existing callers are
unaffected; the citation processor is a post-step a caller composes over results.
"""

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

__all__ = ["QueryFilters", "analyze_query", "CitationProcessor"]

LLMFn = Callable[[str], str]

# Source-type keyword vocabulary for the deterministic fallback. Maps a detected
# keyword to the canonical doc_type/source label used on Document nodes.
_SOURCE_KEYWORDS: dict[str, str] = {
    "paper": "paper",
    "papers": "paper",
    "arxiv": "paper",
    "publication": "paper",
    "ticket": "ticket",
    "tickets": "ticket",
    "incident": "ticket",
    "issue": "issue",
    "issues": "issue",
    "pull request": "issue",
    "pr": "issue",
    "email": "email",
    "emails": "email",
    "message": "message",
    "messages": "message",
    "chat": "message",
    "slack": "message",
    "wiki": "wiki",
    "confluence": "wiki",
    "webpage": "webpage",
    "website": "webpage",
    "doc": "document",
    "docs": "document",
    "document": "document",
}

# Relative-time phrases → ``(pattern, timedelta, label)`` back from "now".
_RELATIVE_TIME: list[tuple[re.Pattern[str], timedelta, str]] = [
    (re.compile(r"\btoday\b", re.I), timedelta(days=1), "today"),
    (re.compile(r"\byesterday\b", re.I), timedelta(days=2), "yesterday"),
    (re.compile(r"\bthis week\b", re.I), timedelta(days=7), "this week"),
    (re.compile(r"\bpast week\b", re.I), timedelta(days=7), "past week"),
    (re.compile(r"\blast week\b", re.I), timedelta(days=14), "last week"),
    (re.compile(r"\bthis month\b", re.I), timedelta(days=31), "this month"),
    (re.compile(r"\bpast month\b", re.I), timedelta(days=31), "past month"),
    (re.compile(r"\blast month\b", re.I), timedelta(days=62), "last month"),
    (re.compile(r"\bthis year\b", re.I), timedelta(days=366), "this year"),
    (re.compile(r"\bpast year\b", re.I), timedelta(days=366), "past year"),
    (re.compile(r"\brecent(?:ly)?\b", re.I), timedelta(days=30), "recent"),
]
_LAST_N_RE = re.compile(r"\b(?:last|past)\s+(\d+)\s+(day|week|month|year)s?\b", re.I)
_UNIT_DAYS = {"day": 1, "week": 7, "month": 31, "year": 366}


class QueryFilters(BaseModel):
    """Filters derived from a query (CONCEPT:ECO-4.32).

    Attributes:
        source_types: Document/source types the query restricts to (empty = all).
        as_of: ISO instant the query's time window starts at (``None`` = no
            time filter). Maps onto ``retrieve_hybrid(as_of=…)``.
        time_range: Human-readable description of the detected window (for logs).
        cleaned_query: The query with detected filter phrases left intact (callers
            may use the original; provided for parity with Onyx).
    """

    source_types: list[str] = Field(default_factory=list)
    as_of: str | None = None
    time_range: str = ""
    cleaned_query: str = ""


_LLM_PROMPT = (
    "Extract retrieval filters from the user query. Respond with STRICT JSON: "
    '{"source_types": [..], "since_days": <int or null>}. '
    "source_types is a subset of [paper, ticket, issue, email, message, wiki, "
    "webpage, document]; since_days is how many days back the query asks about "
    "(null if no time constraint). Query: "
)


def _now() -> datetime:
    return datetime.now(UTC)


def _detect_source_types(query: str) -> list[str]:
    low = query.lower()
    found: list[str] = []
    for kw, label in _SOURCE_KEYWORDS.items():
        if re.search(rf"\b{re.escape(kw)}\b", low) and label not in found:
            found.append(label)
    return found


def _detect_time(query: str) -> tuple[str | None, str]:
    """Return ``(as_of_iso, description)`` for the query's time window."""
    m = _LAST_N_RE.search(query)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        days = n * _UNIT_DAYS.get(unit, 1)
        return (_now() - timedelta(days=days)).isoformat(), f"last {n} {unit}(s)"
    for pat, delta, label in _RELATIVE_TIME:
        if pat.search(query):
            return (_now() - delta).isoformat(), label
    return None, ""


def analyze_query(query: str, llm_fn: LLMFn | None = None) -> QueryFilters:
    """Derive source/time filters from ``query`` (CONCEPT:ECO-4.32).

    Uses ``llm_fn`` when provided (strict-JSON extraction); always falls back to —
    and validates against — the deterministic regex/keyword detector so the result
    is sensible offline.
    """
    source_types = _detect_source_types(query)
    as_of, time_range = _detect_time(query)

    if llm_fn is not None:
        try:
            import json

            raw = llm_fn(_LLM_PROMPT + query)
            data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
            llm_sources = [
                s
                for s in data.get("source_types", [])
                if s in set(_SOURCE_KEYWORDS.values())
            ]
            if llm_sources:
                source_types = sorted(set(source_types) | set(llm_sources))
            since = data.get("since_days")
            if isinstance(since, int) and since > 0 and as_of is None:
                as_of = (_now() - timedelta(days=since)).isoformat()
                time_range = f"last {since} day(s)"
        except Exception:  # noqa: BLE001 — LLM optional; deterministic result stands
            pass

    return QueryFilters(
        source_types=source_types,
        as_of=as_of,
        time_range=time_range,
        cleaned_query=query.strip(),
    )


def filter_nodes_by_source(
    nodes: list[dict[str, Any]], source_types: list[str]
) -> list[dict[str, Any]]:
    """Keep only nodes whose ``doc_type``/``type`` matches a requested source type.

    CONCEPT:ECO-4.32. A node with no resolvable type is kept (we never silently
    drop unclassifiable results). Returns the input unchanged when no source-type
    filter is requested.
    """
    if not source_types:
        return nodes
    wanted = {s.lower() for s in source_types}
    out: list[dict[str, Any]] = []
    for n in nodes:
        dt = str(n.get("doc_type") or n.get("type") or "").lower()
        if not dt or dt in wanted:
            out.append(n)
    return out


class CitationProcessor:
    """Attach source citations to retrieval results / answers (CONCEPT:ECO-4.32).

    Builds a numbered citation list from a node's existing provenance, and can
    rewrite ``[n]`` markers in an answer into ``[n](source)`` references.
    """

    _MARKER_RE = re.compile(r"\[(\d+)\]")

    @staticmethod
    def _source_of(node: dict[str, Any]) -> str:
        for key in ("source_url", "source", "file_path", "url"):
            val = node.get(key)
            if isinstance(val, str) and val:
                return val
        return node.get("name") or node.get("id") or ""

    def build_citations(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return ``[{n, id, title, source}]`` citations, 1-indexed by rank."""
        cites: list[dict[str, Any]] = []
        for i, node in enumerate(nodes, start=1):
            cites.append(
                {
                    "n": i,
                    "id": node.get("id", ""),
                    "title": node.get("name")
                    or node.get("title")
                    or node.get("id", ""),
                    "source": self._source_of(node),
                }
            )
        return cites

    def annotate(self, answer: str, nodes: list[dict[str, Any]]) -> str:
        """Rewrite ``[n]`` markers in ``answer`` to ``[n](source)`` links.

        Out-of-range markers are left untouched so the answer is never corrupted.
        """
        cites = {c["n"]: c for c in self.build_citations(nodes)}

        def _sub(m: re.Match[str]) -> str:
            n = int(m.group(1))
            c = cites.get(n)
            if not c or not c["source"]:
                return m.group(0)
            return f"[{n}]({c['source']})"

        return self._MARKER_RE.sub(_sub, answer)
