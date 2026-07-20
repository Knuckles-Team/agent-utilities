#!/usr/bin/python
from __future__ import annotations

"""Direct Corpus Interaction — literal/regex retrieval over raw documents.

CONCEPT:AU-KG.retrieval.memory-first-retrieval — Memory-First Retrieval (direct-corpus mode)

A precise, deterministic retrieval mode that searches document *text* directly
(grep/read) instead of (or alongside) dense vector similarity. Distilled from the
"Is Grep All You Need / Direct Corpus Interaction" research
(``.specify/specs/research-evolution-20260606/`` plan b2-02): for many agentic
search tasks, literal term/regex matching with line-level localization is
competitive with — and more auditable than — embedding retrieval.

Provides composable primitives:

* :meth:`DirectCorpusSearcher.grep` — line-level literal/regex matches.
* :meth:`DirectCorpusSearcher.read` — read a line range of one document.
* :meth:`DirectCorpusSearcher.search` — multi-term ranked search with
  coverage + localization metrics (the paper's trajectory signals).

No model, no network — pure Python, so it is always available and unit-testable.

Concept: direct-corpus-retrieval
"""

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

_WORD = re.compile(r"[a-z0-9]+")


def _terms(query: str) -> list[str]:
    return _WORD.findall((query or "").lower())


class GrepMatch(BaseModel):
    """A single line-level match."""

    doc_id: str
    line_no: int  # 1-based
    line: str
    before: list[str] = Field(default_factory=list)
    after: list[str] = Field(default_factory=list)


class DocHit(BaseModel):
    """A ranked document hit with localization."""

    doc_id: str
    score: float = Field(ge=0.0, le=1.0)
    term_coverage: float = Field(ge=0.0, le=1.0)
    matched_terms: list[str] = Field(default_factory=list)
    match_lines: list[int] = Field(default_factory=list)  # 1-based, localization


class DciResult(BaseModel):
    """Result of a ranked direct-corpus search."""

    query: str
    hits: list[DocHit] = Field(default_factory=list)
    docs_searched: int = 0


@dataclass
class DirectCorpusSearcher:
    """Grep/read/search over an in-memory document corpus.

    Args:
        documents: Mapping of ``doc_id -> text``.
    """

    documents: dict[str, str] = field(default_factory=dict)

    # -- primitives ---------------------------------------------------------

    def grep(
        self,
        pattern: str,
        *,
        regex: bool = False,
        ignore_case: bool = True,
        max_results: int = 100,
        context_lines: int = 0,
    ) -> list[GrepMatch]:
        """Return line-level matches of ``pattern`` across the corpus."""
        flags = re.IGNORECASE if ignore_case else 0
        try:
            rx = re.compile(pattern if regex else re.escape(pattern), flags)
        except re.error as exc:
            raise ValueError(f"invalid regex pattern: {exc}") from exc

        out: list[GrepMatch] = []
        for doc_id, text in self.documents.items():
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if rx.search(line):
                    lo = max(0, i - context_lines)
                    hi = min(len(lines), i + context_lines + 1)
                    out.append(
                        GrepMatch(
                            doc_id=doc_id,
                            line_no=i + 1,
                            line=line,
                            before=lines[lo:i] if context_lines else [],
                            after=lines[i + 1 : hi] if context_lines else [],
                        )
                    )
                    if len(out) >= max_results:
                        return out
        return out

    def read(
        self, doc_id: str, start: int | None = None, end: int | None = None
    ) -> str:
        """Read document ``doc_id``; optionally a 1-based inclusive line range."""
        text = self.documents.get(doc_id)
        if text is None:
            raise KeyError(doc_id)
        if start is None and end is None:
            return text
        lines = text.splitlines()
        s = max(1, start or 1)
        e = min(len(lines), end or len(lines))
        return "\n".join(lines[s - 1 : e])

    # -- ranked search ------------------------------------------------------

    def search(self, query: str, *, top_k: int = 10) -> DciResult:
        """Rank documents by query-term coverage with line localization.

        Score blends term coverage (fraction of distinct query terms found) with
        a saturating match-density term, so a doc that contains more of the query
        and hits it on multiple lines ranks higher. Each hit reports the matched
        terms and the (1-based) lines where they occur — the localization signal.
        """
        terms = list(dict.fromkeys(_terms(query)))  # distinct, order-preserving
        result = DciResult(query=query, docs_searched=len(self.documents))
        if not terms:
            return result

        term_rx = {t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in terms}
        hits: list[DocHit] = []
        for doc_id, text in self.documents.items():
            lines = text.splitlines() or [text]
            matched: set[str] = set()
            match_lines: set[int] = set()
            total_hits = 0
            for i, line in enumerate(lines):
                for t, rx in term_rx.items():
                    n = len(rx.findall(line))
                    if n:
                        matched.add(t)
                        match_lines.add(i + 1)
                        total_hits += n
            if not matched:
                continue
            coverage = len(matched) / len(terms)
            # saturating density bonus in [0,1): 1 - 1/(1+hits)
            density = 1.0 - 1.0 / (1.0 + total_hits)
            score = 0.75 * coverage + 0.25 * density
            hits.append(
                DocHit(
                    doc_id=doc_id,
                    score=round(min(1.0, score), 6),
                    term_coverage=round(coverage, 6),
                    matched_terms=sorted(matched),
                    match_lines=sorted(match_lines),
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        result.hits = hits[:top_k]
        return result


def searcher_from_nodes(
    nodes: list[dict[str, Any]],
    *,
    text_keys: tuple[str, ...] = ("content", "text", "summary", "description", "name"),
) -> DirectCorpusSearcher:
    """Build a :class:`DirectCorpusSearcher` from KG node dicts."""
    docs: dict[str, str] = {}
    for n in nodes:
        nid = str(n.get("id", "")) or f"node:{len(docs)}"
        body = " \n".join(str(n.get(k, "")) for k in text_keys if n.get(k))
        if body.strip():
            docs[nid] = body
    return DirectCorpusSearcher(documents=docs)
