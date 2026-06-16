"""Extract research-paper links from web/markdown content (CONCEPT:KG-2.7).

Turns a page like a "latest papers" research roundup into a list of downloadable
:class:`PaperRef`s (arXiv ids, DOIs, direct PDF urls), and decides whether a page
*is* such a roundup (enough distinct scholarly links) so the ingestion engine can
auto-acquire the papers it points at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A page is treated as a research roundup at/above this many distinct papers.
ROUNDUP_MIN_PAPERS = 3

# arXiv: abs/pdf URLs and bare ids (YYMM.NNNNN[vN], plus old-style cs/9901001).
_ARXIV_URL = re.compile(
    r"https?://arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)",
    re.IGNORECASE,
)
_ARXIV_BARE = re.compile(r"(?<![\w./])([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)(?![\w/])")
# DOI: the standard 10.<registrant>/<suffix> shape (trim trailing punctuation).
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\)\]\"<>]+)", re.IGNORECASE)
# Direct PDF links (non-arxiv handled as a raw download).
_PDF_URL = re.compile(r"https?://[^\s\)\]\"<>]+?\.pdf\b", re.IGNORECASE)


@dataclass(frozen=True)
class PaperRef:
    """One discovered paper reference. ``ident`` is what the downloader consumes."""

    kind: str  # "arxiv" | "doi" | "pdf"
    ident: str  # arxiv id, DOI, or full PDF url
    url: str = ""  # canonical source url (for provenance / MENTIONS edge)

    def key(self) -> str:
        return f"{self.kind}:{self.ident}".lower()


def extract_paper_links(text: str) -> list[PaperRef]:
    """Find all distinct research-paper references in markdown/HTML text.

    arXiv is matched first (most specific); a PDF/DOI that is really an arXiv link
    is normalized to the arXiv id so we don't download it twice.
    """
    refs: list[PaperRef] = []
    seen: set[str] = set()

    def add(ref: PaperRef) -> None:
        k = ref.key()
        if k not in seen:
            seen.add(k)
            refs.append(ref)

    for m in _ARXIV_URL.finditer(text):
        add(PaperRef("arxiv", m.group(1), url=m.group(0)))
    for m in _ARXIV_BARE.finditer(text):
        add(PaperRef("arxiv", m.group(1), url=f"https://arxiv.org/abs/{m.group(1)}"))
    for m in _DOI.finditer(text):
        add(
            PaperRef(
                "doi", m.group(1).rstrip(".,);"), url=f"https://doi.org/{m.group(1)}"
            )
        )
    for m in _PDF_URL.finditer(text):
        url = m.group(0)
        if "arxiv.org" in url.lower():
            continue  # already captured as an arxiv id above
        add(PaperRef("pdf", url, url=url))
    return refs


def is_research_roundup(
    refs: list[PaperRef], *, min_papers: int = ROUNDUP_MIN_PAPERS
) -> bool:
    """True when a page points at enough distinct scholarly papers to auto-acquire.

    Only arXiv/DOI references count toward the threshold — a page with a couple of
    incidental PDF links is not a roundup; a "latest research" digest cites many.
    """
    scholarly = [r for r in refs if r.kind in ("arxiv", "doi")]
    return len(scholarly) >= min_papers
