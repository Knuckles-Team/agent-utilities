#!/usr/bin/python
from __future__ import annotations

"""AU-INTEGRATION-R018.1 — docs page-count / orphan-page parity model.

Typed shape for the first bounded slice of the public-docs gate suite: full
page-count parity between the published Markdown pages and the MkDocs
navigation, with zero orphan pages (a publishable page reachable from
neither the navigation nor the generated documentation catalog).

This is the `.1` slice: the typed report and the refusal case -- any orphan
page, or any count mismatch between the page set and the navigation plus
catalog union, refuses parity. The remaining gates named by AU-INTEGRATION-R018
(README/theme parity, home-page tense, strict MkDocs build, and the
current-only, privacy, accessibility, theme, workflow and version gates)
are each their own later slice.
"""

from pydantic import BaseModel, Field


class DocsPageParityReport(BaseModel):
    """Inputs for one page-count/orphan-page parity check."""

    page_paths: frozenset[str] = Field(min_length=1)
    navigation_paths: frozenset[str]
    catalog_paths: frozenset[str]


class DocsPageParityDecision(BaseModel):
    """Outcome of a page-count/orphan-page parity check."""

    parity: bool
    orphan_pages: frozenset[str]
    reason: str = Field(min_length=1)


def evaluate_docs_page_parity(report: DocsPageParityReport) -> DocsPageParityDecision:
    """Apply the AU-INTEGRATION-R018.1 parity rule.

    Refuses parity when any publishable page is reachable from neither the
    navigation nor the generated catalog (an orphan page), or when the
    reachable set does not exactly match the full page set (full page-count
    parity against the site navigation).
    """
    reachable = report.navigation_paths | report.catalog_paths
    orphans = frozenset(report.page_paths - reachable)
    if orphans:
        return DocsPageParityDecision(
            parity=False,
            orphan_pages=orphans,
            reason=f"refused: {len(orphans)} orphan page(s) unreachable from nav or catalog",
        )
    unreachable_extra = reachable - report.page_paths
    if len(reachable) != len(report.page_paths) or unreachable_extra:
        return DocsPageParityDecision(
            parity=False,
            orphan_pages=frozenset(),
            reason="refused: navigation/catalog references a page outside the page set",
        )
    return DocsPageParityDecision(
        parity=True,
        orphan_pages=frozenset(),
        reason="full page-count parity with zero orphan pages",
    )
