"""Tests for AU-INTEGRATION-R018.1's docs page-count/orphan-page parity model.

Covers the full-parity acceptance case and the refusal case the requirement
names directly: a page reachable from neither the navigation nor the
generated catalog (an orphan page) must always refuse parity.
"""

from __future__ import annotations

from agent_utilities.core.docs_page_parity import (
    DocsPageParityReport,
    evaluate_docs_page_parity,
)


def test_full_parity_with_no_orphans_is_accepted() -> None:
    report = DocsPageParityReport(
        page_paths=frozenset({"docs/index.md", "docs/guide.md"}),
        navigation_paths=frozenset({"docs/index.md"}),
        catalog_paths=frozenset({"docs/guide.md"}),
    )

    decision = evaluate_docs_page_parity(report)

    assert decision.parity is True
    assert decision.orphan_pages == frozenset()


def test_orphan_page_refuses_parity() -> None:
    """Refusal case: a page in neither nav nor catalog is an orphan."""
    report = DocsPageParityReport(
        page_paths=frozenset({"docs/index.md", "docs/orphan.md"}),
        navigation_paths=frozenset({"docs/index.md"}),
        catalog_paths=frozenset(),
    )

    decision = evaluate_docs_page_parity(report)

    assert decision.parity is False
    assert decision.orphan_pages == frozenset({"docs/orphan.md"})
    assert "orphan page" in decision.reason


def test_navigation_referencing_a_page_outside_the_set_refuses_parity() -> None:
    report = DocsPageParityReport(
        page_paths=frozenset({"docs/index.md"}),
        navigation_paths=frozenset({"docs/index.md", "docs/removed.md"}),
        catalog_paths=frozenset(),
    )

    decision = evaluate_docs_page_parity(report)

    assert decision.parity is False
    assert "outside the page set" in decision.reason
