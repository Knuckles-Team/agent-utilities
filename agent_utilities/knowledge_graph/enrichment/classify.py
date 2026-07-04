"""Test-quality classification rules (CONCEPT:EG-KG.storage.nonblocking-checkpoint Phase 1).

Pure functions over the extracted ``TestEntity`` metrics. These mirror the
"needs work" axioms the software ontology asserts in OWL — keeping the logic in
one place means the Python path and the OWL reasoner agree. The output is stored
on the graph node (``needs_work`` + ``issues``) so the question "which pytests
need work" becomes a graph query, not an ad-hoc scan.
"""

from __future__ import annotations

from pydantic import BaseModel

from .models import TestEntity


class TestThresholds(BaseModel):
    __test__ = False  # not a pytest test class

    mock_heavy_min_mocks: int = 3
    mock_heavy_max_assertions: int = 1
    overfixtured_min: int = 6


class TestIssue(BaseModel):
    __test__ = False  # not a pytest test class

    code: str  # e.g. "MockHeavyTest"
    severity: str  # "high" | "medium" | "low"
    detail: str


def classify_test(
    t: TestEntity, thresholds: TestThresholds | None = None
) -> list[TestIssue]:
    """Return the quality issues for a single test (empty == healthy)."""
    th = thresholds or TestThresholds()
    issues: list[TestIssue] = []

    if t.is_skipped:
        issues.append(
            TestIssue(
                code="DormantTest",
                severity="medium",
                detail=f"skipped via marks {t.marks} — not exercising code",
            )
        )

    # Don't double-flag a skipped test for weak assertions (it doesn't run).
    if not t.is_skipped:
        if t.effective_assertions == 0:
            issues.append(
                TestIssue(
                    code="AssertionFreeTest",
                    severity="high",
                    detail="no assert / pytest.raises — verifies nothing",
                )
            )
        if (
            t.mock_count >= th.mock_heavy_min_mocks
            and t.effective_assertions <= th.mock_heavy_max_assertions
        ):
            issues.append(
                TestIssue(
                    code="MockHeavyTest",
                    severity="high",
                    detail=(
                        f"{t.mock_count} mocks vs {t.effective_assertions} assertions "
                        "— asserts on wiring, not behaviour"
                    ),
                )
            )

    if t.fixture_count >= th.overfixtured_min:
        issues.append(
            TestIssue(
                code="OverFixturedTest",
                severity="low",
                detail=f"{t.fixture_count} fixtures — heavy setup, brittle",
            )
        )

    return issues


def test_needs_work(t: TestEntity, thresholds: TestThresholds | None = None) -> bool:
    return bool(classify_test(t, thresholds))
