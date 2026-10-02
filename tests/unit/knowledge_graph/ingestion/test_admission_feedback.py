"""Retrieval usage yields reviewed admission PROPOSALS, never changes."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from agent_utilities.decide.learning.session import LearningSession
from agent_utilities.knowledge_graph.ingestion import embedding_admission
from agent_utilities.knowledge_graph.ingestion.admission_feedback import (
    AdmissionProposal,
    admitted_classes,
    propose_admission_changes,
    review_admission_feedback,
)


def _usage(rows: list[dict[str, Any]], outcomes: int) -> dict[str, Any]:
    return {"result": "usage", "min_support": 10, "outcomes": outcomes, "rows": rows}


def test_absence_over_enough_traffic_proposes_a_demotion() -> None:
    usage = _usage([], outcomes=500)
    proposals = propose_admission_changes(usage, admitted=["prose", "sql_enum"])
    assert [(p.content_class, p.finding) for p in proposals] == [
        ("prose", "never_retrieved"),
        ("sql_enum", "never_retrieved"),
    ]
    assert all(p.status == "proposed" for p in proposals)
    assert propose_admission_changes(_usage([], outcomes=5), admitted=["prose"]) == []


def test_k_anonymised_rows_are_not_absences() -> None:
    rows = [{"content_class": "prose", "returned": 0, "cited": 0}]
    assert propose_admission_changes(_usage(rows, 500), admitted=["prose"]) == []


def test_retrieved_but_never_cited_is_proposed_and_cited_is_not() -> None:
    rows = [
        {"content_class": "prose", "returned": 40, "cited": 0},
        {"content_class": "generated", "returned": 40, "cited": 3},
    ]
    proposals = propose_admission_changes(
        _usage(rows, 50), admitted=["generated", "prose"]
    )
    assert [(p.content_class, p.finding) for p in proposals] == [
        ("prose", "retrieved_never_cited")
    ]
    assert proposals[0].evidence == {"outcomes": 50, "returned": 40, "cited": 0}


class _Transport:
    async def sql(self, query: str) -> Any:
        if "decision_class_usage" in query:
            return {"columns": ["content_class", "returned", "cited"], "rows": []}
        assert "decision_retrieval_outcomes" in query
        return {"columns": ["runs"], "rows": [[1000]]}

    def run(self, call: Any) -> Any:
        return asyncio.run(call)


def test_review_reads_eg_usage_and_leaves_the_table_alone() -> None:
    before = set(embedding_admission.NEVER_EMBED_CLASSES)
    seen: list[Sequence[AdmissionProposal]] = []
    session = LearningSession(_Transport(), "tenant-t")
    proposals = asyncio.run(review_admission_feedback(session, seen.append))
    assert seen == [proposals]
    assert {p.content_class for p in proposals} == set(admitted_classes())
    assert set(embedding_admission.NEVER_EMBED_CLASSES) == before
