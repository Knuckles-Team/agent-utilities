"""Characterization tests for ``cohort_member_status`` (CX-AU-09).

CCN 16 at time of writing
(``agent_utilities/knowledge_graph/research/cohort.py``). These tests pin the
OBSERVED, black-box behaviour before any decomposition: the cohort_id and
SYNTHESIZE_TASK_TYPE filters, the exact status->bucket mapping (including the
scheduled-vs-pending split inside "ready" and the terminal double-count), and
the best-effort fallback when the work-item index cannot be read.

Per the two-commit discipline, this file must be added and pass GREEN
against the UNMODIFIED ``cohort.py`` before any refactor commit, and must
not change during the refactor commit that follows.
"""

from __future__ import annotations

import time

from agent_utilities.knowledge_graph.research.cohort import (
    SYNTHESIZE_TASK_TYPE,
    cohort_member_status,
)


class _Engine:
    def __init__(self, work=None, raise_on_index=False):
        self._work = work or {}
        self._raise = raise_on_index

    def _ingest_work_item_index(self):
        if self._raise:
            raise RuntimeError("index unavailable")
        return self._work


def _item(status, cohort_id="c1", type_="research_paper_fetch", next_retry_at=None):
    meta = {"cohort_id": cohort_id, "type": type_}
    item: dict = {"status": status, "metadata": meta}
    if next_retry_at is not None:
        # OBSERVED: next_retry_at is read off the top-level WorkItem dict, NOT
        # out of its metadata.
        item["next_retry_at"] = next_retry_at
    return item


def test_index_read_failure_degrades_to_one_unknown() -> None:
    counts = cohort_member_status(_Engine(raise_on_index=True), "c1")
    assert counts == {
        "total": 1,
        "pending": 0,
        "running": 0,
        "scheduled": 0,
        "blocked": 0,
        "completed": 0,
        "failed": 0,
        "terminal": 0,
        "unknown": 1,
    }


def test_items_from_other_cohorts_are_excluded() -> None:
    work = {"a": _item("succeeded", cohort_id="other-cohort")}
    counts = cohort_member_status(_Engine(work), "c1")
    assert counts["total"] == 0


def test_synthesize_gate_task_is_excluded_even_when_cohort_matches() -> None:
    work = {"a": _item("succeeded", type_=SYNTHESIZE_TASK_TYPE)}
    counts = cohort_member_status(_Engine(work), "c1")
    assert counts["total"] == 0


def test_succeeded_counts_completed_and_terminal() -> None:
    work = {"a": _item("succeeded")}
    counts = cohort_member_status(_Engine(work), "c1")
    assert counts["total"] == 1
    assert counts["completed"] == 1
    assert counts["terminal"] == 1
    assert counts["failed"] == 0


def test_failed_dead_letter_and_cancelled_all_count_as_failed_and_terminal() -> None:
    work = {
        "a": _item("failed"),
        "b": _item("dead_letter"),
        "c": _item("cancelled"),
    }
    counts = cohort_member_status(_Engine(work), "c1")
    assert counts["total"] == 3
    assert counts["failed"] == 3
    assert counts["terminal"] == 3
    assert counts["completed"] == 0


def test_leased_and_running_both_count_as_running_not_terminal() -> None:
    work = {"a": _item("leased"), "b": _item("running")}
    counts = cohort_member_status(_Engine(work), "c1")
    assert counts["running"] == 2
    assert counts["terminal"] == 0


def test_submitted_counts_as_blocked() -> None:
    work = {"a": _item("submitted")}
    counts = cohort_member_status(_Engine(work), "c1")
    assert counts["blocked"] == 1
    assert counts["pending"] == 0


def test_ready_with_future_retry_counts_as_scheduled() -> None:
    future = time.time() + 3600
    work = {"a": _item("ready", next_retry_at=future)}
    counts = cohort_member_status(_Engine(work), "c1")
    assert counts["scheduled"] == 1
    assert counts["pending"] == 0


def test_ready_with_past_or_absent_retry_counts_as_pending() -> None:
    past = time.time() - 3600
    work = {
        "a": _item("ready", next_retry_at=past),
        "b": _item("ready"),  # no next_retry_at at all -> defaults to 0.0 (past)
    }
    counts = cohort_member_status(_Engine(work), "c1")
    assert counts["pending"] == 2
    assert counts["scheduled"] == 0


def test_empty_status_counts_as_unknown() -> None:
    work = {"a": _item("")}
    counts = cohort_member_status(_Engine(work), "c1")
    assert counts["unknown"] == 1
    assert counts["total"] == 1


def test_unrecognized_nonempty_status_falls_back_to_pending() -> None:
    work = {"a": _item("some-weird-status")}
    counts = cohort_member_status(_Engine(work), "c1")
    assert counts["pending"] == 1
    assert counts["unknown"] == 0


def test_status_matching_is_case_insensitive() -> None:
    work = {"a": _item("SUCCEEDED")}
    counts = cohort_member_status(_Engine(work), "c1")
    assert counts["completed"] == 1


def test_mixed_batch_totals_and_terminal_sum_correctly() -> None:
    work = {
        "a": _item("succeeded"),
        "b": _item("failed"),
        "c": _item("running"),
        "d": _item("submitted"),
        "e": _item("ready"),
        "f": _item(""),
        "other-cohort": _item("succeeded", cohort_id="not-c1"),
        "gate": _item("succeeded", type_=SYNTHESIZE_TASK_TYPE),
    }
    counts = cohort_member_status(_Engine(work), "c1")
    assert counts["total"] == 6
    assert counts["terminal"] == 2  # succeeded + failed only
    assert (
        counts["completed"]
        + counts["failed"]
        + counts["running"]
        + counts["blocked"]
        + counts["pending"]
        + counts["scheduled"]
        + counts["unknown"]
        == 6
    )
