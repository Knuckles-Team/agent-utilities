"""Characterization tests for ``create_cohort`` (CX-AU-09).

CCN 11 at time of writing
(``agent_utilities/knowledge_graph/research/cohort.py``). These tests pin
the OBSERVED, black-box behaviour before any decomposition: the two
validation failures (bad arXiv ref, non-HTTPS repo), the exact job-id
naming scheme, the exact per-task extra_meta shape, the cohort node's
committed properties, and the returned summary dict shape.

Per the two-commit discipline, this file must be added and pass GREEN
against the UNMODIFIED ``cohort.py`` before any refactor commit, and must
not change during the refactor commit that follows.
"""

from __future__ import annotations

import time

import pytest

from agent_utilities.knowledge_graph.research.cohort import (
    SYNTHESIZE_TASK_TYPE,
    create_cohort,
)


@pytest.fixture(autouse=True)
def _native_graph_slice(monkeypatch):
    from agent_utilities.knowledge_graph.ingestion import envelope_ingest

    calls: list[dict] = []

    def _apply(engine, connector, entities, relationships=None, **kwargs):
        calls.append(
            {
                "connector": connector,
                "entities": entities,
                "relationships": relationships or [],
                "kwargs": kwargs,
            }
        )
        for entity in entities:
            row = dict(entity)
            node_id = row.pop("id")
            node_type = row.pop("node_type")
            engine.add_node(node_id, node_type=node_type, properties=row)
        return {"status": "success"}

    monkeypatch.setattr(envelope_ingest, "ingest_graph_slice", _apply)
    return calls


class _Engine:
    """Records every submit_task call verbatim, plus a minimal add_node."""

    def __init__(self):
        self.calls: list[dict] = []
        self._nodes: dict = {}

    def submit_task(
        self,
        target,
        is_codebase,
        provenance,
        task_type=None,
        extra_meta=None,
        job_id=None,
        skip_dedupe=False,
        **_kw,
    ):
        self.calls.append(
            {
                "target": target,
                "is_codebase": is_codebase,
                "provenance": provenance,
                "task_type": task_type,
                "extra_meta": extra_meta,
                "job_id": job_id,
                "skip_dedupe": skip_dedupe,
            }
        )
        return job_id

    def add_node(self, node_id, node_type=None, properties=None, **_kw):
        self._nodes[node_id] = {
            **self._nodes.get(node_id, {}),
            **(properties or {}),
            "type": node_type,
            "id": node_id,
        }


def test_invalid_paper_ref_raises_value_error(_native_graph_slice) -> None:
    eng = _Engine()
    with pytest.raises(ValueError, match="arXiv"):
        create_cohort(eng, papers=["not-a-valid-arxiv-ref"], repos=[])
    # OBSERVED: validation happens before ANY task is submitted or node committed.
    assert eng.calls == []
    assert eng._nodes == {}


def test_non_https_repo_raises_value_error(_native_graph_slice) -> None:
    eng = _Engine()
    with pytest.raises(ValueError, match="HTTPS"):
        create_cohort(eng, papers=[], repos=["/local/path/repo"])
    assert eng.calls == []
    assert eng._nodes == {}


def test_falsy_paper_and_repo_entries_are_filtered_before_validation(
    _native_graph_slice,
) -> None:
    # empty-string entries are dropped by the `if p` / `if r` filters before
    # _arxiv_id / the https check ever see them, so they cannot raise.
    eng = _Engine()
    out = create_cohort(eng, papers=["", "2606.10001"], repos=["", "https://x.test/r"])
    assert out["papers"] == 1
    assert out["repos"] == 1


def test_job_id_naming_scheme(_native_graph_slice) -> None:
    eng = _Engine()
    out = create_cohort(
        eng,
        papers=["2606.10001", "2606.10002"],
        repos=["https://example.test/r0.git"],
    )
    cid = out["cohort_id"]
    job_ids = [c["job_id"] for c in eng.calls]
    assert job_ids == [f"{cid}:p0", f"{cid}:p1", f"{cid}:r0", f"{cid}:synth"]


def test_paper_task_shape(_native_graph_slice) -> None:
    eng = _Engine()
    create_cohort(eng, papers=["2606.10001"], repos=[])
    call = eng.calls[0]
    assert call["target"] == "2606.10001"
    assert call["is_codebase"] is False
    assert call["provenance"] == {}
    assert call["task_type"] == "research_paper_fetch"
    assert call["skip_dedupe"] is True
    assert call["extra_meta"]["paper"] == {
        "id": "2606.10001",
        "url": "https://arxiv.org/abs/2606.10001",
        "score": 1.0,
    }


def test_repo_task_shape(_native_graph_slice) -> None:
    eng = _Engine()
    out = create_cohort(eng, papers=[], repos=["https://example.test/r0.git"])
    call = eng.calls[0]
    assert call["target"] == "https://example.test/r0.git"
    assert call["is_codebase"] is True
    assert call["task_type"] == "codebase"
    assert call["skip_dedupe"] is True
    # OBSERVED: repo extra_meta carries ONLY cohort_id, nothing else.
    assert call["extra_meta"] == {"cohort_id": out["cohort_id"]}


def test_synthesize_gate_is_last_and_carries_deadline(_native_graph_slice) -> None:
    before = time.time()
    eng = _Engine()
    out = create_cohort(eng, papers=["2606.10001"], repos=[], max_wait_s=120.0)
    after = time.time()
    synth_call = eng.calls[-1]
    assert synth_call["task_type"] == SYNTHESIZE_TASK_TYPE
    assert synth_call["target"] == f"cohort:{out['cohort_id']}"
    assert synth_call["extra_meta"]["cohort_id"] == out["cohort_id"]
    deadline = synth_call["extra_meta"]["deadline_unix"]
    assert before + 120.0 <= deadline <= after + 120.0


def test_cohort_node_committed_with_ingesting_status(_native_graph_slice) -> None:
    eng = _Engine()
    out = create_cohort(
        eng, papers=["2606.10001"], repos=["https://x.test/r"], goal="improve recall"
    )
    node = eng._nodes[out["cohort_id"]]
    assert node["status"] == "ingesting"
    assert node["member_count"] == 2
    assert node["papers"] == 1
    assert node["repos"] == 1
    assert node["concept"] == "AU-KG.coordination.research-cohort-barrier"


def test_returned_summary_shape_and_member_order(_native_graph_slice) -> None:
    eng = _Engine()
    out = create_cohort(
        eng,
        papers=["2606.10001", "2606.10002"],
        repos=["https://x.test/r0", "https://x.test/r1"],
    )
    assert out["papers"] == 2
    assert out["repos"] == 2
    assert len(out["members"]) == 4
    # OBSERVED: members list is papers first, then repos, in input order --
    # matches the exact job_id sequence.
    assert out["members"] == [
        f"{out['cohort_id']}:p0",
        f"{out['cohort_id']}:p1",
        f"{out['cohort_id']}:r0",
        f"{out['cohort_id']}:r1",
    ]
    assert out["synthesize_job"] == f"{out['cohort_id']}:synth"


def test_empty_cohort_still_submits_only_the_gate(_native_graph_slice) -> None:
    eng = _Engine()
    out = create_cohort(eng, papers=[], repos=[])
    assert out["members"] == []
    assert len(eng.calls) == 1
    assert eng.calls[0]["task_type"] == SYNTHESIZE_TASK_TYPE
