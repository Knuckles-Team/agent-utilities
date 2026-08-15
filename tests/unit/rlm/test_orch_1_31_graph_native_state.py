"""CONCEPT:AU-ORCH.optimization.graph-native-optimization-state — Graph-Native Optimization State (resumable GEPA)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_utilities.rlm.gepa import (
    Candidate,
    GEPAOptimizer,
    ParetoCandidatePool,
)


def _evaluator(instance, prediction, prompt):
    return {"accuracy": 1.0, "efficiency": 1.0}, "ok"


class _Sig(BaseModel):
    """t"""

    result: str = ""


@pytest.mark.concept(id="AU-ORCH.optimization.graph-native-optimization-state")
def test_frontier_snapshot_roundtrip():
    pool = ParetoCandidatePool(objectives=["accuracy", "efficiency"])
    pool.update(
        [
            Candidate(
                id="a",
                prompt_text="A",
                generation=1,
                scores={"accuracy": 0.9, "efficiency": 0.5},
                parent_ids=["base"],
            ),
            Candidate(
                id="b",
                prompt_text="B",
                generation=2,
                scores={"accuracy": 0.5, "efficiency": 0.9},
            ),
        ]
    )
    snap = pool.to_snapshot()
    assert isinstance(snap, list) and {r["id"] for r in snap} >= {"a", "b"}

    # Restore into a fresh pool — candidates + ancestry survive.
    fresh = ParetoCandidatePool(objectives=["accuracy", "efficiency"])
    n = fresh.load_snapshot(snap)
    assert n == len(snap)
    restored = {c.id: c for c in fresh.get_frontier()}
    assert "a" in restored and restored["a"].parent_ids == ["base"]
    assert restored["a"].prompt_text == "A"


@pytest.mark.concept(id="AU-ORCH.optimization.graph-native-optimization-state")
@pytest.mark.asyncio
async def test_persist_and_resume_are_best_effort_without_backend(monkeypatch):
    """``persist_frontier``/``resume_frontier`` degrade gracefully with no backend.

    Both reach the ONE process-wide ``IntelligenceGraphEngine`` singleton
    (``agent_utilities/graph/client.py``). That singleton, once constructed by
    ANY earlier test in this worker's session (e.g. one requesting
    ``tiny_engine``/``engine_graph``), stays alive for the rest of the session —
    unlike ``GraphComputeEngine``, whose singleton ``tests/conftest.py``'s
    autouse ``isolate_graph_compute_engine`` resets per-test. So "no backend" is
    not something this test can rely on the ambient environment to provide (it
    depends on pytest-xdist's file-to-worker grouping); force it explicitly so
    the assertion is deterministic regardless of test order.
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    def _no_backend(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("The process graph engine has no active backend")

    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_or_create", classmethod(_no_backend)
    )

    opt = GEPAOptimizer(signature_class=_Sig, base_prompt="p", evaluator_fn=_evaluator)
    # No live graph backend under test → both return falsy, never raise.
    persisted = await opt.persist_frontier("run-xyz")
    assert persisted in (True, False)
    restored = await opt.resume_frontier("run-xyz")
    assert restored == 0
