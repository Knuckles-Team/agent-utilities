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
async def test_persist_and_resume_are_best_effort_without_backend():
    opt = GEPAOptimizer(signature_class=_Sig, base_prompt="p", evaluator_fn=_evaluator)
    # No live graph backend under test → both return falsy, never raise.
    persisted = await opt.persist_frontier("run-xyz")
    assert persisted in (True, False)
    restored = await opt.resume_frontier("run-xyz")
    assert restored == 0
