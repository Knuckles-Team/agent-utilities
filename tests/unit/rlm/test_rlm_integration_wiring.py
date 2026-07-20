"""Integration checks for the live Predict-RLM path.

Each test exercises an *existing* class/entry point to prove the new behavior is actually invoked.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_utilities.rlm.predict_rlm import OutputField, PredictRLM
from agent_utilities.rlm.skills import Skill
from agent_utilities.rlm.telemetry import classify_failure


@pytest.mark.concept(id="ORCH-1.28")
def test_mounted_skill_instructions_reach_the_prompt():
    """C integration: a mounted Skill's SOP instructions must appear in the generated prompt."""

    class Sig(BaseModel):
        """solve it"""

        result: str = OutputField(description="answer")

    rlm = PredictRLM(Sig)
    rlm.mount_skill_unit(
        Skill(name="sop", instructions="STEP 1: always call login first.")
    )
    prompt = rlm._generate_instruction_prompt({})
    assert "SKILL INSTRUCTIONS" in prompt
    assert (
        "STEP 1: always call login first." in prompt
    )  # the SOP actually reaches the model


@pytest.mark.concept(id="ORCH-1.29")
@pytest.mark.asyncio
async def test_run_rlm_classifies_failures_on_live_path():
    """D integration: run_rlm attaches a failure_class on failure (typed signal for the optimizer)."""
    from agent_utilities.rlm.runner import run_rlm

    out = await run_rlm("do a thing", input_text="x")
    # Under test the runtime may succeed (TestModel) or fail; if it fails it MUST be classified.
    assert "ok" in out
    if out["ok"] is False:
        assert out["failure_class"] == classify_failure(out["error"])
