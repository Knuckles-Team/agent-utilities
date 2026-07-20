"""Live Predict-RLM entry-point contract."""

from __future__ import annotations

import pytest

from agent_utilities.rlm.runner import _dynamic_signature, run_rlm


@pytest.mark.concept(id="AU-ORCH.execution.predict-rlm-runtime")
def test_dynamic_signature_has_input_and_output():
    sig = _dynamic_signature("summarize the text", output_field="summary")
    fields = sig.model_fields
    assert "input_text" in fields and "summary" in fields
    assert sig.__doc__ == "summarize the text"
    # input_text marked input; summary marked output.
    assert fields["input_text"].json_schema_extra.get("is_input") is True  # type: ignore[union-attr]
    assert fields["summary"].json_schema_extra.get("is_output") is True  # type: ignore[union-attr]


@pytest.mark.concept(id="AU-ORCH.execution.predict-rlm-runtime")
@pytest.mark.asyncio
async def test_run_rlm_returns_structured_dict_without_raising():
    # Under AGENT_UTILITIES_TESTING the RLM may not fully execute; the entry must still return a
    # dict (ok True or False) and never raise.
    out = await run_rlm("echo the input", input_text="hello")
    assert isinstance(out, dict)
    assert out["task"] == "echo the input"
    assert "ok" in out
