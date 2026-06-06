"""CONCEPT:ORCH-1.12 / ORCH-1.13 — RLM-GEPA live entry-point glue.

Verifies the dynamic signature builder, the default GEPA evaluator (pure), and that the entry
functions return structured dicts without raising (robust MCP/CLI surface).
"""

from __future__ import annotations

import pytest

from agent_utilities.rlm.runner import (
    _default_evaluator,
    _dynamic_signature,
    run_rlm,
)


@pytest.mark.concept(id="ORCH-1.12")
def test_dynamic_signature_has_input_and_output():
    sig = _dynamic_signature("summarize the text", output_field="summary")
    fields = sig.model_fields
    assert "input_text" in fields and "summary" in fields
    assert sig.__doc__ == "summarize the text"
    # input_text marked input; summary marked output.
    assert fields["input_text"].json_schema_extra.get("is_input") is True  # type: ignore[union-attr]
    assert fields["summary"].json_schema_extra.get("is_output") is True  # type: ignore[union-attr]


@pytest.mark.concept(id="ORCH-1.13")
def test_default_evaluator_scores_match():
    class _Inst:
        reference_output = "Paris"

    hit = _default_evaluator(_Inst(), "The answer is Paris.", "prompt")
    miss = _default_evaluator(_Inst(), "I don't know", "prompt")
    assert hit["accuracy"] == 1.0 and "matched" in hit["feedback"]
    assert miss["accuracy"] == 0.0 and "missing" in miss["feedback"]
    # Efficiency rewards shorter prompts.
    assert _default_evaluator(_Inst(), "x", "a")["efficiency"] > _default_evaluator(
        _Inst(), "x", "a" * 5000
    )["efficiency"]


@pytest.mark.concept(id="ORCH-1.12")
@pytest.mark.asyncio
async def test_run_rlm_returns_structured_dict_without_raising():
    # Under AGENT_UTILITIES_TESTING the RLM may not fully execute; the entry must still return a
    # dict (ok True or False) and never raise.
    out = await run_rlm("echo the input", input_text="hello")
    assert isinstance(out, dict)
    assert out["task"] == "echo the input"
    assert "ok" in out
