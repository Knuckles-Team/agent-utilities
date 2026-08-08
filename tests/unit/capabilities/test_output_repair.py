"""Structured-output failure taxonomy + bounded repair (CONCEPT:AU-AHE.harness.structured-output-repair).

Covers: pure classification for each taxonomy class, a live-path repair that
succeeds and stamps ``output_repair_attempts`` on the result, a live-path
repair that exhausts its bound and fails closed with the typed
``StructuredOutputRepairExhausted`` (never silently swallowed), a budget
exhaustion mid-output that is recorded but NEVER retried, and the composition
seam (default-ON, opt-out, retries threading).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from agent_utilities.capabilities.output_repair import (
    DEFAULT_MAX_OUTPUT_REPAIRS,
    StructuredOutputRepair,
    StructuredOutputRepairExhausted,
    classify_raw_output,
    classify_validation_error,
    output_repair_retries,
)
from agent_utilities.core.contextual_model import (
    create_context_agent,
    use_grounding_policy,
)


class Out(BaseModel):
    value: int


# ---------------------------------------------------------------------------
# classify_raw_output -- pre-validation taxonomy (malformed_json / truncated /
# refused / empty)
# ---------------------------------------------------------------------------


def test_classify_empty_string():
    assert classify_raw_output("") == "empty"
    assert classify_raw_output("   ") == "empty"


def test_classify_empty_none_and_dict():
    assert classify_raw_output(None) == "empty"
    assert classify_raw_output({}) == "empty"


def test_classify_refused():
    assert classify_raw_output("I cannot help with that request.") == "refused"
    assert classify_raw_output("I'm sorry, but I won't do that.") == "refused"


def test_classify_malformed_json():
    assert classify_raw_output("{not valid json at all,,,}") == "malformed_json"


def test_classify_truncated():
    # Cut off mid-object: unbalanced braces, inside nothing unterminated but
    # missing the closing brace.
    assert classify_raw_output('{"value": 4') == "truncated"


def test_classify_truncated_mid_string():
    assert classify_raw_output('{"value": "abc') == "truncated"


def test_classify_valid_json_returns_none():
    assert classify_raw_output('{"value": 42}') is None


def test_classify_non_json_text_is_malformed():
    # This hook only fires for structured (schema-validated) output, so any
    # non-JSON text reaching it means the model didn't even attempt the
    # requested shape.
    assert classify_raw_output("hello there") == "malformed_json"


# ---------------------------------------------------------------------------
# classify_validation_error -- post-parse taxonomy (schema_invalid / wrong_type)
# ---------------------------------------------------------------------------


def test_classify_wrong_type_error():
    try:
        Out.model_validate({"value": "not-an-int-nor-numeric"})
    except ValidationError as exc:
        assert classify_validation_error(exc) == "wrong_type"
    else:
        pytest.fail("expected a ValidationError")


def test_classify_schema_invalid_missing_field():
    try:
        Out.model_validate({})
    except ValidationError as exc:
        assert classify_validation_error(exc) == "schema_invalid"
    else:
        pytest.fail("expected a ValidationError")


# ---------------------------------------------------------------------------
# output_repair_retries -- the Agent-level retries= bump
# ---------------------------------------------------------------------------


def test_output_repair_retries_enabled():
    retries = output_repair_retries(structured_output_repair=True, max_output_repairs=2)
    assert retries == {"output": 2}


def test_output_repair_retries_disabled_returns_none():
    assert (
        output_repair_retries(structured_output_repair=False, max_output_repairs=2)
        is None
    )


# ---------------------------------------------------------------------------
# Live-path: repair succeeds and is visible in provenance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_succeeds_after_malformed_json_and_is_visible_on_result():
    calls = {"n": 0}

    def func(messages: list[Any], info: Any) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(parts=[TextPart(content="not json at all")])
        return ModelResponse(parts=[TextPart(content='{"value": 42}')])

    agent = create_context_agent(
        FunctionModel(func),
        output_type=Out,
        capabilities=[StructuredOutputRepair()],
        retries={"output": DEFAULT_MAX_OUTPUT_REPAIRS},
        default_capabilities=False,
    )

    with use_grounding_policy("none"):
        result = await agent.run("go")

    assert result.output == Out(value=42)
    # Wire-First / truthfulness: a repaired run must be visible, never
    # indistinguishable from a clean first-try success.
    attempts = result.output_repair_attempts
    assert len(attempts) == 1
    assert attempts[0]["classification"] == "malformed_json"
    assert attempts[0]["action"] == "retried"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_repair_succeeds_with_no_attempts_when_first_try_is_valid():
    def func(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='{"value": 1}')])

    agent = create_context_agent(
        FunctionModel(func),
        output_type=Out,
        capabilities=[StructuredOutputRepair()],
        default_capabilities=False,
    )

    with use_grounding_policy("none"):
        result = await agent.run("go")

    assert result.output == Out(value=1)
    assert not hasattr(result, "output_repair_attempts")


# ---------------------------------------------------------------------------
# Live-path: repair exhausts its bound and fails closed with a typed error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_exhausts_and_fails_closed_with_typed_error():
    def func(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="still not json")])

    agent = create_context_agent(
        FunctionModel(func),
        output_type=Out,
        capabilities=[StructuredOutputRepair(max_repairs=2)],
        retries={"output": 2},
        default_capabilities=False,
    )

    with pytest.raises(StructuredOutputRepairExhausted) as exc_info:
        with use_grounding_policy("none"):
            await agent.run("go")

    exc = exc_info.value
    assert len(exc.attempts) == 3  # 2 retried + 1 exhausted
    assert [a.action for a in exc.attempts] == ["retried", "retried", "exhausted"]
    assert all(a.classification == "malformed_json" for a in exc.attempts)


@pytest.mark.asyncio
async def test_repair_schema_invalid_retries_then_succeeds():
    calls = {"n": 0}

    def func(messages: list[Any], info: Any) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(
                parts=[TextPart(content='{"value": "not-coercible-abc"}')]
            )
        return ModelResponse(parts=[TextPart(content='{"value": 7}')])

    agent = create_context_agent(
        FunctionModel(func),
        output_type=Out,
        capabilities=[StructuredOutputRepair()],
        retries={"output": DEFAULT_MAX_OUTPUT_REPAIRS},
        default_capabilities=False,
    )

    with use_grounding_policy("none"):
        result = await agent.run("go")

    assert result.output == Out(value=7)
    assert result.output_repair_attempts[0]["classification"] == "wrong_type"


# ---------------------------------------------------------------------------
# Live-path: budget exhaustion mid-output is recorded but NEVER retried
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_exceeded_mid_output_is_recorded_and_not_retried():
    def func(messages: list[Any], info: Any) -> ModelResponse:
        # Force pydantic-ai's own usage-limit check to trip: consume more
        # tokens than allowed by returning a large response and setting a
        # tiny limit below.
        return ModelResponse(
            parts=[TextPart(content='{"value": 1}')],
        )

    from pydantic_ai.usage import UsageLimits

    cap = StructuredOutputRepair()
    agent = create_context_agent(
        FunctionModel(func),
        output_type=Out,
        capabilities=[cap],
        default_capabilities=False,
    )

    with pytest.raises(UsageLimitExceeded):
        with use_grounding_policy("none"):
            await agent.run("go", usage_limits=UsageLimits(request_limit=0))


# ---------------------------------------------------------------------------
# composition.py -- default-ON, opt-out, retries threading
# ---------------------------------------------------------------------------


def test_default_runtime_capabilities_includes_structured_output_repair():
    from agent_utilities.capabilities.composition import default_runtime_capabilities

    defaults = default_runtime_capabilities()
    matches = [c for c in defaults if isinstance(c, StructuredOutputRepair)]
    assert len(matches) == 1
    assert matches[0].max_repairs == DEFAULT_MAX_OUTPUT_REPAIRS


def test_default_runtime_capabilities_structured_output_repair_false_excludes_it():
    from agent_utilities.capabilities.composition import default_runtime_capabilities

    defaults = default_runtime_capabilities(structured_output_repair=False)
    assert not any(isinstance(c, StructuredOutputRepair) for c in defaults)


def test_default_runtime_capabilities_honors_max_output_repairs_override():
    from agent_utilities.capabilities.composition import default_runtime_capabilities

    defaults = default_runtime_capabilities(max_output_repairs=5)
    matches = [c for c in defaults if isinstance(c, StructuredOutputRepair)]
    assert matches[0].max_repairs == 5


def test_create_context_agent_threads_output_repair_retries_by_default():
    from agent_utilities.core.contextual_model import create_context_agent

    def func(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='{"value": 1}')])

    agent = create_context_agent(FunctionModel(func), output_type=Out)
    assert agent._max_output_retries == DEFAULT_MAX_OUTPUT_REPAIRS


def test_create_context_agent_respects_explicit_retries():
    from agent_utilities.core.contextual_model import create_context_agent

    def func(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='{"value": 1}')])

    agent = create_context_agent(
        FunctionModel(func), output_type=Out, retries={"output": 9}
    )
    assert agent._max_output_retries == 9


@pytest.mark.asyncio
async def test_budget_exceeded_mid_output_provenance_survives_the_raise():
    """The recorded attempt must reach the caller, not a dropped instance.

    Regression (waves 1-5 gate): ``on_run_error`` recorded the attempt onto the
    per-run instance ``for_run()`` hands out, then raised — so nothing retained
    that instance and the record was provably unrecoverable. The test that
    covered this path asserted only ``pytest.raises(UsageLimitExceeded)`` and
    would have passed with the ``_record(...)`` call deleted outright.
    """
    capability = await StructuredOutputRepair().for_run(MagicMock())
    error = UsageLimitExceeded(
        "Exceeded the total_tokens_limit of 10 (total_tokens=11)"
    )

    with pytest.raises(UsageLimitExceeded) as excinfo:
        await capability.on_run_error(MagicMock(), error=error)

    attempts = getattr(excinfo.value, "output_repair_attempts", None)
    assert attempts, "the propagated exception must carry the recorded attempt"
    assert attempts[0]["classification"] == "budget_exceeded_mid_output"
    assert attempts[0]["action"] == "propagated"


@pytest.mark.asyncio
async def test_immutable_exception_type_still_propagates_unchanged():
    """A __slots__ exception cannot carry provenance — the ORIGINAL error must
    still propagate untouched rather than being masked by an AttributeError."""

    class _Slotted(UsageLimitExceeded):
        __slots__ = ()

    capability = await StructuredOutputRepair().for_run(MagicMock())
    error = _Slotted("Exceeded the total_tokens_limit of 10 (total_tokens=11)")

    with pytest.raises(_Slotted) as excinfo:
        await capability.on_run_error(MagicMock(), error=error)
    assert excinfo.value is error
