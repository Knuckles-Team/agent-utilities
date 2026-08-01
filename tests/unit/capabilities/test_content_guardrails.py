"""Default-ON content guardrails (CONCEPT:AU-OS.safety.harness-guardrails-adoption).

Covers: the live agent path for all three guardrails that replace the dead
``PolicyEngine`` (D-48) — a PII input/output actually gets redacted, a
secret-leak (forbidden-content) output actually gets redacted (D-OP-2: this
used to ``block`` the whole output; changed to redact-in-place so a legitimate
answer that happens to carry a credential-shaped substring — e.g. a k8s
manifest — survives with only the secret removed, same posture as the PII
guard), and an output-schema violation actually gets caught — all exercised
through a real ``pydantic_ai.Agent`` run (``FunctionModel``), never by calling
the guard helper directly. Also covers the composition seam (default-ON,
opt-out, required-keys no-op) mirroring ``test_output_repair.py``'s pattern.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness import InputGuardrail, OutputGuardrail

from agent_utilities.capabilities.content_guardrails import (
    output_schema_guardrail,
    pii_redaction_guardrails,
    secret_leak_guardrail,
)
from agent_utilities.core.contextual_model import (
    create_context_agent,
    use_grounding_policy,
)
from agent_utilities.http.redaction import REDACTED

# ---------------------------------------------------------------------------
# Live path: PII redaction (output)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pii_in_output_is_redacted_live_path():
    def func(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="My SSN is 123-45-6789.")])

    agent = create_context_agent(
        FunctionModel(func), output_type=str, capabilities=pii_redaction_guardrails(),
        default_capabilities=False,
    )
    with use_grounding_policy("none"):
        result = await agent.run("go")

    assert "123-45-6789" not in result.output
    assert "[REDACTED_SSN]" in result.output


@pytest.mark.asyncio
async def test_clean_output_passes_pii_guard_unchanged_live_path():
    def func(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="The answer is 4.")])

    agent = create_context_agent(
        FunctionModel(func), output_type=str, capabilities=pii_redaction_guardrails(),
        default_capabilities=False,
    )
    with use_grounding_policy("none"):
        result = await agent.run("go")

    assert result.output == "The answer is 4."


@pytest.mark.asyncio
async def test_pii_in_input_is_redacted_before_reaching_model_live_path():
    seen_prompts: list[str] = []

    def func(messages: list[Any], info: Any) -> ModelResponse:
        for message in messages:
            for part in message.parts:
                if hasattr(part, "content") and isinstance(part.content, str):
                    seen_prompts.append(part.content)
        return ModelResponse(parts=[TextPart(content="ok")])

    agent = create_context_agent(
        FunctionModel(func), output_type=str, capabilities=pii_redaction_guardrails(),
        default_capabilities=False,
    )
    with use_grounding_policy("none"):
        await agent.run("Contact john@example.com about my SSN 123-45-6789")

    combined = " ".join(seen_prompts)
    assert "123-45-6789" not in combined
    assert "john@example.com" not in combined
    assert "[REDACTED_SSN]" in combined
    assert "[REDACTED_EMAIL]" in combined


# ---------------------------------------------------------------------------
# Live path: secret-leak redaction (D-OP-2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_secret_shaped_output_is_redacted_not_blocked_live_path():
    fixture_value = (
        "AKIAEXAMPLEABCDEFGHI"  # sanitizer:ignore - synthetic, not a live credential
    )
    fixture_secret = f"here is the key: {fixture_value}"

    def func(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=fixture_secret)])

    agent = create_context_agent(
        FunctionModel(func), output_type=str, capabilities=[secret_leak_guardrail()],
        default_capabilities=False,
    )
    with use_grounding_policy("none"):
        result = await agent.run("go")

    assert fixture_value not in result.output
    assert REDACTED in result.output
    assert "here is the key:" in result.output


@pytest.mark.asyncio
async def test_k8s_manifest_secret_is_redacted_and_content_survives_live_path():
    """Regression for D-OP-2: a real k8s manifest (``ExternalSecret``/
    ``stringData`` output) trips ``contains_secret`` as a TRUE positive on
    content agents are legitimately asked to handle in this workspace. The old
    ``block`` behaviour discarded the entire answer; the guard must now strip
    only the credential value and let the rest of the manifest through."""
    manifest = (
        "apiVersion: v1\n"
        "kind: Secret\n"
        "metadata:\n"
        "  name: graphos-env\n"
        "type: Opaque\n"
        "stringData:\n"
        "  secret: hunter2\n"
    )

    def func(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=manifest)])

    agent = create_context_agent(
        FunctionModel(func), output_type=str, capabilities=[secret_leak_guardrail()],
        default_capabilities=False,
    )
    with use_grounding_policy("none"):
        result = await agent.run("go")

    assert "hunter2" not in result.output
    assert REDACTED in result.output
    # Surrounding manifest structure survives -- this is not a block.
    assert "kind: Secret" in result.output
    assert "name: graphos-env" in result.output
    assert "stringData:" in result.output


@pytest.mark.asyncio
async def test_clean_output_passes_secret_leak_guard_live_path():
    def func(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="The weather is sunny today.")])

    agent = create_context_agent(
        FunctionModel(func), output_type=str, capabilities=[secret_leak_guardrail()],
        default_capabilities=False,
    )
    with use_grounding_policy("none"):
        result = await agent.run("go")

    assert result.output == "The weather is sunny today."


@pytest.mark.asyncio
async def test_secret_leak_guard_does_not_trip_on_ordinary_prose_live_path():
    """Measured non-triggers from the D-OP-2 investigation: prose mentioning
    "secret"/"token" in a non-assignment shape, and an excluded scheme keyword
    as the value, must pass through unchanged."""
    prose = "The secret: this design is not finished yet. token: bearer semantics."

    def func(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=prose)])

    agent = create_context_agent(
        FunctionModel(func), output_type=str, capabilities=[secret_leak_guardrail()],
        default_capabilities=False,
    )
    with use_grounding_policy("none"):
        result = await agent.run("go")

    assert result.output == prose


# ---------------------------------------------------------------------------
# Live path: output-schema violation
#
# CONCEPT:AU-AHE.harness.loop-exit-conditions — reconciled against the harness's ``retry`` output-guardrail
# verdict (D-track-7): a missing-key / invalid-JSON violation is RECOVERABLE, so the
# guard now hands the output back to the model (``GuardrailResult.retry()``) instead
# of discarding it outright (``block()``). See ``output_schema_guardrail``'s docstring.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_missing_required_keys_sends_it_back_to_the_model_to_fix_live_path():
    """The model gets ONE retry with the specific missing keys named, fixes it,
    and the run succeeds — this is the ``retry`` verdict actually earning its
    keep over a hard ``block``: the salvageable answer is not discarded."""
    calls: list[str] = []

    def func(messages: list[Any], info: Any) -> ModelResponse:
        calls.append("call")
        if len(calls) == 1:
            return ModelResponse(parts=[TextPart(content='{"status": "ok"}')])
        return ModelResponse(
            parts=[TextPart(content='{"status": "ok", "result": 42}')]
        )

    agent = create_context_agent(
        FunctionModel(func),
        output_type=str,
        capabilities=[output_schema_guardrail(["status", "result"])],
        default_capabilities=False,
    )
    with use_grounding_policy("none"):
        result = await agent.run("go")

    assert len(calls) == 2  # the guard forced exactly one retry
    assert result.output == '{"status": "ok", "result": 42}'


@pytest.mark.asyncio
async def test_output_still_missing_required_keys_after_retries_exhausted_live_path():
    """A model that never fixes it still fails closed — ``retry`` is not an
    infinite loop, it is pydantic-ai's own bounded output-retry budget."""

    def func(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='{"status": "ok"}')])

    agent = create_context_agent(
        FunctionModel(func),
        output_type=str,
        capabilities=[output_schema_guardrail(["status", "result"])],
        default_capabilities=False,
    )

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        with use_grounding_policy("none"):
            await agent.run("go")

    assert "output retries" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_output_not_valid_json_sends_it_back_to_the_model_to_fix_live_path():
    calls: list[str] = []

    def func(messages: list[Any], info: Any) -> ModelResponse:
        calls.append("call")
        if len(calls) == 1:
            return ModelResponse(parts=[TextPart(content="not json at all")])
        return ModelResponse(
            parts=[TextPart(content='{"status": "ok", "result": 42}')]
        )

    agent = create_context_agent(
        FunctionModel(func),
        output_type=str,
        capabilities=[output_schema_guardrail(["status", "result"])],
        default_capabilities=False,
    )
    with use_grounding_policy("none"):
        result = await agent.run("go")

    assert len(calls) == 2
    assert result.output == '{"status": "ok", "result": 42}'


@pytest.mark.asyncio
async def test_output_with_required_keys_present_passes_live_path():
    def func(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='{"status": "ok", "result": 42}')])

    agent = create_context_agent(
        FunctionModel(func),
        output_type=str,
        capabilities=[output_schema_guardrail(["status", "result"])],
        default_capabilities=False,
    )
    with use_grounding_policy("none"):
        result = await agent.run("go")

    assert result.output == '{"status": "ok", "result": 42}'


@pytest.mark.asyncio
async def test_output_schema_guardrail_is_a_noop_with_no_required_keys_live_path():
    def func(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="not json at all")])

    agent = create_context_agent(
        FunctionModel(func),
        output_type=str,
        capabilities=[output_schema_guardrail([])],
        default_capabilities=False,
    )
    with use_grounding_policy("none"):
        result = await agent.run("go")

    assert result.output == "not json at all"


# ---------------------------------------------------------------------------
# composition.py -- default-ON, opt-out, required-keys threading
# ---------------------------------------------------------------------------


def test_default_runtime_capabilities_includes_content_guardrails():
    from agent_utilities.capabilities.composition import default_runtime_capabilities

    defaults = default_runtime_capabilities()
    input_guardrails = [c for c in defaults if isinstance(c, InputGuardrail)]
    output_guardrails = [c for c in defaults if isinstance(c, OutputGuardrail)]

    # Assert WHICH guards are wired, not just how many. A bare count breaks the
    # moment a legitimate new guard is added (it did: the migrated G8
    # prompt-injection guard, D-OB-17) without saying anything about whether the
    # guards that matter are still there — which is the actual contract.
    from agent_utilities.capabilities.content_guardrails import (
        _pii_guard,
        _prompt_injection_guard,
        _secret_leak_guard,
    )

    input_guards = [c.guard for c in input_guardrails]
    output_guards = [c.guard for c in output_guardrails]

    # PII redaction contributes one InputGuardrail + one OutputGuardrail; the
    # migrated prompt-injection guard contributes the second InputGuardrail; the
    # secret-leak and output-schema guardrails each contribute one more
    # OutputGuardrail.
    assert input_guards.count(_pii_guard) == 1
    assert input_guards.count(_prompt_injection_guard) == 1
    assert len(input_guardrails) == 2
    assert output_guards.count(_pii_guard) == 1
    assert output_guards.count(_secret_leak_guard) == 1
    assert len(output_guardrails) == 3


def test_default_runtime_capabilities_content_guardrails_false_excludes_them():
    from agent_utilities.capabilities.composition import default_runtime_capabilities

    defaults = default_runtime_capabilities(content_guardrails=False)
    assert not any(isinstance(c, InputGuardrail) for c in defaults)
    assert not any(isinstance(c, OutputGuardrail) for c in defaults)


@pytest.mark.asyncio
async def test_create_context_agent_wires_content_guardrails_by_default_live_path():
    """The whole point of D-48: the guard must fire via the shared composition
    seam (``create_context_agent``), not only when constructed by hand."""
    from agent_utilities.core.contextual_model import (
        create_context_agent,
        use_grounding_policy,
    )

    def func(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="My SSN is 123-45-6789.")])

    agent = create_context_agent(FunctionModel(func), output_type=str)
    with use_grounding_policy("none"):
        result = await agent.run("go")

    assert "123-45-6789" not in result.output
    assert "[REDACTED_SSN]" in result.output
