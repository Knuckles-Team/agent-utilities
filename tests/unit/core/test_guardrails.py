"""Tests for the Policy / Guardrails Engine.

Concept: policy-guardrails
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.security.guardrails import (
    ContentFilterPolicy,
    CostBudgetPolicy,
    MaxTokensPolicy,
    OutputSchemaPolicy,
    PolicyEngine,
    PolicyViolation,
    guardrail,
)

# ---------------------------------------------------------------------------
# MaxTokensPolicy
# ---------------------------------------------------------------------------


@pytest.mark.concept("policy-guardrails")
def test_max_tokens_policy_blocks() -> None:
    policy = MaxTokensPolicy(max_tokens=10)
    result = policy.evaluate("", " ".join(["word"] * 100))
    assert result.allowed is False
    assert "exceeds" in result.reason


@pytest.mark.concept("policy-guardrails")
def test_max_tokens_policy_allows() -> None:
    policy = MaxTokensPolicy(max_tokens=10_000)
    result = policy.evaluate("", "A short response.")
    assert result.allowed is True
    assert result.reason == ""


# ---------------------------------------------------------------------------
# ContentFilterPolicy
# ---------------------------------------------------------------------------


@pytest.mark.concept("policy-guardrails")
def test_content_filter_detects_ssn() -> None:
    policy = ContentFilterPolicy()
    result = policy.evaluate("", "My SSN is 123-45-6789")
    assert result.allowed is False
    assert "ssn" in result.reason.lower() or "sensitive" in result.reason.lower()
    assert result.metadata["matches"]["ssn"] == 1


@pytest.mark.concept("policy-guardrails")
def test_content_filter_clean_output() -> None:
    policy = ContentFilterPolicy()
    result = policy.evaluate("What is 2+2?", "The answer is 4.")
    assert result.allowed is True


@pytest.mark.concept("policy-guardrails")
def test_content_filter_detects_email() -> None:
    policy = ContentFilterPolicy()
    result = policy.evaluate("", "Contact john@example.com for details")
    assert result.allowed is False
    assert result.metadata["matches"]["email"] == 1


# ---------------------------------------------------------------------------
# OutputSchemaPolicy
# ---------------------------------------------------------------------------


@pytest.mark.concept("policy-guardrails")
def test_output_schema_policy_valid() -> None:
    policy = OutputSchemaPolicy(required_keys=["status", "result"])
    output = json.dumps({"status": "ok", "result": 42})
    result = policy.evaluate("", output)
    assert result.allowed is True


@pytest.mark.concept("policy-guardrails")
def test_output_schema_policy_missing_keys() -> None:
    policy = OutputSchemaPolicy(required_keys=["status", "data"])
    output = json.dumps({"status": "ok"})
    result = policy.evaluate("", output)
    assert result.allowed is False
    assert "data" in result.reason


# ---------------------------------------------------------------------------
# CostBudgetPolicy
# ---------------------------------------------------------------------------


@pytest.mark.concept("policy-guardrails")
def test_cost_budget_tracks_usage() -> None:
    policy = CostBudgetPolicy(max_total_tokens=1000, max_cost_usd=1.0)
    ctx = {"agent_id": "test_agent"}
    # First call — should be within budget
    r1 = policy.evaluate("short input", "short output", ctx)
    assert r1.allowed is True

    usage = policy.get_usage("test_agent")
    assert usage["total_tokens"] > 0


@pytest.mark.concept("policy-guardrails")
def test_cost_budget_blocks_over_limit() -> None:
    policy = CostBudgetPolicy(max_total_tokens=10, max_cost_usd=0.001)
    ctx = {"agent_id": "expensive_agent"}
    # Pump tokens well over budget
    big_text = " ".join(["word"] * 500)
    r1 = policy.evaluate(big_text, big_text, ctx)
    assert r1.allowed is False
    assert "exceeded" in r1.reason.lower()


@pytest.mark.concept("policy-guardrails")
def test_cost_budget_reset() -> None:
    policy = CostBudgetPolicy(max_total_tokens=100)
    policy.record_usage("agent_x", 200)
    policy.reset("agent_x")
    usage = policy.get_usage("agent_x")
    assert usage["total_tokens"] == 0


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------


@pytest.mark.concept("policy-guardrails")
def test_policy_engine_runs_all_rules() -> None:
    engine = PolicyEngine()
    engine.register(MaxTokensPolicy(max_tokens=10_000))
    engine.register(ContentFilterPolicy())

    results = engine.evaluate(input_text="hello", output_text="world")
    assert len(results) == 2
    assert all(r.allowed for r in results)


@pytest.mark.concept("policy-guardrails")
def test_policy_engine_blocks_on_severity() -> None:
    engine = PolicyEngine()
    engine.register(MaxTokensPolicy(max_tokens=5))  # Will block
    engine.register(ContentFilterPolicy())

    big_text = " ".join(["word"] * 100)
    with pytest.raises(PolicyViolation) as exc_info:
        engine.evaluate(output_text=big_text, raise_on_block=True)

    assert len(exc_info.value.violations) >= 1
    assert exc_info.value.violations[0].policy_name == "max_tokens"


@pytest.mark.concept("policy-guardrails")
@pytest.mark.asyncio
async def test_guardrail_decorator_pre_check() -> None:
    engine = PolicyEngine()
    engine.register(ContentFilterPolicy())

    @guardrail(engine, check_input=True, check_output=False)
    async def generate(prompt: str) -> str:
        return "safe response"

    # SSN in input should be blocked
    with pytest.raises(PolicyViolation):
        await generate("My SSN is 123-45-6789")
