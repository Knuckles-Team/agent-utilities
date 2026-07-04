"""Tests for GuardrailEngine — Input/Output Interception (CONCEPT:AU-OS.config.secrets-authentication).

@pytest.mark.concept("AU-OS.deployment.platform-journey")
"""

import pytest

from agent_utilities.security.threat_defense_engine import (
    GuardrailAction,
    GuardrailEngine,
    GuardrailPhase,
    GuardrailRule,
)


@pytest.fixture
def engine() -> GuardrailEngine:
    return GuardrailEngine(
        rules=[
            GuardrailRule(
                name="block_secrets",
                pattern=r"(?:sk-|api_key\s*=\s*)[A-Za-z0-9]+",
                action=GuardrailAction.BLOCK,
                phase=GuardrailPhase.INPUT,
                description="Block API key leaks",
            ),
            GuardrailRule(
                name="redact_ssn",
                pattern=r"\b\d{3}-\d{2}-\d{4}\b",
                action=GuardrailAction.REDACT,
                phase=GuardrailPhase.OUTPUT,
                replacement="[SSN REDACTED]",
                description="Redact SSN from output",
            ),
            GuardrailRule(
                name="warn_profanity",
                pattern="badword",
                is_regex=False,
                action=GuardrailAction.WARN,
                phase=GuardrailPhase.INPUT,
                description="Warn on profanity",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Rule model
# ---------------------------------------------------------------------------


class TestGuardrailRule:
    def test_default_action(self):
        r = GuardrailRule(pattern="test")
        assert r.action == GuardrailAction.BLOCK
        assert r.phase == GuardrailPhase.INPUT

    def test_auto_id(self):
        r = GuardrailRule(name="my_rule", pattern="test")
        assert r.id.startswith("guardrail:my_rule:")

    def test_disabled_rule(self):
        r = GuardrailRule(pattern="test", enabled=False)
        assert r.enabled is False


# ---------------------------------------------------------------------------
# Input blocking
# ---------------------------------------------------------------------------


class TestInputBlocking:
    def test_blocks_api_key(self, engine):
        result = engine.check_input("Here is my key: sk-abc123xyz")
        assert result.should_block is True
        assert len(result.block_reasons) >= 1

    def test_allows_clean_input(self, engine):
        result = engine.check_input("Hello, how are you?")
        assert result.should_block is False
        assert len(result.triggered_results) == 0

    def test_case_insensitive_regex(self, engine):
        result = engine.check_input("my api_key = SECRET123")
        assert result.should_block is True


# ---------------------------------------------------------------------------
# Output redaction
# ---------------------------------------------------------------------------


class TestOutputRedaction:
    def test_redacts_ssn(self, engine):
        result = engine.check_output("Your SSN is 123-45-6789.")
        assert result.redacted_text == "Your SSN is [SSN REDACTED]."
        assert len(result.triggered_results) == 1
        assert result.triggered_results[0].action == GuardrailAction.REDACT

    def test_no_redaction_clean_output(self, engine):
        result = engine.check_output("Everything is fine.")
        assert result.redacted_text == "Everything is fine."

    def test_multiple_ssn_redaction(self, engine):
        result = engine.check_output("SSN: 111-22-3333 and 444-55-6666")
        assert "[SSN REDACTED]" in result.redacted_text
        assert "111-22-3333" not in result.redacted_text


# ---------------------------------------------------------------------------
# Warn-only mode
# ---------------------------------------------------------------------------


class TestWarnMode:
    def test_warn_does_not_block(self, engine):
        result = engine.check_input("This contains badword in it")
        assert result.should_block is False
        assert len(result.triggered_results) >= 1
        warn_results = [
            r for r in result.triggered_results if r.action == GuardrailAction.WARN
        ]
        assert len(warn_results) >= 1

    def test_keyword_match(self, engine):
        result = engine.check_input("badword detected")
        triggered = [r for r in result.triggered_results if r.triggered]
        assert len(triggered) >= 1


# ---------------------------------------------------------------------------
# Multi-rule aggregation
# ---------------------------------------------------------------------------


class TestMultiRuleAggregation:
    def test_block_overrides_warn(self):
        engine = GuardrailEngine(
            rules=[
                GuardrailRule(
                    name="warn",
                    pattern="test",
                    is_regex=False,
                    action=GuardrailAction.WARN,
                    phase=GuardrailPhase.INPUT,
                ),
                GuardrailRule(
                    name="block",
                    pattern="dangerous",
                    is_regex=False,
                    action=GuardrailAction.BLOCK,
                    phase=GuardrailPhase.INPUT,
                ),
            ]
        )
        result = engine.check_input("test and dangerous content")
        assert result.should_block is True

    def test_disabled_rules_skipped(self):
        engine = GuardrailEngine(
            rules=[
                GuardrailRule(
                    pattern="secret",
                    is_regex=False,
                    action=GuardrailAction.BLOCK,
                    phase=GuardrailPhase.INPUT,
                    enabled=False,
                ),
            ]
        )
        result = engine.check_input("this is secret")
        assert result.should_block is False


# ---------------------------------------------------------------------------
# from_config construction
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_from_config_basic(self):
        config = [
            {"name": "r1", "pattern": "test", "action": "block", "phase": "input"},
            {"name": "r2", "pattern": r"\d+", "action": "warn", "phase": "output"},
        ]
        engine = GuardrailEngine.from_config(config)
        assert engine.has_guardrails is True

    def test_from_config_invalid_skipped(self):
        config = [
            {"name": "r1", "pattern": "test"},
            {"invalid": "no_pattern"},  # Missing required field
        ]
        engine = GuardrailEngine.from_config(config)
        assert len(engine._rules) >= 1

    def test_from_config_empty(self):
        engine = GuardrailEngine.from_config([])
        assert engine.has_guardrails is False


# ---------------------------------------------------------------------------
# PolicyEngine adapter
# ---------------------------------------------------------------------------


class TestPolicyAdapter:
    def test_adapter_creates_policy_result(self, engine):
        adapter = engine.to_policy_adapter()
        assert adapter.name == "threat_defense_engine"
        result = adapter.evaluate("clean input", "clean output")
        assert result.allowed is True

    def test_adapter_blocks_on_input(self, engine):
        adapter = engine.to_policy_adapter()
        result = adapter.evaluate("sk-secretkey123", "clean output")
        assert result.allowed is False
        assert result.severity == "block"

    def test_adapter_detects_output_trigger(self, engine):
        adapter = engine.to_policy_adapter()
        result = adapter.evaluate("clean input", "SSN: 123-45-6789")
        # Redact is not a block, so should still be allowed
        assert result.metadata["output_triggered"] >= 1


# ---------------------------------------------------------------------------
# Static redaction utility
# ---------------------------------------------------------------------------


class TestRedactionUtility:
    def test_regex_redaction(self):
        text = GuardrailEngine.apply_redaction(
            "Call 555-1234 now",
            r"\d{3}-\d{4}",
            "[PHONE]",
        )
        assert text == "Call [PHONE] now"

    def test_keyword_redaction(self):
        text = GuardrailEngine.apply_redaction(
            "Remove password here",
            "password",
            "[REDACTED]",
            is_regex=False,
        )
        assert text == "Remove [REDACTED] here"

    def test_invalid_regex_falls_back(self):
        text = GuardrailEngine.apply_redaction(
            "test [bad regex",
            "[bad regex",
            "[FIXED]",
        )
        assert "[FIXED]" in text


# ---------------------------------------------------------------------------
# Trigger log
# ---------------------------------------------------------------------------


class TestTriggerLog:
    def test_trigger_log_accumulates(self, engine):
        engine.check_input("sk-key123")
        engine.check_output("SSN: 111-22-3333")
        assert len(engine.trigger_log) >= 2


# ---------------------------------------------------------------------------
# PII Sanitizer & Ephemeral Context Tests
# ---------------------------------------------------------------------------

from agent_utilities.security.guardrails import (
    EphemeralContext,
    PiiSanitizer,
    PIISanitizerPolicy,
)


class TestPiiSanitizer:
    def test_sanitize_text_ssn(self):
        sanitizer = PiiSanitizer()
        text = "My SSN is 123-45-6789."
        assert sanitizer.sanitize_text(text) == "My SSN is [REDACTED_SSN]."

    def test_sanitize_text_tax_id(self):
        sanitizer = PiiSanitizer()
        # Test 50-state tax_id / EIN pattern
        text = "Company EIN is 12-3456789."
        assert sanitizer.sanitize_text(text) == "Company EIN is [REDACTED_TAX_ID]."

    def test_sanitize_dict_recursive(self):
        sanitizer = PiiSanitizer()
        data = {
            "name": "John Doe",
            "sensitive": {
                "ssn": "123-45-6789",
                "nested_list": ["12-3456789", "clean text"],
            },
        }
        sanitized = sanitizer.sanitize_dict(data)
        assert sanitized["sensitive"]["ssn"] == "[REDACTED_SSN]"
        assert sanitized["sensitive"]["nested_list"][0] == "[REDACTED_TAX_ID]"
        assert sanitized["sensitive"]["nested_list"][1] == "clean text"


class TestEphemeralContext:
    def test_scrub_and_zero(self):
        buf = bytearray(b"sensitive_password")
        data_list = ["temp", "data"]
        data_dict = {"key": "val"}

        with EphemeralContext(buf=buf, data_list=data_list, data_dict=data_dict) as ctx:
            assert ctx["buf"] == buf

        # Verify scrubbing
        assert all(b == 0 for b in buf)
        assert len(data_list) == 0
        assert len(data_dict) == 0


class TestPIISanitizerPolicy:
    def test_policy_blocks_pii(self):
        policy = PIISanitizerPolicy()
        result = policy.evaluate("clean input", "My SSN is 123-45-6789")
        assert result.allowed is False
        assert "PII detected" in result.reason
        assert "ssn" in result.metadata["detected_types"]

    def test_policy_allows_clean(self):
        policy = PIISanitizerPolicy()
        result = policy.evaluate("clean input", "clean output")
        assert result.allowed is True
