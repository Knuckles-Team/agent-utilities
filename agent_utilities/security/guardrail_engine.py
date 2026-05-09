#!/usr/bin/python
"""Guardrail Callback Engine — Input/Output Interception (CONCEPT:OS-5.8).

Push-based guardrail interception with block, redact, and warn actions
on both input and output. Ported from MATE's guardrail_callback.py.

Integrates with the existing PolicyEngine as an adapter, adding
automatic interception rather than manual evaluate() calls.

OWL: :GuardrailTrigger rdfs:subClassOf :SecurityFinding
"""

from __future__ import annotations

import logging
import re
import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GuardrailAction(StrEnum):
    """Action to take when a guardrail is triggered. CONCEPT:OS-5.8."""

    BLOCK = "block"
    REDACT = "redact"
    WARN = "warn"
    LOG = "log"


class GuardrailPhase(StrEnum):
    """Phase at which the guardrail runs. CONCEPT:OS-5.8."""

    INPUT = "input"
    OUTPUT = "output"


class GuardrailRule(BaseModel):
    """A single guardrail rule definition. CONCEPT:OS-5.8.

    Ported from MATE's guardrail config JSON schema. Each rule
    defines a pattern (regex or keyword), an action, and the phase
    at which it applies.
    """

    id: str = ""
    name: str = ""
    pattern: str  # regex or keyword
    is_regex: bool = True
    action: GuardrailAction = GuardrailAction.BLOCK
    phase: GuardrailPhase = GuardrailPhase.INPUT
    replacement: str = "[REDACTED]"
    description: str = ""
    enabled: bool = True

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = f"guardrail:{self.name or self.pattern[:20]}:{time.time()}"


class GuardrailResult(BaseModel):
    """Result of a single guardrail check. CONCEPT:OS-5.8."""

    rule_id: str = ""
    guardrail_type: str = ""
    triggered: bool = False
    action: GuardrailAction = GuardrailAction.LOG
    phase: GuardrailPhase = GuardrailPhase.INPUT
    matched_content: str = ""
    redacted_text: str = ""
    details: str = ""
    timestamp: float = Field(default_factory=time.time)


class GuardrailCheckSummary(BaseModel):
    """Aggregated results from checking all guardrail rules. CONCEPT:OS-5.8."""

    phase: GuardrailPhase = GuardrailPhase.INPUT
    total_rules_checked: int = 0
    triggered_results: list[GuardrailResult] = Field(default_factory=list)
    should_block: bool = False
    block_reasons: list[str] = Field(default_factory=list)
    redacted_text: str = ""


class GuardrailEngine:
    """Push-based guardrail interception engine. CONCEPT:OS-5.8.

    Ported from MATE's guardrail_callback.py. Provides automatic
    input/output interception with block, redact, and warn actions.

    Unlike the existing PolicyEngine (pull-based), this engine
    runs checks proactively on every input/output and can modify
    content via redaction.
    """

    def __init__(self, rules: list[GuardrailRule] | None = None) -> None:
        self._rules = rules or []
        self._trigger_log: list[GuardrailResult] = []

    @property
    def has_guardrails(self) -> bool:
        """Whether any guardrail rules are configured."""
        return any(r.enabled for r in self._rules)

    @property
    def trigger_log(self) -> list[GuardrailResult]:
        """History of all triggered guardrail results."""
        return list(self._trigger_log)

    @classmethod
    def from_config(cls, config: list[dict[str, Any]]) -> GuardrailEngine:
        """Construct from JSON/dict config list.

        Mirrors MATE's GuardrailEngine.from_json() pattern.

        Parameters
        ----------
        config : list[dict]
            List of rule definitions with pattern, action, phase, etc.

        Returns
        -------
        GuardrailEngine
            Configured engine instance.
        """
        rules = []
        for item in config:
            try:
                rule = GuardrailRule(**item)
                rules.append(rule)
            except Exception as exc:
                logger.warning("Failed to parse guardrail rule: %s — %s", item, exc)
        return cls(rules=rules)

    def add_rule(self, rule: GuardrailRule) -> None:
        """Add a guardrail rule."""
        self._rules.append(rule)

    def check_input(self, text: str) -> GuardrailCheckSummary:
        """Check input text against all INPUT-phase guardrail rules.

        Parameters
        ----------
        text : str
            The input text to check.

        Returns
        -------
        GuardrailCheckSummary
            Aggregated results including whether to block.
        """
        return self._check(text, GuardrailPhase.INPUT)

    def check_output(self, text: str) -> GuardrailCheckSummary:
        """Check output text against all OUTPUT-phase guardrail rules.

        Parameters
        ----------
        text : str
            The output text to check.

        Returns
        -------
        GuardrailCheckSummary
            Aggregated results including whether to block.
        """
        return self._check(text, GuardrailPhase.OUTPUT)

    def _check(self, text: str, phase: GuardrailPhase) -> GuardrailCheckSummary:
        """Core check logic for a given phase."""
        applicable = [r for r in self._rules if r.enabled and r.phase == phase]
        summary = GuardrailCheckSummary(
            phase=phase,
            total_rules_checked=len(applicable),
        )

        current_text = text
        for rule in applicable:
            matched = self._match_rule(current_text, rule)
            if not matched:
                continue

            result = GuardrailResult(
                rule_id=rule.id,
                guardrail_type=rule.name or rule.pattern[:30],
                triggered=True,
                action=rule.action,
                phase=phase,
                matched_content=matched,
                details=f"Rule '{rule.name}' triggered: {rule.description}",
            )

            if rule.action == GuardrailAction.BLOCK:
                summary.should_block = True
                summary.block_reasons.append(
                    f"{rule.name or rule.pattern}: {rule.description}"
                )

            elif rule.action == GuardrailAction.REDACT:
                current_text = self.apply_redaction(
                    current_text, rule.pattern, rule.replacement, rule.is_regex
                )
                result.redacted_text = current_text

            summary.triggered_results.append(result)
            self._trigger_log.append(result)

        summary.redacted_text = current_text
        return summary

    @staticmethod
    def _match_rule(text: str, rule: GuardrailRule) -> str:
        """Check if a rule's pattern matches the text.

        Returns the matched content, or empty string if no match.
        """
        try:
            if rule.is_regex:
                match = re.search(rule.pattern, text, re.IGNORECASE)
                if match:
                    return match.group(0)
            else:
                if rule.pattern.lower() in text.lower():
                    return rule.pattern
        except re.error as exc:
            logger.warning("Invalid regex in guardrail rule %s: %s", rule.id, exc)
        return ""

    @staticmethod
    def apply_redaction(
        text: str,
        pattern: str,
        replacement: str = "[REDACTED]",
        is_regex: bool = True,
    ) -> str:
        """Apply redaction to text using a pattern.

        Parameters
        ----------
        text : str
            The text to redact.
        pattern : str
            Pattern to match (regex or keyword).
        replacement : str
            Replacement string.
        is_regex : bool
            Whether pattern is a regex.

        Returns
        -------
        str
            Redacted text.
        """
        try:
            if is_regex:
                return re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            else:
                return text.replace(pattern, replacement)
        except re.error:
            return text.replace(pattern, replacement)

    def to_policy_adapter(self) -> Any:
        """Create a PolicyEngine-compatible adapter.

        Returns a PolicyRule-compatible object that can be registered
        with the existing PolicyEngine for unified evaluation.
        """
        from agent_utilities.security.guardrails import PolicyResult

        engine = self

        class _GuardrailPolicyAdapter:
            name = "guardrail_engine"

            def evaluate(
                self,
                input_text: str,
                output_text: str,
                context: dict[str, Any] | None = None,
            ) -> PolicyResult:
                input_check = engine.check_input(input_text)
                output_check = engine.check_output(output_text)
                blocked = input_check.should_block or output_check.should_block
                reasons = input_check.block_reasons + output_check.block_reasons
                return PolicyResult(
                    allowed=not blocked,
                    policy_name="guardrail_engine",
                    reason="; ".join(reasons) if reasons else "",
                    severity="block" if blocked else "warn",
                    metadata={
                        "input_triggered": len(input_check.triggered_results),
                        "output_triggered": len(output_check.triggered_results),
                    },
                )

        return _GuardrailPolicyAdapter()
