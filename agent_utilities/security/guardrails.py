#!/usr/bin/python
from __future__ import annotations

"""Policy & Guardrails Engine.

CONCEPT:AU-OS.config.secrets-authentication

Provides automated policy enforcement for agent inputs and outputs.
Rules implement the :class:`PolicyRule` protocol and are aggregated by
:class:`PolicyEngine`.  Includes built-in rules for content filtering,
output validation, token limits, and cost-budget tracking.

Concept: policy-guardrails
"""


import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class PolicyResult(BaseModel):
    """Result of evaluating a single policy rule."""

    allowed: bool
    policy_name: str
    reason: str = ""
    severity: Literal["warn", "block"] = "block"
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyViolation(Exception):
    """Raised when a blocking policy rule is violated."""

    def __init__(self, violations: list[PolicyResult]) -> None:
        self.violations = violations
        names = [v.policy_name for v in violations]
        super().__init__(f"Policy violation(s): {', '.join(names)}")


# ---------------------------------------------------------------------------
# Policy rule protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PolicyRule(Protocol):
    """Protocol for policy rules."""

    name: str

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        context: dict[str, Any] | None = None,
    ) -> PolicyResult:
        """Evaluate input/output against this policy."""
        return PolicyResult(allowed=True, policy_name=self.name)


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------

# Common PII patterns
_PII_PATTERNS: dict[str, str] = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "tax_id": r"\b\d{2}-\d{7}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
}


import gc


class EphemeralContext:
    """Context manager for securely cleaning up transient memory.

    Zeroes out bytearrays and mutable collections, then runs GC.
    """

    def __init__(self, **kwargs) -> None:
        self.transients = kwargs

    def __enter__(self) -> dict[str, Any]:
        return self.transients

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.scrub()

    def scrub(self) -> None:
        """Explicitly overwrite and clear transient data structures."""

        def _clear(item: Any):
            if isinstance(item, dict):
                for k, v in list(item.items()):
                    _clear(v)
                    item[k] = None
                item.clear()
            elif isinstance(item, list):
                for i in range(len(item)):
                    _clear(item[i])
                    item[i] = None
                item.clear()
            elif isinstance(item, bytearray):
                for i in range(len(item)):
                    item[i] = 0
            elif hasattr(item, "__dict__"):
                for k, v in list(item.__dict__.items()):
                    _clear(v)
                    setattr(item, k, None)

        for key, val in list(self.transients.items()):
            _clear(val)
            self.transients[key] = None
        self.transients.clear()
        gc.collect()


class PiiSanitizer:
    """Named Entity Recognition & Regex-based PII Sanitizer.

    Dynamically redacts sensitive identifiers from text, lists, and dicts.
    """

    def __init__(self, patterns: dict[str, str] | None = None) -> None:
        self.patterns = patterns or _PII_PATTERNS
        self._compiled = {k: re.compile(v) for k, v in self.patterns.items()}

    def sanitize_text(self, text: str) -> str:
        """Redact PII from string content."""
        if not text or not isinstance(text, str):
            return text

        # Simple NER and pattern redaction
        sanitized = text
        for label, regex in self._compiled.items():
            replacement = f"[REDACTED_{label.upper()}]"
            sanitized = regex.sub(replacement, sanitized)

        return sanitized

    def sanitize_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Deep-sanitize dictionary values and keys."""
        if not isinstance(data, dict):
            return data

        sanitized: dict[Any, Any] = {}
        for k, v in data.items():
            # Sanitize keys too
            clean_k = (
                self.sanitize_text(k)
                if isinstance(k, str)
                else self.sanitize_text(str(k))
            )
            if isinstance(v, dict):
                sanitized[clean_k] = self.sanitize_dict(v)
            elif isinstance(v, list):
                sanitized[clean_k] = [self.sanitize(item) for item in v]
            elif isinstance(v, str):
                sanitized[clean_k] = self.sanitize_text(v)
            else:
                sanitized[clean_k] = v
        return sanitized

    def sanitize(self, data: Any) -> Any:
        """Generic deep-sanitizer for strings, lists, dicts."""
        if isinstance(data, str):
            return self.sanitize_text(data)
        elif isinstance(data, dict):
            return self.sanitize_dict(data)
        elif isinstance(data, list):
            return [self.sanitize(item) for item in data]
        return data


@dataclass
class PIISanitizerPolicy:
    """Enforces zero-PII policies on inputs and outputs.

    Concept: policy-guardrails
    """

    name: str = "pii_sanitizer"
    sanitizer: PiiSanitizer = field(default_factory=PiiSanitizer)
    severity: Literal["warn", "block"] = "block"

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        context: dict[str, Any] | None = None,
    ) -> PolicyResult:
        # Check input and output for presence of PII
        combined = f"{input_text}\n{output_text}"
        detected = []
        for label, regex in self.sanitizer._compiled.items():
            if regex.search(combined):
                detected.append(label)

        allowed = len(detected) == 0
        return PolicyResult(
            allowed=allowed,
            policy_name=self.name,
            reason="" if allowed else f"PII detected: {detected}",
            severity=self.severity,
            metadata={"detected_types": detected},
        )


@dataclass
class MaxTokensPolicy:
    """Blocks if output exceeds a maximum token count (estimated by words).

    Concept: policy-guardrails
    """

    name: str = "max_tokens"
    max_tokens: int = 10_000

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        context: dict[str, Any] | None = None,
    ) -> PolicyResult:
        # Rough token estimate: ~0.75 tokens per word
        estimated_tokens = int(len(output_text.split()) * 1.33)
        allowed = estimated_tokens <= self.max_tokens
        return PolicyResult(
            allowed=allowed,
            policy_name=self.name,
            reason=""
            if allowed
            else f"Output exceeds {self.max_tokens} tokens (est. {estimated_tokens})",
            severity="block",
            metadata={
                "estimated_tokens": estimated_tokens,
                "max_tokens": self.max_tokens,
            },
        )


@dataclass
class ContentFilterPolicy:
    """Blocks if output matches forbidden regex patterns (PII, sensitive data).

    Concept: policy-guardrails
    """

    name: str = "content_filter"
    patterns: dict[str, str] = field(default_factory=lambda: dict(_PII_PATTERNS))
    severity: Literal["warn", "block"] = "block"

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        context: dict[str, Any] | None = None,
    ) -> PolicyResult:
        matches: dict[str, list[str]] = {}
        combined = f"{input_text}\n{output_text}"
        for label, pattern in self.patterns.items():
            found = re.findall(pattern, combined)
            if found:
                matches[label] = found

        allowed = len(matches) == 0
        return PolicyResult(
            allowed=allowed,
            policy_name=self.name,
            reason=""
            if allowed
            else f"Detected sensitive content: {list(matches.keys())}",
            severity=self.severity,
            metadata={"matches": {k: len(v) for k, v in matches.items()}},
        )


@dataclass
class OutputSchemaPolicy:
    """Blocks if structured output doesn't contain expected keys.

    Concept: policy-guardrails
    """

    name: str = "output_schema"
    required_keys: list[str] = field(default_factory=list)

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        context: dict[str, Any] | None = None,
    ) -> PolicyResult:
        import json as _json

        try:
            data = _json.loads(output_text)
        except (ValueError, TypeError):
            # Non-JSON output — only block if we have required keys
            if self.required_keys:
                return PolicyResult(
                    allowed=False,
                    policy_name=self.name,
                    reason="Output is not valid JSON but schema validation was required",
                    severity="block",
                )
            return PolicyResult(allowed=True, policy_name=self.name)

        if not isinstance(data, dict):
            return PolicyResult(
                allowed=not self.required_keys,
                policy_name=self.name,
                reason="Output is JSON but not an object",
                severity="block" if self.required_keys else "warn",
            )

        missing = [k for k in self.required_keys if k not in data]
        allowed = len(missing) == 0
        return PolicyResult(
            allowed=allowed,
            policy_name=self.name,
            reason="" if allowed else f"Missing required keys: {missing}",
            severity="block",
            metadata={"missing_keys": missing},
        )


@dataclass
class CostBudgetPolicy:
    """Tracks and enforces per-agent cost budgets (tokens and estimated $).

    Uses a simple in-memory ledger. In production, this would be backed
    by a persistent store or the Knowledge Graph.

    Concept: policy-guardrails
    """

    name: str = "cost_budget"
    max_total_tokens: int = 500_000
    max_cost_usd: float = 10.0
    cost_per_1k_tokens: float = 0.002  # Default: GPT-4o-mini pricing

    # Internal ledger — keyed by agent_id
    _ledger: dict[str, dict[str, float]] = field(default_factory=dict)

    def _get_agent_id(self, context: dict[str, Any] | None) -> str:
        if context and "agent_id" in context:
            return context["agent_id"]
        return "__default__"

    def record_usage(self, agent_id: str, tokens: int) -> None:
        """Record token usage for an agent."""
        entry = self._ledger.setdefault(
            agent_id, {"total_tokens": 0, "total_cost": 0.0}
        )
        entry["total_tokens"] += tokens
        entry["total_cost"] += (tokens / 1000) * self.cost_per_1k_tokens

    def get_usage(self, agent_id: str) -> dict[str, float]:
        """Get current usage for an agent."""
        return self._ledger.get(agent_id, {"total_tokens": 0, "total_cost": 0.0})

    def reset(self, agent_id: str | None = None) -> None:
        """Reset usage for a specific agent or all agents."""
        if agent_id:
            self._ledger.pop(agent_id, None)
        else:
            self._ledger.clear()

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        context: dict[str, Any] | None = None,
    ) -> PolicyResult:
        agent_id = self._get_agent_id(context)

        # Estimate tokens for this request
        estimated_tokens = int(len(f"{input_text} {output_text}".split()) * 1.33)

        # Record this usage
        self.record_usage(agent_id, estimated_tokens)

        usage = self.get_usage(agent_id)
        over_tokens = usage["total_tokens"] > self.max_total_tokens
        over_cost = usage["total_cost"] > self.max_cost_usd

        allowed = not (over_tokens or over_cost)
        reasons = []
        if over_tokens:
            reasons.append(
                f"Token budget exceeded: {usage['total_tokens']:.0f}/{self.max_total_tokens}"
            )
        if over_cost:
            reasons.append(
                f"Cost budget exceeded: ${usage['total_cost']:.4f}/${self.max_cost_usd:.2f}"
            )

        return PolicyResult(
            allowed=allowed,
            policy_name=self.name,
            reason="; ".join(reasons),
            severity="block",
            metadata={
                "agent_id": agent_id,
                "total_tokens": usage["total_tokens"],
                "total_cost": usage["total_cost"],
                "estimated_request_tokens": estimated_tokens,
            },
        )


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------


@dataclass
class PolicyEngine:
    """Runs all registered policy rules and enforces results.

    Concept: policy-guardrails
    """

    rules: list[PolicyRule] = field(default_factory=list)

    def register(self, rule: PolicyRule) -> None:
        """Register a policy rule."""
        self.rules.append(rule)

    def evaluate(
        self,
        input_text: str = "",
        output_text: str = "",
        context: dict[str, Any] | None = None,
        raise_on_block: bool = False,
    ) -> list[PolicyResult]:
        """Run all rules and return results.

        Args:
            input_text: The agent input.
            output_text: The agent output.
            context: Optional context (may contain agent_id, etc.).
            raise_on_block: If True, raise PolicyViolation on blocking failures.

        Returns:
            List of PolicyResult from each rule.

        Raises:
            PolicyViolation: If raise_on_block=True and any blocking rule fails.
        """
        results: list[PolicyResult] = []
        for rule in self.rules:
            try:
                result = rule.evaluate(input_text, output_text, context)
                results.append(result)
            except Exception as exc:
                logger.warning("Policy rule %s failed: %s", rule.name, exc)
                results.append(
                    PolicyResult(
                        allowed=True,
                        policy_name=rule.name,
                        reason=f"Rule error (fail-open): {exc}",
                        severity="warn",
                    )
                )

        if raise_on_block:
            blocked = [r for r in results if not r.allowed and r.severity == "block"]
            if blocked:
                raise PolicyViolation(blocked)

        return results


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def guardrail(
    engine: PolicyEngine,
    check_input: bool = True,
    check_output: bool = True,
):
    """Decorator that wraps an async function with policy checks.

    Args:
        engine: The PolicyEngine to use for evaluation.
        check_input: If True, evaluate the first positional arg as input.
        check_output: If True, evaluate the return value as output.

    Example::

        @guardrail(engine, check_input=True, check_output=True)
        async def generate(prompt: str) -> str:
            return await llm.generate(prompt)
    """

    def decorator(func):
        import functools

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Pre-check: validate input
            if check_input and args:
                input_text = str(args[0]) if args else ""
                pre_results = engine.evaluate(
                    input_text=input_text, raise_on_block=True
                )
                logger.debug("Pre-check: %d rules passed", len(pre_results))

            # Execute the function
            result = await func(*args, **kwargs)

            # Post-check: validate output
            if check_output and result is not None:
                output_text = str(result)
                input_text = str(args[0]) if args else ""
                engine.evaluate(
                    input_text=input_text,
                    output_text=output_text,
                    raise_on_block=True,
                )

            return result

        return wrapper

    return decorator
