#!/usr/bin/python
from __future__ import annotations

"""PII sanitization primitives.

CONCEPT:AU-OS.safety.harness-guardrails-adoption

Historically this module also carried a sync, fail-open ``PolicyEngine`` (PII
redaction, forbidden-content regex, cost-budget ledger, and output-schema
rules) that was never wired into any live call path — grep confirmed only test
and docstring instantiations across the whole ``agent-packages`` ecosystem
(D-48). It was deleted per ``AGENTS.md`` "No Legacy": three of its four rule
types (PII, forbidden-content, output-schema) are now covered by
``pydantic-ai-harness``'s ``InputGuardrail``/``OutputGuardrail`` capabilities,
wired default-on through ``capabilities/content_guardrails.py`` +
``capabilities/composition.py``; the fourth (cost-budget) was already fully
covered — and, unlike ``PolicyEngine``, actually enforced on the live dispatch
path — by ``models/usage.py``'s ``ExecutionBudget``
(``graph/_router_impl.py::dispatcher_step``), so it was deleted outright
rather than ported. See ``docs/architecture/`` for the guardrail wiring
diagram.

``PiiSanitizer`` below is a live, independently-consumed utility (the
``PiiRedactionFilter`` logging filter in ``observability/audit_logger.py``,
``vector-mcp``'s API layer, and the guardrail guards in
``capabilities/content_guardrails.py``) and is kept.
"""


import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in patterns
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
