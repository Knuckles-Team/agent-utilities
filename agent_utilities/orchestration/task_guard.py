"""Engine-independent screening and persistence redaction for agent tasks."""

from __future__ import annotations

from agent_utilities.security.threat_defense_engine import (
    PromptInjectionScanner,
    ScanResult,
)


class TaskInjectionRejected(ValueError):
    """The task text did not pass AU's prompt-injection policy."""


def scan_agent_task(
    task: str, *, scanner: PromptInjectionScanner | None = None
) -> ScanResult:
    """Return AU's canonical prompt-injection verdict for one task."""
    active_scanner = scanner or PromptInjectionScanner()
    return active_scanner.scan_text(task)


def redact_agent_task(task: str) -> str:
    """Return the canonical PII-redacted form written to durable state."""
    from agent_utilities.capabilities.content_guardrails import _pii_guard

    verdict = _pii_guard(task)
    replacement = getattr(verdict, "replacement", None)
    return replacement if isinstance(replacement, str) else task


def screen_and_redact_agent_task(task: str) -> str:
    """Reject injection before redacting PII from task text for persistence."""
    verdict = scan_agent_task(task)
    if verdict.is_malicious:
        raise TaskInjectionRejected("task rejected by AU input screening policy")
    return redact_agent_task(task)
