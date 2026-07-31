#!/usr/bin/python
from __future__ import annotations

"""Default-ON content guardrails: PII redaction, secret-leak redaction, output-schema.

CONCEPT:AU-OS.safety.harness-guardrails-adoption — adopts ``pydantic-ai-harness``'s
``InputGuardrail``/``OutputGuardrail`` capabilities (shipped in harness 0.7.0, already
an installed dependency here at 0.14.0) as the live replacement for three of the four
rule types the dead ``agent_utilities.security.guardrails.PolicyEngine`` implemented
(D-48): PII redaction, forbidden-content blocking, and output-schema validation.
``PolicyEngine`` had **zero live callers** anywhere in the ``agent-packages`` ecosystem
— only test and docstring instantiations — so it read as implemented safety while
never actually running. ``InputGuardrail``/``OutputGuardrail`` use the same
``AbstractCapability`` hook contract every other capability in this package builds on
(``output_repair.StructuredOutputRepair``, ``stuck_loop.StuckLoopDetection``, …), so
these guards run through the SAME live agent path every other reliability capability
does.

``PromptInjectionPolicy`` is migrated too, as :func:`prompt_injection_guardrail`
(G8): it was only a ``PolicyEngine``-shaped adapter around
``PromptInjectionScanner``, which survives untouched and is still enforced
directly on ``Orchestrator._scan_task`` -- the chokepoint every
``dispatch_task``/``execute_agent``/``execute_workflow`` call passes through.
The adapter's deletion therefore removed a redundant wrapper, not the defence;
attaching the scanner here additionally closes the real gap, which was that the
AGENT path had no injection gate of its own.

The fifth rule type, ``CostBudgetPolicy``, has no port here: ``models/usage.py``'s
``ExecutionBudget`` (``max_total_tokens``/``max_cost_usd``/``max_duration_seconds``/
``max_tool_calls``) already covers the same ground and — unlike ``CostBudgetPolicy`` —
is actually enforced on the live dispatch path
(``graph/_router_impl.py::dispatcher_step``), so porting it would have reintroduced a
second, redundant cost-budget notion.

Wired as default-ON capabilities (``capabilities/composition.py``), like every other
reliability capability here: the PII guard only redacts when a PII pattern actually
matches, the secret-leak guard only redacts when a credential-shaped token is present
(D-OP-2: it used to ``block`` the whole output, which discarded legitimate answers
containing e.g. a k8s manifest's ``stringData`` block — see :func:`_secret_leak_guard`),
and the output-schema guard is a pure no-op when the caller passes no required keys —
so a normal run carrying none of that content is unaffected.
"""

from collections.abc import Sequence
from typing import Any

from pydantic_ai_harness import GuardrailResult, InputGuardrail, OutputGuardrail

from agent_utilities.http.redaction import contains_secret, redact_text
from agent_utilities.security.guardrails import PiiSanitizer
from agent_utilities.security.threat_defense_engine import PromptInjectionScanner

#: Refusal prefix for the prompt-injection input guardrail (G8).
PROMPT_INJECTION_BLOCK_PREFIX = "Input blocked: prompt-injection attempt detected."

#: One scanner instance: ``PromptInjectionScanner.__init__`` compiles its whole
#: pattern set, so rebuilding it per prompt would recompile on every agent turn.
_INJECTION_SCANNER = PromptInjectionScanner()


def _prompt_injection_guard(value: object) -> GuardrailResult:
    """Block a prompt carrying a detected injection/jailbreak attempt (G8).

    This is the genuine MIGRATION of the deleted
    ``threat_defense_engine.PromptInjectionPolicy`` onto the harness Guardrails
    mechanism — NOT a dropped capability. That class was only a
    ``PolicyEngine``-shaped adapter around :class:`PromptInjectionScanner`, and
    the scanner itself is untouched and still enforced directly on the
    orchestrator's own chokepoint (``Orchestrator._scan_task``, called from
    ``dispatch_task``/``execute_agent``/``execute_workflow``/
    ``execute_dynamic_workflow``). Deleting the adapter removed a second wrapper,
    never the defence. What was genuinely missing is this: the AGENT path had no
    injection gate of its own, so a prompt reaching an agent by any route other
    than the orchestrator was unscanned. Attaching it here closes that gap with
    the same scanner and the same verdict.
    """
    if not isinstance(value, str):
        return GuardrailResult.allow()
    result = _INJECTION_SCANNER.scan_text(value)
    if result.is_malicious:
        return GuardrailResult.block(
            f"{PROMPT_INJECTION_BLOCK_PREFIX} {result.explanation}"
        )
    return GuardrailResult.allow()


def prompt_injection_guardrail() -> InputGuardrail[Any]:
    """Build the default-ON prompt-injection input guardrail (G8)."""
    return InputGuardrail(guard=_prompt_injection_guard)


def _pii_guard(value: object) -> GuardrailResult:
    """Redact PII (SSN / tax-id / credit-card / email) rather than block it.

    A no-op (``allow``) for non-string values (typed/structured output) and for
    text with no PII match. ``InputGuardrail`` and ``OutputGuardrail`` share this
    guard: an ``InputGuardrail`` sees only ever ``str`` (the user prompt);
    ``OutputGuardrail`` may see a non-string typed output, which this guard passes
    through unchanged (the output-schema guardrail below governs typed output
    shape; PII redaction only applies to free text).
    """
    if not isinstance(value, str):
        return GuardrailResult.allow()
    sanitized = PiiSanitizer().sanitize_text(value)
    if sanitized == value:
        return GuardrailResult.allow()
    return GuardrailResult.replace(sanitized)


def _secret_leak_guard(value: object) -> GuardrailResult:
    """Redact secret-shaped tokens (API key, DSN, PAT, ...) rather than block.

    D-OP-2: this guard used to ``block`` the entire output outright on a
    ``contains_secret`` hit. That made it a **true positive on legitimate
    content** this workspace routinely asks agents to handle — a k8s
    ``ExternalSecret``/manifest snippet such as ``"stringData:\n  secret:
    hunter2"`` trips ``contains_secret`` and discarded the whole response, not
    just the credential. Mirroring ``_pii_guard``, this now reuses
    :func:`redact_text` (the same detector set as ``contains_secret``) so the
    credential never reaches the caller while the surrounding answer survives.
    A no-op (``allow``) for non-string values and for text with no secret
    match.
    """
    if not isinstance(value, str):
        return GuardrailResult.allow()
    if not contains_secret(value):
        return GuardrailResult.allow()
    return GuardrailResult.replace(redact_text(value))


def pii_redaction_guardrails() -> list[InputGuardrail[Any] | OutputGuardrail[Any]]:
    """Build the default-ON PII-redaction guardrail pair (input + output)."""
    return [InputGuardrail(guard=_pii_guard), OutputGuardrail(guard=_pii_guard)]


def secret_leak_guardrail() -> OutputGuardrail[Any]:
    """Build the default-ON forbidden-content (secret-leak) output guardrail."""
    return OutputGuardrail(guard=_secret_leak_guard)


def output_schema_guardrail(required_keys: Sequence[str]) -> OutputGuardrail[Any]:
    """Build an output-schema guardrail blocking output missing required keys.

    A pure no-op when ``required_keys`` is empty — safe to attach unconditionally.
    Handles the three output shapes ``PolicyEngine``'s ``OutputSchemaPolicy`` covered:
    a Pydantic ``BaseModel`` (``model_dump()`` normalizes it to a dict first), a plain
    ``dict``, and a JSON-text ``str`` output (parsed before the key check; non-JSON
    text blocks outright when keys are required). Any other output type is passed
    through unchanged, since a required-keys contract does not apply to it.
    """
    keys = tuple(required_keys)

    def _schema_guard(value: object) -> GuardrailResult:
        if not keys:
            return GuardrailResult.allow()
        data: object = value
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        elif isinstance(data, str):
            import json

            try:
                data = json.loads(data)
            except (TypeError, ValueError):
                return GuardrailResult.block(
                    "Output is not valid JSON but schema validation was required"
                )
        if not isinstance(data, dict):
            return GuardrailResult.allow()
        missing = [key for key in keys if key not in data]
        if missing:
            return GuardrailResult.block(f"Output missing required keys: {missing}")
        return GuardrailResult.allow()

    return OutputGuardrail(guard=_schema_guard)
