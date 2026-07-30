#!/usr/bin/python
from __future__ import annotations

"""Structured-output failure taxonomy + bounded repair.

CONCEPT:AU-AHE.harness.structured-output-repair — a delegated ``pydantic_graph``
run against ServiceNow exceeded its token budget at 6,982 tokens *after* a
successful tool call, and (separately) a fleet tool silently ignored an unknown
``limit`` argument and returned 212 KB from one call. Neither failure was a
malformed-output problem, but both exposed the same gap one layer up: when a
model DOES produce bad structured output (truncated by a budget, invalid JSON,
wrong shape, an outright refusal), pydantic-ai's own retry machinery treats every
cause identically — one undifferentiated ``ValidationError``/``ModelRetry``
bounce — and a caller cannot tell *why* it failed or *that a repair was even
attempted* once the run finally succeeds or gives up.

This module classifies every structured-output failure into an explicit
taxonomy, drives a **bounded** classify-then-repair loop (re-ask the model with
a targeted, classification-specific instruction), and — critically — makes the
outcome **visible**:

* every attempt (classification + what repair was tried) is appended to
  :attr:`StructuredOutputRepair.attempts` and, on success, stamped onto the
  ``AgentRunResult`` as ``output_repair_attempts`` (and, on the failure path,
  onto the propagated exception) so a caller holding the result or the error can
  never read a repaired run back as an indistinguishable clean first-try success
  (CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency's truthfulness
  contract, extended to output repair). SCOPE: this is visibility to the direct
  caller only -- ``orchestration/agent_runner.py::_record_execution_trace``,
  which writes the RunTrace/KG/OTel record, does not yet read
  ``output_repair_attempts``, so a repaired run is still indistinguishable from
  a clean one in the persisted trace. Recorded as D-W15-13 in
  ``reports/deferred/waves1-5-gate.md``;
* exhausting the bound raises :class:`StructuredOutputRepairExhausted` — a
  plain (non-``ModelRetry``) exception carrying the full attempt list — so the
  run fails closed with a typed, inspectable error instead of either an
  unbounded retry storm or pydantic-ai's own generic
  ``UnexpectedModelBehavior: Exceeded maximum output retries``;
* a budget exhausted *mid-output* (``UsageLimitExceeded``) or an upstream
  content-filter refusal (``ContentFilterError``) is recorded and NEVER
  retried — repairing a budget violation would only spend more of the budget
  that already tripped, exactly the anti-pattern
  ``graph/verification.py::error_recovery_step`` was hardened against
  (CONCEPT:AU-ORCH.execution.execution-budget-caps).

Wired as a default-ON capability (``capabilities/composition.py``), like every
other reliability capability here it only acts on its own trigger: it is a
pure no-op for the many agents built with a plain-text ``output_type`` (the
underlying hooks do not fire for unstructured output — see
``before_output_validate``'s own docs), and costs nothing for a structured
agent whose first response already validates.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pydantic import ValidationError
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ContentFilterError, ModelRetry, UsageLimitExceeded

logger = logging.getLogger(__name__)

#: The minimum viable structured-output failure taxonomy (task requirement).
OutputFailureClass = Literal[
    "malformed_json",
    "schema_invalid",
    "truncated",
    "refused",
    "empty",
    "wrong_type",
    "budget_exceeded_mid_output",
]

#: Default bounded-repair budget: how many targeted re-asks are attempted
#: before failing closed. Also threaded as the underlying Agent's
#: ``retries={"output": ...}`` budget (``output_repair_retries``) so
#: pydantic-ai's own generic retry-exhaustion exception never preempts this
#: module's classified, attempt-recording exhaustion path.
DEFAULT_MAX_OUTPUT_REPAIRS = 2

_PREVIEW_CHARS = 300

_REFUSAL_PATTERN = re.compile(
    r"^\s*(i'?m sorry|i cannot\b|i can'?t\b|i won'?t\b|i am unable|i'?m unable|"
    r"as an ai\b|i'?m not able)",
    re.IGNORECASE,
)

#: pydantic ``ValidationError`` error ``type`` codes that indicate the model
#: produced schema-*shaped* output with a value of the wrong Python type
#: (coercion failures) rather than a structurally invalid document (a missing
#: required field, an unrecognized extra field, a bad literal/enum, ...).
_WRONG_TYPE_ERROR_TYPES = frozenset(
    {
        "int_parsing",
        "int_type",
        "float_parsing",
        "float_type",
        "bool_parsing",
        "bool_type",
        "string_type",
        "bytes_type",
        "dict_type",
        "list_type",
    }
)


def _safe_preview(output: Any) -> str:
    """A short, log/trace-safe rendering of a raw or erroring output value."""
    try:
        text = output if isinstance(output, str) else str(output)
    except Exception:
        # A broken __str__ must not break classification.
        return "<unrepresentable>"
    text = text.strip()
    if len(text) > _PREVIEW_CHARS:
        return text[:_PREVIEW_CHARS] + "…"
    return text


def _looks_truncated(text: str) -> bool:
    """Heuristic: ``text`` looks like JSON that was cut off mid-token.

    Balances brace/bracket/quote nesting rather than re-parsing (the caller
    already knows ``json.loads`` failed): genuinely malformed JSON usually
    still *closes* its structure (a bad comma, an unquoted key), while a
    truncation ends with unbalanced open delimiters or inside an unterminated
    string.
    """
    depth = 0
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
    return in_string or depth > 0


def classify_raw_output(output: Any) -> OutputFailureClass | None:
    """Classify a pre-validation raw structured-output value. ``None`` = looks fine.

    Runs in :meth:`StructuredOutputRepair.before_output_validate`, before
    pydantic-ai attempts to parse/validate ``output`` against the schema —
    exactly where malformed JSON, truncation, an empty response, and an
    outright textual refusal are all still visible as raw text (they would
    otherwise surface, if at all, as an undifferentiated parse/validation
    error one step later).
    """
    if output is None:
        return "empty"
    if isinstance(output, dict):
        return "empty" if not output else None
    if not isinstance(output, str):
        return None
    text = output.strip()
    if not text:
        return "empty"
    if _REFUSAL_PATTERN.match(text):
        return "refused"
    # This hook only fires for structured output that requires parsing (see
    # ``StructuredOutputRepair.before_output_validate``), so any text reaching
    # here is expected to be JSON — a leading ``{``/``[`` is not required to
    # treat a parse failure as malformed/truncated.
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return "truncated" if _looks_truncated(text) else "malformed_json"
    return None


def classify_validation_error(error: ValidationError) -> OutputFailureClass:
    """Classify a post-parse pydantic ``ValidationError`` from schema validation.

    ``wrong_type`` when every reported error is a type-coercion failure (the
    document's *shape* was right but a value's type was not); ``schema_invalid``
    for anything structural (missing/extra fields, bad literals, nested model
    mismatches, ...) or when the error list is empty/unavailable.
    """
    try:
        errors = error.errors()
    except Exception:
        # A malformed ValidationError must not crash classification.
        return "schema_invalid"
    if errors and all(e.get("type") in _WRONG_TYPE_ERROR_TYPES for e in errors):
        return "wrong_type"
    return "schema_invalid"


def _summarize_validation_error(error: ValidationError) -> str:
    try:
        return "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', '')}"
            for e in error.errors()[:5]
        )
    except Exception:
        # Fall back to the plain string form.
        return _safe_preview(error)


_REPAIR_MESSAGES: dict[OutputFailureClass, str] = {
    "empty": "Your previous response was empty. Return the complete structured output.",
    "refused": (
        "Do not explain, apologize, or refuse — return ONLY the requested "
        "structured output."
    ),
    "malformed_json": "Your previous response was not valid JSON ({detail}). Return ONLY valid JSON matching the schema.",
    "truncated": (
        "Your previous response was cut off before it finished. Return the "
        "COMPLETE output in a single response."
    ),
    "schema_invalid": "Your previous response did not match the required schema ({detail}). Return the corrected structured output.",
    "wrong_type": "One or more fields had the wrong type ({detail}). Return the corrected structured output with the right types.",
}


def _repair_message(classification: OutputFailureClass, detail: str) -> str:
    template = _REPAIR_MESSAGES.get(
        classification, "Your previous output was invalid ({detail}). Try again."
    )
    return template.format(detail=detail) if "{detail}" in template else template


def output_repair_retries(
    *, structured_output_repair: bool, max_output_repairs: int
) -> dict[str, int] | None:
    """The Agent-level ``retries`` override needed for bounded repair to be authoritative.

    pydantic-ai's own output-validation retry budget defaults to 1
    (``AgentRetries``/``Agent(retries=...)``). :class:`StructuredOutputRepair`
    raises ``ModelRetry`` up to ``max_output_repairs`` times before switching to
    its own typed :class:`StructuredOutputRepairExhausted`; if the underlying
    Agent's own budget were smaller, pydantic-ai would raise its generic
    ``UnexpectedModelBehavior: Exceeded maximum output retries`` first, on a
    retry this module itself issued, pre-empting the classified/attempt-recorded
    exhaustion path entirely. Returns ``None`` (no override) when the capability
    is disabled, so an explicit caller-supplied ``retries=`` is never overwritten
    with an unrelated budget.
    """
    if not structured_output_repair:
        return None
    return {"output": int(max_output_repairs)}


@dataclass(frozen=True)
class RepairAttempt:
    """One classify-then-repair (or terminal) decision, in trace-ready shape."""

    classification: OutputFailureClass
    attempt: int
    action: Literal["retried", "exhausted", "propagated"]
    detail: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "attempt": self.attempt,
            "action": self.action,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }


class StructuredOutputRepairExhausted(RuntimeError):
    """Fail-closed terminal error: bounded structured-output repair did not converge.

    Deliberately NOT a :class:`~pydantic_ai.exceptions.ModelRetry` — raising one
    of those here would let pydantic-ai's own retry machinery swallow and retry
    it again, defeating the bound. Carries the full ``attempts`` list (every
    classification and repair message tried) so the caller's provenance/trace
    records exactly what happened instead of a bare "the model misbehaved".
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: list[RepairAttempt],
        last_error: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = list(attempts)
        self.last_error = last_error
        if last_error is not None:
            self.__cause__ = last_error


@dataclass
class StructuredOutputRepair(AbstractCapability[Any]):
    """Classify → bounded targeted repair → fail closed, for structured agent output.

    Attach once (default-ON via ``capabilities/composition.py``); ``for_run``
    returns a fresh per-run instance so concurrent runs never share attempt
    state, mirroring :class:`~agent_utilities.capabilities.stuck_loop.StuckLoopDetection`.
    """

    max_repairs: int = DEFAULT_MAX_OUTPUT_REPAIRS
    attempts: list[RepairAttempt] = field(default_factory=list)

    async def for_run(self, ctx: RunContext[Any]) -> StructuredOutputRepair:
        del ctx
        return replace(self, attempts=[])

    def _record(
        self,
        classification: OutputFailureClass,
        action: Literal["retried", "exhausted", "propagated"],
        detail: str,
    ) -> RepairAttempt:
        attempt = RepairAttempt(
            classification=classification,
            attempt=len(self.attempts) + 1,
            action=action,
            detail=detail,
        )
        self.attempts.append(attempt)
        return attempt

    def _decide(
        self,
        classification: OutputFailureClass,
        detail: str,
        *,
        cause: BaseException | None,
    ) -> None:
        """Record this failure and either raise a targeted ``ModelRetry`` (bounded)
        or fail closed with :class:`StructuredOutputRepairExhausted`."""
        if len(self.attempts) < self.max_repairs:
            self._record(classification, "retried", detail)
            raise ModelRetry(_repair_message(classification, detail))
        attempt = self._record(classification, "exhausted", detail)
        raise StructuredOutputRepairExhausted(
            f"structured output repair exhausted after {attempt.attempt} attempt(s) "
            f"(last classification={classification!r}): {detail}",
            attempts=self.attempts,
            last_error=cause,
        )

    async def before_output_validate(
        self, ctx: RunContext[Any], *, output_context: Any, output: Any
    ) -> Any:
        del output_context
        if ctx.partial_output:
            return output
        classification = classify_raw_output(output)
        if classification is None:
            return output
        self._decide(classification, _safe_preview(output), cause=None)
        return output  # pragma: no cover - _decide always raises

    async def on_output_validate_error(
        self,
        ctx: RunContext[Any],
        *,
        output_context: Any,
        output: Any,
        error: ValidationError | ModelRetry,
    ) -> Any:
        del ctx, output_context, output
        if isinstance(error, ModelRetry):
            # A downstream (application) output_validator rejected the value for
            # business-logic reasons — schema-shaped but semantically invalid.
            classification: OutputFailureClass = "schema_invalid"
            detail = str(error)
        else:
            classification = classify_validation_error(error)
            detail = _summarize_validation_error(error)
        self._decide(classification, detail, cause=error)

    async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> Any:
        del ctx
        # A budget tripped or the provider itself refused mid-generation. Both
        # are recorded for provenance but NEVER retried here: repairing a
        # budget violation would only spend more of the very budget that
        # already tripped (the same anti-pattern
        # graph/verification.py::error_recovery_step guards against for the
        # multi-node graph budget).
        if isinstance(error, UsageLimitExceeded):
            self._record("budget_exceeded_mid_output", "propagated", str(error))
        elif isinstance(error, ContentFilterError):
            self._record("refused", "propagated", str(error))
        if self.attempts:
            # ``for_run`` hands every run its OWN instance, and this path raises
            # rather than returning an ``AgentRunResult`` -- so an attempt list
            # kept only on ``self`` is dropped the instant this returns and is
            # provably unrecoverable by any caller. Attach it to the exception,
            # which is the one object that does survive out to the caller, so
            # the recording is real provenance and not a write to nowhere.
            try:
                error.output_repair_attempts = [  # type: ignore[attr-defined]
                    a.to_dict() for a in self.attempts
                ]
            except AttributeError:  # noqa: BLE001 - an exception type with
                # __slots__ (or a read-only attribute) cannot carry provenance;
                # the ORIGINAL error must still propagate untouched, so this is
                # a genuine no-op rather than a swallowed failure.
                logger.debug(
                    "output-repair provenance could not be attached to %s "
                    "(immutable exception type); the error itself is unchanged",
                    type(error).__name__,
                )
        raise error

    async def after_run(self, ctx: RunContext[Any], *, result: Any) -> Any:
        del ctx
        if self.attempts:
            # Never let a repaired run look like a clean first-try success --
            # stamp the result so every caller/provenance reader sees it
            # happened, even when the repair itself succeeded.
            result.output_repair_attempts = [a.to_dict() for a in self.attempts]
        return result


__all__ = [
    "DEFAULT_MAX_OUTPUT_REPAIRS",
    "OutputFailureClass",
    "RepairAttempt",
    "StructuredOutputRepair",
    "StructuredOutputRepairExhausted",
    "classify_raw_output",
    "classify_validation_error",
    "output_repair_retries",
]
