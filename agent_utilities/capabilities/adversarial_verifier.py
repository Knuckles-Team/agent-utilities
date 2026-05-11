#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AHE-3.1 — Adversarial Verification (Opt-In).

Provides an optional adversarial verification pass that runs alongside
the standard quality gate in :func:`verifier_step`.  When enabled via
``ADVERSARIAL_VERIFICATION=true``, a "Hacker Agent" attempts to break
the implementation by probing for:
- Boundary value failures
- Concurrency issues
- Idempotency violations
- Orphan reference errors

The adversarial pass runs AFTER the quality gate passes.  If the
hacker agent finds a flaw, the verification fails with specific
feedback, triggering a re-dispatch.

Integrates with:
    - CONCEPT:ORCH-1.2 (Signal Board): Adversarial findings are emitted as
      ``security_concern`` or ``quality_gap`` signals.
    - Verifier prompt: Extends the existing adversarial probes section.
    - ``GraphState``: Uses standard feedback/re-dispatch flow.

See docs/pillars/3_agentic_harness_engineering.md §CONCEPT:AHE-3.1
"""


import asyncio
import logging
import os
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..graph.state import GraphDeps, GraphState

logger = logging.getLogger(__name__)

# Default: disabled.  Set ADVERSARIAL_VERIFICATION=true to enable.
ADVERSARIAL_ENABLED = os.getenv("ADVERSARIAL_VERIFICATION", "false").lower() in (
    "true",
    "1",
    "yes",
)


class AdversarialResult(BaseModel):
    """Structured output from the adversarial verification agent."""

    vulnerabilities_found: bool = False
    severity: str = Field(
        default="none",
        description="Severity level: none, low, medium, high, critical",
    )
    findings: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)
    confidence: float = Field(
        default=0.0,
        description="Confidence in the findings (0.0 to 1.0)",
    )


async def run_adversarial_pass(
    state: GraphState,
    deps: GraphDeps,
    results_summary: str,
) -> AdversarialResult | None:
    """Run an adversarial verification pass against execution results.

    This function spawns a "Hacker Agent" that attempts to find
    weaknesses in the implementation.  It only runs when:
    1. ``ADVERSARIAL_VERIFICATION=true`` (config flag)
    2. The standard quality gate has already passed
    3. Code was generated (mode == "execute" or code artifacts exist)

    Args:
        state: The current ``GraphState`` with execution results.
        deps: The ``GraphDeps`` with model and event queue.
        results_summary: Consolidated results from execution steps.

    Returns:
        An ``AdversarialResult`` if findings exist, or ``None`` if clean.
    """
    if not ADVERSARIAL_ENABLED:
        return None

    # Only adversarial-verify when there's code to attack
    has_code = any(
        keyword in results_summary.lower()
        for keyword in ("def ", "class ", "function ", "import ", "```python", "```ts")
    )
    if not has_code and state.mode != "execute":
        logger.debug("Adversarial pass skipped: no code artifacts detected")
        return None

    try:
        from pydantic_ai import Agent

        adversarial_agent = Agent(
            model=deps.agent_model,
            output_type=AdversarialResult,
            system_prompt=(
                "You are a security-focused adversarial reviewer. Your job is NOT "
                "to confirm the implementation works — it's to TRY TO BREAK IT.\n\n"
                "## ADVERSARIAL PROBES\n"
                "1. **Boundary values**: 0, -1, empty string, very long strings, "
                "Unicode, None, MAX_INT\n"
                "2. **Concurrency**: What happens with parallel requests?\n"
                "3. **Idempotency**: Same mutating operation twice — safe?\n"
                "4. **Orphan references**: IDs that don't exist, deleted entities\n"
                "5. **Injection**: SQL injection, command injection, path traversal\n"
                "6. **Resource exhaustion**: Unbounded loops, missing pagination\n"
                "7. **Error handling**: What fails silently? What leaks stack traces?\n\n"
                "## RULES\n"
                "- Only report REAL vulnerabilities you can reason about from the code\n"
                "- Do NOT fabricate issues — false positives waste developer time\n"
                "- Rate severity honestly: none/low/medium/high/critical\n"
                "- Include specific code references and suggested fixes\n\n"
                f"## CONTEXT\n"
                f"Original Query: {state.query}\n\n"
                f"## IMPLEMENTATION TO ATTACK\n{results_summary}"
            ),
        )

        result = await asyncio.wait_for(
            adversarial_agent.run("Break this implementation."),
            timeout=deps.verifier_timeout,
        )

        adversarial = result.output
        if not isinstance(adversarial, AdversarialResult):
            return None

        if adversarial.vulnerabilities_found:
            logger.warning(
                "[CONCEPT:AHE-3.1] Adversarial verification found %d issue(s) "
                "(severity: %s, confidence: %.0f%%)",
                len(adversarial.findings),
                adversarial.severity,
                adversarial.confidence * 100,
            )

            # CONCEPT:ORCH-1.0 — Emit findings to the signal board
            for finding in adversarial.findings[:5]:
                signal_type = (
                    "security_concern"
                    if adversarial.severity in ("high", "critical")
                    else "quality_gap"
                )
                if signal_type not in state.signal_board:
                    state.signal_board[signal_type] = []
                state.signal_board[signal_type].append(f"[adversarial] {finding[:200]}")

            # Emit event for UI transparency
            from ..graph.config_helpers import emit_graph_event

            emit_graph_event(
                deps.event_queue,
                "adversarial_result",
                vulnerabilities_found=True,
                severity=adversarial.severity,
                finding_count=len(adversarial.findings),
                confidence=adversarial.confidence,
            )

            return adversarial

        logger.info(
            "[CONCEPT:AHE-3.1] Adversarial verification passed — no issues found"
        )
        return None

    except Exception as e:
        logger.warning(f"[CONCEPT:AHE-3.1] Adversarial pass failed (non-fatal): {e}")
        return None
