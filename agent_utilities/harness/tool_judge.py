#!/usr/bin/python
from __future__ import annotations

"""Agentic tool-judge over large traces (CONCEPT:AHE-3.66).

A multi-MB agent trace blows the judge's context window if stuffed in whole. Absorbed
from Opik's agentic judge: instead of inlining the trace, give the judge TOOLS to navigate
the span subgraph on demand — list spans, drill into one, read the final I/O — and let it
investigate before ruling. Built on the same model the rest of eval uses (vLLM/OpenAI).

The judge is selected automatically: small traces use the cheap inline judge; only traces
over a size threshold pay for the tool loop (``should_use``).
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Serialized-trace size (chars) above which the tool-judge is worth its tool loop.
TOOL_JUDGE_THRESHOLD = 8000
#: ...or this many spans+generations — a complex run is better navigated than inlined.
TOOL_JUDGE_MAX_SPANS = 12


def should_use(
    entry: dict[str, Any],
    threshold: int = TOOL_JUDGE_THRESHOLD,
    max_spans: int = TOOL_JUDGE_MAX_SPANS,
) -> bool:
    """True when a trace is complex enough to warrant navigating it with tools — either
    many spans/generations (a deep agent run) or a large serialized payload."""
    n = len(entry.get("spans", [])) + len(entry.get("generations", []))
    if n > max_spans:
        return True
    try:
        size = len(json.dumps(_serialize(entry), default=str))
    except Exception:
        size = 0
    return size > threshold


def _serialize(entry: dict[str, Any]) -> dict[str, Any]:
    trace = entry.get("trace")
    return {
        "input": getattr(trace, "input", ""),
        "output": getattr(trace, "output", ""),
        "spans": [getattr(s, "name", "") for s in entry.get("spans", [])],
        "generations": [getattr(g, "name", "") for g in entry.get("generations", [])],
    }


class ToolEnabledJudge:
    """Judge a trace against a criteria by letting the model navigate it via tools."""

    def judge(self, entry: dict[str, Any], criteria: str) -> tuple[float, str]:
        """Return ``(1.0|0.0, reasoning)``. Falls back to (0.0, reason) with no model."""
        spans = entry.get("spans", [])
        generations = entry.get("generations", [])
        trace = entry.get("trace")

        # ── tools the judge can call to navigate the span subgraph on demand ──
        def list_spans() -> list[str]:
            """List the names of all spans and generations in the trace."""
            return [getattr(s, "name", "") for s in spans] + [
                getattr(g, "name", "") for g in generations
            ]

        def span_detail(name: str) -> str:
            """Get the details (kind, latency, error, model, tokens) of one span by name."""
            for s in spans:
                if getattr(s, "name", "") == name:
                    return (
                        f"span kind={getattr(s, 'span_kind', '?')} "
                        f"latency_ms={getattr(s, 'latency_ms', None)} "
                        f"error={getattr(s, 'error', None)}"
                    )
            for g in generations:
                if getattr(g, "name", "") == name:
                    return (
                        f"generation model={getattr(g, 'model', '?')} "
                        f"in={getattr(g, 'input_tokens', 0)} "
                        f"out={getattr(g, 'output_tokens', 0)} "
                        f"error={getattr(g, 'error', None)}"
                    )
            return f"no span named {name!r}"

        def trace_io() -> str:
            """Get the trace's top-level input and final output text."""
            return (
                f"INPUT: {getattr(trace, 'input', '')}\n"
                f"OUTPUT: {getattr(trace, 'output', '')}"
            )

        try:
            from pydantic_ai import Agent

            from agent_utilities.core.model_factory import create_model

            agent = Agent(
                create_model(),
                tools=[list_spans, span_detail, trace_io],
                system_prompt=(
                    "You are a trace evaluator. Use the tools to investigate the trace "
                    "(list_spans, then span_detail / trace_io as needed). When you have "
                    "enough evidence, judge it. Respond with a single JSON line: "
                    '{"pass": <true|false>, "reasoning": "<brief>"}'
                ),
            )
            result = agent.run_sync(
                f"Does this trace satisfy the criteria: {criteria}?"
            )
            text = (result.output if hasattr(result, "output") else str(result)).strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            passed = bool(parsed.get("pass", False))
            return (1.0 if passed else 0.0, str(parsed.get("reasoning", "")))
        except Exception as exc:  # pragma: no cover - model optional / parse failure
            logger.debug("tool-judge unavailable/failed: %s", exc)
            return (0.0, f"tool-judge unavailable: {exc}")


__all__ = ["ToolEnabledJudge", "should_use", "TOOL_JUDGE_THRESHOLD"]
