"""Run one L3 node (an ``AgentSpec``) through its selected harness.

The parallel engine calls :func:`run_agent_spec` for every node whose spec
names a non-native harness. The node's outcome is recorded at L5 and returned
in the engine's own ``AgentExecutionResult`` shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_utilities.layers.harness_port import HarnessRequest, RunOutcome
from agent_utilities.layers.harness_record import run_and_record
from agent_utilities.layers.harness_registry import HarnessRegistry, harness_registry
from agent_utilities.models.execution_manifest import AgentExecutionResult, AgentSpec


def request_for_spec(spec: AgentSpec, task: str, timeout_s: float) -> HarnessRequest:
    """The harness request one node spec describes."""
    workspace = Path(spec.harness_workspace) if spec.harness_workspace else None
    return HarnessRequest(
        agent_name=spec.agent_id,
        task=task,
        workspace=workspace,
        timeout_s=timeout_s,
        model=spec.model_id or None,
        response_format="json" if spec.output_schema else "text",
        allowed_tools=tuple(spec.tools),
    )


def node_result(spec: AgentSpec, outcome: RunOutcome) -> AgentExecutionResult:
    """A harness outcome in the parallel engine's result shape."""
    usage = outcome.usage
    return AgentExecutionResult(
        agent_id=spec.agent_id,
        role=spec.role,
        partition=spec.partitions[0] if spec.partitions else "",
        output=outcome.final_text,
        success=outcome.succeeded,
        error=outcome.error or "",
        duration_ms=outcome.duration_ms,
        model_id=outcome.model,
        token_usage={} if usage is None else usage.token_usage(),
        metadata={"run_outcome": outcome.model_dump(mode="json")},
    )


async def run_agent_spec(
    spec: AgentSpec,
    task: str,
    *,
    timeout_s: float,
    engine: Any = None,
    registry: HarnessRegistry | None = None,
) -> AgentExecutionResult:
    """Select the node's harness, run it, record L5 and return the result."""
    port = (registry or harness_registry()).select(spec)
    request = request_for_spec(spec, task, timeout_s)
    outcome = await run_and_record(port, request, engine=engine)
    return node_result(spec, outcome)


__all__ = ["node_result", "request_for_spec", "run_agent_spec"]
