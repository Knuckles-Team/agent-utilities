"""``AgentExecutionPort`` through the L4 :class:`HarnessPort` (RF-ADR-010 §6).

Agent execution has one path: an :class:`AgentExecutionRequest` becomes a
digest-bound :class:`RunSpec`, the selected harness negotiates it and the run
goes through :func:`~agent_utilities.layers.execution.run_to_completion`. The
default harness is the in-process AU runtime (``pydantic-ai``), so existing
callers keep the runtime's behaviour; the verified session is bound for the
whole run, so every graph read/write is attributed to and authorized as the
caller.

Failure mapping keeps the control plane's error contract: a harness that is
refused, absent or unconfigured is :class:`AgentControlPlaneUnavailable`; a
run that failed inside the AU runtime re-raises the runtime's own exception
(``ScopeError``, ``LookupError``, ...); anything else raises its typed harness
error.
"""

from __future__ import annotations

import uuid
from typing import Any

from agent_utilities.api.agent_control_contracts import (
    AgentControlPlaneUnavailable,
    AgentExecutionRequest,
    AgentExecutionResult,
)
from agent_utilities.api.session import GraphSession, use_session
from agent_utilities.layers.contracts import (
    HarnessNotConfigured,
    HarnessNotInstalled,
    HarnessOutcomeUncertain,
    HarnessRefused,
    HarnessRunFailed,
    RunSpec,
    RunToolset,
)
from agent_utilities.layers.execution import (
    HarnessRegistry,
    RunOutcome,
    default_registry,
    host_workspace_lease,
    release_workspace,
    run_to_completion,
)
from agent_utilities.layers.negotiation import HarnessPolicy

#: The in-process runtime reads context through the session-bound EG client,
#: not an MCP connection, so it does not need an EG MCP endpoint.
IN_PROCESS_POLICY = HarnessPolicy(require_context_endpoint=False)
IN_PROCESS_HARNESS = "pydantic-ai"

#: Request fields forwarded to the runtime as ``RunSpec.runtime_options``
#: (request field -> runtime option name).
_RUNTIME_FIELDS: dict[str, str] = {
    "max_steps": "max_steps",
    "return_mermaid": "return_mermaid",
    "context": "context",
    "budget_tokens": "budget_tokens",
    "context_ref": "context_ref",
    "session_ref": "session_id",
    "open_channel": "open_channel",
    "memento_source": "memento_source",
    "execution_profile": "execution_profile",
    "reasoning_effort": "reasoning_effort",
    "model_class": "model_class",
    "response_format": "response_format",
    "include_run_summary": "include_run_summary",
    "skill_name": "skill_name",
    "tool_server": "tool_server",
    "execution_mode": "execution_mode",
    "grounding": "grounding",
}


def run_spec_for(request: AgentExecutionRequest, run_id: str) -> RunSpec:
    """The digest-bound RunSpec equivalent of one execution request."""
    options: dict[str, Any] = {
        option: getattr(request, field)
        for field, option in _RUNTIME_FIELDS.items()
        if getattr(request, field) is not None
    }
    return RunSpec(
        run_id=run_id,
        task=request.task,
        agent_ref=request.agent_name,
        toolset=RunToolset(
            allowed_tools=request.allowed_tools,
            required_tools=request.required_tools or (),
        ),
        account_mode="api_key" if request.credential_ref else "subscription",
        account_ref=request.credential_ref,
        runtime_options=options,
    )


def _raise_for(outcome: RunOutcome) -> None:
    status = outcome.result.status
    if status == "succeeded":
        return
    cause = outcome.failure.__cause__ if outcome.failure is not None else None
    if isinstance(cause, Exception):
        raise cause
    if status == "outcome_uncertain":
        raise HarnessOutcomeUncertain(outcome.result.error)
    raise HarnessRunFailed(outcome.result.error or status)


class HarnessAgentExecutor:
    """:class:`AgentExecutionPort` that runs every request through a harness."""

    def __init__(
        self,
        registry: HarnessRegistry,
        *,
        harness: str = IN_PROCESS_HARNESS,
        policy: HarnessPolicy = IN_PROCESS_POLICY,
        workspace_root: str | None = None,
    ) -> None:
        self._registry = registry
        self._harness = harness
        self._policy = policy
        self._workspace_root = workspace_root

    @classmethod
    def in_process(cls, runner: Any) -> HarnessAgentExecutor:
        """The default executor over AU's own runtime."""
        if not callable(getattr(runner, "execute_agent", None)):
            raise TypeError("an AU agent runner with execute_agent is required")
        return cls(default_registry(runner))

    async def execute_agent(
        self, request: AgentExecutionRequest, *, session: GraphSession
    ) -> AgentExecutionResult:
        run_id = request.run_id or f"run-{uuid.uuid4().hex}"
        spec = run_spec_for(request, run_id)
        root = self._workspace_root
        lease = host_workspace_lease(root, run_id) if root else None
        try:
            with use_session(session):
                outcome = await run_to_completion(
                    self._registry.get(self._harness),
                    spec,
                    policy=self._policy,
                    lease=lease,
                )
        except (HarnessRefused, HarnessNotInstalled, HarnessNotConfigured) as exc:
            raise AgentControlPlaneUnavailable(str(exc)) from exc
        finally:
            if root and lease is not None:
                release_workspace(root, lease)
        _raise_for(outcome)
        mode = None if request.execution_mode == "auto" else request.execution_mode
        return AgentExecutionResult(
            run_id=run_id, output=outcome.result.output, execution_mode=mode
        )


__all__ = [
    "IN_PROCESS_HARNESS",
    "IN_PROCESS_POLICY",
    "HarnessAgentExecutor",
    "run_spec_for",
]
