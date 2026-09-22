"""The fully bound agent control plane a host composes per verified caller.

A host (GraphOS) holds only two things it may legitimately own: the
session-routed epistemic-graph client for the caller's graph and the verified
:class:`GraphSession`. :func:`compose_hosted_agent_control_plane` binds every
other port from AU's own runtime and configuration, so the host never imports
AU internals:

* capability search and the WorkItem store over EG
  (:mod:`agent_utilities.api.agent_control_adapters`);
* agent execution on the process runtime's ``Orchestrator``;
* :class:`QueueSignedAgentDispatch`, which signs a ``work_item_turn`` carrier
  (including the tool allowlist) onto AU's ``agent_turns`` queue;
* :class:`ProcessRunOutputReader`, the redacted run-output read (AU-5).

The process runtime is never created here: a host must have opened it (see
:mod:`agent_utilities.api.runtime`), and every port fails closed with
:class:`AgentControlPlaneUnavailable` when it has not.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING, Any

from agent_utilities.api.agent_control_adapters import (
    AuthenticationMethod,
    EgCapabilitySearch,
    EgWorkItemStore,
    OrchestratorAgentExecutor,
)
from agent_utilities.api.agent_control_contracts import (
    AgentControlPlaneUnavailable,
    RunOutput,
    RunOutputRequest,
    RunStatus,
    SignedAgentDispatchReceipt,
    SignedAgentDispatchRequest,
)
from agent_utilities.api.session import GraphSession, use_session

if TYPE_CHECKING:
    from agent_utilities.api.agent_control_plane import AgentControlPlane

#: Longest run output returned to a caller; longer output is truncated.
MAX_RUN_OUTPUT_CHARS = 100_000

_RUN_STATUSES: dict[str, RunStatus] = {
    "running": "running",
    "success": "succeeded",
    "succeeded": "succeeded",
    "completed": "succeeded",
    "error": "failed",
    "failed": "failed",
    "timeout": "failed",
    "cancelled": "failed",
    "refused": "failed",
}


def process_engine() -> Any:
    """The already-open AU process runtime engine, or fail closed."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if engine is None:
        raise AgentControlPlaneUnavailable(
            "the AU process runtime is not open; call open_process_runtime first"
        )
    return engine


def _digest(label: str, value: str) -> str:
    return hashlib.sha256(f"{label}\0{value}".encode()).hexdigest()


def admission_digests(session: GraphSession) -> tuple[str, str, str]:
    """``(policy, catalog, model)`` provenance digests for WorkItem admission.

    Each names what governed the admission: the session's policy version, the
    AU release that owns the capability catalog and the configured chat
    model set.
    """
    from agent_utilities._version import __version__
    from agent_utilities.core.config import config

    models = sorted(
        str(getattr(model, "id", "") or "")
        for model in getattr(config, "chat_models", ()) or ()
    )
    return (
        _digest("au-policy", str(session.policy_version)),
        _digest("au-catalog", __version__),
        _digest("au-model", ",".join(models)),
    )


def authentication_method_for(session: GraphSession) -> AuthenticationMethod:
    """The EG provenance label for how the session's principal authenticated."""
    from agent_utilities.security.request_identity import _is_local_process_context

    if _is_local_process_context(session.engine_verified_context()):
        return "local_process"
    return "oidc"


class _ProcessRunner:
    """Resolves the process ``Orchestrator`` per call, never at composition."""

    async def execute_agent(self, agent_name: str, task: str, **options: Any) -> str:
        from agent_utilities.orchestration.manager import Orchestrator

        runner = Orchestrator(process_engine())
        return await runner.execute_agent(agent_name, task, **options)


class QueueSignedAgentDispatch:
    """``SignedAgentDispatchPort`` onto AU's signed ``agent_turns`` queue.

    The WorkItem was already admitted by the EG store, so this only publishes
    a signed ``work_item_turn`` carrier referencing it; the worker claims and
    commits that same WorkItem, and the queue ack waits for its terminal state.
    """

    def __init__(self, queue: Any = None) -> None:
        self._queue = queue

    async def enqueue(
        self, request: SignedAgentDispatchRequest, *, session: GraphSession
    ) -> SignedAgentDispatchReceipt:
        from agent_utilities.orchestration.agent_dispatch import (
            KIND_WORK_ITEM_TURN,
            AgentTurnEnvelope,
            DispatchCarrierError,
            DispatchQueueFull,
        )

        envelope = AgentTurnEnvelope(
            job_id=request.job_id,
            session_id=request.session_ref,
            kind=KIND_WORK_ITEM_TURN,
            payload_ref=request.work_item_id,
            agent_name=request.agent_name,
            tenant=session.tenant,
            allowed_tools=request.allowed_tools,
        )
        try:
            accepted = await asyncio.to_thread(self._publish, envelope)
        except DispatchQueueFull as exc:
            raise AgentControlPlaneUnavailable(
                "the agent dispatch queue is at its admission bound"
            ) from exc
        except DispatchCarrierError as exc:
            raise AgentControlPlaneUnavailable(
                "the dispatch carrier could not be signed"
            ) from exc
        return SignedAgentDispatchReceipt(job_id=request.job_id, accepted=accepted)

    def _publish(self, envelope: Any) -> bool:
        from agent_utilities.core.config import config
        from agent_utilities.orchestration.agent_dispatch import (
            DispatchQueueFull,
            dispatch_queue_depth,
            get_dispatch_queue,
        )

        envelope.ensure_authenticated_carrier()
        queue = self._queue if self._queue is not None else get_dispatch_queue()
        max_depth = int(config.agent_dispatch_max_depth)
        if dispatch_queue_depth(queue) >= max_depth:
            raise DispatchQueueFull("agent dispatch queue is at its admission bound")
        return bool(queue.put_if_below(envelope.to_item(), max_depth))


def _bounded_redacted(text: str) -> tuple[str, bool]:
    from agent_utilities.orchestration.task_guard import redact_agent_task

    redacted = redact_agent_task(text) if text else ""
    if len(redacted) <= MAX_RUN_OUTPUT_CHARS:
        return redacted, False
    return redacted[:MAX_RUN_OUTPUT_CHARS], True


def run_output_from_trace(run_id: str, trace: dict[str, Any]) -> RunOutput | None:
    """Project one ``:RunTrace`` read to the public, redacted run output."""
    raw_status = str(trace.get("status") or "")
    if raw_status == "not_found":
        return None
    output, truncated = _bounded_redacted(str(trace.get("result_preview") or ""))
    return RunOutput(
        run_id=run_id,
        status=_RUN_STATUSES.get(raw_status.casefold(), "unknown"),
        output=output,
        truncated=truncated,
    )


class ProcessRunOutputReader:
    """``RunOutputPort`` over the process runtime's RunTrace, under the caller's session.

    The read runs with the verified session bound, so the graph routing and
    read authorization are the caller's; a run outside that scope is not found.
    """

    async def get_run_output(
        self, request: RunOutputRequest, *, session: GraphSession
    ) -> RunOutput | None:
        from agent_utilities.orchestration.manager import Orchestrator

        runner = Orchestrator(process_engine())

        def _read() -> dict[str, Any]:
            with use_session(session):
                return runner.get_run_trace(request.run_id)

        trace = await asyncio.to_thread(_read)
        return run_output_from_trace(request.run_id, trace)


def compose_hosted_agent_control_plane(
    eg_client: Any, session: GraphSession
) -> AgentControlPlane:
    """The control plane for one verified caller, bound entirely by AU (AU-1)."""
    from agent_utilities.api.agent_control_plane import compose_agent_control_plane

    policy_digest, catalog_digest, model_digest = admission_digests(session)
    return compose_agent_control_plane(
        eg_client,
        session,
        capability_search=EgCapabilitySearch(eg_client),
        agent_executor=OrchestratorAgentExecutor(_ProcessRunner()),
        work_item_store=EgWorkItemStore(
            eg_client,
            authentication_method=authentication_method_for(session),
            policy_digest=policy_digest,
            catalog_digest=catalog_digest,
            model_digest=model_digest,
        ),
        signed_dispatch=QueueSignedAgentDispatch(),
        run_output=ProcessRunOutputReader(),
    )


__all__ = [
    "MAX_RUN_OUTPUT_CHARS",
    "ProcessRunOutputReader",
    "QueueSignedAgentDispatch",
    "admission_digests",
    "authentication_method_for",
    "compose_hosted_agent_control_plane",
    "process_engine",
    "run_output_from_trace",
]
