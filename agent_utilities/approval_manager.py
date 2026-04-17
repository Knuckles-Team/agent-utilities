#!/usr/bin/python
# coding: utf-8
"""Approval Manager Module.

Provides a protocol-agnostic, asyncio-based pause/resume mechanism for
human-in-the-loop tool approval and MCP elicitation.  The core primitive
is an :class:`ApprovalManager` that maps request IDs to
:class:`asyncio.Future` objects.  Graph executor code ``await``s a
future to pause, and a server endpoint (``/api/approve``) resolves
it to resume.

The :func:`run_with_approvals` helper wraps pydantic-ai's two-call
``DeferredToolRequests`` → ``DeferredToolResults`` pattern into a
single blocking call that transparently handles approval rounds.

Design notes
------------
* Uses only pydantic-ai's native data structures (``DeferredToolRequests``,
  ``DeferredToolResults``, ``ToolApproved``, ``ToolDenied``).
* Completely decoupled from ACP — works with AG-UI, SSE, or any protocol
  that can POST a JSON response to ``/api/approve``.
* When running inside an ACP session, pydantic-acp's own
  ``NativeApprovalBridge`` handles approvals at the wrapper-agent level,
  so this manager is bypassed automatically.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDenied,
)

logger = logging.getLogger(__name__)

_MAX_APPROVAL_ROUNDS = 8


@dataclass
class ApprovalRequest:
    """A pending approval request that the UI must resolve."""

    request_id: str
    tool_calls: list[dict[str, Any]]
    """Serialised tool call metadata for UI rendering."""


class ApprovalManager:
    """Manages pending approval / elicitation requests using asyncio Futures.

    The graph executor pushes a request and ``await``s the corresponding
    future.  A server endpoint (``/api/approve``) looks up the future by
    ID and resolves it, unblocking the executor.

    Thread-safety: all operations are expected to run on a single asyncio
    event loop.
    """

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Producer side (called by the graph executor / MCP callback)
    # ------------------------------------------------------------------

    async def wait_for_approval(
        self,
        request_id: str,
        timeout: float = 0.0,
    ) -> dict[str, Any]:
        """Create a future and block until the UI resolves it.

        Args:
            request_id: Unique identifier for this approval request.
            timeout: Maximum seconds to wait.  ``0`` means wait forever.

        Returns:
            The resolution payload sent by the UI (typically contains
            per-tool-call decisions).

        Raises:
            asyncio.TimeoutError: If *timeout* > 0 and no response arrives.

        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future
        try:
            if timeout > 0:
                return await asyncio.wait_for(future, timeout=timeout)
            return await future
        finally:
            self._pending.pop(request_id, None)

    # ------------------------------------------------------------------
    # Consumer side (called by /api/approve endpoint)
    # ------------------------------------------------------------------

    def resolve(self, request_id: str, result: dict[str, Any]) -> bool:
        """Resolve a pending approval, unblocking the waiting coroutine.

        Args:
            request_id: The ID that was passed to :meth:`wait_for_approval`.
            result: The resolution payload (decisions, feedback, etc.).

        Returns:
            ``True`` if the request was found and resolved, ``False``
            otherwise.

        """
        future = self._pending.get(request_id)
        if future is not None and not future.done():
            future.set_result(result)
            return True
        return False

    @property
    def pending_count(self) -> int:
        """Number of currently pending requests."""
        return len(self._pending)

    def has_pending(self, request_id: str) -> bool:
        """Check whether *request_id* is still waiting."""
        return request_id in self._pending


# ------------------------------------------------------------------
# Helper: run a specialist agent with transparent approval handling
# ------------------------------------------------------------------


async def run_with_approvals(
    agent: Any,
    query: str | Sequence[Any],
    *,
    approval_manager: ApprovalManager,
    event_queue: Optional[asyncio.Queue] = None,
    request_id_prefix: str = "",
    approval_timeout: float = 0.0,
    max_rounds: int = _MAX_APPROVAL_ROUNDS,
    **run_kwargs: Any,
) -> Any:
    """Run a pydantic-ai agent, handling approval rounds transparently.

    If the agent returns :class:`DeferredToolRequests`, this function:

    1. Pushes an ``approval_required`` event to *event_queue*.
    2. Awaits user decisions via *approval_manager*.
    3. Re-runs the agent with :class:`DeferredToolResults`.
    4. Repeats up to *max_rounds* times.

    Args:
        agent: A ``pydantic_ai.Agent`` instance.
        query: The user query or structured message parts.
        approval_manager: The :class:`ApprovalManager` to use.
        event_queue: Optional sideband queue for streaming events to UIs.
        request_id_prefix: Prefix for generated approval request IDs.
        approval_timeout: Per-round timeout (0 = no timeout).
        max_rounds: Maximum approval round-trips before giving up.
        **run_kwargs: Extra kwargs forwarded to ``agent.run()``.

    Returns:
        The agent's final ``AgentRunResult``.

    """
    import uuid

    message_history = run_kwargs.pop("message_history", None)
    deferred_tool_results: DeferredToolResults | None = None

    for round_idx in range(max_rounds):
        result = await agent.run(
            query if round_idx == 0 else None,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            **run_kwargs,
        )

        output = result.output
        if not isinstance(output, DeferredToolRequests):
            return result

        if not output.approvals:
            return result

        # Serialize tool calls for the UI
        request_id = f"{request_id_prefix}{uuid.uuid4().hex[:12]}"
        tool_call_infos = []
        for tc in output.approvals:
            tool_call_infos.append(
                {
                    "tool_call_id": tc.tool_call_id,
                    "tool_name": tc.tool_name,
                    "args": tc.args_as_dict(),
                }
            )

        logger.info(
            f"Approval round {round_idx + 1}: {len(output.approvals)} tool(s) "
            f"need approval (request_id={request_id})"
        )

        # Push approval event to the UI via sideband queue
        if event_queue is not None:
            await event_queue.put(
                {
                    "type": "approval_required",
                    "request_id": request_id,
                    "tool_calls": tool_call_infos,
                }
            )

        # PAUSE: wait for the UI to resolve via /api/approve
        try:
            resolution = await approval_manager.wait_for_approval(
                request_id, timeout=approval_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Approval timed out for {request_id}")
            # Deny all on timeout
            resolution = {
                "decisions": {tc["tool_call_id"]: "deny" for tc in tool_call_infos}
            }

        # Build DeferredToolResults from user decisions
        decisions = resolution.get("decisions", {})
        approvals: dict[str, ToolApproved | ToolDenied] = {}
        for tc in tool_call_infos:
            cid = tc["tool_call_id"]
            decision = decisions.get(cid, "deny")
            if decision in ("accept", "approve", True):
                approvals[cid] = ToolApproved()
            else:
                approvals[cid] = ToolDenied(
                    message=resolution.get("feedback", "Tool call denied by user.")
                )

        deferred_tool_results = DeferredToolResults(approvals=approvals)
        message_history = result.all_messages()
        query = ""  # Don't resend prompt on continuation

        if event_queue is not None:
            await event_queue.put(
                {
                    "type": "approval_resolved",
                    "request_id": request_id,
                    "decisions": {
                        cid: "approved" if isinstance(v, ToolApproved) else "denied"
                        for cid, v in approvals.items()
                    },
                }
            )

    logger.error(f"Max approval rounds ({max_rounds}) exceeded")
    return result


# ------------------------------------------------------------------
# MCP Elicitation callback
# ------------------------------------------------------------------

# Context variable that carries the active elicitation queue into MCP
# server callbacks.  Set by the server/AG-UI endpoint before running
# the agent and read by the callback below.

elicitation_queue_var: contextvars.ContextVar[Optional[asyncio.Queue]] = (
    contextvars.ContextVar("elicitation_queue", default=None)
)

# Singleton manager reused for both tool approval and MCP elicitation.
# The /api/approve endpoint resolves requests from both sources.
elicitation_manager = ApprovalManager()


async def global_elicitation_callback(context: Any = None, params: Any = None) -> Any:
    """Standardised elicitation callback for MCP servers.

    When an MCP tool calls ``ctx.elicit()``, this callback:

    1. Pushes an ``elicitation`` event to the sideband queue (streamed
       to the UI via SSE).
    2. Awaits a resolution from :data:`elicitation_manager` (which the
       ``/api/approve`` endpoint resolves).

    Both the terminal UI and web UI can render the elicitation form and
    POST the user's response to ``/api/approve``.

    """
    queue = elicitation_queue_var.get()
    if not queue:
        logger.warning("No elicitation queue in context — blocking request.")
        return {"status": "error", "message": "No elicitation queue"}

    if params is not None:
        request_data = params.model_dump() if hasattr(params, "model_dump") else params
    elif isinstance(context, dict):
        request_data = context
    else:
        return {"status": "error", "message": "Invalid elicitation arguments"}

    import uuid as _uuid

    request_id = request_data.get("id") or _uuid.uuid4().hex
    request_data["id"] = request_id

    logger.info(f"MCP elicitation triggered: {request_id}")
    await queue.put({"type": "elicitation", **request_data})

    # PAUSE — wait for the UI to resolve via /api/approve
    result = await elicitation_manager.wait_for_approval(request_id)

    # If called from MCP session, try to return ElicitResult
    try:
        from mcp import types as mcp_types

        return mcp_types.ElicitResult(**result)
    except Exception:
        return result
