"""Native GraphOS ``io.modelcontextprotocol/tasks`` extension, backed by WorkItem.

CONCEPT:AU-ECO.mcp.tasks-workitem-bridge

Reaches the same WorkItem-backed Tasks contract the isolated
``mcp_v2_gateway`` sidecar already implements for stateless 2026-07-28
clients (``mcp_v2_gateway/gateway.py``), but for a client connected directly
to GraphOS's own FastMCP 4 server -- ``mcp>=2.0.0`` is now the default (see
``server_factory.py``, ``docs/architecture/fastmcp4-default.md``).

Deliberately does **not** mount ``fastmcp_tasks.extension.TasksExtension``:
that package's execution engine is hard-wired to a Docket/Redis backend
(``fastmcp_tasks.handlers``/``creation.py``, ``pydocket`` dependency) -- a
second, parallel job system this codebase's one ``WorkItem`` state machine
(``AU-P1-1``) already forbids duplicating (``AGENTS.md``: extend before you
add). This module only reuses the extension's *registration* mechanism
(``fastmcp.server.extensions.ServerExtension``/``MethodBinding``) -- never
its execution engine -- and backs every method with
``agent_utilities.orchestration.work_item`` directly, the exact same
authority ``graph_jobs`` (``agent_utilities/mcp/tools/job_tools.py``) and the
isolated gateway both already use. The wire model field names below mirror
``fastmcp_tasks.models`` (the current SEP-2663 draft: flat task fields,
``inputRequests`` for ``input_required``) rather than the older, reduced
pinned revision the isolated gateway locks to for its own separate,
documented reasons (``mcp_v2_gateway/gateway.py``'s
``TASKS_EXTENSION_REVISION``).
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fastmcp.server.extensions import (
    MethodBinding,
    ServerExtension,
    read_client_extension_settings,
)
from mcp.shared.exceptions import MCPError
from mcp_types import RequestParams, Result
from mcp_types.jsonrpc import MISSING_REQUIRED_CLIENT_CAPABILITY
from mcp_types.version import MODERN_PROTOCOL_VERSIONS
from pydantic import ConfigDict, Field

if TYPE_CHECKING:
    from mcp.server.context import ServerRequestContext

logger = logging.getLogger(__name__)

TASKS_EXTENSION_ID = "io.modelcontextprotocol/tasks"
_TASK_METHOD_VERSIONS = frozenset(MODERN_PROTOCOL_VERSIONS)

# WorkItem raw statuses that project onto the extension's "working" (or, with
# a live pending_input_request, "input_required") wire status. Mirrors
# mcp_v2_gateway.gateway.GraphOSV2Gateway._WORKING_RAW_STATUSES, but this
# module targets `graph_jobs`'s orchestrator-dispatch WorkItems, whose raw
# vocabulary never includes "queued"/"pending"/"executing" (those are
# ingest/legacy-bridge synonyms projected only by the gateway's downstream
# `graph_jobs(action="status")` JSON, not the raw WORK_ITEM_STATES tuple).
_WORKING_RAW_STATUSES = frozenset({"submitted", "ready", "leased", "running"})


class _GetTaskParams(RequestParams):
    model_config = ConfigDict(populate_by_name=True)

    task_id: str = Field(alias="taskId")


_CancelTaskParams = _GetTaskParams


class _UpdateTaskParams(RequestParams):
    model_config = ConfigDict(populate_by_name=True)

    task_id: str = Field(alias="taskId")
    input_responses: dict[str, Any] = Field(alias="inputResponses")


class _GetTaskResult(Result):
    """``GetTaskResult`` (SEP-2663): flat task fields plus exactly one of
    ``result``/``error``/``input_requests`` depending on ``status``."""

    model_config = ConfigDict(populate_by_name=True)

    result_type: str = Field(default="complete", serialization_alias="resultType")
    task_id: str = Field(serialization_alias="taskId")
    status: str
    created_at: str = Field(serialization_alias="createdAt")
    last_updated_at: str = Field(serialization_alias="lastUpdatedAt")
    ttl_ms: float | None = Field(default=None, serialization_alias="ttlMs")
    status_message: str | None = Field(
        default=None, serialization_alias="statusMessage"
    )
    poll_interval_ms: float | None = Field(
        default=1_000, serialization_alias="pollIntervalMs"
    )
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    input_requests: dict[str, Any] | None = Field(
        default=None, serialization_alias="inputRequests"
    )


class _AckResult(Result):
    """Acknowledgement shape for ``tasks/update``/``tasks/cancel``."""

    model_config = ConfigDict(populate_by_name=True)

    result_type: str = Field(default="complete", serialization_alias="resultType")


def _iso_timestamp(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (
            datetime.fromtimestamp(value, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
    return (
        datetime.fromtimestamp(time.time(), tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class WorkItemTasksExtension(ServerExtension):
    """Backs ``tasks/get``/``tasks/update``/``tasks/cancel`` with native WorkItems.

    Only ``orchestrator``-kind WorkItems (``graph_jobs(action="dispatch")``'s
    ``orch-<hex>`` job ids) are addressable through this extension today --
    the same scope the gateway's Tasks projection already has, since both
    adapters exist to expose ``graph_jobs``' durable handle, not every
    WorkItem kind in the graph.
    """

    identifier = TASKS_EXTENSION_ID

    def methods(self) -> list[MethodBinding]:
        return [
            MethodBinding(
                method="tasks/get",
                params_type=_GetTaskParams,
                handler=self._handle_get,
                protocol_versions=_TASK_METHOD_VERSIONS,
            ),
            MethodBinding(
                method="tasks/update",
                params_type=_UpdateTaskParams,
                handler=self._handle_update,
                protocol_versions=_TASK_METHOD_VERSIONS,
            ),
            MethodBinding(
                method="tasks/cancel",
                params_type=_CancelTaskParams,
                handler=self._handle_cancel,
                protocol_versions=_TASK_METHOD_VERSIONS,
            ),
        ]

    def _require_tasks_capability(self, ctx: ServerRequestContext[Any, Any]) -> None:
        """SEP-2663: reject a task method the client did not opt into this request."""
        if read_client_extension_settings(ctx, TASKS_EXTENSION_ID) is None:
            raise MCPError(
                code=MISSING_REQUIRED_CLIENT_CAPABILITY,
                message=(
                    f"This request targets the tasks extension "
                    f"({TASKS_EXTENSION_ID}); the client did not declare it "
                    "for this request."
                ),
                data={"requiredCapabilities": {"extensions": {TASKS_EXTENSION_ID: {}}}},
            )

    @staticmethod
    def _engine() -> Any:
        from agent_utilities.mcp import kg_server

        engine = kg_server._get_engine()
        if engine is None:
            raise MCPError(code=-32603, message="IntelligenceGraphEngine not active.")
        return getattr(engine, "_work_item_engine", engine)

    async def _handle_get(
        self, ctx: ServerRequestContext[Any, Any], params: _GetTaskParams
    ) -> _GetTaskResult:
        self._require_tasks_capability(ctx)
        return self._project(params.task_id)

    async def _handle_cancel(
        self, ctx: ServerRequestContext[Any, Any], params: _GetTaskParams
    ) -> _AckResult:
        self._require_tasks_capability(ctx)
        from agent_utilities.orchestration import work_item as _wi

        item_id = _wi.orchestrator_work_item_id(params.task_id)
        if not _wi.cancel_work_item(self._engine(), item_id):
            raise MCPError(code=-32602, message="Failed to cancel task")
        return _AckResult()

    async def _handle_update(
        self, ctx: ServerRequestContext[Any, Any], params: _UpdateTaskParams
    ) -> _AckResult:
        self._require_tasks_capability(ctx)
        from agent_utilities.knowledge_graph.core.session import resolve_session
        from agent_utilities.orchestration import work_item as _wi

        item_id = _wi.orchestrator_work_item_id(params.task_id)
        session = resolve_session(required_scope="kg:write")
        if not _wi.submit_work_item_input(
            self._engine(),
            item_id,
            tenant=session.tenant,
            response=params.input_responses,
        ):
            raise MCPError(code=-32602, message="Failed to submit task input")
        return _AckResult()

    def _project(self, task_id: str) -> _GetTaskResult:
        from agent_utilities.orchestration import work_item as _wi

        item_id = _wi.orchestrator_work_item_id(task_id)
        item = _wi.get_work_item(self._engine(), item_id)
        if item is None:
            raise MCPError(code=-32602, message="Unknown task")
        raw_status = str(item.get("status") or "").lower()
        metadata = item.get("metadata")
        pending = (
            metadata.get("pending_input_request")
            if isinstance(metadata, dict)
            else None
        )
        if raw_status in _WORKING_RAW_STATUSES and isinstance(pending, dict):
            status = "input_required"
        elif raw_status in _WORKING_RAW_STATUSES:
            status = "working"
        elif raw_status == "succeeded":
            status = "completed"
        elif raw_status in {"failed", "dead_letter"}:
            status = "failed"
        elif raw_status == "cancelled":
            status = "cancelled"
        else:
            raise MCPError(code=-32603, message="Unknown WorkItem status")
        result = _GetTaskResult(
            task_id=task_id,
            status=status,
            created_at=_iso_timestamp(item.get("created_at")),
            last_updated_at=_iso_timestamp(
                item.get("updated_at") or item.get("created_at")
            ),
            # Native WorkItems have no automatic record-expiration policy: they
            # are durable graph records (mirrors the gateway's own ttlMs: null).
            ttl_ms=None,
        )
        if status == "input_required":
            result.input_requests = {"request": pending}
        elif status == "completed":
            ref = item.get("result_ref")
            result.result = (
                {"resultRef": ref} if ref is not None else {"status": "completed"}
            )
        elif status == "failed":
            result.error = {"code": -32603, "message": "GraphOS WorkItem failed"}
        return result
