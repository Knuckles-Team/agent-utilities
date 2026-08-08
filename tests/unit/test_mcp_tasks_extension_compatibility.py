"""Wire-level compatibility guard for the 2026 MCP Tasks extension.

GraphOS mounts a native, WorkItem-backed `io.modelcontextprotocol/tasks`
extension (`agent_utilities/mcp/tasks_extension.py`,
CONCEPT:AU-ECO.mcp.tasks-workitem-bridge) -- NOT the separate `fastmcp-tasks`
package (SEP-2663), whose execution engine is hard-wired to Docket/Redis, a
second job system this codebase's one `WorkItem` state machine (AU-P1-1)
forbids duplicating. This exercises real FastMCP initialization metadata and
handler registration rather than a local capability model.

Until 2026-07-31 this test asserted the OPPOSITE (GraphOS must NOT advertise
the extension until wired up) -- see `docs/architecture/mcp_v2_gateway.md`
and the isolated `mcp_v2_gateway` sidecar's own Tasks↔WorkItem mapping, which
landed first. Left passing unchanged, that negative assertion would have hidden
the feature actually shipping (a Wire-First violation), so it is updated here
to the real, current contract instead.

CONCEPT:AU-ECO.mcp.protocol-compat-bridge — MCP SDK v2's `LowLevelServer` (the
`mcp._mcp_server` this test drives) renamed the request-handler store from a
public `request_handlers` dict keyed by request-model CLASS to a private
`_request_handlers` dict keyed by wire-protocol METHOD STRING (e.g. `"tools/list"`),
reached through the public `get_request_handler(method: str) -> HandlerEntry | None`
accessor.
"""

from __future__ import annotations

import pytest


def test_graphos_advertises_and_serves_the_workitem_backed_tasks_extension() -> None:
    from agent_utilities.mcp.server_factory import create_mcp_server
    from agent_utilities.mcp.tasks_extension import (
        TASKS_EXTENSION_AVAILABLE,
        TASKS_EXTENSION_ID,
        WorkItemTasksExtension,
    )

    # Proof #1 of 2 (D-FMC-1): with a real fastmcp>=4.0.0b1 install (this
    # repo's own env), the defensive import in tasks_extension.py must be a
    # no-op -- the extension registers EXACTLY as before the guard was added.
    assert TASKS_EXTENSION_AVAILABLE is True

    _args, mcp, _middlewares = create_mcp_server(
        name="graph-os-test",
        command_args=[],
    )

    capabilities = mcp._mcp_server.create_initialization_options().capabilities
    assert TASKS_EXTENSION_ID in (capabilities.extensions or {})

    # The tasks extension registers exactly these three methods when mounted,
    # each bound to WorkItemTasksExtension's own handler -- reachable, and
    # backed by the native WorkItem authority, not fastmcp-tasks' Docket engine.
    for method in ("tasks/get", "tasks/update", "tasks/cancel"):
        entry = mcp._mcp_server.get_request_handler(method)
        assert entry is not None
        # The bound params type is this module's own wire model, not
        # fastmcp_tasks' -- proof it's WorkItemTasksExtension that answered,
        # not some other tasks-extension implementation.
        assert entry.params_type.__module__ == WorkItemTasksExtension.__module__


def test_tasks_extension_degrades_gracefully_when_fastmcp_server_extensions_is_absent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """D-FMC-1 / D-W2C2-2: proof #2 of 2, the fastmcp-3 fleet-image path.

    58 fleet pods hostPath-mount this canonical working tree over their own
    site-packages but ship fastmcp 3.x images (D-SH-3); on those images
    ``fastmcp.server.extensions`` does not exist, and an unguarded module-scope
    import there crashed the WHOLE server before anything else ran. This
    simulates that absence with a patched ``__import__`` (rather than actually
    uninstalling fastmcp 4 from this env) and proves three things:

    1. the module no longer raises at import time -- it degrades
       ``TASKS_EXTENSION_AVAILABLE`` to ``False`` instead;
    2. the ORIGINAL cause is logged, not swallowed (this program's hard-won
       rule after two outages hid behind swallowed exceptions today);
    3. ``server_factory.create_mcp_server`` still builds a working server,
       simply without the Tasks extension's three methods mounted.

    Removing the ``try/except ImportError`` guard around the
    ``fastmcp.server.extensions`` import in ``tasks_extension.py`` makes THIS
    test fail with the very ``ModuleNotFoundError`` it exists to catch (the
    reimport below raises instead of returning a degraded module).
    """
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "fastmcp.server.extensions":
            raise ModuleNotFoundError("No module named 'fastmcp.server.extensions'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    sys.modules.pop("agent_utilities.mcp.tasks_extension", None)
    sys.modules.pop("agent_utilities.mcp.server_factory", None)

    try:
        with caplog.at_level("WARNING", logger="agent_utilities.mcp.tasks_extension"):
            te = importlib.import_module("agent_utilities.mcp.tasks_extension")

        # (1) degraded, not crashed.
        assert te.TASKS_EXTENSION_AVAILABLE is False

        # (2) the cause is logged, naming fastmcp 4 as the requirement -- an
        # operator must be able to tell "Tasks extension unavailable on this
        # image" from "server broken".
        assert "fastmcp>=4.0.0b1" in caplog.text
        assert "fastmcp.server.extensions" in caplog.text
        assert "unavailable" in caplog.text

        # (3) the server still builds, and simply doesn't mount the extension.
        from agent_utilities.mcp.server_factory import create_mcp_server

        _args, mcp, _middlewares = create_mcp_server(
            name="graph-os-test-degraded",
            command_args=[],
        )
        capabilities = mcp._mcp_server.create_initialization_options().capabilities
        assert te.TASKS_EXTENSION_ID not in (capabilities.extensions or {})
        for method in ("tasks/get", "tasks/update", "tasks/cancel"):
            assert mcp._mcp_server.get_request_handler(method) is None
    finally:
        # Force the NEXT import (this file's other tests, or any later test
        # module) to pick up the real, guarded module against the real
        # fastmcp 4 install -- monkeypatch undoes `__import__` automatically,
        # but a stale degraded module would otherwise linger in sys.modules.
        sys.modules.pop("agent_utilities.mcp.tasks_extension", None)
        sys.modules.pop("agent_utilities.mcp.server_factory", None)


def test_tasks_error_path_uses_v1_error_data_shape(monkeypatch) -> None:
    """SDK v1's McpError receives one ErrorData model on a real Tasks error path."""
    from types import SimpleNamespace

    from mcp.shared import exceptions as mcp_exceptions

    import agent_utilities.mcp.kg_server as kg
    import agent_utilities.mcp.protocol_compat as protocol_compat
    from agent_utilities.mcp.tasks_extension import WorkItemTasksExtension

    class _ErrorData:
        def __init__(self, *, code, message, data=None):
            self.code = code
            self.message = message
            self.data = data

    class _V1Error(BaseException):
        def __init__(self, error):
            self.error = error

    monkeypatch.delattr(mcp_exceptions, "MCPError", raising=False)
    monkeypatch.setattr(mcp_exceptions, "McpError", _V1Error, raising=False)
    monkeypatch.setattr(
        protocol_compat,
        "mcp_types_module",
        lambda: SimpleNamespace(ErrorData=_ErrorData),
    )
    monkeypatch.setattr(kg, "_get_engine", lambda: None)

    with pytest.raises(_V1Error) as exc_info:
        WorkItemTasksExtension._engine()

    assert isinstance(exc_info.value.error, _ErrorData)
    assert (
        exc_info.value.error.code,
        exc_info.value.error.message,
        exc_info.value.error.data,
    ) == (-32603, "IntelligenceGraphEngine not active.", None)


@pytest.mark.asyncio
async def test_graph_jobs_cancel_uses_the_dispatched_work_item(monkeypatch) -> None:
    import json

    import agent_utilities.mcp.kg_server as kg
    from agent_utilities.mcp.tools.job_tools import register_job_tools
    from agent_utilities.orchestration.work_item import orchestrator_work_item_id

    class _MCP:
        def tool(self, **_kwargs):
            return lambda function: function

    job_id = "job:mcp-task-cancel"
    item_id = orchestrator_work_item_id(job_id)

    class _Authority:
        def __init__(self) -> None:
            self.node = {
                "id": item_id,
                "tenant": "tenant-test",
                "status": "ready",
                "depends_on": [],
                "downstream_ids": [],
                "metadata": {},
            }
            self.cancel_requests: list[dict] = []

        def query_cypher(self, _query: str, params: dict | None = None):
            return [self.node] if (params or {}).get("id") == item_id else []

        def cancel_work_item(self, request: dict):
            self.cancel_requests.append(request)
            self.node["status"] = "cancelled"
            return {"status": "cancelled"}

    authority = _Authority()

    class _Engine:
        _work_item_engine = authority

    engine = _Engine()
    register_job_tools(_MCP())
    monkeypatch.setattr(kg, "_get_engine", lambda: engine)

    cancelled = json.loads(
        await kg._execute_tool("graph_jobs", action="cancel", job_id=job_id)
    )

    assert cancelled == {"status": "cancelled", "job_id": job_id}
    assert authority.node["status"] == "cancelled"
    assert authority.cancel_requests[0]["work_item_id"] == item_id


@pytest.mark.asyncio
async def test_completed_task_surfaces_the_real_run_trace_output_not_the_opaque_marker(
    monkeypatch,
) -> None:
    """D-25-4: a completed orchestrator task's ``tasks/get`` result must
    surface the REAL agent output (the correlated ``:RunTrace``'s
    ``result_preview``) rather than the opaque ``result_ref`` completion
    marker string, when the run was pinned to the task's own id
    (``_execute_orchestrator_turn``'s ``run_id=envelope.job_id``,
    D-25-4's fix in ``agent_dispatch_worker.py``)."""
    import agent_utilities.mcp.kg_server as kg
    from agent_utilities.mcp.tasks_extension import WorkItemTasksExtension
    from agent_utilities.observability.trace_ontology import (
        trace_id as canonical_trace_id,
    )
    from agent_utilities.orchestration.work_item import orchestrator_work_item_id

    job_id = "job:mcp-task-result"
    item_id = orchestrator_work_item_id(job_id)
    trace_node_id = canonical_trace_id(job_id)

    class _Authority:
        def query_cypher(self, _query: str, params: dict | None = None):
            if (params or {}).get("id") == item_id:
                return [
                    {
                        "id": item_id,
                        "tenant": "tenant-test",
                        "status": "succeeded",
                        "depends_on": [],
                        "downstream_ids": [],
                        "metadata": {},
                        "result_ref": f"orchestrator:{job_id}:completed",
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-01T00:01:00Z",
                    }
                ]
            return []

    class _Backend:
        def execute(self, query: str, params: dict | None = None):
            if (params or {}).get("tid") != trace_node_id:
                return []
            if "ToolCall" in query:
                return []
            return [
                {
                    "status": "succeeded",
                    "attribution_ref": None,
                    "task": "do the thing",
                    "timestamp": "2026-01-01T00:01:00Z",
                    "duration_ms": 42,
                    "result_preview": "the real agent output text",
                    "error": None,
                    "execution_mode": "auto",
                    "graph_evidence_schema_version": None,
                    "graph_topology": None,
                    "graph_topology_digest": None,
                    "graph_version_digest": None,
                    "graph_runtime_version": None,
                    "graph_node_sequence": None,
                    "graph_transition_sequence": None,
                    "graph_transition_count": None,
                    "graph_checkpoint_ids": None,
                    "graph_resume_supported": None,
                    "skill_ref": None,
                    "server_ref": None,
                    "model_ref": None,
                    "model_class": None,
                    "skill_instruction_digest": None,
                    "event_sequence": None,
                }
            ]

    authority = _Authority()
    backend_double = _Backend()

    class _Engine:
        _work_item_engine = authority
        backend = backend_double

    engine = _Engine()
    monkeypatch.setattr(kg, "_get_engine", lambda: engine)

    extension = WorkItemTasksExtension()
    result = extension._project(job_id)

    assert result.status == "completed"
    assert result.result == {
        "resultPreview": "the real agent output text",
        "runId": job_id,
    }


@pytest.mark.asyncio
async def test_completed_task_falls_back_to_opaque_marker_with_no_run_trace(
    monkeypatch,
) -> None:
    """A completed WorkItem with no correlated RunTrace (a non-orchestrator
    job kind, or a run predating D-25-4) still degrades to the prior,
    real opaque-marker behavior -- never an error."""
    import agent_utilities.mcp.kg_server as kg
    from agent_utilities.mcp.tasks_extension import WorkItemTasksExtension
    from agent_utilities.orchestration.work_item import orchestrator_work_item_id

    job_id = "job:mcp-task-no-trace"
    item_id = orchestrator_work_item_id(job_id)

    class _Authority:
        def query_cypher(self, _query: str, params: dict | None = None):
            if (params or {}).get("id") == item_id:
                return [
                    {
                        "id": item_id,
                        "tenant": "tenant-test",
                        "status": "succeeded",
                        "depends_on": [],
                        "downstream_ids": [],
                        "metadata": {},
                        "result_ref": f"orchestrator:{job_id}:completed",
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-01T00:01:00Z",
                    }
                ]
            return []

    class _Backend:
        def execute(self, _query: str, _params: dict | None = None):
            return []  # no RunTrace row -- correlated trace never landed

    authority = _Authority()
    backend_double = _Backend()

    class _Engine:
        _work_item_engine = authority
        backend = backend_double

    engine = _Engine()
    monkeypatch.setattr(kg, "_get_engine", lambda: engine)

    extension = WorkItemTasksExtension()
    result = extension._project(job_id)

    assert result.status == "completed"
    assert result.result == {"resultRef": f"orchestrator:{job_id}:completed"}
