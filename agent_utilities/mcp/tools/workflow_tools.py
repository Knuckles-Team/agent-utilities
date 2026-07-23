"""Focused graph-os workflow lifecycle operations."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import public_error_text

_WORKFLOW_TASKS: dict[str, asyncio.Task[Any]] = {}


def _workflow_gate(engine: Any, name: str) -> dict[str, Any]:
    from agent_utilities.knowledge_graph.core.workflow_gate import (
        gate_workflow_execution,
    )

    return gate_workflow_execution(engine, name)


def _gate_denial(name: str, gate: dict[str, Any]) -> str:
    return json.dumps(
        {
            "error": "workflow definition failed ontology validation — execution refused",
            "workflow": name,
            "workflow_id": gate.get("workflow_id"),
            "violations": gate.get("violations", []),
        },
        default=str,
    )


def _workflow_mermaid(engine: Any, name: str) -> str | None:
    """Read the persisted topology diagram without failing the workflow operation."""
    try:
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        return WorkflowStore(engine).get_mermaid(name)
    except Exception:
        return None


def register_workflow_tools(mcp: Any) -> None:
    """Register compile, execute, dispatch, inspect, and export operations."""

    @mcp.tool(
        name="graph_workflows",
        description=(
            "Manage governed WorkflowDefinitions. Actions: 'compile', 'compile_process', "
            "'list', 'execute', 'dispatch', 'status', and 'export'. Execute and dispatch "
            "apply the same ontology/ACL gate; dispatch returns the exact runner session id."
        ),
        tags=["graph-os", "workflow", "orchestration"],
    )
    async def graph_workflows(
        action: str = Field(
            default="list",
            description="compile | compile_process | list | execute | dispatch | status | export",
        ),
        workflow: str = Field(
            default="", description="Workflow name, process id, or run/session id."
        ),
        task: str = Field(
            default="", description="Compilation description or workflow input task."
        ),
        name: str = Field(default="", description="Optional compiled workflow name."),
        export_format: str = Field(
            default="bpmn", description="bpmn | json | skill (export)."
        ),
        max_steps: int = Field(default=30, ge=1),
        limit: int = Field(default=50, ge=1),
    ) -> str:
        engine = kg_server._get_engine()
        if engine is None:
            return "Error: IntelligenceGraphEngine not active."
        try:
            from agent_utilities.orchestration.manager import Orchestrator

            orchestrator = Orchestrator(engine)
            if action == "compile":
                compiled_name = name or f"compiled_{uuid.uuid4().hex}"
                workflow_id = await orchestrator.compile_workflow(
                    name=compiled_name, task=task
                )
                return json.dumps(
                    {
                        "status": "compiled",
                        "workflow_id": workflow_id,
                        "name": compiled_name,
                        "mermaid": _workflow_mermaid(engine, compiled_name),
                    },
                    default=str,
                )

            if action == "compile_process":
                if not workflow:
                    raise ValueError("workflow must contain the BusinessProcess id")
                from agent_utilities.knowledge_graph.process_plan_compiler import (
                    ProcessPlanCompiler,
                )

                report = await ProcessPlanCompiler(engine).compile_and_store(
                    workflow, name=name or None
                )
                report["status"] = "compiled"
                report["mermaid"] = _workflow_mermaid(engine, report["name"])
                return json.dumps(report, default=str)

            if action == "list":
                from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

                return json.dumps(
                    {
                        "source": "kg",
                        "workflows": WorkflowStore(engine).list_workflows(limit=limit),
                    },
                    default=str,
                )

            if action in {"execute", "dispatch"}:
                if not workflow:
                    raise ValueError("workflow is required")
                gate = _workflow_gate(engine, workflow)
                if gate.get("allowed") is not True:
                    return _gate_denial(workflow, gate)

                if action == "execute":
                    result = await orchestrator.execute_workflow(
                        workflow_id=workflow,
                        task=task,
                        max_steps=max_steps,
                    )
                    return json.dumps(
                        {
                            "result": result,
                            "mermaid": _workflow_mermaid(engine, workflow),
                        },
                        default=str,
                    )

                from agent_utilities.workflows.runner import WorkflowRunner

                session_id = f"wf-{uuid.uuid4().hex}"
                runner = WorkflowRunner()
                background = asyncio.create_task(
                    runner.execute_by_name(
                        workflow,
                        engine,
                        trace_session=session_id,
                        task=task or None,
                    ),
                    name=f"workflow:{session_id}",
                )
                _WORKFLOW_TASKS[session_id] = background
                return json.dumps(
                    {
                        "status": "dispatched",
                        "session_id": session_id,
                        "status_url": "/api/graph/workflows",
                        "status_request": {
                            "action": "status",
                            "workflow": session_id,
                        },
                    }
                )

            if action == "status":
                if not workflow:
                    raise ValueError("workflow must contain the run/session id")
                status_task = _WORKFLOW_TASKS.get(workflow)
                if status_task is not None and not status_task.done():
                    return json.dumps(
                        {"session_id": workflow, "status": "running"}, default=str
                    )
                if status_task is not None:
                    try:
                        task_result = status_task.result()
                    except Exception as exc:
                        return public_error_text(exc)
                    return json.dumps(task_result.to_dict(), default=str)

                from agent_utilities.workflows.runner import _active_workflows

                stored_result = _active_workflows.get(workflow)
                if stored_result is None:
                    return json.dumps({"session_id": workflow, "status": "not_found"})
                return json.dumps(stored_result.to_dict(), default=str)

            if action == "export":
                if not workflow:
                    raise ValueError("workflow is required for export")
                from agent_utilities.knowledge_graph.governance_import import (
                    export_workflow,
                )

                return json.dumps(
                    export_workflow(engine, workflow, fmt=export_format),
                    indent=2,
                    default=str,
                )

            return f"Error: Unknown graph_workflows action '{action}'"
        except PermissionError:
            raise
        except Exception as exc:
            return public_error_text(exc)

    kg_server.REGISTERED_TOOLS["graph_workflows"] = graph_workflows
    kg_server.ACTION_TOOL_ROUTES["graph_workflows"] = "/graph/workflows"
