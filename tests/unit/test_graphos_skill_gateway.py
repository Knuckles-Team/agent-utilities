"""Live-path contract for KG-resolved local-vLLM delegation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_utilities.orchestration.manager import Orchestrator


class _FakeMCP:
    def tool(self, *, name: str, description: str = "", tags=None):
        def decorator(fn):
            return fn

        return decorator


def _orchestrator(engine: MagicMock) -> Orchestrator:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.engine = engine
    orchestrator._scan_task = MagicMock()  # type: ignore[method-assign]
    return orchestrator


def test_resolve_capability_prefers_typed_kg_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.knowledge_graph.core import secured_reads

    monkeypatch.setattr(secured_reads, "permit", lambda node_ids: node_ids)
    engine = MagicMock()
    engine.search_hybrid.return_value = [
        {"id": "chunk:unrelated", "type": "Chunk", "score": 0.99},
        {
            "id": "resource:skill:github-review",
            "node_type": "CallableResource",
            "resource_type": "AGENT_SKILL",
            "name": "github-review",
            "score": 0.91,
        },
        {
            "id": "skill_workflow:review-release",
            "node_type": "WorkflowDefinition",
            "name": "review-release",
            "score": 0.83,
        },
    ]

    resolved = _orchestrator(engine).resolve_capability("review GitHub PR 458")

    assert resolved["kind"] == "skill"
    assert resolved["name"] == "github-review"
    assert resolved["source"] == "kg_hybrid"
    assert resolved["alternatives"][0]["kind"] == "workflow"


def test_resolve_capability_hides_unpermitted_cross_tenant_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.knowledge_graph.core import secured_reads

    monkeypatch.setattr(secured_reads, "permit", lambda _node_ids: [])
    engine = MagicMock()
    engine.search_hybrid.return_value = [
        {
            "id": "skill_workflow:stale-local-workflow",
            "node_type": "WorkflowDefinition",
            "name": "stale-local-workflow",
            "score": 0.99,
        }
    ]

    resolved = _orchestrator(engine).resolve_capability("review a pull request")

    assert resolved["kind"] == "agent"
    assert resolved["name"] == "agent-utilities-expert"
    assert resolved["source"] == "default"


def test_resolve_capability_falls_back_to_kg_bound_expert() -> None:
    engine = MagicMock()
    engine.search_hybrid.return_value = [
        {"id": "document:1", "type": "Document", "score": 0.88}
    ]

    resolved = _orchestrator(engine).resolve_capability("handle an unusual task")

    assert resolved == {
        "kind": "agent",
        "name": "agent-utilities-expert",
        "id": "",
        "score": 0.0,
        "source": "default",
        "alternatives": [],
    }


def test_resolve_capability_recognizes_a_bare_tool_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``:Tool`` node (CONCEPT:AU-KG.retrieval.unified-capability-contract) must resolve
    to kind="tool" with its owning server, not be silently dropped."""
    from agent_utilities.knowledge_graph.core import secured_reads

    monkeypatch.setattr(secured_reads, "permit", lambda node_ids: node_ids)
    engine = MagicMock()
    engine.search_hybrid.return_value = [
        {
            "id": "tool_github-mcp_list_issues",
            "node_type": "Tool",
            "name": "list_issues",
            "mcp_server": "github-mcp",
            "score": 0.87,
        }
    ]

    resolved = _orchestrator(engine).resolve_capability("list open GitHub issues")

    assert resolved["kind"] == "tool"
    assert resolved["name"] == "list_issues"
    assert resolved["server"] == "github-mcp"


@pytest.mark.asyncio
async def test_execute_capability_binds_a_resolved_tool_via_capability_contract() -> (
    None
):
    """When resolution lands on a bare tool (no caller skill_name/agent_name),
    execute_capability must bind it through the same Capability contract a
    ranked find/find_tools result would — the default expert scoped to just
    that one tool — instead of mis-using the tool name as an agent name."""
    engine = MagicMock()
    orchestrator = _orchestrator(engine)
    orchestrator.resolve_capability = MagicMock(
        return_value={
            "kind": "tool",
            "name": "list_issues",
            "id": "tool_github-mcp_list_issues",
            "score": 0.87,
            "source": "kg_hybrid",
            "server": "github-mcp",
            "alternatives": [],
        }
    )
    orchestrator.execute_agent = AsyncMock(
        return_value=json.dumps(
            {
                "output": "issues listed",
                "run_id": "run:0123456789abcdef0123456789abcdef",
                "run_summary": {"outcome": "ok"},
            }
        )
    )
    orchestrator._run_provenance = MagicMock(
        return_value={
            "status": "completed",
            "tool_call_count": 1,
            "tool_calls": [{"tool_name": "list_issues", "status": "ok"}],
        }
    )

    result = await orchestrator.execute_capability(task="List open GitHub issues.")

    assert result["resolution"]["kind"] == "tool"
    call = orchestrator.execute_agent.await_args.kwargs
    assert call["agent_name"] == "agent-utilities-expert"
    assert call["tool_server"] == "github-mcp"
    assert call["allowed_tools"] == ["list_issues"]
    # The delegate must be named on BOTH keywords: run_agent enforces
    # "tool_server requires skill_name" AND "skill_name must match the
    # dispatched agent_name". This previously asserted None, which encoded the
    # defect -- the real run_agent (mocked out here) raised ValueError on every
    # auto-resolved Tool. See tests/unit/test_capability_binding_survives_real_guards.py.
    assert call["skill_name"] == "agent-utilities-expert"


@pytest.mark.asyncio
async def test_execute_capability_runs_resolved_skill_and_returns_bounded_evidence() -> (
    None
):
    engine = MagicMock()
    orchestrator = _orchestrator(engine)
    orchestrator.resolve_capability = MagicMock(  # type: ignore[method-assign]
        return_value={
            "kind": "skill",
            "name": "github-review",
            "id": "resource:skill:github-review",
            "score": 0.9,
            "source": "kg_hybrid",
            "alternatives": [],
        }
    )
    orchestrator.execute_agent = AsyncMock(  # type: ignore[method-assign]
        return_value=json.dumps(
            {
                "output": "evidence-backed result",
                "run_id": "run:0123456789abcdef0123456789abcdef",
                "mermaid": None,
                "run_summary": {
                    "outcome": "ok",
                    "trace_ref": "trace:opaque",
                },
            }
        )
    )
    orchestrator._run_provenance = MagicMock(  # type: ignore[method-assign]
        return_value={
            "trace_ref": "trace:opaque",
            "tool_call_count": 1,
            "tool_calls": [
                {
                    "sequence": 0,
                    "tool_name": "github_pull_request_read",
                    "status": "completed",
                }
            ],
        }
    )

    result = await orchestrator.execute_capability(
        task="Review GitHub PR 458.",
        allowed_tools=["github_pull_request_read"],
    )

    assert result["resolution"]["name"] == "github-review"
    assert result["output"] == "evidence-backed result"
    assert result["provenance"]["tool_call_count"] == 1
    assert result["approval_request"] is None
    orchestrator.execute_agent.assert_awaited_once()
    call = orchestrator.execute_agent.await_args.kwargs
    assert call["agent_name"] == "github-review"
    assert call["include_run_summary"] is True
    assert call["allowed_tools"] == ["github_pull_request_read"]


@pytest.mark.asyncio
async def test_execute_capability_forwards_explicit_pydantic_graph_contract() -> None:
    engine = MagicMock()
    orchestrator = _orchestrator(engine)
    orchestrator.resolve_capability = MagicMock(  # type: ignore[method-assign]
        return_value={
            "kind": "agent",
            "name": "change-review",
            "id": "",
            "score": 1.0,
            "source": "caller",
            "alternatives": [],
        }
    )
    orchestrator.execute_agent = AsyncMock(  # type: ignore[method-assign]
        return_value=json.dumps(
            {
                "output": "validated",
                "run_id": "run:0123456789abcdef0123456789abcdef",
                "run_summary": {"outcome": "ok"},
                "execution_evidence": {
                    "schema_version": "graph-execution-evidence-v1",
                    "topology": "basic",
                    "topology_digest": "sha256:topology",
                    "version_digest": "sha256:version",
                    "runtime_version": "2.21.0",
                    "node_sequence": ["router", "dispatcher", "__end__"],
                    "transitions": [],
                    "checkpoint_ids": [],
                    "resume_supported": False,
                },
            }
        )
    )
    orchestrator._run_provenance = MagicMock(  # type: ignore[method-assign]
        return_value={
            "status": "completed",
            "tool_call_count": 2,
            "tool_calls": [
                {"tool_name": "get_change", "status": "ok"},
                {"tool_name": "list_approvals", "status": "ok"},
            ],
        }
    )

    result = await orchestrator.execute_capability(
        task="Review one synthetic change.",
        skill_name="change-review",
        tool_server="itsm-api",
        execution_mode="pydantic_graph",
        allowed_tools=["get_change", "list_approvals"],
        required_tools=["get_change", "list_approvals"],
    )

    assert result["resolution"]["kind"] == "skill"
    assert result["resolution"]["source"] == "caller_skill"
    assert result["execution_evidence"]["schema_version"] == (
        "graph-execution-evidence-v1"
    )
    assert result["execution_evidence"]["node_sequence"][-1] == "__end__"
    call = orchestrator.execute_agent.await_args.kwargs
    assert call["skill_name"] == "change-review"
    assert call["tool_server"] == "itsm-api"
    assert call["execution_mode"] == "pydantic_graph"
    assert call["allowed_tools"] == ["get_change", "list_approvals"]
    assert call["required_tools"] == ["get_change", "list_approvals"]


def test_required_tools_must_be_inside_allowed_catalog() -> None:
    from agent_utilities.orchestration.execution_contract import (
        missing_required_tools,
        validate_pydantic_graph_contract,
        validate_tool_contract,
    )

    with pytest.raises(PermissionError, match="subset"):
        validate_tool_contract(["get_change"], ["delete_change"])
    with pytest.raises(ValueError, match="skill_name, tool_server, allowed_tools"):
        validate_pydantic_graph_contract(
            "pydantic_graph",
            skill_name="",
            tool_server="",
            allowed_tools=None,
        )
    assert missing_required_tools(
        ["get_change", "list_approvals"],
        ["get_change"],
    ) == ["list_approvals"]


@pytest.mark.asyncio
async def test_graph_orchestrate_public_tool_exposes_pydantic_graph_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_utilities.mcp.kg_server as kg_server
    from agent_utilities.mcp.tools.analysis_tools import register_analysis_tools

    register_analysis_tools(_FakeMCP())
    monkeypatch.setattr(kg_server, "_get_engine", lambda: MagicMock())
    execute = AsyncMock(
        return_value={
            "output": "validated",
            "run_id": "run:0123456789abcdef0123456789abcdef",
            "provenance": {"tool_call_count": 1},
            "execution_evidence": {
                "schema_version": "graph-execution-evidence-v1",
                "topology": "basic",
                "topology_digest": "sha256:topology",
                "version_digest": "sha256:version",
                "runtime_version": "2.21.0",
                "node_sequence": ["router", "dispatcher", "__end__"],
                "transitions": [],
                "checkpoint_ids": [],
                "resume_supported": False,
            },
        }
    )
    monkeypatch.setattr(Orchestrator, "execute_capability", execute)

    raw = await kg_server._execute_tool(
        "graph_orchestrate",
        task="Review a synthetic change.",
        skill_name="change-review",
        tool_server="itsm-api",
        execution_mode="pydantic_graph",
        allowed_tools=["get_change"],
        required_tools=["get_change"],
    )

    response = json.loads(raw)
    assert response["output"] == "validated"
    assert response["execution_evidence"]["schema_version"] == (
        "graph-execution-evidence-v1"
    )
    call = execute.await_args.kwargs
    assert call["skill_name"] == "change-review"
    assert call["tool_server"] == "itsm-api"
    assert call["execution_mode"] == "pydantic_graph"
    assert call["allowed_tools"] == ["get_change"]
    assert call["required_tools"] == ["get_change"]


@pytest.mark.asyncio
async def test_graph_orchestrate_binds_a_tool_and_a_skill_through_the_same_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Capability resolved from ranked intent search binds through
    ``graph_orchestrate`` without the caller branching on ``kind``
    (CONCEPT:AU-KG.retrieval.unified-capability-contract): spreading a tool's
    ``to_binding()`` and a skill's ``to_binding()`` reach the SAME MCP tool
    with the SAME keyword surface."""
    import agent_utilities.mcp.kg_server as kg_server
    from agent_utilities.core.capability_contract import Capability
    from agent_utilities.mcp.tools.analysis_tools import register_analysis_tools

    register_analysis_tools(_FakeMCP())
    monkeypatch.setattr(kg_server, "_get_engine", lambda: MagicMock())
    execute = AsyncMock(
        return_value={
            "output": "ok",
            "run_id": "run:0123456789abcdef0123456789abcdef",
            "provenance": {"tool_call_count": 1},
        }
    )
    monkeypatch.setattr(Orchestrator, "execute_capability", execute)

    tool_capability = Capability(
        kind="tool",
        id="tool_github-mcp_list_issues",
        name="list_issues",
        server="github-mcp",
    )
    skill_capability = Capability(
        kind="skill",
        id="skill_docs-mcp_release-notes-writer",
        name="release-notes-writer",
        server="docs-mcp",
    )

    await kg_server._execute_tool(
        "graph_orchestrate", task="List open issues.", **tool_capability.to_binding()
    )
    tool_call = execute.await_args.kwargs
    assert tool_call["allowed_tools"] == ["list_issues"]
    assert tool_call["tool_server"] == "github-mcp"
    # Was `== ""`, which encoded the defect: execute_capability rejects
    # tool_server without skill_name before any work begins.
    assert tool_call["skill_name"] == "agent-utilities-expert"

    await kg_server._execute_tool(
        "graph_orchestrate",
        task="Draft release notes.",
        **skill_capability.to_binding(),
    )
    skill_call = execute.await_args.kwargs
    assert skill_call["skill_name"] == "release-notes-writer"
    assert skill_call["tool_server"] == "docs-mcp"


@pytest.mark.asyncio
async def test_execute_capability_refuses_ungrounded_tool_required_envelope() -> None:
    """The result wrapper must not turn pseudo tool-call text into gateway success."""
    engine = MagicMock()
    orchestrator = _orchestrator(engine)
    orchestrator.resolve_capability = MagicMock(  # type: ignore[method-assign]
        return_value={
            "kind": "agent",
            "name": "github-mcp",
            "id": "",
            "score": 1.0,
            "source": "caller",
            "alternatives": [],
        }
    )
    orchestrator.execute_agent = AsyncMock(  # type: ignore[method-assign]
        return_value=json.dumps(
            {
                "output": '```json\n{"tool": "repository-manager"}\n```',
                "run_id": "run:0123456789abcdef0123456789abcdef",
                "run_summary": {"outcome": "ok"},
            }
        )
    )
    orchestrator._run_provenance = MagicMock(  # type: ignore[method-assign]
        return_value={"status": "completed", "tool_call_count": 0, "tool_calls": []}
    )

    result = await orchestrator.execute_capability(
        task="List repositories.",
        agent_name="github-mcp",
        allowed_tools=["gith__repos"],
    )

    assert result["resolution"]["name"] == "github-mcp"
    assert result["run_summary"]["outcome"] == "degraded"
    assert result["provenance"]["status"] == "degraded"
    assert "no recorded ToolCall provenance" in result["output"]


@pytest.mark.asyncio
async def test_execute_capability_runs_governed_workflow_and_surfaces_gate_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.knowledge_graph.core import workflow_gate

    engine = MagicMock()
    orchestrator = _orchestrator(engine)
    orchestrator.resolve_capability = MagicMock(  # type: ignore[method-assign]
        return_value={
            "kind": "workflow",
            "name": "release-review",
            "id": "skill_workflow:release-review",
            "score": 0.92,
            "source": "kg_hybrid",
            "alternatives": [],
        }
    )
    monkeypatch.setattr(
        workflow_gate,
        "gate_workflow_execution",
        lambda _engine, _name: {"allowed": True},
    )
    orchestrator.execute_workflow = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "workflow_name": "release-review",
            "run_id": "wf-0123456789abcdef",
            "status": "suspended",
            "step_results": [
                {
                    "node_id": "approve-release",
                    "status": "blocked_on_approval",
                    "error": "awaiting gate satisfaction",
                }
            ],
            "mermaid": "flowchart TD\nA-->B",
        }
    )
    orchestrator._workflow_provenance = MagicMock(  # type: ignore[method-assign]
        return_value={"session_id": "wf-0123456789abcdef", "run_count": 1}
    )

    result = await orchestrator.execute_capability(task="Run the release review.")

    assert result["resolution"]["kind"] == "workflow"
    assert result["approval_request"] == {
        "required": True,
        "approval_id": None,
        "status": "suspended",
        "reason": None,
    }
    assert result["provenance"]["run_count"] == 1
