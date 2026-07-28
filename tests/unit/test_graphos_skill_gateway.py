"""Live-path contract for KG-resolved local-vLLM delegation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_utilities.orchestration.manager import Orchestrator


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
                "output": "```json\n{\"tool\": \"repository-manager\"}\n```",
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
