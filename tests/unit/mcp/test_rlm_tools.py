"""Tests for the graph_rlm MCP tool's `evolve_prompt` action (CONCEPT:AU-ORCH.
optimization.optimize-skill-prompt-gepa).

Live-path proof that ``graph_rlm(action="evolve_prompt")`` actually drives
:class:`~agent_utilities.rlm.gepa.GEPAOptimizer` (which enables DW-GRPO dynamic
reward weighting, :mod:`~agent_utilities.rlm.dynamic_reward`, by default) rather
than merely importing it. Mirrors ``tests/unit/core/test_rlm_gepa.py``'s
LLM-mocking pattern (no live model calls) and the ``_CollectingMCP`` +
``kg_server.REGISTERED_TOOLS`` pattern used across the other MCP tool-surface
tests (e.g. ``test_audit_tools.py``).
"""

from __future__ import annotations

import json
import unittest.mock

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools.rlm_tools import register_rlm_tools


class _CollectingMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name, description="", tags=None):  # noqa: ANN001
        def _deco(fn):
            self.tools[name] = fn
            return fn

        return _deco


def _register() -> object:
    mcp = _CollectingMCP()
    register_rlm_tools(mcp)
    return mcp.tools["graph_rlm"]


def test_registered_on_graphos_tool_table():
    tool = _register()
    assert kg_server.REGISTERED_TOOLS.get("graph_rlm") is tool
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_rlm") == "/graph/rlm"


class _MockMutateResponse:
    output = (
        '{"rationale": "Tightened the answer format.", '
        '"mutated_prompt": "Answer precisely."}'
    )


async def _mock_mutate_run(self_agent, prompt, **kwargs):  # noqa: ANN001, ARG001
    return _MockMutateResponse()


async def _mock_create_or_merge(node):  # noqa: ANN001, ARG001
    return {"status": "merged"}


@pytest.mark.asyncio
async def test_evolve_prompt_runs_gepa_with_dynamic_reward_weighting_on():
    """Live-path: the actual MCP action drives GEPAOptimizer.optimize() end to
    end (mocked LLM boundaries only), and the DW-GRPO weighter is active — the
    real wiring for both agent_utilities.rlm.gepa and .dynamic_reward."""
    tool = _register()

    async def mock_harness_run(self_harness, **inputs):  # noqa: ANN001
        query = str(inputs.get("query") or "")
        # The built-in default dataset's first row expects "Paris".
        answer = "Paris" if "France" in query else "unknown"
        return self_harness.signature(query=query, response=answer)

    with (
        unittest.mock.patch("pydantic_ai.Agent.run", new=_mock_mutate_run),
        unittest.mock.patch(
            "agent_utilities.rlm.predict_rlm.PredictRLM.run", new=mock_harness_run
        ),
        unittest.mock.patch(
            "agent_utilities.rlm.gepa.create_or_merge_node",
            new=_mock_create_or_merge,
        ),
    ):
        raw = await tool(
            action="evolve_prompt",
            task="",
            data_json=json.dumps({"iterations": 1, "batch_size": 2}),
        )

    out = json.loads(raw)
    assert out["ok"] is True
    assert out["winning_prompt"]
    assert "accuracy" in out["scores"]
    # DW-GRPO reward weights present + normalized — proves dynamic_reward.py ran
    # as GEPAOptimizer's default, not just that gepa.py imported successfully.
    assert set(out["reward_weights"]) == {"accuracy"}
    assert abs(sum(out["reward_weights"].values()) - 1.0) < 1e-9
    assert out["frontier_size"] >= 1


@pytest.mark.asyncio
async def test_evolve_prompt_rejects_empty_dataset():
    tool = _register()
    raw = await tool(action="evolve_prompt", data_json=json.dumps({"dataset": []}))
    # public_error_text degrades the raised ValueError to a safe structured
    # payload, never a raw traceback.
    out = json.loads(raw)
    assert "error" in out


def test_unknown_action():
    tool = _register()
    import asyncio

    out = asyncio.run(tool(action="bogus"))
    assert "Unknown graph_rlm action" in out
