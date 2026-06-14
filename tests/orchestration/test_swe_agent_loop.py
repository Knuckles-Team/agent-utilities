"""CONCEPT:ORCH-1.47 — KG-grounded SWE agent loop, driven deterministically (no live LLM).

A scripted ``FunctionModel`` plays the role of the model: it grounds (a graph tool), edits a
file, runs tests, then finishes. The assertions prove the loop actually acted in the workspace
(the edit landed, tests ran) — i.e. the edit→run→test loop is wired end-to-end.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_utilities.models import AgentDeps
from agent_utilities.orchestration.swe_agent import build_swe_agent, run_swe_task
from agent_utilities.runtime import DevWorkspace, LocalWorkspace
from agent_utilities.tools.swe_workspace_tools import edit_file, run_command, run_tests

# ── direct tool tests (deterministic, local workspace) ────────────────────────


class _Ctx:
    """Minimal RunContext stand-in carrying deps (tools only read ctx.deps)."""

    def __init__(self, deps):
        self.deps = deps


async def test_swe_workspace_tools_act_in_workspace():
    ws = DevWorkspace(LocalWorkspace(), run_id="t")
    await ws.start()
    deps = AgentDeps(workspace=ws)
    ctx = _Ctx(deps)
    try:
        await run_command(ctx, "printf 'a = 1\\n' > m.py")
        out = await edit_file(ctx, "m.py", "a = 1", "a = 2")
        assert "Applied 1 replacement" in out
        cat = await run_command(ctx, "cat m.py")
        assert "a = 2" in cat
    finally:
        await ws.stop()


async def test_workspace_tool_without_workspace_is_graceful():
    ctx = _Ctx(AgentDeps(workspace=None))
    out = await run_tests(ctx, selector="x")
    assert "No developer workspace" in out


# ── full loop, scripted model ─────────────────────────────────────────────────


def _script():
    """Return a FunctionModel that grounds, edits, tests, then finishes."""
    steps = iter(
        [
            ToolCallPart(tool_name="find_definition", args={"symbol": "add"}),
            ToolCallPart(
                tool_name="edit_file",
                args={"path": "calc.py", "old": "return a - b", "new": "return a + b"},
            ),
            ToolCallPart(tool_name="run_tests", args={"selector": "test_calc.py"}),
            TextPart(content="Fixed the sign bug in add(); tests pass."),
        ]
    )

    def model_fn(messages: list, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[next(steps)])

    return FunctionModel(model_fn)


async def test_full_swe_loop_edits_and_runs_tests():
    ws = DevWorkspace(LocalWorkspace(), run_id="loop")
    await ws.start()
    deps = AgentDeps(workspace=ws)
    # seed a buggy repo
    await run_command(_Ctx(deps), "git init -q . 2>/dev/null || true")
    await run_command(
        _Ctx(deps),
        "printf 'def add(a, b):\\n    return a - b\\n' > calc.py",
    )
    await run_command(
        _Ctx(deps),
        "printf 'from calc import add\\n\\ndef test_add():\\n    assert add(2, 3) == 5\\n' > test_calc.py",
    )
    try:
        agent = build_swe_agent(model=_script())
        result = await run_swe_task("Fix add() so tests pass.", deps, agent=agent)
        # the agent's edit landed
        cat = await run_command(_Ctx(deps), "cat calc.py")
        assert "a + b" in cat
        assert "Fixed the sign bug" in result.output
        assert result.tool_calls >= 3
    finally:
        await ws.stop()
