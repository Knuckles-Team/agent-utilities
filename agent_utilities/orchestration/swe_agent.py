"""CONCEPT:ORCH-1.47 — KG-grounded software-engineering agent (CodeAct equivalent).

The loop *is* Pydantic-AI's own tool-calling loop: a focused :class:`pydantic_ai.Agent` bound to
the code-intelligence (graph) tools (KG-2.65) and the SWE workspace (action) tools that execute
inside ``deps.workspace`` (OS-5.33). The agent grounds itself with graph queries, edits files,
runs tests, and iterates until the change verifies — and because the workspace mirrors every
action to the KG (KG-2.64), the whole trajectory is provenance the golden loop (AHE-3.23) can
learn from.

``build_swe_agent`` is model-injectable so tests can drive the full loop deterministically with
a Pydantic-AI ``FunctionModel``/``TestModel`` (no live LLM required).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent

from agent_utilities.core.model_factory import create_model
from agent_utilities.models import AgentDeps
from agent_utilities.runtime.events import CmdRunAction

from ..agent.capability_resolver import resolve_capabilities
from ..agent.swe_prompts import SWE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# The SWE agent's tools are resolved from bounded capability intents (ECO-4.45), never a
# hard-coded name list — the same contract its prompts/swe_engineer.json blueprint declares.
SWE_CAPABILITIES = ["software-engineering", "web-browsing"]


@dataclass
class SweResult:
    """Outcome of one SWE task run."""

    output: str
    tool_calls: int
    patch: str  # unified diff of the changes the agent made (empty if not a git repo)


def build_swe_agent(
    model: Any | None = None, *, extra_tools: list[Any] | None = None
) -> Agent:
    """Assemble the SWE agent: graph tools + workspace tools + graph-first system prompt.

    ``model`` may be a Pydantic-AI model object (or ``None`` to resolve the configured default).
    """
    mdl = model if model is not None else create_model()
    tools = [*resolve_capabilities(SWE_CAPABILITIES), *(extra_tools or [])]
    return Agent(
        model=mdl,
        deps_type=AgentDeps,
        system_prompt=SWE_SYSTEM_PROMPT,
        tools=tools,
        name="swe-agent",
    )


async def run_swe_task(
    task: str,
    deps: AgentDeps,
    *,
    model: Any | None = None,
    agent: Agent | None = None,
) -> SweResult:
    """Run the SWE agent on ``task`` inside ``deps.workspace`` and return the result + patch.

    The workspace must already be started and (for a real task) seeded with the repo. The patch
    is extracted with ``git diff`` inside the workspace — the fidelity-preserving way to capture
    exactly what the agent changed (vs reconstructing from edit observations).
    """
    agent = agent or build_swe_agent(model)
    result = await agent.run(task, deps=deps)
    tool_calls = _count_tool_calls(result)
    patch = await _extract_patch(getattr(deps, "workspace", None))
    return SweResult(output=str(result.output), tool_calls=tool_calls, patch=patch)


async def _extract_patch(workspace: Any | None) -> str:
    if workspace is None:
        return ""
    # Exclude bytecode/caches a test run leaves behind so the patch is just the source change.
    exclude = "':(exclude)**/__pycache__/**' ':(exclude)*.pyc' ':(exclude).au_*'"
    # Prefer diffing against the harness's pre-solve tag (au_base) so the patch captures the
    # agent's change whether it left it uncommitted OR self-committed; fall back to staged/working.
    obs = await workspace.act(
        CmdRunAction(
            command=(
                f"git add -A >/dev/null 2>&1; "
                f"git diff au_base -- . {exclude} 2>/dev/null "
                f"|| git diff --cached -- . {exclude} 2>/dev/null "
                f"|| git diff -- . {exclude} 2>/dev/null || true"
            )
        )
    )
    return getattr(obs, "stdout", "")


def _count_tool_calls(result: Any) -> int:
    """Best-effort count of tool calls across the run's messages."""
    try:
        messages = result.all_messages()
    except Exception:  # noqa: BLE001
        return 0
    count = 0
    for msg in messages:
        for part in getattr(msg, "parts", []) or []:
            if type(part).__name__ == "ToolCallPart":
                count += 1
    return count
