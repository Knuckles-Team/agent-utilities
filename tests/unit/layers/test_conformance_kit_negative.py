"""The conformance kit rejects known-bad adapters (a gate must catch bad input)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.layers.adapters.claude_code import ClaudeCodeHarness
from agent_utilities.layers.adapters.codex import DESCRIPTOR as CODEX
from agent_utilities.layers.adapters.codex import CodexHarness
from agent_utilities.layers.conformance import (
    ConformanceFailure,
    ConformanceTarget,
    Rig,
    TranscriptLauncher,
    run_check,
)
from agent_utilities.layers.contracts import (
    HarnessDescriptor,
    RunEvent,
    RunSpec,
    RunToolset,
)
from agent_utilities.layers.execution import host_workspace_lease
from agent_utilities.layers.negotiation import HarnessPolicy
from agent_utilities.layers.ports import RunHandle

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "harness_transcripts"
POLICY = HarnessPolicy(require_context_endpoint=False)


class _ClaimsFinalOutput(CodexHarness):
    def describe(self) -> HarnessDescriptor:
        return CODEX.model_copy(update={"fidelity": "final-output"})


class _ApiKeyOnly(CodexHarness):
    def describe(self) -> HarnessDescriptor:
        return CODEX.model_copy(update={"account_modes": frozenset({"api_key"})})


class _ReorderedTrace(CodexHarness):
    def trace(self, handle: RunHandle) -> tuple[RunEvent, ...]:
        return tuple(reversed(super().trace(handle)))


def _target(
    harness_type: type[CodexHarness], lines: list[str], root: Path
) -> ConformanceTarget:
    def build(run_id: str) -> Rig:
        return Rig(
            harness=harness_type(launcher=TranscriptLauncher(lines)),
            spec=RunSpec(run_id=run_id, task="t", agent_ref="a"),
            lease=host_workspace_lease(str(root), run_id),
        )

    return ConformanceTarget(
        name="codex",
        evidence="recorded-live-transcript",
        policy=POLICY,
        tool_name="command_execution",
        scenarios={"success": build},
    )


def _codex_lines() -> list[str]:
    return (FIXTURES / "codex_success.jsonl").read_text().splitlines()


@pytest.mark.parametrize(
    ("harness_type", "check"),
    [
        (_ClaimsFinalOutput, "tool_call_surfacing"),
        (_ApiKeyOnly, "descriptor_honesty"),
        (_ReorderedTrace, "streaming_events"),
    ],
)
async def test_kit_rejects_a_dishonest_adapter(
    harness_type: type[CodexHarness], check: str, tmp_path: Path
) -> None:
    with pytest.raises(ConformanceFailure):
        await run_check(_target(harness_type, _codex_lines(), tmp_path), check)


async def test_kit_rejects_a_run_without_startup_inventory(tmp_path: Path) -> None:
    lines = (FIXTURES / "claude_code_success.jsonl").read_text().splitlines()[1:]

    def build(run_id: str) -> Rig:
        return Rig(
            harness=ClaudeCodeHarness(launcher=TranscriptLauncher(lines)),
            spec=RunSpec(
                run_id=run_id,
                task="t",
                agent_ref="a",
                toolset=RunToolset(required_tools=("Bash",)),
            ),
            lease=host_workspace_lease(str(tmp_path), run_id),
        )

    target = ConformanceTarget(
        name="claude-code",
        evidence="recorded-live-transcript",
        policy=POLICY,
        tool_name="Bash",
        scenarios={"success": build},
    )
    with pytest.raises(ConformanceFailure, match="status failed"):
        await run_check(target, "session_lifecycle")
