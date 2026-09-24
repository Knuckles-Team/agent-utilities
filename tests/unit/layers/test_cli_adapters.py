"""CLI adapter delivery: argv, per-run config, credentials by reference, inventory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_utilities.layers.adapters.claude_code import ClaudeCodeHarness
from agent_utilities.layers.adapters.codex import CodexHarness
from agent_utilities.layers.adapters.grok import GrokHarness
from agent_utilities.layers.cli_process import child_environment
from agent_utilities.layers.conformance import TranscriptLauncher
from agent_utilities.layers.contracts import (
    McpEndpoint,
    RunBudget,
    RunSpec,
    RunToolset,
    SkillRef,
)
from agent_utilities.layers.execution import host_workspace_lease, run_to_completion
from agent_utilities.layers.negotiation import HarnessPolicy

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "harness_transcripts"
POLICY = HarnessPolicy(require_context_endpoint=False)
EG = McpEndpoint(name="eg", url="https://eg.example/mcp", bearer_ref="env://EG_TOKEN")
SKILL = SkillRef(name="planted-skill", digest="b" * 64, body="---\nname: x\n---\nhi\n")


class _Secrets:
    values = {"env://EG_TOKEN": "eg-secret", "env://API_KEY": "api-secret"}

    def resolve(self, ref: str) -> str | None:
        return self.values.get(ref)


def _lines(name: str) -> list[str]:
    return (FIXTURES / name).read_text().splitlines()


def _init_with(**fields) -> list[str]:
    lines = _lines("claude_code_success.jsonl")
    init = {**json.loads(lines[0]), **fields}
    return [json.dumps(init), *lines[1:]]


async def _run(harness, spec: RunSpec, root: Path):
    lease = host_workspace_lease(str(root), spec.run_id)
    return lease, await run_to_completion(harness, spec, policy=POLICY, lease=lease)


async def test_claude_delivers_mcp_skills_fence_and_credentials(tmp_path) -> None:
    launcher = TranscriptLauncher(
        _init_with(mcp_servers=[{"name": "eg", "status": "connected"}])
    )
    spec = RunSpec(
        run_id="c1",
        task="do it",
        agent_ref="a",
        toolset=RunToolset(
            context_endpoint=EG,
            allowed_tools=("Bash(echo:*)",),
            required_tools=("Bash",),
            skills=(SKILL,),
        ),
        budget=RunBudget(max_cost_usd=0.5),
        account_mode="api_key",
        account_ref="env://API_KEY",
    )
    harness = ClaudeCodeHarness(launcher=launcher, credentials=_Secrets())
    lease, outcome = await _run(harness, spec, tmp_path)
    assert outcome.result.status == "succeeded"
    assert outcome.result.usage.quality == "measured"
    (launch,) = launcher.launches
    argv = launch["argv"]
    for flag in ("--strict-mcp-config", "--verbose", "--no-session-persistence"):
        assert flag in argv
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert argv[argv.index("--setting-sources") + 1] == "project"
    assert argv[argv.index("--allowedTools") + 1] == "Bash(echo:*)"
    assert argv[argv.index("--max-budget-usd") + 1] == "0.5000"
    assert launch["stdin"] == "do it"
    assert launch["env"]["ANTHROPIC_API_KEY"] == "api-secret"
    assert launch["env"]["AU_MCP_TOKEN_0"] == "eg-secret"
    workspace = Path(lease.workspace)
    mcp = (workspace / ".au-run" / "mcp.json").read_text()
    assert "eg-secret" not in mcp and "${AU_MCP_TOKEN_0}" in mcp
    skill = workspace / ".claude" / "skills" / "planted-skill" / "SKILL.md"
    assert skill.read_text() == SKILL.body
    settings = json.loads((workspace / ".claude" / "settings.json").read_text())
    assert settings["permissions"]["deny"]
    assert "bypassPermissions" not in json.dumps(settings)


@pytest.mark.parametrize(
    ("fields", "toolset", "missing"),
    [
        ({}, RunToolset(required_tools=("mcp__eg__search",)), "tool:mcp__eg__search"),
        (
            {"mcp_servers": [{"name": "eg", "status": "failed"}]},
            RunToolset(context_endpoint=EG),
            "mcp:eg",
        ),
        ({"skills": []}, RunToolset(skills=(SKILL,)), "skill:planted-skill"),
    ],
)
async def test_claude_fails_loudly_on_a_startup_inventory_gap(
    fields, toolset, missing, tmp_path
) -> None:
    launcher = TranscriptLauncher(_init_with(**fields))
    harness = ClaudeCodeHarness(launcher=launcher, credentials=_Secrets())
    spec = RunSpec(run_id="c2", task="t", agent_ref="a", toolset=toolset)
    _lease, outcome = await _run(harness, spec, tmp_path)
    assert outcome.result.status == "failed"
    assert outcome.result.error_kind == "HarnessToolInventoryMismatch"
    assert missing in outcome.result.error
    assert launcher.processes[0].terminated
    assert not [event for event in outcome.trace if event.kind == "tool_call"]


async def test_codex_passes_mcp_by_override_and_measures_usage(tmp_path) -> None:
    launcher = TranscriptLauncher(_lines("codex_success.jsonl"))
    spec = RunSpec(
        run_id="x1",
        task="t",
        agent_ref="a",
        toolset=RunToolset(context_endpoint=EG),
        account_mode="api_key",
        account_ref="env://API_KEY",
    )
    harness = CodexHarness(launcher=launcher, credentials=_Secrets())
    _lease, outcome = await _run(harness, spec, tmp_path)
    argv = launcher.launches[0]["argv"]
    assert 'mcp_servers.eg.url="https://eg.example/mcp"' in argv
    assert 'mcp_servers.eg.bearer_token_env_var="AU_MCP_TOKEN_0"' in argv
    assert argv[-1] == "-" and "--ephemeral" in argv and "--ignore-user-config" in argv
    assert launcher.launches[0]["env"]["CODEX_API_KEY"] == "api-secret"
    assert outcome.result.output == "DONE"
    assert outcome.result.usage.input_tokens == 30256
    assert outcome.result.provider_session == "01a0cbdc-d145-7823-94fc-f4903d2faab6"


async def test_grok_prompt_never_reaches_the_command_line(tmp_path) -> None:
    launcher = TranscriptLauncher(_lines("grok_success.jsonl"))
    spec = RunSpec(run_id="g1", task="secret task text", agent_ref="a")
    lease, outcome = await _run(GrokHarness(launcher=launcher), spec, tmp_path)
    argv = launcher.launches[0]["argv"]
    assert "secret task text" not in " ".join(argv)
    prompt = Path(argv[argv.index("--prompt-file") + 1])
    assert prompt.read_text() == "secret task text"
    assert str(prompt).startswith(lease.workspace)
    assert outcome.result.output == "DONE"
    assert outcome.result.usage.quality == "unavailable"


async def test_failure_after_tool_use_with_effects_is_uncertain(tmp_path) -> None:
    lines = _lines("codex_success.jsonl")[:5]
    launcher = TranscriptLauncher(lines, exit_code=1)
    spec = RunSpec(run_id="x2", task="t", agent_ref="a", side_effects="idempotent")
    _lease, outcome = await _run(CodexHarness(launcher=launcher), spec, tmp_path)
    assert outcome.result.status == "outcome_uncertain"
    assert outcome.result.trace == "incomplete"


async def test_cli_run_without_a_leased_workspace_is_refused(tmp_path) -> None:
    harness = CodexHarness(launcher=TranscriptLauncher(_lines("codex_success.jsonl")))
    spec = RunSpec(run_id="x3", task="t", agent_ref="a")
    outcome = await run_to_completion(harness, spec, policy=POLICY, lease=None)
    assert outcome.result.status == "failed"
    assert outcome.result.error_kind == "SandboxBoundaryError"


def test_child_environment_is_an_allowlist(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    env = child_environment({"AU_MCP_TOKEN_0": "t"})
    assert env["HOME"] == str(tmp_path)
    assert env["AU_MCP_TOKEN_0"] == "t"
    assert "OPENAI_API_KEY" not in env
