"""AU-HARNESS-R007..R010: the L4 harness port, its two adapters, selection and L5."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from agent_utilities.layers import harness_record
from agent_utilities.layers.harness_cli import (
    CLAUDE_PINNED_FLAGS,
    ClaudeCodeHarness,
    graphos_mcp_config,
    write_graphos_mcp_config,
)
from agent_utilities.layers.harness_native import NativeHarness
from agent_utilities.layers.harness_node import run_agent_spec
from agent_utilities.layers.harness_port import (
    NATIVE_HARNESS,
    DiffStat,
    HarnessPort,
    HarnessRequest,
    RunOutcome,
    UsageReport,
)
from agent_utilities.layers.harness_process import parse_shortstat, read_diff_stat
from agent_utilities.layers.harness_registry import (
    HarnessRegistry,
    UnknownHarness,
    harness_name,
)
from agent_utilities.models.execution_manifest import AgentSpec
from tests.unit.layers.harness_fakes import read_record, write_fake_claude

_STATUSES = {"completed", "degraded", "failed", "timeout", "refused"}


def _envelope(output: str, outcome: str = "ok", **extra: Any) -> str:
    summary = {"outcome": outcome, "trace_ref": "trace:run:abc", **extra}
    return json.dumps(
        {
            "output": output,
            "run_id": "run:abc",
            "run_summary": summary,
            "provenance_recorded": True,
        }
    )


class _Runner:
    def __init__(self, raw: str, delay: float = 0.0) -> None:
        self.raw = raw
        self.delay = delay
        self.kwargs: dict[str, Any] = {}

    async def __call__(self, **kwargs: Any) -> str:
        self.kwargs = kwargs
        await asyncio.sleep(self.delay)
        return self.raw


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "worktree"
    ws.mkdir()
    return ws


@pytest.fixture
def mcp_config(tmp_path: Path) -> Path:
    return write_graphos_mcp_config(
        tmp_path / "mcp.json", "https://graph-os.example/mcp", token_env="GRAPHOS_TOKEN"
    )


def _claude(tmp_path: Path, mcp_config: Path, mode: str = "ok", **kw: Any):
    executable, record = write_fake_claude(tmp_path)

    async def no_diff(_: Path) -> DiffStat | None:
        return None

    harness = ClaudeCodeHarness(
        mcp_config=mcp_config,
        binary=str(executable),
        env={"FAKE_MODE": mode},
        diff_reader=kw.pop("diff_reader", no_diff),
        **kw,
    )
    return harness, record


async def _assert_conforms(port: Any, request: HarnessRequest) -> RunOutcome:
    """The shared protocol conformance check every adapter passes."""
    assert isinstance(port, HarnessPort)
    assert isinstance(port.name, str) and port.name
    outcome = await port.run(request)
    assert isinstance(outcome, RunOutcome)
    assert outcome.harness == port.name
    assert outcome.run_id == request.run_id
    assert outcome.agent_name == request.agent_name
    assert outcome.status in _STATUSES
    assert outcome.duration_ms >= 0
    return outcome


# -- protocol conformance ---------------------------------------------------


async def test_native_adapter_conforms() -> None:
    runner = _Runner(_envelope("hello"))
    request = HarnessRequest(agent_name="a", task="t", run_id="run:abc")
    outcome = await _assert_conforms(NativeHarness(runner=runner), request)
    assert outcome.status == "completed"


async def test_claude_adapter_conforms(
    tmp_path: Path, workspace: Path, mcp_config: Path
) -> None:
    harness, _ = _claude(tmp_path, mcp_config)
    request = HarnessRequest(agent_name="a", task="fix it", workspace=workspace)
    outcome = await _assert_conforms(harness, request)
    assert outcome.status == "completed"


def test_outcome_is_immutable_and_strict() -> None:
    outcome = RunOutcome(run_id="r", harness="h", agent_name="a", status="completed")
    with pytest.raises(ValueError):
        outcome.status = "failed"  # type: ignore[misc]
    with pytest.raises(ValueError):
        RunOutcome(run_id="r", harness="h", agent_name="a", status="nope")  # type: ignore[arg-type]


# -- native adapter ---------------------------------------------------------


async def test_native_keeps_run_agent_contract() -> None:
    runner = _Runner(_envelope('{"answer": 42}'))
    request = HarnessRequest(
        agent_name="a",
        task="t",
        run_id="run:abc",
        response_format="json",
        allowed_tools=("x",),
    )
    outcome = await NativeHarness(runner=runner, engine="E").run(request)
    assert runner.kwargs["include_run_summary"] is True
    assert runner.kwargs["run_id"] == "run:abc"
    assert runner.kwargs["engine"] == "E"
    assert runner.kwargs["response_format"] == "json"
    assert runner.kwargs["allowed_tools"] == ["x"]
    assert outcome.structured_output == {"answer": 42}
    assert outcome.trace_ref == "trace:run:abc"
    assert outcome.recorded is True


async def test_native_maps_degraded_failure() -> None:
    failure = {"raw": "boom", "translated": "tool server down"}
    runner = _Runner(_envelope("partial", outcome="degraded", failure=failure))
    request = HarnessRequest(agent_name="a", task="t")
    outcome = await NativeHarness(runner=runner).run(request)
    assert outcome.status == "degraded"
    assert outcome.error == "tool server down"


async def test_native_plain_string_result() -> None:
    outcome = await NativeHarness(runner=_Runner("bare text")).run(
        HarnessRequest(agent_name="a", task="t")
    )
    assert outcome.status == "completed"
    assert outcome.final_text == "bare text"
    assert outcome.structured_output is None
    assert outcome.recorded is False


async def test_native_timeout_is_typed() -> None:
    runner = _Runner(_envelope("late"), delay=5.0)
    request = HarnessRequest(agent_name="a", task="t", timeout_s=0.05)
    outcome = await NativeHarness(runner=runner).run(request)
    assert outcome.status == "timeout"
    assert outcome.error


# -- claude-code CLI adapter (fake executable; no real CLI calls) ------------


async def test_claude_pinned_argv_cwd_and_stdin(
    tmp_path: Path, workspace: Path, mcp_config: Path
) -> None:
    harness, record = _claude(tmp_path, mcp_config, max_budget_usd=1.5)
    request = HarnessRequest(
        agent_name="coder",
        task="add a test",
        workspace=workspace,
        model="opus",
        allowed_tools=("mcp__graph-os", "Edit"),
    )
    await harness.run(request)
    seen = read_record(record)
    argv = seen["argv"]
    assert tuple(argv[: len(CLAUDE_PINNED_FLAGS)]) == CLAUDE_PINNED_FLAGS
    assert argv[-2:] == ["--mcp-config", str(mcp_config)]
    assert ["--permission-mode", "dontAsk"] == argv[6:8]
    assert "--model" in argv and "opus" in argv
    assert "--max-budget-usd" in argv and "1.5" in argv
    assert "mcp__graph-os,Edit" in argv
    assert "add a test" not in argv  # the prompt goes on stdin only
    assert seen["prompt"] == "add a test"
    assert Path(seen["cwd"]).resolve() == workspace.resolve()


async def test_claude_typed_outcome(
    tmp_path: Path, workspace: Path, mcp_config: Path
) -> None:
    harness, _ = _claude(tmp_path, mcp_config)
    outcome = await harness.run(
        HarnessRequest(agent_name="a", task="go", workspace=workspace)
    )
    assert outcome.exit_code == 0
    assert outcome.final_text == "done: go"
    assert outcome.transcript_ref == "claude-code:session:sess-123"
    assert outcome.model == "fake-model-1"
    assert outcome.usage == UsageReport(
        input_tokens=11,
        output_tokens=7,
        cache_read_input_tokens=3,
        cache_creation_input_tokens=2,
        cost_usd=0.0421,
    )


async def test_claude_child_env_is_allowlisted(
    tmp_path: Path, workspace: Path, mcp_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PARENT_SECRET_TOKEN", "x")
    harness, record = _claude(tmp_path, mcp_config)
    await harness.run(HarnessRequest(agent_name="a", task="go", workspace=workspace))
    env = read_record(record)["env"]
    assert "PARENT_SECRET_TOKEN" not in env
    assert "FAKE_MODE" in env


async def test_claude_error_result_is_failed(
    tmp_path: Path, workspace: Path, mcp_config: Path
) -> None:
    harness, _ = _claude(tmp_path, mcp_config, mode="error")
    outcome = await harness.run(
        HarnessRequest(agent_name="a", task="go", workspace=workspace)
    )
    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert outcome.error == "error_during_execution"


async def test_claude_unparseable_output_is_failed(
    tmp_path: Path, workspace: Path, mcp_config: Path
) -> None:
    harness, _ = _claude(tmp_path, mcp_config, mode="garbage")
    outcome = await harness.run(
        HarnessRequest(agent_name="a", task="go", workspace=workspace)
    )
    assert outcome.status == "failed"
    assert outcome.exit_code == 1
    assert outcome.usage is None
    assert outcome.error and "no JSON result" in outcome.error


async def test_claude_timeout_kills_the_child(
    tmp_path: Path, workspace: Path, mcp_config: Path
) -> None:
    harness, _ = _claude(tmp_path, mcp_config, mode="sleep")
    start = time.monotonic()
    outcome = await harness.run(
        HarnessRequest(agent_name="a", task="go", workspace=workspace, timeout_s=0.5)
    )
    assert time.monotonic() - start < 10
    assert outcome.status == "timeout"
    assert outcome.exit_code is None


@pytest.mark.parametrize("case", ["no-workspace", "no-config", "no-binary"])
async def test_claude_refuses_before_launch(
    case: str, tmp_path: Path, workspace: Path, mcp_config: Path
) -> None:
    harness, record = _claude(tmp_path, mcp_config)
    request = HarnessRequest(agent_name="a", task="go", workspace=workspace)
    if case == "no-workspace":
        request = HarnessRequest(agent_name="a", task="go")
    elif case == "no-config":
        harness = ClaudeCodeHarness(
            mcp_config=tmp_path / "absent.json", binary=harness._binary
        )
    else:
        harness = ClaudeCodeHarness(
            mcp_config=mcp_config, binary="definitely-not-a-claude-binary"
        )
    outcome = await harness.run(request)
    assert outcome.status == "refused"
    assert outcome.error
    assert not record.exists()


async def test_claude_reports_worktree_diff(
    tmp_path: Path, workspace: Path, mcp_config: Path
) -> None:
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], cwd=workspace, check=True)
    (workspace / "a.txt").write_text("one\n")
    subprocess.run([*git, "add", "a.txt"], cwd=workspace, check=True)
    subprocess.run([*git, "commit", "-qm", "base"], cwd=workspace, check=True)
    (workspace / "a.txt").write_text("one\ntwo\nthree\n")
    harness, _ = _claude(tmp_path, mcp_config, diff_reader=read_diff_stat)
    outcome = await harness.run(
        HarnessRequest(agent_name="a", task="go", workspace=workspace)
    )
    assert outcome.diff_stat == DiffStat(files_changed=1, insertions=2, deletions=0)


async def test_diff_stat_outside_git_is_none(workspace: Path) -> None:
    assert await read_diff_stat(workspace) is None


def test_parse_shortstat() -> None:
    text = " 3 files changed, 10 insertions(+), 4 deletions(-)\n"
    assert parse_shortstat(text) == DiffStat(
        files_changed=3, insertions=10, deletions=4
    )
    assert parse_shortstat(" 1 file changed, 1 deletion(-)") == DiffStat(
        files_changed=1, deletions=1
    )
    assert parse_shortstat("") == DiffStat()


def test_graphos_mcp_config_never_holds_a_token() -> None:
    config = graphos_mcp_config("https://g.example/mcp", token_env="GRAPHOS_TOKEN")
    server = config["mcpServers"]["graph-os"]
    assert server == {
        "type": "http",
        "url": "https://g.example/mcp",
        "headers": {"Authorization": "Bearer ${GRAPHOS_TOKEN}"},
    }


# -- selection --------------------------------------------------------------


class _FakePort:
    def __init__(self, name: str = "fake", status: str = "completed") -> None:
        self._name = name
        self._status = status
        self.requests: list[HarnessRequest] = []

    @property
    def name(self) -> str:
        return self._name

    async def run(self, request: HarnessRequest) -> RunOutcome:
        self.requests.append(request)
        return RunOutcome(
            run_id=request.run_id,
            harness=self._name,
            agent_name=request.agent_name,
            status=self._status,  # type: ignore[arg-type]
            final_text="ok",
            usage=UsageReport(input_tokens=5, output_tokens=1, cost_usd=0.01),
        )


def test_selection_defaults_to_native() -> None:
    registry = HarnessRegistry(native=_FakePort(NATIVE_HARNESS))
    assert harness_name(AgentSpec(agent_id="n")) == NATIVE_HARNESS
    assert harness_name(object()) == NATIVE_HARNESS
    assert registry.select(AgentSpec(agent_id="n")).name == NATIVE_HARNESS
    assert isinstance(HarnessRegistry().get(NATIVE_HARNESS), NativeHarness)


def test_selection_by_node_field_and_unknown_is_an_error() -> None:
    registry = HarnessRegistry(native=_FakePort(NATIVE_HARNESS))
    registry.register(_FakePort("claude-code"))
    assert registry.names() == ["claude-code", NATIVE_HARNESS]
    spec = AgentSpec(agent_id="n", harness="claude-code")
    assert registry.select(spec).name == "claude-code"
    with pytest.raises(UnknownHarness):
        registry.select(AgentSpec(agent_id="n", harness="langgraph"))
    with pytest.raises(TypeError):
        registry.register(object())  # type: ignore[arg-type]


# -- L5 recording -----------------------------------------------------------


@pytest.fixture
def l5_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    calls: dict[str, list] = {"trace": [], "usage": []}

    async def fake_trace(engine, run_id, agent_name, task, /, **kwargs):
        calls["trace"].append((engine, run_id, agent_name, task, kwargs))
        return True

    class _Recorder:
        def record_run(self, **kwargs):
            calls["usage"].append(kwargs)
            return True

    from agent_utilities.orchestration import agent_runner
    from agent_utilities.usage import recorder

    monkeypatch.setattr(agent_runner, "_record_execution_trace_ordered", fake_trace)
    monkeypatch.setattr(recorder, "get_usage_recorder", lambda: _Recorder())
    return calls


async def test_cli_outcome_feeds_existing_trace_and_usage(l5_calls) -> None:
    request = HarnessRequest(agent_name="coder", task="go", run_id="run:1")
    outcome = RunOutcome(
        run_id="run:1",
        harness="claude-code",
        agent_name="coder",
        status="timeout",
        final_text="x",
        model="m",
        usage=UsageReport(input_tokens=3, output_tokens=4, cost_usd=0.5),
        error="slow",
    )
    assert await harness_record.record_outcome("E", request, outcome) is True
    engine, run_id, agent, task, kwargs = l5_calls["trace"][0]
    assert (engine, run_id, agent, task) == ("E", "run:1", "coder", "go")
    assert kwargs["execution_mode"] == "harness:claude-code"
    assert kwargs["status"] == "failed"
    assert kwargs["error"] == "slow"
    assert l5_calls["usage"][0]["token_usage"]["output_tokens"] == 4
    assert l5_calls["usage"][0]["run_id"] == "run:1"


async def test_native_outcome_is_not_written_twice(l5_calls) -> None:
    request = HarnessRequest(agent_name="a", task="t")
    outcome = RunOutcome(
        run_id=request.run_id,
        harness=NATIVE_HARNESS,
        agent_name="a",
        status="completed",
        recorded=True,
    )
    assert await harness_record.record_outcome(None, request, outcome) is True
    assert l5_calls == {"trace": [], "usage": []}


async def test_node_runs_through_selected_harness_and_records(l5_calls) -> None:
    registry = HarnessRegistry(native=_FakePort(NATIVE_HARNESS))
    port = _FakePort("claude-code")
    registry.register(port)
    spec = AgentSpec(
        agent_id="coder",
        role="dev",
        harness="claude-code",
        harness_workspace="/srv/wt",
        tools=["mcp__graph-os"],
        output_schema='{"type": "object"}',
    )
    result = await run_agent_spec(spec, "do it", timeout_s=9.0, registry=registry)
    sent = port.requests[0]
    assert sent.workspace == Path("/srv/wt")
    assert sent.timeout_s == 9.0
    assert sent.response_format == "json"
    assert sent.allowed_tools == ("mcp__graph-os",)
    assert result.success is True
    assert result.output == "ok"
    assert result.token_usage["input_tokens"] == 5
    assert result.metadata["run_outcome"]["recorded"] is True
    assert l5_calls["trace"][0][4]["execution_mode"] == "harness:claude-code"


async def test_parallel_engine_dispatches_non_native_nodes(
    monkeypatch: pytest.MonkeyPatch, l5_calls
) -> None:
    from agent_utilities.graph.parallel_engine import ParallelEngine
    from agent_utilities.layers import harness_registry as registry_module
    from agent_utilities.models.execution_manifest import ExecutionManifest

    registry = HarnessRegistry(native=_FakePort(NATIVE_HARNESS))
    port = _FakePort("claude-code")
    registry.register(port)
    monkeypatch.setattr(registry_module, "_registry", registry)
    spec = AgentSpec(agent_id="coder", harness="claude-code", task_template="t1")
    manifest = ExecutionManifest(agents=[spec], query="q")
    engine = ParallelEngine(engine=None)
    result = await engine._execute_agent(spec, manifest, None, [])
    assert result.success is True
    assert port.requests[0].task.startswith("t1")
