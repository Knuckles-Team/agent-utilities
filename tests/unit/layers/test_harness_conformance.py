"""Every ruled harness adapter against the one shared conformance kit.

Evidence per target (also carried on every ``CheckResult``):

* ``pydantic-ai`` -- the real in-process adapter over a scripted AU runner
  (``real-harness``: the adapter and its runtime contract are real; the model
  is replaced by the runner double).
* ``claude-code`` / ``codex`` -- ``recorded-live-transcript``: JSONL captured
  from real ``claude -p`` / ``codex exec`` runs on 2026-09-22 (see
  ``tests/fixtures/harness_transcripts/provenance.json``); failure and
  blocking scenarios reuse the recorded init records with a synthetic tail.
* ``grok`` / ``devin`` -- ``synthetic-transcript``: authored from the vendor
  documentation because neither harness is installed/credentialed on the
  recording host.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agent_utilities.layers.adapters.claude_code import ClaudeCodeHarness
from agent_utilities.layers.adapters.codex import CodexHarness
from agent_utilities.layers.adapters.devin import DevinHarness
from agent_utilities.layers.adapters.grok import GrokHarness
from agent_utilities.layers.adapters.pydantic_ai import PydanticAiHarness
from agent_utilities.layers.conformance import (
    CHECKS,
    ConformanceTarget,
    Rig,
    TranscriptLauncher,
    run_check,
)
from agent_utilities.layers.contracts import (
    EvidenceSource,
    RunSpec,
    RunToolset,
    SkillRef,
)
from agent_utilities.layers.execution import host_workspace_lease
from agent_utilities.layers.negotiation import HarnessPolicy

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "harness_transcripts"
TASK = "Run the shell command: echo harness-conformance, then reply DONE."
POLICY = HarnessPolicy(require_context_endpoint=False)
SKILL_BODY = "---\nname: planted-skill\ndescription: planted\n---\nReply PLANTED.\n"
PLANTED_SKILL = SkillRef(
    name="planted-skill",
    digest=hashlib.sha256(SKILL_BODY.encode()).hexdigest(),
    body=SKILL_BODY,
)


def _lines(name: str) -> list[str]:
    return (FIXTURES / name).read_text(encoding="utf-8").splitlines()


class _Secrets:
    def resolve(self, ref: str) -> str | None:
        return {"env://HARNESS_TOKEN": "token-value"}.get(ref)


def _spec(run_id: str, **overrides: Any) -> RunSpec:
    return RunSpec(run_id=run_id, task=TASK, agent_ref="conformance", **overrides)


# --- CLI targets -------------------------------------------------------------


def _cli_rig(
    harness_type: Callable[..., Any],
    root: Path,
    launcher: TranscriptLauncher,
    spec: RunSpec,
) -> Rig:
    harness = harness_type(launcher=launcher, credentials=_Secrets())
    lease = host_workspace_lease(str(root), spec.run_id)

    def observe() -> dict[str, Any]:
        paths = [launch["cwd"] for launch in launcher.launches]
        paths += [
            arg
            for launch in launcher.launches
            for arg in launch["argv"][1:]
            if arg.startswith("/")
        ]
        stopped = any(process.terminated for process in launcher.processes)
        return {"paths": paths, "stopped": stopped}

    return Rig(harness=harness, spec=spec, lease=lease, observe=observe)


@dataclass(frozen=True)
class _CliDialect:
    name: str
    harness_type: Callable[..., Any]
    success: list[str]
    blocking: list[str]
    failing: list[str]
    tool_name: str
    evidence: EvidenceSource
    spec_overrides: dict[str, Any]


def _claude() -> _CliDialect:
    success = _lines("claude_code_success.jsonl")
    failure = {
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": True,
        "result": "",
        "session_id": "synthetic",
    }
    return _CliDialect(
        name="claude-code",
        harness_type=ClaudeCodeHarness,
        success=success,
        blocking=success[:1],
        failing=[success[0], json.dumps(failure)],
        tool_name="Bash",
        evidence="recorded-live-transcript",
        spec_overrides={
            "toolset": RunToolset(
                allowed_tools=("Bash(echo:*)",),
                required_tools=("Bash",),
                skills=(PLANTED_SKILL,),
            )
        },
    )


def _codex() -> _CliDialect:
    success = _lines("codex_success.jsonl")
    failure = {"type": "turn.failed", "error": {"message": "synthetic refusal"}}
    return _CliDialect(
        name="codex",
        harness_type=CodexHarness,
        success=success,
        blocking=success[:2],
        failing=[*success[:2], json.dumps(failure)],
        tool_name="command_execution",
        evidence="recorded-live-transcript",
        spec_overrides={},
    )


def _grok() -> _CliDialect:
    success = _lines("grok_success.jsonl")
    failure = {"jsonrpc": "2.0", "id": 2, "error": {"code": -32000, "message": "x"}}
    return _CliDialect(
        name="grok",
        harness_type=GrokHarness,
        success=success,
        blocking=success[:1],
        failing=[json.dumps(failure)],
        tool_name="run_command",
        evidence="synthetic-transcript",
        spec_overrides={},
    )


def _cli_target(dialect: _CliDialect, root: Path) -> ConformanceTarget:
    def factory(lines: list[str], **launcher: Any) -> Callable[[str], Rig]:
        def build(run_id: str) -> Rig:
            spec = _spec(run_id, **dialect.spec_overrides)
            return _cli_rig(
                dialect.harness_type, root, TranscriptLauncher(lines, **launcher), spec
            )

        return build

    return ConformanceTarget(
        name=dialect.name,
        evidence=dialect.evidence,
        policy=POLICY,
        tool_name=dialect.tool_name,
        scenarios={
            "success": factory(dialect.success),
            "blocking": factory(dialect.blocking, hold_open=True),
            "failing": factory(dialect.failing, exit_code=1),
            "unavailable": factory(dialect.success, installed=False),
        },
    )


# --- in-process target -------------------------------------------------------


@dataclass(frozen=True)
class _Progress:
    stage: str
    status: str
    detail: str = ""
    evidence: dict | None = None


class _ScriptedRunner:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.stopped = False

    async def execute_agent(self, agent_name: str, task: str, **options: Any) -> str:
        sink = options["progress_sink"]
        if self.mode == "failing":
            raise LookupError("no authorized capability")
        await sink(_Progress("tool_call", "started", "run_command"))
        if self.mode == "blocking":
            try:
                await asyncio.Event().wait()
            finally:
                self.stopped = True
        await sink(_Progress("tool_result", "ok", "run_command"))
        await sink(_Progress("done", "ok", "completed"))
        return "DONE"


def _in_process_target() -> ConformanceTarget:
    def factory(mode: str, **spec: Any) -> Callable[[str], Rig]:
        def build(run_id: str) -> Rig:
            runner = _ScriptedRunner(mode)
            return Rig(
                harness=PydanticAiHarness(runner),
                spec=_spec(run_id, **spec),
                lease=None,
                observe=lambda: {"stopped": runner.stopped},
            )

        return build

    return ConformanceTarget(
        name="pydantic-ai",
        evidence="real-harness",
        policy=POLICY,
        tool_name="tool_call",
        scenarios={
            "success": factory("success"),
            "blocking": factory("blocking"),
            "failing": factory("failing"),
            "unavailable": factory("success", account_mode="api_key"),
        },
    )


# --- Devin target ------------------------------------------------------------


class _RecordedDevinApi:
    """Synthetic replay of the Devin v3 session API (docs.devin.ai OpenAPI)."""

    def __init__(self, statuses: list[tuple[str, str]]) -> None:
        self.statuses = list(statuses)
        self.terminated: list[str] = []
        self.closed = False

    async def create_session(self, body: dict[str, Any]) -> dict[str, Any]:
        return {"session_id": "devin-abc123", "url": "https://app.devin.ai/s/abc"}

    async def get_session(self, session_id: str) -> dict[str, Any]:
        status, detail = self.statuses[0]
        if len(self.statuses) > 1:
            self.statuses.pop(0)
        return {"status": status, "status_detail": detail, "acus_consumed": 1.5}

    async def list_messages(self, session_id: str, after: str | None) -> dict:
        if after is not None:
            return {"items": [], "end_cursor": after, "has_next_page": False}
        item = {"event_id": "e1", "source": "devin", "message": "DONE", "created_at": 1}
        return {"items": [item], "end_cursor": "c1", "has_next_page": False}

    async def terminate(self, session_id: str) -> None:
        self.terminated.append(session_id)

    async def aclose(self) -> None:
        self.closed = True


def _devin_target() -> ConformanceTarget:
    def factory(
        statuses: list[tuple[str, str]], org_id: str | None = "org-1"
    ) -> Callable[[str], Rig]:
        def build(run_id: str) -> Rig:
            api = _RecordedDevinApi(statuses)
            harness = DevinHarness(
                org_id=org_id,
                credentials=_Secrets(),
                api_factory=lambda token, org: api,
                poll_interval_s=0.01,
            )
            spec = _spec(
                run_id,
                allowed_environments=frozenset({"provider-managed-remote"}),
                account_mode="api_key",
                account_ref="env://HARNESS_TOKEN",
            )
            return Rig(
                harness=harness,
                spec=spec,
                lease=None,
                observe=lambda: {"stopped": bool(api.terminated)},
            )

        return build

    working = [("running", "working"), ("running", "finished")]
    return ConformanceTarget(
        name="devin",
        evidence="synthetic-transcript",
        policy=POLICY,
        tool_name="",
        scenarios={
            "success": factory(working),
            "blocking": factory([("running", "working")]),
            "failing": factory([("error", "")]),
            "unavailable": factory(working, org_id=None),
        },
    )


_TARGETS: dict[str, Callable[[Path], ConformanceTarget]] = {
    "pydantic-ai": lambda root: _in_process_target(),
    "claude-code": lambda root: _cli_target(_claude(), root),
    "codex": lambda root: _cli_target(_codex(), root),
    "grok": lambda root: _cli_target(_grok(), root),
    "devin": lambda root: _devin_target(),
}


@pytest.mark.parametrize("check", sorted(CHECKS))
@pytest.mark.parametrize("harness", sorted(_TARGETS))
async def test_adapter_passes_the_conformance_kit(
    harness: str, check: str, tmp_path: Path
) -> None:
    target = _TARGETS[harness](tmp_path)
    result = await run_check(target, check)
    assert result.passed
    assert result.harness == harness
    assert result.evidence == target.evidence


def test_every_ruled_harness_has_a_conformance_target() -> None:
    assert set(_TARGETS) == {"pydantic-ai", "claude-code", "codex", "devin", "grok"}


def test_recorded_fixtures_are_labelled() -> None:
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    labels = {name: entry["evidence"] for name, entry in provenance.items()}
    assert labels == {
        "claude_code_success.jsonl": "recorded-live-transcript",
        "codex_success.jsonl": "recorded-live-transcript",
        "grok_success.jsonl": "synthetic-transcript",
    }
