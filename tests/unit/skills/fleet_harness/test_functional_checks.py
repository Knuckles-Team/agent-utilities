"""Functional layer: MUST degrade gracefully — never hang, never false-PASS."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from agent_utilities.skills.fleet_harness.discovery import SkillRecord
from agent_utilities.skills.fleet_harness.functional_checks import (
    probe_reachable,
    referenced_graphos_tools,
    run_functional_checks,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _record_for(name: str, *, repo_name: str = "fixtures") -> SkillRecord:
    skill_md = _FIXTURES / name / "SKILL.md"
    return SkillRecord(
        skill_md=skill_md,
        skill_dir=skill_md.parent,
        repo_root=_FIXTURES,
        repo_name=repo_name,
    )


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeClient:
    def __init__(self, tool_names: list[str]) -> None:
        self._tool_names = tool_names

    async def list_tools(self):
        return [_FakeTool(n) for n in self._tool_names]


def _reachable_factory(tool_names: list[str]):
    @asynccontextmanager
    async def factory():
        yield _FakeClient(tool_names)

    return factory


def _unreachable_factory(*, hang: bool = False):
    @asynccontextmanager
    async def factory():
        if hang:
            await asyncio.sleep(
                3600
            )  # never actually reached under the harness timeout
        raise ConnectionRefusedError("no graph-os listening")
        yield  # pragma: no cover - unreachable, keeps this an async generator

    return factory


def test_referenced_graphos_tools_extracts_only_graphos_convention():
    body = "Use `graph_ask` and `engine_query`. Do not call `ask` (an English verb) or `random_thing`."
    assert referenced_graphos_tools(body) == ["graph_ask", "engine_query"]


def test_referenced_graphos_tools_dedupes_preserving_order():
    body = "`graph_ask` ... later again `graph_ask` ... then `engine_query`."
    assert referenced_graphos_tools(body) == ["graph_ask", "engine_query"]


class TestProbeReachable:
    @pytest.mark.asyncio
    async def test_reachable_returns_live_tool_names(self):
        reachable, tools, detail = await probe_reachable(
            _reachable_factory(["graph_ask", "engine_query"])
        )
        assert reachable is True
        assert tools == frozenset({"graph_ask", "engine_query"})
        assert "connected" in detail

    @pytest.mark.asyncio
    async def test_unreachable_returns_false_never_raises(self):
        reachable, tools, detail = await probe_reachable(_unreachable_factory())
        assert reachable is False
        assert tools == frozenset()
        assert "unreachable" in detail

    @pytest.mark.asyncio
    async def test_never_hangs_past_the_configured_timeout(self, monkeypatch):
        import agent_utilities.skills.fleet_harness.functional_checks as fc

        monkeypatch.setattr(fc, "_CONNECT_TIMEOUT_SECONDS", 0.05)
        monkeypatch.setattr(fc, "_CALL_TIMEOUT_SECONDS", 0.05)
        start = time.monotonic()
        reachable, tools, detail = await probe_reachable(
            _unreachable_factory(hang=True)
        )
        elapsed = time.monotonic() - start
        assert reachable is False
        assert elapsed < 2.0  # bounded, not the 3600s the fake would otherwise sleep
        assert "unreachable" in detail


class TestRunFunctionalChecks:
    @pytest.mark.asyncio
    async def test_skill_with_no_tool_references_is_not_applicable(self):
        # `bad_missing_frontmatter` has no `graph_*`/`engine_*` backtick
        # references in its body at all — not graph-os-routed.
        record = SkillRecord(
            skill_md=_FIXTURES / "bad_missing_frontmatter" / "SKILL.md",
            skill_dir=_FIXTURES / "bad_missing_frontmatter",
            repo_root=_FIXTURES,
            repo_name="agent-utilities",
        )
        results = await run_functional_checks(
            [record], _reachable_factory(["graph_ask"])
        )
        assert results[0].status == "SKIPPED-not-applicable"

    @pytest.mark.asyncio
    async def test_skill_referencing_live_tool_passes(self):
        record = _record_for("good-skill", repo_name="agent-utilities")
        results = await run_functional_checks(
            [record], _reachable_factory(["graph_ask"])
        )
        assert results[0].status == "PASS"
        assert results[0].referenced_tools == ("graph_ask",)

    @pytest.mark.asyncio
    async def test_skill_referencing_dead_tool_fails(self):
        record = _record_for("good-skill", repo_name="agent-utilities")
        # graph_ask is NOT in the live tool surface below -> must FAIL, not PASS.
        results = await run_functional_checks(
            [record], _reachable_factory(["some_other_tool"])
        )
        assert results[0].status == "FAIL"
        assert "graph_ask" in results[0].detail

    @pytest.mark.asyncio
    async def test_unreachable_endpoint_skips_rather_than_fails_or_hangs(self):
        record = _record_for("good-skill", repo_name="agent-utilities")
        results = await run_functional_checks([record], _unreachable_factory())
        assert results[0].status == "SKIPPED-unreachable"
        # never a false PASS and never a FAIL when the harness itself couldn't reach graph-os
        assert results[0].status not in {"PASS", "FAIL"}
