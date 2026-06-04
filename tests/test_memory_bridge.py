#!/usr/bin/python
"""Tests for KG-2.7 Observational Memory Bridge.

CONCEPT:KG-2.1 — Observational Memory Bridge
CONCEPT:ECO-4.0 — Agent Hook Installer

Covers:
- MemoryMaterializer: KG -> Markdown rendering + bidirectional sync
- StartupContextBuilder: budgeted payload generation
- HookInstaller: hook file writing for agent surfaces
- Observer/Reflector: transcript -> observation -> reflection pipeline
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


@pytest.fixture()
def memory_tmpdir(tmp_path: Path):
    """Override memory dir to a temp directory."""
    with patch.dict(os.environ, {"AGENT_UTILITIES_MEMORY_DIR": str(tmp_path)}):
        yield tmp_path


@pytest.fixture()
def engine():
    """Create a minimal IntelligenceGraphEngine for testing."""
    os.environ["AGENT_UTILITIES_TESTING"] = "true"
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    g = GraphComputeEngine(backend_type="rust")

    # Seed with test data
    g.add_node(
        "user_1",
        type="user",
        name="Test User",
        role="developer",
        communication_style="concise",
        id="user_1",
    )
    g.add_node(
        "pref_1",
        type="preference",
        name="Dark mode preference",
        value="Prefers dark mode in all tools",
        category="ui",
        description="Dark mode",
        importance_score=0.8,
        id="pref_1",
    )
    g.add_node(
        "obs_1",
        type="observation",
        name="Decision about KG",
        content="User decided to use KG-first architecture",
        description="KG-first architecture",
        priority="critical",
        source="claude",
        timestamp="2026-05-17T10:00:00Z",
        importance_score=0.9,
        id="obs_1",
    )
    g.add_node(
        "obs_2",
        type="observation",
        name="CI fix session",
        content="Codex session focused on CI pipeline fixes",
        description="CI pipeline fixes",
        priority="important",
        source="codex",
        timestamp="2026-05-17T14:00:00Z",
        importance_score=0.6,
        id="obs_2",
    )
    g.add_node(
        "ref_1",
        type="reflection",
        name="KG architecture pattern",
        content="Agent consistently prefers graph-native persistence",
        description="Graph-native persistence",
        category="architecture",
        confidence=0.9,
        importance_score=0.7,
        id="ref_1",
    )
    g.add_node(
        "fact_1",
        type="fact",
        name="Python version",
        content="Project uses Python 3.12",
        description="Python 3.12",
        certainty=1.0,
        id="fact_1",
    )
    g.add_node(
        "goal_1",
        type="goal",
        name="Complete memory bridge",
        goal_text="Implement cross-agent memory bridge (KG-2.7)",
        description="Memory bridge",
        status="active",
        importance_score=0.9,
        id="goal_1",
    )
    g.add_node(
        "ep_1",
        type="episode",
        name="Morning session",
        summary="Refactored KG ingestion pipeline",
        source="claude",
        timestamp="2026-05-17T09:00:00Z",
        description="KG refactor",
        id="ep_1",
    )
    g.add_node(
        "thread_1",
        type="thread",
        name="Memory Bridge Implementation",
        title="Memory Bridge Implementation",
        created_at="2026-05-17T10:00:00Z",
        id="thread_1",
    )

    engine = IntelligenceGraphEngine(db_path=":memory:")
    yield engine
    IntelligenceGraphEngine._ACTIVE_ENGINE = None


# === MemoryMaterializer Tests ===


class TestMemoryMaterializer:
    def test_materialize_creates_four_files(self, engine, memory_tmpdir):
        from agent_utilities.knowledge_graph.memory import MemoryMaterializer

        m = MemoryMaterializer(engine)
        paths = m.materialize()
        assert len(paths) == 4
        assert all(p.exists() for p in paths.values())
        assert set(paths.keys()) == {
            "observations.md",
            "reflections.md",
            "profile.md",
            "active.md",
        }

    def test_observations_contain_priority_emoji(self, engine, memory_tmpdir):
        from agent_utilities.knowledge_graph.memory import MemoryMaterializer

        m = MemoryMaterializer(engine)
        paths = m.materialize()
        obs = paths["observations.md"].read_text()
        assert "\U0001f534" in obs  # critical
        assert "\U0001f7e1" in obs  # important
        assert "KG-first architecture" in obs
        assert "[claude]" in obs

    def test_profile_contains_identity(self, engine, memory_tmpdir):
        from agent_utilities.knowledge_graph.memory import MemoryMaterializer

        m = MemoryMaterializer(engine)
        paths = m.materialize()
        profile = paths["profile.md"].read_text()
        assert "Test User" in profile or "developer" in profile
        assert "Prefers dark mode" in profile

    def test_active_contains_goals(self, engine, memory_tmpdir):
        from agent_utilities.knowledge_graph.memory import MemoryMaterializer

        m = MemoryMaterializer(engine)
        paths = m.materialize()
        active = paths["active.md"].read_text()
        assert "cross-agent memory bridge" in active.lower() or "KG-2.7" in active

    def test_detect_edits_after_modification(self, engine, memory_tmpdir):
        from agent_utilities.knowledge_graph.memory import MemoryMaterializer

        m = MemoryMaterializer(engine)
        m.materialize()

        # No edits yet
        assert m.detect_edits() == []

        # Modify profile.md
        profile_path = memory_tmpdir / "profile.md"
        profile_path.write_text(
            profile_path.read_text() + "\n- I prefer vim over emacs\n"
        )
        edited = m.detect_edits()
        assert "profile.md" in edited

    def test_ingest_edits_roundtrip(self, engine, memory_tmpdir):
        from agent_utilities.knowledge_graph.memory import MemoryMaterializer

        m = MemoryMaterializer(engine)
        m.materialize()

        # Add a new preference to profile.md
        profile_path = memory_tmpdir / "profile.md"
        content = profile_path.read_text()
        content += "\n- I prefer PostgreSQL for production databases\n"
        profile_path.write_text(content)

        results = m.ingest_edits()
        assert "profile.md" in results
        assert results["profile.md"] > 0

    def test_cursor_persistence(self, engine, memory_tmpdir):
        from agent_utilities.knowledge_graph.memory import MemoryMaterializer

        m = MemoryMaterializer(engine)
        m.materialize()

        cursor_path = memory_tmpdir / ".memory_cursor.json"
        assert cursor_path.exists()
        cursor = json.loads(cursor_path.read_text())
        assert "_materialized_at" in cursor
        assert "observations.md" in cursor


# === StartupContextBuilder Tests ===


class TestStartupContextBuilder:
    def test_build_payload_within_budget(self, engine, memory_tmpdir):
        from agent_utilities.knowledge_graph.memory import StartupContextBuilder

        builder = StartupContextBuilder(engine)
        payload = builder.build_payload(budget_chars=4000)
        assert len(payload.text) <= 4000
        assert payload.budget_chars == 4000

    def test_payload_contains_routing_info(self, engine, memory_tmpdir):
        from agent_utilities.knowledge_graph.memory import StartupContextBuilder

        builder = StartupContextBuilder(engine)
        payload = builder.build_payload(
            agent="codex",
            cwd="/home/user/projects/agent-utilities",
            task="fix CI pipeline",
        )
        assert "codex" in payload.text.lower() or "Agent: codex" in payload.text
        assert "Recall" in payload.text

    def test_payload_has_overflow_handles(self, engine, memory_tmpdir):
        from agent_utilities.knowledge_graph.memory import StartupContextBuilder

        builder = StartupContextBuilder(engine)
        # Very small budget to force overflow
        payload = builder.build_payload(budget_chars=2000)
        assert isinstance(payload.overflow, list)
        assert isinstance(payload.included_handles, list)

    def test_recall_handle_expansion(self, engine, memory_tmpdir):
        from agent_utilities.knowledge_graph.memory import StartupContextBuilder

        builder = StartupContextBuilder(engine)
        builder.build_payload()  # Ensure files exist
        profile = builder.recall_handle("startup:profile")
        assert "User Profile" in profile or "profile" in profile.lower()


# === HookInstaller Tests ===


class TestHookInstaller:
    def test_install_claude_hooks(self, tmp_path):
        from agent_utilities.ecosystem.hook_installer import HookInstaller

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        with patch(
            "agent_utilities.ecosystem.hook_installer._home", return_value=tmp_path
        ):
            installer = HookInstaller()
            results = installer.install(["claude"])
            assert results.get("claude") == "installed"
            config = claude_dir / "settings.json"
            assert config.exists()
            data = json.loads(config.read_text())
            assert "hooks" in data
            assert "SessionStart" in data["hooks"]

    def test_install_codex_hooks(self, tmp_path):
        from agent_utilities.ecosystem.hook_installer import HookInstaller

        with patch(
            "agent_utilities.ecosystem.hook_installer._home", return_value=tmp_path
        ):
            installer = HookInstaller()
            results = installer.install(["codex"])
            assert results.get("codex") == "installed"
            config = tmp_path / ".codex" / "hooks.json"
            assert config.exists()

    def test_doctor_reports_status(self, tmp_path):
        from agent_utilities.ecosystem.hook_installer import HookInstaller

        with patch(
            "agent_utilities.ecosystem.hook_installer._home", return_value=tmp_path
        ):
            installer = HookInstaller()
            installer.install(["claude"])
            report = installer.doctor()
            assert "claude" in report
            assert report["claude"]["status"] == "healthy"
            assert "codex" in report
            assert report["codex"]["status"] == "not_installed"

    def test_uninstall_removes_hooks(self, tmp_path):
        from agent_utilities.ecosystem.hook_installer import HookInstaller

        with patch(
            "agent_utilities.ecosystem.hook_installer._home", return_value=tmp_path
        ):
            installer = HookInstaller()
            installer.install(["grok"])
            grok_config = tmp_path / ".grok" / "hooks" / "agent-utilities-memory.json"
            assert grok_config.exists()
            installer.uninstall(["grok"])
            assert not grok_config.exists()

    def test_unknown_agent_handled(self):
        from agent_utilities.ecosystem.hook_installer import HookInstaller

        installer = HookInstaller()
        results = installer.install(["nonexistent_agent"])
        assert results["nonexistent_agent"] == "unknown_agent"

    def test_agent_terminal_ui_integrated(self):
        from agent_utilities.ecosystem.hook_installer import HookInstaller

        installer = HookInstaller()
        results = installer.install(["agent-terminal-ui"])
        assert results["agent-terminal-ui"] == "integrated"


# === Convenience Function Tests ===


class TestConvenienceFunctions:
    def test_materialize_memory(self, engine, memory_tmpdir):
        from agent_utilities.knowledge_graph.memory import materialize_memory

        paths = materialize_memory(engine)
        assert len(paths) == 4

    def test_build_startup_payload(self, engine, memory_tmpdir):
        from agent_utilities.knowledge_graph.memory import build_startup_payload

        payload = build_startup_payload(engine)
        assert len(payload.text) > 0
        assert payload.budget_chars == 24000
