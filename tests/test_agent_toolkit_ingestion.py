"""Tests for the unified Agent Toolkit Ingestion pipeline.

CONCEPT:ECO-4.0 — Unified MCP/Skill/A2A ingestion pipeline tests.
CONCEPT:ECO-4.1 — MCP Live Discovery tests (parse_mcp_config, _parse_tool_flags).

Uses real mcp_config.json fixtures from the agent-packages ecosystem
(portainer, langfuse, gitlab, container-manager, etc.) to verify
ingestion, auto-detection, tool flag parsing, and idempotent refresh.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
import pytest

# -- Test fixtures paths (real configs from agent-packages) ---
AGENTS_DIR = Path(__file__).resolve().parents[1] / ".." / "agents"

# Known config paths for agent-packages
_MCP_CONFIGS = {
    "portainer": AGENTS_DIR / "portainer-agent" / "mcp_config.json",
    "langfuse": AGENTS_DIR / "langfuse-agent" / "mcp_config.json",
    "container-manager": AGENTS_DIR / "container-manager-mcp" / "mcp_config.json",
    "tunnel-manager": AGENTS_DIR / "tunnel-manager" / "mcp_config.json",
    "systems-manager": AGENTS_DIR / "systems-manager" / "mcp_config.json",
    "github": AGENTS_DIR / "github-agent" / "mcp_config.json",
    "gitlab": AGENTS_DIR / "gitlab-api" / "mcp_config.json",
    "archivebox": AGENTS_DIR / "archivebox-api" / "archivebox_api" / "mcp_config.json",
}

ANTIGRAVITY_MCP_CONFIG = Path(
    os.environ.get(
        "ANTIGRAVITY_MCP_CONFIG",
        os.path.expanduser("~/.gemini/antigravity/mcp_config.json"),
    )
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_engine():
    """Create a minimal IntelligenceGraphEngine for testing."""
    os.environ["AGENT_UTILITIES_TESTING"] = "true"
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    graph = GraphComputeEngine(backend_type="rust")
    engine = IntelligenceGraphEngine(graph=graph, backend=None)
    return engine


def _load_config(name: str) -> dict[str, Any] | None:
    """Load a real MCP config if it exists on disk."""
    path = _MCP_CONFIGS.get(name)
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ---------------------------------------------------------------------------
# Component 1: Tool Flag Parsing (MCPDiscoveryMixin)
# ---------------------------------------------------------------------------


class TestToolFlagParsing:
    """Tests for _parse_tool_flags() extraction logic."""

    def test_portainer_tool_flags(self):
        """Portainer config has 10 TOOL=True env vars."""
        engine = _create_engine()
        env = {
            "PORTAINER_TOKEN": "secret",
            "PORTAINER_URL": "https://port.local",
            "AUTHTOOL": "True",
            "DOCKERTOOL": "True",
            "EDGETOOL": "True",
            "ENVIRONMENTTOOL": "True",
            "KUBERNETESTOOL": "True",
            "REGISTRYTOOL": "True",
            "STACKTOOL": "True",
            "SYSTEMTOOL": "True",
            "TEMPLATETOOL": "True",
            "USERTOOL": "True",
        }
        flags = engine._parse_tool_flags(env)
        assert len(flags) == 10
        assert "docker" in flags
        assert "kubernetes" in flags
        assert "auth" in flags
        # Secrets should NOT be in flags
        assert "portainer_toke" not in flags

    def test_langfuse_tool_flags(self):
        """Langfuse config has 18+ TOOL=True env vars."""
        config = _load_config("langfuse")
        if config is None:
            pytest.skip("langfuse mcp_config.json not found on disk")

        assert config is not None  # narrow for mypy
        engine = _create_engine()
        env = config["mcpServers"]["langfuse-agent"]["env"]
        flags = engine._parse_tool_flags(env)
        # Langfuse has many tool groups
        assert len(flags) >= 15
        assert "trace" in flags or "trace_" in flags

    def test_no_tool_flags(self):
        """Config with no TOOL env vars returns empty list."""
        engine = _create_engine()
        env = {"API_KEY": "secret", "BASE_URL": "http://localhost"}
        flags = engine._parse_tool_flags(env)
        assert flags == []

    def test_case_insensitive_true(self):
        """Tool flag parsing handles various True representations."""
        engine = _create_engine()
        env = {"DOCKERTOOL": "true", "STACKTOOL": "1", "SYSTEMTOOL": "yes"}
        flags = engine._parse_tool_flags(env)
        assert len(flags) == 3


# ---------------------------------------------------------------------------
# Component 2: MCP Config Parsing
# ---------------------------------------------------------------------------


class TestMCPConfigParsing:
    """Tests for parse_mcp_config() server extraction."""

    def test_parse_portainer_config(self):
        """Parse portainer single-server config."""
        config = _load_config("portainer")
        if config is None:
            pytest.skip("portainer mcp_config.json not found on disk")

        engine = _create_engine()
        servers = engine.parse_mcp_config(config)
        assert len(servers) == 1
        assert servers[0]["name"] == "portainer-agent"
        assert servers[0]["command"] == "uv"
        assert "portainer-mcp" in servers[0]["args"]
        assert len(servers[0]["tool_flags"]) == 10

    def test_parse_multi_server_config(self):
        """Parse Antigravity IDE config with 3 servers."""
        if not ANTIGRAVITY_MCP_CONFIG.exists():
            pytest.skip("Antigravity mcp_config.json not found")

        engine = _create_engine()
        config = json.loads(ANTIGRAVITY_MCP_CONFIG.read_text(encoding="utf-8"))
        servers = engine.parse_mcp_config(config)

        # Should have active servers
        active_servers = [s for s in servers]
        assert len(active_servers) >= 1

        names = [s["name"] for s in active_servers]
        assert "mcp-multiplexer" in names or "graph-os" in names or "scholarx" in names

    def test_parse_gitlab_config(self):
        """Parse gitlab config with 17+ tool groups."""
        config = _load_config("gitlab")
        if config is None:
            pytest.skip("gitlab mcp_config.json not found on disk")

        engine = _create_engine()
        servers = engine.parse_mcp_config(config)
        assert len(servers) == 1
        assert servers[0]["name"] == "gitlab-api"
        assert len(servers[0]["tool_flags"]) >= 15

    def test_disabled_server_skipped(self):
        """Servers with disabled=true are excluded."""
        engine = _create_engine()
        config = {
            "mcpServers": {
                "active": {"command": "uv", "args": ["run", "active-mcp"], "env": {}},
                "disabled": {
                    "command": "uv",
                    "args": ["run", "disabled-mcp"],
                    "env": {},
                    "disabled": True,
                },
            }
        }
        servers = engine.parse_mcp_config(config)
        assert len(servers) == 1
        assert servers[0]["name"] == "active"

    def test_config_hash_deterministic(self):
        """Same config produces same hash."""
        engine = _create_engine()
        h1 = engine._compute_config_hash("test", "uv", ["run", "test"], {"A": "1"})
        h2 = engine._compute_config_hash("test", "uv", ["run", "test"], {"A": "1"})
        assert h1 == h2

    def test_config_hash_changes_on_diff(self):
        """Different config produces different hash."""
        engine = _create_engine()
        h1 = engine._compute_config_hash("test", "uv", ["run", "test"], {"A": "1"})
        h2 = engine._compute_config_hash("test", "uv", ["run", "test"], {"A": "2"})
        assert h1 != h2


# ---------------------------------------------------------------------------
# Component 3: Auto-Detection
# ---------------------------------------------------------------------------


class TestAutoDetection:
    """Tests for _detect_toolkit_source_type() heuristics."""

    def test_detect_url_as_a2a(self):
        """HTTP URLs are detected as A2A agent endpoints."""
        from agent_utilities.knowledge_graph.core.engine_ingestion import IngestionMixin

        assert (
            IngestionMixin._detect_toolkit_source_type("http://agent.local")
            == "a2a_url"
        )
        assert (
            IngestionMixin._detect_toolkit_source_type("https://agent.local:8080")
            == "a2a_url"
        )

    def test_detect_json_url_as_remote(self):
        """URLs ending in .json are detected as remote JSON."""
        from agent_utilities.knowledge_graph.core.engine_ingestion import IngestionMixin

        assert (
            IngestionMixin._detect_toolkit_source_type(
                "https://example.com/mcp_config.json"
            )
            == "remote_json"
        )

    def test_detect_mcp_config_file(self):
        """Local .json files with mcpServers key are detected as MCP configs."""
        config = _load_config("portainer")
        if config is None:
            pytest.skip("portainer mcp_config.json not found on disk")

        from agent_utilities.knowledge_graph.core.engine_ingestion import IngestionMixin

        path = str(_MCP_CONFIGS["portainer"])
        assert IngestionMixin._detect_toolkit_source_type(path) == "mcp_config"

    def test_detect_skill_directory(self):
        """Directories with SKILL.md are detected as skill directories."""
        from agent_utilities.knowledge_graph.core.engine_ingestion import IngestionMixin

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_md = Path(tmpdir) / "SKILL.md"
            skill_md.write_text("---\nname: test-skill\n---\n# Test")
            assert (
                IngestionMixin._detect_toolkit_source_type(tmpdir) == "skill_directory"
            )

    def test_detect_unknown(self):
        """Non-existent paths return unknown."""
        from agent_utilities.knowledge_graph.core.engine_ingestion import IngestionMixin

        assert (
            IngestionMixin._detect_toolkit_source_type("/nonexistent/path") == "unknown"
        )


# ---------------------------------------------------------------------------
# Component 4: Skill Ingestion
# ---------------------------------------------------------------------------


class TestSkillIngestion:
    """Tests for skill directory ingestion."""

    def test_ingest_skill_directory(self):
        """Ingesting a skill directory creates a CallableResource node."""
        engine = _create_engine()

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_md = Path(tmpdir) / "SKILL.md"
            skill_md.write_text(
                "---\nname: test-skill\ndescription: A test skill\n---\n# Test Skill\nDo stuff."
            )

            summary: dict[str, Any] = {
                "mcp_servers": 0,
                "tools_discovered": 0,
                "skills": 0,
                "a2a_agents": 0,
                "errors": [],
                "skipped": 0,
            }
            engine._ingest_skill_from_directory(tmpdir, summary)
            assert summary["skills"] == 1
            assert len(summary["errors"]) == 0

    def test_ingest_skill_missing_skillmd(self):
        """Directory without SKILL.md produces an error."""
        engine = _create_engine()

        with tempfile.TemporaryDirectory() as tmpdir:
            summary: dict[str, Any] = {
                "mcp_servers": 0,
                "tools_discovered": 0,
                "skills": 0,
                "a2a_agents": 0,
                "errors": [],
                "skipped": 0,
            }
            engine._ingest_skill_from_directory(tmpdir, summary)
            assert summary["skills"] == 0
            assert len(summary["errors"]) == 1

    def test_parse_skill_frontmatter(self):
        """YAML frontmatter is correctly parsed."""
        from agent_utilities.knowledge_graph.core.engine_ingestion import IngestionMixin

        content = '---\nname: my-skill\ndescription: "Does cool things"\n---\n# Main'
        fm = IngestionMixin._parse_skill_frontmatter(content)
        assert fm["name"] == "my-skill"
        assert fm["description"] == "Does cool things"


# ---------------------------------------------------------------------------
# Component 5: Unified Toolkit Ingestion (async)
# ---------------------------------------------------------------------------


class TestUnifiedIngestion:
    """Tests for ingest_agent_toolkit() end-to-end."""

    @pytest.mark.asyncio
    async def test_ingest_mixed_list(self):
        """Mixed list with MCP config + skill dir processes both."""
        engine = _create_engine()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a skill directory
            skill_dir = Path(tmpdir) / "my-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: my-skill\ndescription: Test\n---\n# Test"
            )

            # Create an MCP config
            mcp_config = Path(tmpdir) / "mcp_config.json"
            mcp_config.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "test-server": {
                                "command": "echo",
                                "args": ["hello"],
                                "env": {"TESTTOOL": "True"},
                            }
                        }
                    }
                )
            )

            # Patch live discovery to avoid subprocess
            with patch.object(
                engine, "discover_mcp_tools", new_callable=AsyncMock
            ) as mock_discover:
                mock_discover.return_value = []

                result = await engine.ingest_agent_toolkit(
                    [str(mcp_config), str(skill_dir)]
                )

            assert result["skills"] == 1
            assert result["mcp_servers"] == 1
            assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_ingest_error_handling(self):
        """Bad paths produce errors without crashing."""
        engine = _create_engine()
        result = await engine.ingest_agent_toolkit(
            ["/nonexistent/path.json", "/also/missing"]
        )
        assert len(result["errors"]) >= 1

    @pytest.mark.asyncio
    async def test_ingest_a2a_mock(self):
        """A2A URL ingestion with mocked HTTP response."""
        engine = _create_engine()

        mock_card = {
            "name": "test-a2a-agent",
            "description": "A test A2A agent",
            "capabilities": ["testing"],
            "skills": [{"name": "test-skill"}],
        }

        with patch.object(
            engine, "_fetch_a2a_card", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = mock_card
            result = await engine.ingest_agent_toolkit(["http://test-agent.local"])

        assert result["a2a_agents"] == 1
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_ingest_a2a_override_path(self):
        """A2A ingestion respects agent_card_path override."""
        engine = _create_engine()

        with patch.object(
            engine, "_fetch_a2a_card", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = {"name": "custom-agent", "capabilities": []}
            await engine.ingest_agent_toolkit(
                ["http://custom-agent.local"],
                agent_card_path="/agent-card.json",
            )
            # Verify the custom path was used
            mock_fetch.assert_called_with(
                "http://custom-agent.local", "/agent-card.json"
            )

    @pytest.mark.asyncio
    async def test_ingest_idempotent_refresh(self):
        """Ingesting same config twice doesn't duplicate when fresh."""
        engine = _create_engine()

        with tempfile.TemporaryDirectory() as tmpdir:
            mcp_config = Path(tmpdir) / "mcp_config.json"
            mcp_config.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "repeat-server": {
                                "command": "echo",
                                "args": ["hello"],
                                "env": {"TESTTOOL": "True"},
                            }
                        }
                    }
                )
            )

            with patch.object(
                engine, "discover_mcp_tools", new_callable=AsyncMock
            ) as mock_discover:
                mock_discover.return_value = []

                # First ingestion
                r1 = await engine.ingest_agent_toolkit([str(mcp_config)])
                assert r1["mcp_servers"] == 1

                # Second ingestion — should skip due to freshness (if backend existed)
                # Without backend, it will re-ingest (expected for memory-only mode)
                r2 = await engine.ingest_agent_toolkit([str(mcp_config)])
                assert (
                    r2["mcp_servers"] == 1
                )  # Re-ingested (no backend for freshness check)

    @pytest.mark.asyncio
    async def test_ingest_real_portainer_config(self):
        """Ingest real portainer config with mocked live discovery."""
        portainer_path = _MCP_CONFIGS.get("portainer")
        if not portainer_path or not portainer_path.exists():
            pytest.skip("portainer mcp_config.json not found on disk")

        engine = _create_engine()

        with patch.object(
            engine, "discover_mcp_tools", new_callable=AsyncMock
        ) as mock_discover:
            # Simulate live discovery returning 5 tools
            mock_discover.return_value = [
                {"name": f"portainer_tool_{i}", "description": f"Tool {i}"}
                for i in range(5)
            ]
            result = await engine.ingest_agent_toolkit([str(portainer_path)])

        assert result["mcp_servers"] == 1
        assert result["tools_discovered"] == 5
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_ingest_real_portainer_fallback_to_flags(self):
        """When live discovery fails, fall back to tool flag extraction."""
        portainer_path = _MCP_CONFIGS.get("portainer")
        if not portainer_path or not portainer_path.exists():
            pytest.skip("portainer mcp_config.json not found on disk")

        engine = _create_engine()

        with patch.object(
            engine, "discover_mcp_tools", new_callable=AsyncMock
        ) as mock_discover:
            mock_discover.return_value = []  # Live discovery fails
            result = await engine.ingest_agent_toolkit([str(portainer_path)])

        assert result["mcp_servers"] == 1
        # Should have fallen back to tool flags (10 flags for portainer)
        assert result["tools_discovered"] == 10
        assert len(result["errors"]) == 0
