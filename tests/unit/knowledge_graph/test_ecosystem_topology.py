#!/usr/bin/python
from __future__ import annotations

"""Tests for Ecosystem Topology Builder.

CONCEPT:AU-ECO.messaging.native-backend-abstraction — Ecosystem Topology Map

Validates package discovery, dependency graph construction,
impact radius computation, MCP categorization, and KG persistence.
"""


import textwrap
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.core.ecosystem_topology import (
    EcosystemTopologyBuilder,
    PackageCategory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Create a minimal multi-package workspace."""
    # Kernel
    kernel = tmp_path / "agent-utilities"
    kernel.mkdir()
    (kernel / "pyproject.toml").write_text(
        textwrap.dedent("""\
            [project]
            name = "agent-utilities"
            version = "0.5.0"
            description = "Agent OS Kernel"
            dependencies = []
        """)
    )

    # Frontend (TUI)
    tui = tmp_path / "agent-terminal-ui"
    tui.mkdir()
    (tui / "pyproject.toml").write_text(
        textwrap.dedent("""\
            [project]
            name = "agent-terminal-ui"
            version = "0.4.0"
            description = "Terminal UI"
            dependencies = ["agent-utilities>=0.4.0", "textual>=0.40"]
        """)
    )

    # Frontend (WebUI)
    webui = tmp_path / "agent-webui"
    webui.mkdir()
    (webui / "pyproject.toml").write_text(
        textwrap.dedent("""\
            [project]
            name = "agent-webui"
            version = "0.8.2"
            description = "Web UI"
            dependencies = ["agent-utilities>=0.4.0"]
        """)
    )

    # MCP Server (Infrastructure)
    mcp1 = tmp_path / "container-manager-mcp"
    mcp1.mkdir()
    (mcp1 / "pyproject.toml").write_text(
        textwrap.dedent("""\
            [project]
            name = "container-manager-mcp"
            version = "0.1.0"
            description = "Container management"
            dependencies = ["agent-utilities>=0.4.0", "docker>=7.0"]
        """)
    )

    # MCP Server (Media)
    mcp2 = tmp_path / "jellyfin-mcp"
    mcp2.mkdir()
    (mcp2 / "pyproject.toml").write_text(
        textwrap.dedent("""\
            [project]
            name = "jellyfin-mcp"
            version = "0.1.0"
            description = "Jellyfin media server"
            dependencies = ["agent-utilities>=0.4.0"]
        """)
    )

    # Skills package
    skills = tmp_path / "universal-skills"
    skills.mkdir()
    (skills / "pyproject.toml").write_text(
        textwrap.dedent("""\
            [project]
            name = "universal-skills"
            version = "0.2.0"
            description = "Skill library"
            dependencies = []
        """)
    )

    return tmp_path


@pytest.fixture()
def builder(workspace: Path) -> EcosystemTopologyBuilder:
    """Create a builder for the test workspace."""
    return EcosystemTopologyBuilder(workspace)


# ---------------------------------------------------------------------------
# Package Discovery
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-OS.deployment.infra-orchestration")
class TestPackageDiscovery:
    """Tests for pyproject.toml scanning and metadata extraction."""

    def test_discovers_all_packages(
        self, builder: EcosystemTopologyBuilder, workspace: Path
    ) -> None:
        packages = builder.discover_packages()
        names = {p.name for p in packages}
        assert "agent-utilities" in names
        assert "agent-terminal-ui" in names
        assert "agent-webui" in names
        assert "container-manager-mcp" in names
        assert "jellyfin-mcp" in names
        assert "universal-skills" in names
        assert len(packages) == 6

    def test_kernel_classification(self, builder: EcosystemTopologyBuilder) -> None:
        packages = builder.discover_packages()
        kernel = next(p for p in packages if p.name == "agent-utilities")
        assert kernel.is_kernel is True
        assert kernel.is_frontend is False
        assert kernel.is_mcp_server is False
        assert kernel.category == PackageCategory.KERNEL

    def test_frontend_classification(self, builder: EcosystemTopologyBuilder) -> None:
        packages = builder.discover_packages()
        tui = next(p for p in packages if p.name == "agent-terminal-ui")
        assert tui.is_frontend is True
        assert tui.is_kernel is False
        assert tui.category == PackageCategory.FRONTEND

    def test_mcp_server_classification(self, builder: EcosystemTopologyBuilder) -> None:
        packages = builder.discover_packages()
        container = next(p for p in packages if p.name == "container-manager-mcp")
        assert container.is_mcp_server is True
        assert container.category == PackageCategory.INFRASTRUCTURE

    def test_media_mcp_categorization(self, builder: EcosystemTopologyBuilder) -> None:
        packages = builder.discover_packages()
        jellyfin = next(p for p in packages if p.name == "jellyfin-mcp")
        assert jellyfin.is_mcp_server is True
        assert jellyfin.category == PackageCategory.MEDIA

    def test_skill_package_classification(
        self, builder: EcosystemTopologyBuilder
    ) -> None:
        packages = builder.discover_packages()
        skills = next(p for p in packages if p.name == "universal-skills")
        assert skills.is_skill_package is True
        assert skills.category == PackageCategory.SKILLS

    def test_dependencies_extracted(self, builder: EcosystemTopologyBuilder) -> None:
        packages = builder.discover_packages()
        tui = next(p for p in packages if p.name == "agent-terminal-ui")
        assert "agent-utilities" in tui.dependencies
        assert "textual" in tui.dependencies

    def test_version_extracted(self, builder: EcosystemTopologyBuilder) -> None:
        packages = builder.discover_packages()
        kernel = next(p for p in packages if p.name == "agent-utilities")
        assert kernel.version == "0.5.0"

    def test_empty_workspace(self, tmp_path: Path) -> None:
        builder = EcosystemTopologyBuilder(tmp_path)
        packages = builder.discover_packages()
        assert packages == []


# ---------------------------------------------------------------------------
# Dependency Graph
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-OS.deployment.infra-orchestration")
class TestDependencyGraph:
    """Tests for inter-package dependency graph construction."""

    def test_builds_internal_deps_only(self, builder: EcosystemTopologyBuilder) -> None:
        packages = builder.discover_packages()
        graph = builder.build_dependency_graph(packages)
        # TUI depends on agent-utilities (internal) but not textual (external)
        assert "agent-utilities" in graph["agent-terminal-ui"]
        assert "textual" not in graph["agent-terminal-ui"]

    def test_kernel_has_no_internal_deps(
        self, builder: EcosystemTopologyBuilder
    ) -> None:
        packages = builder.discover_packages()
        graph = builder.build_dependency_graph(packages)
        assert graph["agent-utilities"] == []

    def test_all_packages_in_graph(self, builder: EcosystemTopologyBuilder) -> None:
        packages = builder.discover_packages()
        graph = builder.build_dependency_graph(packages)
        assert len(graph) == len(packages)


# ---------------------------------------------------------------------------
# Impact Radius
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-OS.deployment.infra-orchestration")
class TestImpactRadius:
    """Tests for transitive impact radius computation."""

    def test_kernel_impacts_all_dependents(
        self, builder: EcosystemTopologyBuilder
    ) -> None:
        packages = builder.discover_packages()
        graph = builder.build_dependency_graph(packages)
        impact = builder.get_impact_radius("agent-utilities", graph)
        # All packages that depend on agent-utilities
        assert "agent-terminal-ui" in impact
        assert "agent-webui" in impact
        assert "container-manager-mcp" in impact
        assert "jellyfin-mcp" in impact

    def test_leaf_has_no_impact(self, builder: EcosystemTopologyBuilder) -> None:
        packages = builder.discover_packages()
        graph = builder.build_dependency_graph(packages)
        impact = builder.get_impact_radius("jellyfin-mcp", graph)
        assert impact == []

    def test_unknown_package(self, builder: EcosystemTopologyBuilder) -> None:
        packages = builder.discover_packages()
        graph = builder.build_dependency_graph(packages)
        impact = builder.get_impact_radius("nonexistent", graph)
        assert impact == []


# ---------------------------------------------------------------------------
# MCP Coverage
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-OS.deployment.infra-orchestration")
class TestMCPCoverage:
    """Tests for frontend-to-MCP server coverage mapping."""

    def test_frontends_consume_all_mcp(self, builder: EcosystemTopologyBuilder) -> None:
        packages = builder.discover_packages()
        coverage = builder.compute_mcp_coverage(packages)
        assert "agent-terminal-ui" in coverage
        assert "agent-webui" in coverage
        assert "container-manager-mcp" in coverage["agent-terminal-ui"]
        assert "jellyfin-mcp" in coverage["agent-webui"]


# ---------------------------------------------------------------------------
# Category Groups
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-OS.deployment.infra-orchestration")
class TestCategoryGroups:
    """Tests for intelligent package category grouping."""

    def test_groups_by_category(self, builder: EcosystemTopologyBuilder) -> None:
        packages = builder.discover_packages()
        groups = builder.get_category_groups(packages)
        assert "kernel" in groups
        assert "frontend" in groups
        assert "infrastructure" in groups
        assert "media" in groups
        assert "skills" in groups


# ---------------------------------------------------------------------------
# Topology Report
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-OS.deployment.infra-orchestration")
class TestTopologyReport:
    """Tests for markdown report generation."""

    def test_report_contains_sections(self, builder: EcosystemTopologyBuilder) -> None:
        packages = builder.discover_packages()
        report = builder.generate_topology_report(packages)
        assert "# Ecosystem Topology Report" in report
        assert "## Package Categories" in report
        assert "## Kernel Impact Radius" in report
        assert "## Frontend MCP Coverage" in report

    def test_report_contains_package_names(
        self, builder: EcosystemTopologyBuilder
    ) -> None:
        packages = builder.discover_packages()
        report = builder.generate_topology_report(packages)
        assert "agent-utilities" in report
        assert "container-manager-mcp" in report
