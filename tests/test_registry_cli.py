#!/usr/bin/python
"""Tests for CONCEPT:OS-5.0 — Agent Registry."""

from __future__ import annotations

import json

import pytest

from agent_utilities.core.registry_cli import AgentRegistry, SpecialistPackage


@pytest.fixture
def registry_path(tmp_path):
    """Create a temporary registry directory."""
    return str(tmp_path / "registry")


@pytest.fixture
def mcp_config_path(tmp_path):
    """Create a temporary MCP config file."""
    path = tmp_path / "mcp_config.json"
    path.write_text(json.dumps({"mcpServers": {}}))
    return str(path)


@pytest.fixture
def registry(registry_path, mcp_config_path) -> AgentRegistry:
    """Create an AgentRegistry with temp paths."""
    return AgentRegistry(
        registry_path=registry_path,
        mcp_config_path=mcp_config_path,
    )


@pytest.fixture
def sample_package(registry_path) -> SpecialistPackage:
    """Create a sample package and write it to available/."""
    pkg = SpecialistPackage(
        name="test-specialist",
        version="0.1.0",
        description="A test specialist",
        mcp_config={
            "name": "test-server",
            "command": "uvx",
            "args": ["test-mcp"],
        },
        tools=["list_items", "create_item", "delete_item"],
        tags=["test", "demo"],
    )
    # Write to available directory
    import os

    os.makedirs(os.path.join(registry_path, "available"), exist_ok=True)
    path = os.path.join(registry_path, "available", f"{pkg.name}.json")
    with open(path, "w") as f:
        f.write(pkg.model_dump_json(indent=2))

    return pkg


class TestSpecialistPackage:
    """Test SpecialistPackage model."""

    def test_defaults(self) -> None:
        pkg = SpecialistPackage(name="test")
        assert pkg.version == "0.1.0"
        assert pkg.tools == []
        assert pkg.dependencies == []
        assert pkg.tags == []

    def test_full(self) -> None:
        pkg = SpecialistPackage(
            name="gitlab-specialist",
            version="1.2.0",
            description="GitLab integration",
            tools=["list_repos", "create_mr"],
            tags=["git", "ci-cd"],
        )
        assert len(pkg.tools) == 2
        assert "git" in pkg.tags


class TestInstallation:
    """Test install/uninstall lifecycle."""

    @pytest.mark.asyncio
    async def test_install_success(
        self,
        registry: AgentRegistry,
        sample_package: SpecialistPackage,
    ) -> None:
        result = await registry.install("test-specialist")
        assert "✓ Installed" in result
        assert "test-specialist" in result

    @pytest.mark.asyncio
    async def test_install_not_found(self, registry: AgentRegistry) -> None:
        result = await registry.install("nonexistent")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_install_merges_mcp_config(
        self,
        registry: AgentRegistry,
        sample_package: SpecialistPackage,
        mcp_config_path: str,
    ) -> None:
        await registry.install("test-specialist")

        with open(mcp_config_path) as f:
            config = json.load(f)

        assert "test-server" in config["mcpServers"]

    @pytest.mark.asyncio
    async def test_install_moves_to_installed(
        self,
        registry: AgentRegistry,
        sample_package: SpecialistPackage,
    ) -> None:
        await registry.install("test-specialist")
        installed = registry.list_installed()
        assert len(installed) == 1
        assert installed[0].name == "test-specialist"

    @pytest.mark.asyncio
    async def test_uninstall_success(
        self,
        registry: AgentRegistry,
        sample_package: SpecialistPackage,
    ) -> None:
        await registry.install("test-specialist")
        result = await registry.uninstall("test-specialist")
        assert "✓ Uninstalled" in result
        assert len(registry.list_installed()) == 0

    @pytest.mark.asyncio
    async def test_uninstall_not_installed(
        self, registry: AgentRegistry
    ) -> None:
        result = await registry.uninstall("nonexistent")
        assert "not installed" in result

    @pytest.mark.asyncio
    async def test_uninstall_removes_mcp_config(
        self,
        registry: AgentRegistry,
        sample_package: SpecialistPackage,
        mcp_config_path: str,
    ) -> None:
        await registry.install("test-specialist")
        await registry.uninstall("test-specialist")

        with open(mcp_config_path) as f:
            config = json.load(f)

        assert "test-specialist" not in config.get("mcpServers", {})


class TestDependencies:
    """Test dependency management."""

    @pytest.mark.asyncio
    async def test_install_missing_dependency(
        self, registry: AgentRegistry, registry_path: str
    ) -> None:
        import os

        pkg = SpecialistPackage(
            name="dependent-pkg",
            version="0.1.0",
            dependencies=["base-pkg"],
        )
        os.makedirs(os.path.join(registry_path, "available"), exist_ok=True)
        path = os.path.join(registry_path, "available", "dependent-pkg.json")
        with open(path, "w") as f:
            f.write(pkg.model_dump_json(indent=2))

        result = await registry.install("dependent-pkg")
        assert "Missing dependency" in result

    @pytest.mark.asyncio
    async def test_uninstall_blocked_by_dependent(
        self,
        registry: AgentRegistry,
        sample_package: SpecialistPackage,
        registry_path: str,
    ) -> None:
        import os

        # Install base package
        await registry.install("test-specialist")

        # Create and install a dependent package
        dep_pkg = SpecialistPackage(
            name="dependent-pkg",
            version="0.1.0",
            dependencies=["test-specialist"],
        )
        path = os.path.join(
            registry_path, "available", "dependent-pkg.json"
        )
        with open(path, "w") as f:
            f.write(dep_pkg.model_dump_json(indent=2))
        await registry.install("dependent-pkg")

        # Try to uninstall base — should be blocked
        result = await registry.uninstall("test-specialist")
        assert "depends on it" in result


class TestDiscovery:
    """Test listing and searching."""

    @pytest.mark.asyncio
    async def test_list_installed(
        self,
        registry: AgentRegistry,
        sample_package: SpecialistPackage,
    ) -> None:
        await registry.install("test-specialist")
        installed = registry.list_installed()
        assert len(installed) == 1
        assert installed[0].name == "test-specialist"

    def test_list_available(
        self,
        registry: AgentRegistry,
        sample_package: SpecialistPackage,
    ) -> None:
        available = registry.list_available()
        assert len(available) == 1
        assert available[0].name == "test-specialist"

    @pytest.mark.asyncio
    async def test_search_by_name(
        self,
        registry: AgentRegistry,
        sample_package: SpecialistPackage,
    ) -> None:
        results = registry.search("test")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_by_tag(
        self,
        registry: AgentRegistry,
        sample_package: SpecialistPackage,
    ) -> None:
        results = registry.search("demo")
        assert len(results) == 1

    def test_search_no_results(self, registry: AgentRegistry) -> None:
        results = registry.search("nonexistent")
        assert len(results) == 0
