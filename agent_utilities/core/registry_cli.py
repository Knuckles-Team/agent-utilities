#!/usr/bin/python
"""CONCEPT:OS-5.0 — Agent Registry (Package-Manager-Style Specialist Installation).

Provides a CLI and programmatic API for installing, removing, and managing
specialist capabilities at runtime — the ``apt-get`` for agents.

Architecture:
    - **Registry index**: A local directory or remote manifest of available
      specialist packages, each containing MCP config fragments and metadata.
    - **Install flow**:
      1. Load specialist definition from registry
      2. Merge MCP config entries into active ``mcp_config.json``
      3. Hydrate KG specialist nodes via ``engine_registry.sync_mcp_agents()``
      4. Register new graph step nodes
      5. Trigger hot-reload via ``/mcp/reload`` (if server running)
    - **Uninstall flow**: Reverse of install — removes KG nodes, MCP config,
      and graph steps.

Integrates with:
    - CONCEPT:ECO-4.1 (Agent Tool System): MCP config merging
    - CONCEPT:KG-2.0 (KG OGM): Specialist node hydration
    - CONCEPT:ORCH-1.2 (Registry Cache): Cache invalidation on install/uninstall
    - ``systems-manager``: Privileged install operations

See docs/agent-registry.md §CONCEPT:OS-5.2.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

from ..models.knowledge_graph import (
    SpecialistPackageNode,
)

logger = logging.getLogger(__name__)


class ContainerConfig(BaseModel):
    """Container deployment configuration for a specialist package.

    References existing Dockerfiles and images from agent repos —
    does not duplicate them.  Used by ``container-manager-mcp``
    for containerized specialist deployment.

    Attributes:
        image: Container image reference (e.g. ``knucklessg1/salesforce-agent:latest``).
        compose_ref: Path to compose file relative to repo root (e.g. ``compose.yml``).
        ports: Port mappings as ``{host_port: container_port}``.
        env: Environment variables to inject into the container.
        labels: Container labels for discovery/filtering.
        health_check: Optional health check command.
    """

    image: str = ""
    compose_ref: str = ""
    ports: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=lambda: {"managed-by": "agent-os"})
    health_check: str = ""


class SpecialistPackage(BaseModel):
    """Definition of an installable specialist package.

    Each package provides a self-contained specialist capability:
    an MCP server definition, metadata for KG registration, and
    optional dependency declarations.

    Attributes:
        name: Package name (e.g. ``salesforce-specialist``).
        version: Semantic version string.
        description: Human-readable description.
        mcp_config: MCP server definition fragment to merge.
        specialist_metadata: Additional metadata for KG registration.
        tools: List of tool names this package provides.
        dependencies: Other packages this one depends on.
        tags: Searchable tags.
        container_config: Optional container deployment config for
            specialists that run as Docker/Podman containers.
    """

    name: str
    version: str = "0.1.0"
    description: str = ""
    mcp_config: dict[str, Any] = Field(default_factory=dict)
    specialist_metadata: dict[str, Any] = Field(default_factory=dict)
    tools: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    container_config: ContainerConfig | None = None


class AgentRegistry:
    """Package-manager-style specialist registry.

    CONCEPT:OS-5.0 — Agent Registry

    Manages the lifecycle of specialist packages: discovery, installation,
    uninstallation, and listing.  Packages are stored as JSON definitions
    in the registry directory and tracked as ``SpecialistPackageNode``
    entries in the KG.

    Args:
        registry_path: Path to the local specialist registry directory.
            Defaults to ``~/.agent-utilities/registry/``.
        mcp_config_path: Path to the active ``mcp_config.json``.
        engine: Optional KG engine for package tracking.
    """

    def __init__(
        self,
        registry_path: str | None = None,
        mcp_config_path: str | None = None,
        engine: IntelligenceGraphEngine | None = None,
    ) -> None:
        self.registry_path = registry_path or os.path.expanduser(
            "~/.agent-utilities/registry"
        )
        self.mcp_config_path = mcp_config_path
        self.engine = engine

        # Ensure registry directory exists
        os.makedirs(self.registry_path, exist_ok=True)
        os.makedirs(os.path.join(self.registry_path, "installed"), exist_ok=True)
        os.makedirs(os.path.join(self.registry_path, "available"), exist_ok=True)

        # Seed default catalog on first use
        self._seed_defaults_if_empty()

    # ── Default Catalog Seeding ────────────────────────────────────────

    def _seed_defaults_if_empty(self) -> None:
        """Seed the registry with the default catalog on first use.

        OS subsystem packages are auto-installed into ``installed/``.
        All other packages go into ``available/`` for on-demand install.
        Skips seeding if either directory already contains entries.
        """
        available_dir = os.path.join(self.registry_path, "available")
        installed_dir = os.path.join(self.registry_path, "installed")

        # Don't overwrite existing state
        if os.listdir(available_dir) or os.listdir(installed_dir):
            return

        try:
            from .default_catalog import get_default_catalog

            for pkg in get_default_catalog():
                if "os_subsystem" in pkg.tags:
                    target = os.path.join(installed_dir, f"{pkg.name}.json")
                else:
                    target = os.path.join(available_dir, f"{pkg.name}.json")
                with open(target, "w") as f:
                    f.write(pkg.model_dump_json(indent=2))

            logger.info(
                "Seeded default catalog: %d packages",
                len(get_default_catalog()),
            )
        except Exception as e:
            logger.debug("Default catalog seeding skipped: %s", e)

    def reseed_defaults(self) -> str:
        """Re-seed the registry with the latest default catalog.

        Overwrites ``available/`` entries for default packages without
        touching ``installed/`` or user-created packages.  Analogous
        to ``apt update`` refreshing the package index.

        Returns:
            A human-readable status message.
        """
        try:
            from .default_catalog import get_default_catalog
        except Exception as e:
            return f"Failed to load default catalog: {e}"

        available_dir = os.path.join(self.registry_path, "available")
        count = 0
        for pkg in get_default_catalog():
            if not self._is_installed(pkg.name):
                path = os.path.join(available_dir, f"{pkg.name}.json")
                with open(path, "w") as f:
                    f.write(pkg.model_dump_json(indent=2))
                count += 1
        return f"✓ Re-seeded {count} available packages"

    # ── Installation ───────────────────────────────────────────────────

    async def install(self, package_name: str) -> str:
        """Install a specialist package.

        Steps:
        1. Load package definition from the ``available/`` directory
        2. Merge its MCP config into the active config
        3. Register in the KG as a ``SpecialistPackageNode``
        4. Move definition to ``installed/``

        Args:
            package_name: Name of the package to install.

        Returns:
            A human-readable status message.
        """
        # Load package definition
        pkg = self._load_available(package_name)
        if not pkg:
            return f"Package '{package_name}' not found in registry"

        # Check dependencies
        for dep in pkg.dependencies:
            if not self._is_installed(dep):
                return f"Missing dependency: '{dep}' — install it first"

        # Merge MCP config
        if pkg.mcp_config and self.mcp_config_path:
            self._merge_mcp_config(package_name, pkg.mcp_config)

        # Register in KG
        self._register_in_kg(pkg)

        # Move to installed
        self._mark_installed(pkg)

        # Invalidate registry cache
        try:
            from ..graph.config_helpers import invalidate_registry_cache

            invalidate_registry_cache()
        except Exception:
            pass  # nosec B110

        logger.info(
            "Installed specialist package: %s v%s (%d tools)",
            pkg.name,
            pkg.version,
            len(pkg.tools),
        )

        return (
            f"✓ Installed '{pkg.name}' v{pkg.version} "
            f"({len(pkg.tools)} tools, MCP server: {pkg.mcp_config.get('name', 'N/A')})"
        )

    async def uninstall(self, package_name: str) -> str:
        """Uninstall a specialist package.

        Removes KG nodes, MCP config entries, and moves the package
        definition back to ``available/``.

        Args:
            package_name: Name of the package to uninstall.

        Returns:
            A human-readable status message.
        """
        pkg = self._load_installed(package_name)
        if not pkg:
            return f"Package '{package_name}' is not installed"

        # Check reverse dependencies
        for installed in self.list_installed():
            if package_name in installed.dependencies:
                return (
                    f"Cannot uninstall '{package_name}': "
                    f"'{installed.name}' depends on it"
                )

        # Remove from MCP config
        if self.mcp_config_path:
            self._remove_mcp_config(package_name)

        # Remove from KG
        self._unregister_from_kg(package_name)

        # Move back to available
        self._mark_uninstalled(pkg)

        # Invalidate cache
        try:
            from ..graph.config_helpers import invalidate_registry_cache

            invalidate_registry_cache()
        except Exception:
            pass  # nosec B110

        logger.info("Uninstalled specialist package: %s", pkg.name)
        return f"✓ Uninstalled '{pkg.name}'"

    # ── Discovery ──────────────────────────────────────────────────────

    def list_installed(self) -> list[SpecialistPackage]:
        """Return all installed specialist packages.

        Returns:
            List of installed ``SpecialistPackage`` instances.
        """
        installed_dir = os.path.join(self.registry_path, "installed")
        packages: list[SpecialistPackage] = []

        for filename in os.listdir(installed_dir):
            if filename.endswith(".json"):
                path = os.path.join(installed_dir, filename)
                try:
                    with open(path) as f:
                        packages.append(SpecialistPackage(**json.load(f)))
                except Exception as e:
                    logger.warning("Failed to load package %s: %s", filename, e)

        return packages

    def list_available(self) -> list[SpecialistPackage]:
        """Return all available (not yet installed) specialist packages.

        Returns:
            List of available ``SpecialistPackage`` instances.
        """
        available_dir = os.path.join(self.registry_path, "available")
        packages: list[SpecialistPackage] = []

        for filename in os.listdir(available_dir):
            if filename.endswith(".json"):
                path = os.path.join(available_dir, filename)
                try:
                    with open(path) as f:
                        packages.append(SpecialistPackage(**json.load(f)))
                except Exception as e:
                    logger.warning("Failed to load package %s: %s", filename, e)

        return packages

    def search(self, query: str) -> list[SpecialistPackage]:
        """Search available and installed packages by name or tag.

        Args:
            query: Search term (case-insensitive).

        Returns:
            List of matching ``SpecialistPackage`` instances.
        """
        query_lower = query.lower()
        results: list[SpecialistPackage] = []

        for pkg in self.list_available() + self.list_installed():
            if (
                query_lower in pkg.name.lower()
                or query_lower in pkg.description.lower()
                or any(query_lower in tag.lower() for tag in pkg.tags)
            ):
                results.append(pkg)

        return results

    # ── Private Helpers ────────────────────────────────────────────────

    def _load_available(self, name: str) -> SpecialistPackage | None:
        """Load a package definition from the available directory."""
        path = os.path.join(self.registry_path, "available", f"{name}.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as f:
                return SpecialistPackage(**json.load(f))
        except Exception as e:
            logger.error("Failed to load available package %s: %s", name, e)
            return None

    def _load_installed(self, name: str) -> SpecialistPackage | None:
        """Load a package definition from the installed directory."""
        path = os.path.join(self.registry_path, "installed", f"{name}.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as f:
                return SpecialistPackage(**json.load(f))
        except Exception as e:
            logger.error("Failed to load installed package %s: %s", name, e)
            return None

    def _is_installed(self, name: str) -> bool:
        """Check if a package is currently installed."""
        path = os.path.join(self.registry_path, "installed", f"{name}.json")
        return os.path.isfile(path)

    def _mark_installed(self, pkg: SpecialistPackage) -> None:
        """Move a package from available to installed."""
        src = os.path.join(self.registry_path, "available", f"{pkg.name}.json")
        dst = os.path.join(self.registry_path, "installed", f"{pkg.name}.json")

        # Write to installed (always, even if not in available)
        with open(dst, "w") as f:
            f.write(pkg.model_dump_json(indent=2))

        # Remove from available if present
        if os.path.isfile(src):
            os.remove(src)

    def _mark_uninstalled(self, pkg: SpecialistPackage) -> None:
        """Move a package from installed back to available."""
        src = os.path.join(self.registry_path, "installed", f"{pkg.name}.json")
        dst = os.path.join(self.registry_path, "available", f"{pkg.name}.json")

        with open(dst, "w") as f:
            f.write(pkg.model_dump_json(indent=2))

        if os.path.isfile(src):
            os.remove(src)

    def _merge_mcp_config(
        self, package_name: str, mcp_fragment: dict[str, Any]
    ) -> None:
        """Merge an MCP config fragment into the active config."""
        if not self.mcp_config_path:
            return

        try:
            config: dict[str, Any] = {}
            if os.path.isfile(self.mcp_config_path):
                with open(self.mcp_config_path) as f:
                    config = json.load(f)

            servers = config.setdefault("mcpServers", {})
            server_name = mcp_fragment.get("name", package_name)
            servers[server_name] = mcp_fragment

            with open(self.mcp_config_path, "w") as f:
                json.dump(config, f, indent=2)

            logger.info(
                "Merged MCP config for '%s' into %s",
                server_name,
                self.mcp_config_path,
            )
        except Exception as e:
            logger.error("Failed to merge MCP config: %s", e)

    def _remove_mcp_config(self, package_name: str) -> None:
        """Remove an MCP config entry for a package."""
        if not self.mcp_config_path or not os.path.isfile(self.mcp_config_path):
            return

        try:
            with open(self.mcp_config_path) as f:
                config = json.load(f)

            servers = config.get("mcpServers", {})
            if package_name in servers:
                del servers[package_name]
                with open(self.mcp_config_path, "w") as f:
                    json.dump(config, f, indent=2)
                logger.info("Removed MCP config for '%s'", package_name)
        except Exception as e:
            logger.error("Failed to remove MCP config: %s", e)

    def _register_in_kg(self, pkg: SpecialistPackage) -> None:
        """Register a specialist package in the Knowledge Graph."""
        if not self.engine:
            return

        try:
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            node = SpecialistPackageNode(
                id=f"pkg:{pkg.name}",
                name=f"Package: {pkg.name}",
                description=pkg.description,
                version=pkg.version,
                mcp_server_name=pkg.mcp_config.get("name", pkg.name),
                tool_count=len(pkg.tools),
                installed_at=ts,
                source_registry="local",
                importance_score=0.7,
                timestamp=ts,
                metadata={
                    "tools": pkg.tools,
                    "tags": pkg.tags,
                    "dependencies": pkg.dependencies,
                },
            )
            self.engine.graph.add_node(node.id, **node.model_dump())
            logger.debug("Registered package %s in KG", pkg.name)
        except Exception as e:
            logger.debug("Failed to register package %s in KG: %s", pkg.name, e)

    def _unregister_from_kg(self, package_name: str) -> None:
        """Remove a specialist package from the Knowledge Graph."""
        if not self.engine:
            return

        try:
            node_id = f"pkg:{package_name}"
            if node_id in self.engine.graph:
                self.engine.graph.remove_node(node_id)
                logger.debug("Removed package %s from KG", package_name)
        except Exception as e:
            logger.debug("Failed to remove package %s from KG: %s", package_name, e)
