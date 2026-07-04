#!/usr/bin/python
from __future__ import annotations

"""Ecosystem Topology Map.

CONCEPT:AU-ECO.interop.ecosystem-topology-map — Ecosystem Topology Map

This module materializes the multi-repository agent ecosystem as a
first-class Knowledge Graph topology.  It scans the workspace for
``pyproject.toml`` files, extracts package metadata, and creates
:class:`EcosystemPackageNode` entries linked by ``DEPENDS_ON``,
``PROVIDES_CAPABILITY_TO``, and ``CONSUMES_FROM_KERNEL`` edges.

Key capabilities enabled by this module:

- **Impact radius computation**: Uses the transitive ``dependsOn`` OWL
  property to find all downstream packages affected by a change.
- **MCP coverage mapping**: Identifies which MCP tools are consumed by
  which frontends.
- **Package category inference**: Groups MCP servers into intelligent
  categories (Infrastructure, Media, Productivity, Data Science,
  DevOps, Communication) using keyword heuristics.

Architecture::

    ┌─────────────────────┐
    │  pyproject.toml(s)  │
    └─────────┬───────────┘
              │ scan
    ┌─────────▼───────────┐
    │ EcosystemTopology   │
    │     Builder         │
    ├─────────────────────┤
    │ - discover_packages │
    │ - build_dep_graph   │
    │ - categorize_mcp    │
    │ - impact_radius     │
    └─────────┬───────────┘
              │ persist
    ┌─────────▼───────────┐
    │  Knowledge Graph    │
    │  (EcosystemPackage  │
    │   Nodes + Edges)    │
    └─────────────────────┘

See Also:
    - :class:`EcosystemPackageNode` — Pydantic model for package nodes.
    - :mod:`agent_utilities.knowledge_graph.owl_bridge` — OWL promotion.
"""


import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class PackageCategory(StrEnum):
    """Intelligent category groupings for ecosystem packages.

    CONCEPT:AU-ECO.interop.ecosystem-topology-map — Ecosystem Topology Map
    """

    KERNEL = "kernel"
    FRONTEND = "frontend"
    INFRASTRUCTURE = "infrastructure"
    MEDIA = "media"
    PRODUCTIVITY = "productivity"
    DATA_SCIENCE = "data_science"
    DEVOPS = "devops"
    COMMUNICATION = "communication"
    FINANCE = "finance"
    SECURITY = "security"
    SKILLS = "skills"
    GENERAL = "general"


# Keyword-based MCP server categorization rules
_CATEGORY_RULES: dict[PackageCategory, list[str]] = {
    PackageCategory.INFRASTRUCTURE: [
        "container",
        "portainer",
        "docker",
        "kubernetes",
        "systems",
        "tunnel",
        "technitium",
        "home-assistant",
        "uptime",
    ],
    PackageCategory.MEDIA: [
        "media",
        "jellyfin",
        "audio",
        "transcriber",
        "owncast",
        "qbittorrent",
        "arr",
    ],
    PackageCategory.PRODUCTIVITY: [
        "atlassian",
        "plane",
        "nextcloud",
        "postiz",
        "mealie",
        "archivebox",
        "stirlingpdf",
    ],
    PackageCategory.DATA_SCIENCE: [
        "data-science",
        "vector",
        "scholarx",
    ],
    PackageCategory.DEVOPS: [
        "github",
        "gitlab",
        "repository",
        "servicenow",
        "langfuse",
    ],
    PackageCategory.COMMUNICATION: [
        "microsoft",
    ],
    PackageCategory.FINANCE: [
        "leanix",
    ],
    PackageCategory.SECURITY: [
        "searxng",
        "documentdb",
    ],
}


@dataclass
class PackageInfo:
    """Extracted metadata from a ``pyproject.toml`` file.

    CONCEPT:AU-ECO.interop.ecosystem-topology-map — Ecosystem Topology Map

    Attributes:
        name: The package name (from ``[project].name``).
        version: The package version string.
        path: Absolute path to the package directory.
        category: Inferred category for MCP servers.
        dependencies: List of direct dependency package names.
        description: Package description from pyproject.toml.
        is_kernel: True if this is the ``agent-utilities`` kernel package.
        is_frontend: True if this is a frontend package (TUI/WebUI).
        is_mcp_server: True if this is an MCP server package.
        is_skill_package: True if this is a skills package.
    """

    name: str = ""
    version: str = ""
    path: str = ""
    category: PackageCategory = PackageCategory.GENERAL
    dependencies: list[str] = field(default_factory=list)
    description: str = ""
    is_kernel: bool = False
    is_frontend: bool = False
    is_mcp_server: bool = False
    is_skill_package: bool = False


class EcosystemTopologyBuilder:
    """Builds and manages the ecosystem topology in the Knowledge Graph.

    CONCEPT:AU-ECO.interop.ecosystem-topology-map — Ecosystem Topology Map

    Scans a workspace directory tree for ``pyproject.toml`` files, extracts
    package metadata, infers inter-package dependencies, and creates
    :class:`EcosystemPackageNode` entries in the Knowledge Graph.

    Args:
        workspace_path: Root path of the multi-repo workspace.

    Example::

        builder = EcosystemTopologyBuilder("/home/apps/workspace/agent-packages")
        packages = builder.discover_packages()
        dep_graph = builder.build_dependency_graph(packages)
        impact = builder.get_impact_radius("agent-utilities", dep_graph)
        print(f"Changing agent-utilities affects {len(impact)} packages")
    """

    def __init__(self, workspace_path: str | Path) -> None:
        self.workspace_path = Path(workspace_path)

    def discover_packages(self, max_depth: int = 3) -> list[PackageInfo]:
        """Scan the workspace for packages via ``pyproject.toml`` files.

        CONCEPT:AU-ECO.interop.ecosystem-topology-map — Ecosystem Topology Map

        Args:
            max_depth: Maximum directory depth to search.

        Returns:
            List of discovered :class:`PackageInfo` instances.
        """
        packages: list[PackageInfo] = []
        seen_paths: set[str] = set()

        for depth in range(1, max_depth + 1):
            pattern = "/".join(["*"] * depth) + "/pyproject.toml"
            for toml_path in self.workspace_path.glob(pattern):
                pkg_dir = str(toml_path.parent)
                if pkg_dir in seen_paths:
                    continue
                seen_paths.add(pkg_dir)

                info = self._parse_pyproject(toml_path)
                if info and info.name:
                    packages.append(info)

        logger.info("Discovered %d ecosystem packages", len(packages))
        return packages

    def _parse_pyproject(self, toml_path: Path) -> PackageInfo | None:
        """Parse a ``pyproject.toml`` file and extract package metadata.

        Args:
            toml_path: Path to the pyproject.toml file.

        Returns:
            :class:`PackageInfo` or None if parsing fails.
        """
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                logger.debug("Neither tomllib nor tomli available")
                return None

        try:
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", toml_path, exc)
            return None

        project = data.get("project", {})
        name = project.get("name", "")
        if not name:
            return None

        version = project.get("version", "")
        description = project.get("description", "")
        raw_deps = project.get("dependencies", [])

        # Extract dependency names (strip version specifiers)
        deps: list[str] = []
        for dep in raw_deps:
            dep_name = re.split(r"[<>=!~;\[\s]", str(dep))[0].strip()
            if dep_name:
                deps.append(dep_name)

        info = PackageInfo(
            name=name,
            version=version,
            path=str(toml_path.parent),
            dependencies=deps,
            description=description,
        )

        # Classify package
        info.is_kernel = name == "agent-utilities"
        info.is_frontend = name in ("agent-terminal-ui", "agent-webui")
        info.is_skill_package = name in ("universal-skills", "skill-graphs")
        info.is_mcp_server = (
            not info.is_kernel
            and not info.is_frontend
            and not info.is_skill_package
            and "agent-utilities" in deps
        )

        # Categorize
        info.category = self._categorize_package(info)

        return info

    def _categorize_package(self, info: PackageInfo) -> PackageCategory:
        """Infer the category of a package from its name and metadata.

        CONCEPT:AU-ECO.interop.ecosystem-topology-map — Ecosystem Topology Map

        Args:
            info: The package info to categorize.

        Returns:
            The inferred :class:`PackageCategory`.
        """
        if info.is_kernel:
            return PackageCategory.KERNEL
        if info.is_frontend:
            return PackageCategory.FRONTEND
        if info.is_skill_package:
            return PackageCategory.SKILLS

        name_lower = info.name.lower()
        for category, keywords in _CATEGORY_RULES.items():
            if any(kw in name_lower for kw in keywords):
                return category

        return PackageCategory.GENERAL

    def build_dependency_graph(
        self, packages: list[PackageInfo]
    ) -> dict[str, list[str]]:
        """Build an inter-package dependency adjacency list.

        CONCEPT:AU-ECO.interop.ecosystem-topology-map — Ecosystem Topology Map

        Only includes edges where both source and target are ecosystem
        packages (not external PyPI dependencies).

        Args:
            packages: List of discovered packages.

        Returns:
            Adjacency list mapping package names to their ecosystem
            dependency names.
        """
        ecosystem_names = {p.name for p in packages}
        dep_graph: dict[str, list[str]] = {}

        for pkg in packages:
            internal_deps = [d for d in pkg.dependencies if d in ecosystem_names]
            dep_graph[pkg.name] = internal_deps

        return dep_graph

    def get_impact_radius(
        self,
        package_name: str,
        dep_graph: dict[str, list[str]],
    ) -> list[str]:
        """Compute the transitive impact radius of a package change.

        CONCEPT:AU-ECO.interop.ecosystem-topology-map — Ecosystem Topology Map

        Uses BFS over the *reverse* dependency graph to find all packages
        that directly or transitively depend on ``package_name``.  This
        leverages the same transitivity as the OWL ``dependsOn`` property.

        Args:
            package_name: The package that changed.
            dep_graph: Adjacency list from :meth:`build_dependency_graph`.

        Returns:
            List of all transitively affected package names (excluding
            the source package itself).

        Example::

            impact = builder.get_impact_radius("agent-utilities", dep_graph)
            # Returns: ["agent-terminal-ui", "agent-webui", "genius-agent", ...]
        """
        # Build reverse graph (dependents)
        reverse: dict[str, list[str]] = {name: [] for name in dep_graph}
        for pkg, deps in dep_graph.items():
            for dep in deps:
                if dep in reverse:
                    reverse[dep].append(pkg)

        # BFS from package_name
        visited: set[str] = set()
        queue = [package_name]
        while queue:
            current = queue.pop(0)
            for dependent in reverse.get(current, []):
                if dependent not in visited:
                    visited.add(dependent)
                    queue.append(dependent)

        return sorted(visited)

    def compute_mcp_coverage(self, packages: list[PackageInfo]) -> dict[str, list[str]]:
        """Map which MCP servers are consumed by which frontends.

        CONCEPT:AU-ECO.interop.ecosystem-topology-map — Ecosystem Topology Map

        Args:
            packages: List of discovered packages.

        Returns:
            Dict mapping frontend package names to lists of MCP server
            names they can consume.
        """
        mcp_servers = [p.name for p in packages if p.is_mcp_server]
        frontends = [p.name for p in packages if p.is_frontend]

        coverage: dict[str, list[str]] = {}
        for fe in frontends:
            # Frontends consume all MCP servers via the kernel
            coverage[fe] = mcp_servers

        return coverage

    def get_category_groups(self, packages: list[PackageInfo]) -> dict[str, list[str]]:
        """Group packages by their inferred category.

        CONCEPT:AU-ECO.interop.ecosystem-topology-map — Ecosystem Topology Map

        Args:
            packages: List of discovered packages.

        Returns:
            Dict mapping category names to lists of package names.
        """
        groups: dict[str, list[str]] = {}
        for pkg in packages:
            cat = pkg.category.value
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(pkg.name)

        return groups

    def persist_to_kg(self, packages: list[PackageInfo]) -> int:
        """Persist ecosystem topology nodes and edges to the Knowledge Graph.

        CONCEPT:AU-ECO.interop.ecosystem-topology-map — Ecosystem Topology Map

        Creates :class:`EcosystemPackageNode` entries and inter-package
        dependency edges.  Requires an active ``IntelligenceGraphEngine``.

        Args:
            packages: List of discovered packages to persist.

        Returns:
            Number of nodes created or updated.
        """
        try:
            from agent_utilities.knowledge_graph.core.engine import (
                IntelligenceGraphEngine,
            )
        except ImportError:
            logger.debug("IntelligenceGraphEngine not available")
            return 0

        engine = IntelligenceGraphEngine.get_active()
        if not engine or not engine.backend:
            logger.debug("No active KG engine for ecosystem persistence")
            return 0

        ecosystem_names = {p.name for p in packages}
        persisted = 0

        for pkg in packages:
            # Determine node type
            if pkg.is_kernel:
                node_type = "kernel_package"
            elif pkg.is_frontend:
                node_type = "frontend_package"
            elif pkg.is_mcp_server:
                node_type = "mcp_server_package"
            elif pkg.is_skill_package:
                node_type = "skill_package"
            else:
                node_type = "ecosystem_package"

            # Create or update node
            query = """
            MERGE (p:EcosystemPackage {id: $id})
            SET p.name = $name,
                p.version = $version,
                p.type = $type,
                p.category = $category,
                p.description = $description,
                p.path = $path,
                p.importance_score = $importance
            """
            importance = 1.0 if pkg.is_kernel else 0.8 if pkg.is_frontend else 0.5
            engine.backend.execute(
                query,
                {
                    "id": f"pkg:{pkg.name}",
                    "name": pkg.name,
                    "version": pkg.version,
                    "type": node_type,
                    "category": pkg.category.value,
                    "description": pkg.description,
                    "path": pkg.path,
                    "importance": importance,
                },
            )
            persisted += 1

            # Create dependency edges
            for dep in pkg.dependencies:
                if dep in ecosystem_names:
                    edge_query = """
                    MATCH (a:EcosystemPackage {id: $src})
                    MATCH (b:EcosystemPackage {id: $tgt})
                    MERGE (a)-[:DEPENDS_ON]->(b)
                    """
                    engine.backend.execute(
                        edge_query,
                        {"src": f"pkg:{pkg.name}", "tgt": f"pkg:{dep}"},
                    )

            # Create kernel consumption edges for non-kernel packages
            if not pkg.is_kernel and "agent-utilities" in pkg.dependencies:
                consume_query = """
                MATCH (a:EcosystemPackage {id: $src})
                MATCH (k:EcosystemPackage {id: $kernel})
                MERGE (a)-[:CONSUMES_FROM_KERNEL]->(k)
                """
                engine.backend.execute(
                    consume_query,
                    {"src": f"pkg:{pkg.name}", "kernel": "pkg:agent-utilities"},
                )

        logger.info("Persisted %d ecosystem packages to KG", persisted)
        return persisted

    def generate_topology_report(self, packages: list[PackageInfo]) -> str:
        """Generate a human-readable topology report.

        CONCEPT:AU-ECO.interop.ecosystem-topology-map — Ecosystem Topology Map

        Args:
            packages: List of discovered packages.

        Returns:
            Formatted markdown report of the ecosystem topology.
        """
        dep_graph = self.build_dependency_graph(packages)
        categories = self.get_category_groups(packages)
        coverage = self.compute_mcp_coverage(packages)

        lines: list[str] = [
            "# Ecosystem Topology Report",
            "",
            f"**Total Packages**: {len(packages)}",
            "",
            "## Package Categories",
            "",
        ]

        for cat, pkgs in sorted(categories.items()):
            lines.append(f"### {cat.replace('_', ' ').title()} ({len(pkgs)})")
            for p in sorted(pkgs):
                lines.append(f"- {p}")
            lines.append("")

        # Impact analysis for kernel
        if "agent-utilities" in dep_graph:
            impact = self.get_impact_radius("agent-utilities", dep_graph)
            lines.extend(
                [
                    "## Kernel Impact Radius",
                    "",
                    f"Changing `agent-utilities` affects **{len(impact)}** packages:",
                    "",
                ]
            )
            for p in impact:
                lines.append(f"- {p}")
            lines.append("")

        # MCP coverage
        lines.append("## Frontend MCP Coverage")
        lines.append("")
        for fe, servers in coverage.items():
            lines.append(f"### {fe}")
            lines.append(f"Consumes **{len(servers)}** MCP servers")
            lines.append("")

        return "\n".join(lines)


__all__ = [
    "EcosystemTopologyBuilder",
    "PackageCategory",
    "PackageInfo",
]
