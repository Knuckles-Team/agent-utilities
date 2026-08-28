"""The specialist-package value types — the leaf of the agent-registry package.

CONCEPT:AU-OS.config.agent-registry — Agent Registry (Package-Manager-Style Specialist Installation).

★ WHY THIS MODULE EXISTS (BUG-CX-004 / WD10-B-004)
``ContainerConfig``/``SpecialistPackage`` used to live in
:mod:`agent_utilities.core.registry.package_adapter`, which
:mod:`agent_utilities.core.default_catalog` imports at module scope to build the
38-package shipped catalog — while ``AgentRegistry._seed_defaults_if_empty()``
needs ``get_default_catalog()`` back from ``default_catalog``. That is a genuine
circular dependency, previously papered over with a function-local import.

Holding the shared *types* in a dependency-free leaf that both sides import is
the honest fix. Keep this module free of intra-package imports — its whole job
is to have no outgoing edges.

``package_adapter`` re-exports both names, so every existing
``from ...core.registry.package_adapter import SpecialistPackage`` keeps working.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ContainerConfig(BaseModel):
    """Container deployment configuration for a specialist package.

    References existing Dockerfiles and images from agent repos —
    does not duplicate them.  Used by ``container-manager-mcp``
    for containerized specialist deployment.

    Attributes:
        image: Container image reference (e.g. ``example/salesforce-agent:latest``).
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
            adaptive_agent_router that run as Docker/Podman containers.
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
