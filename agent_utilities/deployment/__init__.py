"""Deployment tooling — self-setup, complete config generation, and validation.

A thin layer that makes a *full* agent-utilities deployment reproducible from one
entry point: generate a complete, profile-seeded ``config.json`` covering every
:class:`~agent_utilities.core.config.AgentConfig` option, validate a deployment's
config completeness/health (``config_doctor``), and render a grouped reference of
all options. Composed by the ``setup-config`` CLI, the ``graph_configure`` MCP
actions, and the ``agent-utilities-deployment`` skill.
"""

from .config_generator import (
    PROFILES,
    config_doctor,
    config_reference,
    generate_config,
    generate_mcp_config,
    is_restart_required,
    write_config,
)
from .doctor import CHECKS, run_doctor
from .preflight import run_preflight
from .repo_templates import (
    CI_TEMPLATES,
    PROFILE_REPO_SETS,
    STANDARD_REPOS,
    RepoTemplate,
    manifest_summary,
    provision_plan,
    render_skeleton,
    runner_plan,
    standard_repos,
)

__all__ = [
    "CHECKS",
    "CI_TEMPLATES",
    "PROFILES",
    "PROFILE_REPO_SETS",
    "STANDARD_REPOS",
    "RepoTemplate",
    "config_doctor",
    "config_reference",
    "generate_config",
    "generate_mcp_config",
    "is_restart_required",
    "manifest_summary",
    "provision_plan",
    "render_skeleton",
    "run_doctor",
    "run_preflight",
    "runner_plan",
    "standard_repos",
    "write_config",
]
