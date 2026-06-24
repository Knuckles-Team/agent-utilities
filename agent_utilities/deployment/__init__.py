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

__all__ = [
    "CHECKS",
    "PROFILES",
    "config_doctor",
    "config_reference",
    "generate_config",
    "generate_mcp_config",
    "is_restart_required",
    "run_doctor",
    "run_preflight",
    "write_config",
]
