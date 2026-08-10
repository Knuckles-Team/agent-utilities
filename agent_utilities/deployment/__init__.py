"""Deployment tooling — self-setup, complete config generation, and validation.

A thin layer that makes a *full* agent-utilities deployment reproducible from one
entry point: generate a complete, profile-seeded ``config.json`` covering every
:class:`~agent_utilities.core.config.AgentConfig` option, validate a deployment's
config completeness/health (``config_doctor``), and render a grouped reference of
all options. Composed by the ``setup-config`` CLI, the ``graph_configure`` MCP
actions, and the ``agent-utilities-deployment`` skill.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .codex_registration import (
    CODEX_GRAPHOS_COMMAND,
    CODEX_GRAPHOS_SERVER,
    CodexRegistrationError,
    graphos_stdio_spec,
    register_codex_graphos,
)
from .config_generator import (
    PROFILES,
    config_doctor,
    config_reference,
    generate_config,
    is_restart_required,
    write_config,
)
from .genesis_environments import (
    EnvironmentProfile,
    EnvironmentProfileError,
    MissingSecretReferenceError,
    list_environment_profiles,
    load_environment_profile,
    profile_summary,
    validate_environment_profile,
)
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

if TYPE_CHECKING:
    from .doctor import CHECKS, run_doctor
    from .preflight import run_preflight

_LAZY_SUBMODULES = {
    "doctor": ".doctor",
    "preflight": ".preflight",
}

__all__ = [
    "CHECKS",
    "CI_TEMPLATES",
    "CODEX_GRAPHOS_COMMAND",
    "CODEX_GRAPHOS_SERVER",
    "CodexRegistrationError",
    "EnvironmentProfile",
    "EnvironmentProfileError",
    "MissingSecretReferenceError",
    "PROFILES",
    "PROFILE_REPO_SETS",
    "STANDARD_REPOS",
    "RepoTemplate",
    "config_doctor",
    "config_reference",
    "generate_config",
    "graphos_stdio_spec",
    "is_restart_required",
    "list_environment_profiles",
    "load_environment_profile",
    "manifest_summary",
    "profile_summary",
    "provision_plan",
    "render_skeleton",
    "register_codex_graphos",
    "run_doctor",
    "run_preflight",
    "runner_plan",
    "standard_repos",
    "validate_environment_profile",
    "write_config",
]


def _load_submodule(name: str) -> Any:
    """Import and cache one fixed deployment submodule on first access."""
    module = import_module(_LAZY_SUBMODULES[name], __name__)
    globals()[name] = module
    return module


def __getattr__(name: str) -> Any:
    """Load doctor/preflight exports only when a caller actually requests them.

    ``python -m agent_utilities.deployment.doctor`` imports this package before
    executing its target module. Eagerly importing ``doctor`` here therefore
    pre-populated ``sys.modules`` and made :mod:`runpy` emit a RuntimeWarning.
    Keep the public facade intact without pre-importing the module entry point.
    """
    if name in _LAZY_SUBMODULES:
        return _load_submodule(name)
    if name in {"CHECKS", "run_doctor"}:
        doctor = _load_submodule("doctor")

        exports = {"CHECKS": doctor.CHECKS, "run_doctor": doctor.run_doctor}
        globals().update(exports)
        return exports[name]
    if name == "run_preflight":
        preflight = _load_submodule("preflight")

        globals()[name] = preflight.run_preflight
        return preflight.run_preflight
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose static and lazy public members to discovery/introspection callers."""
    return sorted(set(globals()) | set(__all__) | set(_LAZY_SUBMODULES))
