"""Ecosystem Package — Agent Configuration, Hooks, and Plugin Distribution.

CONCEPT:ECO-4.0 — Ecosystem Management

Provides the infrastructure for managing agent configurations, lifecycle
hooks, plugin bundles, and permission policies across the ecosystem.

Modules:
    - ``bridge`` — Routing between internal and external ecosystem components
    - ``hook_installer`` — Cross-agent hook distribution
    - ``agents_md_reflector`` — Stop hook for AGENTS.md self-improvement (ECO-4.17)
    - ``lint_enforcement_hook`` — Deterministic lint enforcement (ECO-4.11)
    - ``plugin_bundle`` — Unified plugin bundle system (ECO-4.12)
    - ``permission_policy`` — Permission deny/allow engine (ECO-4.13)
    - ``config_staleness_auditor`` — Periodic config staleness audit (ECO-4.21)
    - ``governance_workflow`` — Unified governance pipeline (ECO-4.22)
    - ``agent_manager_dashboard`` — Governance dashboard CLI
"""

from __future__ import annotations

from .agents_md_reflector import AgentsMdReflector, create_reflector_hook
from .bridge import EcosystemBridge
from .config_staleness_auditor import ConfigStalenessAuditor, StalenessReport
from .governance_workflow import (
    ChangeProposal,
    ChangeType,
    GovernanceDecision,
    GovernanceReport,
    GovernanceWorkflow,
)
from .lint_enforcement_hook import LintEnforcementHook, create_lint_hook
from .permission_policy import PermissionPolicyEngine, create_permission_hook
from .plugin_bundle import PluginBundle, PluginBundleManager

__all__ = [
    "AgentsMdReflector",
    "ChangeProposal",
    "ChangeType",
    "ConfigStalenessAuditor",
    "EcosystemBridge",
    "GovernanceDecision",
    "GovernanceReport",
    "GovernanceWorkflow",
    "LintEnforcementHook",
    "PermissionPolicyEngine",
    "PluginBundle",
    "PluginBundleManager",
    "StalenessReport",
    "create_lint_hook",
    "create_permission_hook",
    "create_reflector_hook",
]
