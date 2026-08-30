#!/usr/bin/python
"""Agent Utilities Core Module.

This module serves as the primary entry point for the agent-utilities package,
providing a unified interface for agent creation, graph orchestration, workspace
management, and various helper utilities.
"""

from importlib import import_module
from typing import Any

from agent_utilities._version import __version__
from agent_utilities.core.log_privacy import install_log_privacy_boundary

# ruff: noqa: E402, F401

install_log_privacy_boundary()

# D-EGK-1: assert this package was actually loaded from its live hostPath
# mount, not a stale image-baked copy left behind by a pythonX.Y mount-path
# drift. Import-time side effect is intentional -- see
# agent_utilities/core/live_mount_guard.py for why this is the one guard
# that still fires when the other two (a parity gate, a version-independent
# mount path) were bypassed or skipped.
live_mount_guard = import_module("agent_utilities.core.live_mount_guard")


# Lazy imports for all modules to avoid heavy import chains. Each tuple maps a
# relative module path to the names it exports; imports happen only on lookup.
_LAZY_MODULE_EXPORTS: tuple[tuple[str, str], ...] = (
    (
        ".base_utilities",
        "get_logger optional_import_block require_optional_import retrieve_package_name safe_load_model safe_save_model to_boolean to_dict to_float to_integer to_list ensure_package_installed",
    ),
    (".agent.factory", "create_agent_parser create_agent"),
    (".agent.discovery", "discover_agents discover_all_specialists"),
    (".core.config", "DEFAULT_GRAPH_PERSISTENCE_PATH"),
    (".core.embedding_utilities", "create_embedding_model"),
    (".core.model_factory", "create_model"),
    (".core.agentspec_catalog", "AgentSpecGenerator"),
    (".core.wasm_runner", "WasmAgentRunner"),
    (".core.cognitive_scheduler", "CognitiveScheduler"),
    (
        ".graph",
        "GraphState build_tag_env_map create_graph_agent create_master_graph get_graph_mermaid initialize_graph_from_workspace register_on_enter_hook register_on_exit_hook run_graph run_graph_stream run_orthogonal_regions validate_graph",
    ),
    (
        ".graph.reactive",
        "EventLedger BehaviorDispatcher reactive_behavior BudgetGuard BudgetTrippedException",
    ),
    (".prompting.builder", "build_system_prompt_from_workspace load_identity"),
    (".server", "create_agent_server"),
    (".gateway_client", "GatewayClient"),
    (".knowledge_graph.core.codemaps", "CodemapGenerator"),
    (".agent_chat.parser", "parse_codemap_mentions"),
    (
        ".core.workspace",
        "CORE_FILES append_to_md_file get_mcp_config_path get_workspace_path initialize_workspace list_workspace_files load_workspace_file read_md_file write_md_file write_workspace_file",
    ),
    (
        ".core.chat_persistence",
        "delete_chat_from_disk get_chat_from_disk list_chats_from_disk save_chat_to_disk",
    ),
    (
        ".models",
        "CodemapArtifact CodemapNode DiscoveredSpecialist ImplementationPlan NestedStructure PeriodicTask ProjectConstitution Spec StructuredPrompt Task Tasks",
    ),
    (".models.imodel", "DisplayComplexityBudget"),
    (".security.secrets_client", "SecretsClient create_secrets_client"),
    (".security.auth", "verify_credentials"),
    (".security.sandboxed_executor", "SandboxedExecutor SandboxLimits SandboxResult"),
    (".sdd", "SDDManager"),
    (
        ".harness.continuous_evaluation_engine",
        "EvalRunner EvalStrategy TestCase EvalResult InterpretabilityTestSuite InterpretabilityGrader",
    ),
    (".harness.imodel_evolver", "IModelEvolver ParetoFrontier"),
    (
        ".harness.engineering",
        "EngineeringPatternOrchestrator PatternType PatternResult",
    ),
    (".observability.token_tracker", "TokenUsageTracker TokenUsageRecord TokenBucket"),
    (".observability.audit_logger", "AuditLogger AuditRecord"),
    (
        ".observability.config_versioning",
        "AgentConfigVersionManager AgentConfigSnapshot",
    ),
    (
        ".observability.replay_engine",
        "DistributedReplayEngine ReplayManifest InteractionRecord",
    ),
    (".observability", "TelemetryEngine"),
    (".knowledge_graph.core.model_display", "ModelDisplayOptimizer"),
    (
        ".knowledge_graph.core.ecosystem_topology",
        "EcosystemTopologyBuilder PackageCategory PackageInfo",
    ),
    (
        ".knowledge_graph.core.synergy_engine",
        "SynergyEngine ConceptBridge PillarCoupling SynergyInsight",
    ),
    (".knowledge_graph.retrieval.chat_search", "ChatSearchResult search_sessions"),
    (
        ".knowledge_graph.core.agents_md",
        "load_agents_md inject_project_context find_agents_md",
    ),
    (".tools.jupyter_adapter", "JupyterKernelAdapter"),
    (".tools.sandbox_executor", "SandboxExecutor"),
    (".orchestration.distributed_coordinator", "DistributedCoordinator"),
    (".knowledge_graph.memory.agent_context", "SemanticCompactor"),
)
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    name: (module_name, name)
    for module_name, names in _LAZY_MODULE_EXPORTS
    for name in names.split()
}


def __getattr__(name: str) -> Any:
    """Resolve one explicitly exported symbol without eager module imports."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    return getattr(import_module(module_name, package=__name__), attribute_name)


# ── Graph Integration ────────────────────────────────────────────────
# Disabled by default to avoid import overhead during testing
# Can be enabled by setting ENABLE_GRAPH_INTEGRATION=true and calling initialize_graph_integration() explicitly

__all__ = [
    # Agent creation
    "create_agent",
    "create_agent_parser",
    "create_agent_server",
    "GatewayClient",
    # Graph orchestration
    "GraphState",
    "create_graph_agent",
    "create_master_graph",
    "run_graph",
    "run_graph_stream",
    "build_tag_env_map",
    "get_graph_mermaid",
    "validate_graph",
    "initialize_graph_from_workspace",
    # Workspace
    "CORE_FILES",
    "get_workspace_path",
    "get_mcp_config_path",
    "initialize_workspace",
    "load_workspace_file",
    "write_workspace_file",
    "list_workspace_files",
    "read_md_file",
    "write_md_file",
    "append_to_md_file",
    # Prompt / Identity
    "load_identity",
    "build_system_prompt_from_workspace",
    # Model factory
    "create_model",
    # A2A
    "discover_agents",
    "discover_all_specialists",
    # Chat persistence
    "save_chat_to_disk",
    "list_chats_from_disk",
    "get_chat_from_disk",
    "delete_chat_from_disk",
    # Config
    "DEFAULT_GRAPH_PERSISTENCE_PATH",
    # Base utilities
    "to_boolean",
    "to_integer",
    "to_float",
    "to_list",
    "to_dict",
    "retrieve_package_name",
    "get_logger",
    "ensure_package_installed",
    "optional_import_block",
    "require_optional_import",
    "safe_save_model",
    "safe_load_model",
    # Embedding
    "create_embedding_model",
    # HSM hooks
    "register_on_enter_hook",
    "register_on_exit_hook",
    "run_orthogonal_regions",
    # Models
    "PeriodicTask",
    "DiscoveredSpecialist",
    "ProjectConstitution",
    "Spec",
    "ImplementationPlan",
    "Tasks",
    "Task",
    "StructuredPrompt",
    "NestedStructure",
    # SDD
    "SDDManager",
    # Codemaps
    "CodemapNode",
    "CodemapArtifact",
    "CodemapGenerator",
    "parse_codemap_mentions",
    # Secrets & Auth (CONCEPT:AU-OS.config.secrets-authentication)
    "SecretsClient",
    "create_secrets_client",
    "verify_credentials",
    # MATE Integration — Evaluation (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort)
    "EvalRunner",
    "EvalStrategy",
    "TestCase",
    "EvalResult",
    # MATE Integration — Token Tracking (CONCEPT:AU-OS.config.secrets-authentication)
    "TokenUsageTracker",
    "TokenUsageRecord",
    "TokenBucket",
    # MATE Integration — Audit Logging (CONCEPT:AU-OS.config.secrets-authentication)
    "AuditLogger",
    "AuditRecord",
    # MATE Integration — Config Versioning (CONCEPT:AU-AHE.harness.evolutionary-aggregation)
    "AgentConfigVersionManager",
    "AgentConfigSnapshot",
    # Ecosystem Topology Map (CONCEPT:AU-ECO.messaging.native-backend-abstraction)
    "EcosystemTopologyBuilder",
    "PackageCategory",
    "PackageInfo",
    # Cross-Pillar Synergy Engine (CONCEPT:AU-KG.compute.cross-pillar-synergy)
    "SynergyEngine",
    "ConceptBridge",
    "PillarCoupling",
    "SynergyInsight",
    # Chat Search Facade (CONCEPT:AU-KG.memory.tiered-memory-caching)
    "ChatSearchResult",
    "search_sessions",
    # Agents MD Facade (CONCEPT:AU-KG.memory.tiered-memory-caching)
    "load_agents_md",
    "inject_project_context",
    "find_agents_md",
    # Engineering Patterns Facade (CONCEPT:AU-AHE.harness.evolutionary-aggregation)
    "EngineeringPatternOrchestrator",
    "PatternType",
    "PatternResult",
    # Agent-Runtimes Capabilities
    "JupyterKernelAdapter",
    "SandboxExecutor",
    "AgentSpecGenerator",
    # Reactive Framework (CONCEPT:AU-ORCH.reactive.event-sourcing-ledger)
    "EventLedger",
    "BehaviorDispatcher",
    "reactive_behavior",
    "BudgetGuard",
    "BudgetTrippedException",
    # WASM Agent Runner (CONCEPT:AU-OS.governance.wasm-micro-agent-sandbox)
    "WasmAgentRunner",
    # Cognitive Scheduler (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)
    "CognitiveScheduler",
    # Distributed Coordination (CONCEPT:AU-OS.host.homeostatic-recovery-daemon)
    "DistributedCoordinator",
    # Semantic Compactor (CONCEPT:AU-KG.query.vendor-agnostic-traversal)
    "SemanticCompactor",
    # Replay Engine (CONCEPT:AU-OS.observability.deterministic-replay)
    "DistributedReplayEngine",
    "ReplayManifest",
    "InteractionRecord",
    # Telemetry Engine (CONCEPT:AU-OS.config.secrets-authentication)
    "TelemetryEngine",
    # Sandboxed Executor (CONCEPT:AU-OS.observability.deterministic-replay)
    "SandboxedExecutor",
    "SandboxLimits",
    "SandboxResult",
]
