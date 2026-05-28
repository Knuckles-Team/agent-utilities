#!/usr/bin/python
"""Agent Utilities Core Module.

This module serves as the primary entry point for the agent-utilities package,
providing a unified interface for agent creation, graph orchestration, workspace
management, and various helper utilities.

Warning suppression is centralized here so every downstream import inherits
the filters without needing per-file boilerplate.
"""

import os
import warnings

# ruff: noqa: E402, F401

# ── Centralized warning suppression ──────────────────────────────────
# All library-level noise is filtered once at package init so that
# downstream modules (server, mcp_utilities, base_utilities, etc.)
# don't need their own copies.

# 1. requests/urllib3 version-mismatch noise
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from requests.exceptions import RequestsDependencyWarning

        warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    except ImportError:
        pass

warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")
warnings.filterwarnings("ignore", message=".*urllib3.*or charset_normalizer.*")
warnings.filterwarnings("ignore", message=r".*urllib3 v2.*only supports OpenSSL.*")

# 2. InsecureRequestWarning (emitted when ssl_verify=False)
try:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:  # nosec B110
    pass
warnings.filterwarnings("ignore", message=".*Unverified HTTPS request.*")

# 3. DeprecationWarnings from third-party libs
warnings.filterwarnings("ignore", category=DeprecationWarning, module="fastmcp")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="httpx")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

# 4. PydanticDeprecatedSince20 (noisy in older pydantic shims)
try:
    from pydantic import PydanticDeprecatedSince20

    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
except ImportError:
    pass

# ── End warning suppression ──────────────────────────────────────────


# Lazy imports for all modules to avoid heavy import chains
def __getattr__(name):
    if name in [
        "get_logger",
        "optional_import_block",
        "require_optional_import",
        "retrieve_package_name",
        "safe_load_model",
        "safe_save_model",
        "to_boolean",
        "to_dict",
        "to_float",
        "to_integer",
        "to_list",
        "ensure_package_installed",
    ]:
        from .base_utilities import (
            ensure_package_installed,
            get_logger,
            optional_import_block,
            require_optional_import,
            retrieve_package_name,
            safe_load_model,
            safe_save_model,
            to_boolean,
            to_dict,
            to_float,
            to_integer,
            to_list,
        )

        return locals()[name]
    elif name == "create_agent_parser":
        from .agent.factory import create_agent_parser

        return create_agent_parser
    elif name == "DEFAULT_GRAPH_PERSISTENCE_PATH":
        from .core.config import DEFAULT_GRAPH_PERSISTENCE_PATH

        return DEFAULT_GRAPH_PERSISTENCE_PATH
    elif name in ["discover_agents", "discover_all_specialists"]:
        from .agent.discovery import discover_agents, discover_all_specialists

        return (
            discover_agents if name == "discover_agents" else discover_all_specialists
        )
    elif name == "create_embedding_model":
        from .core.embedding_utilities import create_embedding_model

        return create_embedding_model
    elif name in [
        "GraphState",
        "build_tag_env_map",
        "create_agent",
        "create_master_graph",
        "get_graph_mermaid",
        "initialize_graph_from_workspace",
        "register_on_enter_hook",
        "register_on_exit_hook",
        "run_graph",
        "run_graph_stream",
        "run_orthogonal_regions",
        "validate_graph",
    ]:
        from .graph import (
            GraphState,
            build_tag_env_map,
            create_agent,
            create_master_graph,
            get_graph_mermaid,
            initialize_graph_from_workspace,
            register_on_enter_hook,
            register_on_exit_hook,
            run_graph,
            run_graph_stream,
            run_orthogonal_regions,
            validate_graph,
        )

        return locals()[name]
    elif name == "create_model":
        from .core.model_factory import create_model

        return create_model
    elif name in ["build_system_prompt_from_workspace", "load_identity"]:
        from .prompting.builder import build_system_prompt_from_workspace, load_identity

        return (
            build_system_prompt_from_workspace
            if name == "build_system_prompt_from_workspace"
            else load_identity
        )
    elif name == "create_agent_server":
        from .server import create_agent_server

        return create_agent_server
    elif name == "CodemapGenerator":
        from .knowledge_graph.core.codemaps import CodemapGenerator

        return CodemapGenerator
    elif name == "parse_codemap_mentions":
        from .agent_chat.parser import parse_codemap_mentions

        return parse_codemap_mentions
    elif name in [
        "CORE_FILES",
        "append_to_md_file",
        "get_mcp_config_path",
        "get_workspace_path",
        "initialize_workspace",
        "list_workspace_files",
        "load_workspace_file",
        "read_md_file",
        "write_md_file",
        "write_workspace_file",
    ]:
        from .core.workspace import (
            CORE_FILES,
            append_to_md_file,
            get_mcp_config_path,
            get_workspace_path,
            initialize_workspace,
            list_workspace_files,
            load_workspace_file,
            read_md_file,
            write_md_file,
            write_workspace_file,
        )

        return locals()[name]
    elif name in [
        "delete_chat_from_disk",
        "get_chat_from_disk",
        "list_chats_from_disk",
        "save_chat_to_disk",
    ]:
        from .core.chat_persistence import (
            delete_chat_from_disk,
            get_chat_from_disk,
            list_chats_from_disk,
            save_chat_to_disk,
        )

        return locals()[name]
    elif name in [
        "CodemapArtifact",
        "CodemapNode",
        "DiscoveredSpecialist",
        "ImplementationPlan",
        "NestedStructure",
        "PeriodicTask",
        "ProjectConstitution",
        "Spec",
        "StructuredPrompt",
        "Task",
        "Tasks",
    ]:
        from .models import (
            CodemapArtifact,
            CodemapNode,
            DiscoveredSpecialist,
            ImplementationPlan,
            NestedStructure,
            PeriodicTask,
            ProjectConstitution,
            Spec,
            StructuredPrompt,
            Task,
            Tasks,
        )

        return locals()[name]
    elif name in ["SecretsClient", "create_secrets_client"]:
        from .security.secrets_client import SecretsClient, create_secrets_client

        return SecretsClient if name == "SecretsClient" else create_secrets_client
    elif name == "verify_credentials":
        from .security.auth import verify_credentials

        return verify_credentials
    elif name == "SDDManager":
        from .sdd import SDDManager

        return SDDManager
    elif name in ["EvalRunner", "EvalStrategy", "TestCase", "EvalResult"]:
        from .harness.continuous_evaluation_engine import (
            EvalResult,
            EvalRunner,
            EvalStrategy,
            TestCase,
        )

        return locals()[name]
    elif name in ["TokenUsageTracker", "TokenUsageRecord", "TokenBucket"]:
        from .observability.token_tracker import (
            TokenBucket,
            TokenUsageRecord,
            TokenUsageTracker,
        )

        return locals()[name]
    elif name in ["AuditLogger", "AuditRecord"]:
        from .observability.audit_logger import AuditLogger, AuditRecord

        return AuditLogger if name == "AuditLogger" else AuditRecord
    elif name in ["GuardrailEngine", "GuardrailRule", "GuardrailAction"]:
        from .security.threat_defense_engine import (
            GuardrailAction,
            GuardrailEngine,
            GuardrailRule,
        )

        return locals()[name]
    elif name in ["AgentConfigVersionManager", "AgentConfigSnapshot"]:
        from .observability.config_versioning import (
            AgentConfigSnapshot,
            AgentConfigVersionManager,
        )

        return (
            AgentConfigVersionManager
            if name == "AgentConfigVersionManager"
            else AgentConfigSnapshot
        )
    elif name in [
        "IModelEvolver",
        "ParetoFrontier",
        "InterpretabilityTestSuite",
        "InterpretabilityGrader",
        "ModelDisplayOptimizer",
        "DisplayComplexityBudget",
    ]:
        if name in ["IModelEvolver", "ParetoFrontier"]:
            from .harness.imodel_evolver import IModelEvolver, ParetoFrontier

            return locals()[name]
        elif name in ["InterpretabilityTestSuite", "InterpretabilityGrader"]:
            from .harness.continuous_evaluation_engine import (
                InterpretabilityGrader,
                InterpretabilityTestSuite,
            )

            return locals()[name]
        else:
            from .knowledge_graph.core.model_display import ModelDisplayOptimizer
            from .models.imodel import DisplayComplexityBudget

            return (
                ModelDisplayOptimizer
                if name == "ModelDisplayOptimizer"
                else DisplayComplexityBudget
            )
    # Ecosystem Topology Map (CONCEPT:ECO-4.0)
    elif name in ["EcosystemTopologyBuilder", "PackageCategory", "PackageInfo"]:
        from .knowledge_graph.core.ecosystem_topology import (
            EcosystemTopologyBuilder,
            PackageCategory,
            PackageInfo,
        )

        return locals()[name]
    # Cross-Pillar Synergy Engine (CONCEPT:KG-2.4)
    elif name in ["SynergyEngine", "ConceptBridge", "PillarCoupling", "SynergyInsight"]:
        from .knowledge_graph.core.synergy_engine import (
            ConceptBridge,
            PillarCoupling,
            SynergyEngine,
            SynergyInsight,
        )

        return locals()[name]
    # Chat Search Facade (CONCEPT:KG-2.1)
    elif name in ["ChatSearchResult", "search_sessions"]:
        from .knowledge_graph.retrieval.chat_search import (
            ChatSearchResult,
            search_sessions,
        )

        return locals()[name]
    # Agents MD Facade (CONCEPT:KG-2.1)
    elif name in ["load_agents_md", "inject_project_context", "find_agents_md"]:
        from .knowledge_graph.core.agents_md import (
            find_agents_md,
            inject_project_context,
            load_agents_md,
        )

        return locals()[name]
    # Engineering Patterns Facade (CONCEPT:AHE-3.2)
    elif name in ["EngineeringPatternOrchestrator", "PatternType", "PatternResult"]:
        from .harness.engineering import (
            EngineeringPatternOrchestrator,
            PatternResult,
            PatternType,
        )

        return locals()[name]
    # Agent-Runtimes Capabilities (CONCEPT:ECO-4.0, ECO-4.12, AHE-3.23)
    elif name == "DurableExecutionManager":
        from .orchestration.durable_execution import DurableExecutionManager

        return DurableExecutionManager
    elif name == "JupyterKernelAdapter":
        from .tools.jupyter_adapter import JupyterKernelAdapter

        return JupyterKernelAdapter
    elif name == "SandboxExecutor":
        from .tools.sandbox_executor import SandboxExecutor

        return SandboxExecutor
    elif name == "AgentSpecGenerator":
        from .core.agentspec_catalog import AgentSpecGenerator

        return AgentSpecGenerator
    # Reactive Framework (CONCEPT:ORCH-1.28)
    elif name in [
        "EventLedger",
        "BehaviorDispatcher",
        "reactive_behavior",
        "BudgetGuard",
        "BudgetTrippedException",
    ]:
        from .graph.reactive import (
            BehaviorDispatcher,
            BudgetGuard,
            BudgetTrippedException,
            EventLedger,
            reactive_behavior,
        )

        return locals()[name]
    # WASM Agent Runner (CONCEPT:OS-5.4)
    elif name == "WasmAgentRunner":
        from .core.wasm_runner import WasmAgentRunner

        return WasmAgentRunner
    # Cognitive Scheduler (CONCEPT:OS-5.2)
    elif name == "CognitiveScheduler":
        from .core.cognitive_scheduler import CognitiveScheduler

        return CognitiveScheduler
    # Distributed Coordination (CONCEPT:OS-5.6)
    elif name == "DistributedCoordinator":
        from .orchestration.distributed_coordinator import DistributedCoordinator

        return DistributedCoordinator
    elif name == "RecoveryDaemon":
        from .orchestration.recovery_daemon import RecoveryDaemon

        return RecoveryDaemon
    # Semantic Compactor (CONCEPT:KG-2.20)
    elif name == "SemanticCompactor":
        from .knowledge_graph.memory.memory_compaction import SemanticCompactor

        return SemanticCompactor
    # Replay Engine (CONCEPT:OS-5.7)
    elif name in ["DistributedReplayEngine", "ReplayManifest", "InteractionRecord"]:
        from .observability.replay_engine import (
            DistributedReplayEngine,
            InteractionRecord,
            ReplayManifest,
        )

        return locals()[name]
    # Telemetry Engine (CONCEPT:OS-5.1)
    elif name == "TelemetryEngine":
        from .observability import TelemetryEngine

        return TelemetryEngine
    # Sandboxed Executor (CONCEPT:OS-5.7)
    elif name in ["SandboxedExecutor", "SandboxLimits", "SandboxResult"]:
        from .security.sandboxed_executor import (
            SandboxedExecutor,
            SandboxLimits,
            SandboxResult,
        )

        return locals()[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Set environment variables without using to_boolean
os.environ.setdefault("OTEL_ENABLE_OTEL", "false")
os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")
if os.environ.get("ENABLE_OTEL", "True").lower() in ["true", "1", "yes"]:
    os.environ.setdefault("OTEL_ENABLE_OTEL", "True")

# ── Graph Integration ────────────────────────────────────────────────
# Disabled by default to avoid import overhead during testing
# Can be enabled by setting ENABLE_GRAPH_INTEGRATION=true and calling initialize_graph_integration() explicitly

__version__ = "0.18.0"

__all__ = [
    # Agent creation (graph-based)
    "create_agent",
    "create_agent_parser",
    "create_agent_server",
    # Graph orchestration
    "GraphState",
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
    # Secrets & Auth (CONCEPT:OS-5.1)
    "SecretsClient",
    "create_secrets_client",
    "verify_credentials",
    # MATE Integration — Evaluation (CONCEPT:AHE-3.1)
    "EvalRunner",
    "EvalStrategy",
    "TestCase",
    "EvalResult",
    # MATE Integration — Token Tracking (CONCEPT:OS-5.1)
    "TokenUsageTracker",
    "TokenUsageRecord",
    "TokenBucket",
    # MATE Integration — Audit Logging (CONCEPT:OS-5.1)
    "AuditLogger",
    "AuditRecord",
    # MATE Integration — Guardrail Engine (CONCEPT:OS-5.1)
    "GuardrailEngine",
    "GuardrailRule",
    "GuardrailAction",
    # MATE Integration — Config Versioning (CONCEPT:AHE-3.2)
    "AgentConfigVersionManager",
    "AgentConfigSnapshot",
    # Ecosystem Topology Map (CONCEPT:ECO-4.0)
    "EcosystemTopologyBuilder",
    "PackageCategory",
    "PackageInfo",
    # Cross-Pillar Synergy Engine (CONCEPT:KG-2.4)
    "SynergyEngine",
    "ConceptBridge",
    "PillarCoupling",
    "SynergyInsight",
    # Chat Search Facade (CONCEPT:KG-2.1)
    "ChatSearchResult",
    "search_sessions",
    # Agents MD Facade (CONCEPT:KG-2.1)
    "load_agents_md",
    "inject_project_context",
    "find_agents_md",
    # Engineering Patterns Facade (CONCEPT:AHE-3.2)
    "EngineeringPatternOrchestrator",
    "PatternType",
    "PatternResult",
    # Agent-Runtimes Capabilities
    "DurableExecutionManager",
    "JupyterKernelAdapter",
    "SandboxExecutor",
    "AgentSpecGenerator",
    # Reactive Framework (CONCEPT:ORCH-1.28)
    "EventLedger",
    "BehaviorDispatcher",
    "reactive_behavior",
    "BudgetGuard",
    "BudgetTrippedException",
    # WASM Agent Runner (CONCEPT:OS-5.4)
    "WasmAgentRunner",
    # Cognitive Scheduler (CONCEPT:OS-5.2)
    "CognitiveScheduler",
    # Distributed Coordination (CONCEPT:OS-5.6)
    "DistributedCoordinator",
    "RecoveryDaemon",
    # Semantic Compactor (CONCEPT:KG-2.20)
    "SemanticCompactor",
    # Replay Engine (CONCEPT:OS-5.7)
    "DistributedReplayEngine",
    "ReplayManifest",
    "InteractionRecord",
    # Telemetry Engine (CONCEPT:OS-5.1)
    "TelemetryEngine",
    # Sandboxed Executor (CONCEPT:OS-5.7)
    "SandboxedExecutor",
    "SandboxLimits",
    "SandboxResult",
]
