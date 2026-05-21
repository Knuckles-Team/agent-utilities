#!/usr/bin/python
from __future__ import annotations

"""Unified Service Registry (CONCEPT:ORCH-1.4).

Central nervous system wiring all concept modules into the KG-driven
orchestration pipeline via lazy-load registration. Each module registers
its capabilities as discoverable services that the TopologyEngine and
KGTeamComposer can invoke at runtime.

Architecture::

    TopologyEngine / KGTeamComposer
           │
           ▼
    ServiceRegistry.discover(domain, capability)
           │
           ▼
    Lazy-loaded module function invocation
"""


import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ServiceDescriptor:
    """Describes a registered service capability."""

    module_path: str
    function_name: str
    capability: str
    domain: str = "general"
    layer: str = "core"
    description: str = ""
    _cached_fn: Callable[..., Any] | None = field(default=None, repr=False)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Lazily import and invoke the service function."""
        if self._cached_fn is None:
            try:
                mod = importlib.import_module(self.module_path)
                self._cached_fn = getattr(mod, self.function_name)
            except (ImportError, AttributeError) as e:
                logger.warning(
                    "Failed to load service %s.%s: %s",
                    self.module_path,
                    self.function_name,
                    e,
                )
                raise
        return self._cached_fn(*args, **kwargs)

    def get_class(self) -> type | None:
        """Lazily import and return the service class."""
        try:
            mod = importlib.import_module(self.module_path)
            return getattr(mod, self.function_name, None)
        except (ImportError, AttributeError) as e:
            logger.warning(
                "Failed to load class %s.%s: %s",
                self.module_path,
                self.function_name,
                e,
            )
            return None


# ---------------------------------------------------------------------------
# Service definitions: every concept module mapped to its entry point
# ---------------------------------------------------------------------------

_SERVICE_DEFINITIONS: list[dict[str, str]] = [
    # --- Layer 1: Core Orchestration ---
    {
        "module": "agent_utilities.graph.team_composer",
        "entry": "KGTeamComposer",
        "capability": "team_composition",
        "layer": "orchestration",
        "domain": "general",
        "desc": "KG-driven team assembly (ORCH-1.15)",
    },
    {
        "module": "agent_utilities.graph.topology_engine",
        "entry": "TopologyEngine",
        "capability": "topology_materialization",
        "layer": "orchestration",
        "domain": "general",
        "desc": "Dynamic graph materialization (ORCH-1.17)",
    },
    {
        "module": "agent_utilities.graph.state_checkpoint",
        "entry": "StateCheckpointer",
        "capability": "state_checkpoint",
        "layer": "orchestration",
        "domain": "general",
        "desc": "Session state persistence (ORCH-1.16)",
    },
    {
        "module": "agent_utilities.graph.persistent_agents",
        "entry": "PersistentAgentManager",
        "capability": "persistent_agents",
        "layer": "orchestration",
        "domain": "general",
        "desc": "Background agent lifecycle (ORCH-1.19)",
    },
    {
        "module": "agent_utilities.graph.adaptive_agent_router",
        "entry": "TopologicalRoutingPolicy",
        "capability": "topological_routing",
        "layer": "orchestration",
        "domain": "general",
        "desc": "KG-native routing (ORCH-1.18)",
    },
    {
        "module": "agent_utilities.graph.subagent_patterns",
        "entry": "SubagentPatternRouter",
        "capability": "subagent_patterns",
        "layer": "orchestration",
        "domain": "general",
        "desc": "Subagent lifecycle patterns (ORCH-1.6)",
    },
    {
        "module": "agent_utilities.security.execution_stability_engine",
        "entry": "RetryManager",
        "capability": "structured_retry",
        "layer": "orchestration",
        "domain": "general",
        "desc": "Structured retry with hooks (AHE-3.11)",
    },
    {
        "module": "agent_utilities.knowledge_graph.memory.elastic_context_manager",
        "entry": "ElasticContextManager",
        "capability": "context_filtering",
        "layer": "orchestration",
        "domain": "general",
        "desc": "Context-window pruning (KG-2.50)",
    },
    # --- Layer 2: Security & Guards ---
    {
        "module": "agent_utilities.security.threat_defense_engine",
        "entry": "PromptInjectionScanner",
        "capability": "prompt_scanning",
        "layer": "security",
        "domain": "general",
        "desc": "Prompt injection detection (OS-5.4/5.12)",
    },
    {
        "module": "agent_utilities.security.execution_stability_engine",
        "entry": "DoomLoopDetector",
        "capability": "doom_loop_detection",
        "layer": "security",
        "domain": "general",
        "desc": "Infinite loop prevention (OS-5.18)",
    },
    {
        "module": "agent_utilities.security.execution_stability_engine",
        "entry": "RepetitionGuard",
        "capability": "execution_stability_engine",
        "layer": "security",
        "domain": "general",
        "desc": "Tool repetition guard (OS-5.5)",
    },
    {
        "module": "agent_utilities.security.threat_defense_engine",
        "entry": "TopologicalScanner",
        "capability": "vulnerability_scanning",
        "layer": "security",
        "domain": "general",
        "desc": "Graph vulnerability scanning (OS-5.11)",
    },
    {
        "module": "agent_utilities.security.permissions_kernel",
        "entry": "PermissionsKernel",
        "capability": "permissions",
        "layer": "security",
        "domain": "general",
        "desc": "Auth & permissions (OS-5.1)",
    },
    {
        "module": "agent_utilities.security.threat_defense_engine",
        "entry": "GuardrailRule",
        "capability": "guardrails",
        "layer": "security",
        "domain": "general",
        "desc": "Input/output guardrails (OS-5.8)",
    },
    {
        "module": "agent_utilities.server.concurrency",
        "entry": "AsyncioConcurrencyManager",
        "capability": "concurrency_control",
        "layer": "security",
        "domain": "general",
        "desc": "Session concurrency (OS-5.3)",
    },
    # --- Layer 3: KG Intelligence ---
    {
        "module": "agent_utilities.core.cognitive_scheduler",
        "entry": "CognitiveScheduler",
        "capability": "scheduling",
        "layer": "kg_intelligence",
        "domain": "general",
        "desc": "Resource scheduling (OS-5.2)",
    },
    {
        "module": "agent_utilities.knowledge_graph.core.spectral_navigator",
        "entry": "SpectralClusterNavigator",
        "capability": "spectral_clustering",
        "layer": "kg_intelligence",
        "domain": "general",
        "desc": "Spectral cluster discovery (KG-2.34)",
    },
    {
        "module": "agent_utilities.knowledge_graph.core.blast_radius",
        "entry": "BlastRadiusAnalyzer",
        "capability": "blast_radius",
        "layer": "kg_intelligence",
        "domain": "general",
        "desc": "Symbol impact analysis (KG-2.35)",
    },
    {
        "module": "agent_utilities.knowledge_graph.memory.auto_similarity",
        "entry": "AutoSimilarityLinker",
        "capability": "auto_similarity",
        "layer": "kg_intelligence",
        "domain": "general",
        "desc": "Similarity edge creation (KG-2.36)",
    },
    {
        "module": "agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine",
        "entry": "GraphDistillationMigrator",
        "capability": "semantic_retrieval_engine",
        "layer": "kg_intelligence",
        "domain": "general",
        "desc": "RAG→KG migration (KG-2.40)",
    },
    {
        "module": "agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine",
        "entry": "KGNativeRetrievalRetriever",
        "capability": "unified_retrieval",
        "layer": "kg_intelligence",
        "domain": "general",
        "desc": "KG-native retrieval (KG-2.38)",
    },
    {
        "module": "agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine",
        "entry": "HybridSearchScorer",
        "capability": "hybrid_search",
        "layer": "kg_intelligence",
        "domain": "general",
        "desc": "Semantic+keyword search (KG-2.37)",
    },
    {
        "module": "agent_utilities.knowledge_graph.memory.knowledge_stability_engine",
        "entry": "KnowledgeStabilityEngine",
        "capability": "embedding_diagnostics",
        "layer": "kg_intelligence",
        "domain": "general",
        "desc": "Embedding quality & alignment diagnostics (AHE-3.6)",
    },
    {
        "module": "agent_utilities.knowledge_graph.core.formal_reasoning_core",
        "entry": "CausalVerifier",
        "capability": "formal_reasoning_core",
        "layer": "kg_intelligence",
        "domain": "general",
        "desc": "Structural causal models (KG-2.43)",
    },
    {
        "module": "agent_utilities.knowledge_graph.core.formal_reasoning_core",
        "entry": "BayesianBeliefPropagator",
        "capability": "formal_reasoning_core",
        "layer": "kg_intelligence",
        "domain": "general",
        "desc": "Bayesian belief propagation (KG-2.45)",
    },
    {
        "module": "agent_utilities.knowledge_graph.core.formal_reasoning_core",
        "entry": "RandomWalkExplorer",
        "capability": "graph_theory",
        "layer": "kg_intelligence",
        "domain": "general",
        "desc": "Formal graph algorithms (KG-2.41)",
    },
    {
        "module": "agent_utilities.knowledge_graph.core.formal_reasoning_core",
        "entry": "FormalStateMachine",
        "capability": "formal_reasoning_core",
        "layer": "kg_intelligence",
        "domain": "general",
        "desc": "Equivalence classes (KG-2.47)",
    },
    {
        "module": "agent_utilities.knowledge_graph.core.formal_reasoning_core",
        "entry": "FormalStateMachine",
        "capability": "state_invariants",
        "layer": "kg_intelligence",
        "domain": "general",
        "desc": "DFA invariant validation (KG-2.48)",
    },
    {
        "module": "agent_utilities.knowledge_graph.core.formal_reasoning_core",
        "entry": "MarkovTransitionModel",
        "capability": "markov_forecasting",
        "layer": "kg_intelligence",
        "domain": "general",
        "desc": "Markov chain prediction (KG-2.49)",
    },
    {
        "module": "agent_utilities.knowledge_graph.core.optimal_execution",
        "entry": "AlmgrenChrissDiscrete",
        "capability": "optimal_execution",
        "layer": "kg_intelligence",
        "domain": "finance",
        "desc": "Almgren-Chriss/Cartea-Jaimungal (KG-2.46)",
    },
    {
        "module": "agent_utilities.knowledge_graph.memory.knowledge_stability_engine",
        "entry": "KnowledgeStabilityEngine",
        "capability": "knowledge_stability",
        "layer": "kg_intelligence",
        "domain": "general",
        "desc": "Knowledge stability: EWC, drift, collapse, diagnostics (AHE-3.6)",
    },
    # --- Layer 4: Harness & Evolution ---
    {
        "module": "agent_utilities.harness.continuous_evaluation_engine",
        "entry": "TraceDistiller",
        "capability": "trace_distillation",
        "layer": "harness",
        "domain": "general",
        "desc": "Execution trace distillation (AHE-3.1)",
    },
    {
        "module": "agent_utilities.harness.variant_pool",
        "entry": "VariantPool",
        "capability": "prompt_evolution",
        "layer": "harness",
        "domain": "general",
        "desc": "Prompt mutation & selection (AHE-3.2)",
    },
    {
        "module": "agent_utilities.harness.continuous_evaluation_engine",
        "entry": "BacktestHarness",
        "capability": "backtesting",
        "layer": "harness",
        "domain": "finance",
        "desc": "Strategy backtesting (AHE-3.8)",
    },
    {
        "module": "agent_utilities.prompting.provider_adapter",
        "entry": "ProviderPromptAdapter",
        "capability": "prompt_adaptation",
        "layer": "harness",
        "domain": "general",
        "desc": "Provider-specific optimization (ECO-4.5)",
    },
    {
        "module": "agent_utilities.protocols.data_connector",
        "entry": "DataConnectorProtocol",
        "capability": "data_connectors",
        "layer": "harness",
        "domain": "general",
        "desc": "Market data connectors (ECO-4.4)",
    },
    # --- Layer 5: Research Pipeline ---
    {
        "module": "agent_utilities.automation.research_pipeline",
        "entry": "ResearchPipelineRunner",
        "capability": "research_pipeline",
        "layer": "research",
        "domain": "general",
        "desc": "End-to-end research ingestion (KG-2.11)",
    },
    {
        "module": "agent_utilities.knowledge_graph.orchestration.research_subagent",
        "entry": "ResearchSubagent",
        "capability": "research_subagent",
        "layer": "research",
        "domain": "general",
        "desc": "Isolated research context (KG-2.33)",
    },
    {
        "module": "agent_utilities.knowledge_graph.orchestration.research_orchestrator",
        "entry": "ResearchOrchestrator",
        "capability": "research_orchestration",
        "layer": "research",
        "domain": "general",
        "desc": "Daily research cycles (KG-2.39)",
    },
    # --- Layer 6: Finance Domain ---
    {
        "module": "agent_utilities.domains.finance.alpha_factors",
        "entry": "AlphaFactorLibrary",
        "capability": "alpha_factors",
        "layer": "domain",
        "domain": "finance",
        "desc": "Alpha factor library (KG-2.6)",
    },
    {
        "module": "agent_utilities.domains.finance.risk_manager",
        "entry": "RiskManager",
        "capability": "risk_management",
        "layer": "domain",
        "domain": "finance",
        "desc": "VaR & stress testing (KG-2.6)",
    },
    {
        "module": "agent_utilities.domains.finance.portfolio_optimizer",
        "entry": "MeanVarianceOptimizer",
        "capability": "portfolio_optimization",
        "layer": "domain",
        "domain": "finance",
        "desc": "Markowitz/Risk Parity (KG-2.6)",
    },
    {
        "module": "agent_utilities.domains.finance.versioned_orders",
        "entry": "OrderHistory",
        "capability": "versioned_orders",
        "layer": "domain",
        "domain": "finance",
        "desc": "Trading-as-Git (KG-2.6)",
    },
    {
        "module": "agent_utilities.domains.finance.market_data",
        "entry": "DataRegistry",
        "capability": "market_data",
        "layer": "domain",
        "domain": "finance",
        "desc": "Data provider abstraction (KG-2.6)",
    },
    {
        "module": "agent_utilities.domains.finance.payments",
        "entry": "X402PaymentClient",
        "capability": "ai_payments",
        "layer": "domain",
        "domain": "finance",
        "desc": "x402 AI payments (KG-2.6)",
    },
    {
        "module": "agent_utilities.domains.finance.profit_attribution",
        "entry": "ProfitAttributor",
        "capability": "profit_attribution",
        "layer": "domain",
        "domain": "finance",
        "desc": "P&L decomposition (KG-2.6)",
    },
    {
        "module": "agent_utilities.domains.finance.kronos_forecaster",
        "entry": "KronosForecaster",
        "capability": "time_series_forecast",
        "layer": "domain",
        "domain": "finance",
        "desc": "Kronos foundation model (KG-2.6)",
    },
    {
        "module": "agent_utilities.domains.finance.trading_swarm",
        "entry": "TradingSwarm",
        "capability": "trading_swarm",
        "layer": "domain",
        "domain": "finance",
        "desc": "8-role trading swarm (KG-2.6)",
    },
    {
        "module": "agent_utilities.domains.finance.visual_ta",
        "entry": "VisualTAEngine",
        "capability": "visual_ta",
        "layer": "domain",
        "domain": "finance",
        "desc": "Chart pattern detection (KG-2.6)",
    },
    {
        "module": "agent_utilities.domains.finance.market_feeds",
        "entry": "MarketFeedBus",
        "capability": "market_feeds",
        "layer": "domain",
        "domain": "finance",
        "desc": "Real-time market feeds (KG-2.6)",
    },
    {
        "module": "agent_utilities.domains.finance.strategy_export",
        "entry": "StrategyExporter",
        "capability": "strategy_export",
        "layer": "domain",
        "domain": "finance",
        "desc": "Multi-platform export (KG-2.6)",
    },
    {
        "module": "agent_utilities.domains.finance.research_autopilot",
        "entry": "ResearchAutopilot",
        "capability": "research_autopilot",
        "layer": "domain",
        "domain": "finance",
        "desc": "Automated hypothesis testing (KG-2.6)",
    },
    {
        "module": "agent_utilities.domains.finance.strategy_sharing",
        "entry": "StrategyRegistry",
        "capability": "strategy_sharing",
        "layer": "domain",
        "domain": "finance",
        "desc": "Community strategy marketplace (KG-2.6)",
    },
    {
        "module": "agent_utilities.knowledge_graph.core.markov_regime",
        "entry": "MarkovRegimeModel",
        "capability": "markov_regime_detection",
        "layer": "domain",
        "domain": "finance",
        "desc": "Markov chain market regime forecasting (KG-2.6)",
    },
    {
        "module": "agent_utilities.knowledge_graph.core.markov_regime",
        "entry": "HiddenMarkovRegimeModel",
        "capability": "hmm_regime_detection",
        "layer": "domain",
        "domain": "finance",
        "desc": "Gaussian HMM latent regime detection (KG-2.6)",
    },
    # --- Consolidated Engine Facades ---
    {
        "module": "agent_utilities.observability",
        "entry": "TelemetryEngine",
        "capability": "telemetry",
        "layer": "observability",
        "domain": "general",
        "desc": "Audit + token tracking facade (OS-5.6/5.7/5.9)",
    },
    {
        "module": "agent_utilities.knowledge_graph.core.topological_analysis_engine",
        "entry": "TopologicalAnalysisEngine",
        "capability": "topological_analysis",
        "layer": "topology",
        "domain": "general",
        "desc": "Community + analogy + spectral + blast radius (KG-2.5/2.15/2.34/2.35)",
    },
    {
        "module": "agent_utilities.knowledge_graph.research.research_intelligence_engine",
        "entry": "ResearchIntelligenceEngine",
        "capability": "research_intelligence",
        "layer": "research",
        "domain": "general",
        "desc": "Pipeline + citation + orchestration (KG-2.11/2.33/2.39)",
    },
    {
        "module": "agent_utilities.harness.agentic_evolution_engine",
        "entry": "AgenticEvolutionEngine",
        "capability": "agentic_evolution",
        "layer": "harness",
        "domain": "general",
        "desc": "Variant pool + skill evolution (AHE-3.2/ECO-4.1)",
    },
    {
        "module": "agent_utilities.harness.evaluation_engine",
        "entry": "EvaluationEngine",
        "capability": "evaluation",
        "layer": "harness",
        "domain": "general",
        "desc": "Reward decomposition + trace distillation (AHE-3.1/3.10/KG-2.4)",
    },
]


class ServiceRegistry:
    """Central service registry for KG-driven orchestration.

    CONCEPT:ORCH-1.4 — Unified Service Discovery

    Lazily loads and registers all concept modules, making them
    discoverable by the TopologyEngine and KGTeamComposer at runtime.
    Each service is a ``ServiceDescriptor`` with capability type, domain,
    and layer classification.

    Usage::

        registry = ServiceRegistry()
        registry.initialize()

        # Discover services by capability
        team_service = registry.get("team_composition")

        # Discover by domain
        finance_services = registry.discover(domain="finance")

        # Discover by layer
        security_services = registry.discover(layer="security")
    """

    _instance: ServiceRegistry | None = None

    def __init__(self) -> None:
        self._services: dict[str, ServiceDescriptor] = {}
        self._initialized = False

    @classmethod
    def instance(cls) -> ServiceRegistry:
        """Get or create the singleton registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self) -> int:
        """Register all service definitions.

        Returns:
            Number of services registered.
        """
        if self._initialized:
            return len(self._services)

        for defn in _SERVICE_DEFINITIONS:
            desc = ServiceDescriptor(
                module_path=defn["module"],
                function_name=defn["entry"],
                capability=defn["capability"],
                domain=defn.get("domain", "general"),
                layer=defn.get("layer", "core"),
                description=defn.get("desc", ""),
            )
            self._services[desc.capability] = desc

        # Load external legacy plugins
        try:
            import os

            from agent_utilities.graph.adapters.external_plugin_adapter import (
                ExternalPluginAdapter,
            )

            plugin_dir = os.environ.get(
                "EXTERNAL_PLUGIN_DIR", os.path.join(os.getcwd(), "plugins")
            )
            external_plugins = ExternalPluginAdapter.load_plugins_from_directory(
                plugin_dir
            )
            for plugin_desc in external_plugins:
                self._services[plugin_desc.capability] = plugin_desc
                logger.info(
                    "[CONCEPT:ORCH-1.4] Registered external plugin: %s",
                    plugin_desc.capability,
                )
        except Exception as e:
            logger.warning("[CONCEPT:ORCH-1.4] Failed to load external plugins: %s", e)

        self._initialized = True
        logger.info(
            "[CONCEPT:ORCH-1.4] Service registry initialized with %d services",
            len(self._services),
        )
        return len(self._services)

    def get(self, capability: str) -> ServiceDescriptor | None:
        """Get a service by capability name."""
        if not self._initialized:
            self.initialize()
        return self._services.get(capability)

    def discover(
        self,
        domain: str | None = None,
        layer: str | None = None,
    ) -> list[ServiceDescriptor]:
        """Discover services by domain and/or layer."""
        if not self._initialized:
            self.initialize()

        results = list(self._services.values())
        if domain:
            results = [
                s for s in results if s.domain == domain or s.domain == "general"
            ]
        if layer:
            results = [s for s in results if s.layer == layer]
        return results

    def list_capabilities(self) -> list[str]:
        """List all registered capability names."""
        if not self._initialized:
            self.initialize()
        return sorted(self._services.keys())

    def get_layer_summary(self) -> dict[str, int]:
        """Get count of services per layer."""
        if not self._initialized:
            self.initialize()
        summary: dict[str, int] = {}
        for s in self._services.values():
            summary[s.layer] = summary.get(s.layer, 0) + 1
        return summary

    def validate_loadable(self) -> tuple[list[str], list[str]]:
        """Validate which services can actually be imported.

        Returns:
            Tuple of (loadable, failed) capability names.
        """
        if not self._initialized:
            self.initialize()

        loadable: list[str] = []
        failed: list[str] = []

        for cap, desc in self._services.items():
            try:
                importlib.import_module(desc.module_path)
                loadable.append(cap)
            except Exception:
                failed.append(cap)

        return loadable, failed

    def register_with_kg(self, engine: Any) -> int:
        """Register all services as CallableResource nodes in the KG.

        Args:
            engine: The IntelligenceGraphEngine.

        Returns:
            Number of nodes registered.
        """
        if not self._initialized:
            self.initialize()

        count = 0
        for cap, desc in self._services.items():
            try:
                node_id = f"svc:{cap}"
                engine._upsert_node(
                    "CallableResource",
                    node_id,
                    {
                        "id": node_id,
                        "name": desc.description,
                        "type": "callable_resource",
                        "resource_type": "SERVICE",
                        "module_path": desc.module_path,
                        "entry_point": desc.function_name,
                        "capability": cap,
                        "domain": desc.domain,
                        "layer": desc.layer,
                    },
                )
                count += 1
            except Exception as e:
                logger.debug("Failed to register service '%s': %s", cap, e)

        logger.info("[CONCEPT:ORCH-1.4] Registered %d services with KG", count)
        return count
