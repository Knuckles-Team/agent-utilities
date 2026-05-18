#!/usr/bin/python
from __future__ import annotations

"""Registry Graph Models Module.

This module defines the Pydantic models used for the hybrid graph representation
of the agent registry (NODE_AGENTS.md). It supports topological and semantic
discovery of adaptive_agent_router and their tools.
"""


from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class RegistryNodeType(StrEnum):
    """Enumeration of node types in the registry graph."""

    AGENT = "agent"
    TOOL = "tool"
    SKILL = "skill"
    PROMPT = "prompt"
    MEMORY = "memory"
    FILE = "file"
    SYMBOL = "symbol"
    MODULE = "module"
    CLIENT = "client"
    USER = "user"
    PREFERENCE = "preference"
    JOB = "job"
    LOG = "log"
    MESSAGE = "message"
    CHAT_SUMMARY = "chat_summary"
    THREAD = "thread"
    HEARTBEAT = "heartbeat"
    # Enhanced Memory & Reasoning
    REASONING_TRACE = "reasoning_trace"
    TOOL_CALL = "tool_call"
    ENTITY = "entity"
    EVENT = "event"
    REFLECTION = "reflection"
    GOAL = "goal"
    EPISODE = "episode"
    FACT = "fact"
    CONCEPT = "concept"
    CAPABILITY = "capability"
    # Callable Resources & Agents
    CALLABLE_RESOURCE = "callable_resource"
    TOOL_METADATA = "tool_metadata"
    SPAWNED_AGENT = "spawned_agent"
    SYSTEM_PROMPT = "system_prompt"
    # Self-Improvement & Learning
    OUTCOME_EVALUATION = "outcome_evaluation"
    CRITIQUE = "critique"
    SELF_EVALUATION = "self_evaluation"
    EXPERIMENT = "experiment"
    PROPOSED_SKILL = "proposed_skill"
    EXPERIENCE = "experience"
    # Knowledge Base
    KNOWLEDGE_BASE = "knowledge_base"
    ARTICLE = "article"
    RAW_SOURCE = "raw_source"
    KB_CONCEPT = "kb_concept"
    KB_FACT = "kb_fact"
    KB_INDEX = "kb_index"
    CHECKPOINT = "checkpoint"
    TEAM = "team"
    TASK = "task"
    POLICY = "policy"
    PROCESS_FLOW = "process_flow"
    PROCESS_STEP = "process_step"
    KNOWLEDGE_BASE_TOPIC = "knowledge_base_topic"
    SOURCE = "source"
    EVIDENCE = "evidence"
    PERSON = "person"
    PATTERN_TEMPLATE = "pattern_template"
    ORGANIZATION = "organization"
    ROLE = "role"
    PLACE = "place"
    PHASE = "phase"
    DECISION = "decision"
    INCIDENT = "incident"
    SYSTEM = "system"
    BELIEF = "belief"
    HYPOTHESIS = "hypothesis"
    PRINCIPLE = "principle"
    OBSERVATION = "observation"
    ACTION = "action"
    # Standard Ontology Node Types (BFO, Schema.org, FIBO)
    DOCUMENT = "document"
    CREATIVE_WORK = "creative_work"
    DATASET = "dataset"
    SOFTWARE_PROJECT = "software_project"
    MEDICAL_ENTITY = "medical_entity"
    PROCEDURE = "procedure"
    REGULATION = "regulation"
    FINANCIAL_INSTRUMENT = "financial_instrument"
    FINANCIAL_TRANSACTION = "financial_transaction"
    ACCOUNT = "account"
    # AHE (Agentic Harness Engineering) Node Types (CONCEPT:AHE-3.0)
    CHANGE_MANIFEST = "change_manifest"
    COMPONENT_EDIT_RECORD = "component_edit_record"
    EVIDENCE_RECORD = "evidence_record"
    CONSTRAINT_STATE = "constraint_state"
    # Emergent Architecture Node Types (CONCEPT:KG-2.0)
    SELF_MODEL = "memory_retriever"
    SWARM_COALITION = "swarm_coalition"
    PROPOSAL = "proposal"
    # Agentic Design Patterns Gap Concepts (CONCEPT:ORCH-1.1 through CONCEPT:AHE-3.2)
    PROMPT_CHAIN = "prompt_chain"
    RESOURCE_USAGE = "resource_usage"
    EVALUATION_RECORD = "evaluation_record"
    PRIORITIZED_TASK = "prioritized_task"
    KNOWLEDGE_GAP = "knowledge_gap"
    EXPLORATION_EXPERIMENT = "exploration_experiment"
    # Engineering Rules Engine (agent-rules-books integration)
    ENGINEERING_RULE = "engineering_rule"
    RULE_BOOK = "rule_book"
    # First-Principles Architecture (CONCEPT:AHE-3.3)
    TEAM_CONFIG = "team_config"
    AGENT_CAPABILITY = "agent_capability"
    # Agent OS Architecture (CONCEPT:OS-5.2)
    AGENT_PROCESS = "agent_process"
    AGENT_IDENTITY = "agent_identity"
    SPECIALIST_PACKAGE = "specialist_package"
    # Agent OS Infrastructure
    HOST = "host"
    INFRASTRUCTURE_TEMPLATE = "infrastructure_template"
    # Squeeze Evolve Routing (CONCEPT:ORCH-1.2)
    ROUTING_DECISION = "routing_decision"
    # Schema Packs (CONCEPT:KG-2.2)
    SCHEMA_PACK = "schema_pack"
    # Entity-Claim Extraction / MAGMA Epistemic (CONCEPT:KG-2.2)
    CLAIM = "claim"
    # Tiered Virtual Context/Memory blocks (CONCEPT:KG-2.1)
    VIRTUAL_CONTEXT_BLOCK = "virtual_context_block"
    # Quiet-STaR rationale persistence (CONCEPT:KG-2.1)
    QUIET_STAR_RATIONALE = "quiet_star_rationale"
    # Topological Mincut Partitioning (CONCEPT:KG-2.5)
    COMMUNITY = "community"
    # Heavy Thinking Orchestration (CONCEPT:AHE-3.5)
    TRAJECTORY = "trajectory"
    DELIBERATION = "deliberation"
    # Financial Trading Pipeline (CONCEPT:KG-2.6)
    TRADING_SIGNAL = "trading_signal"
    ORDER = "order"
    POSITION = "position"
    PORTFOLIO = "portfolio"
    STRATEGY = "strategy"
    TIME_SERIES_FORECAST = "time_series_forecast"
    VERSIONED_TRADE_COMMIT = "versioned_trade_commit"
    EXECUTION_GUARD = "execution_guard"
    UNIFIED_TRADING_ACCOUNT = "unified_trading_account"
    # Market Data Connector Protocol (CONCEPT:ECO-4.3)
    DATA_CONNECTOR = "data_connector"
    DATA_FETCH_RECORD = "data_fetch_record"
    # Swarm Preset Template Engine (CONCEPT:ORCH-1.4)
    SWARM_PRESET = "swarm_preset"
    SWARM_RUN = "swarm_run"
    SWARM_TASK_RECORD = "swarm_task_record"
    # Risk Scoring Ontology (CONCEPT:KG-2.6)
    RISK_ASSESSMENT = "risk_assessment"
    RISK_FACTOR = "risk_factor"
    RISK_MITIGATION = "risk_mitigation"
    # Backtest Evaluation Harness (CONCEPT:AHE-3.6)
    BACKTEST_RUN = "backtest_run"
    BACKTEST_METRIC = "backtest_metric"
    # Topological Analogy Engine (CONCEPT:KG-2.5)
    ANALOGY_MATCH = "analogy_match"
    # OWL-Driven Semantic Subsumption (CONCEPT:KG-2.2)
    SUBSUMPTION_ALIGNMENT = "subsumption_alignment"
    # Topological Vulnerability Scanner (CONCEPT:OS-5.1)
    TOPOLOGICAL_VULNERABILITY = "topological_vulnerability"
    # Agentic-iModels (CONCEPT:AHE-3.3, AHE-3.16, KG-2.17)
    IMODEL = "imodel"
    INTERPRETABILITY_TEST = "interpretability_test"
    MODEL_DISPLAY = "model_display"
    # Ecosystem Topology Map (CONCEPT:ECO-4.2)
    ECOSYSTEM_PACKAGE = "ecosystem_package"
    FRONTEND_PACKAGE = "frontend_package"
    KERNEL_PACKAGE = "kernel_package"
    MCP_SERVER_PACKAGE = "mcp_server_package"
    SKILL_PACKAGE = "skill_package"
    # Cross-Pillar Synergy Engine (CONCEPT:KG-2.4)
    SYNERGY_INSIGHT = "synergy_insight"
    # Knowledge Distillation Engine (CONCEPT:KG-2.2)
    IDEA_BLOCK = "idea_block"
    DISTILLATION_ROUND = "distillation_round"
    # Research Intelligence Sub-Agent (CONCEPT:KG-2.6)
    RESEARCH_SESSION = "research_session"
    CITATION_EDGE = "citation_edge"
    # Spectral Cluster Navigator (CONCEPT:KG-2.5)
    SPECTRAL_CLUSTER = "spectral_cluster"
    # Symbol Blast Radius Analyzer (CONCEPT:KG-2.5)
    BLAST_RADIUS_REPORT = "blast_radius_report"
    # Auto-Similarity Memory Graph (CONCEPT:KG-2.3)
    SIMILARITY_EDGE = "similarity_edge"
    # Hybrid Search Index (CONCEPT:KG-2.3)
    HYBRID_SEARCH_CONFIG = "hybrid_search_config"
    # Enhanced Doom-Loop Detector (CONCEPT:OS-5.0)
    DOOM_LOOP_INCIDENT = "doom_loop_incident"
    # RAG-KG Unification (CONCEPT:KG-2.3)
    UNIFIED_RAG_CONFIG = "unified_rag_config"
    # Research Orchestration (CONCEPT:KG-2.6)
    ORCHESTRATION_CYCLE = "orchestration_cycle"
    # Graph Distillation Migration (CONCEPT:KG-2.6)
    DISTILLATION_INDEX = "distillation_index"
    # Formal Graph Theory Primitives (CONCEPT:KG-2.6)
    MATH_FOUNDATION = "math_foundation"
    CRITICAL_PATH_RESULT = "critical_path_result"
    # Structural Causal Reasoning (CONCEPT:KG-2.6)
    CAUSAL_FACTOR = "causal_factor"
    CAUSAL_MODEL = "causal_model"
    # Optimal Execution Engine (CONCEPT:KG-2.6)
    EXECUTION_PLAN = "execution_plan"
    MARKET_MAKING_QUOTE = "market_making_quote"
    PAIRS_TRADE_SIGNAL = "pairs_trade_signal"
    # KG-Native Orchestration (CONCEPT:ORCH-1.1 through CONCEPT:ORCH-1.4)
    TOPOLOGY_TEMPLATE = "topology_template"
    SESSION_CHECKPOINT = "session_checkpoint"
    PERSISTENT_AGENT = "persistent_agent"
    TOPOLOGY_TRANSITION = "topology_transition"
    # Phase 2-5: Operationalized missing ontology nodes
    AGENT_SWARM = "agent_swarm"
    BUSINESS_UNIT = "business_unit"
    CHART_PATTERN = "chart_pattern"
    COMMUNICATION_MCP = "communication_mcp"
    COMPLIANCE_CONTROL = "compliance_control"
    CROSS_TENANT_INSIGHT = "cross_tenant_insight"
    DATA_SCIENCE_MCP = "data_science_mcp"
    DELEGATED_AUTHORITY = "delegated_authority"
    DEV_OPS_MCP = "dev_ops_mcp"
    ENTERPRISE_RESOURCE = "enterprise_resource"
    EQUIVALENCE_CLASS = "equivalence_class"
    EXECUTION_SIGNAL = "execution_signal"
    EXTERNAL_GRAPH_REFERENCE = "external_graph_reference"
    INFRASTRUCTURE_MCP = "infrastructure_mcp"
    KELLY_SIZING = "kelly_sizing"
    KRONOS_MODEL = "kronos_model"
    LSTM_NETWORK = "lstm_network"
    LEGAL_ENTITY = "legal_entity"
    MARKET_DATA_SOURCE = "market_data_source"
    MARKET_REGIME = "market_regime"
    # Markov Regime Detection (CONCEPT:KG-2.6)
    MARKOV_REGIME_STATE = "markov_regime_state"
    MARKOV_TRANSITION_MATRIX = "markov_transition_matrix"
    REGIME_SIGNAL = "regime_signal"
    MEDIA_MCP = "media_mcp"
    MERGE_REQUEST = "merge_request"
    NEURAL_NETWORK_MODEL = "neural_network_model"
    OPTIMIZATION_GOAL = "optimization_goal"
    ORDER_COMMIT_RECORD = "order_commit_record"
    ORDER_VERSION = "order_version"
    PARETO_FRONTIER_ENTRY = "pareto_frontier_entry"
    PARTIAL_ORDER = "partial_order"
    PAYMENT_BUDGET = "payment_budget"
    PAYMENT_PROOF_ENTITY = "payment_proof_entity"
    PORTFOLIO_ALLOCATION = "portfolio_allocation"
    PRODUCTIVITY_MCP = "productivity_mcp"
    PULL_REQUEST = "pull_request"
    RLM_ACTOR = "rlm_actor"
    REGULATORY_FRAMEWORK = "regulatory_framework"
    RESEARCH_HYPOTHESIS = "research_hypothesis"
    RISK_PROFILE = "risk_profile"
    SCIENTIFIC_ENTITY = "scientific_entity"
    SECURITY_CLEARANCE = "security_clearance"
    STATE_TRANSITION = "state_transition"
    STOCHASTIC_PROCESS = "stochastic_process"
    STRATEGY_CARD_ENTITY = "strategy_card_entity"
    STREAM_CHANNEL = "stream_channel"
    TRADING_STRATEGY = "trading_strategy"
    TRADING_SWARM_ENTITY = "trading_swarm_entity"
    TRANSITION_MATRIX = "transition_matrix"
    VAR_ESTIMATE = "var_estimate"
    VALUE_STREAM = "value_stream"
    # Context Graph Architecture (CONCEPT:KG-2.7)
    ARCHITECTURE_DECISION = "architecture_decision"
    ARCHIMATE_ELEMENT = "archimate_element"
    # KG-Driven Graph Materialization (CONCEPT:ORCH-1.20)
    AGENT_TEMPLATE = "agent_template"
    # Observational Memory Bridge (CONCEPT:KG-2.10)
    OBSERVATION_RECORD = "observation_record"
    USER_PROFILE = "user_profile"
    ACTIVE_CONTEXT = "active_context"
    MEMORY_MATERIALIZATION_EVENT = "memory_materialization_event"


class RegistryEdgeType(StrEnum):
    """Enumeration of relationship types in the registry graph."""

    PROVIDES = "provides"
    HAS_SKILL = "has_skill"
    USES_PROMPT = "uses_prompt"
    DEPENDS_ON = "depends_on"
    RELATED_TO = "related_to"
    MEMORY_OF = "memory_of"
    CALLS = "calls"
    IMPORTS = "imports"
    CONTAINS = "contains"
    INHERITS_FROM = "inherits_from"
    BELONGS_TO = "belongs_to"
    PREFERS = "prefers"
    EXECUTED_BY = "executed_by"
    PART_OF = "part_of"
    REPLY_TO = "reply_to"
    HEARTBEAT_OF = "heartbeat_of"
    # Enhanced Memory Relationships
    HAS_REASONING = "has_reasoning"
    USED_TOOL = "used_tool"
    AFFECTS = "affects"
    CAUSED_BY = "caused_by"
    INFLUENCED = "influenced"
    CONTRADICTS = "contradicts"
    UPDATED_BELIEF = "updated_belief"
    HAS_EVIDENCE = "has_evidence"
    TEMPORALLY_PRECEDES = "temporally_precedes"
    OCCURRED_DURING = "occurred_during"
    EVOLVED_INTO = "evolved_into"
    ENABLES = "enables"
    IMPLIES = "implies"
    INDEXES = "indexes"
    CONSOLIDATES_INTO = "consolidates_into"
    SELF_REFLECTS_ON = "self_reflects_on"
    # Callable Resource Relationships
    HAS_METADATA = "has_metadata"
    PROVIDES_CAPABILITY = "provides_capability"
    DELEGATES_TO = "delegates_to"
    DISCOVERED_VIA = "discovered_via"
    USED_RESOURCE = "used_resource"
    USES_BASE_PROMPT = "uses_base_prompt"
    EVOLVED_FROM = "evolved_from"
    PROVEN_WITH = "proven_with"
    DERIVED_FROM_PROMPT = "derived_from_prompt"
    # Self-Improvement Relationships
    PRODUCED_OUTCOME = "produced_outcome"
    SCORED_BY = "scored_by"
    GENERATED_CRITIQUE = "generated_critique"
    EXPERIENCED_DURING = "experienced_during"
    LED_TO = "led_to"
    SUPERSEDES = "supersedes"
    # Knowledge Base Relationships
    BELONGS_TO_KB = "belongs_to_kb"
    COMPILED_FROM = "compiled_from"
    ABOUT = "about"
    CITES = "cites"
    BACKLINKS = "backlinks"
    CONTRADICTS_KB = "contradicts_kb"
    INDEXES_KB = "indexes_kb"
    SNAPSHOT_OF = "snapshot_of"
    FORKED_FROM = "forked_from"
    ASSIGNED_TO_AGENT = "assigned_to_agent"
    BLOCKED_BY_TASK = "blocked_by_task"
    APPLIES_TO = "applies_to"
    HAS_START = "has_start"
    NEXT = "next"
    GROUNDED_IN = "grounded_in"
    REFERENCES = "references"
    AUTHORED = "authored"
    SUPPORTS = "supports"
    # --- KG V2 edges (see docs/KG_V2_DESIGN.md §3) ---
    HAS_ROLE = "has_role"
    PLAYED_ROLE_DURING = "played_role_during"
    OCCURRED_AT_PLACE = "occurred_at_place"
    OCCURRED_DURING_PHASE = "occurred_during_phase"
    DECIDED_BY = "decided_by"
    MOTIVATED_BY = "motivated_by"
    RESULTED_IN = "resulted_in"
    SUPPORTS_BELIEF = "supports_belief"
    CONTRADICTS_BELIEF = "contradicts_belief"
    GENERALIZES_TO = "generalizes_to"
    INSTANCE_OF_PATTERN = "instance_of_pattern"
    CAUSED_INCIDENT = "caused_incident"
    RESOLVED_INCIDENT = "resolved_incident"
    OWNS_SYSTEM = "owns_system"
    DEPENDS_ON_SYSTEM = "depends_on_system"
    PREDICTS = "predicts"
    OBSERVES = "observes"
    SUPERSEDES_BY = "supersedes_by"
    BELONGS_TO_ORGANIZATION = "belongs_to_organization"
    EMPLOYS = "employs"
    # OWL-related edges
    OBSERVED_BY = "observed_by"
    TRIGGERED_ACTION = "triggered_action"
    # Standard Ontology Edges (PROV-O, SKOS, Dublin Core, FIBO)
    WAS_GENERATED_BY = "was_generated_by"
    WAS_DERIVED_FROM = "was_derived_from"
    WAS_ATTRIBUTED_TO = "was_attributed_to"
    HAS_TEMPORAL_EXTENT = "has_temporal_extent"
    BROADER = "broader"
    NARROWER = "narrower"
    RELATED_CONCEPT = "related_concept"
    EXACT_MATCH = "exact_match"
    CLOSE_MATCH = "close_match"
    BROAD_MATCH = "broad_match"
    CREATOR = "creator"
    CITES_SOURCE = "cites_source"
    HAS_FINANCIAL_INSTRUMENT = "has_financial_instrument"
    EXECUTED_TRANSACTION = "executed_transaction"
    # AHE (Agentic Harness Engineering) Edges (CONCEPT:AHE-3.0)
    EDITED_IN_ROUND = "edited_in_round"
    PREDICTED_FIX = "predicted_fix"
    CAUSED_REGRESSION = "caused_regression"
    CONFIRMED_FIX = "confirmed_fix"
    VERIFIED_BY = "verified_by"
    ESCALATED_TO = "escalated_to"
    # Emergent Architecture Edges (CONCEPT:ORCH-1.0)
    VARIANT_OF = "variant_of"
    CURRENT_SELF_MODEL = "current_memory_retriever"
    SPAWNED_BY = "spawned_by"
    COORDINATED_BY = "coordinated_by"
    PROPOSED_FOR = "proposed_for"
    # Agentic Design Patterns Gap Edges (CONCEPT:ORCH-1.1 through CONCEPT:AHE-3.2)
    CHAIN_STEP = "chain_step"
    BRANCHES_TO = "branches_to"
    CONSUMED_RESOURCE = "consumed_resource"
    EVALUATED_WITH = "evaluated_with"
    CALIBRATED_AGAINST = "calibrated_against"
    BLOCKS = "blocks"
    ASSIGNED_TO_SPECIALIST = "assigned_to_specialist"
    TESTS_HYPOTHESIS = "tests_hypothesis"
    EXPLORED_GAP = "explored_gap"
    RESULTED_IN_DISCOVERY = "resulted_in_discovery"
    # Engineering Rules Engine Edges (agent-rules-books)
    CONFLICTS_WITH = "conflicts_with"
    CORRECTS_BIAS = "corrects_bias"
    APPLICABLE_WHEN = "applicable_when"
    # First-Principles Architecture Edges (CONCEPT:AHE-3.3)
    HAS_CAPABILITY = "has_capability"
    REUSED_TEAM = "reused_team"
    # Agent OS Architecture Edges (CONCEPT:OS-5.2)
    PREEMPTED_BY = "preempted_by"
    CHECKPOINTED_TO = "checkpointed_to"
    HAS_IDENTITY = "has_identity"
    AUTHORIZED_FOR = "authorized_for"
    INSTALLED_FROM = "installed_from"
    # Squeeze Evolve Routing (CONCEPT:ORCH-1.2)
    ROUTED_BY = "routed_by"
    AGGREGATED_FROM = "aggregated_from"
    # Schema Packs (CONCEPT:KG-2.2)
    USES_SCHEMA_PACK = "uses_schema_pack"
    # Entity-Claim Extraction / MAGMA Epistemic (CONCEPT:KG-2.2)
    BUILDS_ON = "builds_on"
    EXEMPLIFIES = "exemplifies"
    AUTHORED_BY = "authored_by"
    # Topological Mincut Partitioning (CONCEPT:KG-2.5)
    PART_OF_COMMUNITY = "part_of_community"
    # Temporal Drift (CONCEPT:AHE-3.4)
    DRIFTED_TO = "drifted_to"
    # Heavy Thinking Orchestration (CONCEPT:AHE-3.5)
    TRAJECTORY_OF = "trajectory_of"
    DELIBERATED_BY = "deliberated_by"
    AGREES_WITH = "agrees_with"
    DISAGREES_WITH = "disagrees_with"
    # Financial Trading Pipeline (CONCEPT:KG-2.6)
    GENERATED_SIGNAL = "generated_signal"
    PLACED_ORDER = "placed_order"
    OPENED_POSITION = "opened_position"
    BELONGS_TO_PORTFOLIO = "belongs_to_portfolio"
    EXECUTES_STRATEGY = "executes_strategy"
    BACKTESTED_WITH = "backtested_with"
    FORECASTED = "forecasted"
    VERSIONED_IN = "versioned_in"
    GUARDED_BY = "guarded_by"
    # Market Data Connector Protocol (CONCEPT:ECO-4.3)
    FETCHED_FROM = "fetched_from"
    FALLS_BACK_TO = "falls_back_to"
    # Swarm Preset Template Engine (CONCEPT:ORCH-1.4)
    PRESET_OF = "preset_of"
    RAN_PRESET = "ran_preset"
    TASK_DEPENDS_ON = "task_depends_on"
    # Risk Scoring Ontology (CONCEPT:KG-2.6)
    ASSESSED_RISK = "assessed_risk"
    HAS_RISK_FACTOR = "has_risk_factor"
    MITIGATED_BY = "mitigated_by"
    PROPAGATES_RISK_TO = "propagates_risk_to"
    # Backtest Evaluation Harness (CONCEPT:AHE-3.6)
    EVALUATED_STRATEGY = "evaluated_strategy"
    HAS_METRIC = "has_metric"
    COMPARED_TO_BENCHMARK = "compared_to_benchmark"
    # Topological Analogy Engine (CONCEPT:KG-2.5)
    ANALOGOUS_TO = "analogous_to"
    # OWL-Driven Semantic Subsumption (CONCEPT:KG-2.2)
    SUBSUMED_BY = "subsumed_by"
    # Topological Vulnerability Scanner (CONCEPT:OS-5.1)
    EXPOSES_VULNERABILITY = "exposes_vulnerability"
    # Agentic-iModels (CONCEPT:AHE-3.3, AHE-3.16, KG-2.17)
    EVOLVED_MODEL = "evolved_model"
    TESTED_INTERPRETABILITY = "tested_interpretability"
    DISPLAY_OF = "display_of"
    PARETO_DOMINATES = "pareto_dominates"
    # Ecosystem Topology Map (CONCEPT:ECO-4.2)
    PROVIDES_CAPABILITY_TO = "provides_capability_to"
    CONSUMES_FROM_KERNEL = "consumes_from_kernel"
    VISUALIZES = "visualizes"
    # Cross-Pillar Synergy Engine (CONCEPT:KG-2.4)
    HAS_SYNERGY_WITH = "has_synergy_with"
    # Knowledge Distillation Engine (CONCEPT:KG-2.2)
    DISTILLED_FROM = "distilled_from"
    PRODUCED_IN_ROUND = "produced_in_round"
    # Research Intelligence Sub-Agent (CONCEPT:KG-2.6)
    CITES_PAPER = "cites_paper"
    CITED_BY_PAPER = "cited_by_paper"
    DISCOVERED_IN_SESSION = "discovered_in_session"
    # Spectral Cluster Navigator (CONCEPT:KG-2.5)
    MEMBER_OF_CLUSTER = "member_of_cluster"
    CLUSTER_PARENT = "cluster_parent"
    # Symbol Blast Radius Analyzer (CONCEPT:KG-2.5)
    SYMBOL_USED_IN = "symbol_used_in"
    SYMBOL_DEFINED_IN = "symbol_defined_in"
    # Auto-Similarity Memory Graph (CONCEPT:KG-2.3)
    SIMILAR_TO = "similar_to"
    # Enhanced Doom-Loop Detector (CONCEPT:OS-5.0)
    TRIGGERED_DOOM_LOOP = "triggered_doom_loop"
    # RAG-KG Unification (CONCEPT:KG-2.3)
    SHORTCUT_RETRIEVAL = "shortcut_retrieval"
    # Research Orchestration (CONCEPT:KG-2.6)
    ORCHESTRATED_BY = "orchestrated_by"
    # KG-Native Orchestration (CONCEPT:ORCH-1.1 through CONCEPT:ORCH-1.4)
    TRANSITIONS_TO = "transitions_to"
    CHECKPOINTED_STATE = "checkpointed_state"
    SUBSCRIBED_TO = "subscribed_to"
    MATERIALIZED_FROM = "materialized_from"
    COMPOSED_TEAM = "composed_team"
    # Formal Graph Theory Primitives (CONCEPT:KG-2.6)
    CRITICAL_PATH_OF = "critical_path_of"
    COLORED_WITH = "colored_with"
    # Structural Causal Reasoning (CONCEPT:KG-2.6)
    CAUSES = "causes"
    CAUSAL_MECHANISM = "causal_mechanism"
    COUNTERFACTUAL_OF = "counterfactual_of"
    # Probabilistic Reasoning (CONCEPT:KG-2.6)
    BELIEF_UPDATE = "belief_update"
    # Optimal Execution (CONCEPT:KG-2.6)
    EXECUTED_VIA = "executed_via"
    PAIRS_WITH = "pairs_with"
    MAKES_MARKET_IN = "makes_market_in"
    # Phase 2-5: Operationalized missing ontology properties
    ABSORBED_INTO = "absorbed_into"
    ACTUAL_SCORE = "actual_score"
    ALT_LABEL = "alt_label"
    ANSWERS_QUESTION = "answers_question"
    APPLICATION_COUNT = "application_count"
    APPLIED_IN_TASK = "applied_in_task"
    ATTRIBUTED_BY = "attributed_by"
    BASELINE_SCORE = "baseline_score"
    BLAST_RADIUS_COUNT = "blast_radius_count"
    BLOCKS_AFTER_DISTILLATION = "blocks_after_distillation"
    BLOCKS_BEFORE_DISTILLATION = "blocks_before_distillation"
    BLOCKS_PROP = "blocks_prop"
    CITATION_DEPTH = "citation_depth"
    CLUSTER_COHERENCE = "cluster_coherence"
    CLUSTER_SCOPE_COUNT = "cluster_scope_count"
    COMPLETENESS_SCORE = "completeness_score"
    CONFLICT_WEIGHT = "conflict_weight"
    CORRECTIVE_PROMPT = "corrective_prompt"
    CORRECTNESS_SCORE = "correctness_score"
    COSINE_SIMILARITY = "cosine_similarity"
    COVERAGE_RATIO = "coverage_ratio"
    CURRENT_LEVEL = "current_level"
    CYCLE_INTERVAL_HOURS = "cycle_interval_hours"
    DECAY_LAMBDA = "decay_lambda"
    DERIVED_FROM_BOOK = "derived_from_book"
    DETECTED_PATTERN = "detected_pattern"
    DISPLAY_STRATEGY = "display_strategy"
    DISTILLATION_RECOMMENDATION = "distillation_recommendation"
    DISTILLATION_THRESHOLD = "distillation_threshold"
    EFFICACY_SCORE = "efficacy_score"
    EIGENGAP_VALUE = "eigengap_value"
    ENDED_AT_TIME = "ended_at_time"
    ENTRY_PRICE = "entry_price"
    EXIT_PRICE = "exit_price"
    EXPORTED_AS = "exported_as"
    FORECASTS = "forecasts"
    GENERATES_SIGNAL = "generates_signal"
    HAS_ALLOCATION = "has_allocation"
    HAS_ALPHA_FACTOR = "has_alpha_factor"
    HAS_CHAIN_STEP = "has_chain_step"
    HAS_CLEARANCE = "has_clearance"
    HAS_DELEGATED_AUTHORITY_FROM = "has_delegated_authority_from"
    HAS_ORDER_VERSION = "has_order_version"
    HAS_PRIORITY = "has_priority"
    HAS_RESOURCE_EFFICIENCY = "has_resource_efficiency"
    HAS_RISK_LIMIT = "has_risk_limit"
    HAS_SEVERITY_SCORE = "has_severity_score"
    IMPACT_SCORE = "impact_score"
    INTERPRETABILITY_SCORE = "interpretability_score"
    MAPPED_TO_EXTERNAL = "mapped_to_external"
    MAX_DRAWDOWN = "max_drawdown"
    MENTIONED_IN = "mentioned_in"
    MITIGATES_RISK = "mitigates_risk"
    MODELS_REGIME = "models_regime"
    MODIFIED_IN = "modified_in"
    PACKAGE_CATEGORY = "package_category"
    PACKAGE_VERSION = "package_version"
    PAID_VIA = "paid_via"
    PAPERS_INGESTED = "papers_ingested"
    PREDICTED_SCORE = "predicted_score"
    PREDICTIVE_RANK = "predictive_rank"
    PREF_LABEL = "pref_label"
    PROVIDED_BY = "provided_by"
    RELEVANCE_SCORE = "relevance_score"
    REPETITION_COUNT = "repetition_count"
    RULE_CLASS = "rule_class"
    RULE_TIER = "rule_tier"
    SAFETY_SCORE = "safety_score"
    SHARED_AS = "shared_as"
    SHARPE_RATIO = "sharpe_ratio"
    SHORTCUT_HIT_COUNT = "shortcut_hit_count"
    SHORTCUT_RETRIEVAL_FROM = "shortcut_retrieval_from"
    SIGNAL_TYPE = "signal_type"
    SIZED_BY = "sized_by"
    STALE_EDGE_COUNT = "stale_edge_count"
    STARTED_AT_TIME = "started_at_time"
    STREAMS_TO = "streams_to"
    SWARM_DECIDED_BY = "swarm_decided_by"
    TESTED_HYPOTHESIS = "tested_hypothesis"
    TOKEN_BUDGET_MAX = "token_budget_max"  # nosec
    TRUSTED_ANSWER = "trusted_answer"
    VIOLATION_COUNT = "violation_count"
    # Context Graph Architecture — ADR edges (CONCEPT:KG-2.7)
    IMPACTS_CONCEPT = "impacts_concept"
    ALTERNATIVES_TO = "alternatives_to"
    # KG-Driven Graph Materialization (CONCEPT:ORCH-1.20)
    REQUIRES_TOOLSET = "requires_toolset"
    COMPATIBLE_WITH_MODEL = "compatible_with_model"
    HAS_RESULT_TYPE = "has_result_type"
    # Observational Memory Bridge (CONCEPT:KG-2.10)
    MATERIALIZED_AS = "materialized_as"
    OBSERVED_THROUGH = "observed_through"
    REFLECTED_FROM = "reflected_from"
    PROFILE_ATTRIBUTE = "profile_attribute"


class RegistryNode(BaseModel):
    """Base class for all nodes in the registry graph."""

    id: str = Field(description="Unique identifier for the node")
    type: RegistryNodeType
    name: str
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance_score: float = 0.0
    timestamp: str | None = None
    embedding: list[float] | None = None
    is_permanent: bool = False
    ewc_fisher_diag: list[float] | None = None
    temporal_drift_score: float = 0.0


class CommunityNode(RegistryNode):
    """Represents an emergent topological cluster of knowledge.

    CONCEPT:KG-2.5 — Topological Mincut Partitioning
    """

    type: RegistryNodeType = RegistryNodeType.COMMUNITY
    coherence_score: float = 1.0
    member_count: int = 0
    community_tags: list[str] = Field(default_factory=list)


class TrajectoryNode(RegistryNode):
    """A persisted reasoning trajectory from a parallel thinker agent.

    CONCEPT:AHE-3.5 — Heavy Thinking Orchestration

    Represents the output of a single parallel reasoning attempt during
    the Heavy Thinking pipeline.  Multiple ``TrajectoryNode`` instances
    are linked to a shared query via ``TRAJECTORY_OF`` edges and are
    consumed by the deliberation phase.

    Attributes:
        thinker_id: Identifier of the parallel thinker that produced this trajectory.
        query_hash: SHA-256 hash of the original query for deduplication.
        answer: The final boxed or extracted answer from this trajectory.
        reasoning_summary: Pruned summary of the reasoning chain (thinking tokens removed).
        score: Evaluation score assigned during deliberation (0.0–1.0).
        is_correct: Whether this trajectory's answer was confirmed correct.
        model_id: The LLM model used by this thinker.
    """

    type: RegistryNodeType = RegistryNodeType.TRAJECTORY
    thinker_id: str = ""
    query_hash: str = ""
    answer: str = ""
    reasoning_summary: str = ""
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_correct: bool | None = None
    model_id: str = ""


class DeliberationNode(RegistryNode):
    """A deliberation synthesis result from the sequential deliberation phase.

    CONCEPT:AHE-3.5 — Heavy Thinking Orchestration

    Represents the output of the deliberation agent that critically
    analyzes multiple parallel trajectories and synthesizes a final
    consensus answer.  Links to consumed trajectories via
    ``DELIBERATED_BY`` edges and records agreement/disagreement via
    ``AGREES_WITH`` / ``DISAGREES_WITH`` edges.

    Attributes:
        trajectories_analyzed: Number of trajectory nodes consumed.
        consensus_answer: The final synthesized answer.
        confidence: Deliberation confidence score (0.0–1.0).
        critical_analysis: Free-text analysis of trajectory differences.
        iteration: The refinement iteration that produced this node (0 = first pass).
        model_id: The LLM model used for deliberation (may differ from thinkers).
    """

    type: RegistryNodeType = RegistryNodeType.DELIBERATION
    trajectories_analyzed: int = 0
    consensus_answer: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    critical_analysis: str = ""
    iteration: int = 0
    model_id: str = ""


class AgentNode(RegistryNode):
    """Represents a specialist agent in the registry."""

    type: RegistryNodeType = RegistryNodeType.AGENT
    agent_type: str  # prompt, mcp, a2a
    system_prompt: str = ""
    endpoint_url: str | None = None
    tool_count: int = 0


class ToolNode(RegistryNode):
    """Represents a specific tool provided by an agent."""

    type: RegistryNodeType = RegistryNodeType.TOOL
    mcp_server: str
    relevance_score: int = 0
    requires_approval: bool = False
    tags: list[str] = Field(default_factory=list)


class SkillNode(RegistryNode):
    """Represents a universal skill or tool graph."""

    type: RegistryNodeType = RegistryNodeType.SKILL
    package_name: str
    capabilities: list[str] = Field(default_factory=list)


class PromptNode(RegistryNode):
    """Represents an existing system prompt template."""

    type: RegistryNodeType = RegistryNodeType.PROMPT
    system_prompt: str
    json_blueprint: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)


class MemoryNode(RegistryNode):
    """Represents a historical memory or decision."""

    type: RegistryNodeType = RegistryNodeType.MEMORY
    category: str = "general"
    content: str = ""
    status: str = "ACTIVE"
    tags: list[str] = Field(default_factory=list)


class CodeNode(RegistryNode):
    """Represents a code entity (File, Class, Function)."""

    repo_path: str | None = None
    file_path: str | None = None
    language: str | None = None
    line_start: int | None = None
    line_end: int | None = None


class RegistryEdge(BaseModel):
    """Represents a relationship between two nodes in the registry graph."""

    source: str
    target: str
    type: RegistryEdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegistryGraphMetadata(BaseModel):
    """Aggregated metrics and metadata for the registry graph."""

    node_count: int = 0
    edge_count: int = 0
    agent_count: int = 0
    tool_count: int = 0
    last_sync: str = ""
    version: str = "1.0.0"


class SymbolMetadata(BaseModel):
    name: str
    type: str  # Class, Function, Method
    line: int
    docstring: str | None = None
    args: list[str] = Field(default_factory=list)
    return_type: str | None = None


class PhaseResult(BaseModel):
    name: str
    duration_ms: float
    output: Any
    success: bool = True
    error: str | None = None


class ResolutionContext(BaseModel):
    file_map: dict[str, str] = Field(default_factory=dict)  # Name to ID
    symbol_map: dict[str, str] = Field(default_factory=dict)  # Name to ID


from agent_utilities.core.config import (
    DEFAULT_ENABLE_KG_EMBEDDINGS,
)


class PipelineConfig(BaseModel):
    """Configuration for the Unified Intelligence Pipeline."""

    workspace_path: str
    enable_embeddings: bool = DEFAULT_ENABLE_KG_EMBEDDINGS
    persist_to_ladybug: bool = True
    ladybug_path: str | None = None
    embedding_provider: str | None = "llama-index"
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            ".git",
            "node_modules",
            "venv",
            "__pycache__",
            ".repo_graph",
            ".ladybug",
        ]
    )
    multimodal: bool = False
    incremental: bool = True
    # Knowledge Base settings
    enable_knowledge_base: bool = True
    kb_auto_ingest_skill_graphs: bool = False  # On-demand by default
    kb_chunk_size: int = 1024
    kb_extraction_model: str | None = None  # None = use default provider model
    kb_archive_age_days: int = 180
    kb_archive_importance_threshold: float = 0.3
    enable_workspace_sync: bool = True
    kb_auto_ingest_cloned_repos: bool = True
    # OWL Reasoning settings
    enable_owl_reasoning: bool = True
    owl_backend: str = "owlready2"
    owl_ontology_path: str | None = None
    owl_promotion_importance_threshold: float = 0.1
    owl_promotion_recency_days: int = 7
    # External Graph Endpoints settings
    enable_external_graphs: bool = True
    external_sparql_endpoints: list[str] = Field(default_factory=list)
    external_lpg_endpoints: dict[str, str] = Field(default_factory=dict)


class ExternalGraphReferenceNode(RegistryNode):
    """Reference to an external Knowledge Graph (SPARQL or LPG).

    CONCEPT:KG-2.9 — External Graph Federation
    """

    type: RegistryNodeType = RegistryNodeType.EXTERNAL_GRAPH_REFERENCE
    endpoint_url: str
    graph_type: str  # "sparql", "lpg", etc.
    properties: dict[str, Any] = Field(default_factory=dict)


class ClientNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.CLIENT


class UserNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.USER
    role: str = "user"


class PreferenceNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.PREFERENCE
    category: str
    value: str


class JobNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.JOB
    schedule: str
    command: str


class LogNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.LOG
    timestamp: str
    status: str
    output: str


class ThreadNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.THREAD
    title: str
    created_at: str


class MessageNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.MESSAGE
    role: str
    content: str
    timestamp: str


class ChatSummaryNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.CHAT_SUMMARY
    summary_text: str
    key_entities: list[str] = Field(default_factory=list)
    importance_score: float = 0.5
    original_count: int = 0


class HeartbeatNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.HEARTBEAT
    agent_name: str
    timestamp: str
    status: str
    issues: list[str] = Field(default_factory=list)
    raw_data: str = ""


# --- Enhanced Memory & Reasoning Nodes ---


class ReasoningTraceNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.REASONING_TRACE
    thought: str
    reflection: str | None = None
    confidence: float = 1.0


class ToolCallNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.TOOL_CALL
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None


class EntityNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.ENTITY
    entity_type: str  # Person, Org, Location, etc.
    properties: dict[str, Any] = Field(default_factory=dict)


class EventNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.EVENT
    timestamp: str
    event_type: str
    severity: str = "info"  # info, warning, error, critical
    payload: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    episode_id: str | None = None


class ReflectionNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.REFLECTION
    content: str
    confidence: float = 1.0


class GoalNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.GOAL
    goal_text: str
    status: str = "active"


class EpisodeNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.EPISODE
    timestamp: str
    source: str  # chat, tool, reflection
    end_time: str | None = None
    event_count: int = 0
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)


class FactNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.FACT
    content: str
    certainty: float = 1.0


class ConceptNode(RegistryNode):
    """Atomic knowledge unit (e.g. 'p53 gene', 'SN2 reaction')."""

    type: RegistryNodeType = RegistryNodeType.CONCEPT
    concept_id: str
    definition: str = ""
    is_permanent: bool = False


class SourceNode(RegistryNode):
    """Reference material (e.g. papers, journals, datasets)."""

    type: RegistryNodeType = RegistryNodeType.SOURCE
    source_id: str
    doi: str | None = None
    url: str | None = None
    publication_date: str | None = None
    authors: list[str] = Field(default_factory=list)


class EvidenceNode(RegistryNode):
    """Claims or findings extracted from sources."""

    type: RegistryNodeType = RegistryNodeType.EVIDENCE
    evidence_id: str
    claim: str
    confidence_score: float = 1.0


class ClaimNode(RegistryNode):
    """A discrete claim, assertion, or thesis extracted from documents.

    CONCEPT:KG-2.2 — Entity-Claim Extraction / MAGMA Epistemic View

    Represents verifiable assertions with confidence scoring and
    epistemic metadata. Claims participate in BUILDS_ON, CONTRADICTS,
    and EXEMPLIFIES relationships for epistemic reasoning.
    """

    type: RegistryNodeType = RegistryNodeType.CLAIM
    claim_text: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    claim_type: str = "assertion"  # assertion, decision, thesis, finding, opinion
    source_ids: list[str] = Field(default_factory=list)
    extracted_from: str | None = None  # source document/article ID
    domain: str | None = None  # business or knowledge domain
    is_verified: bool = False


class VirtualContextBlockNode(RegistryNode):
    """Tiered Virtual Context/Memory blocks.

    CONCEPT:KG-2.1 — Manages tiered memory caching for the graph, ranging
    from 'working_memory' to 'episodic' to 'semantic'.
    """

    type: RegistryNodeType = RegistryNodeType.VIRTUAL_CONTEXT_BLOCK
    tier: str = "working_memory"  # working_memory, episodic, semantic
    block_data: dict[str, Any] = Field(default_factory=dict)
    ttl_seconds: int | None = None


class QuietStarRationaleNode(RegistryNode):
    """Quiet-STaR rationale persistence in the Knowledge Graph.

    CONCEPT:KG-2.1 — Captures the internal chain-of-thought rationale
    used to arrive at decisions or plans, persistently stored for self-improvement.
    """

    type: RegistryNodeType = RegistryNodeType.QUIET_STAR_RATIONALE
    rationale: str
    context_tokens: int = 0
    decision_id: str | None = None
    outcome_reward: float | None = None


class PersonNode(RegistryNode):
    """Researchers, authors, or agents."""

    type: RegistryNodeType = RegistryNodeType.PERSON
    person_id: str
    expertise: list[str] = Field(default_factory=list)
    affiliation: str | None = None


class CapabilityNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.CAPABILITY


# --- Ecosystem Topology Nodes (CONCEPT:ECO-4.2) ---


class EcosystemPackageNode(RegistryNode):
    """Represents a package within the agent-utilities ecosystem.

    CONCEPT:ECO-4.2 — Ecosystem Topology Map

    Models kernel, frontend, MCP server, and skill packages as
    first-class Knowledge Graph nodes for dependency analysis,
    impact radius computation, and OWL reasoning.

    Attributes:
        type: Node type (ecosystem_package, frontend_package, etc.).
        package_name: The PyPI/package name.
        version: Semantic version string.
        category: Intelligent category classification.
        package_path: Filesystem path to the package root.
        is_kernel: True for the agent-utilities kernel.
        is_frontend: True for TUI/WebUI packages.
        is_mcp_server: True for MCP server packages.
        is_skill_package: True for universal-skills/skill-graphs.
        dependency_names: List of ecosystem-internal dependency names.
    """

    type: RegistryNodeType = RegistryNodeType.ECOSYSTEM_PACKAGE
    package_name: str = ""
    version: str = ""
    category: str = "general"
    package_path: str = ""
    is_kernel: bool = False
    is_frontend: bool = False
    is_mcp_server: bool = False
    is_skill_package: bool = False
    dependency_names: list[str] = Field(default_factory=list)


class SynergyInsightNode(RegistryNode):
    """A discovered cross-pillar synergy persisted in the KG.

    CONCEPT:KG-2.4 — Cross-Pillar Synergy Engine

    Attributes:
        source_concept: Source concept ID (e.g., ``AHE-3.5``).
        target_concept: Target concept ID (e.g., ``KG-2.0``).
        relationship_type: The suggested or existing relationship.
        confidence: Synergy confidence score (0.0–1.0).
        rationale: Explanation of the synergy.
        pillar_a: Primary pillar of the source concept.
        pillar_b: Primary pillar of the target concept.
    """

    type: RegistryNodeType = RegistryNodeType.SYNERGY_INSIGHT
    source_concept: str = ""
    target_concept: str = ""
    relationship_type: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    pillar_a: str = ""
    pillar_b: str = ""


# --- Callable Resources & Agent Nodes ---


class ToolMetadataNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.TOOL_METADATA
    tags: list[str] = Field(default_factory=list)
    prompt_template: str | None = None
    source: str = ""
    capabilities: list[str] = Field(default_factory=list)
    resources: dict[str, Any] = Field(default_factory=dict)


class AnalogyMatchNode(RegistryNode):
    """Represents a topological analogy found across domains.

    CONCEPT:KG-2.5 — Topological Analogy Engine
    """

    type: RegistryNodeType = RegistryNodeType.ANALOGY_MATCH
    target_domain: str
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_nodes: int = 0
    analogy_rationale: str = ""


class SubsumptionAlignmentNode(RegistryNode):
    """Represents a zero-shot semantic subsumption alignment.

    CONCEPT:KG-2.2 — OWL-Driven Semantic Subsumption
    """

    type: RegistryNodeType = RegistryNodeType.SUBSUMPTION_ALIGNMENT
    source_entity_id: str
    inferred_parent_class: str
    inferred_lineage: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TopologicalVulnerabilityNode(RegistryNode):
    """Represents a structural vulnerability found in the execution graph.

    CONCEPT:OS-5.1 — Topological Vulnerability Scanner
    """

    type: RegistryNodeType = RegistryNodeType.TOPOLOGICAL_VULNERABILITY
    vulnerability_type: str
    severity: str = "medium"
    detected_pattern: str = ""
    mitigation_strategy: str = ""


class ArchitectureDecisionRecord(RegistryNode):
    """First-class KG node for queryable decision traces.

    CONCEPT:KG-2.7 — Architecture Decision Records

    Captures the full decision context: what was decided, why, what
    alternatives were considered, who approved, and what concepts/pillars
    are impacted. Makes the 'decision trace layer' from the Context Graph
    Architecture assessment fully queryable via Cypher and SPARQL.

    Lifecycle: proposed → accepted → deprecated → superseded
    """

    type: RegistryNodeType = RegistryNodeType.ARCHITECTURE_DECISION
    title: str = ""
    status: Literal["proposed", "accepted", "deprecated", "superseded"] = "proposed"
    context: str = ""  # Why this decision was needed
    decision: str = ""  # What was decided
    rationale: str = ""  # Why this option was chosen
    alternatives: list[str] = Field(default_factory=list)  # Options considered
    consequences: list[str] = Field(default_factory=list)  # Known tradeoffs
    authority: str = ""  # Who/what approved (user, policy, daemon)
    pillar: str = ""  # ORCH | KG | AHE | ECO | OS
    impacted_concepts: list[str] = Field(default_factory=list)  # Concept IDs
    superseded_by: str = ""  # ID of ADR that supersedes this one


class TriggerBinding(BaseModel):
    """CONCEPT:ECO-4.0 — Declarative trigger binding for callable resources.

    Maps a function/tool to an activation trigger (HTTP route, cron
    schedule, or event topic). Enables AgentOS-style category collapse
    where every capability is a self-describing function with trigger
    metadata.
    """

    trigger_type: Literal["http", "cron", "event", "manual"] = "manual"
    binding: str = Field(
        default="",
        description="Route path, cron expression, or event topic name",
    )
    conditions: dict[str, Any] = Field(
        default_factory=dict,
        description="Activation conditions (e.g., {'method': 'POST', 'auth': true})",
    )


class CallableResourceNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.CALLABLE_RESOURCE
    resource_type: str  # MCP_TOOL, A2A_AGENT, INTERNAL_SKILL, AGENT_SKILL
    endpoint: str | None = None
    agent_card: dict[str, Any] | None = None
    skill_code_path: str | None = None
    metadata_id: str
    # ECO-4.3 Community Telemetry
    origin: Literal["local", "community", "upstream"] = "local"
    timestamp: str | None = None
    author: str | None = None
    # CONCEPT:ECO-4.0 — Self-Describing Function Registry
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing the function's input parameters",
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing the function's return type",
    )
    trigger_bindings: list[TriggerBinding] = Field(
        default_factory=list,
        description="Declarative trigger bindings (http, cron, event)",
    )


class SpawnedAgentNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.SPAWNED_AGENT
    system_prompt: str
    tool_ids: list[str] = Field(default_factory=list)
    parent_task_id: str | None = None
    created_at: str


class SystemPromptNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.SYSTEM_PROMPT
    content: str
    version: str
    tags: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    source: str  # MANUAL, GENERATED, etc.


class AgentTemplateNode(RegistryNode):
    """KG node that declaratively defines a pydantic-graph step.

    CONCEPT:ORCH-1.20 — KG-Driven Graph Materialization

    Each AgentTemplateNode maps 1:1 to a dynamically instantiated
    BaseNode subclass in the materialized pydantic-graph. The KG
    edges (USES_PROMPT, REQUIRES_TOOLSET, DEPENDS_ON,
    COMPATIBLE_WITH_MODEL) fully specify the step's configuration.

    Attributes:
        role: Specialist role tag (e.g., 'researcher', 'coder').
        system_prompt_id: FK → Prompt node for system prompt resolution.
        toolset_ids: FK → Tool/CallableResource nodes for MCP binding.
        model_preference: Preferred model ID (routing policy gets final say).
        execution_tier: LLM tier hint: lite | standard | super.
        step_order: Ordering hint for sequential graphs.
        is_parallel: Whether this step can run in parallel with peers.
        max_retries: Maximum retry attempts on failure.
    """

    type: RegistryNodeType = RegistryNodeType.AGENT_TEMPLATE
    role: str = ""
    system_prompt_id: str = ""
    toolset_ids: list[str] = Field(default_factory=list)
    model_preference: str = ""
    execution_tier: str = "standard"
    step_order: int = 0
    is_parallel: bool = False
    max_retries: int = 2


# --- Self-Improvement & Learning Nodes ---


class OutcomeEvaluationNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.OUTCOME_EVALUATION
    reward: float
    success_criteria_met: list[str] = Field(default_factory=list)
    feedback_text: str


class CritiqueNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.CRITIQUE
    textual_gradient: str


class SelfEvaluationNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.SELF_EVALUATION
    confidence_calibration: float
    task_difficulty: float
    evaluation: str = ""


class ExperimentNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.EXPERIMENT
    status: str


class ProposedSkillNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.PROPOSED_SKILL
    code_content: str
    frontmatter: dict[str, Any] = Field(default_factory=dict)


class ExperienceNode(RegistryNode):
    """Context-specific tactical guidance distilled from past successes/failures."""

    type: RegistryNodeType = RegistryNodeType.EXPERIENCE
    condition: str
    action: str
    success_rate: float = 1.0
    source_run_id: str | None = None


class PatternTemplateNode(RegistryNode):
    """Reusable code pattern or TDD cycle (Hoarding)."""

    type: RegistryNodeType = RegistryNodeType.PATTERN_TEMPLATE
    pattern_type: str  # tdd_cycle, code_snippet, architectural_pattern
    content: str
    success_rate: float = 1.0
    tags: list[str] = Field(default_factory=list)


# --- Knowledge Base Nodes ---


class KnowledgeBaseNode(RegistryNode):
    """Top-level namespace node for a named knowledge base."""

    type: RegistryNodeType = RegistryNodeType.KNOWLEDGE_BASE
    topic: str
    source_type: str  # skill_graph, directory, url, mixed
    source_count: int = 0
    article_count: int = 0
    status: str = "ingesting"  # ingesting, ready, updating, error, archived


class ArticleNode(RegistryNode):
    """A compiled wiki article in a knowledge base."""

    type: RegistryNodeType = RegistryNodeType.ARTICLE
    summary: str
    content: str = ""  # May be empty when archived (summary-only)
    word_count: int = 0
    tags: list[str] = Field(default_factory=list)


class RawSourceNode(RegistryNode):
    """An original document ingested as a source for a knowledge base."""

    type: RegistryNodeType = RegistryNodeType.RAW_SOURCE
    file_path: str
    source_type: str  # md, pdf, docx, epub, txt, html, url
    content_hash: str
    file_size: int = 0
    status: str = "processed"  # pending, processed, error


class KBConceptNode(RegistryNode):
    """A key concept extracted from KB articles."""

    type: RegistryNodeType = RegistryNodeType.KB_CONCEPT


class KBFactNode(RegistryNode):
    """An atomic fact with certainty score extracted from KB articles."""

    type: RegistryNodeType = RegistryNodeType.KB_FACT
    content: str
    certainty: float = 1.0
    source_ids: list[str] = Field(default_factory=list)


class KBIndexNode(RegistryNode):
    """An auto-maintained index document for a knowledge base."""

    type: RegistryNodeType = RegistryNodeType.KB_INDEX
    content: str  # Markdown index with article summaries and suggested queries
    kb_id: str
    article_count: int = 0


class CheckpointNode(RegistryNode):
    """Snapshot of conversation state."""

    type: RegistryNodeType = RegistryNodeType.CHECKPOINT
    label: str
    turn: int
    message_count: int
    message_data: str  # JSON serialized messages


class TeamNode(RegistryNode):
    """Agent team management."""

    type: RegistryNodeType = RegistryNodeType.TEAM
    name: str
    status: str = "active"  # active | dissolved
    member_count: int = 0


class TaskNode(RegistryNode):
    """Shared task within a team."""

    type: RegistryNodeType = RegistryNodeType.TASK
    content: str
    status: str = "pending"  # pending | in_progress | completed
    assigned_to: str | None = None
    created_by: str | None = None


class PolicyNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.POLICY
    policy_id: str
    condition: str
    action: str
    priority: int = 50
    applies_to: list[str] = Field(default_factory=list)
    version: str = "1.0"
    created_at: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None


class ProcessFlowNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.PROCESS_FLOW
    flow_id: str
    goal: str
    start_step: str
    version: str = "1.0"
    created_at: str | None = None


class ProcessStepNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.PROCESS_STEP
    step_id: str
    step_type: str
    tool: str | None = None
    condition: str | None = None


class KnowledgeBaseTopicNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.KNOWLEDGE_BASE_TOPIC
    topic_id: str
    source: str | None = None


# --- KG V2: Human-memory-inspired Nodes (see docs/KG_V2_DESIGN.md §2) ---


class OrganizationNode(RegistryNode):
    """First-class organization (company, team, vendor, etc.).

    ACT-R chunk analogue. Promoted from the generic
    ``EntityNode(entity_type="Organization")`` form because orgs are recurring
    causal hubs: they own systems, employ people, publish policies. See
    docs/KG_V2_DESIGN.md §2.2.1.
    """

    type: RegistryNodeType = RegistryNodeType.ORGANIZATION
    org_id: str = Field(description="Stable slug, e.g. 'acme-corp'")
    legal_name: str | None = None
    domain: str | None = Field(default=None, description="Primary DNS domain")
    org_type: Literal["company", "team", "vendor", "opensource", "regulator"] = (
        "company"
    )
    parent_org_id: str | None = Field(
        default=None, description="Points to another OrganizationNode"
    )
    website: str | None = None


class RoleNode(RegistryNode):
    """A time-bounded role or title a Person plays inside an Organization.

    ACT-R chunk / Tulving (1972) semantic-episodic split: the role itself is
    semantic; the (person, role, phase) binding is episodic and lives on the
    ``PLAYED_ROLE_DURING`` edge. See docs/KG_V2_DESIGN.md §2.2.2.
    """

    type: RegistryNodeType = RegistryNodeType.ROLE
    role_id: str = Field(description="Stable slug, e.g. 'sre-oncall'")
    title: str
    responsibilities: list[str] = Field(default_factory=list)
    organization_id: str | None = Field(
        default=None, description="OrganizationNode.id this role belongs to"
    )
    seniority: (
        Literal[
            "intern",
            "ic",
            "senior",
            "staff",
            "principal",
            "lead",
            "manager",
            "exec",
        ]
        | None
    ) = None


class PlaceNode(RegistryNode):
    """A place — physical, virtual, or contextual.

    Peer, Brunec, Newcombe & Epstein (2021) cognitive-graph analogue.
    Supersedes ``EntityNode(entity_type IN {"Location", "PhysicalLocation",
    "VirtualLocation"})``; the ``kind`` discriminator lets EcphoryRAG
    (Balsam et al. 2025) co-location retrieval treat a Teams channel and a
    conference room uniformly as retrieval cues.
    See docs/KG_V2_DESIGN.md §2.2.3.
    """

    type: RegistryNodeType = RegistryNodeType.PLACE
    place_id: str = Field(description="Stable slug")
    kind: Literal["physical", "virtual", "contextual"]
    address: str | None = Field(
        default=None,
        description=(
            "Street address for physical, URI for virtual, tag for contextual"
        ),
    )
    parent_place_id: str | None = None
    geo_lat: float | None = None
    geo_lon: float | None = None


class PhaseNode(RegistryNode):
    """A named temporal interval — event-segmentation theory (Zacks 2007).

    Anchors events ("what happened during Q2 2026?"). Phases may nest (e.g.
    Phase "Incident-2026-04-02" nests inside Phase "Q2 2026").
    See docs/KG_V2_DESIGN.md §2.2.4.
    """

    type: RegistryNodeType = RegistryNodeType.PHASE
    phase_id: str = Field(description="Stable slug, e.g. 'q2-2026'")
    started_at: str = Field(description="ISO-8601 start timestamp")
    ended_at: str | None = Field(
        default=None,
        description="None while phase is ongoing",
    )
    phase_kind: Literal[
        "calendar",
        "project",
        "incident",
        "lifecycle",
        "custom",
    ] = "custom"
    parent_phase_id: str | None = None


class DecisionNode(RegistryNode):
    """A decision — subtype of Event linking Goal → Action → Outcome.

    Glimcher & Fehr (2013) neuroeconomic / ACT-R goal-state model. Explicit
    motivation + alternatives make counterfactual reasoning possible.
    See docs/KG_V2_DESIGN.md §2.2.5.
    """

    type: RegistryNodeType = RegistryNodeType.DECISION
    decision_id: str
    statement: str = Field(description="The decision in plain language")
    motivation: list[str] = Field(
        default_factory=list,
        description="NodeRefs to Goal/Belief/Fact nodes that motivated the decision",
    )
    alternatives_considered: list[str] = Field(
        default_factory=list,
        description="Plain-text alternatives that were rejected",
    )
    chosen_alternative: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    decided_by: list[str] = Field(
        default_factory=list, description="PersonNode/AgentNode IDs"
    )
    decided_at: str = Field(description="ISO-8601 timestamp")
    reversible: bool = True


class IncidentNode(RegistryNode):
    """A production incident or operational disruption — subtype of Event.

    Brown & Kulik (1977) flashbulb-memory analogue; Josselyn & Tonegawa
    (2020) engram salience weighting. Elevated ``importance_score`` floor
    and slower decay (see maintenance.apply_temporal_decay multipliers).
    See docs/KG_V2_DESIGN.md §2.2.6.
    """

    type: RegistryNodeType = RegistryNodeType.INCIDENT
    incident_id: str
    severity: Literal["low", "medium", "high", "critical"]
    detected_at: str
    resolved_at: str | None = None
    status: Literal["detected", "mitigating", "resolved", "postmortem"] = "detected"
    postmortem_article_id: str | None = Field(
        default=None,
        description="ArticleNode.id with the postmortem",
    )
    affected_system_ids: list[str] = Field(default_factory=list)
    root_cause_summary: str | None = None


class SystemNode(RegistryNode):
    """A software system or service — a causal hub distinct from CodeNode.

    Bartlett (1932) / Rumelhart (1980) schema-theory analogue. CodeNode is
    file/class level; SystemNode is the whole logical system
    ("auth-service", "ingestion-pipeline") with explicit ownership and
    dependency edges. See docs/KG_V2_DESIGN.md §2.2.7.
    """

    type: RegistryNodeType = RegistryNodeType.SYSTEM
    system_id: str = Field(description="Stable slug, e.g. 'auth-service'")
    tech_stack: list[str] = Field(default_factory=list)
    owner_role_ids: list[str] = Field(
        default_factory=list, description="RoleNode.id list"
    )
    owner_org_id: str | None = None
    depends_on_system_ids: list[str] = Field(default_factory=list)
    repo_urls: list[str] = Field(default_factory=list)
    criticality: Literal["tier1", "tier2", "tier3", "experimental"] = "tier2"


class BeliefNode(RegistryNode):
    """A claim-with-confidence grounded in evidence.

    Collins & Quillian (1969) / ACT-R declarative-activation analogue.
    Distinct from FactNode (timeless) because beliefs are *held*, can be
    *revised*, and have ``last_reviewed`` per ACT-R activation theory.
    See docs/KG_V2_DESIGN.md §2.2.8.
    """

    type: RegistryNodeType = RegistryNodeType.BELIEF
    statement: str = Field(description="The proposition being believed")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_node_ids: list[str] = Field(
        default_factory=list,
        description="NodeRefs to Fact/Article/Episode",
    )
    contradicted_by_node_ids: list[str] = Field(default_factory=list)
    supported_by_node_ids: list[str] = Field(default_factory=list)
    last_reviewed: str = Field(description="ISO-8601; bumps on evidence update")
    source_agent_id: str | None = None
    scope_node_ids: list[str] = Field(
        default_factory=list,
        description="Concepts/Systems this belief is scoped to",
    )

    @model_validator(mode="after")
    def _validate_support_contradict_mutex(self) -> BeliefNode:
        """Invariant: an id cannot both support and contradict a belief.

        Enforced per docs/KG_V2_DESIGN.md §2.2.8 and §8.1 test plan.
        """
        overlap = set(self.supported_by_node_ids) & set(self.contradicted_by_node_ids)
        if overlap:
            raise ValueError(
                "BeliefNode: the same node(s) cannot both support and "
                f"contradict a belief: {sorted(overlap)}"
            )
        return self


class HypothesisNode(RegistryNode):
    """A predictive belief — a falsifiable expectation about the future.

    Clark (2013) / Friston (2010) predictive-processing analogue.
    ``observation_outcome_ids`` populates as reality arrives; closure into a
    BeliefNode happens in the maintenance loop (Rule 5, §4.3).
    See docs/KG_V2_DESIGN.md §2.2.9.
    """

    type: RegistryNodeType = RegistryNodeType.HYPOTHESIS
    prediction: str = Field(description="The predicted outcome in plain language")
    preconditions_node_ids: list[str] = Field(
        default_factory=list,
        description="Belief/Fact/Phase IDs that must hold for the prediction",
    )
    observation_outcome_ids: list[str] = Field(
        default_factory=list,
        description="Episode/Incident/Fact IDs that confirmed/refuted",
    )
    falsifiable: bool = True
    verdict: Literal["open", "confirmed", "refuted", "inconclusive"] = "open"
    confidence_prior: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_posterior: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Bayesian update after observations",
    )
    expires_at: str | None = None


class PrincipleNode(RegistryNode):
    """A distilled, reusable rule — 'always use TDD', 'never push on Friday'.

    ACT-R production-rule analogue. An IF-THEN rule compiled from repeated
    decisions or reflections; links back to the evidence (decisions,
    episodes) so the rule can be revisited when conditions change.
    See docs/KG_V2_DESIGN.md §2.2.10.
    """

    type: RegistryNodeType = RegistryNodeType.PRINCIPLE
    principle_id: str = Field(description="Stable slug")
    statement: str = Field(description="The rule, imperative form")
    scope_node_ids: list[str] = Field(
        default_factory=list,
        description="Concept/System/Organization this principle applies to",
    )
    exceptions: list[str] = Field(
        default_factory=list, description="Plain-text exceptions"
    )
    derived_from_decision_ids: list[str] = Field(default_factory=list)
    derived_from_episode_ids: list[str] = Field(default_factory=list)
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How enforced (0=guideline, 1=mandatory)",
    )
    review_cadence_days: int = Field(default=180)
    last_reviewed: str | None = None


class EngineeringRuleNode(PrincipleNode):
    """A software engineering rule derived from a canonical book.

    Subclass of PrincipleNode specialized for agent-rules-books integration.
    Stores structured rule metadata for context-budget-aware retrieval and
    AHE efficacy tracking.

    CONCEPT:KG-2.2 — Engineering Rules Engine
    """

    type: RegistryNodeType = RegistryNodeType.ENGINEERING_RULE
    # Rule tier for context-budget selection
    tier: Literal["full", "mini", "nano"] = Field(
        default="mini",
        description="Context-budget tier: full (reference), mini (recommended), nano (fallback)",
    )
    # Classification per agent-rules-books PROCESS.md taxonomy
    rule_class: Literal[
        "book-thesis",
        "decision-changing",
        "micro-decision",
        "conflict-resolver",
        "trigger",
        "checklist-only",
        "framing",
        "default",
    ] = Field(
        default="decision-changing",
        description="Rule classification from PROCESS.md taxonomy",
    )
    # The LLM bias this rule corrects
    bias_corrected: str = Field(
        default="",
        description="The known LLM shortcut or bias this rule prevents",
    )
    # Trigger condition (for trigger rules)
    trigger_condition: str = Field(
        default="",
        description="When this rule activates (e.g., 'adding external dependency')",
    )
    # Task type tags for relevance matching
    task_relevance_tags: list[str] = Field(
        default_factory=list,
        description="Task types: architecture, refactoring, legacy, production, data-systems, domain-modeling, code-quality",
    )
    # Source book reference
    source_book_id: str = Field(
        default="",
        description="ID of the RuleBookNode this rule was derived from",
    )
    # Section within the source (for traceability)
    source_section: str = Field(
        default="",
        description="Section name in the source book (e.g., 'Decision rules', 'Trigger rules')",
    )
    # AHE tracking
    application_count: int = Field(
        default=0,
        description="Number of times this rule has been applied to a task",
    )
    violation_count: int = Field(
        default=0,
        description="Number of times this rule was violated in a task",
    )
    efficacy_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Running efficacy score from AHE feedback (0=ineffective, 1=highly effective)",
    )
    # Conflict tracking
    conflict_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Deterministic weight for conflict resolution (derived from evidence/reasoning/material)",
    )
    version: str = Field(
        default="1.0.0",
        description="Semantic version for rule evolution tracking",
    )


class RuleBookNode(RegistryNode):
    """A canonical software engineering book that is the source of engineering rules.

    Represents the upstream source (e.g., 'Clean Architecture', 'Domain-Driven Design')
    from which EngineeringRuleNode instances are derived.

    CONCEPT:KG-2.2 — Engineering Rules Engine
    """

    type: RegistryNodeType = RegistryNodeType.RULE_BOOK
    book_id: str = Field(description="Kebab-case slug, e.g. 'clean-architecture'")
    author: str = Field(default="", description="Book author(s)")
    # Domain tags for routing
    domain_tags: list[str] = Field(
        default_factory=list,
        description="Applicable domains: architecture, code-quality, domain-modeling, refactoring, legacy, production, data-systems",
    )
    # Bias the book corrects
    primary_bias: str = Field(
        default="",
        description="The book's central corrective bias (from mini.md 'Primary bias to correct')",
    )
    # When to use
    when_to_use: str = Field(
        default="",
        description="Usage context (from mini.md 'When to use')",
    )
    # Rule counts per tier
    full_rule_count: int = Field(default=0)
    mini_rule_count: int = Field(default=0)
    nano_rule_count: int = Field(default=0)
    # Version tracking
    version: str = Field(
        default="1.0.0",
        description="Version of the rule extraction",
    )


class ObservationNode(RegistryNode):
    """An agent observation — a structured record of something perceived.

    First-class node for tracking what the agent observes during operation,
    enabling the OWL layer to reason about observation patterns and
    correlations.
    """

    type: RegistryNodeType = RegistryNodeType.OBSERVATION
    content: str
    confidence: float = Field(default=1.0, ge=0, le=1)
    source: str
    related_event_id: str | None = None


class ActionNode(RegistryNode):
    """An agent action — a structured record of something the agent did.

    First-class node for tracking agent actions, enabling the OWL layer
    to reason about action patterns, success rates, and causal chains.
    """

    type: RegistryNodeType = RegistryNodeType.ACTION
    action_type: str
    status: str = "completed"  # pending, completed, failed
    triggered_by_event_id: str | None = None
    result: str | None = None


# --- Standard Ontology Nodes (BFO, Schema.org, PROV-O, DC, FIBO) ---


class DocumentNode(RegistryNode):
    """A digital or physical document — Dublin Core aligned.

    BFO:GenericallyDependentContinuant, mapped to schema:DigitalDocument
    and bibo:Document. Properties align with Dublin Core Terms.
    """

    type: RegistryNodeType = RegistryNodeType.DOCUMENT
    title: str = Field(description="dc:title — document title")
    creator: str | None = Field(
        default=None, description="dc:creator — author or creator"
    )
    date: str | None = Field(
        default=None, description="dc:date — creation or publication date"
    )
    subject: str | None = Field(
        default=None, description="dc:subject — topic or subject"
    )
    identifier: str | None = Field(
        default=None, description="dc:identifier — DOI, ISBN, URN, etc."
    )
    format: str | None = Field(default=None, description="dc:format — MIME type")
    language: str | None = Field(
        default=None, description="dc:language — language code"
    )
    content: str = ""
    word_count: int = 0
    tags: list[str] = Field(default_factory=list)


class CreativeWorkNode(RegistryNode):
    """A book, manual, SOP, or creative output — Schema.org aligned.

    BFO:GenericallyDependentContinuant, mapped to schema:CreativeWork.
    """

    type: RegistryNodeType = RegistryNodeType.CREATIVE_WORK
    title: str
    creator: str | None = None
    date_published: str | None = None
    genre: str | None = None
    content: str = ""
    tags: list[str] = Field(default_factory=list)


class DatasetNode(RegistryNode):
    """A data collection — Schema.org aligned.

    BFO:GenericallyDependentContinuant, mapped to schema:Dataset.
    """

    type: RegistryNodeType = RegistryNodeType.DATASET
    distribution_url: str | None = None
    temporal_coverage: str | None = Field(
        default=None, description="Time period the dataset covers"
    )
    spatial_coverage: str | None = None
    format: str | None = None
    record_count: int | None = None
    license: str | None = None


class SoftwareProjectNode(RegistryNode):
    """A software project or repository.

    BFO:IndependentContinuant, aligned to schema:SoftwareSourceCode.
    """

    type: RegistryNodeType = RegistryNodeType.SOFTWARE_PROJECT
    repo_url: str | None = None
    language: str | None = None
    license: str | None = None
    version: str | None = None
    stars: int | None = None
    tech_stack: list[str] = Field(default_factory=list)


class MedicalEntityNode(RegistryNode):
    """A medical concept, condition, or entity — stub for medical domain.

    BFO:IndependentContinuant, aligned to schema:MedicalEntity.
    Future extension point for SNOMED-CT, FHIR, etc.
    """

    type: RegistryNodeType = RegistryNodeType.MEDICAL_ENTITY
    entity_type: str = "generic"  # condition, procedure, drug, anatomy
    icd_code: str | None = None
    snomed_id: str | None = None


class ProcedureNode(RegistryNode):
    """An operational procedure or SOP — blue-collar/ops domain.

    BFO:Process (occurrent), aligned to schema:HowTo.
    """

    type: RegistryNodeType = RegistryNodeType.PROCEDURE
    steps: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    estimated_duration: str | None = None
    category: str = "general"


class RegulationNode(RegistryNode):
    """A compliance or regulatory rule.

    BFO:GenericallyDependentContinuant. Supports compliance tracking
    across healthcare, finance, and technology domains.
    """

    type: RegistryNodeType = RegistryNodeType.REGULATION
    jurisdiction: str | None = None
    effective_date: str | None = None
    expiry_date: str | None = None
    authority: str | None = None
    regulation_type: str = "general"  # general, financial, healthcare, data_privacy
    compliance_status: str = (
        "unknown"  # compliant, non_compliant, under_review, unknown
    )


class FinancialInstrumentNode(RegistryNode):
    """A financial instrument — FIBO aligned.

    BFO:GenericallyDependentContinuant. Aligned to FIBO
    FinancialInstrument for finance domain support.
    """

    type: RegistryNodeType = RegistryNodeType.FINANCIAL_INSTRUMENT
    instrument_type: str = "generic"  # stock, bond, derivative, fund, crypto
    ticker: str | None = None
    issuer: str | None = None
    currency: str | None = None
    isin: str | None = Field(
        default=None, description="International Securities Identification Number"
    )


class FinancialTransactionNode(RegistryNode):
    """A financial transaction event — FIBO aligned.

    BFO:Process (occurrent). Represents a buy, sell, transfer, or settlement.
    """

    type: RegistryNodeType = RegistryNodeType.FINANCIAL_TRANSACTION
    transaction_type: str = "generic"  # buy, sell, transfer, settlement, dividend
    amount: float | None = None
    currency: str | None = None
    counterparty: str | None = None
    executed_at: str | None = None
    status: str = "completed"  # pending, completed, failed, reversed


class AccountNode(RegistryNode):
    """A financial account — FIBO aligned.

    BFO:GenericallyDependentContinuant.
    """

    type: RegistryNodeType = RegistryNodeType.ACCOUNT
    account_type: str = "generic"  # checking, savings, brokerage, custodial
    institution: str | None = None
    currency: str | None = None
    status: str = "active"  # active, closed, frozen


# --- Emergent Architecture Nodes (CONCEPT:ORCH-1.0) ---


class MemoryRetrieverNode(RegistryNode):
    """Versioned metacognitive self-model of the agent's capabilities.

    CONCEPT:KG-2.1 — Persistent Self-Model

    Each session creates a new version linked by SUPERSEDES edges, with a
    CURRENT_SELF_MODEL pointer for O(1) lookup. Integrates with OWL via
    ``memory_retriever.promote_to_owl()`` for reasoner-driven metacognition.

    See docs/pillars/architecture_c4.md §CONCEPT:KG-2.1
    """

    type: RegistryNodeType = RegistryNodeType.SELF_MODEL
    version: int = Field(default=1, description="Monotonically increasing version")
    domain_success_rates: dict[str, float] = Field(
        default_factory=dict,
        description="Running average success rate per routed domain",
    )
    capability_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Self-evaluated confidence per capability area",
    )
    tool_proficiency: dict[str, float] = Field(
        default_factory=dict,
        description="Frequency × success rate per tool",
    )
    total_sessions: int = 0
    total_tasks_completed: int = 0
    known_failure_patterns: list[str] = Field(
        default_factory=list,
        description="Recurring failure modes extracted from low-reward episodes",
    )
    pheromone_trails: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "ACO-inspired specialist→task-pattern affinity trails. "
            "Outer key = specialist_id, inner key = task_pattern, "
            "value = trail strength (0.0–1.0). Trails decay by 10%% "
            "per session (evaporation) and strengthen on success."
        ),
    )
    model_synergies: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "CONCEPT:AHE-3.3 — Model Synergy Tracker. Tracks success rates "
            "for model combinations used in sessions. Keys are sorted, "
            "pipe-delimited model IDs (e.g., 'gpt-4o|claude-sonnet'). "
            "Values are EMA success rates [0.0, 1.0]. Enables intelligent "
            "model recombination when preferred models are unavailable. "
            "Inspired by the RL Conductor's adaptive worker pool selection "
            "(Nielsen et al., ICLR 2026)."
        ),
    )
    session_id: str = Field(default="", description="Session that created this version")


# Backward-compatible alias — the class was renamed from SelfModelNode to
# MemoryRetrieverNode during the CONCEPT:KG-2.0 migration. Tests and
# external integrations may still reference the old name.
SelfModelNode = MemoryRetrieverNode


class SwarmCoalitionNode(RegistryNode):
    """A dynamically formed agent coalition for task execution.

    CONCEPT:ORCH-1.0 — Swarm Orchestration

    Tracks the lifecycle of a dynamically spawned swarm: which agents
    participated, the task tree they were assigned, and the achieved
    parallelism ratio. Registered as transient KG nodes for observability.

    See docs/pillars/architecture_c4.md §CONCEPT:KG-2.0
    """

    type: RegistryNodeType = RegistryNodeType.SWARM_COALITION
    agents_spawned: int = 0
    depth_reached: int = 0
    parallelism_achieved: float = Field(
        default=0.0,
        description="Ratio of parallel vs sequential execution",
    )
    task_description: str = ""
    status: str = "active"  # active, completed, failed


class ProposalNode(RegistryNode):
    """A specialist output proposal competing for broadcast.

    CONCEPT:ORCH-1.2 — Global Workspace Attention

    Specialists submit proposals that are scored by relevance, confidence,
    and track record. Winners are broadcast to the KG for integration.

    See docs/pillars/architecture_c4.md §CONCEPT:ORCH-1.2
    """

    type: RegistryNodeType = RegistryNodeType.PROPOSAL
    specialist_id: str = Field(description="ID of the specialist that generated this")
    output: str = ""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    track_record_score: float = Field(default=0.0, ge=0.0, le=1.0)
    composite_score: float = Field(default=0.0, ge=0.0, le=1.0)
    selected: bool = False


# --- Agentic Design Patterns Gap Nodes (CONCEPT:ORCH-1.1 through CONCEPT:AHE-3.2) ---


class PromptChainNode(RegistryNode):
    """A declarative multi-step prompt pipeline.

    CONCEPT:ORCH-1.1 — Prompt Chaining Pattern

    Models a sequence of prompt steps with intermediate validation,
    conditional branching, and result transformation. Persisted to the KG
    for discovery and reuse across sessions.

    BFO:Process (occurrent), aligned to :Procedure.
    See docs/pillars/architecture_c4.md §CONCEPT:ORCH-1.1
    """

    type: RegistryNodeType = RegistryNodeType.PROMPT_CHAIN
    chain_id: str = Field(description="Stable slug, e.g. 'extract-and-summarize'")
    step_count: int = 0
    steps: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of chain step definitions",
    )
    max_retries_per_step: int = 2
    total_executions: int = 0
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0


class ResourceUsageNode(RegistryNode):
    """Per-session resource consumption record.

    CONCEPT:OS-5.2 — Resource-Aware Optimization

    Tracks token usage, cost, and latency per specialist per session.
    Historical data feeds into the MemoryRetriever for trend analysis and
    the OWL reasoner for model selection optimization.

    BFO:Process (occurrent), aligned to :ResourceUsage.
    See docs/pillars/architecture_c4.md §CONCEPT:OS-5.2
    """

    type: RegistryNodeType = RegistryNodeType.RESOURCE_USAGE
    session_id: str = Field(description="Session that generated this record")
    specialist_id: str | None = Field(
        default=None, description="Specialist that consumed resources"
    )
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model_id: str | None = None
    model_tier: str | None = None
    budget_allocated: float = 0.0
    budget_remaining: float = 0.0


class EvaluationRecordNode(RegistryNode):
    """Multi-dimensional evaluation of an agent response.

    CONCEPT:AHE-3.1 — Evaluation & Monitoring

    Provides per-dimension scoring (correctness, completeness, relevance,
    safety) with a composite score for backward compatibility with the
    existing verifier gate. Supports LLM-as-Judge and human calibration.

    BFO:SpecificallyDependentContinuant, aligned to :Observation.
    See docs/pillars/architecture_c4.md §CONCEPT:AHE-3.1
    """

    type: RegistryNodeType = RegistryNodeType.EVALUATION_RECORD
    correctness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    safety_score: float = Field(default=1.0, ge=0.0, le=1.0)
    composite_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evaluator: str = "llm-judge"  # llm-judge, human, automated
    rubric_id: str | None = None
    evidence: str = ""
    session_id: str = ""

    # KG Eval Capture fields (AHE-3.1)
    query: str | None = None
    method: str | None = None
    result_node_ids: list[str] | None = None
    latency_ms: float | None = None
    schema_pack: str | None = None


class PrioritizedTaskNode(RegistryNode):
    """A task with multi-factor priority scoring.

    CONCEPT:ORCH-1.1 — Task Prioritization

    Extends the basic SDD task model with urgency, impact, effort, and risk
    dimensions. Supports dynamic re-prioritization, priority inheritance
    from blocking tasks, and capability-based specialist assignment.

    BFO:Process (occurrent), aligned to :Action.
    See docs/pillars/architecture_c4.md §CONCEPT:ORCH-1.1
    """

    type: RegistryNodeType = RegistryNodeType.PRIORITIZED_TASK
    task_id: str = Field(description="Stable task identifier")
    urgency: float = Field(default=0.5, ge=0.0, le=1.0)
    impact: float = Field(default=0.5, ge=0.0, le=1.0)
    effort: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Estimated complexity (0=trivial, 1=massive)",
    )
    risk: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Failure probability",
    )
    composite_priority: float = Field(default=0.0, ge=0.0, le=1.0)
    status: str = "pending"  # pending, in_progress, completed, blocked
    assigned_specialist: str | None = None
    blocking_task_ids: list[str] = Field(default_factory=list)
    blocked_by_task_ids: list[str] = Field(default_factory=list)


class KnowledgeGapNode(RegistryNode):
    """An identified gap in the agent's knowledge.

    CONCEPT:AHE-3.2 — Exploration & Discovery

    Represents a domain or topic where the agent has identified insufficient
    knowledge. Links to hypotheses generated to fill the gap and experiments
    designed to test those hypotheses.

    BFO:SpecificallyDependentContinuant, aligned to :Observation.
    See docs/pillars/architecture_c4.md §CONCEPT:AHE-3.2
    """

    type: RegistryNodeType = RegistryNodeType.KNOWLEDGE_GAP
    domain: str = Field(description="Domain area of the gap")
    gap_statement: str = Field(description="Description of what is missing")
    severity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How critical this gap is (0=nice-to-know, 1=blocking)",
    )
    status: str = "identified"  # identified, exploring, filled, deferred
    hypothesis_ids: list[str] = Field(default_factory=list)
    discovered_fact_ids: list[str] = Field(default_factory=list)


class ExplorationExperimentNode(RegistryNode):
    """A structured experiment to explore a hypothesis.

    CONCEPT:AHE-3.2 — Exploration & Discovery

    Represents an experiment designed to test a hypothesis, including
    design, variables, success criteria, and results. Supports
    multi-reviewer evaluation with structured scoring.

    BFO:Process (occurrent), aligned to :Procedure.
    See docs/pillars/architecture_c4.md §CONCEPT:AHE-3.2
    """

    type: RegistryNodeType = RegistryNodeType.EXPLORATION_EXPERIMENT
    experiment_id: str = Field(description="Stable experiment identifier")
    hypothesis_id: str = Field(description="HypothesisNode ID being tested")
    design: str = Field(description="Natural language experiment description")
    variables: dict[str, str] = Field(
        default_factory=dict,
        description="Variable name -> description mapping",
    )
    success_criteria: str = ""
    results: str | None = None
    review_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Reviewer name -> score mapping",
    )
    status: str = "designed"  # designed, running, completed, failed


# --- First-Principles Architecture Nodes (CONCEPT:AHE-3.3) ---


class TeamConfigNode(RegistryNode):
    """Reusable team composition template promoted from successful coalitions.

    CONCEPT:AHE-3.3 — Proven Team Reuse

    When a ``SwarmCoalition`` completes successfully (reward > threshold),
    it is promoted into a ``TeamConfigNode``.  The router queries matching
    TeamConfigs before LLM-based planning — if a proven template exists
    with a high enough ``success_rate``, it is reused directly.

    The ``capability_overrides`` field enables RLM + TeamConfig synergy:
    adaptive_agent_router that historically receive large data can have the ``rlm``
    capability auto-attached at reuse time.

    See docs/pillars/architecture_c4.md §CONCEPT:AHE-3.3
    """

    type: RegistryNodeType = RegistryNodeType.TEAM_CONFIG
    task_pattern: str = Field(
        description="Semantic descriptor of the task type this team solves"
    )
    specialist_ids: list[str] = Field(
        default_factory=list,
        description="Agent IDs that form this team",
    )
    tool_assignments: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Mapping of agent_id → assigned tool names",
    )
    prompt_template_ids: list[str] = Field(
        default_factory=list,
        description="Prompt node IDs used by the team",
    )
    capability_overrides: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Mapping of agent_id → capability types to auto-attach",
    )
    success_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Rolling average success rate from OutcomeEvaluations",
    )
    usage_count: int = Field(default=0, description="Number of times reused")
    reuse_threshold: float = Field(
        default=0.72,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity to consider reuse (adaptive)",
    )
    # ECO-4.3 Community Telemetry
    origin: Literal["local", "community", "upstream"] = "local"
    timestamp: str | None = None
    author: str | None = None


class AgentCapabilityNode(RegistryNode):
    """First-class capability assignable to specialist agents.

    CONCEPT:ORCH-1.2 — Agent Capability Type System

    Formalizes capabilities (RLM, Critic, Navigator, etc.) as typed KG
    nodes with handler metadata and trigger conditions.  Linked to agents
    via ``HAS_CAPABILITY`` edges.  When ``auto_activate`` is True and
    trigger conditions are met at runtime, the executor dynamically
    invokes the capability handler.

    See docs/pillars/architecture_c4.md §CONCEPT:ORCH-1.2
    """

    type: RegistryNodeType = RegistryNodeType.AGENT_CAPABILITY
    capability_type: str = Field(
        description=(
            "Capability identifier: 'rlm', 'critic', 'navigator', "
            "'synthesizer', 'researcher', 'data_synthesizer', 'memory_manager'"
        )
    )
    handler_module: str = Field(
        description="Python import path, e.g. 'agent_utilities.rlm.specialist'"
    )
    handler_function: str = Field(
        default="run",
        description="Entry point function name within handler_module",
    )
    trigger_conditions: dict[str, Any] = Field(
        default_factory=dict,
        description="Conditions for auto-activation, e.g. {'input_chars_gt': 50000}",
    )
    performance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Rolling performance score from outcome evaluations",
    )
    auto_activate: bool = Field(
        default=True,
        description="Whether to auto-engage when trigger conditions are met",
    )


# --- KG-Native Orchestration Nodes (CONCEPT:ORCH-1.1 through CONCEPT:ORCH-1.4) ---


class TopologyTemplateNode(RegistryNode):
    """KG-stored graph execution topology template.

    CONCEPT:ORCH-1.2 — Dynamic Topology Materialization

    Defines a reusable execution pattern: which specialist types participate,
    how they are connected (transitions), and what execution mode they use.
    At runtime, the TopologyEngine selects the best template based on domain
    and complexity, then materializes it into a live ``pydantic-graph``.

    Templates are authored as YAML and ingested into the KG.  Successful
    executions increase ``success_rate``; poor ones decrease it, enabling
    evolutionary selection over time.
    """

    type: RegistryNodeType = RegistryNodeType.TOPOLOGY_TEMPLATE
    domain: str = Field(
        default="general",
        description="Domain this topology applies to (general, finance, medical, etc.)",
    )
    complexity_min: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Minimum task complexity this template handles",
    )
    complexity_max: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Maximum task complexity this template handles",
    )
    node_roles: list[str] = Field(
        default_factory=list,
        description="Ordered list of specialist roles in this topology",
    )
    transitions: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Mapping of role -> list of possible next roles",
    )
    execution_mode: str = Field(
        default="sequential",
        description="Execution mode: sequential, parallel, fan_out, fan_in, mixed",
    )
    parallel_groups: list[list[str]] = Field(
        default_factory=list,
        description="Groups of roles that execute in parallel (for mixed mode)",
    )
    tool_assignments: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Mapping of role -> required tool names",
    )
    model_preferences: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of role -> preferred model_id",
    )
    system_prompt_ids: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of role -> PromptNode ID for system prompt",
    )
    memory_channels: list[str] = Field(
        default_factory=list,
        description="KG memory channels this topology reads/writes",
    )
    success_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Rolling success rate from outcome evaluations",
    )
    usage_count: int = Field(default=0, description="Number of times materialized")
    origin: Literal["local", "community", "upstream"] = "local"


class SessionCheckpointNode(RegistryNode):
    """Persisted execution state checkpoint in the Knowledge Graph.

    CONCEPT:ORCH-1.1 — Execution State Persistence

    Bridges the ephemeral ``GraphState`` and the persistent KG.
    Created at HSM transition boundaries and on session completion.
    Enables session resume, cross-session learning, and active
    state queries from other agents.
    """

    type: RegistryNodeType = RegistryNodeType.SESSION_CHECKPOINT
    session_id: str = Field(description="Unique session identifier")
    query: str = Field(default="", description="Original user query")
    plan: str = Field(default="", description="Serialized execution plan")
    specialist_results: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of specialist_id -> result summary",
    )
    node_history: list[str] = Field(
        default_factory=list,
        description="Ordered list of graph nodes visited",
    )
    current_node: str = Field(default="", description="Current/last graph node")
    total_usage_tokens: int = Field(default=0, description="Cumulative token usage")
    state_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary state data for full reconstruction",
    )
    status: str = Field(
        default="active",
        description="Checkpoint status: active, completed, failed, suspended",
    )
    topology_template_id: str = Field(
        default="",
        description="ID of the TopologyTemplate that was materialized for this session",
    )


class PersistentAgentNode(RegistryNode):
    """Long-running background agent coordinated via the Knowledge Graph.

    CONCEPT:ORCH-1.4 — Persistent Background Agents

    Unlike ephemeral request agents, persistent agents maintain state
    across sessions.  They register in the KG with subscription filters
    and are awakened by the EventStreamIngester when matching events occur.

    Lifecycle: registered → idle → running → idle → ... → terminated

    Uses the unified CognitiveScheduler (OS-5.2) for scheduling.
    """

    type: RegistryNodeType = RegistryNodeType.PERSISTENT_AGENT
    agent_type: str = Field(
        default="background",
        description="Agent classification: background, monitor, scheduler, rebalancer",
    )
    subscriptions: list[str] = Field(
        default_factory=list,
        description="Event types this agent reacts to (e.g., 'data.new', 'policy.changed')",
    )
    schedule_cron: str = Field(
        default="",
        description="Cron expression for periodic execution (empty = event-driven only)",
    )
    heartbeat_ts: str = Field(
        default="",
        description="ISO-8601 timestamp of last heartbeat",
    )
    state_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized agent context for session continuity",
    )
    status: str = Field(
        default="idle",
        description="Lifecycle status: idle, running, paused, terminated",
    )
    specialist_ids: list[str] = Field(
        default_factory=list,
        description="Agent IDs this persistent agent can spawn",
    )
    tool_ids: list[str] = Field(
        default_factory=list,
        description="Tools available to this persistent agent",
    )
    model_id: str = Field(
        default="",
        description="Preferred model for this agent's inference calls",
    )
    max_concurrent: int = Field(
        default=1,
        description="Maximum concurrent executions allowed",
    )


class TeamComposition(BaseModel):
    """Result of KG-driven team composition.

    CONCEPT:ORCH-1.1 — KG-Driven Team Composition

    Returned by KGTeamComposer.compose_team(). Contains everything
    needed to materialize and execute a specialist team:
    - Which adaptive_agent_router to spawn (with their roles)
    - What tools each specialist gets
    - What model each specialist uses
    - What system prompts to inject
    - How they are connected (topology)
    - Whether they run in parallel or sequentially
    """

    team_id: str = Field(description="Unique team instance identifier")
    source: str = Field(
        default="composed",
        description="How this team was created: 'reused' (from TeamConfigNode) or 'composed' (from topology)",
    )
    team_config_id: str = Field(
        default="",
        description="If reused, the ID of the TeamConfigNode",
    )
    topology_template_id: str = Field(
        default="",
        description="ID of the TopologyTemplate used",
    )
    adaptive_agent_router: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of specialist configurations: [{role, agent_id, model_id, tools, system_prompt, ...}]",
    )
    execution_mode: str = Field(
        default="sequential",
        description="How adaptive_agent_router execute: sequential, parallel, fan_out, fan_in, mixed",
    )
    parallel_groups: list[list[str]] = Field(
        default_factory=list,
        description="For mixed mode: groups of roles that execute in parallel",
    )
    memory_channels: list[str] = Field(
        default_factory=list,
        description="KG memory channels shared across the team",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this composition (from historical success or topology scoring)",
    )
    reasoning: str = Field(
        default="",
        description="Human-readable explanation of composition decisions",
    )
    coordination_protocol: dict[str, Any] = Field(
        default_factory=dict,
        description="CONCEPT:ORCH-1.5 — Selected coordination protocol metadata "
        "(protocol_id, protocol_type, name, quality_score, converged). "
        "Research: 2605.03310v1",
    )


# --- Agent OS Architecture Nodes (CONCEPT:OS-5.2) ---


class AgentProcessNode(RegistryNode):
    """Running agent process tracked by the Cognitive Scheduler.

    CONCEPT:OS-5.2 — Cognitive Scheduler

    Represents a live or paused specialist agent managed by the
    ``CognitiveScheduler``.  Tracks priority, execution state,
    token quota/usage, and optional checkpoint IDs for context
    paging.  Linked to its parent agent via ``EXECUTED_BY`` edges
    and to checkpoints via ``CHECKPOINTED_TO`` edges.

    See docs/pillars/5_agent_os_infrastructure.md §CONCEPT:OS-5.2
    """

    type: RegistryNodeType = RegistryNodeType.AGENT_PROCESS
    priority: int = Field(
        default=2,
        ge=0,
        le=3,
        description="Scheduler priority: 0=CRITICAL, 1=HIGH, 2=NORMAL, 3=LOW",
    )
    state: str = Field(
        default="waiting",
        description="Process state: waiting, running, paused, completed, failed",
    )
    token_quota: int = Field(
        default=100_000,
        description="Maximum token budget for this process",
    )
    tokens_used: int = Field(
        default=0,
        description="Tokens consumed so far",
    )
    checkpoint_id: str | None = Field(
        default=None,
        description="ID of context checkpoint when paused",
    )
    task_description: str = Field(
        default="",
        description="Human-readable task this process is executing",
    )
    preempted_at: float | None = Field(
        default=None,
        description="Timestamp when process was last preempted",
    )


class AgentIdentityNode(RegistryNode):
    """Signed agent identity for permissions governance.

    CONCEPT:OS-5.1 — Permissions Kernel

    Each specialist agent receives a signed identity when spawned,
    binding it to a role (admin, operator, specialist, sandbox, guest)
    and a set of granted capabilities.  The ``signature`` field contains
    an HMAC-SHA256 of the identity payload for tamper detection.

    Linked to agents via ``HAS_IDENTITY`` edges and to authorized
    tools via ``AUTHORIZED_FOR`` edges.

    See docs/pillars/5_agent_os_infrastructure.md §CONCEPT:OS-5.2
    """

    type: RegistryNodeType = RegistryNodeType.AGENT_IDENTITY
    role: str = Field(
        default="specialist",
        description="Permission role: admin, operator, specialist, sandbox, guest",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="Granted capability identifiers",
    )
    signature: str = Field(
        default="",
        description="HMAC-SHA256 signature for identity verification",
    )
    issued_at: float = Field(
        default=0.0,
        description="Unix timestamp when identity was issued",
    )


class SpecialistPackageNode(RegistryNode):
    """Installed specialist package metadata.

    CONCEPT:OS-5.0 — Agent Registry

    Tracks specialist packages installed via the Agent Registry CLI
    (``agent-utilities install <name>``).  Each package maps to an
    MCP server config fragment, a set of tools, and KG specialist
    nodes.  Linked to the originating registry via ``INSTALLED_FROM``
    edges.

    See docs/pillars/5_agent_os_infrastructure.md §CONCEPT:OS-5.2
    """

    type: RegistryNodeType = RegistryNodeType.SPECIALIST_PACKAGE
    version: str = Field(
        default="0.0.0",
        description="Semantic version of the installed package",
    )
    mcp_server_name: str = Field(
        default="",
        description="Name of the MCP server this package provides",
    )
    tool_count: int = Field(
        default=0,
        description="Number of tools exposed by this package",
    )
    installed_at: str = Field(
        default="",
        description="ISO timestamp when the package was installed",
    )
    source_registry: str = Field(
        default="local",
        description="Registry source: local, remote, or systems-manager",
    )


class HostNode(RegistryNode):
    """A remote host in the Agent OS infrastructure.

    First-class KG citizen representing a managed server.
    Credentials are resolved via the ``secret://`` engine —
    passwords and keys are never stored in plaintext.

    Used by:
        - ``tunnel-manager``: SSH connections, remote exec, file transfer
        - ``container-manager-mcp``: Docker/Podman endpoint targeting
        - ``systems-manager``: Health checks and monitoring

    Attributes:
        hostname: IP address or FQDN of the host.
        alias: Friendly name (e.g. ``media-server``).
        port: SSH port (default 22).
        user: SSH username.
        credential_ref: ``secret://`` URI for password.
        identity_file_ref: ``secret://`` URI for SSH key path.
        os_type: Operating system (e.g. ``linux``, ``darwin``).
        arch: CPU architecture (e.g. ``x86_64``, ``aarch64``).
        labels: Arbitrary key-value labels for filtering.
        docker_endpoint: Docker API endpoint (e.g. ``tcp://192.168.1.10:2375``).
        docker_host: Whether this host can run containers.
        swarm_role: Docker Swarm role (``manager``, ``worker``, or empty).
        container_manager_url: URL of a deployed container-manager-mcp instance.
        services: List of running compose service names.
        last_seen: ISO timestamp from the last health check.
        health_status: Current health (``healthy``, ``degraded``, ``unreachable``).
    """

    type: RegistryNodeType = RegistryNodeType.HOST

    # Network identity
    hostname: str = Field(default="", description="IP address or FQDN")
    alias: str = Field(default="", description="Friendly host name")
    port: int = Field(default=22, description="SSH port")

    # Auth (secret:// references — never plaintext)
    user: str = Field(default="", description="SSH username")
    credential_ref: str = Field(
        default="",
        description="secret:// URI for password (e.g. secret://hosts/media-server/password)",
    )
    identity_file_ref: str = Field(
        default="",
        description="secret:// URI for SSH key (e.g. secret://hosts/media-server/identity)",
    )

    # Host capabilities
    os_type: str = Field(default="", description="OS type: linux, darwin, windows")
    arch: str = Field(default="", description="CPU architecture: x86_64, aarch64")
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary labels for filtering (e.g. role=media, rack=2)",
    )

    # Container runtime
    docker_endpoint: str = Field(
        default="",
        description="Docker API URL: tcp://host:2375 or unix:///var/run/docker.sock",
    )
    docker_host: bool = Field(
        default=False, description="Whether this host can run containers"
    )
    swarm_role: str = Field(
        default="", description="Docker Swarm role: manager, worker, or empty"
    )
    container_manager_url: str = Field(
        default="",
        description="URL of deployed container-manager-mcp instance on this host",
    )

    # State
    services: list[str] = Field(
        default_factory=list, description="Running compose service names"
    )
    last_seen: str = Field(default="", description="ISO timestamp of last health check")
    health_status: str = Field(
        default="unknown",
        description="Current health: healthy, degraded, unreachable",
    )


class InfrastructureTemplateNode(RegistryNode):
    """A deployable infrastructure blueprint.

    References existing compose files from agent repos — does NOT
    duplicate them.  Used by ``container-manager-mcp`` to scaffold
    dependencies on-demand (databases, search engines, observability).

    Attributes:
        compose_ref: Path to compose file relative to workspace root.
        services: Service names defined in the compose file.
        required_env: Environment variables that must be set before deploy.
        optional_env: Optional env vars with defaults.
        depends_on_templates: Other templates that must be deployed first.
        profile: Deployment profile (standalone, traefik, swarm).
        tags: Searchable tags.
    """

    type: RegistryNodeType = RegistryNodeType.INFRASTRUCTURE_TEMPLATE

    compose_ref: str = Field(
        default="",
        description="Compose file path relative to workspace (e.g. agents/langfuse-agent/compose.yml)",
    )
    services: list[str] = Field(
        default_factory=list,
        description="Service names in the compose file",
    )
    required_env: list[str] = Field(
        default_factory=list,
        description="Required env vars (e.g. LANGFUSE_URL, LANGFUSE_TOKEN)",
    )
    optional_env: dict[str, str] = Field(
        default_factory=dict,
        description="Optional env vars with defaults (e.g. PORT=9001)",
    )
    depends_on_templates: list[str] = Field(
        default_factory=list,
        description="Templates that must be deployed first (e.g. postgres)",
    )
    profile: str = Field(
        default="standalone",
        description="Deployment profile: standalone, traefik, swarm",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Searchable tags (e.g. observability, os_service)",
    )


class SchemaPackNode(RegistryNode):
    """A persisted Schema Pack configuration in the Knowledge Graph.

    CONCEPT:KG-2.2 — Schema Packs

    Tracks which domain profile is active for this workspace,
    enabling pack-aware filtering in the OWL bridge, hybrid retriever,
    and inference engine.

    See docs/pillars/2_epistemic_knowledge_graph.md §Schema Packs.
    """

    type: RegistryNodeType = RegistryNodeType.SCHEMA_PACK
    pack_name: str = Field(description="Pack identifier (e.g. 'research-state')")
    mode: str = Field(
        default="additive",
        description="Operating mode: 'additive' or 'exclusive'",
    )
    active_node_types: list[str] = Field(
        default_factory=list,
        description="Node type strings active under this pack",
    )
    active_edge_types: list[str] = Field(
        default_factory=list,
        description="Edge type strings active under this pack",
    )
    backlink_boost_strategy: str = Field(
        default="global",
        description="Backlink boost strategy: global, context_only, disabled",
    )
    backlink_boost_factor: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Logarithmic scaling coefficient for backlink boost",
    )


# --- Schema Definition for Backend Abstraction ---


class TableDefinition(BaseModel):
    name: str
    columns: dict[str, str]  # name: type (Ladybug types)


class RelDefinition(BaseModel):
    type: str
    connections: list[dict[str, str]]  # List of {"from": "Type", "to": "Type"}


class GraphSchemaDefinition(BaseModel):
    nodes: list[TableDefinition]
    edges: list[RelDefinition]


# --- Financial Trading Pipeline Nodes (CONCEPT:KG-2.6) ---


class TradingSignalNode(RegistryNode):
    """A buy/sell/hold signal with confidence and attribution.

    CONCEPT:KG-2.6 — Financial Trading Pipeline

    Captures actionable trading signals generated by strategies or
    analysts. Links to the originating ``StrategyNode`` via
    ``GENERATED_SIGNAL`` edges and to the target ``FinancialInstrument``.

    Attributes:
        signal_type: Direction of the signal (buy, sell, hold).
        confidence: Signal strength from 0.0 (no confidence) to 1.0 (certain).
        instrument_id: Identifier for the target financial instrument.
        attribution: Source of the signal (e.g., 'technical', 'fundamental', 'sentiment').
        price_at_signal: Market price when the signal was generated.
        expiry: Optional ISO timestamp after which the signal is considered stale.
    """

    type: RegistryNodeType = RegistryNodeType.TRADING_SIGNAL
    signal_type: str = "hold"  # buy, sell, hold
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    instrument_id: str = ""
    attribution: str = ""  # technical, fundamental, sentiment, composite
    price_at_signal: float | None = None
    expiry: str | None = None


class OrderNode(RegistryNode):
    """An order in the trading pipeline lifecycle.

    CONCEPT:KG-2.6 — Financial Trading Pipeline

    Models the full order lifecycle: pending → submitted → filled → cancelled.
    Links to ``TradingSignalNode`` via ``PLACED_ORDER`` and to
    ``PositionNode`` via ``OPENED_POSITION``.

    Attributes:
        order_type: Order type (market, limit, stop, stop_limit).
        side: Order side (buy, sell).
        quantity: Number of units.
        price: Limit/stop price (None for market orders).
        filled_price: Actual execution price.
        status: Current lifecycle status.
        instrument_id: Target instrument identifier.
        exchange: Exchange or venue identifier.
        submitted_at: ISO timestamp of order submission.
        filled_at: ISO timestamp of order fill.
    """

    type: RegistryNodeType = RegistryNodeType.ORDER
    order_type: str = "market"  # market, limit, stop, stop_limit
    side: str = "buy"  # buy, sell
    quantity: float = 0.0
    price: float | None = None
    filled_price: float | None = None
    status: str = "pending"  # pending, submitted, partial, filled, cancelled, rejected
    instrument_id: str = ""
    exchange: str = ""
    submitted_at: str | None = None
    filled_at: str | None = None


class PositionNode(RegistryNode):
    """An open or closed position in a portfolio.

    CONCEPT:KG-2.6 — Financial Trading Pipeline

    Tracks position lifecycle with entry/exit prices, P&L, and risk
    metrics. Links to ``PortfolioNode`` via ``BELONGS_TO_PORTFOLIO``.

    Attributes:
        instrument_id: Target instrument identifier.
        side: Position direction (long, short).
        quantity: Position size.
        entry_price: Average entry price.
        exit_price: Average exit price (None if still open).
        current_price: Latest known market price.
        realized_pnl: Closed P&L.
        unrealized_pnl: Open P&L.
        status: Position status (open, closed).
        opened_at: ISO timestamp when position was opened.
        closed_at: ISO timestamp when position was fully closed.
    """

    type: RegistryNodeType = RegistryNodeType.POSITION
    instrument_id: str = ""
    side: str = "long"  # long, short
    quantity: float = 0.0
    entry_price: float = 0.0
    exit_price: float | None = None
    current_price: float | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    status: str = "open"  # open, closed
    opened_at: str | None = None
    closed_at: str | None = None


class PortfolioNode(RegistryNode):
    """An aggregation of positions with allocation and performance tracking.

    CONCEPT:KG-2.6 — Financial Trading Pipeline

    Represents a named portfolio containing multiple positions.
    Performance metrics are computed from constituent positions.

    Attributes:
        total_value: Total portfolio market value.
        cash_balance: Uninvested cash.
        position_count: Number of active positions.
        total_pnl: Aggregate realized + unrealized P&L.
        allocation_weights: Asset-to-weight mapping for rebalancing.
        benchmark_id: Optional benchmark identifier for comparison.
    """

    type: RegistryNodeType = RegistryNodeType.PORTFOLIO
    total_value: float = 0.0
    cash_balance: float = 0.0
    position_count: int = 0
    total_pnl: float = 0.0
    allocation_weights: dict[str, float] = Field(default_factory=dict)
    benchmark_id: str | None = None


class StrategyNode(RegistryNode):
    """A named trading strategy with parameters and performance lineage.

    CONCEPT:KG-2.6 — Financial Trading Pipeline

    Represents a reusable strategy definition. Links to ``BacktestRunNode``
    via ``BACKTESTED_WITH`` and to ``PortfolioNode`` via
    ``EXECUTES_STRATEGY`` for live execution tracking.

    Attributes:
        strategy_type: Category (momentum, mean_reversion, factor, ml, etc.).
        parameters: Strategy-specific configuration parameters.
        version: Strategy version for lineage tracking.
        sharpe_ratio: Latest computed Sharpe ratio.
        max_drawdown: Maximum observed drawdown (0.0 to 1.0).
        win_rate: Historical win rate (0.0 to 1.0).
        universes: List of instrument universes this strategy targets.
    """

    type: RegistryNodeType = RegistryNodeType.STRATEGY
    strategy_type: str = ""  # momentum, mean_reversion, factor, ml, hybrid
    parameters: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    universes: list[str] = Field(default_factory=list)


# --- Market Data Connector Nodes (CONCEPT:ECO-4.3) ---


class DataConnectorNode(RegistryNode):
    """A registered external data source with health and rate-limit metadata.

    CONCEPT:ECO-4.3 — Market Data Connector Protocol

    Represents an external data provider that can be queried by agents.
    The ``FALLS_BACK_TO`` edge creates a prioritized fallback chain.

    Attributes:
        connector_type: Category (market_data, news, fundamental, alternative).
        provider: Provider name (e.g., 'yahoo_finance', 'polygon', 'alpha_vantage').
        base_url: API base URL.
        is_healthy: Current health status.
        rate_limit_rpm: Maximum requests per minute.
        rate_limit_remaining: Remaining requests in current window.
        supported_instruments: List of instrument types this connector supports.
        priority: Fallback priority (lower = tried first).
        last_health_check: ISO timestamp of last health ping.
    """

    type: RegistryNodeType = RegistryNodeType.DATA_CONNECTOR
    connector_type: str = "market_data"
    provider: str = ""
    base_url: str = ""
    is_healthy: bool = True
    rate_limit_rpm: int = 60
    rate_limit_remaining: int = 60
    supported_instruments: list[str] = Field(default_factory=list)
    priority: int = 0
    last_health_check: str | None = None


class DataFetchRecordNode(RegistryNode):
    """Immutable provenance record of a data fetch operation.

    CONCEPT:ECO-4.3 — Market Data Connector Protocol

    Records every data retrieval for audit, debugging, and cost tracking.
    Links to the source ``DataConnectorNode`` via ``FETCHED_FROM``.

    Attributes:
        connector_id: The data connector that served this request.
        query: The query or endpoint that was called.
        row_count: Number of data rows returned.
        latency_ms: Request latency in milliseconds.
        status_code: HTTP status code or equivalent.
        error: Error message if the fetch failed.
        fetched_at: ISO timestamp of the fetch.
    """

    type: RegistryNodeType = RegistryNodeType.DATA_FETCH_RECORD
    connector_id: str = ""
    query: str = ""
    row_count: int = 0
    latency_ms: float = 0.0
    status_code: int = 200
    error: str | None = None
    fetched_at: str | None = None


# --- Swarm Preset Template Nodes (CONCEPT:ORCH-1.4) ---


class SwarmPresetNode(RegistryNode):
    """A declarative multi-agent workflow preset stored in the KG.

    CONCEPT:ORCH-1.4 — Swarm Preset Template Engine

    Represents a reusable DAG-based swarm configuration with agent
    specifications, task dependencies, and template variables.
    Links to ``TeamConfigNode`` via ``PRESET_OF`` for evolutionary
    recommendation.

    Attributes:
        preset_name: Unique preset identifier (e.g., 'investment_committee').
        agent_specs: List of agent role specifications (id, role, tools).
        task_graph: DAG of task definitions with dependency edges.
        variables: Template variables available for substitution.
        success_count: Number of successful executions.
        total_runs: Total number of executions.
        avg_duration_seconds: Average run duration.
    """

    type: RegistryNodeType = RegistryNodeType.SWARM_PRESET
    preset_name: str = ""
    agent_specs: list[dict[str, Any]] = Field(default_factory=list)
    task_graph: list[dict[str, Any]] = Field(default_factory=list)
    variables: dict[str, str] = Field(default_factory=dict)
    success_count: int = 0
    total_runs: int = 0
    avg_duration_seconds: float = 0.0


class SwarmRunNode(RegistryNode):
    """Execution record of a swarm preset run.

    CONCEPT:ORCH-1.4 — Swarm Preset Template Engine

    Links to the originating ``SwarmPresetNode`` via ``RAN_PRESET``.

    Attributes:
        preset_id: Reference to the preset that was executed.
        status: Run status (pending, running, completed, failed, cancelled).
        user_vars: User-provided variable substitutions.
        total_input_tokens: Cumulative input tokens.
        total_output_tokens: Cumulative output tokens.
        duration_seconds: Total run duration.
        task_count: Number of tasks in the run.
        started_at: ISO start timestamp.
        completed_at: ISO completion timestamp.
        final_report: Aggregated output summary.
    """

    type: RegistryNodeType = RegistryNodeType.SWARM_RUN
    preset_id: str = ""
    status: str = "pending"  # pending, running, completed, failed, cancelled
    user_vars: dict[str, str] = Field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    duration_seconds: float = 0.0
    task_count: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    final_report: str | None = None


class SwarmTaskRecordNode(RegistryNode):
    """Individual task execution record within a swarm run.

    CONCEPT:ORCH-1.4 — Swarm Preset Template Engine

    Attributes:
        task_id: Task identifier within the preset DAG.
        agent_id: Agent that executed this task.
        status: Task status (pending, in_progress, completed, failed).
        summary: Output summary from the task.
        depends_on: Upstream task IDs in the DAG.
        duration_seconds: Task execution duration.
    """

    type: RegistryNodeType = RegistryNodeType.SWARM_TASK_RECORD
    task_id: str = ""
    agent_id: str = ""
    status: str = "pending"
    summary: str = ""
    depends_on: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0


# --- Risk Scoring Nodes (CONCEPT:KG-2.6) ---


class RiskAssessmentNode(RegistryNode):
    """Domain-agnostic risk evaluation with multi-dimensional scoring.

    CONCEPT:KG-2.6 — Risk Scoring Ontology Extension

    Represents a point-in-time risk assessment for any entity (financial
    instrument, software component, project, etc.). Links to constituent
    ``RiskFactorNode`` entries via ``HAS_RISK_FACTOR``.

    Attributes:
        entity_id: The ID of the entity being assessed.
        overall_risk_score: Composite risk score (0.0 = safe, 1.0 = critical).
        risk_level: Human-readable risk category.
        assessment_type: Assessment methodology used.
        assessed_at: ISO timestamp of the assessment.
        assessor: Identifier of the agent/user performing the assessment.
    """

    type: RegistryNodeType = RegistryNodeType.RISK_ASSESSMENT
    entity_id: str = ""
    overall_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: str = "low"  # low, medium, high, critical
    assessment_type: str = ""  # var, drawdown, qualitative, composite
    assessed_at: str | None = None
    assessor: str = ""


class RiskFactorNode(RegistryNode):
    """An individual risk factor contributing to an assessment.

    CONCEPT:KG-2.6 — Risk Scoring Ontology Extension

    Represents a specific dimension of risk (market_risk, credit_risk,
    operational_risk, code_quality_risk, etc.). Multiple risk factors
    aggregate into a ``RiskAssessmentNode``.

    The ``PROPAGATES_RISK_TO`` edge enables transitive risk propagation
    via OWL reasoning: if A depends on B and B has high risk, A inherits
    that risk automatically.

    Attributes:
        factor_type: Risk category (market, credit, operational, liquidity, etc.).
        severity: Severity score (0.0 to 1.0).
        probability: Likelihood of the risk materializing (0.0 to 1.0).
        impact: Estimated impact magnitude (0.0 to 1.0).
        mitigation_status: Current mitigation state.
        details: Free-text explanation of the risk.
    """

    type: RegistryNodeType = RegistryNodeType.RISK_FACTOR
    factor_type: str = ""  # market, credit, operational, liquidity, concentration
    severity: float = Field(default=0.0, ge=0.0, le=1.0)
    probability: float = Field(default=0.5, ge=0.0, le=1.0)
    impact: float = Field(default=0.5, ge=0.0, le=1.0)
    mitigation_status: str = "unmitigated"  # unmitigated, partial, mitigated
    details: str = ""


class RiskMitigationNode(RegistryNode):
    """A proposed or applied mitigation for a risk factor.

    CONCEPT:KG-2.6 — Risk Scoring Ontology Extension

    Links to the ``RiskFactorNode`` it addresses via ``MITIGATED_BY``.

    Attributes:
        mitigation_type: Category of mitigation (hedge, diversification, limit, insurance).
        effectiveness: Estimated effectiveness (0.0 to 1.0).
        cost: Estimated cost of implementing the mitigation.
        status: Implementation status (proposed, active, expired).
        applied_at: ISO timestamp when mitigation was activated.
    """

    type: RegistryNodeType = RegistryNodeType.RISK_MITIGATION
    mitigation_type: str = ""  # hedge, diversification, limit, insurance, process
    effectiveness: float = Field(default=0.5, ge=0.0, le=1.0)
    cost: float = 0.0
    status: str = "proposed"  # proposed, active, expired
    applied_at: str | None = None


# --- Backtest / Evaluation Harness Nodes (CONCEPT:AHE-3.6) ---


class BacktestRunNode(RegistryNode):
    """A complete backtesting or evaluation run record.

    CONCEPT:AHE-3.6 — Backtest Evaluation Harness

    Records a full evaluation run including parameters, date range, and
    aggregate results. Links to ``StrategyNode`` via ``EVALUATED_STRATEGY``
    and to individual ``BacktestMetricNode`` entries via ``HAS_METRIC``.

    Attributes:
        strategy_id: Reference to the strategy being evaluated.
        start_date: ISO date for the evaluation start.
        end_date: ISO date for the evaluation end.
        initial_capital: Starting capital for the backtest.
        final_capital: Ending capital after the backtest.
        total_trades: Number of trades executed.
        parameters: Strategy parameters used in this run.
        status: Run status (running, completed, failed).
        walk_forward_windows: Number of walk-forward validation splits.
        benchmark_id: Optional benchmark for comparison.
    """

    type: RegistryNodeType = RegistryNodeType.BACKTEST_RUN
    strategy_id: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 100_000.0
    final_capital: float = 0.0
    total_trades: int = 0
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"  # running, completed, failed
    walk_forward_windows: int = 0
    benchmark_id: str | None = None


class BacktestMetricNode(RegistryNode):
    """An individual metric from a backtest or evaluation run.

    CONCEPT:AHE-3.6 — Backtest Evaluation Harness

    Represents a single named metric (Sharpe ratio, max drawdown, accuracy,
    F1, etc.) for a specific evaluation window. Links to the parent
    ``BacktestRunNode`` via ``HAS_METRIC``.

    Attributes:
        metric_name: Name of the metric (sharpe_ratio, max_drawdown, win_rate, etc.).
        value: Numeric value of the metric.
        window_index: Walk-forward window index (0 for aggregate).
        benchmark_value: Benchmark comparison value (if applicable).
        is_passing: Whether this metric meets the passing threshold.
    """

    type: RegistryNodeType = RegistryNodeType.BACKTEST_METRIC
    metric_name: str = ""
    value: float = 0.0
    window_index: int = 0  # 0 = aggregate, 1+ = per-window
    benchmark_value: float | None = None
    is_passing: bool = True


# --- Knowledge Distillation Nodes (CONCEPT:KG-2.2) ---


class EntityReference(BaseModel):
    """A named entity reference within an IdeaBlock.

    CONCEPT:KG-2.2 — Knowledge Distillation Engine

    Attributes:
        entity_name: The entity name (e.g., 'CLAUDE CODE').
        entity_type: The entity type (e.g., 'PRODUCT', 'ORGANIZATION').
    """

    entity_name: str
    entity_type: str = "CONCEPT"


class IdeaBlockNode(RegistryNode):
    """An atomic, structured knowledge unit.

    CONCEPT:KG-2.2 — Knowledge Distillation Engine

    Represents a single piece of knowledge as a question-answer pair with
    governance metadata, entity references, and retrieval keywords.
    Derived from Blockify's IdeaBlock specification but implemented as
    a pure Pydantic model.

    OWL: ``:IdeaBlock rdfs:subClassOf :Concept, bfo:0000031``

    Attributes:
        critical_question: The question this knowledge answers.
        trusted_answer: The validated, factual response.
        tags: Governance/classification tags (e.g., IMPORTANT, TECHNOLOGY).
        keywords: BM25-optimized retrieval terms.
        entities: Named entity references extracted from the content.
        source_document_id: Provenance link to the source document.
        distillation_round: Which iteration produced this block (0=original).
        merged_from: IDs of blocks merged to create this one.
    """

    type: RegistryNodeType = RegistryNodeType.IDEA_BLOCK
    critical_question: str = ""
    trusted_answer: str = ""
    tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    entities: list[EntityReference] = Field(default_factory=list)
    source_document_id: str | None = None
    distillation_round: int = 0
    merged_from: list[str] = Field(default_factory=list)


class DistillationRoundNode(RegistryNode):
    """A single iteration of the knowledge deduplication process.

    CONCEPT:KG-2.2 — Knowledge Distillation Engine

    Records metrics for one distillation round including the similarity
    threshold, block counts before and after, and the number of merges
    performed.

    OWL: ``:DistillationRound rdfs:subClassOf bfo:0000015``

    Attributes:
        iteration: Round number (1-indexed).
        similarity_threshold: Cosine similarity threshold used.
        blocks_before: Number of blocks entering this round.
        blocks_after: Number of blocks after deduplication.
        pairs_found: Number of similar pairs discovered.
        clusters_merged: Number of clusters that were merged.
    """

    type: RegistryNodeType = RegistryNodeType.DISTILLATION_ROUND
    iteration: int = 1
    similarity_threshold: float = 0.65
    blocks_before: int = 0
    blocks_after: int = 0
    pairs_found: int = 0
    clusters_merged: int = 0


# --- Research Intelligence Sub-Agent Nodes (CONCEPT:KG-2.6) ---


class ResearchSessionNode(RegistryNode):
    """A research sub-agent execution session.

    CONCEPT:KG-2.6 — Research Intelligence Sub-Agent

    Tracks an isolated research context window with its own token budget,
    tool whitelist, and findings. Adapted from ml-intern's research_tool.py
    sub-agent pattern with KG persistence.

    OWL: ``:ResearchSession rdfs:subClassOf :Episode``

    Attributes:
        query: The original research query.
        token_budget_warn: Token count at which warnings trigger.
        token_budget_max: Hard token ceiling for the session.
        tokens_used: Current token consumption.
        tools_allowed: Whitelist of read-only tools for this session.
        papers_discovered: Number of papers found during the session.
        citations_traversed: Depth of citation graph traversal.
        findings_count: Number of Evidence nodes created.
        status: Session state (active, completed, budget_exceeded, doom_looped).
    """

    type: RegistryNodeType = RegistryNodeType.RESEARCH_SESSION
    query: str = ""
    token_budget_warn: int = 170_000
    token_budget_max: int = 190_000
    tokens_used: int = 0
    tools_allowed: list[str] = Field(
        default_factory=lambda: ["search_papers", "read_paper", "search_datasets"]
    )
    papers_discovered: int = 0
    citations_traversed: int = 0
    findings_count: int = 0
    status: str = "active"  # active, completed, budget_exceeded, doom_looped


class CitationEdgeNode(RegistryNode):
    """A citation relationship between two research papers.

    CONCEPT:KG-2.6 — Research Intelligence Sub-Agent

    Represents directional citation edges with semantic metadata.
    Enables transitive citation graph traversal via ``wasDerivedFrom``
    OWL property chains.

    OWL: ``:CitationEdge rdfs:subClassOf :Evidence``

    Attributes:
        citing_paper_id: The paper that contains the citation.
        cited_paper_id: The paper being cited.
        citation_context: Text surrounding the citation in the citing paper.
        is_influential: Whether this citation is marked as influential (S2 flag).
        citation_intent: Purpose of citation (background, method, result, comparison).
        depth: Hop distance from the original search query paper.
    """

    type: RegistryNodeType = RegistryNodeType.CITATION_EDGE
    citing_paper_id: str = ""
    cited_paper_id: str = ""
    citation_context: str = ""
    is_influential: bool = False
    citation_intent: str = "background"  # background, method, result, comparison
    depth: int = 0


# --- Spectral Cluster Navigator Nodes (CONCEPT:KG-2.5) ---


class SpectralClusterNode(RegistryNode):
    """A semantically coherent cluster discovered via spectral decomposition.

    CONCEPT:KG-2.5 — Spectral Cluster Navigator

    Represents an auto-discovered group of semantically related entities
    using normalized Laplacian eigengap heuristics for tuning-free
    k-selection. Integrates with OWL via ``skos:Concept`` alignment.

    OWL: ``:SpectralCluster rdfs:subClassOf skos:Concept``

    Attributes:
        cluster_label: LLM-generated or heuristic label for this cluster.
        member_count: Number of entities in this cluster.
        eigengap_value: The eigengap that determined this cluster count.
        coherence_score: Intra-cluster cosine similarity average.
        centroid_embedding: Mean embedding vector of cluster members.
        parent_cluster_id: ID of parent cluster in hierarchical clustering.
        depth: Hierarchy depth (0 = root).
        domain: Domain context (codebase, research, financial, general).
    """

    type: RegistryNodeType = RegistryNodeType.SPECTRAL_CLUSTER
    cluster_label: str = ""
    member_count: int = 0
    eigengap_value: float = 0.0
    coherence_score: float = 0.0
    centroid_embedding: list[float] | None = None
    parent_cluster_id: str | None = None
    depth: int = 0
    domain: str = "general"  # codebase, research, financial, general


# --- Symbol Blast Radius Analyzer Nodes (CONCEPT:KG-2.5) ---


class BlastRadiusNode(RegistryNode):
    """A symbol-level impact analysis report.

    CONCEPT:KG-2.5 — Symbol Blast Radius Analyzer

    Tracks how widely a code symbol (function, class, variable) is used
    across the codebase. Adapted from contextplus's blast-radius.ts
    with KG integration for structural impact scoring.

    OWL: ``:BlastRadiusReport rdfs:subClassOf :Observation``

    Attributes:
        symbol_name: The symbol being analyzed.
        symbol_type: Type of symbol (function, class, variable, constant).
        definition_file: File where the symbol is defined.
        definition_line: Line number of the definition.
        usage_count: Total number of usages across the codebase.
        file_count: Number of distinct files using this symbol.
        impact_score: Normalized impact score (0.0-1.0).
        is_low_usage: Whether this symbol has suspiciously low usage.
        usages: List of usage locations as dicts with file, line, context.
    """

    type: RegistryNodeType = RegistryNodeType.BLAST_RADIUS_REPORT
    symbol_name: str = ""
    symbol_type: str = "function"  # function, class, variable, constant
    definition_file: str = ""
    definition_line: int = 0
    usage_count: int = 0
    file_count: int = 0
    impact_score: float = 0.0
    is_low_usage: bool = False
    usages: list[dict[str, Any]] = Field(default_factory=list)


# --- Auto-Similarity Memory Graph Nodes (CONCEPT:KG-2.3) ---


class SimilarityEdgeNode(RegistryNode):
    """An auto-created similarity link between memory nodes.

    CONCEPT:KG-2.3 — Auto-Similarity Memory Graph

    Represents a similarity relationship discovered during memory
    node insertion. Uses cosine similarity thresholding with exponential
    decay scoring. Adapted from contextplus's memory-graph.ts auto-linking.

    OWL: ``:SimilarityEdge rdfs:subClassOf :Evidence``

    Attributes:
        source_node_id: The node that triggered the similarity check.
        target_node_id: The node found to be similar.
        cosine_similarity: Raw cosine similarity score at creation time.
        decay_lambda: Exponential decay rate (higher = faster decay).
        current_weight: Decayed weight (updated on access).
        creation_epoch: Unix timestamp of edge creation.
        last_accessed_epoch: Unix timestamp of last access.
        access_count: Number of times this edge was traversed.
    """

    type: RegistryNodeType = RegistryNodeType.SIMILARITY_EDGE
    source_node_id: str = ""
    target_node_id: str = ""
    cosine_similarity: float = 0.0
    decay_lambda: float = 0.01  # ~70-day half-life
    current_weight: float = 1.0
    creation_epoch: float = 0.0
    last_accessed_epoch: float = 0.0
    access_count: int = 0


class MemoryDecayConfig(BaseModel):
    """Configuration for auto-similarity and decay parameters.

    CONCEPT:KG-2.3 — Auto-Similarity Memory Graph

    Attributes:
        similarity_threshold: Minimum cosine similarity to create an edge (default 0.72).
        decay_lambda: Default decay rate for new similarity edges.
        prune_threshold: Minimum weight before an edge is pruned.
        max_edges_per_node: Maximum similarity edges per node (prevents hub explosion).
        batch_window: Number of recent nodes to compare against during insertion.
    """

    similarity_threshold: float = 0.72
    decay_lambda: float = 0.01
    prune_threshold: float = 0.05
    max_edges_per_node: int = 20
    batch_window: int = 100


# --- Hybrid Search Index Configuration (CONCEPT:KG-2.3) ---


class HybridSearchConfig(BaseModel):
    """Configuration for hybrid semantic+keyword search scoring.

    CONCEPT:KG-2.3 — Hybrid Search Index

    Adapted from contextplus's embedding.ts hybrid scoring with
    configurable weight/threshold parameters.

    Attributes:
        semantic_weight: Weight for semantic (vector) similarity (default 0.72).
        keyword_weight: Weight for keyword (BM25-style) matching (default 0.28).
        min_semantic_score: Minimum semantic score to include a result.
        min_keyword_score: Minimum keyword score to include a result.
        min_combined_score: Minimum combined score to include a result.
        phrase_boost: Boost factor for exact phrase matches.
        top_k: Maximum number of results to return.
    """

    semantic_weight: float = 0.72
    keyword_weight: float = 0.28
    min_semantic_score: float = 0.0
    min_keyword_score: float = 0.0
    min_combined_score: float = 0.1
    phrase_boost: float = 0.15
    top_k: int = 10


# --- Enhanced Doom-Loop Detector Nodes (CONCEPT:OS-5.0) ---


class DoomLoopIncidentNode(RegistryNode):
    """A detected tool-call repetition pattern.

    CONCEPT:OS-5.0 — Enhanced Doom-Loop Detector

    Records pattern-aware doom-loop detection incidents including
    consecutive identical calls and repeating multi-tool sequences.
    Adapted from ml-intern's doom_loop.py with KG persistence and
    corrective prompt generation.

    OWL: ``:DoomLoopIncident rdfs:subClassOf :Incident``

    Attributes:
        pattern_type: Type of pattern detected (consecutive, sequence).
        tool_names: Tools involved in the loop.
        signature_hashes: Hashed tool-call signatures that triggered detection.
        repetition_count: Number of times the pattern repeated.
        corrective_prompt: Generated prompt to break the loop.
        session_id: Session where the loop was detected.
        was_broken: Whether the loop was successfully broken.
    """

    type: RegistryNodeType = RegistryNodeType.DOOM_LOOP_INCIDENT
    pattern_type: str = "consecutive"  # consecutive, sequence
    tool_names: list[str] = Field(default_factory=list)
    signature_hashes: list[str] = Field(default_factory=list)
    repetition_count: int = 0
    corrective_prompt: str = ""
    session_id: str = ""
    was_broken: bool = False


# --- RAG-KG Unification Node (CONCEPT:KG-2.3) ---


class UnifiedRAGConfigNode(RegistryNode):
    """Configuration snapshot for a unified RAG-KG retrieval session.

    CONCEPT:KG-2.3 — RAG-KG Unification

    Tracks the configuration and performance metrics for unified
    retrieval sessions that combine similarity shortcuts, spectral
    cluster scoping, and hybrid scoring.

    Attributes:
        enable_similarity_shortcuts: Whether shortcut edges were used.
        enable_cluster_scoping: Whether spectral cluster scoping was used.
        enable_hybrid_scoring: Whether hybrid scoring was used.
        shortcut_hits: Number of nodes found via shortcut edges.
        cluster_scoped: Number of queries scoped to clusters.
        full_scans: Number of fallback full-index scans.
        avg_latency_ms: Average retrieval latency in milliseconds.
    """

    type: RegistryNodeType = RegistryNodeType.UNIFIED_RAG_CONFIG
    enable_similarity_shortcuts: bool = True
    enable_cluster_scoping: bool = True
    enable_hybrid_scoring: bool = True
    shortcut_hits: int = 0
    cluster_scoped: int = 0
    full_scans: int = 0
    avg_latency_ms: float = 0.0


# --- Research Orchestration Node (CONCEPT:KG-2.6) ---


class OrchestrationCycleNode(RegistryNode):
    """Record of a research orchestration cycle execution.

    CONCEPT:KG-2.6 — Research Orchestration Integration

    Tracks the results of an automated research-to-KG cycle including
    discovery, citation traversal, similarity linking, and cluster
    refresh phases.

    Attributes:
        cycle_id: Unique cycle identifier.
        papers_discovered: Number of papers found.
        papers_ingested: Number of papers ingested.
        citations_traversed: Total citation edges traversed.
        similarity_edges_created: Auto-similarity edges created.
        clusters_built: Spectral clusters built/refreshed.
        duration_seconds: Total cycle duration.
        query: Focus query if provided.
    """

    type: RegistryNodeType = RegistryNodeType.ORCHESTRATION_CYCLE
    cycle_id: str = ""
    papers_discovered: int = 0
    papers_ingested: int = 0
    citations_traversed: int = 0
    similarity_edges_created: int = 0
    clusters_built: int = 0
    duration_seconds: float = 0.0
    query: str = ""


# --- Graph Distillation Node (CONCEPT:KG-2.6) ---


class DistillationIndexNode(RegistryNode):
    """Snapshot of the graph distillation index health.

    CONCEPT:KG-2.6 — Graph Distillation Migration

    Records the state of the pre-computed similarity edge index
    for monitoring and operational awareness.

    Attributes:
        total_nodes: Total nodes with embeddings.
        nodes_with_shortcuts: Nodes with at least one shortcut edge.
        total_edges: Total similarity edges.
        coverage_ratio: Fraction of nodes with shortcuts.
        avg_edge_weight: Mean decayed edge weight.
        stale_edge_count: Edges below prune threshold.
        recommendation: System-generated health recommendation.
    """

    type: RegistryNodeType = RegistryNodeType.DISTILLATION_INDEX
    total_nodes: int = 0
    nodes_with_shortcuts: int = 0
    total_edges: int = 0
    coverage_ratio: float = 0.0
    avg_edge_weight: float = 0.0
    stale_edge_count: int = 0
    recommendation: str = ""
