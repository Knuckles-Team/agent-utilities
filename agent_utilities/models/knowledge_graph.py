#!/usr/bin/python
"""Registry Graph Models Module.

This module defines the Pydantic models used for the hybrid graph representation
of the agent registry (NODE_AGENTS.md). It supports topological and semantic
discovery of specialists and their tools.
"""

from __future__ import annotations

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


from ..config import (
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


class PersonNode(RegistryNode):
    """Researchers, authors, or agents."""

    type: RegistryNodeType = RegistryNodeType.PERSON
    person_id: str
    expertise: list[str] = Field(default_factory=list)
    affiliation: str | None = None


class CapabilityNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.CAPABILITY


# --- Callable Resources & Agent Nodes ---


class ToolMetadataNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.TOOL_METADATA
    tags: list[str] = Field(default_factory=list)
    prompt_template: str | None = None
    resources: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    auth_requirements: dict[str, Any] | None = None
    version: str | None = None
    source: str  # MCP, A2A, INTERNAL, AGENT_SKILL


class CallableResourceNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.CALLABLE_RESOURCE
    resource_type: str  # MCP_TOOL, A2A_AGENT, INTERNAL_SKILL, AGENT_SKILL
    endpoint: str | None = None
    agent_card: dict[str, Any] | None = None
    skill_code_path: str | None = None
    metadata_id: str


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
