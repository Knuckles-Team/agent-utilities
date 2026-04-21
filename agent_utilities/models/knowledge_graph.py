#!/usr/bin/python
"""Registry Graph Models Module.

This module defines the Pydantic models used for the hybrid graph representation
of the agent registry (NODE_AGENTS.md). It supports topological and semantic
discovery of specialists and their tools.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


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


class RegistryNode(BaseModel):
    """Base class for all nodes in the registry graph."""

    id: str = Field(description="Unique identifier for the node")
    type: RegistryNodeType
    name: str
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance_score: float = 0.0
    timestamp: str | None = None
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


class PipelineConfig(BaseModel):
    """Configuration for the Unified Intelligence Pipeline."""

    workspace_path: str
    enable_embeddings: bool = True
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
    embedding: list[float] | None = None


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


class FactNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.FACT
    content: str
    certainty: float = 1.0


class ConceptNode(RegistryNode):
    """Atomic knowledge unit (e.g. 'p53 gene', 'SN2 reaction')."""

    type: RegistryNodeType = RegistryNodeType.CONCEPT
    concept_id: str
    definition: str = ""
    embedding: list[float] | None = None
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
    embedding: list[float] | None = None


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
    embedding: list[float] | None = None
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
    embedding: list[float] | None = None


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
    embedding: list[float] | None = None


class KBFactNode(RegistryNode):
    """An atomic fact with certainty score extracted from KB articles."""

    type: RegistryNodeType = RegistryNodeType.KB_FACT
    content: str
    certainty: float = 1.0
    source_ids: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None


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
    embedding: list[float] | None = None


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
