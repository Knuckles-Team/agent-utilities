#!/usr/bin/python
from __future__ import annotations

"""Company Brain Infrastructure Models (CONCEPT:KG-2.6).

Pydantic models for the Company Brain operational state layer.
These models define the data structures for the six infrastructure
primitives that transform agent-utilities from a single-agent
knowledge graph into a multi-writer, multi-reader, multi-tenant
organizational brain.

The Company Brain is **actor-agnostic**: every primitive treats humans,
AI agents, and hybrid human+AI teams as equal first-class participants.
An ``ActorType`` enum distinguishes them where needed, but the
infrastructure itself imposes no hierarchy between organic and
synthetic intelligence.

Design Principles:
    1. **Actor-Agnostic**: Humans, AIs, and hybrid teams are all
       first-class participants with identical capabilities.
    2. **Ontology-First**: All models align with the existing OWL
       ontology (BFO/PROV-O/SKOS/FIBO).
    3. **Composable**: Each model can be used independently or
       composed into larger workflows.
    4. **Provenance-Native**: Every mutation carries attribution
       and derivation metadata.

Infrastructure Primitives:
    - **Concurrency Control** (Gap 1): Version vectors, CAS, locks
    - **Multi-Tenancy** (Gap 2): Tenant isolation, hierarchies
    - **Conflict Resolution** (Gap 3): Contradiction detection, merges
    - **Provenance Tracking** (Gap 4): Trust hierarchies, read audits
    - **Event Streaming** (Gap 5): Webhook adapters, CDC
    - **Data-Level Permissions** (Gap 6): Node ACLs, classification

See Also:
    - :mod:`agent_utilities.knowledge_graph.core.company_brain`
    - ``docs/company_brain/index.md``
"""


import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Core Enums — Actor Types & Assertion Semantics
# ---------------------------------------------------------------------------


class ActorType(StrEnum):
    """Type of actor interacting with the Company Brain.

    The Company Brain is actor-agnostic: humans, AI agents, automated
    services, and hybrid human+AI teams are all first-class participants.
    This enum exists for provenance and audit purposes, not for access
    control differentiation.

    Values:
        HUMAN: A human user (employee, contractor, external stakeholder).
        AI_AGENT: An autonomous AI agent operating within the ecosystem.
        AUTOMATED_SERVICE: A non-AI automated system (CI/CD, cron, webhook).
        HYBRID_TEAM: A human+AI collaborative unit acting as a single actor.
        SYSTEM: The Company Brain infrastructure itself (maintenance, synthesis).
    """

    HUMAN = "human"
    AI_AGENT = "ai_agent"
    AUTOMATED_SERVICE = "automated_service"
    HYBRID_TEAM = "hybrid_team"
    SYSTEM = "system"


class AssertionType(StrEnum):
    """Classification of how a fact entered the Company Brain.

    Every writable node carries an assertion type to distinguish raw
    observations from derived inferences and human judgments. This
    enables conflict resolution to apply source-appropriate strategies.

    Values:
        RAW_DATA: Direct observation from a system of record (CRM, HRIS, git).
        AGENT_INFERENCE: An AI agent's derived conclusion or interpretation.
        HUMAN_JUDGMENT: A human's explicit assessment or decision.
        SYNTHESIZED: Result of the SynthesisEngine distillation process.
        EXTERNAL_IMPORT: Data imported from an external system without verification.
    """

    RAW_DATA = "raw_data"
    AGENT_INFERENCE = "agent_inference"
    HUMAN_JUDGMENT = "human_judgment"
    SYNTHESIZED = "synthesized"
    EXTERNAL_IMPORT = "external_import"


class MergeStrategy(StrEnum):
    """Strategy for resolving conflicting writes to the same node.

    When two actors write conflicting values to the same node, the
    ConflictResolver applies one of these strategies. Strategies
    can be configured globally, per-node-type, or per-tenant.

    Values:
        LAST_WRITE_WINS: Most recent write overwrites previous value.
            Simple but loses information. Suitable for low-stakes state.
        HIGHEST_CONFIDENCE_WINS: Write with higher confidence score
            takes precedence. Requires confidence metadata on mutations.
        SOURCE_AUTHORITY_WINS: Write from a higher-authority source wins.
            Uses the TrustHierarchy to rank sources.
        REQUIRE_HUMAN_ARBITRATION: Conflict is flagged and both values
            are preserved until a human resolves it.
        MERGE_APPEND: Both values are kept as a list. Useful for
            accumulating observations rather than replacing state.
    """

    LAST_WRITE_WINS = "last_write_wins"
    HIGHEST_CONFIDENCE_WINS = "highest_confidence_wins"
    SOURCE_AUTHORITY_WINS = "source_authority_wins"
    REQUIRE_HUMAN_ARBITRATION = "require_human_arbitration"
    MERGE_APPEND = "merge_append"


class DataClassification(StrEnum):
    """Regulatory data classification label for node-level ACLs.

    Applied to nodes to enforce data-level permissions. Access control
    is determined by comparing the requesting actor's clearance level
    against the node's classification.

    Values:
        PUBLIC: Visible to all authenticated actors.
        INTERNAL: Visible to all actors within the tenant.
        CONFIDENTIAL: Visible only to actors with explicit grant.
        RESTRICTED: Visible only to designated data owners and admins.
            Requires audit logging on every access.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class LockMode(StrEnum):
    """Lock mode for graph-level concurrency control.

    Values:
        OPTIMISTIC: No lock acquired upfront; version check at commit time.
            Best for low-contention workloads.
        PESSIMISTIC: Exclusive lock acquired before mutation.
            Guarantees no conflicts but reduces throughput.
        ADVISORY: Lock is informational only; conflicts are detected
            but not prevented. Useful for read-heavy workloads with
            occasional writes.
    """

    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    ADVISORY = "advisory"


class ConflictStatus(StrEnum):
    """Status of a detected conflict.

    Values:
        OPEN: Conflict detected but not yet resolved.
        RESOLVED_AUTO: Resolved automatically by merge strategy.
        RESOLVED_HUMAN: Resolved by human arbitration.
        ESCALATED: Escalated to a higher authority for resolution.
        STALE: Conflict is older than retention window; auto-archived.
    """

    OPEN = "open"
    RESOLVED_AUTO = "resolved_auto"
    RESOLVED_HUMAN = "resolved_human"
    ESCALATED = "escalated"
    STALE = "stale"


class EventSourceType(StrEnum):
    """Source type for real-time event ingestion.

    Values:
        WEBHOOK: External system pushing events via HTTP callback.
        KAFKA: Apache Kafka topic consumer.
        NATS: NATS JetStream consumer.
        REDIS_STREAM: Redis Streams consumer.
        POLLING: Active polling of an external API.
        CDC: Change Data Capture from a database.
        MCP: Model Context Protocol server event.
        A2A: Agent-to-Agent protocol message.
    """

    WEBHOOK = "webhook"
    KAFKA = "kafka"
    NATS = "nats"
    REDIS_STREAM = "redis_stream"
    POLLING = "polling"
    CDC = "cdc"
    MCP = "mcp"
    A2A = "a2a"


# ---------------------------------------------------------------------------
# Concurrency Control Models (Gap 1)
# ---------------------------------------------------------------------------


class VersionVector(BaseModel):
    """Version vector for a single node in the graph.

    Each node maintains a version vector tracking the last-known write
    from each actor. This enables optimistic concurrency control: at
    commit time, the engine compares the mutation's base version against
    the current version and rejects stale writes.

    Attributes:
        node_id: The graph node this version vector tracks.
        versions: Mapping of actor_id → version counter.
        current_version: Monotonically increasing version number.
        last_writer: Actor ID that performed the most recent write.
        last_written_at: ISO timestamp of the most recent write.
    """

    node_id: str
    versions: dict[str, int] = Field(default_factory=dict)
    current_version: int = 0
    last_writer: str = ""
    last_written_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def increment(self, actor_id: str) -> int:
        """Increment the version for a specific actor."""
        self.current_version += 1
        self.versions[actor_id] = self.current_version
        self.last_writer = actor_id
        self.last_written_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return self.current_version

    def is_stale(self, base_version: int) -> bool:
        """Check if a mutation is based on a stale version."""
        return base_version < self.current_version


class GraphLock(BaseModel):
    """A lock on a specific node or edge in the graph.

    Attributes:
        lock_id: Unique identifier for this lock.
        target_id: The node or edge being locked.
        holder_id: Actor holding the lock.
        holder_type: Type of actor holding the lock.
        mode: Lock mode (optimistic, pessimistic, advisory).
        acquired_at: ISO timestamp when lock was acquired.
        expires_at: ISO timestamp when lock automatically expires.
        is_active: Whether the lock is currently active.
    """

    lock_id: str = Field(default_factory=lambda: f"lock:{uuid.uuid4().hex[:12]}")
    target_id: str
    holder_id: str
    holder_type: ActorType = ActorType.AI_AGENT
    mode: LockMode = LockMode.OPTIMISTIC
    acquired_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    expires_at: str = ""
    is_active: bool = True


class CASResult(BaseModel):
    """Result of a Compare-And-Swap operation.

    Attributes:
        success: Whether the CAS succeeded.
        node_id: The node that was targeted.
        expected_version: The version the caller expected.
        actual_version: The version found at commit time.
        new_version: The version after successful CAS (0 if failed).
        conflict_detected: Whether a conflict was detected.
    """

    success: bool
    node_id: str
    expected_version: int
    actual_version: int
    new_version: int = 0
    conflict_detected: bool = False


# ---------------------------------------------------------------------------
# Multi-Tenancy Models (Gap 2)
# ---------------------------------------------------------------------------


class TenantNode(BaseModel):
    """Represents a tenant (team, department, organization) in the graph.

    Tenants form a hierarchy: a parent tenant's admins can see all
    child tenant data, but child tenants cannot see sibling data.

    Attributes:
        tenant_id: Unique identifier for this tenant.
        name: Human-readable tenant name.
        parent_tenant_id: Parent tenant for hierarchical scoping.
        description: Description of this tenant's purpose.
        created_at: ISO timestamp when tenant was created.
        created_by: Actor who created this tenant.
        created_by_type: Type of actor who created this tenant.
        is_active: Whether this tenant is currently active.
        metadata: Additional tenant metadata.
    """

    tenant_id: str = Field(default_factory=lambda: f"tenant:{uuid.uuid4().hex[:8]}")
    name: str
    parent_tenant_id: str = ""
    description: str = ""
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    created_by: str = "system"
    created_by_type: ActorType = ActorType.SYSTEM
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class TenantMembership(BaseModel):
    """Maps an actor to a tenant with a specific role.

    Attributes:
        actor_id: The actor (human or AI) being assigned.
        actor_type: Type of actor.
        tenant_id: The tenant they belong to.
        role: Role within the tenant (admin, member, viewer, service).
        granted_at: ISO timestamp when membership was granted.
        granted_by: Actor who granted this membership.
    """

    actor_id: str
    actor_type: ActorType = ActorType.HUMAN
    tenant_id: str
    role: str = "member"
    granted_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    granted_by: str = "system"


# ---------------------------------------------------------------------------
# Conflict Resolution Models (Gap 3)
# ---------------------------------------------------------------------------


class ConflictNode(BaseModel):
    """Represents a detected conflict between two writes to the same node.

    When two actors write conflicting values to the same node, the
    ConflictResolver emits a ConflictNode preserving both versions
    and their provenance. The conflict remains open until resolved
    by a merge strategy or human arbitration.

    Attributes:
        conflict_id: Unique identifier for this conflict.
        node_id: The graph node where the conflict occurred.
        field_name: The specific field that conflicted.
        value_a: First conflicting value (with provenance).
        value_b: Second conflicting value (with provenance).
        actor_a: Actor who wrote value_a.
        actor_a_type: Type of actor who wrote value_a.
        actor_b: Actor who wrote value_b.
        actor_b_type: Type of actor who wrote value_b.
        assertion_type_a: How value_a entered the brain.
        assertion_type_b: How value_b entered the brain.
        confidence_a: Confidence score of value_a.
        confidence_b: Confidence score of value_b.
        status: Current resolution status.
        resolution_strategy: Strategy used to resolve (if resolved).
        resolved_value: The winning value after resolution.
        resolved_by: Actor who resolved the conflict.
        resolved_at: ISO timestamp of resolution.
        detected_at: ISO timestamp when conflict was detected.
    """

    conflict_id: str = Field(
        default_factory=lambda: f"conflict:{uuid.uuid4().hex[:10]}"
    )
    node_id: str
    field_name: str
    value_a: Any = None
    value_b: Any = None
    actor_a: str = ""
    actor_a_type: ActorType = ActorType.AI_AGENT
    actor_b: str = ""
    actor_b_type: ActorType = ActorType.AI_AGENT
    assertion_type_a: AssertionType = AssertionType.AGENT_INFERENCE
    assertion_type_b: AssertionType = AssertionType.AGENT_INFERENCE
    confidence_a: float = 0.5
    confidence_b: float = 0.5
    status: ConflictStatus = ConflictStatus.OPEN
    resolution_strategy: MergeStrategy | None = None
    resolved_value: Any = None
    resolved_by: str = ""
    resolved_at: str = ""
    detected_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


# ---------------------------------------------------------------------------
# Provenance Tracking Models (Gap 4)
# ---------------------------------------------------------------------------


class ProvenanceRecord(BaseModel):
    """A provenance record tracking how a node was created or modified.

    Every write to the Company Brain generates a ProvenanceRecord that
    captures who wrote it, why, from what source, and with what
    confidence. This enables trust-weighted retrieval and audit trails.

    Attributes:
        record_id: Unique identifier for this provenance record.
        node_id: The node this provenance applies to.
        actor_id: Who performed the write.
        actor_type: Type of actor (human, AI, hybrid team, etc.).
        action: What was done (create, update, delete, merge).
        assertion_type: How this fact entered the brain.
        confidence: Confidence score (0.0–1.0).
        source_system: System of record (CRM, Slack, git, manual).
        derived_from: List of node IDs this was derived from.
        attributed_to: Actor or system this is attributed to.
        rationale: Free-text explanation of why this write happened.
        timestamp: ISO timestamp of the write.
        session_id: Session context for grouping related writes.
        tenant_id: Tenant scope of this write.
    """

    record_id: str = Field(default_factory=lambda: f"prov:{uuid.uuid4().hex[:10]}")
    node_id: str
    actor_id: str
    actor_type: ActorType = ActorType.AI_AGENT
    action: str = "create"
    assertion_type: AssertionType = AssertionType.AGENT_INFERENCE
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source_system: str = ""
    derived_from: list[str] = Field(default_factory=list)
    attributed_to: str = ""
    rationale: str = ""
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    session_id: str = ""
    tenant_id: str = ""


class TrustHierarchyEntry(BaseModel):
    """A single entry in the source trust hierarchy.

    Defines the authority level of a data source for a specific
    data domain. Higher authority sources win during conflict
    resolution when SOURCE_AUTHORITY_WINS strategy is used.

    Example:
        TrustHierarchyEntry(
            source_system="crm",
            data_domain="customer",
            authority_level=0.95,
            rationale="CRM is the system of record for customer data"
        )

    Attributes:
        source_system: The source system identifier.
        data_domain: The data domain this authority applies to.
        authority_level: Authority score (0.0–1.0, higher = more trusted).
        rationale: Why this source has this authority level.
        overrides: List of source systems this one overrides.
        trust_decay_rate: Rate at which trust decays over time without reinforcement (0.0–1.0 per day).
        conflict_penalty: Penalty applied to authority_level when this source is overridden by human arbitration.
    """

    source_system: str
    data_domain: str = "*"
    authority_level: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""
    overrides: list[str] = Field(default_factory=list)
    trust_decay_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    conflict_penalty: float = Field(default=0.05, ge=0.0, le=1.0)


class ReadAuditEntry(BaseModel):
    """An audit record for read operations on the Company Brain.

    Tracks who queried what, when, enabling "who saw it" provenance
    that the gap analysis identified as missing.

    Attributes:
        entry_id: Unique identifier.
        actor_id: Who performed the read.
        actor_type: Type of actor.
        query_type: Type of query (cypher, semantic, traversal).
        nodes_accessed: List of node IDs that were returned.
        query_summary: Summary of the query (not the raw query for security).
        timestamp: ISO timestamp of the read.
        tenant_id: Tenant context of the read.
    """

    entry_id: str = Field(default_factory=lambda: f"read:{uuid.uuid4().hex[:10]}")
    actor_id: str
    actor_type: ActorType = ActorType.AI_AGENT
    query_type: str = "traversal"
    nodes_accessed: list[str] = Field(default_factory=list)
    query_summary: str = ""
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    tenant_id: str = ""


# ---------------------------------------------------------------------------
# Event Streaming Models (Gap 5)
# ---------------------------------------------------------------------------


class EventStreamConfig(BaseModel):
    """Configuration for an event stream ingestion source.

    Attributes:
        stream_id: Unique identifier for this stream.
        name: Human-readable name.
        source_type: Type of event source (webhook, Kafka, etc.).
        endpoint: Connection endpoint (URL, topic, channel).
        tenant_id: Tenant scope for ingested events.
        actor_id: Actor identity for provenance attribution.
        actor_type: Type of actor.
        transform_rules: Optional mapping rules for event → graph mutation.
        enabled: Whether this stream is currently active.
        batch_size: Number of events to batch before committing.
        retry_max: Maximum retry attempts for failed ingestions.
        created_at: ISO timestamp.
    """

    stream_id: str = Field(default_factory=lambda: f"stream:{uuid.uuid4().hex[:8]}")
    name: str
    source_type: EventSourceType
    endpoint: str = ""
    tenant_id: str = ""
    actor_id: str = "system"
    actor_type: ActorType = ActorType.AUTOMATED_SERVICE
    transform_rules: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    batch_size: int = 10
    retry_max: int = 3
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


class WebhookEvent(BaseModel):
    """A single event received from a webhook source.

    Attributes:
        event_id: Unique identifier for this event.
        source_type: Type of source system (slack, jira, github, etc.).
        event_type: The event type (message.posted, issue.created, etc.).
        payload: Raw event payload.
        actor_id: Actor who triggered the event.
        actor_type: Type of actor.
        timestamp: ISO timestamp of the event.
        tenant_id: Tenant scope.
        processed: Whether the event has been ingested into the graph.
        retry_count: Number of ingestion attempts.
    """

    event_id: str = Field(default_factory=lambda: f"evt:{uuid.uuid4().hex[:10]}")
    source_type: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    actor_id: str = ""
    actor_type: ActorType = ActorType.HUMAN
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    tenant_id: str = ""
    processed: bool = False
    retry_count: int = 0


class IngestionResult(BaseModel):
    """Result of ingesting events into the Company Brain.

    Attributes:
        stream_id: The stream that produced this result.
        events_received: Number of events received.
        events_ingested: Number successfully ingested into the graph.
        events_failed: Number that failed ingestion.
        nodes_created: Number of new graph nodes created.
        edges_created: Number of new graph edges created.
        conflicts_detected: Number of conflicts detected during ingestion.
        duration_ms: Time taken for the ingestion batch.
        timestamp: ISO timestamp.
    """

    stream_id: str = ""
    events_received: int = 0
    events_ingested: int = 0
    events_failed: int = 0
    nodes_created: int = 0
    edges_created: int = 0
    conflicts_detected: int = 0
    duration_ms: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


# ---------------------------------------------------------------------------
# Data-Level Permissions Models (Gap 6)
# ---------------------------------------------------------------------------


class NodeACL(BaseModel):
    """Access Control List for a specific node.

    Defines who can read, write, or admin a specific node.
    ACLs are evaluated at query time to filter results based
    on the requesting actor's identity and tenant membership.

    Attributes:
        node_id: The node this ACL protects.
        classification: Data classification level.
        data_owner: Actor who owns this data.
        data_owner_type: Type of data owner.
        read_actors: Actor IDs with explicit read access.
        write_actors: Actor IDs with explicit write access.
        admin_actors: Actor IDs with admin access (can modify ACL).
        read_roles: Roles with read access (e.g., "manager", "analyst").
        write_roles: Roles with write access.
        tenant_id: Tenant scope.
        inherit_from_parent: Whether to inherit ACLs from parent nodes.
        audit_on_access: Whether to log every read to the audit trail.
    """

    node_id: str
    classification: DataClassification = DataClassification.INTERNAL
    data_owner: str = ""
    data_owner_type: ActorType = ActorType.HUMAN
    read_actors: list[str] = Field(default_factory=list)
    write_actors: list[str] = Field(default_factory=list)
    admin_actors: list[str] = Field(default_factory=list)
    read_roles: list[str] = Field(default_factory=list)
    write_roles: list[str] = Field(default_factory=list)
    tenant_id: str = ""
    inherit_from_parent: bool = True
    audit_on_access: bool = False


class PermissionCheckResult(BaseModel):
    """Result of a permission check against a node's ACL.

    Attributes:
        allowed: Whether access is permitted.
        node_id: The node that was checked.
        actor_id: The actor requesting access.
        actor_type: Type of the requesting actor.
        action: The action requested (read, write, admin).
        reason: Human-readable explanation of the decision.
        classification: The node's data classification.
        audit_logged: Whether this check was logged to the audit trail.
    """

    allowed: bool
    node_id: str
    actor_id: str
    actor_type: ActorType = ActorType.AI_AGENT
    action: str = "read"
    reason: str = ""
    classification: DataClassification = DataClassification.INTERNAL
    audit_logged: bool = False


# ---------------------------------------------------------------------------
# Company Brain Node Type & Edge Type Extensions
# ---------------------------------------------------------------------------


class CompanyBrainNodeType(StrEnum):
    """Additional node types for Company Brain infrastructure.

    These extend the existing RegistryNodeType enum with
    Company Brain-specific concepts.
    """

    CONFLICT = "conflict"
    TENANT = "tenant"
    PROVENANCE_RECORD = "provenance_record"
    READ_AUDIT = "read_audit"
    EVENT_STREAM = "event_stream"
    NODE_ACL = "node_acl"
    TRUST_HIERARCHY = "trust_hierarchy"
    VERSION_VECTOR = "version_vector"
    GRAPH_LOCK = "graph_lock"


class CompanyBrainEdgeType(StrEnum):
    """Additional edge types for Company Brain infrastructure.

    These extend the existing RegistryEdgeType enum with
    Company Brain-specific relationships.
    """

    CONFLICTS_WITH_VALUE = "conflicts_with_value"
    RESOLVED_BY_ACTOR = "resolved_by_actor"
    BELONGS_TO_TENANT = "belongs_to_tenant"
    CHILD_TENANT_OF = "child_tenant_of"
    HAS_PROVENANCE = "has_provenance"
    DERIVED_FROM_SOURCE = "derived_from_source"
    ATTRIBUTED_TO_ACTOR = "attributed_to_actor"
    INGESTED_FROM_STREAM = "ingested_from_stream"
    HAS_ACL = "has_acl"
    OWNS_DATA = "owns_data"
    LOCKED_BY = "locked_by"
    VERSION_TRACKED_BY = "version_tracked_by"
    READ_BY_ACTOR = "read_by_actor"
    TRUST_OVERRIDES = "trust_overrides"


__all__ = [
    "ActorType",
    "AssertionType",
    "CASResult",
    "CompanyBrainEdgeType",
    "CompanyBrainNodeType",
    "ConflictNode",
    "ConflictStatus",
    "DataClassification",
    "EventSourceType",
    "EventStreamConfig",
    "GraphLock",
    "IngestionResult",
    "LockMode",
    "MergeStrategy",
    "NodeACL",
    "PermissionCheckResult",
    "ProvenanceRecord",
    "ReadAuditEntry",
    "TenantMembership",
    "TenantNode",
    "TrustHierarchyEntry",
    "VersionVector",
    "WebhookEvent",
]
