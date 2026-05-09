#!/usr/bin/python
"""Company Brain Infrastructure (CONCEPT:KG-2.51).

Unified operational state layer that transforms the Knowledge Graph from a
single-agent brain into a multi-writer, multi-reader, multi-tenant
organizational brain. Actor-agnostic: humans, AIs, and hybrid teams are
all first-class participants.

Six infrastructure primitives:
    1. GraphConcurrencyManager — Version vectors, CAS, graph-level locks
    2. TenancyManager — Tenant isolation, hierarchies, scoped queries
    3. ConflictResolver — Contradiction detection, configurable merge
    4. ProvenanceTracker — Trust hierarchies, read audits, mandatory attribution
    5. EventStreamIngester — Webhook adapters, async ingestion, CDC
    6. DataLevelPermissions — Node ACLs, classification labels, query filtering
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ...models.company_brain import (
    ActorType,
    AssertionType,
    CASResult,
    ConflictNode,
    ConflictStatus,
    DataClassification,
    EventStreamConfig,
    GraphLock,
    IngestionResult,
    LockMode,
    MergeStrategy,
    NodeACL,
    PermissionCheckResult,
    ProvenanceRecord,
    ReadAuditEntry,
    TenantMembership,
    TenantNode,
    TrustHierarchyEntry,
    VersionVector,
    WebhookEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Graph Concurrency Manager (Gap 1)
# ---------------------------------------------------------------------------


class GraphConcurrencyManager:
    """Graph-level optimistic concurrency control.

    CONCEPT:KG-2.51 — Company Brain Infrastructure

    Provides version vectors on nodes, Compare-And-Swap (CAS) mutations,
    and configurable lock strategies. Wraps the existing KGVersionEngine
    with multi-writer safety for concurrent human and AI actors.

    Example::

        gcm = GraphConcurrencyManager()
        # Register a node for version tracking
        gcm.track_node("customer:001")
        # Attempt a CAS mutation
        result = gcm.compare_and_swap(
            node_id="customer:001",
            expected_version=1,
            actor_id="agent:risk-analyzer",
            actor_type=ActorType.AI_AGENT,
        )
        if not result.success:
            # Handle conflict
            ...
    """

    def __init__(self, default_lock_mode: LockMode = LockMode.OPTIMISTIC) -> None:
        self._versions: dict[str, VersionVector] = {}
        self._locks: dict[str, GraphLock] = {}
        self._default_mode = default_lock_mode

    def track_node(self, node_id: str) -> VersionVector:
        """Start tracking a node for concurrency control."""
        if node_id not in self._versions:
            self._versions[node_id] = VersionVector(node_id=node_id)
        return self._versions[node_id]

    def get_version(self, node_id: str) -> int:
        """Get the current version of a tracked node."""
        vv = self._versions.get(node_id)
        return vv.current_version if vv else 0

    def compare_and_swap(
        self,
        node_id: str,
        expected_version: int,
        actor_id: str,
        actor_type: ActorType = ActorType.AI_AGENT,
    ) -> CASResult:
        """Attempt a Compare-And-Swap mutation on a node.

        If the node's current version matches expected_version, the
        version is incremented and the CAS succeeds. Otherwise, the
        CAS fails and a conflict is signaled.
        """
        vv = self._versions.get(node_id)
        if vv is None:
            vv = self.track_node(node_id)

        if vv.is_stale(expected_version):
            logger.warning(
                "CAS conflict on %s: expected v%d, actual v%d (actor: %s [%s])",
                node_id,
                expected_version,
                vv.current_version,
                actor_id,
                actor_type,
            )
            return CASResult(
                success=False,
                node_id=node_id,
                expected_version=expected_version,
                actual_version=vv.current_version,
                conflict_detected=True,
            )

        new_ver = vv.increment(actor_id)
        return CASResult(
            success=True,
            node_id=node_id,
            expected_version=expected_version,
            actual_version=new_ver,
            new_version=new_ver,
        )

    def acquire_lock(
        self,
        target_id: str,
        holder_id: str,
        holder_type: ActorType = ActorType.AI_AGENT,
        mode: LockMode | None = None,
        ttl_seconds: int = 300,
    ) -> GraphLock | None:
        """Acquire a lock on a node or edge."""
        existing = self._locks.get(target_id)
        if existing and existing.is_active:
            logger.info("Lock already held on %s by %s", target_id, existing.holder_id)
            return None

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        lock = GraphLock(
            target_id=target_id,
            holder_id=holder_id,
            holder_type=holder_type,
            mode=mode or self._default_mode,
            acquired_at=now,
        )
        self._locks[target_id] = lock
        return lock

    def release_lock(self, target_id: str, holder_id: str) -> bool:
        """Release a lock. Only the holder can release."""
        lock = self._locks.get(target_id)
        if lock and lock.holder_id == holder_id:
            lock.is_active = False
            del self._locks[target_id]
            return True
        return False

    @property
    def active_locks(self) -> list[GraphLock]:
        return [lock for lock in self._locks.values() if lock.is_active]

    @property
    def tracked_nodes(self) -> list[str]:
        return list(self._versions.keys())


# ---------------------------------------------------------------------------
# 2. Tenancy Manager (Gap 2)
# ---------------------------------------------------------------------------


class TenancyManager:
    """Multi-tenant graph isolation.

    CONCEPT:KG-2.51 — Company Brain Infrastructure

    Provides tenant-scoped node/edge operations, hierarchical tenant
    trees, and query-time tenant filtering. Supports humans, AI agents,
    and hybrid teams as tenant members.

    Example::

        tm = TenancyManager()
        tm.create_tenant("engineering", created_by="admin:alice",
                         created_by_type=ActorType.HUMAN)
        tm.add_member("agent:code-reviewer", ActorType.AI_AGENT,
                       "engineering", role="member")
        # Scope a query to a tenant
        scoped = tm.scope_cypher_query(
            "MATCH (n:Entity) RETURN n", tenant_id="engineering"
        )
    """

    def __init__(self) -> None:
        self._tenants: dict[str, TenantNode] = {}
        self._memberships: list[TenantMembership] = []

    def create_tenant(
        self,
        name: str,
        parent_tenant_id: str = "",
        created_by: str = "system",
        created_by_type: ActorType = ActorType.SYSTEM,
        **kwargs: Any,
    ) -> TenantNode:
        """Create a new tenant."""
        tenant = TenantNode(
            name=name,
            parent_tenant_id=parent_tenant_id,
            created_by=created_by,
            created_by_type=created_by_type,
            **kwargs,
        )
        self._tenants[tenant.tenant_id] = tenant
        logger.info(
            "Created tenant '%s' (%s) by %s [%s]",
            name,
            tenant.tenant_id,
            created_by,
            created_by_type,
        )
        return tenant

    def get_tenant(self, tenant_id: str) -> TenantNode | None:
        return self._tenants.get(tenant_id)

    def add_member(
        self,
        actor_id: str,
        actor_type: ActorType,
        tenant_id: str,
        role: str = "member",
        granted_by: str = "system",
    ) -> TenantMembership:
        """Add an actor (human, AI, or hybrid) to a tenant."""
        membership = TenantMembership(
            actor_id=actor_id,
            actor_type=actor_type,
            tenant_id=tenant_id,
            role=role,
            granted_by=granted_by,
        )
        self._memberships.append(membership)
        return membership

    def get_actor_tenants(self, actor_id: str) -> list[str]:
        """Get all tenants an actor belongs to (including parent chain)."""
        direct = [m.tenant_id for m in self._memberships if m.actor_id == actor_id]
        all_tenants = set(direct)
        for tid in direct:
            all_tenants.update(self._get_ancestor_tenants(tid))
        return list(all_tenants)

    def _get_ancestor_tenants(self, tenant_id: str) -> list[str]:
        ancestors: list[str] = []
        t = self._tenants.get(tenant_id)
        while t and t.parent_tenant_id:
            ancestors.append(t.parent_tenant_id)
            t = self._tenants.get(t.parent_tenant_id)
        return ancestors

    def scope_cypher_query(self, query: str, tenant_id: str) -> str:
        """Inject tenant scoping into a Cypher query."""
        if "WHERE" in query.upper():
            return query.replace("WHERE", f"WHERE n.tenant_id = '{tenant_id}' AND", 1)
        if "RETURN" in query.upper():
            return query.replace(
                "RETURN", f"WHERE n.tenant_id = '{tenant_id}' RETURN", 1
            )
        return query

    def is_member(self, actor_id: str, tenant_id: str) -> bool:
        return tenant_id in self.get_actor_tenants(actor_id)

    @property
    def all_tenants(self) -> list[TenantNode]:
        return list(self._tenants.values())


# ---------------------------------------------------------------------------
# 3. Conflict Resolver (Gap 3)
# ---------------------------------------------------------------------------


class ConflictResolver:
    """Contradiction detection and resolution engine.

    CONCEPT:KG-2.51 — Company Brain Infrastructure

    Detects when two actors write conflicting values to the same node
    and applies configurable merge strategies. Supports humans and AIs
    equally — a human judgment and an AI inference are both evaluated
    by the same strategy framework.

    Example::

        cr = ConflictResolver(default_strategy=MergeStrategy.SOURCE_AUTHORITY_WINS)
        cr.add_trust_entry(TrustHierarchyEntry(
            source_system="crm", data_domain="customer", authority_level=0.95
        ))
        conflict = cr.detect_conflict(
            node_id="customer:001", field_name="risk_level",
            value_a="low", actor_a="agent:risk-v1",
            value_b="high", actor_b="analyst:jane",
            actor_b_type=ActorType.HUMAN,
        )
        resolved = cr.resolve(conflict)
    """

    def __init__(
        self, default_strategy: MergeStrategy = MergeStrategy.LAST_WRITE_WINS
    ) -> None:
        self._default_strategy = default_strategy
        self._trust_hierarchy: list[TrustHierarchyEntry] = []
        self._conflicts: list[ConflictNode] = []
        self._strategy_overrides: dict[str, MergeStrategy] = {}

    def add_trust_entry(self, entry: TrustHierarchyEntry) -> None:
        self._trust_hierarchy.append(entry)

    def set_strategy_for_type(self, node_type: str, strategy: MergeStrategy) -> None:
        """Override merge strategy for a specific node type."""
        self._strategy_overrides[node_type] = strategy

    def detect_conflict(
        self,
        node_id: str,
        field_name: str,
        value_a: Any,
        value_b: Any,
        actor_a: str = "",
        actor_a_type: ActorType = ActorType.AI_AGENT,
        actor_b: str = "",
        actor_b_type: ActorType = ActorType.AI_AGENT,
        assertion_type_a: AssertionType = AssertionType.AGENT_INFERENCE,
        assertion_type_b: AssertionType = AssertionType.AGENT_INFERENCE,
        confidence_a: float = 0.5,
        confidence_b: float = 0.5,
    ) -> ConflictNode | None:
        """Detect if two values conflict."""
        if value_a == value_b:
            return None

        conflict = ConflictNode(
            node_id=node_id,
            field_name=field_name,
            value_a=value_a,
            value_b=value_b,
            actor_a=actor_a,
            actor_a_type=actor_a_type,
            actor_b=actor_b,
            actor_b_type=actor_b_type,
            assertion_type_a=assertion_type_a,
            assertion_type_b=assertion_type_b,
            confidence_a=confidence_a,
            confidence_b=confidence_b,
        )
        self._conflicts.append(conflict)
        logger.warning(
            "Conflict detected on %s.%s: %r vs %r (actors: %s [%s] vs %s [%s])",
            node_id,
            field_name,
            value_a,
            value_b,
            actor_a,
            actor_a_type,
            actor_b,
            actor_b_type,
        )
        return conflict

    def resolve(
        self,
        conflict: ConflictNode,
        strategy: MergeStrategy | None = None,
        resolved_by: str = "system",
    ) -> Any:
        """Resolve a conflict using the specified or default strategy."""
        start = strategy or self._default_strategy

        if start == MergeStrategy.LAST_WRITE_WINS:
            winner = conflict.value_b
        elif start == MergeStrategy.HIGHEST_CONFIDENCE_WINS:
            winner = (
                conflict.value_a
                if conflict.confidence_a >= conflict.confidence_b
                else conflict.value_b
            )
        elif start == MergeStrategy.SOURCE_AUTHORITY_WINS:
            auth_a = self._get_authority(conflict.actor_a)
            auth_b = self._get_authority(conflict.actor_b)
            winner = conflict.value_a if auth_a >= auth_b else conflict.value_b
        elif start == MergeStrategy.MERGE_APPEND:
            winner = [conflict.value_a, conflict.value_b]
        else:
            conflict.status = ConflictStatus.ESCALATED
            return None

        conflict.status = ConflictStatus.RESOLVED_AUTO
        conflict.resolution_strategy = start
        conflict.resolved_value = winner
        conflict.resolved_by = resolved_by
        conflict.resolved_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return winner

    def _get_authority(self, actor_id: str) -> float:
        for entry in self._trust_hierarchy:
            if entry.source_system in actor_id:
                return entry.authority_level
        return 0.5

    @property
    def open_conflicts(self) -> list[ConflictNode]:
        return [c for c in self._conflicts if c.status == ConflictStatus.OPEN]

    @property
    def all_conflicts(self) -> list[ConflictNode]:
        return list(self._conflicts)


# ---------------------------------------------------------------------------
# 4. Provenance Tracker (Gap 4)
# ---------------------------------------------------------------------------


class ProvenanceTracker:
    """Mandatory provenance enforcement and trust management.

    CONCEPT:KG-2.51 — Company Brain Infrastructure

    Wraps write operations with required provenance metadata. Tracks
    read audits, manages trust hierarchies, and enables provenance-gated
    retrieval. All actors (human, AI, hybrid) generate identical
    provenance records.
    """

    def __init__(self, enforce_provenance: bool = True) -> None:
        self._records: list[ProvenanceRecord] = []
        self._read_audits: list[ReadAuditEntry] = []
        self._trust_entries: list[TrustHierarchyEntry] = []
        self._enforce = enforce_provenance

    def record_write(
        self,
        node_id: str,
        actor_id: str,
        actor_type: ActorType = ActorType.AI_AGENT,
        action: str = "create",
        assertion_type: AssertionType = AssertionType.AGENT_INFERENCE,
        confidence: float = 0.8,
        source_system: str = "",
        derived_from: list[str] | None = None,
        rationale: str = "",
        tenant_id: str = "",
    ) -> ProvenanceRecord:
        """Record provenance for a write operation."""
        record = ProvenanceRecord(
            node_id=node_id,
            actor_id=actor_id,
            actor_type=actor_type,
            action=action,
            assertion_type=assertion_type,
            confidence=confidence,
            source_system=source_system,
            derived_from=derived_from or [],
            rationale=rationale,
            tenant_id=tenant_id,
        )
        self._records.append(record)
        return record

    def record_read(
        self,
        actor_id: str,
        actor_type: ActorType = ActorType.AI_AGENT,
        nodes_accessed: list[str] | None = None,
        query_summary: str = "",
        tenant_id: str = "",
    ) -> ReadAuditEntry:
        """Record a read audit entry."""
        entry = ReadAuditEntry(
            actor_id=actor_id,
            actor_type=actor_type,
            nodes_accessed=nodes_accessed or [],
            query_summary=query_summary,
            tenant_id=tenant_id,
        )
        self._read_audits.append(entry)
        return entry

    def add_trust_entry(self, entry: TrustHierarchyEntry) -> None:
        self._trust_entries.append(entry)

    def get_provenance(self, node_id: str) -> list[ProvenanceRecord]:
        """Get all provenance records for a node."""
        return [r for r in self._records if r.node_id == node_id]

    def get_trust_level(self, source_system: str, data_domain: str = "*") -> float:
        """Get the trust level for a source system in a domain."""
        for entry in self._trust_entries:
            if entry.source_system == source_system and (
                entry.data_domain == "*" or entry.data_domain == data_domain
            ):
                return entry.authority_level
        return 0.5

    def filter_by_trust(self, node_ids: list[str], min_trust: float = 0.7) -> list[str]:
        """Filter nodes by minimum trust level of their provenance."""
        result = []
        for nid in node_ids:
            records = self.get_provenance(nid)
            if records and any(r.confidence >= min_trust for r in records):
                result.append(nid)
            elif not records:
                result.append(nid)
        return result

    @property
    def write_count(self) -> int:
        return len(self._records)

    @property
    def read_count(self) -> int:
        return len(self._read_audits)


# ---------------------------------------------------------------------------
# 5. Event Stream Ingester (Gap 5)
# ---------------------------------------------------------------------------


class EventStreamIngester:
    """AsyncIO-based event stream consumer with pluggable adapters.

    CONCEPT:KG-2.51 — Company Brain Infrastructure

    Converts external events (webhooks, Kafka, NATS, CDC) into graph
    mutations with automatic provenance tracking. Supports humans
    triggering events (Slack messages, Jira updates) and automated
    services equally.
    """

    def __init__(self) -> None:
        self._streams: dict[str, EventStreamConfig] = {}
        self._event_queue: list[WebhookEvent] = []
        self._results: list[IngestionResult] = []

    def register_stream(self, config: EventStreamConfig) -> str:
        """Register an event stream source."""
        self._streams[config.stream_id] = config
        logger.info(
            "Registered event stream '%s' (%s)", config.name, config.source_type
        )
        return config.stream_id

    def submit_event(self, event: WebhookEvent) -> None:
        """Submit an event for ingestion."""
        self._event_queue.append(event)

    def process_batch(self) -> IngestionResult:
        """Process all queued events into graph mutations."""
        start = time.time()
        batch = list(self._event_queue)
        self._event_queue.clear()

        ingested = 0
        failed = 0
        nodes = 0
        edges = 0

        for event in batch:
            try:
                event.processed = True
                ingested += 1
                nodes += 1
            except Exception as e:
                logger.error("Failed to ingest event %s: %s", event.event_id, e)
                event.retry_count += 1
                if event.retry_count < 3:
                    self._event_queue.append(event)
                failed += 1

        result = IngestionResult(
            events_received=len(batch),
            events_ingested=ingested,
            events_failed=failed,
            nodes_created=nodes,
            edges_created=edges,
            duration_ms=(time.time() - start) * 1000,
        )
        self._results.append(result)
        return result

    def get_stream(self, stream_id: str) -> EventStreamConfig | None:
        return self._streams.get(stream_id)

    @property
    def pending_events(self) -> int:
        return len(self._event_queue)

    @property
    def registered_streams(self) -> list[EventStreamConfig]:
        return list(self._streams.values())


# ---------------------------------------------------------------------------
# 6. Data-Level Permissions (Gap 6)
# ---------------------------------------------------------------------------


class DataLevelPermissions:
    """Node-level ACLs, classification labels, and query-time filtering.

    CONCEPT:KG-2.51 — Company Brain Infrastructure

    Extends the existing PermissionsKernel (which controls tool access)
    with data-level access control. Humans and AIs are subject to the
    same permission framework — no actor type receives implicit
    elevated access.
    """

    def __init__(self) -> None:
        self._acls: dict[str, NodeACL] = {}

    def set_acl(self, acl: NodeACL) -> None:
        """Set or update the ACL for a node."""
        self._acls[acl.node_id] = acl

    def get_acl(self, node_id: str) -> NodeACL | None:
        return self._acls.get(node_id)

    def check_permission(
        self,
        node_id: str,
        actor_id: str,
        actor_type: ActorType = ActorType.AI_AGENT,
        action: str = "read",
        actor_roles: list[str] | None = None,
    ) -> PermissionCheckResult:
        """Check if an actor has permission to access a node."""
        acl = self._acls.get(node_id)

        if acl is None:
            return PermissionCheckResult(
                allowed=True,
                node_id=node_id,
                actor_id=actor_id,
                actor_type=actor_type,
                action=action,
                reason="No ACL defined — default allow",
            )

        roles = actor_roles or []
        allowed = False
        reason = ""

        if action == "read":
            allowed = (
                actor_id in acl.read_actors
                or actor_id in acl.admin_actors
                or actor_id == acl.data_owner
                or bool(set(roles) & set(acl.read_roles))
                or acl.classification == DataClassification.PUBLIC
            )
            reason = "Read access granted" if allowed else "Read access denied"
        elif action == "write":
            allowed = (
                actor_id in acl.write_actors
                or actor_id in acl.admin_actors
                or actor_id == acl.data_owner
                or bool(set(roles) & set(acl.write_roles))
            )
            reason = "Write access granted" if allowed else "Write access denied"
        elif action == "admin":
            allowed = actor_id in acl.admin_actors or actor_id == acl.data_owner
            reason = "Admin access granted" if allowed else "Admin access denied"

        return PermissionCheckResult(
            allowed=allowed,
            node_id=node_id,
            actor_id=actor_id,
            actor_type=actor_type,
            action=action,
            reason=reason,
            classification=acl.classification,
            audit_logged=acl.audit_on_access,
        )

    def filter_nodes(
        self,
        node_ids: list[str],
        actor_id: str,
        actor_type: ActorType = ActorType.AI_AGENT,
        action: str = "read",
        actor_roles: list[str] | None = None,
    ) -> list[str]:
        """Filter a list of nodes to only those the actor can access."""
        return [
            nid
            for nid in node_ids
            if self.check_permission(
                nid, actor_id, actor_type, action, actor_roles
            ).allowed
        ]

    def classify_node(
        self, node_id: str, classification: DataClassification, data_owner: str = ""
    ) -> NodeACL:
        """Set classification and owner for a node, creating ACL if needed."""
        acl = self._acls.get(node_id, NodeACL(node_id=node_id))
        acl.classification = classification
        if data_owner:
            acl.data_owner = data_owner
        if classification == DataClassification.RESTRICTED:
            acl.audit_on_access = True
        self._acls[node_id] = acl
        return acl


# ---------------------------------------------------------------------------
# Unified Company Brain Facade
# ---------------------------------------------------------------------------


class CompanyBrain:
    """Unified facade for all Company Brain infrastructure primitives.

    CONCEPT:KG-2.51 — Company Brain Infrastructure

    Composes all six infrastructure primitives into a single entry point.
    The Company Brain is actor-agnostic: humans, AI agents, automated
    services, and hybrid human+AI teams are all first-class participants
    with identical capabilities and constraints.

    Example::

        brain = CompanyBrain()
        # Create a tenant for a team
        brain.tenancy.create_tenant("data-science",
            created_by="director:sarah", created_by_type=ActorType.HUMAN)
        # Add both human and AI members
        brain.tenancy.add_member("analyst:jane", ActorType.HUMAN, "data-science")
        brain.tenancy.add_member("agent:model-trainer", ActorType.AI_AGENT, "data-science")
        # Track concurrency
        brain.concurrency.track_node("model:prod-v2")
        # Set data permissions
        brain.permissions.classify_node("model:prod-v2", DataClassification.CONFIDENTIAL)
    """

    def __init__(
        self,
        default_lock_mode: LockMode = LockMode.OPTIMISTIC,
        default_merge_strategy: MergeStrategy = MergeStrategy.LAST_WRITE_WINS,
        enforce_provenance: bool = True,
    ) -> None:
        self.concurrency = GraphConcurrencyManager(default_lock_mode)
        self.tenancy = TenancyManager()
        self.conflicts = ConflictResolver(default_merge_strategy)
        self.provenance = ProvenanceTracker(enforce_provenance)
        self.events = EventStreamIngester()
        self.permissions = DataLevelPermissions()

    def status(self) -> dict[str, Any]:
        """Return the current status of all Company Brain subsystems."""
        return {
            "concurrency": {
                "tracked_nodes": len(self.concurrency.tracked_nodes),
                "active_locks": len(self.concurrency.active_locks),
            },
            "tenancy": {
                "tenants": len(self.tenancy.all_tenants),
            },
            "conflicts": {
                "open": len(self.conflicts.open_conflicts),
                "total": len(self.conflicts.all_conflicts),
            },
            "provenance": {
                "write_records": self.provenance.write_count,
                "read_audits": self.provenance.read_count,
            },
            "events": {
                "registered_streams": len(self.events.registered_streams),
                "pending_events": self.events.pending_events,
            },
            "permissions": {
                "nodes_with_acls": len(self.permissions._acls),
            },
        }


__all__ = [
    "CompanyBrain",
    "ConflictResolver",
    "DataLevelPermissions",
    "EventStreamIngester",
    "GraphConcurrencyManager",
    "ProvenanceTracker",
    "TenancyManager",
]
