from __future__ import annotations

"""Transactional AgentBus inbox-to-WorkItem materialization.

The broker message is not the orchestration authority.  Before it may be
acknowledged, this module durably creates four tenant-qualified records in one
engine transaction:

* ``BusInbox`` — the deduplicated, privacy-sanitized delivery;
* ``WorkItem`` — the sole writable work-state authority;
* ``BusDeliveryOutcome`` — the canonical audit/outcome record;
* ``MutationOutbox`` — projection/CDC work that can be retried independently.

The identifiers are hashes over a random message group and tenant scope; raw
user names, local paths, endpoints, credentials, or broker receipt handles are
never persisted. Missing identity or native transaction authority fails closed.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from agent_utilities.messaging.bus_privacy import bus_reference
from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

__all__ = [
    "BusInboxCommit",
    "BusOutboxCommit",
    "commit_message_outbox",
    "commit_message_to_work_item",
    "mark_message_outbox_delivered",
    "mark_message_outbox_published",
]


@dataclass(frozen=True)
class BusInboxCommit:
    inbox_id: str
    work_item_id: str
    outcome_id: str
    outbox_id: str
    replay: bool = False


@dataclass(frozen=True)
class BusOutboxCommit:
    outbox_id: str
    replay: bool = False


def _opaque_id(*parts: str) -> str:
    framed = "\x00".join(str(part or "") for part in parts).encode("utf-8")
    return hashlib.sha256(b"agent-utilities:bus-inbox:v1\x00" + framed).hexdigest()


def _native_client(engine: Any) -> Any | None:
    candidates = (
        engine,
        getattr(engine, "_client", None),
        getattr(engine, "_control", None),
        getattr(engine, "backend", None),
        getattr(getattr(engine, "backend", None), "graph", None),
        getattr(engine, "graph_compute", None),
        getattr(getattr(engine, "graph_compute", None), "_client", None),
    )
    for candidate in candidates:
        if candidate is not None and getattr(candidate, "txn", None) is not None:
            return candidate
    return None


def _session_graph_and_tenant(tenant: str) -> tuple[str, str]:
    from agent_utilities.knowledge_graph.core.session import current_session
    from agent_utilities.knowledge_graph.core.shard_topology import tenant_graph_name

    session = current_session()
    if session is None or not session.actor.authenticated:
        raise PermissionError("bus inbox requires a verified GraphSession")
    if not session.tenant:
        raise PermissionError("bus inbox requires a tenant-qualified GraphSession")
    if tenant and session.tenant != tenant:
        raise PermissionError("bus tenant differs from the verified GraphSession")
    return session.graph or tenant_graph_name(session.tenant), session.tenant


def _properties(
    message: dict[str, Any], *, tenant: str, recipient: str, now: float
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], BusInboxCommit]:
    from agent_utilities.knowledge_graph.core.engine_tasks import _coerce_prio_bucket
    from agent_utilities.models.knowledge_graph import WorkItemNode

    group = str(message.get("msg_group") or message.get("id") or "")
    if not group:
        raise ValueError("bus message has no stable identity")
    tenant_ref = bus_reference("tenant", tenant or "default")
    recipient_ref = bus_reference("agent", recipient, tenant=tenant_ref)
    digest = _opaque_id(tenant_ref, recipient_ref, group)
    inbox_id = f"businbox:{digest}"
    work_item_id = f"workitem:bus:{digest}"
    outcome_id = f"busoutcome:{digest}"
    outbox_id = f"mutationoutbox:bus:{digest}"

    privacy = PersistencePrivacyGuard()
    clean_payload, payload_report = privacy.sanitize_text(str(message.get("payload") or ""))
    raw_meta = message.get("meta") or {}
    if isinstance(raw_meta, str):
        try:
            raw_meta = json.loads(raw_meta)
        except (TypeError, ValueError):
            raw_meta = {}
    clean_meta, meta_report = privacy.sanitize(raw_meta)
    if not isinstance(clean_meta, dict):
        clean_meta = {}
    privacy_types = sorted(
        set(payload_report.detected_types) | set(meta_report.detected_types)
    )
    sender_ref = bus_reference(
        "agent", str(message.get("sender") or ""), tenant=tenant_ref
    )
    topic_ref = (
        bus_reference("topic", str(message.get("topic") or ""), tenant=tenant_ref)
        if message.get("topic")
        else ""
    )
    payload_digest = hashlib.sha256(clean_payload.encode("utf-8")).hexdigest()
    correlation_id = bus_reference(
        "correlation",
        str(clean_meta.get("correlation_id") or ""),
        tenant=tenant_ref,
    )
    audit_sink = bool(message.get("_audit_sink"))

    inbox = {
        "id": inbox_id,
        "node_type": "BusInbox",
        "tenant": tenant_ref,
        "message_ref": digest,
        "group_ref": bus_reference("message_group", group, tenant=tenant_ref),
        "work_item_ref": work_item_id,
        "sender_ref": sender_ref,
        "recipient_ref": recipient_ref,
        "topic_ref": topic_ref,
        "payload": clean_payload,
        "payload_sha256": payload_digest,
        "metadata": json.dumps(clean_meta, sort_keys=True, separators=(",", ":")),
        "privacy_redactions": payload_report.redactions + meta_report.redactions,
        "privacy_types": privacy_types,
        "status": "committed",
        "created_at": float(message.get("created") or now),
        "committed_at": now,
    }
    work = WorkItemNode(
        id=work_item_id,
        name="WorkItem: bus_message",
        # WorkItem's tenant is an engine authorization key, not user content;
        # it must match the verified GraphSession for native claim fencing.
        tenant=tenant,
        kind="bus_message",
        status="succeeded" if audit_sink else "ready",
        payload_ref=inbox_id,
        idempotency_key=digest,
        depends_on=[],
        dep_count=0,
        prio_bucket=_coerce_prio_bucket(clean_meta.get("priority", 2)),
        deadline_unix=clean_meta.get("deadline_unix"),
        budget=clean_meta.get("budget"),
        resource_class=str(clean_meta.get("resource_class") or "default"),
        fairness_group=bus_reference(
            "fairness_group",
            str(clean_meta.get("fairness_group") or tenant_ref or "default"),
            tenant=tenant_ref,
        ),
        max_attempts=max(1, int(clean_meta.get("max_attempts") or 3)),
        backoff_base_s=float(clean_meta.get("backoff_base_s") or 30.0),
        correlation_id=correlation_id,
        created_at=now,
        updated_at=now,
        submitted_at=now,
        completed_at=now if audit_sink else None,
        result_ref=outcome_id if audit_sink else None,
    ).model_dump(exclude={"id", "type"})
    work["node_type"] = "WorkItem"
    outcome = {
        "id": outcome_id,
        "node_type": "BusDeliveryOutcome",
        "tenant": tenant_ref,
        "message_ref": digest,
        "inbox_ref": inbox_id,
        "work_item_ref": work_item_id,
        "outcome": "topic_audit_committed" if audit_sink else "inbox_committed",
        "correlation_id": correlation_id,
        "created_at": now,
    }
    outbox = {
        "id": outbox_id,
        "node_type": "MutationOutbox",
        "tenant": tenant_ref,
        "aggregate_type": "BusInbox",
        "aggregate_id": inbox_id,
        "event_type": "BusInboxCommitted",
        "payload_ref": outcome_id,
        "status": "pending",
        "attempt": 0,
        "created_at": now,
    }
    return inbox, work, outcome, outbox, BusInboxCommit(
        inbox_id=inbox_id,
        work_item_id=work_item_id,
        outcome_id=outcome_id,
        outbox_id=outbox_id,
    )


def _already_committed(engine: Any, work_item_id: str) -> bool:
    from agent_utilities.orchestration.work_item import get_work_item

    try:
        return get_work_item(engine, work_item_id) is not None
    except Exception as exc:
        raise RuntimeError(
            "bus inbox idempotency read failed "
            f"(error_type={type(exc).__name__})"
        ) from None


def _message_outbox_properties(
    message: dict[str, Any], *, tenant: str, now: float, status: str
) -> tuple[str, dict[str, Any]]:
    group = str(message.get("msg_group") or message.get("id") or "")
    if not group:
        raise ValueError("bus message has no stable identity")
    tenant_ref = bus_reference("tenant", tenant)
    digest = _opaque_id(
        "outbox",
        tenant_ref,
        group,
        str(message.get("recipient") or ""),
        str(message.get("topic") or ""),
    )
    raw_meta = message.get("meta") or {}
    if isinstance(raw_meta, str):
        try:
            raw_meta = json.loads(raw_meta)
        except (TypeError, ValueError):
            raw_meta = {}
    privacy = PersistencePrivacyGuard()
    clean_payload, payload_report = privacy.sanitize_text(
        str(message.get("payload") or "")
    )
    clean_meta, meta_report = privacy.sanitize(raw_meta)
    if not isinstance(clean_meta, dict):
        clean_meta = {}
    props = {
        "id": f"busoutbox:{digest}",
        "node_type": "BusOutbox",
        "tenant": tenant_ref,
        "group_ref": bus_reference("message_group", group, tenant=tenant_ref),
        "sender_ref": bus_reference(
            "agent", str(message.get("sender") or ""), tenant=tenant_ref
        ),
        "recipient_ref": bus_reference(
            "agent", str(message.get("recipient") or ""), tenant=tenant_ref
        ),
        "topic_ref": bus_reference(
            "topic", str(message.get("topic") or ""), tenant=tenant_ref
        ),
        "payload": clean_payload,
        "metadata": json.dumps(clean_meta, sort_keys=True, separators=(",", ":")),
        "privacy_redactions": payload_report.redactions + meta_report.redactions,
        "status": status,
        "created_at": float(message.get("created") or now),
        "updated_at": now,
    }
    if status == "published":
        props["published_at"] = now
    elif status == "delivered":
        props["delivered_at"] = now
    return str(props["id"]), props


def _outbox_status(engine: Any, outbox_id: str) -> str:
    try:
        rows = engine.query_cypher(
            "MATCH (o:BusOutbox {id: $id}) RETURN o.status AS status",
            {"id": outbox_id},
        )
        return str(rows[0].get("status") or "") if rows else ""
    except Exception as exc:
        raise RuntimeError(
            "bus outbox authority read failed "
            f"(error_type={type(exc).__name__})"
        ) from None


def _outbox_exists(engine: Any, outbox_id: str) -> bool:
    return bool(_outbox_status(engine, outbox_id))


def _commit_native_node(engine: Any, graph: str, node_id: str, props: dict[str, Any]) -> None:
    client = _native_client(engine)
    if client is None:
        raise RuntimeError("bus outbox requires the engine-native transaction surface")
    txn = client.txn.begin(graph=graph)
    client.txn.add_node(txn, node_id, props)
    if not client.txn.commit(txn):
        expected = str(props.get("status") or "")
        if _outbox_status(engine, node_id) != expected:
            raise RuntimeError("bus outbox transaction did not commit")


def commit_message_outbox(
    engine: Any,
    message: dict[str, Any],
    *,
    tenant: str,
    now: float | None = None,
) -> BusOutboxCommit:
    """Persist a tenant-qualified send intent before broker publication."""
    graph, effective_tenant = _session_graph_and_tenant(tenant)
    timestamp = time.time() if now is None else float(now)
    outbox_id, props = _message_outbox_properties(
        message, tenant=effective_tenant, now=timestamp, status="pending"
    )
    if _outbox_exists(engine, outbox_id):
        return BusOutboxCommit(outbox_id=outbox_id, replay=True)
    _commit_native_node(engine, graph, outbox_id, props)
    return BusOutboxCommit(outbox_id=outbox_id)


def mark_message_outbox_published(
    engine: Any,
    message: dict[str, Any],
    *,
    tenant: str,
    now: float | None = None,
) -> BusOutboxCommit:
    """Mark a previously persisted send intent published in a native transaction."""
    graph, effective_tenant = _session_graph_and_tenant(tenant)
    timestamp = time.time() if now is None else float(now)
    outbox_id, props = _message_outbox_properties(
        message, tenant=effective_tenant, now=timestamp, status="published"
    )
    _commit_native_node(engine, graph, outbox_id, props)
    return BusOutboxCommit(outbox_id=outbox_id)


def mark_message_outbox_delivered(
    engine: Any,
    message: dict[str, Any],
    *,
    tenant: str,
    now: float | None = None,
) -> BusOutboxCommit:
    """Mark a send outbox delivered only after durable inbox commit and ack."""
    graph, effective_tenant = _session_graph_and_tenant(tenant)
    timestamp = time.time() if now is None else float(now)
    outbox_id, props = _message_outbox_properties(
        message, tenant=effective_tenant, now=timestamp, status="delivered"
    )
    _commit_native_node(engine, graph, outbox_id, props)
    return BusOutboxCommit(outbox_id=outbox_id)


def commit_message_to_work_item(
    engine: Any,
    message: dict[str, Any],
    *,
    tenant: str,
    recipient: str,
    now: float | None = None,
) -> BusInboxCommit:
    """Atomically materialize a broker delivery, returning stable record IDs."""

    graph, effective_tenant = _session_graph_and_tenant(tenant)
    timestamp = time.time() if now is None else float(now)
    inbox, work, outcome, outbox, result = _properties(
        message, tenant=effective_tenant, recipient=recipient, now=timestamp
    )
    if _already_committed(engine, result.work_item_id):
        return BusInboxCommit(**{**result.__dict__, "replay": True})

    client = _native_client(engine)
    if client is None:
        raise RuntimeError("bus delivery requires the engine-native transaction surface")
    txn = client.txn.begin(graph=graph)
    client.txn.add_node(txn, result.inbox_id, inbox)
    client.txn.add_node(txn, result.work_item_id, work)
    client.txn.add_node(txn, result.outcome_id, outcome)
    client.txn.add_node(txn, result.outbox_id, outbox)
    if not client.txn.commit(txn):
        if _already_committed(engine, result.work_item_id):
            return BusInboxCommit(**{**result.__dict__, "replay": True})
        raise RuntimeError("bus inbox transaction did not commit")
    return result
