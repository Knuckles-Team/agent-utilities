"""Durable state primitives for governed WebUI contact delivery."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agent_utilities.knowledge_graph.core.session import resolve_session
from agent_utilities.security.persistence_privacy import persistence_reference

_ACTOR_REFERENCE = re.compile(r"^[0-9a-f]{64}$")
_IDEMPOTENCY_KEY = re.compile(r"^contactreq_[0-9a-f]{32}$")
_DESTINATION = re.compile(
    r"^(?P<platform>[a-z0-9][a-z0-9_-]{0,31}):"
    r"(?P<channel>[^\x00-\x1f\x7f]{1,200})$"
)
_RECEIPT = re.compile(r"^contact_[A-Za-z0-9_-]{16,56}$")
_WORK_ITEM_ID = re.compile(r"^workitem:webui_contact:[0-9a-f]{64}$")
_RATE_LIMIT = 5
_RATE_WINDOW_SECONDS = 60
_CAS_RETRIES = 8


@dataclass(frozen=True, slots=True)
class ContactDeliveryContext:
    tenant: str
    tenant_reference: str
    actor_reference: str
    client_key: str
    destination: tuple[str, str]
    request_digest: str
    item_id: str


def digest(*values: str) -> str:
    framed = b"\x00".join(value.encode("utf-8") for value in values)
    return hashlib.sha256(framed).hexdigest()


def destination_parts(value: str) -> tuple[str, str] | None:
    """Return a valid fixed ``platform:channel`` pair."""
    match = _DESTINATION.fullmatch(value)
    if match is None:
        return None
    channel = match.group("channel")
    if len(channel.encode("utf-8")) > 200:
        return None
    return match.group("platform"), channel


def prepare_contact(
    request: Any, *, fixed_destination: str
) -> ContactDeliveryContext | None:
    """Validate server-owned fields and derive non-reversible graph values."""
    request_destination = request.destination
    destination = destination_parts(request_destination)
    actor = request.actor_reference
    client_key = request.idempotency_key
    valid = all(
        (
            request.retention_days == 0,
            destination is not None and request_destination == fixed_destination,
            _ACTOR_REFERENCE.fullmatch(actor),
            _IDEMPOTENCY_KEY.fullmatch(client_key),
        )
    )
    if not valid or destination is None:
        return None

    session = resolve_session(required_scope="kg:write")
    tenant = str(session.tenant or "").strip()
    actor_id = str(getattr(session.actor, "actor_id", "") or "")
    expected_actor = hashlib.sha256(actor_id.encode("utf-8")).hexdigest()
    if not tenant or actor != expected_actor:
        return None
    tenant_ref = persistence_reference(
        "contact_tenant", tenant, namespace="webui-contact"
    )
    actor_ref = persistence_reference("contact_actor", actor, namespace=tenant_ref)
    submission = request.submission
    canonical = json.dumps(
        {
            "destination": request.destination,
            "email": submission.email,
            "message": submission.message,
            "name": submission.name,
            "retention_days": request.retention_days,
            "subject": submission.subject,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    request_digest = persistence_reference(
        "contact_request", canonical, namespace=f"webui-contact:{tenant}"
    )
    item_digest = digest("webui-contact-v1", tenant, actor, client_key)
    return ContactDeliveryContext(
        tenant=tenant,
        tenant_reference=tenant_ref,
        actor_reference=actor_ref,
        client_key=client_key,
        destination=destination,
        request_digest=request_digest,
        item_id=f"workitem:webui_contact:{item_digest}",
    )


def existing_receipt(row: dict[str, Any] | None, expected_digest: str) -> str | None:
    """Return only a matching, durable successful receipt."""
    metadata = row.get("metadata") if row else None
    digest_matches = (
        isinstance(metadata, dict) and metadata.get("request_digest") == expected_digest
    )
    receipt = row.get("result_ref") if row else None
    succeeded = bool(
        digest_matches
        and row
        and row.get("status") == "succeeded"
        and isinstance(receipt, str)
        and _RECEIPT.fullmatch(receipt)
    )
    return receipt if succeeded and isinstance(receipt, str) else None


def work_item_submission(context: ContactDeliveryContext) -> dict[str, Any]:
    """Build a WorkItem payload containing no form fields or raw routing."""
    idempotency_ref = persistence_reference(
        "contact_idempotency",
        context.client_key,
        namespace=context.actor_reference,
    )
    destination = ":".join(context.destination)
    destination_ref = persistence_reference(
        "contact_destination", destination, namespace=context.tenant_reference
    )
    return {
        "kind": "webui.contact.delivery",
        "queue": "webui_contact_delivery",
        "payload_ref": context.request_digest,
        "tenant": context.tenant,
        "resource_class": "network_io",
        "fairness_group": context.actor_reference,
        "max_attempts": 1,
        "idempotency_key": idempotency_ref,
        "description": "Governed agent-webui contact delivery",
        "created_by": context.actor_reference,
        "metadata": {
            "request_digest": context.request_digest,
            "destination_reference": destination_ref,
        },
        "work_item_id": context.item_id,
    }


def _attempt_ids(value: Any) -> list[str] | None:
    if (
        not isinstance(value, list)
        or len(value) > _RATE_LIMIT
        or not all(
            isinstance(item, str) and _WORK_ITEM_ID.fullmatch(item) for item in value
        )
        or len(set(value)) != len(value)
    ):
        return None
    return value


def _rate_snapshot(rows: Any) -> tuple[int, int, list[str]] | None:
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        return None
    row = rows[0]
    stored_window = row.get("window_start")
    count = row.get("count")
    attempt_ids = _attempt_ids(row.get("attempt_ids"))
    if type(stored_window) is not int or type(count) is not int or attempt_ids is None:
        return None
    if stored_window < 0 or not 1 <= count <= _RATE_LIMIT or count != len(attempt_ids):
        return None
    return stored_window, count, attempt_ids


def _next_rate_state(
    snapshot: tuple[int, int, list[str]],
    context: ContactDeliveryContext,
    window_start: int,
) -> tuple[bool, dict[str, Any] | None]:
    stored_window, count, attempt_ids = snapshot
    same_window = stored_window == window_start
    if same_window and context.item_id in attempt_ids:
        return True, None
    if same_window and count >= _RATE_LIMIT:
        return False, None
    return False, {
        "conditions": {
            "window_start": stored_window,
            "count": count,
            "attempt_ids": attempt_ids,
        },
        "updates": {
            "window_start": window_start,
            "count": count + 1 if same_window else 1,
            "attempt_ids": (
                [*attempt_ids, context.item_id] if same_window else [context.item_id]
            ),
        },
    }


def allow_shared_attempt(
    authority: Any, context: ContactDeliveryContext, *, now: float
) -> bool:
    """Consume one of five shared actor slots in a fixed minute window."""
    window_start = int(now // _RATE_WINDOW_SECONDS) * _RATE_WINDOW_SECONDS
    node_id = "contact_rate:" + digest(
        "webui-contact-rate-v1",
        context.tenant_reference,
        context.actor_reference,
    )
    properties = {
        "id": node_id,
        "node_type": "ContactRateLimit",
        "tenant_reference": context.tenant_reference,
        "actor_reference": context.actor_reference,
        "window_start": window_start,
        "count": 1,
        "attempt_ids": [context.item_id],
    }
    if authority.create_node_if_absent(node_id, properties=properties):
        return True

    query = (
        "MATCH (r:ContactRateLimit {id: $id}) "
        "RETURN r.window_start AS window_start, r.count AS count, "
        "r.attempt_ids AS attempt_ids LIMIT 2"
    )
    allowed = False
    for _ in range(_CAS_RETRIES):
        snapshot = _rate_snapshot(authority.query_cypher(query, {"id": node_id}))
        if snapshot is None:
            break
        allowed, transition = _next_rate_state(snapshot, context, window_start)
        if allowed or transition is None:
            break
        allowed = authority.compare_and_set_node_fields(
            node_id,
            transition["conditions"],
            transition["updates"],
        )
        if allowed:
            break
    return allowed


def contact_delivery_factory_kwargs(
    app_factory: Callable[..., Any],
    sync_runner: Callable[[Callable[[], Any]], Awaitable[Any]],
) -> dict[str, Any]:
    """Return the host port only for a contract-bearing agent-webui factory."""
    try:
        supports_contact = (
            "contact_delivery" in inspect.signature(app_factory).parameters
        )
    except (TypeError, ValueError):
        return {}
    if not supports_contact:
        return {}
    try:
        from agent_utilities.server.webui_contact_delivery import (
            build_webui_contact_delivery,
        )

        delivery = build_webui_contact_delivery(sync_runner)
    except ImportError:
        delivery = None
    return {"contact_delivery": delivery}
