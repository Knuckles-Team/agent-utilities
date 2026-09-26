"""AgentBus delivery adapter over epistemic-graph's partitioned stream log.

EG owns stream placement, offsets, and consumer cursors. AU owns the message
shape, verified tenant, and inbox transaction. A cursor advances only after
AU has committed every recipient's inbox and WorkItem.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from agent_utilities.messaging.bus_privacy import bus_reference, sanitize_bus_content

logger = logging.getLogger(__name__)
BUS_LOG_BACKENDS = ("engine",)
_MATERIALIZER_GROUP = "agent-bus-inbox-v1"


class BusLogUnavailable(RuntimeError):
    """The EG stream authority cannot deliver AgentBus messages."""


def current_bus_tenant() -> str:
    """Return the verified session tenant for a bus operation."""
    from agent_utilities.knowledge_graph.core.session import current_session

    session = current_session()
    if session is None or not session.actor.authenticated or not session.tenant:
        raise PermissionError("AgentBus requires a verified tenant GraphSession")
    session.actor.ensure_credential_current()
    return session.tenant


def _verified_tenant(tenant: str) -> str:
    verified = current_bus_tenant()
    if tenant != verified:
        raise PermissionError("bus tenant differs from the verified GraphSession")
    return verified


def bus_partition_key(tenant: str, target: str) -> str:
    tenant_ref = bus_reference("tenant", tenant)
    target_ref = bus_reference("route", target, tenant=tenant_ref)
    return f"{tenant_ref}:{target_ref}"


def encode_envelope(
    *,
    tenant: str,
    group: str,
    sender: str,
    recipient: str,
    topic: str,
    payload: str,
    meta_json: str,
    created: float,
) -> dict[str, Any]:
    tenant_ref = bus_reference("tenant", tenant)
    group = bus_reference("message_group", group, tenant=tenant_ref)
    sender = bus_reference("agent", sender, tenant=tenant_ref)
    recipient = bus_reference("agent", recipient, tenant=tenant_ref)
    topic = bus_reference("topic", topic, tenant=tenant_ref)
    try:
        metadata = json.loads(meta_json) if meta_json else {}
    except (TypeError, ValueError):
        metadata = {}
    payload, meta_json, _report = sanitize_bus_content(payload, metadata)
    return {
        "id": f"busmsg:{group}:{recipient or topic}",
        "msg_group": group,
        "sender": sender,
        "recipient": recipient,
        "topic": topic,
        "payload": payload,
        "meta": meta_json,
        "status": "sent",
        "created": created,
        "tenant": tenant_ref,
    }


def decode_envelope(raw: bytes) -> dict[str, Any] | None:
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("payload"), str):
        return None
    if not all(
        isinstance(obj.get(k), str) and obj[k]
        for k in ("tenant", "msg_group", "sender")
    ):
        return None
    if bool(obj.get("recipient")) == bool(obj.get("topic")):
        return None
    return obj


class EngineStreamBusLog:
    """Thin synchronous AU adapter; EG implements partition/cursor semantics."""

    name = "engine"

    def __init__(self, client: Any, *, partitions: int = 6) -> None:
        from epistemic_graph.partitioned_stream import SyncPartitionedStreamLog

        broker = getattr(client, "broker", None)
        if broker is None or not all(
            callable(getattr(broker, name, None))
            for name in (
                "stream_publish",
                "stream_read",
                "stream_commit_offset",
                "stream_committed_offset",
            )
        ):
            raise BusLogUnavailable(
                "connected EG client has no synchronous stream broker"
            )
        self._log = SyncPartitionedStreamLog(
            broker,
            namespace="agent_bus",
            partitions=partitions,
        )
        self._dlq = SyncPartitionedStreamLog(
            broker,
            namespace="agent_bus_dlq",
            partitions=partitions,
        )
        self.partitions = partitions

    def _publish(self, tenant: str, route: str, envelope: dict[str, Any]) -> bool:
        _verified_tenant(tenant)
        wire = json.dumps(
            envelope, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        self._log.append(
            tenant, route, wire, now_ms=int(float(envelope["created"]) * 1000)
        )
        return True

    def publish_direct(
        self,
        *,
        tenant: str,
        group: str,
        sender: str,
        to: str,
        payload: str,
        meta_json: str,
        created: float,
    ) -> bool:
        envelope = encode_envelope(
            tenant=tenant,
            group=group,
            sender=sender,
            recipient=to,
            topic="",
            payload=payload,
            meta_json=meta_json,
            created=created,
        )
        return self._publish(tenant, str(envelope["recipient"]), envelope)

    def publish_topic(
        self,
        *,
        tenant: str,
        group: str,
        sender: str,
        topic: str,
        payload: str,
        meta_json: str,
        created: float,
    ) -> bool:
        envelope = encode_envelope(
            tenant=tenant,
            group=group,
            sender=sender,
            recipient="",
            topic=topic,
            payload=payload,
            meta_json=meta_json,
            created=created,
        )
        return self._publish(tenant, str(envelope["topic"]), envelope)

    def receive(
        self,
        *,
        tenant: str,
        agent_id: str,
        topics: list[str],
        max_messages: int = 200,
    ) -> list[dict[str, Any]]:
        _verified_tenant(tenant)
        del agent_id, topics  # one bounded materializer scans fixed tenant partitions
        messages: list[dict[str, Any]] = []
        remaining = max(
            0, min(int(max_messages), self.partitions * self._log.max_batch)
        )
        for partition in range(self.partitions):
            if remaining == 0:
                break
            quota = min(
                self._log.max_batch,
                max(
                    1,
                    (remaining + self.partitions - partition - 1)
                    // (self.partitions - partition),
                ),
            )
            for record in self._log.read(
                tenant, partition, _MATERIALIZER_GROUP, limit=quota
            ):
                envelope = decode_envelope(record.payload)
                if envelope is None:
                    self._dead_letter(tenant, record, "decode_error")
                    self._log.commit(tenant, _MATERIALIZER_GROUP, record)
                    continue
                envelope["_receipt"] = {"tenant": tenant, "record": record}
                messages.append(envelope)
                remaining -= 1
        # Preserve each partition's offset order. Sorting globally by message
        # timestamp would permit committing a later offset before an earlier one.
        return messages

    def _dead_letter(self, tenant: str, record: Any, reason: str) -> None:
        diagnostic = json.dumps(
            {
                "stream": record.stream,
                "offset": record.offset,
                "sha256": hashlib.sha256(record.payload).hexdigest(),
                "bytes": len(record.payload),
                "reason": reason,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._dlq.append(tenant, "poison", diagnostic, now_ms=0)

    def _receipt(self, message: dict[str, Any]) -> tuple[str, Any] | None:
        receipt = message.get("_receipt")
        if not isinstance(receipt, dict):
            return None
        tenant = receipt.get("tenant")
        record = receipt.get("record")
        if not isinstance(tenant, str) or record is None:
            return None
        _verified_tenant(tenant)
        return tenant, record

    def ack(self, message: dict[str, Any]) -> bool:
        receipt = self._receipt(message)
        if receipt is None:
            return False
        tenant, record = receipt
        self._log.commit(tenant, _MATERIALIZER_GROUP, record)
        return True

    def nack(self, message: dict[str, Any], *, requeue: bool = True) -> bool:
        receipt = self._receipt(message)
        if receipt is None:
            return False
        if not requeue:
            tenant, record = receipt
            self._dead_letter(tenant, record, "rejected_envelope")
            self._log.commit(tenant, _MATERIALIZER_GROUP, record)
        return True

    def read_dlq(self, *, tenant: str, max_messages: int = 50) -> list[dict[str, Any]]:
        _verified_tenant(tenant)
        rows: list[dict[str, Any]] = []
        for partition in range(self.partitions):
            for record in self._dlq.read(
                tenant,
                partition,
                "agent-bus-dlq-inspection",
                limit=min(max_messages, self._dlq.max_batch),
            ):
                rows.append(json.loads(record.payload))
                if len(rows) >= max_messages:
                    return rows
        return rows

    def stats(self) -> dict[str, Any]:
        return {"backend": self.name, "partitions": self.partitions}


def resolve_bus_log_backend(
    *, engine: Any = None, config: Any = None
) -> EngineStreamBusLog:
    """Resolve the single EG stream writer; no alternate broker can be selected."""
    if config is None:
        from agent_utilities.core.config import config as _config

        config = _config
    selected = (
        str(getattr(config, "agent_bus_log_backend", "engine") or "engine")
        .strip()
        .lower()
    )
    if selected != "engine":
        raise BusLogUnavailable("AgentBus requires the epistemic-graph stream backend")
    partitions = int(getattr(config, "agent_bus_partitions", 6) or 6)
    if engine is not None and getattr(engine, "broker", None) is not None:
        return EngineStreamBusLog(engine, partitions=partitions)
    from agent_utilities.mcp.tools import engine_tools

    client = engine_tools._client_for("")
    return EngineStreamBusLog(client, partitions=partitions)
