"""Current durable FastA2A broker and storage on Epistemic Graph.

CONCEPT:AU-ECO.messaging.native-backend-abstraction

The adapter has one persistence plane and one authority plane: the process-owned
``SyncEpistemicGraphClient`` under a verified ``GraphSession``.  Task creation,
dispatch, execution leases, cancellation, context updates, and terminal writes
are all fenced.  No in-memory or external-database fallback exists.

Only tenant-qualified opaque identifiers and privacy-approved execution material
cross the persistence boundary.  Inline file bodies are rejected; file inputs
must be opaque governed content references.  Bounds are checked before schema
projection or copying.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar, cast

import anyio
from fasta2a.broker import Broker, TaskOperation
from fasta2a.pydantic_ai import AgentWorker
from fasta2a.schema import (
    Artifact,
    Message,
    Skill,
    Task,
    TaskIdParams,
    TaskSendParams,
    TaskState,
    TaskStatus,
)
from fasta2a.storage import Storage
from opentelemetry.trace import get_current_span, get_tracer, use_span
from pydantic import TypeAdapter, ValidationError
from pydantic_ai.messages import ModelMessage, UserPromptPart

from agent_utilities.security.persistence_privacy import (
    persistence_reference,
    sanitize_for_persistence,
)

__all__ = [
    "A2AStorageConflict",
    "EpistemicGraphA2ABroker",
    "EpistemicGraphA2ARuntime",
    "EpistemicGraphA2AStorage",
    "EpistemicGraphAgentWorker",
    "agent_to_epistemic_a2a",
    "build_epistemic_graph_a2a_backends",
]

_T = TypeVar("_T")
_TASK_ADAPTER = TypeAdapter(Task)
_MESSAGE_ADAPTER = TypeAdapter(Message)
_ARTIFACT_ADAPTER = TypeAdapter(Artifact)
_STATE_ADAPTER = TypeAdapter(TaskState)
_TASK_ID_PARAMS_ADAPTER = TypeAdapter(TaskIdParams)
_TASK_SEND_PARAMS_ADAPTER = TypeAdapter(TaskSendParams)
_CONTEXT_ADAPTER = TypeAdapter(list[ModelMessage])
_SCHEMA_VERSION = 1
_TASK_RECORD_KIND = "a2a_task_v1"
_CONTEXT_RECORD_KIND = "a2a_context_v1"
_TERMINAL_STATES = frozenset({"completed", "canceled", "failed", "rejected"})
_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "submitted": frozenset({"working", "canceled", "failed", "rejected"}),
    "working": frozenset(
        {
            "input-required",
            "completed",
            "canceled",
            "failed",
            "rejected",
            "auth-required",
        }
    ),
    "input-required": frozenset({"working", "canceled", "failed", "rejected"}),
    "auth-required": frozenset({"working", "canceled", "failed", "rejected"}),
    "unknown": frozenset({"working", "canceled", "failed", "rejected"}),
}
_HEX_32 = re.compile(r"^[0-9a-f]{32}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_LOWER_HEX = re.compile(r"^[0-9a-f]+$")
_GOVERNED_CONTENT_REF = re.compile(r"^urn:agent-utilities:content:sha256:[0-9a-f]{64}$")
_MAX_STRUCTURE_DEPTH = 32
_MAX_STRUCTURE_ITEMS = 4_096
_MAX_ID_CHARS = 512
_DISPATCH_PAGE_LIMIT = 64
_DISPATCH_RUN = 1
_DISPATCH_CANCEL = 2
_TASK_RECORD_FIELDS = frozenset(
    {
        "record_kind",
        "node_type",
        "tenant_ref",
        "context_id",
        "context_revision",
        "context_payload_ref",
        "revision",
        "state",
        "payload",
        "payload_ref",
        "execution_tag",
        "execution_consumer",
        "run_dispatch_state",
        "run_operation",
        "cancel_dispatch_state",
    }
)
_CONTEXT_RECORD_FIELDS = frozenset(
    {"record_kind", "node_type", "tenant_ref", "revision", "payload", "payload_ref"}
)
_tracer = get_tracer(__name__)


class A2AStorageConflict(RuntimeError):
    """A durable A2A update lost its optimistic concurrency fence."""


class _A2ADeliveryRetry(RuntimeError):
    """Internal signal used to nack one delivery without stopping the worker."""


@dataclass
class _ExecutionBinding:
    task_id: str
    context_id: str
    expected_task_revision: int
    expected_task_payload_ref: str
    expected_context_revision: int
    expected_context_payload_ref: str
    delivery_tag: int
    consumer: str


@dataclass
class _DeliveryControl:
    task_id: str
    delivery_tag: int
    consumer: str
    monitor_cancellation: bool
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    abort_reason: str = ""

    def abort(self, reason: str) -> None:
        if not self.abort_reason:
            self.abort_reason = reason
            self.abort_event.set()


_EXECUTION_BINDING: ContextVar[_ExecutionBinding | None] = ContextVar(
    "a2a_execution_binding", default=None
)
_DELIVERY_CONTROL: ContextVar[_DeliveryControl | None] = ContextVar(
    "a2a_delivery_control", default=None
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError):
        raise ValueError("A2A payload is not deterministic JSON") from None


def _bounded(value: Any, *, maximum: int, label: str) -> bytes:
    encoded = _json_bytes(value)
    if not encoded or len(encoded) > maximum:
        raise ValueError(f"{label} exceeds the configured persistence bound")
    return encoded


def _admit_structure(
    value: Any,
    *,
    maximum: int,
    label: str,
    maximum_items: int = _MAX_STRUCTURE_ITEMS,
) -> None:
    """Bound a JSON-shaped value before validation, copying, or serialization."""

    budget = int(maximum)
    items = 0
    stack: list[tuple[Any, int]] = [(value, 0)]
    containers: set[int] = set()
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_STRUCTURE_DEPTH:
            raise ValueError(f"{label} exceeds the configured nesting bound")
        items += 1
        if items > maximum_items:
            raise ValueError(f"{label} exceeds the configured collection bound")
        if current is None or isinstance(current, bool):
            budget -= 4
        elif isinstance(current, int | float):
            budget -= 32
        elif isinstance(current, str):
            # Count UTF-8 bytes without allocating a second copy of an attacker-
            # controlled string.
            budget -= (
                sum(
                    1
                    if ord(character) < 0x80
                    else 2
                    if ord(character) < 0x800
                    else 3
                    if ord(character) < 0x10000
                    else 4
                    for character in current
                )
                + 2
            )
        elif isinstance(current, dict):
            identity = id(current)
            if identity in containers:
                raise ValueError(f"{label} contains a cyclic structure")
            containers.add(identity)
            budget -= 2
            for key, item in current.items():
                if not isinstance(key, str):
                    raise ValueError(f"{label} contains a non-string object key")
                budget -= (
                    sum(
                        1
                        if ord(character) < 0x80
                        else 2
                        if ord(character) < 0x800
                        else 3
                        if ord(character) < 0x10000
                        else 4
                        for character in key
                    )
                    + 3
                )
                stack.append((item, depth + 1))
        elif isinstance(current, list | tuple):
            identity = id(current)
            if identity in containers:
                raise ValueError(f"{label} contains a cyclic structure")
            containers.add(identity)
            budget -= 2
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, datetime):
            budget -= 40
        elif is_dataclass(current) and not isinstance(current, type):
            identity = id(current)
            if identity in containers:
                raise ValueError(f"{label} contains a cyclic structure")
            containers.add(identity)
            budget -= 2
            stack.extend(
                (getattr(current, item.name), depth + 1) for item in fields(current)
            )
        else:
            # Typed model objects are projected only after the caller has bounded
            # their containing collection. Arbitrary object reprs are never read.
            values = getattr(current, "__dict__", None)
            if not isinstance(values, dict):
                raise ValueError(f"{label} contains unsupported execution material")
            stack.append((values, depth + 1))
        if budget < 0:
            raise ValueError(f"{label} exceeds the configured admission bound")


def _validated_json(adapter: TypeAdapter[_T], value: Any, *, label: str) -> _T:
    """Validate and JSON-project without returning validation input in errors."""

    try:
        validated = adapter.validate_python(value, by_name=True)
        return cast(_T, adapter.dump_python(validated, mode="json"))
    except (RecursionError, TypeError, ValueError, ValidationError):
        raise ValueError(f"{label} violates the current FastA2A schema") from None


def _privacy_json(value: Any, *, label: str) -> Any:
    clean, report = sanitize_for_persistence(value)
    if not isinstance(clean, (dict, list)):
        raise ValueError(f"{label} did not produce structured persistence material")
    if report.changed:
        raise ValueError(
            f"{label} contains execution material prohibited at the persistence boundary"
        )
    return clean


def _digest_component(kind: str, value: Any, *, namespace: str) -> str:
    reference = persistence_reference(kind, value, namespace=namespace)
    digest = reference.rsplit("_", 1)[-1]
    if not _HEX_64.fullmatch(digest):
        raise RuntimeError("persistence reference did not produce an opaque digest")
    return digest


def _payload_ref(value: Any, *, tenant_key: str) -> str:
    content_digest = hashlib.sha256(_json_bytes(value)).hexdigest()
    digest = _digest_component(
        "payload", content_digest, namespace=f"a2a:{tenant_key}:payload"
    )
    return f"a2a.payload.{digest}"


def _valid_payload_ref(value: Any) -> bool:
    rendered = str(value or "")
    return rendered.startswith("a2a.payload.") and bool(
        _HEX_64.fullmatch(rendered.removeprefix("a2a.payload."))
    )


def _prepare_governed_parts(
    value: dict[str, Any], *, label: str
) -> tuple[dict[str, Any], list[tuple[int, str]]]:
    """Blank validated opaque refs during generic location-field sanitization."""

    projected = json.loads(_json_bytes(value))
    restored: list[tuple[int, str]] = []
    parts = projected.get("parts")
    if not isinstance(parts, list):
        return projected, restored
    for index, part in enumerate(parts):
        if not isinstance(part, dict) or part.get("kind") != "file":
            continue
        file_value = part.get("file")
        if not isinstance(file_value, dict):
            raise ValueError(f"{label} file part is invalid")
        if "bytes" in file_value:
            raise ValueError(
                f"{label} inline file bytes are prohibited; use an opaque governed reference"
            )
        reference = str(file_value.get("uri") or "")
        if not _GOVERNED_CONTENT_REF.fullmatch(reference):
            raise ValueError(f"{label} file input is not an opaque governed reference")
        file_value["uri"] = ""
        restored.append((index, reference))
    return projected, restored


def _restore_governed_parts(
    value: dict[str, Any], restored: list[tuple[int, str]], *, label: str
) -> None:
    parts = value.get("parts")
    if not isinstance(parts, list):
        raise ValueError(f"{label} parts disappeared during privacy validation")
    for index, reference in restored:
        try:
            file_value = parts[index]["file"]
        except (IndexError, KeyError, TypeError):
            raise ValueError(
                f"{label} governed reference disappeared during privacy validation"
            ) from None
        if not isinstance(file_value, dict) or file_value.get("uri") != "":
            raise ValueError(
                f"{label} governed reference changed during privacy validation"
            )
        file_value["uri"] = reference


@dataclass
class EpistemicGraphA2ARuntime:
    """Shared verified authority for the FastA2A broker and storage adapters."""

    client: Any | None = None
    session: Any | None = None
    tenant_ref: str = field(default="", init=False)
    tenant_key: str = field(default="", init=False)
    _start_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _started: bool = field(default=False, init=False)

    async def start(self) -> None:
        async with self._start_lock:
            if self._started:
                return
            if (self.client is None) != (self.session is None):
                raise RuntimeError(
                    "A2A runtime client and session must be supplied together"
                )
            if self.client is None:
                from agent_utilities.core.config import config
                from agent_utilities.security.request_identity import (
                    acquire_process_identity_token,
                    actor_from_bearer_token,
                    mint_graph_session,
                )

                token = await anyio.to_thread.run_sync(
                    acquire_process_identity_token, config
                )
                actor = await actor_from_bearer_token(token)
                session = await anyio.to_thread.run_sync(mint_graph_session, actor)

                def _resolve_client() -> Any:
                    from agent_utilities.knowledge_graph.core.graph_compute import (
                        GraphComputeEngine,
                    )
                    from agent_utilities.knowledge_graph.core.session import use_session
                    from agent_utilities.security.brain_context import use_actor

                    with use_actor(session.actor), use_session(session):
                        return GraphComputeEngine.get_or_create(
                            graph_name=session.graph
                        ).client

                self.client = await anyio.to_thread.run_sync(_resolve_client)
                self.session = session

            session = self.session
            tenant = str(getattr(session, "tenant", "") or "").strip()
            if (
                session is None
                or not tenant
                or not getattr(
                    getattr(session, "actor", None), "authenticated", False
                )
            ):
                raise RuntimeError("A2A persistence requires verified tenant authority")
            session.require_scope("kg:read")
            session.require_scope("kg:write")
            self.tenant_ref = persistence_reference(
                "tenant", tenant, namespace="a2a-runtime"
            )
            self.tenant_key = _digest_component(
                "tenant", tenant, namespace="a2a-runtime"
            )
            self._started = True

    async def call(self, namespace: str, method: str, *args: Any, **kwargs: Any) -> Any:
        await self.start()
        session = self.session
        if session is None:
            raise RuntimeError("A2A persistence requires a started session")
        client = self.client
        surface = getattr(client, namespace, None)
        operation = getattr(surface, method, None)
        if not callable(operation):
            raise RuntimeError(
                f"Epistemic Graph lacks the required A2A {namespace}.{method} capability"
            )

        def _invoke() -> Any:
            from agent_utilities.knowledge_graph.core.session import use_session
            from agent_utilities.security.brain_context import use_actor

            with use_actor(session.actor), use_session(session):
                return operation(*args, **kwargs)

        return await anyio.to_thread.run_sync(_invoke)

    def task_prefix(self) -> str:
        if not self.tenant_key:
            raise RuntimeError("A2A runtime has not started")
        return f"a2a.task.{self.tenant_key}."

    def context_id(self, value: str) -> str:
        if not self.tenant_key:
            raise RuntimeError("A2A runtime has not started")
        rendered = str(value or "").strip()
        if (
            not rendered
            or len(rendered) > _MAX_ID_CHARS
            or any(ord(character) < 32 for character in rendered)
        ):
            raise ValueError("A2A context id is invalid")
        prefix = f"a2a.context.{self.tenant_key}."
        if rendered.startswith(prefix):
            suffix = rendered.removeprefix(prefix)
            if not _HEX_64.fullmatch(suffix):
                raise ValueError("A2A context id is not a canonical opaque identifier")
            return rendered
        digest = _digest_component(
            "context", rendered, namespace=f"a2a:{self.tenant_key}"
        )
        return f"{prefix}{digest}"

    def require_task_id(self, value: str) -> str:
        rendered = str(value or "").strip()
        prefix = self.task_prefix()
        suffix = rendered.removeprefix(prefix)
        if not rendered.startswith(prefix) or not _HEX_32.fullmatch(suffix):
            raise ValueError("task id is not an opaque identifier for this tenant")
        return rendered


class EpistemicGraphA2AStorage(Storage[list[ModelMessage]]):
    """FastA2A task/context storage on exact, CAS-fenced graph records."""

    def __init__(
        self,
        runtime: EpistemicGraphA2ARuntime,
        *,
        max_payload_bytes: int = 262_144,
        max_history: int = 100,
        max_artifacts: int = 50,
        max_context_messages: int = 100,
        update_retries: int = 4,
    ) -> None:
        self.runtime = runtime
        self.max_payload_bytes = int(max_payload_bytes)
        self.max_history = int(max_history)
        self.max_artifacts = int(max_artifacts)
        self.max_context_messages = int(max_context_messages)
        self.update_retries = int(update_retries)
        if (
            min(
                self.max_payload_bytes,
                self.max_history,
                self.max_artifacts,
                self.max_context_messages,
                self.update_retries,
            )
            <= 0
        ):
            raise ValueError("A2A storage bounds must be positive")

    def _task_id(self) -> str:
        return f"{self.runtime.task_prefix()}{uuid.uuid4().hex}"

    def _message(self, value: Message, *, task_id: str, context_id: str) -> Message:
        _admit_structure(
            value,
            maximum=self.max_payload_bytes,
            maximum_items=max(64, self.max_history * 8),
            label="A2A message",
        )
        raw = cast(
            dict[str, Any],
            _validated_json(_MESSAGE_ADAPTER, value, label="A2A message"),
        )
        if len(raw.get("parts") or []) > self.max_history:
            raise ValueError("A2A message has too many parts")
        for key in ("reference_task_ids", "extensions"):
            if len(raw.get(key) or []) > self.max_history:
                raise ValueError(f"A2A message {key} exceeds the collection bound")
        projected, restored = _prepare_governed_parts(raw, label="A2A message")
        clean = cast(dict[str, Any], _privacy_json(projected, label="A2A message"))
        _restore_governed_parts(clean, restored, label="A2A message")
        clean["task_id"] = task_id
        clean["context_id"] = context_id
        clean["message_id"] = "a2a.message." + _digest_component(
            "message",
            raw.get("message_id"),
            namespace=f"a2a:{self.runtime.tenant_key}",
        )
        if "reference_task_ids" in raw:
            clean["reference_task_ids"] = [
                "a2a.taskref."
                + _digest_component(
                    "task", item, namespace=f"a2a:{self.runtime.tenant_key}"
                )
                for item in raw.get("reference_task_ids") or []
            ]
        if "extensions" in raw:
            clean["extensions"] = [
                "a2a.extension."
                + _digest_component(
                    "extension", item, namespace=f"a2a:{self.runtime.tenant_key}"
                )
                for item in raw.get("extensions") or []
            ]
        message = _validated_json(
            _MESSAGE_ADAPTER, clean, label="privacy-approved A2A message"
        )
        _bounded(
            message,
            maximum=self.max_payload_bytes,
            label="privacy-approved A2A message",
        )
        return message

    def _artifact(self, value: Artifact) -> Artifact:
        _admit_structure(
            value,
            maximum=self.max_payload_bytes,
            maximum_items=max(64, self.max_artifacts * 8),
            label="A2A artifact",
        )
        raw = cast(
            dict[str, Any],
            _validated_json(_ARTIFACT_ADAPTER, value, label="A2A artifact"),
        )
        if len(raw.get("parts") or []) > self.max_history:
            raise ValueError("A2A artifact has too many parts")
        projected, restored = _prepare_governed_parts(raw, label="A2A artifact")
        clean = cast(dict[str, Any], _privacy_json(projected, label="A2A artifact"))
        _restore_governed_parts(clean, restored, label="A2A artifact")
        clean["artifact_id"] = "a2a.artifact." + _digest_component(
            "artifact",
            raw.get("artifact_id"),
            namespace=f"a2a:{self.runtime.tenant_key}",
        )
        artifact = _validated_json(
            _ARTIFACT_ADAPTER, clean, label="privacy-approved A2A artifact"
        )
        _bounded(
            artifact,
            maximum=self.max_payload_bytes,
            label="privacy-approved A2A artifact",
        )
        return artifact

    def _context_record(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != _CONTEXT_RECORD_FIELDS:
            raise RuntimeError("native A2A context record is invalid")
        if (
            value.get("record_kind") != _CONTEXT_RECORD_KIND
            or value.get("node_type") != "A2AContext"
            or value.get("tenant_ref") != self.runtime.tenant_ref
            or type(value.get("revision")) is not int
            or value["revision"] < 0
            or not isinstance(value.get("payload"), list)
            or not _valid_payload_ref(value.get("payload_ref"))
            or value["payload_ref"]
            != _payload_ref(value["payload"], tenant_key=self.runtime.tenant_key)
        ):
            raise RuntimeError("native A2A context record is invalid")
        return value

    def _task_record(self, value: Any, task_id: str) -> tuple[dict[str, Any], Task]:
        self.runtime.require_task_id(task_id)
        if not isinstance(value, dict) or set(value) != _TASK_RECORD_FIELDS:
            raise RuntimeError("native A2A task record is invalid")
        if (
            value.get("record_kind") != _TASK_RECORD_KIND
            or value.get("node_type") != "A2ATask"
            or value.get("tenant_ref") != self.runtime.tenant_ref
            or type(value.get("revision")) is not int
            or value["revision"] < 0
            or not isinstance(value.get("payload"), dict)
            or not _valid_payload_ref(value.get("payload_ref"))
            or value["payload_ref"]
            != _payload_ref(value["payload"], tenant_key=self.runtime.tenant_key)
            or type(value.get("context_revision")) is not int
            or value["context_revision"] < 0
            or not _valid_payload_ref(value.get("context_payload_ref"))
            or value.get("run_dispatch_state")
            not in {"pending", "published", "suppressed"}
            or value.get("cancel_dispatch_state")
            not in {"none", "pending", "published"}
        ):
            raise RuntimeError("native A2A task record is invalid")
        task = _validated_json(_TASK_ADAPTER, value["payload"], label="stored A2A task")
        context_id = self.runtime.context_id(str(task.get("context_id") or ""))
        if (
            task.get("id") != task_id
            or value.get("context_id") != context_id
            or value.get("state") != task["status"]["state"]
        ):
            raise RuntimeError("native A2A task identity does not match its record")
        tag = value.get("execution_tag")
        consumer = value.get("execution_consumer")
        if (tag is None) != (consumer is None) or (
            tag is not None
            and (
                type(tag) is not int
                or tag <= 0
                or not isinstance(consumer, str)
                or not consumer
            )
        ):
            raise RuntimeError("native A2A task execution fence is invalid")
        run_operation = value.get("run_operation")
        if not isinstance(run_operation, dict):
            raise RuntimeError("native A2A task dispatch record is invalid")
        self._validate_run_operation(run_operation, task_id, context_id)
        return value, task

    def _validate_run_operation(
        self, value: dict[str, Any], task_id: str, context_id: str
    ) -> dict[str, Any]:
        if set(value) != {"id", "context_id", "message"}:
            raise RuntimeError("native A2A run dispatch is invalid")
        params = cast(
            dict[str, Any],
            _validated_json(
                _TASK_SEND_PARAMS_ADAPTER, value, label="stored A2A run dispatch"
            ),
        )
        message = params.get("message")
        if (
            params.get("id") != task_id
            or params.get("context_id") != context_id
            or not isinstance(message, dict)
            or set(message)
            != {
                "role",
                "parts",
                "kind",
                "message_id",
                "task_id",
                "context_id",
            }
            or message.get("parts") != []
            or message.get("task_id") != task_id
            or message.get("context_id") != context_id
            or not isinstance(message.get("message_id"), str)
            or not message["message_id"].startswith("a2a.message.")
            or not _HEX_64.fullmatch(message["message_id"].removeprefix("a2a.message."))
        ):
            raise RuntimeError("native A2A run dispatch is invalid")
        return params

    def _task_conditions(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "record_kind": _TASK_RECORD_KIND,
            "node_type": "A2ATask",
            "tenant_ref": self.runtime.tenant_ref,
            "context_id": record["context_id"],
            "revision": record["revision"],
            "state": record["state"],
            "payload_ref": record["payload_ref"],
            "execution_tag": record["execution_tag"],
            "execution_consumer": record["execution_consumer"],
        }

    def _context_conditions(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "record_kind": _CONTEXT_RECORD_KIND,
            "node_type": "A2AContext",
            "tenant_ref": self.runtime.tenant_ref,
            "revision": record["revision"],
            "payload_ref": record["payload_ref"],
        }

    async def _context_fence(self, context_id: str) -> tuple[int, str]:
        initial_payload: list[Any] = []
        initial = {
            "record_kind": _CONTEXT_RECORD_KIND,
            "node_type": "A2AContext",
            "tenant_ref": self.runtime.tenant_ref,
            "revision": 0,
            "payload": initial_payload,
            "payload_ref": _payload_ref(
                initial_payload, tenant_key=self.runtime.tenant_key
            ),
        }
        created = await self.runtime.call(
            "nodes", "create_if_absent", context_id, initial
        )
        if type(created) is not bool:
            raise RuntimeError("native A2A create-if-absent returned an invalid result")
        properties = await self.runtime.call("nodes", "properties", context_id)
        record = self._context_record(properties)
        return int(record["revision"]), str(record["payload_ref"])

    async def load_task(
        self, task_id: str, history_length: int | None = None
    ) -> Task | None:
        await self.runtime.start()
        task_id = self.runtime.require_task_id(task_id)
        properties = await self.runtime.call("nodes", "properties", task_id)
        if properties is None:
            return None
        _record, task = self._task_record(properties, task_id)
        result = json.loads(_json_bytes(task))
        requested = self.max_history if history_length is None else int(history_length)
        if requested < 0:
            raise ValueError("A2A history length cannot be negative")
        requested = min(requested, self.max_history)
        if "history" in result:
            result["history"] = result["history"][-requested:] if requested else []
        return cast(Task, result)

    async def submit_task(self, context_id: str, message: Message) -> Task:
        await self.runtime.start()
        context_id = self.runtime.context_id(context_id)
        for _attempt in range(self.update_retries):
            task_id = self._task_id()
            clean_message = self._message(
                message, task_id=task_id, context_id=context_id
            )
            context_revision, context_payload_ref = await self._context_fence(
                context_id
            )
            task: Task = {
                "id": task_id,
                "context_id": context_id,
                "kind": "task",
                "status": TaskStatus(state="submitted", timestamp=_now_iso()),
                "history": [clean_message],
            }
            task = _validated_json(_TASK_ADAPTER, task, label="A2A task")
            _bounded(task, maximum=self.max_payload_bytes, label="A2A task")
            run_operation = {
                "id": task_id,
                "context_id": context_id,
                "message": {
                    "role": clean_message["role"],
                    "parts": [],
                    "kind": "message",
                    "message_id": clean_message["message_id"],
                    "task_id": task_id,
                    "context_id": context_id,
                },
            }
            self._validate_run_operation(run_operation, task_id, context_id)
            properties = {
                "record_kind": _TASK_RECORD_KIND,
                "node_type": "A2ATask",
                "tenant_ref": self.runtime.tenant_ref,
                "context_id": context_id,
                "context_revision": context_revision,
                "context_payload_ref": context_payload_ref,
                "revision": 0,
                "state": "submitted",
                "payload": task,
                "payload_ref": _payload_ref(task, tenant_key=self.runtime.tenant_key),
                "execution_tag": None,
                "execution_consumer": None,
                "run_dispatch_state": "pending",
                "run_operation": run_operation,
                "cancel_dispatch_state": "none",
            }
            created = await self.runtime.call(
                "nodes", "create_if_absent", task_id, properties
            )
            if type(created) is not bool:
                raise RuntimeError(
                    "native A2A create-if-absent returned an invalid result"
                )
            if not created:
                continue
            stored = await self.runtime.call("nodes", "properties", task_id)
            self._task_record(stored, task_id)
            return cast(Task, json.loads(_json_bytes(task)))
        raise A2AStorageConflict("A2A task id allocation exceeded its retry budget")

    @staticmethod
    def _validate_transition(current: str, requested: str) -> None:
        if current == requested:
            return
        if current in _TERMINAL_STATES or requested not in _ALLOWED_TRANSITIONS.get(
            current, frozenset()
        ):
            raise A2AStorageConflict("A2A task state transition is not permitted")

    def _require_binding(self, task_id: str) -> _ExecutionBinding:
        binding = _EXECUTION_BINDING.get()
        if binding is None or binding.task_id != task_id:
            raise A2AStorageConflict(
                "A2A task update lacks the delivery's execution fence"
            )
        return binding

    def _binding_matches(
        self, binding: _ExecutionBinding, record: dict[str, Any]
    ) -> bool:
        return self._binding_base_matches(binding, record) and (
            record["state"] == "submitted"
            or (
                record["execution_tag"] == binding.delivery_tag
                and record["execution_consumer"] == binding.consumer
            )
        )

    @staticmethod
    def _binding_base_matches(
        binding: _ExecutionBinding, record: dict[str, Any]
    ) -> bool:
        return (
            record["revision"] == binding.expected_task_revision
            and record["payload_ref"] == binding.expected_task_payload_ref
            and record["context_revision"] == binding.expected_context_revision
            and record["context_payload_ref"] == binding.expected_context_payload_ref
            and record["context_id"] == binding.context_id
        )

    async def update_task(
        self,
        task_id: str,
        state: TaskState,
        new_artifacts: list[Artifact] | None = None,
        new_messages: list[Message] | None = None,
    ) -> Task:
        await self.runtime.start()
        task_id = self.runtime.require_task_id(task_id)
        binding = self._require_binding(task_id)
        requested_state = _STATE_ADAPTER.validate_python(state)
        if new_artifacts is not None and len(new_artifacts) > self.max_artifacts:
            raise ValueError("A2A artifact update exceeds the collection bound")
        if new_messages is not None and len(new_messages) > self.max_history:
            raise ValueError("A2A message update exceeds the collection bound")
        properties = await self.runtime.call("nodes", "properties", task_id)
        if properties is None:
            raise KeyError("A2A task does not exist")
        record, current = self._task_record(properties, task_id)
        claiming_execution = requested_state == "working" and record["state"] in {
            "submitted",
            "working",
        }
        if not (
            self._binding_base_matches(binding, record)
            if claiming_execution
            else self._binding_matches(binding, record)
        ):
            raise A2AStorageConflict("A2A task execution fence changed")
        self._validate_transition(current["status"]["state"], requested_state)
        updated = cast(Task, json.loads(_json_bytes(current)))
        updated["status"] = TaskStatus(state=requested_state, timestamp=_now_iso())
        context_id = self.runtime.context_id(updated["context_id"])
        if new_artifacts:
            artifacts = [self._artifact(item) for item in new_artifacts]
            updated["artifacts"] = [
                *(updated.get("artifacts") or []),
                *artifacts,
            ][-self.max_artifacts :]
        if new_messages:
            messages = [
                self._message(item, task_id=task_id, context_id=context_id)
                for item in new_messages
            ]
            updated["history"] = [
                *(updated.get("history") or []),
                *messages,
            ][-self.max_history :]
        updated = _validated_json(_TASK_ADAPTER, updated, label="updated A2A task")
        _bounded(updated, maximum=self.max_payload_bytes, label="updated A2A task")
        revision = int(record["revision"])
        payload_ref = _payload_ref(updated, tenant_key=self.runtime.tenant_key)
        terminal = requested_state in _TERMINAL_STATES
        updates = {
            "revision": revision + 1,
            "state": requested_state,
            "payload": updated,
            "payload_ref": payload_ref,
            "execution_tag": None if terminal else binding.delivery_tag,
            "execution_consumer": None if terminal else binding.consumer,
        }
        applied = await self.runtime.call(
            "nodes",
            "compare_and_set",
            task_id,
            self._task_conditions(record),
            updates,
        )
        if type(applied) is not bool or not applied:
            raise A2AStorageConflict("A2A task update lost its execution fence")
        binding.expected_task_revision = revision + 1
        binding.expected_task_payload_ref = payload_ref
        return cast(Task, json.loads(_json_bytes(updated)))

    async def cancel_task(self, task_id: str) -> Task:
        """Durably cancel before the HTTP request returns or queues a wake record."""

        await self.runtime.start()
        task_id = self.runtime.require_task_id(task_id)
        for _attempt in range(self.update_retries):
            properties = await self.runtime.call("nodes", "properties", task_id)
            if properties is None:
                raise KeyError("A2A task does not exist")
            record, current = self._task_record(properties, task_id)
            current_state = str(current["status"]["state"])
            if current_state == "canceled":
                return current
            if current_state in _TERMINAL_STATES:
                raise A2AStorageConflict("terminal A2A task cannot be canceled")
            updated = cast(Task, json.loads(_json_bytes(current)))
            updated["status"] = TaskStatus(state="canceled", timestamp=_now_iso())
            updated = _validated_json(_TASK_ADAPTER, updated, label="canceled A2A task")
            payload_ref = _payload_ref(updated, tenant_key=self.runtime.tenant_key)
            applied = await self.runtime.call(
                "nodes",
                "compare_and_set",
                task_id,
                self._task_conditions(record),
                {
                    "revision": record["revision"] + 1,
                    "state": "canceled",
                    "payload": updated,
                    "payload_ref": payload_ref,
                    "execution_tag": None,
                    "execution_consumer": None,
                    "cancel_dispatch_state": "pending",
                },
            )
            if type(applied) is not bool:
                raise RuntimeError("native A2A CAS returned an invalid result")
            if applied:
                return updated
        raise A2AStorageConflict("A2A cancellation exceeded its CAS retry budget")

    async def load_context(self, context_id: str) -> list[ModelMessage] | None:
        await self.runtime.start()
        context_id = self.runtime.context_id(context_id)
        properties = await self.runtime.call("nodes", "properties", context_id)
        if properties is None:
            return None
        record = self._context_record(properties)
        binding = _EXECUTION_BINDING.get()
        if binding is not None and binding.context_id == context_id:
            if (
                binding.expected_context_revision != record["revision"]
                or binding.expected_context_payload_ref != record["payload_ref"]
            ):
                raise A2AStorageConflict(
                    "A2A context changed after task submission; refusing stale execution"
                )
            task_properties = await self.runtime.call(
                "nodes", "properties", binding.task_id
            )
            task_record, _task = self._task_record(task_properties, binding.task_id)
            if not self._binding_matches(binding, task_record):
                raise A2AStorageConflict("A2A task changed before context load")
        try:
            return _CONTEXT_ADAPTER.validate_python(record["payload"])
        except (RecursionError, TypeError, ValueError, ValidationError):
            raise RuntimeError("native A2A context payload is invalid") from None

    def _normalize_context(self, context: list[ModelMessage]) -> list[Any]:
        if len(context) > self.max_context_messages:
            raise ValueError("A2A model context exceeds the collection bound")
        _admit_structure(
            context,
            maximum=self.max_payload_bytes,
            maximum_items=max(64, self.max_context_messages * 32),
            label="A2A model context",
        )
        try:
            raw = _CONTEXT_ADAPTER.dump_python(context, mode="json")
        except (RecursionError, TypeError, ValueError, ValidationError):
            raise ValueError("A2A model context violates the current schema") from None
        clean = _privacy_json(raw, label="A2A model context")
        try:
            normalized = _CONTEXT_ADAPTER.dump_python(
                _CONTEXT_ADAPTER.validate_python(clean), mode="json"
            )
        except (RecursionError, TypeError, ValueError, ValidationError):
            raise ValueError("privacy-approved A2A model context is invalid") from None
        _bounded(
            normalized,
            maximum=self.max_payload_bytes,
            label="privacy-approved A2A model context",
        )
        return list(normalized)

    async def _atomic_context_task_update(
        self,
        *,
        binding: _ExecutionBinding,
        context_record: dict[str, Any],
        context_updates: dict[str, Any],
        task_record: dict[str, Any],
        task_updates: dict[str, Any],
    ) -> None:
        txn_id = await self.runtime.call("txn", "begin")
        if not isinstance(txn_id, str) or not txn_id:
            raise RuntimeError("native A2A transaction did not return an id")
        committed = False
        try:
            staged_context = await self.runtime.call(
                "txn",
                "cas",
                txn_id,
                binding.context_id,
                self._context_conditions(context_record),
                context_updates,
            )
            staged_task = await self.runtime.call(
                "txn",
                "cas",
                txn_id,
                binding.task_id,
                self._task_conditions(task_record),
                task_updates,
            )
            if type(staged_context) is not bool or type(staged_task) is not bool:
                raise RuntimeError(
                    "native A2A transaction staging returned invalid results"
                )
            if not staged_context or not staged_task:
                raise A2AStorageConflict("A2A transaction rejected an execution fence")
            result = await self.runtime.call("txn", "commit", txn_id)
            if type(result) is not bool:
                raise RuntimeError(
                    "native A2A transaction commit returned an invalid result"
                )
            committed = result
        finally:
            if not committed:
                with contextlib.suppress(Exception):
                    await self.runtime.call("txn", "rollback", txn_id)
        if not committed:
            raise A2AStorageConflict("A2A context/task transaction lost its fence")

    async def update_context(
        self, context_id: str, context: list[ModelMessage]
    ) -> None:
        await self.runtime.start()
        context_id = self.runtime.context_id(context_id)
        binding = _EXECUTION_BINDING.get()
        if binding is None or binding.context_id != context_id:
            raise A2AStorageConflict(
                "A2A context update lacks the task's execution fence"
            )
        normalized = self._normalize_context(context)
        context_properties = await self.runtime.call("nodes", "properties", context_id)
        context_record = self._context_record(context_properties)
        task_properties = await self.runtime.call(
            "nodes", "properties", binding.task_id
        )
        task_record, _task = self._task_record(task_properties, binding.task_id)
        if (
            context_record["revision"] != binding.expected_context_revision
            or context_record["payload_ref"] != binding.expected_context_payload_ref
            or not self._binding_matches(binding, task_record)
            or task_record["state"] != "working"
        ):
            raise A2AStorageConflict("A2A context/task execution fence changed")
        context_ref = _payload_ref(normalized, tenant_key=self.runtime.tenant_key)
        task_revision = task_record["revision"] + 1
        await self._atomic_context_task_update(
            binding=binding,
            context_record=context_record,
            context_updates={
                "revision": context_record["revision"] + 1,
                "payload": normalized,
                "payload_ref": context_ref,
            },
            task_record=task_record,
            task_updates={
                "revision": task_revision,
                "context_revision": context_record["revision"] + 1,
                "context_payload_ref": context_ref,
            },
        )
        binding.expected_context_revision = context_record["revision"] + 1
        binding.expected_context_payload_ref = context_ref
        binding.expected_task_revision = task_revision

    async def complete_task(
        self,
        task_id: str,
        context: list[ModelMessage],
        *,
        new_artifacts: list[Artifact],
        new_messages: list[Message],
    ) -> Task:
        """Atomically commit model context and the terminal task result."""

        await self.runtime.start()
        task_id = self.runtime.require_task_id(task_id)
        binding = self._require_binding(task_id)
        if (
            len(new_artifacts) > self.max_artifacts
            or len(new_messages) > self.max_history
        ):
            raise ValueError("A2A completion exceeds a configured collection bound")
        context_properties = await self.runtime.call(
            "nodes", "properties", binding.context_id
        )
        context_record = self._context_record(context_properties)
        task_properties = await self.runtime.call("nodes", "properties", task_id)
        task_record, current = self._task_record(task_properties, task_id)
        if (
            context_record["revision"] != binding.expected_context_revision
            or context_record["payload_ref"] != binding.expected_context_payload_ref
            or not self._binding_matches(binding, task_record)
            or task_record["state"] != "working"
        ):
            raise A2AStorageConflict("A2A completion lost its execution fence")
        normalized = self._normalize_context(context)
        artifacts = [self._artifact(item) for item in new_artifacts]
        messages = [
            self._message(item, task_id=task_id, context_id=binding.context_id)
            for item in new_messages
        ]
        updated = cast(Task, json.loads(_json_bytes(current)))
        updated["status"] = TaskStatus(state="completed", timestamp=_now_iso())
        if artifacts:
            updated["artifacts"] = [
                *(updated.get("artifacts") or []),
                *artifacts,
            ][-self.max_artifacts :]
        if messages:
            updated["history"] = [
                *(updated.get("history") or []),
                *messages,
            ][-self.max_history :]
        updated = _validated_json(_TASK_ADAPTER, updated, label="completed A2A task")
        _bounded(updated, maximum=self.max_payload_bytes, label="completed A2A task")
        context_ref = _payload_ref(normalized, tenant_key=self.runtime.tenant_key)
        task_ref = _payload_ref(updated, tenant_key=self.runtime.tenant_key)
        task_revision = task_record["revision"] + 1
        await self._atomic_context_task_update(
            binding=binding,
            context_record=context_record,
            context_updates={
                "revision": context_record["revision"] + 1,
                "payload": normalized,
                "payload_ref": context_ref,
            },
            task_record=task_record,
            task_updates={
                "revision": task_revision,
                "state": "completed",
                "payload": updated,
                "payload_ref": task_ref,
                "context_revision": context_record["revision"] + 1,
                "context_payload_ref": context_ref,
                "execution_tag": None,
                "execution_consumer": None,
            },
        )
        binding.expected_context_revision = context_record["revision"] + 1
        binding.expected_context_payload_ref = context_ref
        binding.expected_task_revision = task_revision
        binding.expected_task_payload_ref = task_ref
        return cast(Task, json.loads(_json_bytes(updated)))

    async def record_for_execution(
        self, task_id: str, context_id: str
    ) -> tuple[dict[str, Any], Task]:
        properties = await self.runtime.call("nodes", "properties", task_id)
        record, task = self._task_record(properties, task_id)
        if record["context_id"] != context_id:
            raise ValueError("native A2A task context does not match its operation")
        return record, task

    async def execution_control_state(self, binding: _ExecutionBinding) -> str:
        properties = await self.runtime.call("nodes", "properties", binding.task_id)
        if properties is None:
            return "lost"
        try:
            record, _task = self._task_record(properties, binding.task_id)
        except (RuntimeError, ValueError):
            return "lost"
        if record["state"] == "canceled":
            return "canceled"
        if record["state"] in _TERMINAL_STATES:
            return "terminal"
        return "active" if self._binding_matches(binding, record) else "lost"

    async def mark_dispatch(self, task_id: str, kind: str, state: str) -> None:
        field_name = f"{kind}_dispatch_state"
        if kind not in {"run", "cancel"} or state not in {"published", "suppressed"}:
            raise ValueError("A2A dispatch state is invalid")
        for _attempt in range(self.update_retries):
            properties = await self.runtime.call("nodes", "properties", task_id)
            record, _task = self._task_record(properties, task_id)
            current = record[field_name]
            if current == state:
                return
            if current not in {"pending", "none"}:
                raise A2AStorageConflict("A2A dispatch state is not pending")
            applied = await self.runtime.call(
                "nodes",
                "compare_and_set",
                task_id,
                {
                    "record_kind": _TASK_RECORD_KIND,
                    "tenant_ref": self.runtime.tenant_ref,
                    field_name: current,
                },
                {field_name: state},
            )
            if type(applied) is not bool:
                raise RuntimeError("native A2A CAS returned an invalid result")
            if applied:
                return
        raise A2AStorageConflict("A2A dispatch update exceeded its CAS retry budget")

    async def dispatch_page(
        self, *, after: str | None, limit: int
    ) -> tuple[list[tuple[str, str, dict[str, Any], int]], str | None]:
        rows = await self.runtime.call(
            "nodes", "list_by_label", "A2ATask", limit, after=after
        )
        if not isinstance(rows, list) or len(rows) > limit:
            raise RuntimeError("native A2A dispatch scan returned an invalid page")
        pending: list[tuple[str, str, dict[str, Any], int]] = []
        cursor: str | None = None
        for row in rows:
            if (
                not isinstance(row, (tuple, list))
                or len(row) != 2
                or not isinstance(row[0], str)
            ):
                raise RuntimeError("native A2A dispatch scan returned an invalid row")
            task_id, properties = row
            record, _task = self._task_record(properties, task_id)
            cursor = task_id
            if record["run_dispatch_state"] == "pending":
                if record["state"] in {"submitted", "working"}:
                    pending.append(
                        (task_id, "run", record["run_operation"], _DISPATCH_RUN)
                    )
                else:
                    await self.mark_dispatch(task_id, "run", "suppressed")
            if record["cancel_dispatch_state"] == "pending":
                pending.append((task_id, "cancel", {"id": task_id}, _DISPATCH_CANCEL))
        return pending, cursor


@dataclass
class EpistemicGraphA2ABroker(Broker):
    """FastA2A broker backed by the fenced native durable engine broker."""

    runtime: EpistemicGraphA2ARuntime
    storage: EpistemicGraphA2AStorage
    poll_interval_ms: int = 100
    lease_ms: int = 300_000
    prefetch: int = 1
    max_payload_bytes: int = 262_144
    message_ttl_ms: int = 86_400_000
    max_delivery_count: int = 5
    reconcile_interval_ms: int = 1_000
    reconcile_limit: int = _DISPATCH_PAGE_LIMIT
    cancellation_poll_interval_ms: int = 1_000
    _active: bool = field(default=False, init=False)
    _exchange: str = field(default="", init=False)
    _queue: str = field(default="", init=False)
    _consumer: str = field(
        default_factory=lambda: f"worker-{uuid.uuid4().hex}", init=False
    )
    _reconcile_cursor: str | None = field(default=None, init=False)
    _reconcile_task: asyncio.Task[None] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not 10 <= self.reconcile_interval_ms <= 60_000:
            raise ValueError("A2A dispatch reconcile interval is outside its bound")
        if not 1 <= self.reconcile_limit <= 1_024:
            raise ValueError("A2A dispatch reconcile page limit is outside its bound")
        if not 10 <= self.cancellation_poll_interval_ms <= 60_000:
            raise ValueError("A2A cancellation poll interval is outside its bound")

    async def __aenter__(self) -> EpistemicGraphA2ABroker:
        await self.runtime.start()
        tenant_key = self.runtime.tenant_key
        self._exchange = f"a2a.operations.{tenant_key}"
        self._queue = f"a2a.worker.{tenant_key}"
        declared = await self.runtime.call(
            "broker", "declare_exchange", self._exchange, kind="direct"
        )
        queued = await self.runtime.call(
            "broker",
            "declare_queue",
            self._queue,
            max_delivery_count=self.max_delivery_count,
            message_ttl_ms=self.message_ttl_ms,
        )
        bound = await self.runtime.call(
            "broker", "bind_queue", self._exchange, self._queue, "task"
        )
        if (declared, queued, bound) != ("ok", "ok", "ok"):
            raise RuntimeError(
                "native A2A broker declaration returned an invalid result"
            )
        self._active = True
        await self._reconcile_once()
        self._reconcile_task = asyncio.create_task(
            self._reconcile_loop(), name="a2a-dispatch-reconciler"
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._active = False
        task = self._reconcile_task
        self._reconcile_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _message_ref(self, message: Any) -> str:
        raw = message if isinstance(message, dict) else {}
        value = str(raw.get("message_id") or "")
        if value.startswith("a2a.message.") and _HEX_64.fullmatch(
            value.removeprefix("a2a.message.")
        ):
            return value
        return "a2a.message." + _digest_component(
            "message", value, namespace=f"a2a:{self.runtime.tenant_key}"
        )

    def _safe_run_params(self, params: TaskSendParams) -> dict[str, Any]:
        task_id = self.runtime.require_task_id(str(params.get("id") or ""))
        context_id = self.runtime.context_id(str(params.get("context_id") or ""))
        raw_message = params.get("message")
        role = str(raw_message.get("role") if isinstance(raw_message, dict) else "user")
        if role not in {"user", "agent"}:
            raise ValueError("A2A broker message role is invalid")
        return {
            "id": task_id,
            "context_id": context_id,
            "message": {
                "role": role,
                "parts": [],
                "kind": "message",
                "message_id": self._message_ref(raw_message),
                "task_id": task_id,
                "context_id": context_id,
            },
        }

    async def _publish_dispatch(
        self, task_id: str, kind: str, params: dict[str, Any], sequence: int
    ) -> None:
        payload = _bounded(
            {"schema_version": _SCHEMA_VERSION, "operation": kind, "params": params},
            maximum=self.max_payload_bytes,
            label="A2A broker operation",
        )
        producer_id = "a2a.producer." + _digest_component(
            "producer", task_id, namespace=f"a2a:{self.runtime.tenant_key}"
        )
        result = await self.runtime.call(
            "broker",
            "publish_idempotent",
            self._exchange,
            "task",
            payload,
            producer_id=producer_id,
            seq=sequence,
        )
        if not isinstance(result, dict) or set(result) != {
            "confirmed",
            "duplicate",
            "delivered",
        }:
            raise RuntimeError(
                "native A2A idempotent publish returned an invalid result"
            )
        confirmed = result["confirmed"]
        duplicate = result["duplicate"]
        delivered = result["delivered"]
        if (
            type(confirmed) is not bool
            or type(duplicate) is not bool
            or type(delivered) is not int
            or not confirmed
            or (duplicate and delivered != 0)
            or (not duplicate and delivered != 1)
        ):
            raise RuntimeError("native A2A operation was not durably routed once")
        await self.storage.mark_dispatch(task_id, kind, "published")

    async def run_task(self, params: TaskSendParams) -> None:
        await self.runtime.start()
        self._raise_reconciler_failure()
        safe_params = self._safe_run_params(params)
        record, _task = await self.storage.record_for_execution(
            safe_params["id"], safe_params["context_id"]
        )
        # The persisted operation is authoritative for crash recovery. The HTTP
        # bridge's transient copy must resolve to exactly the same reference shape.
        if record["run_operation"] != safe_params:
            raise A2AStorageConflict("A2A run dispatch does not match its task record")
        await self._publish_dispatch(
            safe_params["id"], "run", safe_params, _DISPATCH_RUN
        )

    async def cancel_task(self, params: TaskIdParams) -> None:
        await self.runtime.start()
        self._raise_reconciler_failure()
        validated = cast(
            dict[str, Any],
            _validated_json(
                _TASK_ID_PARAMS_ADAPTER, params, label="A2A cancel parameters"
            ),
        )
        task_id = self.runtime.require_task_id(str(validated.get("id") or ""))
        await self.storage.cancel_task(task_id)
        await self._publish_dispatch(
            task_id, "cancel", {"id": task_id}, _DISPATCH_CANCEL
        )

    async def _reconcile_once(self) -> None:
        pending, cursor = await self.storage.dispatch_page(
            after=self._reconcile_cursor, limit=self.reconcile_limit
        )
        self._reconcile_cursor = cursor if cursor is not None else None
        for task_id, kind, params, sequence in pending:
            await self._publish_dispatch(task_id, kind, params, sequence)

    async def _reconcile_loop(self) -> None:
        while self._active:
            await anyio.sleep(self.reconcile_interval_ms / 1000)
            await self._reconcile_once()

    def _raise_reconciler_failure(self) -> None:
        """Surface a dead crash-recovery loop to the active protocol worker."""

        task = self._reconcile_task
        if not self._active or task is None or not task.done():
            return
        if task.cancelled():
            raise RuntimeError("native A2A dispatch reconciler stopped unexpectedly")
        error = task.exception()
        if error is None:
            raise RuntimeError("native A2A dispatch reconciler stopped unexpectedly")
        raise RuntimeError(
            f"native A2A dispatch reconciliation failed ({type(error).__name__})"
        ) from None

    @staticmethod
    def _delivery_tag(properties: dict[str, Any]) -> int:
        value = properties.get("delivery_tag")
        if type(value) is not int or value <= 0:
            raise RuntimeError("native A2A broker delivery tag is invalid")
        return value

    async def _nack_tag(self, tag: int, *, requeue: bool, allow_absent: bool) -> str:
        outcome = await self.runtime.call(
            "broker",
            "nack_tag",
            tag,
            consumer=self._consumer,
            requeue=requeue,
            now_ms=_now_ms(),
        )
        allowed = {"requeued", "dead-lettered", "dropped"}
        if allow_absent:
            allowed.add("absent")
        if not isinstance(outcome, str) or outcome not in allowed:
            raise RuntimeError("native A2A broker nack returned an invalid result")
        return outcome

    async def _fail_exhausted_delivery(self, binding: _ExecutionBinding) -> None:
        """Resolve an operation whose broker retry budget is exhausted."""

        try:
            await self.storage.update_task(binding.task_id, state="failed")
        except A2AStorageConflict:
            latest = await self.storage.load_task(binding.task_id)
            if latest is not None and latest["status"]["state"] in _TERMINAL_STATES:
                return
            raise

    async def _decode_claim(
        self, properties: dict[str, Any]
    ) -> tuple[TaskOperation, _ExecutionBinding | None, int]:
        tag = self._delivery_tag(properties)
        if properties.get("owner_consumer") != self._consumer:
            raise RuntimeError("native A2A broker returned another consumer's claim")
        raw = properties.get("payload")
        if (
            not isinstance(raw, str)
            or not raw
            or len(raw) % 2
            or len(raw) > self.max_payload_bytes * 2
            or not _LOWER_HEX.fullmatch(raw)
        ):
            raise ValueError("native A2A broker payload is invalid")
        try:
            payload = bytes.fromhex(raw)
            if not payload or len(payload) > self.max_payload_bytes:
                raise ValueError
            envelope = json.loads(payload)
        except (RecursionError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise ValueError("native A2A broker payload is invalid") from None
        _admit_structure(
            envelope,
            maximum=self.max_payload_bytes,
            label="native A2A broker envelope",
        )
        if not isinstance(envelope, dict) or set(envelope) != {
            "schema_version",
            "operation",
            "params",
        }:
            raise ValueError("native A2A broker envelope is invalid")
        if envelope["schema_version"] != _SCHEMA_VERSION:
            raise ValueError("native A2A broker schema version is unsupported")
        operation = envelope["operation"]
        params = envelope["params"]
        if operation not in {"run", "cancel"} or not isinstance(params, dict):
            raise ValueError("native A2A broker operation is invalid")
        binding: _ExecutionBinding | None = None
        if operation == "run":
            params = cast(
                dict[str, Any],
                _validated_json(
                    _TASK_SEND_PARAMS_ADAPTER,
                    params,
                    label="native A2A run parameters",
                ),
            )
            task_id = self.runtime.require_task_id(str(params["id"]))
            context_id = self.runtime.context_id(str(params["context_id"]))
            record, _task = await self.storage.record_for_execution(task_id, context_id)
            if record["run_operation"] != params:
                raise ValueError(
                    "native A2A run operation differs from its task record"
                )
            binding = _ExecutionBinding(
                task_id=task_id,
                context_id=context_id,
                expected_task_revision=record["revision"],
                expected_task_payload_ref=record["payload_ref"],
                expected_context_revision=record["context_revision"],
                expected_context_payload_ref=record["context_payload_ref"],
                delivery_tag=tag,
                consumer=self._consumer,
            )
        else:
            if set(params) != {"id"}:
                raise ValueError("native A2A cancel parameters are invalid")
            params = cast(
                dict[str, Any],
                _validated_json(
                    _TASK_ID_PARAMS_ADAPTER,
                    params,
                    label="native A2A cancel parameters",
                ),
            )
            params["id"] = self.runtime.require_task_id(str(params["id"]))
        operation_value: TaskOperation = cast(
            TaskOperation,
            {
                "operation": operation,
                "params": params,
                "_current_span": get_current_span(),
            },
        )
        return operation_value, binding, tag

    async def _maintain_lease(
        self,
        control: _DeliveryControl,
        binding: _ExecutionBinding | None,
    ) -> None:
        renewal_interval = max(0.05, self.lease_ms / 3_000)
        control_interval = max(0.01, self.cancellation_poll_interval_ms / 1_000)
        interval = (
            min(renewal_interval, control_interval)
            if control.monitor_cancellation
            else renewal_interval
        )
        loop = asyncio.get_running_loop()
        renew_at = loop.time() + renewal_interval
        while not control.stop_event.is_set():
            try:
                await asyncio.wait_for(control.stop_event.wait(), timeout=interval)
                return
            except TimeoutError:
                pass
            try:
                if loop.time() >= renew_at:
                    renewed = await self.runtime.call(
                        "broker",
                        "renew_tag",
                        control.delivery_tag,
                        consumer=control.consumer,
                        now_ms=_now_ms(),
                        lease_ms=self.lease_ms,
                    )
                    if type(renewed) is not bool or not renewed:
                        control.abort("lease_lost")
                        return
                    renew_at = loop.time() + renewal_interval
                if control.monitor_cancellation and binding is not None:
                    state = await self.storage.execution_control_state(binding)
                    if state == "canceled":
                        control.abort("task_canceled")
                        return
                    if state == "terminal":
                        control.abort("task_terminal")
                        return
                    if state == "lost":
                        control.abort("lease_lost")
                        return
            except Exception:
                control.abort("lease_lost")
                return

    async def receive_task_operations(self) -> AsyncGenerator[TaskOperation, None]:
        if not self._active:
            raise RuntimeError("native A2A broker is not entered")
        while self._active:
            self._raise_reconciler_failure()
            claimed = await self.runtime.call(
                "broker",
                "consume",
                self._queue,
                group="a2a-workers",
                consumer=self._consumer,
                now_ms=_now_ms(),
                lease_ms=self.lease_ms,
                prefetch=self.prefetch,
            )
            if claimed is None:
                await anyio.sleep(self.poll_interval_ms / 1000)
                continue
            if (
                not isinstance(claimed, (tuple, list))
                or len(claimed) != 2
                or not isinstance(claimed[0], str)
                or not isinstance(claimed[1], dict)
            ):
                raise RuntimeError(
                    "native A2A broker returned an invalid consume tuple"
                )
            _node_id, properties = claimed
            tag = self._delivery_tag(properties)
            delivery_count = properties.get("delivery_count")
            if type(delivery_count) is not int or delivery_count <= 0:
                raise RuntimeError("native A2A broker delivery count is invalid")
            exhausts_retries = delivery_count >= self.max_delivery_count
            try:
                task_operation, binding, tag = await self._decode_claim(properties)
            except ValueError:
                await self._nack_tag(tag, requeue=False, allow_absent=False)
                continue

            control = _DeliveryControl(
                task_id=str(task_operation["params"]["id"]),
                delivery_tag=tag,
                consumer=self._consumer,
                monitor_cancellation=task_operation["operation"] == "run",
            )
            binding_token: Token[_ExecutionBinding | None] | None = None
            if binding is not None:
                binding_token = _EXECUTION_BINDING.set(binding)
            control_token = _DELIVERY_CONTROL.set(control)
            heartbeat = asyncio.create_task(
                self._maintain_lease(control, binding),
                name="a2a-delivery-heartbeat",
            )
            try:
                yield task_operation
            except BaseException:
                failed = False
                if binding is not None and exhausts_retries:
                    await self._fail_exhausted_delivery(binding)
                    failed = True
                outcome = await self._nack_tag(
                    tag,
                    requeue=True,
                    allow_absent=control.abort_reason == "lease_lost",
                )
                if (
                    binding is not None
                    and not failed
                    and outcome in {"dead-lettered", "dropped"}
                ):
                    await self._fail_exhausted_delivery(binding)
                raise
            else:
                if control.abort_reason == "lease_lost":
                    await self._nack_tag(tag, requeue=True, allow_absent=True)
                    raise _A2ADeliveryRetry("A2A delivery lease was lost")
                acknowledged = await self.runtime.call(
                    "broker", "ack_tag", tag, consumer=self._consumer
                )
                if type(acknowledged) is not bool or not acknowledged:
                    raise RuntimeError(
                        "native A2A broker could not acknowledge its fenced delivery"
                    )
            finally:
                control.stop_event.set()
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat
                try:
                    _DELIVERY_CONTROL.reset(control_token)
                except ValueError:
                    # Async-generator shutdown can be driven by the event loop's
                    # finalizer context after its owning task has already ended.
                    _DELIVERY_CONTROL.set(None)
                if binding_token is not None:
                    try:
                        _EXECUTION_BINDING.reset(binding_token)
                    except ValueError:
                        _EXECUTION_BINDING.set(None)


class EpistemicGraphAgentWorker(AgentWorker):
    """Current FastA2A worker with fenced completion and cross-process cancel."""

    broker: EpistemicGraphA2ABroker
    storage: EpistemicGraphA2AStorage

    async def _mark_failed(self, task_id: str) -> None:
        try:
            await self.storage.update_task(task_id, state="failed")
        except A2AStorageConflict:
            latest = await self.storage.load_task(task_id)
            if latest is not None and latest["status"]["state"] in _TERMINAL_STATES:
                return
            raise

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params["id"])
        if task is None:
            raise ValueError("A2A task is unavailable")
        if task["status"]["state"] in _TERMINAL_STATES:
            return
        if task["status"]["state"] not in {"submitted", "working"}:
            raise A2AStorageConflict("A2A task is not executable")
        await self.storage.update_task(task["id"], state="working")
        try:
            message_history = await self.storage.load_context(task["context_id"]) or []
            message_history.extend(self.build_message_history(task.get("history", [])))
            try:
                result = await self.agent.run(message_history=message_history)
            except Exception:
                await self._mark_failed(task["id"])
                return

            a2a_messages: list[Message] = []
            for message in result.new_messages():
                from pydantic_ai.messages import ModelRequest

                if isinstance(message, ModelRequest):
                    continue
                a2a_parts = self._response_parts_to_a2a(message.parts)
                if a2a_parts:
                    a2a_messages.append(
                        Message(
                            role="agent",
                            parts=a2a_parts,
                            kind="message",
                            message_id=str(uuid.uuid4()),
                        )
                    )
            artifacts = self.build_artifacts(result.output)
            try:
                await self.storage.complete_task(
                    task["id"],
                    result.all_messages(),
                    new_artifacts=artifacts,
                    new_messages=a2a_messages,
                )
            except ValueError:
                await self._mark_failed(task["id"])
        except asyncio.CancelledError:
            raise
        except A2AStorageConflict:
            latest = await self.storage.load_task(task["id"])
            if latest is not None and latest["status"]["state"] in _TERMINAL_STATES:
                return
            raise

    async def cancel_task(self, params: TaskIdParams) -> None:
        try:
            await self.storage.cancel_task(params["id"])
        except A2AStorageConflict:
            latest = await self.storage.load_task(params["id"])
            if latest is not None and latest["status"]["state"] in _TERMINAL_STATES:
                return
            raise

    def _request_parts_from_a2a(self, parts: list[Any]) -> list[Any]:
        model_parts: list[Any] = []
        for part in parts:
            if isinstance(part, dict) and part.get("kind") == "file":
                file_value = part.get("file")
                reference = (
                    str(file_value.get("uri") or "")
                    if isinstance(file_value, dict)
                    else ""
                )
                if not _GOVERNED_CONTENT_REF.fullmatch(reference):
                    raise ValueError(
                        "A2A file input is not a governed content reference"
                    )
                model_parts.append(
                    UserPromptPart(content=f"Governed content reference: {reference}")
                )
            else:
                model_parts.extend(super()._request_parts_from_a2a([part]))
        return model_parts

    async def _handle_task_operation(self, task_operation: TaskOperation) -> None:
        with use_span(task_operation["_current_span"]):
            with _tracer.start_as_current_span(
                f"{task_operation['operation']} task",
                attributes={"logfire.tags": ["fasta2a"]},
            ):
                if task_operation["operation"] == "run":
                    await self.run_task(task_operation["params"])
                elif task_operation["operation"] == "cancel":
                    await self.cancel_task(task_operation["params"])
                else:
                    raise RuntimeError(
                        "native A2A worker received an invalid operation"
                    )

    async def _loop(self) -> None:
        iterator = self.broker.receive_task_operations()
        active_handler: asyncio.Task[None] | None = None
        abort_wait: asyncio.Task[bool] | None = None
        try:
            while True:
                task_operation = await anext(iterator)
                control = _DELIVERY_CONTROL.get()
                if control is None:
                    raise RuntimeError("native A2A delivery control is unavailable")
                active_handler = asyncio.create_task(
                    self._handle_task_operation(task_operation),
                    name="a2a-task-handler",
                )
                abort_wait = asyncio.create_task(
                    control.abort_event.wait(), name="a2a-delivery-abort-wait"
                )
                done, _pending = await asyncio.wait(
                    {active_handler, abort_wait}, return_when=asyncio.FIRST_COMPLETED
                )
                if abort_wait in done and control.abort_reason:
                    active_handler.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await active_handler
                    active_handler = None
                    abort_wait = None
                    if control.abort_reason in {"task_canceled", "task_terminal"}:
                        continue
                    try:
                        await iterator.athrow(
                            _A2ADeliveryRetry("A2A delivery lease was lost")
                        )
                    except _A2ADeliveryRetry:
                        pass
                    iterator = self.broker.receive_task_operations()
                    continue
                abort_wait.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await abort_wait
                abort_wait = None
                exception = active_handler.exception()
                active_handler = None
                if exception is not None:
                    try:
                        await iterator.athrow(
                            _A2ADeliveryRetry("A2A task handler did not commit")
                        )
                    except _A2ADeliveryRetry:
                        pass
                    iterator = self.broker.receive_task_operations()
        finally:
            for task in (abort_wait, active_handler):
                if task is not None and not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            with contextlib.suppress(BaseException):
                await iterator.aclose()


def agent_to_epistemic_a2a(
    agent: Any,
    *,
    broker: EpistemicGraphA2ABroker,
    storage: EpistemicGraphA2AStorage,
    name: str,
    description: str,
    version: str,
    skills: Sequence[Skill],
    debug: bool = False,
) -> Any:
    """Build the current FastA2A app with the fenced native worker."""

    from fasta2a.applications import FastA2A

    worker = EpistemicGraphAgentWorker(agent=agent, broker=broker, storage=storage)

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        async with app.task_manager, agent:
            async with worker.run():
                yield

    return FastA2A(
        storage=storage,
        broker=broker,
        name=name,
        description=description,
        version=version,
        skills=list(skills),
        debug=debug,
        lifespan=lifespan,
    )


def build_epistemic_graph_a2a_backends(
    config: Any,
) -> tuple[EpistemicGraphA2ABroker, EpistemicGraphA2AStorage]:
    """Build the sole current FastA2A persistence pair from ``AgentConfig``."""

    if (
        config.a2a_broker != "epistemic_graph"
        or config.a2a_storage != "epistemic_graph"
    ):
        raise ValueError(
            "A2A_BROKER and A2A_STORAGE must both select 'epistemic_graph'"
        )
    runtime = EpistemicGraphA2ARuntime()
    storage = EpistemicGraphA2AStorage(
        runtime,
        max_payload_bytes=config.a2a_max_payload_bytes,
        max_history=config.a2a_max_history,
        max_artifacts=config.a2a_max_artifacts,
        max_context_messages=config.a2a_max_context_messages,
        update_retries=config.a2a_storage_update_retries,
    )
    broker = EpistemicGraphA2ABroker(
        runtime,
        storage,
        poll_interval_ms=config.a2a_broker_poll_interval_ms,
        lease_ms=config.a2a_broker_lease_ms,
        prefetch=config.a2a_broker_prefetch,
        max_payload_bytes=config.a2a_max_payload_bytes,
        message_ttl_ms=config.a2a_broker_message_ttl_ms,
        max_delivery_count=config.a2a_broker_max_delivery_count,
        reconcile_interval_ms=config.a2a_dispatch_reconcile_interval_ms,
        reconcile_limit=config.a2a_dispatch_reconcile_limit,
        cancellation_poll_interval_ms=config.a2a_cancellation_poll_interval_ms,
    )
    return broker, storage
