"""Concrete EG- and AU-backed implementations of the agent-control ports.

The control plane (:mod:`agent_utilities.api.agent_control_plane`) only knows
typed ports. This module supplies the concrete adapters a composition root
binds to them:

* :class:`EgWorkItemStore` -- the ``WorkItemStorePort`` over EG's native,
  engine-owned WorkItem command log (``SubmitWorkItem``, ``CancelWorkItem``,
  and the session-authorized ``GetWorkItem``/``ListWorkItems`` reads).
* :class:`EgCapabilitySearch` -- the ``CapabilitySearchPort`` over EG's typed
  ``AgentComponent.Search`` capability query.
* :class:`OrchestratorAgentExecutor` -- the ``AgentExecutionPort`` over AU's
  own agent runtime (``Orchestrator.execute_agent``).

Every adapter derives tenant, graph and principal from the verified
:class:`GraphSession`; no caller-supplied authority reaches EG. Payloads are
privacy-sanitized before they are digested or sent, and the engine's
created/replayed idempotency outcome is surfaced verbatim. An adapter whose EG
method is not served by the connected engine fails closed with
:class:`AgentControlPlaneUnavailable`; it never reconstructs the operation
from raw queries or a local store.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, get_args

from pydantic import JsonValue

from agent_utilities.api.agent_control_contracts import (
    AgentControlPlaneUnavailable,
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentWorkItemNotCancelable,
    CapabilityCandidate,
    CapabilityKind,
    CapabilitySearchRequest,
    WorkItemCancelRequest,
    WorkItemGetRequest,
    WorkItemListRequest,
    WorkItemPage,
    WorkItemSnapshot,
    WorkItemStatus,
    WorkItemSubmission,
    WorkItemSubmissionResult,
)
from agent_utilities.api.session import GraphSession, use_session

if TYPE_CHECKING:
    from agent_utilities.api.agent_control_contracts import SignedAgentDispatchPort
    from agent_utilities.api.agent_control_plane import AgentControlPlane

AuthenticationMethod = Literal[
    "workload_identity", "oidc", "mutual_tls", "local_process"
]

#: Reserved metadata keys the adapter owns inside EG's WorkItem metadata.
#: A caller may not supply them; they carry the sanitized task body (EG
#: persists no free-form body outside bounded metadata) and its digest.
DESCRIPTION_METADATA_KEY = "au:description"
PAYLOAD_DIGEST_METADATA_KEY = "au:payload_digest"
_RESERVED_METADATA_PREFIX = "au:"

#: EG's native metadata bound (``MAX_SUBMIT_METADATA_BYTES``); checked here so
#: an oversize body fails with a typed AU error before any engine round trip.
MAX_WORK_ITEM_METADATA_BYTES = 64 * 1024

_WORK_ITEM_STATUSES: frozenset[str] = frozenset(get_args(WorkItemStatus))

_COMPONENT_KIND_TO_CAPABILITY: dict[str, CapabilityKind] = {
    "skill": "skill",
    "a2a_agent_card": "agent",
}


class WorkItemIdempotencyConflict(ValueError):
    """An idempotency key was reused with a different sanitized payload."""


class WorkItemPayloadTooLarge(ValueError):
    """The sanitized WorkItem payload exceeds EG's native metadata bound."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _session_graph(session: GraphSession) -> str:
    graph = str(session.graph or "").strip()
    if not graph:
        raise AgentControlPlaneUnavailable(
            "the verified session is not bound to a graph"
        )
    return graph


def _is_idempotency_conflict(exc: BaseException) -> bool:
    return str(exc).startswith("IDEMPOTENCY_CONFLICT")


def sanitize_work_item_payload(
    description: str, metadata: Mapping[str, JsonValue]
) -> tuple[str, dict[str, JsonValue]]:
    """Privacy-sanitize a WorkItem body and metadata before persistence."""
    from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

    guard = PersistencePrivacyGuard()
    clean_description, _report = guard.sanitize_text(description)
    clean_metadata, _metadata_report = guard.sanitize(dict(metadata))
    if not isinstance(clean_metadata, dict):
        raise AgentControlPlaneUnavailable("WorkItem metadata sanitization failed")
    return clean_description, clean_metadata


def _reject_reserved_metadata(metadata: Mapping[str, Any]) -> None:
    if any(str(key).startswith(_RESERVED_METADATA_PREFIX) for key in metadata):
        raise ValueError("WorkItem metadata keys may not use the reserved 'au:' prefix")


def _key_sorted(value: Any) -> Any:
    """Recursively key-sort mappings, matching a Rust ``BTreeMap`` encoding."""
    if isinstance(value, Mapping):
        return {str(key): _key_sorted(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [_key_sorted(item) for item in value]
    return value


def _bounded_metadata(metadata: dict[str, JsonValue]) -> dict[str, JsonValue]:
    import msgpack

    if len(msgpack.packb(metadata, use_bin_type=True)) > MAX_WORK_ITEM_METADATA_BYTES:
        raise WorkItemPayloadTooLarge(
            "the sanitized WorkItem payload exceeds the 64 KiB native bound"
        )
    return metadata


def request_context_for(
    session: GraphSession,
    authentication_method: AuthenticationMethod,
) -> dict[str, Any]:
    """The EG ``RequestContext`` v2 derived only from the verified session.

    EG binds this durable-provenance context to the verified carrier: tenant,
    agent, audience and policy must equal the carrier and every scope must be
    one the carrier already allows, so the claims are copied, never widened.
    """
    claims = session.engine_verified_context()
    now_ms = int(time.time() * 1000)
    lease = getattr(session.actor, "credential_lease", None)
    expiry = (
        lease.expires_at
        if lease is not None
        else getattr(session.actor, "credential_expires_at", None)
    )
    expires_at_ms = max(now_ms, int(expiry) * 1000) if expiry is not None else now_ms
    return {
        "schema_version": "2",
        "request_id": f"au-req:{uuid.uuid4().hex}",
        "subject_id": str(claims["principal"]),
        "tenant_id": str(claims["tenant"]),
        "agent_id": str(claims["agent_id"]),
        "scopes": list(claims["scopes"]),
        "audience": str(claims["audience"]),
        "authentication_method": authentication_method,
        "policy_version": str(claims["policy_version"]),
        "graph": _session_graph(session),
        "placement_epoch": None,
        "trace_id": str(session.trace_context or f"au-trace:{uuid.uuid4().hex}"),
        "issued_at_ms": now_ms,
        "expires_at_ms": expires_at_ms,
    }


# ---------------------------------------------------------------------------
# WorkItem store
# ---------------------------------------------------------------------------


def _snapshot_status(value: Any) -> WorkItemStatus:
    status = str(value or "")
    if status not in _WORK_ITEM_STATUSES:
        raise AgentControlPlaneUnavailable(
            "the WorkItem authority returned an unknown status"
        )
    return cast(WorkItemStatus, status)


def _public_metadata(metadata: Mapping[str, Any]) -> dict[str, JsonValue]:
    return {
        str(key): value
        for key, value in metadata.items()
        if not str(key).startswith(_RESERVED_METADATA_PREFIX)
    }


def snapshot_from_row(row: Mapping[str, Any]) -> WorkItemSnapshot:
    """Project one EG ``GetWorkItem``/``ListWorkItems`` row to the AU view."""
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return WorkItemSnapshot(
        work_item_id=str(row["work_item_id"]),
        kind=str(row["kind"]),
        status=_snapshot_status(row.get("status")),
        payload_ref=str(row.get("input_ref") or ""),
        description=str(metadata.get(DESCRIPTION_METADATA_KEY) or ""),
        metadata=_public_metadata(metadata),
        version=int(row["version"]),
        updated_at_ms=int(row["updated_at_ms"]),
    )


class _WorkItemCommands(Protocol):
    async def submit(self, request: dict[str, Any]) -> dict[str, Any]: ...

    async def cancel(
        self,
        *,
        tenant: str,
        work_item_id: str,
        idempotency_key: str,
        now_ms: int,
        reason_ref: str | None = None,
    ) -> dict[str, Any]: ...


class EgWorkItemStore:
    """``WorkItemStorePort`` over EG's native, tenant-scoped WorkItem authority."""

    def __init__(
        self,
        eg_client: Any,
        *,
        authentication_method: AuthenticationMethod,
        policy_digest: str,
        catalog_digest: str,
        model_digest: str,
        graph: str | None = None,
    ) -> None:
        """``eg_client`` must route to ``graph`` (the WorkItem authority graph);
        ``None`` means the session's own graph."""
        if eg_client is None:
            raise ValueError("an epistemic-graph client is required")
        for label, digest in (
            ("policy", policy_digest),
            ("catalog", catalog_digest),
            ("model", model_digest),
        ):
            if not str(digest).strip():
                raise ValueError(f"a non-empty {label} digest is required")
        self._client = eg_client
        self._graph = graph
        self._authentication_method = authentication_method
        self._digests = (policy_digest, catalog_digest, model_digest)

    def _work_session(self, session: GraphSession) -> GraphSession:
        """``session`` retargeted onto the WorkItem graph (identity unchanged).

        A graph-scoped client refuses an ambient session bound to another
        graph, so every command runs under this narrowed session.
        """
        if self._graph and self._graph != session.graph:
            return session.with_graph(self._graph)
        return session

    @property
    def _commands(self) -> _WorkItemCommands:
        commands = getattr(self._client, "work_items", None)
        if commands is None:
            raise AgentControlPlaneUnavailable(
                "the epistemic-graph client has no native WorkItem namespace"
            )
        return cast(_WorkItemCommands, commands)

    def _submit_request(
        self, request: WorkItemSubmission, session: GraphSession
    ) -> tuple[dict[str, Any], str]:
        _reject_reserved_metadata(request.metadata)
        description, metadata = sanitize_work_item_payload(
            request.description, request.metadata
        )
        payload_digest = _canonical_digest(
            {"kind": request.kind, "description": description, "metadata": metadata}
        )
        metadata = _bounded_metadata(
            {
                **metadata,
                DESCRIPTION_METADATA_KEY: description,
                PAYLOAD_DIGEST_METADATA_KEY: payload_digest,
            }
        )
        policy_digest, catalog_digest, model_digest = self._digests
        command_digest = _canonical_digest(
            {
                "work_item_id": request.work_item_id,
                "kind": request.kind,
                "priority": request.priority,
                "max_attempts": request.max_attempts,
                "deadline_unix": request.deadline_unix,
                "payload": payload_digest,
            }
        )
        # Rust declaration order of ``SubmitWorkItemRequest``, with metadata
        # key-sorted like its ``BTreeMap``: the ``eg2.`` envelope MAC covers
        # the server's re-serialization of the typed request, so any other
        # order fails authentication (EG CONTRACT-REQUEST R5).
        body = {
            "schema_version": "1",
            "context": request_context_for(session, self._authentication_method),
            "work_item_id": request.work_item_id,
            "idempotency_key": request.idempotency_key,
            "command_digest": command_digest,
            "kind": request.kind,
            "priority": request.priority,
            "depends_on": [],
            "input_ref": f"au-task:sha256:{payload_digest}",
            "policy_digest": policy_digest,
            "catalog_digest": catalog_digest,
            "model_digest": model_digest,
            "max_attempts": request.max_attempts,
            "deadline_unix": request.deadline_unix,
            "metadata": _key_sorted(metadata),
            "provenance_refs": [],
            "max_tenant_in_flight": 0,
        }
        return body, description

    async def submit(
        self, request: WorkItemSubmission, *, session: GraphSession
    ) -> WorkItemSubmissionResult:
        work_session = self._work_session(session)
        body, description = self._submit_request(request, work_session)
        try:
            with use_session(work_session):
                result = await self._commands.submit(body)
        except RuntimeError as exc:
            if _is_idempotency_conflict(exc):
                raise WorkItemIdempotencyConflict(
                    "the idempotency key was already used for a different payload"
                ) from exc
            raise
        if result["work_item_id"] != request.work_item_id:
            raise AgentControlPlaneUnavailable(
                "the WorkItem authority admitted an unrelated work item"
            )
        if result["replayed"]:
            replay = await self.get(
                WorkItemGetRequest(work_item_id=request.work_item_id), session=session
            )
            if replay is None:
                raise AgentControlPlaneUnavailable(
                    "a replayed WorkItem admission is not readable"
                )
            return WorkItemSubmissionResult(item=replay, created=False, replayed=True)
        item = WorkItemSnapshot(
            work_item_id=result["work_item_id"],
            kind=request.kind,
            status=_snapshot_status(result["status"]),
            payload_ref=body["input_ref"],
            description=description,
            metadata=_public_metadata(body["metadata"]),
            version=1,
            updated_at_ms=body["context"]["issued_at_ms"],
        )
        return WorkItemSubmissionResult(item=item, created=True, replayed=False)

    async def _read_method(self, method: str) -> Any:
        supports = getattr(self._client, "supports", None)
        reader = getattr(
            self._commands, "get" if method == "GetWorkItem" else "list", None
        )
        if supports is None or reader is None or await supports(method) is not True:
            raise AgentControlPlaneUnavailable(
                f"the connected epistemic-graph does not serve {method}"
            )
        return reader

    async def get(
        self, request: WorkItemGetRequest, *, session: GraphSession
    ) -> WorkItemSnapshot | None:
        with use_session(self._work_session(session)):
            reader = await self._read_method("GetWorkItem")
            row = await reader(tenant=session.tenant, work_item_id=request.work_item_id)
        return None if row is None else snapshot_from_row(row)

    async def list(
        self, request: WorkItemListRequest, *, session: GraphSession
    ) -> WorkItemPage:
        with use_session(self._work_session(session)):
            reader = await self._read_method("ListWorkItems")
            page = await reader(
                tenant=session.tenant,
                cursor=request.cursor,
                limit=request.limit,
                kind=request.kind,
            )
        return WorkItemPage(
            items=tuple(snapshot_from_row(row) for row in page["items"]),
            next_cursor=page.get("next_cursor"),
        )

    async def cancel(
        self, request: WorkItemCancelRequest, *, session: GraphSession
    ) -> WorkItemSnapshot | None:
        with use_session(self._work_session(session)):
            transition = await self._commands.cancel(
                tenant=session.tenant,
                work_item_id=request.work_item_id,
                idempotency_key=f"au-cancel:{request.work_item_id}:{request.reason}",
                now_ms=int(time.time() * 1000),
                reason_ref=f"au-cancel-reason:{request.reason}",
            )
        status = str(transition.get("status") or "")
        if status == "missing":
            return None
        if status in {"in_flight", "not_cancellable"}:
            raise AgentWorkItemNotCancelable(
                "the WorkItem cannot be cancelled in its current state"
            )
        if status not in {"cancelled", "noop"}:
            raise AgentControlPlaneUnavailable(
                "the WorkItem authority returned an unknown cancel status"
            )
        return await self.get(
            WorkItemGetRequest(work_item_id=request.work_item_id), session=session
        )


# ---------------------------------------------------------------------------
# Capability search
# ---------------------------------------------------------------------------


def _candidate_name(entry: Any) -> str:
    attributes = entry.attributes or {}
    return str(attributes.get("name") or entry.component_id)


def candidates_from_entries(entries: Sequence[Any]) -> tuple[CapabilityCandidate, ...]:
    """Project EG-ranked component entries to AU capability candidates.

    EG returns an ordered page, not similarity scores; the score is the
    entry's ordinal rank mapped into ``(0, 1]`` so EG's order is preserved.
    """
    usable = [
        entry
        for entry in entries
        if _COMPONENT_KIND_TO_CAPABILITY.get(getattr(entry.kind, "value", entry.kind))
    ]
    total = len(usable)
    return tuple(
        CapabilityCandidate(
            kind=_COMPONENT_KIND_TO_CAPABILITY[
                getattr(entry.kind, "value", entry.kind)
            ],
            name=_candidate_name(entry),
            component_id=entry.component_id,
            score=(total - rank) / total,
            source="eg_agent_component",
        )
        for rank, entry in enumerate(usable)
    )


#: Bound on catalog pages walked to resolve an explicitly named agent.
MAX_NAME_LOOKUP_PAGES = 16


class EgCapabilitySearch:
    """``CapabilitySearchPort`` over EG's typed ``AgentComponent.Search``.

    Free text never reaches EG as a task term. A request carrying a typed
    ``task_iri`` (one of EG's five native task terms) is searched through the
    ontology; a request naming an agent is resolved by walking the bounded,
    kind-scoped catalog; anything else has no authorized match.
    """

    def __init__(self, eg_client: Any) -> None:
        if eg_client is None:
            raise ValueError("an epistemic-graph client is required")
        self._client = eg_client

    async def _page(
        self,
        session: GraphSession,
        *,
        task_iri: str | None,
        limit: int,
        cursor: str | None = None,
    ) -> Any:
        from epistemic_graph.generated.agent_component import (
            AgentComponentKind,
            AgentComponentSearchRequest,
        )
        from epistemic_graph.generated.storage import send_agent_component_search

        typed = AgentComponentSearchRequest(
            tenant_id=session.tenant,
            task=task_iri,
            kinds=[AgentComponentKind.SKILL, AgentComponentKind.A2A_AGENT_CARD],
            limit=limit,
            cursor=cursor,
        )
        try:
            return await send_agent_component_search(
                self._client, typed, _session_graph(session)
            )
        except ValueError as exc:
            raise AgentControlPlaneUnavailable(
                "the epistemic-graph client does not serve typed task-capability search"
            ) from exc

    async def _by_task(
        self, task_iri: str, limit: int, session: GraphSession
    ) -> tuple[CapabilityCandidate, ...]:
        page = await self._page(session, task_iri=task_iri, limit=limit)
        return candidates_from_entries(page.entries)

    async def _by_name(
        self, name: str | None, session: GraphSession
    ) -> tuple[CapabilityCandidate, ...]:
        """Walk the bounded catalog for ``name``; no name has no match."""
        cursor: str | None = None
        pages = MAX_NAME_LOOKUP_PAGES if name is not None else 0
        for _page_index in range(pages):
            page = await self._page(session, task_iri=None, limit=256, cursor=cursor)
            found = tuple(
                candidate
                for candidate in candidates_from_entries(page.entries)
                if candidate.name == name
            )
            if found or page.next_cursor is None:
                return found
            cursor = page.next_cursor
        return ()

    async def search(
        self, request: CapabilitySearchRequest, *, session: GraphSession
    ) -> Sequence[CapabilityCandidate]:
        if request.task_iri is not None:
            return await self._by_task(request.task_iri, request.limit, session)
        return await self._by_name(request.agent_name, session)


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------


def _execution_options(request: AgentExecutionRequest, run_id: str) -> dict[str, Any]:
    return {
        "max_steps": request.max_steps,
        "return_mermaid": request.return_mermaid,
        "context": request.context,
        "budget_tokens": request.budget_tokens,
        "context_ref": request.context_ref,
        "allowed_tools": _optional_list(request.allowed_tools),
        "required_tools": _optional_list(request.required_tools),
        "cred_ref": request.credential_ref,
        "session_id": request.session_ref,
        "open_channel": request.open_channel,
        "memento_source": request.memento_source,
        "execution_profile": request.execution_profile,
        "reasoning_effort": request.reasoning_effort,
        "model_class": request.model_class,
        "response_format": request.response_format,
        "run_id": run_id,
        "include_run_summary": request.include_run_summary,
        "skill_name": request.skill_name,
        "tool_server": request.tool_server,
        "execution_mode": request.execution_mode,
        "grounding": request.grounding,
    }


def _optional_list(values: tuple[str, ...] | None) -> list[str] | None:
    return None if values is None else list(values)


class OrchestratorAgentExecutor:
    """``AgentExecutionPort`` over AU's own agent runtime.

    The verified session is bound as the ambient graph authority for the whole
    run, so every graph read/write the agent makes is attributed to, and
    authorized as, the caller -- never a process identity.
    """

    def __init__(self, runner: Any) -> None:
        if not callable(getattr(runner, "execute_agent", None)):
            raise TypeError("an AU agent runner with execute_agent is required")
        self._runner = runner

    async def execute_agent(
        self, request: AgentExecutionRequest, *, session: GraphSession
    ) -> AgentExecutionResult:
        run_id = request.run_id or f"run-{uuid.uuid4().hex}"
        with use_session(session):
            output = await self._runner.execute_agent(
                request.agent_name,
                request.task,
                **_execution_options(request, run_id),
            )
        mode = None if request.execution_mode == "auto" else request.execution_mode
        return AgentExecutionResult(
            run_id=run_id, output=str(output), execution_mode=mode
        )


def compose_eg_agent_control_plane(
    eg_client: Any,
    session: GraphSession,
    *,
    runner: Any,
    authentication_method: AuthenticationMethod,
    policy_digest: str,
    catalog_digest: str,
    model_digest: str,
    signed_dispatch: SignedAgentDispatchPort | None = None,
) -> AgentControlPlane:
    """Compose the control plane with the concrete EG and AU adapters.

    ``runner`` is AU's agent runtime (an ``Orchestrator`` bound to the
    process engine); ``signed_dispatch`` stays an explicit injection because
    queue signing belongs to the hosting process.
    """
    from agent_utilities.api.agent_control_plane import compose_agent_control_plane

    return compose_agent_control_plane(
        eg_client,
        session,
        capability_search=EgCapabilitySearch(eg_client),
        agent_executor=OrchestratorAgentExecutor(runner),
        work_item_store=EgWorkItemStore(
            eg_client,
            authentication_method=authentication_method,
            policy_digest=policy_digest,
            catalog_digest=catalog_digest,
            model_digest=model_digest,
        ),
        signed_dispatch=signed_dispatch,
    )


__all__ = [
    "DESCRIPTION_METADATA_KEY",
    "MAX_WORK_ITEM_METADATA_BYTES",
    "PAYLOAD_DIGEST_METADATA_KEY",
    "AuthenticationMethod",
    "EgCapabilitySearch",
    "EgWorkItemStore",
    "OrchestratorAgentExecutor",
    "WorkItemIdempotencyConflict",
    "WorkItemPayloadTooLarge",
    "candidates_from_entries",
    "compose_eg_agent_control_plane",
    "request_context_for",
    "sanitize_work_item_payload",
    "snapshot_from_row",
]
