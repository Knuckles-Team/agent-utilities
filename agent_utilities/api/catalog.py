"""Authorized agent and workflow catalog read ports.

The graph is the authority for both catalog families.  This module is the
small AU application adapter consumed by other control-plane processes: it
does not define a second store, open the backend directly, or manufacture a
catalog from a cache.  Every read inherits the verified ambient
``GraphSession`` and goes through ``IntelligenceGraphEngine.query_cypher`` so
tenant, visibility, classification, ACL, and read-audit enforcement remain in
the existing graph read path.

The asynchronous methods are the stable cross-process composition surface.
The explicitly named ``*_sync`` methods are useful to AU's synchronous
application callers and are kept on the same authority object so both forms
have exactly the same authorization and projection behavior.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from agent_utilities.knowledge_graph.core.session import GraphSession, resolve_session

CatalogStatus = Literal["active", "retired", "withdrawn"]


class CatalogReadError(RuntimeError):
    """Raised when an authoritative catalog row cannot be projected safely."""


@dataclass(frozen=True, slots=True)
class WorkflowCatalogRecord:
    """The AU-owned current workflow-definition projection.

    ``definition_digest`` projects the source ``WorkflowDefinition.content_hash``
    as a content-addressed ``sha256:...`` value.
    """

    workflow_id: str
    name: str
    description: str
    status: CatalogStatus
    revision: int
    definition_digest: str


@dataclass(frozen=True, slots=True)
class AgentCatalogRecord:
    """An authorized current agent-definition projection."""

    agent_id: str
    name: str
    description: str
    system_prompt: str | None = None
    tools: tuple[str, ...] | None = None


@runtime_checkable
class WorkflowCatalogReadPort(Protocol):
    """Async read port GraphOS uses for the AU workflow catalog."""

    async def list_current_workflows(self) -> Sequence[WorkflowCatalogRecord]: ...


@runtime_checkable
class AgentCatalogReadPort(Protocol):
    """Async read port GraphOS uses for authorized AU agents."""

    async def list_authorized_agents(self) -> Sequence[AgentCatalogRecord]: ...


_AGENT_CATALOG_QUERY = """
MATCH (a:Agent)
OPTIONAL MATCH (a)-[:USES|USES_TOOL]->(t)
RETURN a.id AS id,
       a.agent_id AS agent_id,
       a.name AS name,
       a.description AS description,
       a.system_prompt AS system_prompt,
       a.status AS status,
       t.id AS tool_id,
       t.name AS tool_name
ORDER BY a.name, a.id, t.name
""".strip()

_WORKFLOW_CATALOG_QUERY = """
MATCH (w:WorkflowDefinition)
RETURN w.id AS id,
       w.name AS name,
       w.description AS description,
       w.status AS status,
       w.revision AS revision,
       w.version AS version,
       w.content_hash AS content_hash
ORDER BY w.name, w.id
""".strip()

_ACTIVE_STATUS_VALUES = frozenset({"active", "enabled", "published"})


def _row_dict(row: Any, *, kind: str) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise CatalogReadError(f"{kind} catalog returned a non-mapping row")
    return dict(row)


def _text(value: Any, *, field: str, identity: str, required: bool = False) -> str:
    if value is None:
        if required:
            raise CatalogReadError(f"{identity} catalog row has no {field}")
        return ""
    if not isinstance(value, str):
        raise CatalogReadError(f"{identity} catalog field {field} is not text")
    result = value.strip()
    if required and not result:
        raise CatalogReadError(f"{identity} catalog row has an empty {field}")
    return result


def _status(value: Any, *, identity: str) -> CatalogStatus:
    """Normalize legacy publication spellings without accepting unknown state."""

    if value is None:
        return "active"
    if not isinstance(value, str):
        raise CatalogReadError(f"{identity} catalog status is not text")
    normalized = value.strip().casefold()
    if not normalized:
        return "active"
    if normalized in _ACTIVE_STATUS_VALUES:
        return "active"
    if normalized == "retired":
        return "retired"
    if normalized == "withdrawn":
        return "withdrawn"
    raise CatalogReadError(f"{identity} catalog returned unsupported status {value!r}")


def _revision(row: Mapping[str, Any], *, identity: str) -> int:
    value = row.get("revision")
    if value is None:
        value = row.get("version")
    if value is None:
        raise CatalogReadError(f"{identity} catalog row has no revision")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CatalogReadError(f"{identity} catalog revision is invalid")
    return value


def _digest_text(value: Any, *, identity: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CatalogReadError(f"{identity} catalog digest is not text")
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.startswith("sha256:"):
        hex_digest = normalized.removeprefix("sha256:")
    else:
        hex_digest = normalized
    if len(hex_digest) != 64 or any(
        c not in "0123456789abcdefABCDEF" for c in hex_digest
    ):
        raise CatalogReadError(
            f"{identity} catalog digest is not a sha256 content digest"
        )
    return f"sha256:{hex_digest.lower()}"


@dataclass(slots=True)
class _AgentAccumulator:
    agent_id: str
    name: str
    description: str
    system_prompt: str | None
    tools: set[str]


@dataclass(frozen=True, slots=True)
class _AgentRowProjection:
    agent_id: str
    name: str
    description: str
    system_prompt: str | None
    tool_name: str


def _agent_row_projection(raw_row: Any) -> _AgentRowProjection | None:
    row = _row_dict(raw_row, kind="agent")
    status = _status(row.get("status"), identity="agent")
    if status != "active":
        return None

    fallback_id = _text(row.get("id"), field="id", identity="agent")
    explicit_id = _text(row.get("agent_id"), field="agent_id", identity="agent")
    agent_id = explicit_id or fallback_id
    if not agent_id:
        raise CatalogReadError("agent catalog row has no agent identity")
    name = _text(row.get("name"), field="name", identity=agent_id, required=True)
    description = _text(row.get("description"), field="description", identity=agent_id)
    prompt_value = row.get("system_prompt")
    if prompt_value is None:
        system_prompt = None
    else:
        system_prompt = (
            _text(prompt_value, field="system_prompt", identity=agent_id) or None
        )

    tool_id = _text(row.get("tool_id"), field="tool_id", identity=agent_id)
    tool_name = _text(row.get("tool_name"), field="tool_name", identity=agent_id)
    if tool_id and not tool_name:
        raise CatalogReadError(
            f"{agent_id} catalog row references a tool without a name"
        )
    return _AgentRowProjection(
        agent_id=agent_id,
        name=name,
        description=description,
        system_prompt=system_prompt,
        tool_name=tool_name,
    )


def _merge_agent_row(records: dict[str, _AgentAccumulator], raw_row: Any) -> None:
    projection = _agent_row_projection(raw_row)
    if projection is None:
        return
    existing = records.get(projection.agent_id)
    if existing is None:
        existing = _AgentAccumulator(
            agent_id=projection.agent_id,
            name=projection.name,
            description=projection.description,
            system_prompt=projection.system_prompt,
            tools=set(),
        )
        records[projection.agent_id] = existing
    elif (
        existing.name != projection.name
        or existing.description != projection.description
        or existing.system_prompt != projection.system_prompt
    ):
        raise CatalogReadError(
            f"agent catalog returned conflicting rows for {projection.agent_id!r}"
        )
    if projection.tool_name:
        existing.tools.add(projection.tool_name)


def _agent_record(accumulator: _AgentAccumulator) -> AgentCatalogRecord:
    return AgentCatalogRecord(
        agent_id=accumulator.agent_id,
        name=accumulator.name,
        description=accumulator.description,
        system_prompt=accumulator.system_prompt,
        tools=tuple(sorted(accumulator.tools)) if accumulator.tools else None,
    )


def _project_agent_rows(rows: Any) -> tuple[AgentCatalogRecord, ...]:
    records: dict[str, _AgentAccumulator] = {}
    for raw_row in rows:
        _merge_agent_row(records, raw_row)
    return tuple(
        _agent_record(record)
        for record in sorted(
            records.values(), key=lambda item: (item.name.casefold(), item.agent_id)
        )
    )


class CatalogReadAuthority(AgentCatalogReadPort, WorkflowCatalogReadPort):
    """AU application read authority for GraphOS catalog composition.

    The constructor accepts the existing AU graph engine only.  It deliberately
    does not accept tenant, actor, or a backend: those are supplied by the
    verified ambient ``GraphSession`` and enforced by ``query_cypher``.  A
    missing/expired session, missing ``kg:read`` scope, unavailable backend, or
    row-policy failure therefore propagates as a denial/error instead of being
    converted into an empty catalog.
    """

    def __init__(self, engine: Any, *, session: GraphSession | None = None) -> None:
        query = getattr(engine, "query_cypher", None)
        if not callable(query):
            raise TypeError("catalog authority requires an AU graph query engine")
        self._engine = engine
        self._session = session

    def list_authorized_agents_sync(self) -> tuple[AgentCatalogRecord, ...]:
        """Return the caller-authorized current agents under ``kg:read``."""

        session = resolve_session(self._session, required_scope="kg:read")
        rows = self._engine.query_cypher(_AGENT_CATALOG_QUERY, session=session)
        if rows is None:
            raise CatalogReadError("agent catalog authority returned no result")
        return _project_agent_rows(rows)

    async def list_authorized_agents(self) -> tuple[AgentCatalogRecord, ...]:
        """Async GraphOS port; preserve the ambient session in the worker."""

        return await asyncio.to_thread(self.list_authorized_agents_sync)

    def list_current_workflows_sync(self) -> tuple[WorkflowCatalogRecord, ...]:
        """Return current workflow definitions under ``kg:read``."""

        session = resolve_session(self._session, required_scope="kg:read")
        rows = self._engine.query_cypher(_WORKFLOW_CATALOG_QUERY, session=session)
        if rows is None:
            raise CatalogReadError("workflow catalog authority returned no result")

        records: dict[str, WorkflowCatalogRecord] = {}
        for raw_row in rows:
            row = _row_dict(raw_row, kind="workflow")
            graph_id = _text(row.get("id"), field="id", identity="workflow")
            if not graph_id:
                raise CatalogReadError("workflow catalog row has no workflow identity")
            workflow_id = graph_id
            name = _text(
                row.get("name"), field="name", identity=workflow_id, required=True
            )
            description = _text(
                row.get("description"), field="description", identity=workflow_id
            )
            status = _status(row.get("status"), identity=workflow_id)
            revision = _revision(row, identity=workflow_id)
            digest = _digest_text(row.get("content_hash"), identity=workflow_id)
            if digest is None:
                raise CatalogReadError(
                    f"{workflow_id} catalog row has no workflow definition digest"
                )
            record = WorkflowCatalogRecord(
                workflow_id=workflow_id,
                name=name,
                description=description,
                status=status,
                revision=revision,
                definition_digest=digest,
            )
            existing = records.get(workflow_id)
            if existing is not None and existing != record:
                raise CatalogReadError(
                    f"workflow catalog returned conflicting rows for {workflow_id!r}"
                )
            records[workflow_id] = record

        return tuple(
            sorted(
                records.values(),
                key=lambda item: (item.name.casefold(), item.workflow_id),
            )
        )

    async def list_current_workflows(self) -> tuple[WorkflowCatalogRecord, ...]:
        """Async GraphOS port; preserve the ambient session in the worker."""

        return await asyncio.to_thread(self.list_current_workflows_sync)


def catalog_read_ports(
    engine: Any, session: GraphSession
) -> tuple[WorkflowCatalogReadPort, AgentCatalogReadPort]:
    """Bind AU's two catalog read ports to one engine and session.

    The returned tuple is ordered ``(workflows, agents)`` for GraphOS's
    composition root.  Both entries intentionally share one authority object;
    this is a tuple of views over one AU owner, not two stores.  The bound
    session is still checked against the verified ambient session on every
    read, so composition cannot widen or retarget authority.
    """

    authority = CatalogReadAuthority(engine, session=session)
    return authority, authority


__all__ = [
    "AgentCatalogReadPort",
    "AgentCatalogRecord",
    "CatalogReadAuthority",
    "CatalogReadError",
    "CatalogStatus",
    "catalog_read_ports",
    "WorkflowCatalogReadPort",
    "WorkflowCatalogRecord",
]
