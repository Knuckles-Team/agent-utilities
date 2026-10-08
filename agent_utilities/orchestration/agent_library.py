"""The Agent Library: durable, delegatable agent records (AU-CONTROL-R024–R026).

One library entry is the ``CallableResource`` node ``run_agent`` resolves for
delegation. No second record type exists. A locally built or role agent is a
``Skill`` + ``CallableResource(resource_type=AGENT_SKILL)`` pair that carries
the atomic-skill field contract (``source_ref`` ``skill://...`` and
``instruction_digest``). An assembled agent graph is a
``CallableResource(resource_type=AGENT_GRAPH)`` that carries the EG graph draft
and its committed decision reference.

The record fields mirror EG's ``AgentLibraryEntryDraft`` (role, system prompt,
tools, skills, model profile). A later EG ``AgentLibrary`` publish maps 1:1.

Every surface reads and writes through :class:`AgentLibrary`:

* the agent-webui ``/api/enhanced/agent-library/*`` routes;
* the graph-os intent surface (``agent_library`` tool, ``find``/``ask``/``manage``);
* L3 assembly (:mod:`agent_utilities.decide.consumers.assembly`), which saves
  every solved agent graph here for reuse.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "AGENT_LIBRARY_PROVIDER_REF",
    "LIBRARY_KINDS",
    "AgentLibrary",
    "AgentRecord",
    "assembled_graph_record",
    "role_agent_from_blueprint",
    "role_agents_from_prompts",
    "seed_role_agents",
]

#: Marks an entry the library itself wrote (the agent-webui contract).
AGENT_LIBRARY_PROVIDER_REF = "provider://agent-webui-library"
#: ``local``: built by an operator; ``role``: generated from a fleet package
#: prompt; ``agent_graph``: an L3 assembly saved for reuse.
LIBRARY_KINDS = ("local", "role", "agent_graph")
MAX_INSTRUCTIONS_BYTES = 32_000
MAX_TOOLS = 256

_SKILL = "AGENT_SKILL"
_A2A = "A2A_AGENT"
_GRAPH = "AGENT_GRAPH"
_LIBRARY_TYPES = frozenset({_SKILL, _A2A, _GRAPH})
_LIST_QUERY = (
    "MATCH (r:CallableResource) WHERE r.resource_type = $a2a "
    "OR ((r.resource_type = $skill OR r.resource_type = $graph) "
    "AND r.provider_ref = $ref) RETURN r LIMIT {limit}"
)
_GET_QUERY = "MATCH (r:CallableResource {id: $id}) RETURN r"
_TOOLS_QUERY = "MATCH (r {id: $id})-[:USES_TOOL]->(t) RETURN t.name AS name, t.id AS id"
_SERVER_TOOLS_QUERY = "MATCH (t:Tool) WHERE t.mcp_server = $s RETURN t.id AS id"
_PROMPTS_QUERY = "MATCH (p:Prompt) RETURN p LIMIT {limit}"

Reader = Callable[[str, dict[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class AgentRecord:
    """One built agent: prompt, tools, skills, model profile, context policy, role."""

    name: str
    system_prompt: str = ""
    description: str = ""
    role: str = ""
    kind: str = "local"
    tools: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    model_profile: str = ""
    mcp_server: str = ""
    context_policy: Mapping[str, Any] = field(default_factory=dict)
    source: str = ""
    graph: Mapping[str, Any] | None = None
    agent_id: str = ""
    status: str = "active"
    timestamp: str = ""
    runnable: bool = True
    tool_refs: tuple[Mapping[str, Any], ...] = ()
    #: A2A entries only: the published endpoint and agent card.
    endpoint: str = ""
    agent_card: Any = None

    def validate(self) -> AgentRecord:
        """Fail closed on a record no surface can store or run."""
        if not self.name.strip() or len(self.name) > 120:
            raise ValueError("Agent name is required")
        if self.kind not in LIBRARY_KINDS:
            raise ValueError(f"unknown agent library kind {self.kind!r}")
        if self.kind != "agent_graph" and not self.system_prompt.strip():
            raise ValueError("Agent instructions are required")
        if len(self.system_prompt.encode("utf-8")) > MAX_INSTRUCTIONS_BYTES:
            raise ValueError("Instructions exceed the safety bound")
        return self

    def view(self) -> dict[str, Any]:
        """The API projection (the agent-webui ``LibraryAgent`` shape plus fields)."""
        return {
            "id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "kind": "a2a" if self.kind == "a2a" else "local",
            "library_kind": self.kind,
            "role": self.role,
            "mcp_server": self.mcp_server or None,
            "model_preference": self.model_profile or None,
            "timestamp": self.timestamp or None,
            "status": self.status,
            "runnable_bound": self.runnable,
            "skills": list(self.skills),
            "context_policy": dict(self.context_policy),
            "source": self.source,
            "endpoint": self.endpoint or None,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _json_map(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        loaded = json.loads(value) if isinstance(value, str) and value else {}
    except ValueError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _row_kind(row: Mapping[str, Any]) -> str:
    resource_type = str(row.get("resource_type") or "")
    if resource_type == _A2A:
        return "a2a"
    if resource_type == _GRAPH:
        return "agent_graph"
    kind = str(row.get("library_kind") or "local")
    return kind if kind in LIBRARY_KINDS else "local"


def _text(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key)
    return default if value in (None, "") else str(value)


def _record_from_row(row: Mapping[str, Any]) -> AgentRecord:
    kind = _row_kind(row)
    graph = _json_map(row.get("graph_json")) if kind == "agent_graph" else None
    return AgentRecord(
        agent_id=_text(row, "id"),
        name=_text(row, "name"),
        description=_text(row, "description"),
        system_prompt=_text(row, "system_prompt"),
        role=_text(row, "role"),
        kind=kind,
        skills=_strings(row.get("skill_refs")),
        model_profile=_text(row, "model_preference"),
        mcp_server=_text(row, "mcp_server"),
        context_policy=_json_map(row.get("context_policy_json")),
        source=_text(row, "library_source"),
        graph=graph,
        status=_text(row, "status", "active"),
        timestamp=_text(row, "timestamp"),
        runnable=bool(row.get("runnable_bound", kind == "a2a")),
        endpoint=_text(row, "endpoint"),
        agent_card=row.get("agent_card"),
    )


def _tool_ref(row: Any) -> dict[str, Any]:
    """``{id, name}`` of one bound tool row; empty when it names neither."""
    if not isinstance(row, Mapping) or not (row.get("id") or row.get("name")):
        return {}
    return {"id": row.get("id"), "name": row.get("name")}


def _row(item: Any) -> Mapping[str, Any] | None:
    if not isinstance(item, Mapping):
        return None
    node = item.get("r", item)
    return node if isinstance(node, Mapping) else None


class AgentLibrary:
    """Save, get and list Agent Library records over one graph engine.

    ``read`` overrides the row reader (for example a tenant+commons union read);
    the default is the engine's own ``query_cypher``.
    """

    def __init__(self, engine: Any, *, read: Reader | None = None) -> None:
        self.engine = engine
        self._read = read or engine.query_cypher

    def list(self, kind: str | None = None, *, limit: int = 500) -> list[AgentRecord]:
        """Every non-archived entry, sorted by name; ``kind`` filters one kind."""
        params = {
            "a2a": _A2A,
            "skill": _SKILL,
            "graph": _GRAPH,
            "ref": AGENT_LIBRARY_PROVIDER_REF,
        }
        query = _LIST_QUERY.format(limit=int(limit))
        rows = [_row(item) for item in self._read(query, params) or []]
        records = [
            _record_from_row(row)
            for row in rows
            if row is not None and str(row.get("status") or "") != "ARCHIVED"
        ]
        if kind:
            records = [record for record in records if record.kind == kind]
        return sorted(records, key=lambda record: record.name.lower())

    def get(self, agent_id: str) -> AgentRecord | None:
        """One entry with its bound tools, or ``None`` when absent."""
        rows = [_row(item) for item in self._read(_GET_QUERY, {"id": agent_id}) or []]
        row = next((row for row in rows if row is not None), None)
        if row is None or _text(row, "resource_type") not in _LIBRARY_TYPES:
            return None
        tools = self._read(_TOOLS_QUERY, {"id": agent_id}) or []
        refs = tuple(_tool_ref(t) for t in tools if _tool_ref(t))
        return replace(_record_from_row(row), tool_refs=refs)

    def save(self, record: AgentRecord) -> AgentRecord:
        """Write one record; returns it with its id and the tools actually bound."""
        record = record.validate()
        if record.kind == "agent_graph":
            return self._save_graph(record)
        return self._save_runnable(record)

    def bound_tools(self, tool_ids: Iterable[str], mcp_server: str = "") -> list[str]:
        """The given tool ids plus every ingested tool of ``mcp_server``."""
        resolved = list(tool_ids)
        if mcp_server:
            rows = self.engine.backend.execute(_SERVER_TOOLS_QUERY, {"s": mcp_server})
            resolved.extend(
                str(r["id"]) for r in rows or [] if isinstance(r, dict) and r.get("id")
            )
        return resolved

    def bind_tools(self, resource_id: str, tool_ids: Iterable[str]) -> list[str]:
        """Link each distinct tool id under ``USES_TOOL``; returns them sorted."""
        seen: set[str] = set()
        for tool_id in tool_ids:
            if tool_id in seen or len(seen) >= MAX_TOOLS:
                continue
            seen.add(tool_id)
            self.engine.link_nodes(resource_id, tool_id, "USES_TOOL")
        return sorted(seen)

    def _common(self, record: AgentRecord, source_ref: str) -> dict[str, Any]:
        common: dict[str, Any] = {
            "name": record.name,
            "description": record.description or record.name,
            "source_ref": source_ref,
            "provider_ref": AGENT_LIBRARY_PROVIDER_REF,
            "timestamp": _now(),
            "library_kind": record.kind,
            "role": record.role,
            "skill_refs": list(record.skills),
            "context_policy_json": json.dumps(dict(record.context_policy)),
            "library_source": record.source,
        }
        if record.mcp_server:
            common["mcp_server"] = record.mcp_server
        if record.model_profile:
            common["model_preference"] = record.model_profile
        return common

    def _save_runnable(self, record: AgentRecord) -> AgentRecord:
        from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
            runnable_skill_digest,
            skill_reference,
        )

        source_ref = skill_reference(record.name)
        skill_id = f"skill:{source_ref.removeprefix('skill://')}"
        resource_id = f"resource:{skill_id}"
        common = self._common(record, source_ref)
        common["instruction_digest"] = runnable_skill_digest(record.system_prompt)
        body = record.system_prompt
        tools = self.bound_tools(record.tools, record.mcp_server)
        self.engine.add_node(
            skill_id, "Skill", {**common, "body": body, "instruction": body}
        )
        self.engine.add_node(
            resource_id,
            "CallableResource",
            {
                **common,
                "resource_type": _SKILL,
                "system_prompt": body,
                "runnable_bound": True,
            },
        )
        self.engine.link_nodes(skill_id, resource_id, "BINDS_RUNNABLE")
        bound = self.bind_tools(resource_id, tools)
        return replace(record, agent_id=resource_id, tools=tuple(bound))

    def _save_graph(self, record: AgentRecord) -> AgentRecord:
        graph = dict(record.graph or {})
        graph_id = str(graph.get("graph_id") or record.name)
        resource_id = f"resource:agent-graph:{_slug(graph_id)}"
        props = {
            **self._common(record, f"agent-graph://{_slug(graph_id)}"),
            "resource_type": _GRAPH,
            "system_prompt": record.system_prompt,
            "graph_json": json.dumps(graph, sort_keys=True, default=str),
            "runnable_bound": False,
        }
        # Assembled tools are EG component ids, not ``:Tool`` nodes: they stay in
        # ``graph_json`` and bind no ``USES_TOOL`` edge.
        self.engine.add_node(resource_id, "CallableResource", props)
        return replace(record, agent_id=resource_id, runnable=False)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "graph"


# ── Pre-built role agents (AU-CONTROL-R025) ───────────────────────────────


def _identity(blueprint: Mapping[str, Any]) -> Mapping[str, Any]:
    identity = blueprint.get("identity")
    return identity if isinstance(identity, Mapping) else {}


def _directive(blueprint: Mapping[str, Any]) -> str:
    instructions = blueprint.get("instructions")
    if isinstance(instructions, Mapping):
        return str(instructions.get("core_directive") or "").strip()
    return str(blueprint.get("content") or "").strip()


def role_agent_from_blueprint(
    blueprint: Mapping[str, Any], provider: str = ""
) -> AgentRecord | None:
    """A ``role`` record from one packaged prompt blueprint.

    Only a blueprint that names an ``identity.role`` and a directive is a role
    agent; generic orchestrator prompts (no role) produce ``None``. The package
    that ships the prompt is the agent's MCP server, so its tools bind on save.
    """
    role = str(_identity(blueprint).get("role") or "").strip()
    directive = _directive(blueprint)
    if not role or not directive:
        return None
    source = str(blueprint.get("source") or provider or "").strip()
    task = str(blueprint.get("task") or role)
    return AgentRecord(
        name=f"role-{_slug(task)}",
        system_prompt=directive,
        description=str(blueprint.get("description") or role),
        role=role,
        kind="role",
        skills=_strings(blueprint.get("skills")),
        mcp_server=source,
        source=f"prompt:{source}/{task}" if source else f"prompt:{task}",
    )


def role_agents_from_prompts(engine: Any, *, limit: int = 1000) -> list[AgentRecord]:
    """Role records for every ``:Prompt`` node whose blueprint names a role.

    The ``:Prompt`` corpus already holds the base, fleet-provider and harvested
    fleet prompts (``registry_builder.ingest_prompt_node``), so this reads no
    filesystem path.
    """
    rows = engine.query_cypher(_PROMPTS_QUERY.format(limit=int(limit)), {}) or []
    records: dict[str, AgentRecord] = {}
    for item in rows:
        node = item.get("p", item) if isinstance(item, Mapping) else None
        blueprint = _json_map(node.get("json_blueprint")) if node else {}
        record = role_agent_from_blueprint(blueprint)
        if record is not None:
            records.setdefault(record.name, record)
    return [records[name] for name in sorted(records)]


def seed_role_agents(library: AgentLibrary) -> list[AgentRecord]:
    """Save every role agent the prompt corpus defines; returns the saved records."""
    return [library.save(record) for record in role_agents_from_prompts(library.engine)]


# ── Assembled agent graphs (AU-CONTROL-R026) ──────────────────────────────


def _component_ids(agent: Mapping[str, Any], key: str) -> tuple[str, ...]:
    return tuple(str(d.get("component_id")) for d in agent.get(key) or [])


def _prompt_ref(agent: Mapping[str, Any]) -> str:
    prompt = agent.get("system_prompt")
    if not isinstance(prompt, Mapping):
        return ""
    return f"component:{prompt.get('component_id', '')}"


def _graph_id(graph: Mapping[str, Any] | None, agent: Mapping[str, Any]) -> str:
    return str((graph or {}).get("graph_id") or agent.get("agent_id") or "")


def assembled_graph_record(
    result: Mapping[str, Any],
    *,
    committed: Mapping[str, Any] | None = None,
    published: Any = None,
) -> AgentRecord | None:
    """An ``agent_graph`` record for one solved assembly, or ``None``.

    The record keeps the EG graph draft, the assembled agents, the committed
    decision record id and the publish receipt, so a later caller can reuse
    the graph instead of assembling again.
    """
    raw = result.get("graph")
    graph = raw if isinstance(raw, Mapping) else None
    agents = [a for a in result.get("agents") or [] if isinstance(a, Mapping)]
    first = agents[0] if agents else {}
    graph_id = _graph_id(graph, first)
    if not graph_id:
        return None
    decision = _jsonable(committed) or {}
    payload = {
        "graph_id": graph_id,
        "graph": None if graph is None else dict(graph),
        "agents": [dict(a) for a in agents],
        "decision_record_id": decision.get("record_id"),
        "published": _jsonable(published),
    }
    return AgentRecord(
        name=graph_id,
        description="assembled by EG AgentAssemble",
        role=_text(first, "role"),
        kind="agent_graph",
        system_prompt=_prompt_ref(first),
        tools=_component_ids(first, "tools"),
        skills=_component_ids(first, "skills"),
        model_profile=_text(first, "model_identity"),
        source="assembly",
        graph=payload,
    )


def _jsonable(value: Any) -> Any:
    value = getattr(value, "payload", value)
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else value
