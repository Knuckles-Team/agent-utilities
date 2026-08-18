#!/usr/bin/python
from __future__ import annotations

"""Relational read model for the MCP/skill fleet catalog (CONCEPT:AU-KG.ingest.fleet-catalog-relational-tables).

Today the fleet catalog (MCP servers/tools/prompts/skills-over-MCP) exists
**only** as knowledge-graph nodes — ``source_sync._write_fleet_nodes`` writes
``:MCPServer``/``:Tool``/``:Skill`` via Cypher on the hourly
``fleet-tool-schema-sync`` schedule. There is no relational table for any of
it, so a frontend that wants "list the servers/tools" has nothing cheap to
read and instead live-probes the multiplexer across the whole fleet on every
request.

This module is the missing relational tier — **normal, ordinary Postgres-style
tables**, written through the engine's own SQL surface
(``GraphComputeEngine.sql_exec``, KG-2.266), exactly the pattern
:mod:`~.table_ingest` already established for connector/ETL mirroring. Do
**not** duplicate that pattern with a second SQL-writing mechanism — this
module extends it (reuses :func:`~.table_ingest._safe_ident`,
:func:`~.table_ingest._sql_literal`, :func:`~.table_ingest._bounded_columns`
directly) rather than re-deriving SQL-identifier/literal safety a second time.

**Relational tables are the PRIMARY, cheap read path; the KG nodes and any
vectorization are secondary enrichment.** So this write must never depend on —
and must never be blocked by — whatever gates the KG write:

* The engine's SQL user-table surface (``sql_exec`` / ``TableStore``,
  ``epistemic-graph`` ``src/server/sql_tables.rs``) is an **owner-scoped
  catalog** keyed by the verified tenant+actor's own signed carrier authority
  — a private redb file auto-provisioned per owner, gated only on
  authentication (a valid signed request), never on the named-graph
  Read/Write ``Pattern`` grant :class:`IsolationLayer::check_access` enforces
  for Cypher (the gate that has been blocking ``tenant__homelab____commons__``
  writes — see ``agent_utilities/security/tenant_rbac_admission.py``). Reads
  over a served ``SELECT`` similarly execute against the unfiltered owner
  store (only the ``nodes``/``edges`` graph snapshot is RLS-filtered via
  ``IsolationLayer::filter_view`` — a plain user table is not). So this write
  path is **not** behind the same missing-grant wall that blocks Cypher.
* Every write here is best-effort and independently wrapped: a failure (no
  engine SQL surface configured, a transient engine error) is caught by the
  caller and reported, never allowed to abort the KG write it runs alongside.
* Reachability/errors are stored as **honest data** — a server row is written
  for every probed server, including an unreachable one (``reachable=false``,
  ``last_error=<text>``), never silently omitted the way the KG write skips
  an errored server entirely. "Unavailable" must never look identical to
  "empty".

Schema (5 tables, normalized, FK-style relations via ``server_id``):

* ``mcp_servers``  — id, name, transport, url, enabled, reachable,
  last_probe_at, last_error, tool_count, skill_count, prompt_count,
  resource_count, updated_at.
* ``mcp_tools``    — id, server_id, server_name, name, description,
  input_schema (JSON text), tool_mode, enabled, updated_at.
* ``mcp_prompts``  — id, server_id, server_name, name, description, uri,
  updated_at.
* ``mcp_resources`` — id, server_id, server_name, uri, name, description,
  mime_type, resource_kind, updated_at. Today populated from the ``skill://``
  and ``prompt://`` resource subsets the multiplexer already discovers
  (``resource_kind`` = ``"skill"``/``"prompt"``) — both ARE MCP Resources
  under the hood (see ``multiplexer._bounded_skill_catalog``/
  ``_bounded_prompt_catalog`` docstrings). Extending probe coverage to other
  resource kinds is a multiplexer change, not a new mechanism here.
* ``skills``       — id, name, description, uri, skill_type, classification,
  provider, mcp_server, enabled, updated_at. ``skill_type`` is the raw
  frontmatter/catalog-declared value (``skill``/``workflow``/``graph``/
  ``mcp_skill``); ``classification`` is its stored display label — a
  **stored column**, not a runtime KG-dependent lookup that falls back to
  "Unclassified" whenever a read fails or ingestion hasn't run yet
  (:func:`classify_skill_type` never leaves ``skill_type`` blank).

Ids reuse the exact KG node-id convention (``mcp_server_<name>``,
``tool_<server>_<name>``, ``skill_<server>_<name>``) so the relational rows
and the KG nodes describing the same fleet object share one identity across
modalities (graph, relational, and — on a parallel enrichment queue —
vector), per the operator's explicit "leverage each modality" design.

Idempotent: every write is ``INSERT ... ON CONFLICT (id) DO UPDATE SET ...``
(the engine's native upsert, ``epistemic-graph`` ``crates/eg-query/src/sql/
classify.rs::decode_on_conflict``) — a re-sync updates the existing row in
place rather than duplicating it.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from .table_ingest import _bounded_columns, _safe_ident, _sql_literal

logger = logging.getLogger(__name__)

TABLE_MCP_SERVERS = "mcp_servers"
TABLE_MCP_TOOLS = "mcp_tools"
TABLE_MCP_PROMPTS = "mcp_prompts"
TABLE_MCP_RESOURCES = "mcp_resources"
TABLE_SKILLS = "skills"

# Display label for a stored ``skill_type``. Anything not in this map still
# gets a readable classification (title-cased) rather than "Unclassified" —
# only an entirely absent/blank declaration is defaulted, in
# :func:`classify_skill_type`, and it is defaulted to ``"skill"``, never left
# blank.
_SKILL_TYPE_CLASSIFICATION: dict[str, str] = {
    "skill": "Atomic Skill",
    "workflow": "Workflow",
    "graph": "Skill Graph",
    "mcp_skill": "MCP Skill",
}

_DDL: dict[str, str] = {
    TABLE_MCP_SERVERS: """CREATE TABLE IF NOT EXISTS mcp_servers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    transport TEXT NOT NULL,
    url TEXT NOT NULL,
    enabled BOOLEAN NOT NULL,
    reachable BOOLEAN NOT NULL,
    last_probe_at TEXT NOT NULL,
    last_error TEXT NOT NULL,
    tool_count BIGINT NOT NULL,
    skill_count BIGINT NOT NULL,
    prompt_count BIGINT NOT NULL,
    resource_count BIGINT NOT NULL,
    updated_at TEXT NOT NULL
)""",
    TABLE_MCP_TOOLS: """CREATE TABLE IF NOT EXISTS mcp_tools (
    id TEXT PRIMARY KEY,
    server_id TEXT NOT NULL,
    server_name TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    input_schema TEXT NOT NULL,
    tool_mode TEXT NOT NULL,
    enabled BOOLEAN NOT NULL,
    updated_at TEXT NOT NULL
)""",
    TABLE_MCP_PROMPTS: """CREATE TABLE IF NOT EXISTS mcp_prompts (
    id TEXT PRIMARY KEY,
    server_id TEXT NOT NULL,
    server_name TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    uri TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
    TABLE_MCP_RESOURCES: """CREATE TABLE IF NOT EXISTS mcp_resources (
    id TEXT PRIMARY KEY,
    server_id TEXT NOT NULL,
    server_name TEXT NOT NULL,
    uri TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    resource_kind TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
    TABLE_SKILLS: """CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    uri TEXT NOT NULL,
    skill_type TEXT NOT NULL,
    classification TEXT NOT NULL,
    provider TEXT NOT NULL,
    mcp_server TEXT NOT NULL,
    enabled BOOLEAN NOT NULL,
    updated_at TEXT NOT NULL
)""",
}

# One-time-per-store DDL cache (keyed by the engine's ``graph_compute``
# identity) — ``ensure_fleet_catalog_tables`` is safe to call before every
# write, but re-issuing 5 ``CREATE TABLE IF NOT EXISTS`` engine round trips
# per row would be exactly the per-element engine-call anti-pattern this
# codebase forbids (batch, never per-element). A boot pass can write hundreds
# of skill rows in one process, so this matters.
_ensured_stores: set[int] = set()


def _graph_compute(engine: Any) -> Any:
    """The engine's ``GraphComputeEngine`` (the SQL wire handle), or None."""
    return getattr(engine, "graph_compute", None) if engine is not None else None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _privacy_safe(text: str) -> str:
    """Redact fleet-supplied prose before it becomes relational state.

    Mirrors ``source_sync._privacy_safe`` / ``skill_workflow_ingest``'s own
    ``PersistencePrivacyGuard`` use exactly — a tool/skill description is
    EXTERNAL material that routinely embeds absolute paths or other host
    detail, and a relational table is exactly as much "graph state" as a KG
    node property for that purpose.
    """
    from ...security.persistence_privacy import PersistencePrivacyGuard

    safe, _privacy = PersistencePrivacyGuard().sanitize_text(str(text or ""))
    return safe


def ensure_fleet_catalog_tables(engine: Any) -> bool:
    """``CREATE TABLE IF NOT EXISTS`` for all 5 fleet-catalog tables.

    Returns ``False`` (never raises) when the engine has no SQL surface —
    the same graceful degrade :mod:`~.table_ingest` uses, so a caller that
    also writes KG nodes is never blocked by this table not being creatable
    yet (e.g. in a unit test with a bare fake engine).
    """
    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec"):
        return False
    key = id(gc)
    if key in _ensured_stores:
        return True
    for ddl in _DDL.values():
        gc.sql_exec(ddl)
    _ensured_stores.add(key)
    return True


def _upsert(
    engine: Any, table: str, row: dict[str, Any], *, conflict_col: str = "id"
) -> bool:
    """``INSERT ... ON CONFLICT (conflict_col) DO UPDATE SET ...`` for one row.

    Idempotent: re-upserting the same id updates the existing row rather than
    duplicating it (the engine's native upsert — see the module docstring).
    """
    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec"):
        return False
    tbl = _safe_ident(table)
    columns = _bounded_columns(list(row.keys()))
    values = [_sql_literal(row[c]) for c in columns]
    set_clause = ", ".join(
        f"{c} = {_sql_literal(row[c])}" for c in columns if c != conflict_col
    )
    stmt = (
        f"INSERT INTO {tbl} ({', '.join(columns)}) VALUES ({', '.join(values)}) "
        f"ON CONFLICT ({_safe_ident(conflict_col)}) DO UPDATE SET {set_clause}"
    )
    gc.sql_exec(stmt)
    return True


def classify_skill_type(skill_type: str | None) -> tuple[str, str]:
    """Normalize a raw ``skill_type`` and derive its stored display label.

    Never returns a blank ``skill_type`` — an absent/blank declaration
    defaults to ``"skill"`` (an ordinary atomic, runnable skill), so a stored
    row is never left to fall back to "Unclassified" the way the prior
    KG-dependent lookup did whenever ingestion had not run yet or a read
    failed. A ``skill_type`` outside the known set still gets a readable
    ``classification`` (title-cased) rather than "Unclassified" — the corpus
    said *something*, so that is what is stored.
    """
    normalized = str(skill_type or "").strip().lower() or "skill"
    classification = _SKILL_TYPE_CLASSIFICATION.get(
        normalized, normalized.replace("_", " ").title()
    )
    return normalized, classification


def write_skill_row(
    engine: Any,
    *,
    skill_id: str,
    name: str,
    description: str = "",
    uri: str = "",
    provider: str = "",
    mcp_server: str = "",
    skill_type: str | None = None,
    disabled: bool = False,
) -> bool:
    """Upsert one row of the ``skills`` relational table.

    Best-effort: returns ``False`` (never raises) when the engine has no SQL
    surface — the caller (:func:`~..ingestion.skill_workflow_ingest.ingest_runnable_skill`)
    must never let this abort the KG write it runs alongside.
    """
    if not ensure_fleet_catalog_tables(engine):
        return False
    normalized_type, classification = classify_skill_type(skill_type)
    row = {
        "id": skill_id,
        "name": _privacy_safe(name),
        "description": _privacy_safe(description),
        "uri": uri,
        "skill_type": normalized_type,
        "classification": classification,
        "provider": _privacy_safe(provider),
        "mcp_server": mcp_server,
        "enabled": not disabled,
        "updated_at": _now_iso(),
    }
    return _upsert(engine, TABLE_SKILLS, row)


def _derive_tool_mode(input_schema: dict[str, Any] | None) -> str:
    """Classify a served tool as ``condensed`` or ``verbose``.

    Deliberately re-derived (not imported) rather than reused from
    ``source_sync._derive_tool_mode`` — importing it would make this module
    depend on ``source_sync``, which itself will import THIS module (see the
    module docstring's "same operation that writes KG nodes"), a circular
    dependency. The check itself is a 2-line structural test, not a policy
    worth centralizing at the cost of that cycle.
    """
    props = (input_schema or {}).get("properties")
    if isinstance(props, dict) and "action" in props and "params_json" in props:
        return "condensed"
    return "verbose"


def write_fleet_catalog(
    engine: Any,
    catalog: dict[str, dict],
    *,
    configs: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Mirror one probed multiplexer ``catalog`` into the 5 fleet-catalog tables.

    ``catalog`` is the SAME ``{server: {"tools": [...], "skills": [...],
    "prompts": [...], "error": str|None}}`` map
    :func:`~..core.source_sync._write_fleet_nodes` consumes — call this with
    that exact object (never a second probe) so the relational rows and the
    KG nodes can never observe a different fleet state.

    ``configs``, when supplied, is the multiplexer's own
    ``{server: {"command"|"url": ..., "disabled": bool}}`` config map
    (``MCPMultiplexer.load_catalog()``) — used ONLY for ``transport``/``url``/
    ``enabled`` on the server row. ``command``/``args`` themselves are never
    stored (they can carry local paths/secrets — the same discipline
    ``skill_reference`` documents for skill identity).

    Unlike the KG write, a server is written here EVEN WHEN unreachable
    (``reachable=false``, ``last_error=<text>``) — "unavailable" must never
    look identical to "empty" to a caller reading this table.

    Returns a counts dict; never raises (the caller wraps this, but every
    per-row failure here degrades to "table not written" rather than an
    exception escaping mid-catalog).
    """
    if not ensure_fleet_catalog_tables(engine):
        return {"status": "skipped", "reason": "no engine SQL surface"}

    configs = configs or {}
    now = _now_iso()
    servers_written = 0
    tools_written = 0
    prompts_written = 0
    resources_written = 0
    skills_written = 0
    servers_unreachable = 0

    for server_name, info in (catalog or {}).items():
        if not isinstance(info, dict):
            continue
        err = info.get("error")
        reachable = err is None
        if not reachable:
            servers_unreachable += 1
        tools = info.get("tools") or []
        skills = info.get("skills") or []
        prompts = info.get("prompts") or []

        cfg = configs.get(server_name) or {}
        transport = (
            "http" if cfg.get("url") else ("stdio" if cfg.get("command") else "")
        )
        server_id = f"mcp_server_{server_name}"

        if _upsert(
            engine,
            TABLE_MCP_SERVERS,
            {
                "id": server_id,
                "name": server_name,
                "transport": transport,
                "url": str(cfg.get("url") or ""),
                "enabled": not bool(cfg.get("disabled", False)),
                "reachable": reachable,
                "last_probe_at": now,
                "last_error": _privacy_safe(str(err or "")),
                "tool_count": len(tools),
                "skill_count": len(skills),
                "prompt_count": len(prompts),
                "resource_count": len(skills) + len(prompts),
                "updated_at": now,
            },
        ):
            servers_written += 1

        for entry in tools:
            if not isinstance(entry, dict):
                continue
            tool_name = entry.get("name")
            if not tool_name:
                continue
            if _upsert(
                engine,
                TABLE_MCP_TOOLS,
                {
                    "id": f"tool_{server_name}_{tool_name}",
                    "server_id": server_id,
                    "server_name": server_name,
                    "name": tool_name,
                    "description": _privacy_safe(entry.get("description", "")),
                    "input_schema": json.dumps(
                        entry.get("inputSchema") or {}, default=str
                    ),
                    "tool_mode": _derive_tool_mode(entry.get("inputSchema")),
                    "enabled": True,
                    "updated_at": now,
                },
            ):
                tools_written += 1

        for entry in skills:
            if not isinstance(entry, dict):
                continue
            skill_name = entry.get("name")
            if not skill_name:
                continue
            skill_id = f"skill_{server_name}_{skill_name}"
            if write_skill_row(
                engine,
                skill_id=skill_id,
                name=skill_name,
                description=entry.get("description", ""),
                uri=str(entry.get("uri") or ""),
                provider=f"mcp:{server_name}",
                mcp_server=server_name,
                skill_type="mcp_skill",
            ):
                skills_written += 1
            if _upsert(
                engine,
                TABLE_MCP_RESOURCES,
                {
                    "id": f"resource_{server_name}_skill_{skill_name}",
                    "server_id": server_id,
                    "server_name": server_name,
                    "uri": str(entry.get("uri") or ""),
                    "name": skill_name,
                    "description": _privacy_safe(entry.get("description", "")),
                    "mime_type": "text/markdown",
                    "resource_kind": "skill",
                    "updated_at": now,
                },
            ):
                resources_written += 1

        for entry in prompts:
            if not isinstance(entry, dict):
                continue
            prompt_name = entry.get("name")
            if not prompt_name:
                continue
            provider_tag = entry.get("provider") or ""
            prompt_id = f"prompt_{server_name}_{provider_tag}_{prompt_name}".strip("_")
            if _upsert(
                engine,
                TABLE_MCP_PROMPTS,
                {
                    "id": prompt_id,
                    "server_id": server_id,
                    "server_name": server_name,
                    "name": prompt_name,
                    "description": _privacy_safe(entry.get("description", "")),
                    "uri": str(entry.get("uri") or ""),
                    "updated_at": now,
                },
            ):
                prompts_written += 1
            if _upsert(
                engine,
                TABLE_MCP_RESOURCES,
                {
                    "id": f"resource_{server_name}_prompt_{prompt_name}",
                    "server_id": server_id,
                    "server_name": server_name,
                    "uri": str(entry.get("uri") or ""),
                    "name": prompt_name,
                    "description": _privacy_safe(entry.get("description", "")),
                    "mime_type": "text/plain",
                    "resource_kind": "prompt",
                    "updated_at": now,
                },
            ):
                resources_written += 1

    return {
        "status": "ok",
        "servers_written": servers_written,
        "servers_unreachable": servers_unreachable,
        "tools_written": tools_written,
        "prompts_written": prompts_written,
        "resources_written": resources_written,
        "skills_written": skills_written,
    }
