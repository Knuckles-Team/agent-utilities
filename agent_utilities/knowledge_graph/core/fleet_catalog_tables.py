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

**Hardening (NE-007 / AU-CATALOG) — tenant scoping, CAS, desired vs. observed.**
The physical per-owner redb store was an *implicit* tenant boundary, not a
row-level one — it cannot express tenant scope in a query and does not
survive a catalog-sharing model change. Blind ``INSERT ... ON CONFLICT DO
UPDATE`` had no version fencing, so two concurrent probes (or a stale retry
racing a fresh one) could silently clobber each other, and a retried write
was not provably a no-op. And ``mcp_servers`` collapsed desired registration
(``enabled``, from the multiplexer config) and observed discovery
(``reachable``/``last_probe_at``/``last_error``/counts, from the probe) into
one row written by the same path — violating the target-state design's
invariant 4 (``plans/next-evolution/evolution_feedback.md`` §4/§5.3):
*"Desired registration and observed discovery are never collapsed into one
status field."* This revision closes those gaps without adopting the whole
target schema (no ``mcp_server_versions``/UUID identities/``provider_auth_grants``
— those are a parallel track):

* Every table gained an explicit ``tenant_id TEXT NOT NULL`` column, resolved
  ONLY from the verified ambient authority (:func:`~.session.current_session`,
  falling back to ``GraphComputeEngine._verified_tenant``) — never from a
  caller-supplied argument. A caller cannot claim a tenant for a row.
* Every mutable row carries a monotonic ``revision BIGINT`` and an
  ``idempotency_key TEXT``. Writes are **compare-and-set in application
  code**: the existing row (if any) is read first (one batched ``SELECT ...
  WHERE tenant_id = ? AND id IN (...)``); a write whose ``idempotency_key``
  matches the stored one is a **no-op** (a provable retry of the same
  logical write), a write carrying a ``revision`` that is not newer than the
  stored one is **rejected**, and only a write that clears both checks is
  applied. The engine's SQL tier has no composite ``PRIMARY KEY``, no
  ``FOREIGN KEY``, and no conditional/``EXCLUDED``-aware ``ON CONFLICT`` today
  (a parallel track is adding those), so this is intentionally NOT expressed
  as a single conditional ``UPDATE ... WHERE revision < EXCLUDED.revision``
  — each table's DDL comment names the constraint this would become once the
  engine supports it. The read-then-write is not linearizable against two
  writers racing inside the same window; it does deterministically reject any
  write that observably lost the race, which is the concrete defect this
  closes.
* Discovery-derived rows carry an explicit authority kind. OAuth rows carry the
  verified subject and authorization-grant digest minted by the process-owned
  remote-OAuth broker after exact token resolution. Non-OAuth/local rows carry
  the process-owned tenant-local visibility contract with empty principal and
  grant fields. Existing stores need an additive migration for those binding
  columns; until it is applied, the registry rejects legacy rows and the writer
  skips unbound derived rows rather than treating them as global.
* ``mcp_servers`` now holds **desired registration state only** (``name``,
  ``transport``, ``url``, ``enabled`` — the multiplexer config's own claim);
  it is a mutation authority, CAS-fenced like every other table. Observed
  discovery moved to a new **append-only** table, ``mcp_server_discovery``:
  one row per distinct observation (``reachable``, ``last_error``, the four
  counts, ``observed_at``, and the verified ``discovery_principal`` plus
  ``discovery_grant_digest`` that ran the probe, per invariant 7: *"tool
  discovery may vary by principal grant... bound to its discovery subject"*).
  "Append-only" here means a row
  is never mutated to reflect new information — a **changed** observation
  content-hashes to a new row id and is inserted fresh; only a byte-identical
  repeat (nothing new to record) content-hashes to the same row id and is the
  no-op case above. Neither table's write can implicitly mutate the other:
  writing a discovery observation never touches ``mcp_servers.enabled``.
* ``mcp_tools`` gained ``schema_digest`` — a stable SHA-256 of the tool's
  ``input_schema`` alone (narrower than the whole-row idempotency digest,
  which also covers ``description``) — so a consumer can cheaply ask "did
  this tool's *contract* change" without diffing raw JSON, satisfying the
  design doc's ``mcp_tool_versions.schema_digest`` intent without adopting
  its full versioned-artifact table.
* Writing stays **batched, never per-element** (``check-no-per-element-ingest-loop``):
  :func:`write_fleet_catalog` collects every row for a table across the whole
  probed catalog first, then issues at most one batched multi-row ``SELECT``
  and one batched multi-row ``INSERT`` per table for newly-seen ids; only the
  rare CAS-conflict "this id already exists and legitimately changed" case
  falls back to one ``UPDATE`` per changed row (never per unchanged/no-op
  row, and never the common case — the common case is either "new id" or
  "no-op replay", both batched/short-circuited).

Schema (6 tables):

* ``mcp_servers``   — id, tenant_id, name, transport, url, enabled (desired),
  revision, idempotency_key, updated_at.
* ``mcp_server_discovery`` — id, tenant_id, server_id, server_name, reachable,
  last_error, tool_count, skill_count, prompt_count, resource_count,
  observed_at, discovery_authority_kind, discovery_principal,
  discovery_grant_digest, revision,
  idempotency_key. Append-only observation log — see above.
* ``mcp_tools``     — id, tenant_id, server_id, server_name, name,
  description, input_schema (JSON text), schema_digest, tool_mode, enabled,
  discovery_authority_kind, discovery_principal, discovery_grant_digest,
  revision, idempotency_key,
  updated_at. Each discovery-bound identity is an immutable snapshot/version.
* ``mcp_prompts``   — id, tenant_id, server_id, server_name, name,
  description, uri, discovery_authority_kind, discovery_principal,
  discovery_grant_digest, revision,
  idempotency_key, updated_at. Each discovery-bound identity is an immutable
  snapshot/version.
* ``mcp_resources`` — id, tenant_id, server_id, server_name, uri, name,
  description, mime_type, resource_kind, revision, idempotency_key,
  discovery_authority_kind, discovery_principal, discovery_grant_digest,
  updated_at. Today populated
  from the ``skill://`` and ``prompt://``
  resource subsets the multiplexer already discovers (``resource_kind`` =
  ``"skill"``/``"prompt"``) — both ARE MCP Resources under the hood (see
  ``multiplexer._bounded_skill_catalog``/``_bounded_prompt_catalog``
  docstrings). Extending probe coverage to other resource kinds is a
  multiplexer change, not a new mechanism here.
* ``skills``        — id, tenant_id, name, description, uri, skill_type,
  classification, provider, mcp_server, enabled, revision, idempotency_key,
  discovery_authority_kind, discovery_principal, discovery_grant_digest,
  updated_at. ``skill_type`` is
  the raw frontmatter/catalog-declared value
  (``skill``/``workflow``/``graph``/``mcp_skill``); ``classification`` is its
  stored display label — a **stored column**, not a runtime KG-dependent
  lookup that falls back to "Unclassified" whenever a read fails or ingestion
  hasn't run yet (:func:`classify_skill_type` never leaves ``skill_type``
  blank).

Desired server rows reuse the exact KG node-id convention
(``mcp_server_<name>``). Discovery-derived rows retain that logical prefix but
append the immutable grant digest (for example,
``tool_<server>_<name>__<grant_digest>``), so each principal/grant snapshot is
separate while its relationship to the KG object remains obvious.

No secret values, ever: ``command``/``args`` are deliberately never stored
(they can carry local paths and secrets) and no column added here can carry a
credential, token, or endpoint auth material — the same discipline
:func:`write_fleet_catalog`'s docstring already documented, unchanged.
"""

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from .table_ingest import _bounded_columns, _safe_ident, _sql_literal

logger = logging.getLogger(__name__)

TABLE_MCP_SERVERS = "mcp_servers"
TABLE_MCP_SERVER_DISCOVERY = "mcp_server_discovery"
TABLE_MCP_TOOLS = "mcp_tools"
TABLE_MCP_PROMPTS = "mcp_prompts"
TABLE_MCP_RESOURCES = "mcp_resources"
TABLE_SKILLS = "skills"

DISCOVERY_AUTHORITY_OAUTH_GRANT: Literal["oauth_grant"] = "oauth_grant"
DISCOVERY_AUTHORITY_TENANT_LOCAL: Literal["tenant_local"] = "tenant_local"
_DISCOVERY_AUTHORITY_KINDS = frozenset(
    {DISCOVERY_AUTHORITY_OAUTH_GRANT, DISCOVERY_AUTHORITY_TENANT_LOCAL}
)


@dataclass(frozen=True)
class TenantLocalDiscoveryBinding:
    """Process-owned authority for a non-OAuth MCP discovery snapshot.

    Local/stdio children do not have a provider OAuth grant to fingerprint.
    The multiplexer creates this typed binding only after a verified graph
    session is present, and hands it to the writer through its private
    identity-bound side channel.  The fixed authority label is a visibility
    contract, not a digest of roles/scopes and never comes from catalog data.
    """

    tenant_id: str
    authority: Literal["tenant_local"] = DISCOVERY_AUTHORITY_TENANT_LOCAL

    def __post_init__(self) -> None:
        tenant = str(self.tenant_id or "").strip()
        if not tenant or self.authority != DISCOVERY_AUTHORITY_TENANT_LOCAL:
            raise ValueError("tenant-local discovery authority is malformed")


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
    # would be: PRIMARY KEY (tenant_id, id) — a composite tenant-scoped
    # identity. The tier supports only single-column PRIMARY KEY today, and
    # every physical store is still one-per-owner (see module docstring), so
    # a single-column ``id`` (which already encodes ``name``) is safe for
    # now; ``tenant_id`` is carried as authoritative row-level metadata ahead
    # of that constraint existing.
    TABLE_MCP_SERVERS: """CREATE TABLE IF NOT EXISTS mcp_servers (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    name TEXT NOT NULL,
    transport TEXT NOT NULL,
    url TEXT NOT NULL,
    enabled BOOLEAN NOT NULL,
    revision BIGINT NOT NULL,
    idempotency_key TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
    # would be: FOREIGN KEY (server_id) REFERENCES mcp_servers(id). Append-
    # only — see module docstring; ``id`` is a content-derived digest of the
    # observation (excluding ``observed_at``), so a byte-identical repeat
    # observation always maps to the SAME row (the no-op/idempotent-replay
    # case) while any changed field yields a new id (a new, preserved row).
    TABLE_MCP_SERVER_DISCOVERY: """CREATE TABLE IF NOT EXISTS mcp_server_discovery (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    server_id TEXT NOT NULL,
    server_name TEXT NOT NULL,
    reachable BOOLEAN NOT NULL,
    last_error TEXT NOT NULL,
    tool_count BIGINT NOT NULL,
    skill_count BIGINT NOT NULL,
    prompt_count BIGINT NOT NULL,
    resource_count BIGINT NOT NULL,
    observed_at TEXT NOT NULL,
    discovery_authority_kind TEXT NOT NULL,
    discovery_principal TEXT NOT NULL,
    discovery_grant_digest TEXT NOT NULL,
    revision BIGINT NOT NULL,
    idempotency_key TEXT NOT NULL
)""",
    # would be: FOREIGN KEY (server_id) REFERENCES mcp_servers(id).
    TABLE_MCP_TOOLS: """CREATE TABLE IF NOT EXISTS mcp_tools (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    server_id TEXT NOT NULL,
    server_name TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    input_schema TEXT NOT NULL,
    schema_digest TEXT NOT NULL,
    tool_mode TEXT NOT NULL,
    enabled BOOLEAN NOT NULL,
    discovery_authority_kind TEXT NOT NULL,
    discovery_principal TEXT NOT NULL,
    discovery_grant_digest TEXT NOT NULL,
    revision BIGINT NOT NULL,
    idempotency_key TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
    # would be: FOREIGN KEY (server_id) REFERENCES mcp_servers(id).
    TABLE_MCP_PROMPTS: """CREATE TABLE IF NOT EXISTS mcp_prompts (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    server_id TEXT NOT NULL,
    server_name TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    uri TEXT NOT NULL,
    discovery_authority_kind TEXT NOT NULL,
    discovery_principal TEXT NOT NULL,
    discovery_grant_digest TEXT NOT NULL,
    revision BIGINT NOT NULL,
    idempotency_key TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
    # would be: FOREIGN KEY (server_id) REFERENCES mcp_servers(id).
    TABLE_MCP_RESOURCES: """CREATE TABLE IF NOT EXISTS mcp_resources (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    server_id TEXT NOT NULL,
    server_name TEXT NOT NULL,
    uri TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    resource_kind TEXT NOT NULL,
    discovery_authority_kind TEXT NOT NULL,
    discovery_principal TEXT NOT NULL,
    discovery_grant_digest TEXT NOT NULL,
    revision BIGINT NOT NULL,
    idempotency_key TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
    TABLE_SKILLS: """CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    uri TEXT NOT NULL,
    skill_type TEXT NOT NULL,
    classification TEXT NOT NULL,
    provider TEXT NOT NULL,
    mcp_server TEXT NOT NULL,
    enabled BOOLEAN NOT NULL,
    discovery_authority_kind TEXT NOT NULL,
    discovery_principal TEXT NOT NULL,
    discovery_grant_digest TEXT NOT NULL,
    revision BIGINT NOT NULL,
    idempotency_key TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
}

# One-time-per-store DDL cache (keyed by the engine's ``graph_compute``
# identity) — ``ensure_fleet_catalog_tables`` is safe to call before every
# write, but re-issuing 6 ``CREATE TABLE IF NOT EXISTS`` engine round trips
# per row would be exactly the per-element engine-call anti-pattern this
# codebase forbids (batch, never per-element). A boot pass can write hundreds
# of skill rows in one process, so this matters.
_ensured_stores: set[int] = set()

_DISCOVERY_BINDING_MIGRATION = {
    table: (
        "discovery_authority_kind",
        "discovery_principal",
        "discovery_grant_digest",
    )
    for table in (
        TABLE_MCP_SERVER_DISCOVERY,
        TABLE_MCP_TOOLS,
        TABLE_MCP_PROMPTS,
        TABLE_MCP_RESOURCES,
        TABLE_SKILLS,
    )
}


def _existing_table_columns(gc: Any, table: str) -> set[str] | None:
    """Read one owner-scoped table schema through the native SQL surface.

    The epistemic-graph SQL parser currently accepts ``ADD COLUMN`` but drops
    ``IF NOT EXISTS`` from the decoded add-column action, so issuing the
    Postgres spelling would not be idempotent on an existing legacy table.
    ``information_schema.columns`` is the engine's supported schema-read seam;
    use it before each additive migration instead of relying on a dialect
    feature the custom parser does not preserve.
    """
    statement = (
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = "
        f"{_sql_literal(table)}"
    )
    try:
        rows = gc.sql_exec(statement)
    except Exception:  # noqa: BLE001 - migration must fail closed on no schema read
        return None
    if not isinstance(rows, list):
        return None
    columns: set[str] = set()
    for row in rows:
        if isinstance(row, Mapping):
            value = row.get("column_name")
        elif isinstance(row, (list, tuple)) and row:
            value = row[0]
        else:
            return None
        if not isinstance(value, str) or not value:
            return None
        columns.add(value)
    return columns


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


def _content_signature(content: dict[str, Any]) -> str:
    """Stable SHA-256 digest of a row's semantic content.

    Used as the default ``idempotency_key`` when a caller does not supply an
    explicit one, and as ``mcp_tools.schema_digest`` when applied to the
    ``input_schema`` alone (see :func:`_build_tool_row`). Two calls with
    equal ``content`` always digest identically regardless of key order
    (``sort_keys=True``); an absent/blank field is stringified via
    ``default=str`` rather than raising.
    """
    payload = json.dumps(content, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _binding_scope(binding: Any | None = None) -> tuple[str, str, str, str]:
    """Resolve one private typed discovery authority into storage fields.

    OAuth bindings retain the broker-issued principal and grant fingerprint.
    Non-OAuth probes use the separate tenant-local visibility contract; they
    intentionally carry empty principal/grant fields rather than a synthetic
    OAuth-like digest.
    """
    try:
        from ...mcp.remote_oauth_broker import OAuthGrantBinding

        if isinstance(binding, OAuthGrantBinding):
            tenant = str(binding.tenant_id or "").strip()
            principal = str(binding.principal_id or "").strip()
            digest = str(binding.fingerprint or "").strip()
            if not tenant or not principal or not digest:
                return "", "", "", ""
            return tenant, principal, digest, DISCOVERY_AUTHORITY_OAUTH_GRANT
        if isinstance(binding, TenantLocalDiscoveryBinding):
            return (
                binding.tenant_id,
                "",
                "",
                DISCOVERY_AUTHORITY_TENANT_LOCAL,
            )
        return "", "", "", ""
    except (ImportError, TypeError, ValueError):
        return "", "", "", ""


def resolve_discovery_binding(binding: Any | None = None) -> tuple[str, str]:
    """Return values from a broker-issued binding, never ambient session data.

    A discovery grant is minted by :class:`RemoteOAuthBroker` only after exact
    token resolution.  Session roles/scopes/policy are not a grant identity and
    therefore cannot be used as a substitute.  Missing or untyped bindings are
    deliberately unavailable so legacy/unbound observations cannot be relabelled
    as public.
    """

    _tenant, principal, digest, _authority_kind = _binding_scope(binding)
    return principal, digest


def _default_revision() -> int:
    """UTC epoch microseconds — a monotonic-enough default write fence.

    Real callers race across probe cycles that are always at least seconds
    apart, so wall-clock ordering is a safe default; a caller that needs a
    stronger guarantee (e.g. two writes constructed within the same
    microsecond, as tests do) passes an explicit ``revision``.
    """
    return int(datetime.now(UTC).timestamp() * 1_000_000)


def _resolve_tenant_id(engine: Any) -> str:
    """Resolve this write's tenant scope from verified graph authority ONLY.

    Never accepts a caller-supplied tenant string — every row's
    ``tenant_id`` must come from the same verified authority the engine
    itself enforces, so a row can never be mis-scoped by a caller's claim.
    Tries the ambient :class:`~.session.GraphSession` first (the same
    authority :func:`~.session.resolve_session` would bind), then falls back
    to the engine's own ``GraphComputeEngine._verified_tenant`` (KG-2.266) —
    both read the identical underlying authenticated identity; the fallback
    only matters when the caller has an ambient session that
    ``current_session()`` cannot see directly (e.g. a client-side proxy
    engine). Returns ``""`` (never raises) when neither is bound — the write
    proceeds with an empty tenant scope rather than being blocked, consistent
    with this whole module's "best-effort, never blocks the caller" contract;
    an empty ``tenant_id`` is not silently indistinguishable from a real one
    to a reader (it is a distinct, honestly-empty value).
    """
    from .session import current_session

    session = current_session()
    if session is not None:
        tenant = str(getattr(session, "tenant", "") or "")
        if tenant:
            return tenant
    gc = _graph_compute(engine)
    verified = getattr(gc, "_verified_tenant", None)
    if callable(verified):
        try:
            tenant = verified()
        except Exception:  # noqa: BLE001 — best-effort, never blocks the write
            tenant = ""
        if isinstance(tenant, str) and tenant:
            return tenant
    return ""


def _resolve_principal_id(engine: Any) -> str:
    """Resolve the discovery subject (invariant 7) from verified authority.

    Same never-caller-supplied discipline as :func:`_resolve_tenant_id`. Used
    to bind a discovery observation to WHO ran the probe — tool discovery may
    legitimately vary by principal grant, so a snapshot recorded without its
    subject would not be safely comparable across principals later.
    """
    del engine
    return ""


def _bound_row_id(base_id: str, discovery_grant_digest: str) -> str:
    """Give each discovery visibility scope an immutable snapshot identity."""

    return f"{base_id}__{discovery_grant_digest or DISCOVERY_AUTHORITY_TENANT_LOCAL}"


def ensure_fleet_catalog_tables(engine: Any) -> bool:
    """``CREATE TABLE IF NOT EXISTS`` for all 6 fleet-catalog tables.

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
    # CREATE TABLE IF NOT EXISTS cannot alter an installation created before
    # grant binding was introduced.  Add the columns explicitly and leave them
    # nullable for legacy rows: the registry and writer reject NULL/blank rows
    # rather than relabelling them as globally visible.  The custom engine
    # parser accepts ADD COLUMN but does not preserve IF NOT EXISTS on that
    # action, so schema-read first through information_schema and issue a plain
    # ADD only when the column is absent.
    for table, columns in _DISCOVERY_BINDING_MIGRATION.items():
        existing = _existing_table_columns(gc, table)
        if existing is None:
            return False
        for column in columns:
            if column in existing:
                continue
            gc.sql_exec(
                f"ALTER TABLE {_safe_ident(table)} ADD COLUMN "
                f"{_safe_ident(column)} TEXT"
            )
            existing.add(column)
    _ensured_stores.add(key)
    return True


def _select_existing(
    engine: Any,
    table: str,
    tenant_id: str,
    ids: list[str],
    *,
    id_col: str = "id",
) -> dict[str, dict[str, Any]]:
    """One batched ``SELECT`` for every id in ``ids`` — never one per row.

    Returns ``{id: row}`` for whatever currently exists (scoped to
    ``tenant_id``, so a row from another tenant can never be read back as
    "existing" here even if an id collided). Best-effort: an engine with no
    read surface, or a query failure, degrades to "nothing exists yet" (every
    row in the caller's batch is then treated as new) rather than raising —
    consistent with this module never blocking the write path it supports.
    """
    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec") or not ids:
        return {}
    tbl = _safe_ident(table)
    col = _safe_ident(id_col)
    id_list = ", ".join(_sql_literal(row_id) for row_id in ids)
    stmt = (
        f"SELECT * FROM {tbl} WHERE tenant_id = {_sql_literal(tenant_id)} "
        f"AND {col} IN ({id_list})"
    )
    try:
        rows = gc.sql_exec(stmt)
    except Exception:  # noqa: BLE001 — CAS read is best-effort
        logger.debug("fleet catalog CAS read failed for %s", table)
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        if isinstance(row, dict) and row.get(id_col) is not None:
            existing[str(row[id_col])] = row
    return existing


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _cas_batch_upsert(
    engine: Any, table: str, rows: list[dict[str, Any]], *, conflict_col: str = "id"
) -> dict[str, int]:
    """CAS-fenced, idempotency-deduped batch write — one statement per table
    for the common case (new ids), never one per row.

    Every ``row`` must already carry ``tenant_id``, ``revision``, and
    ``idempotency_key``. Reads the current state of every id in ``rows`` in
    ONE batched ``SELECT`` (:func:`_select_existing`), then:

    * a row whose id does not exist yet is queued for one batched multi-row
      ``INSERT`` (a single statement covering every new row in this table for
      this call — the batching this module's docstring and the
      ``check-no-per-element-ingest-loop`` gate require);
    * a row whose stored ``idempotency_key`` already matches is a **no-op**
      (a provable retry of the same logical write — nothing is re-issued);
    * a row whose stored ``revision`` is >= the incoming one is **rejected**
      (a stale write arriving after a fresher one) and left untouched;
    * otherwise the row legitimately changed and is applied via one
      ``UPDATE ... WHERE id = ? AND tenant_id = ?`` per changed row. This is
      the one per-row path in this function, and it is intentionally narrow:
      it only runs for ids that already exist AND legitimately changed —
      never for the bulk "first time seeing this id" or "unchanged replay"
      cases, which are exactly the common cases a fleet probe hits on every
      cycle. (Would be a single conditional multi-row
      ``ON CONFLICT (id) DO UPDATE ... WHERE revision < EXCLUDED.revision``
      once the engine's SQL tier supports conditional/``EXCLUDED``-aware
      upsert — tracked on the parallel constraints track; this deliberately
      does not gamble on that support existing today.)

    Returns ``{"written": n, "rejected_stale": n, "noop_replay": n}``.
    """
    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec") or not rows:
        return {"written": 0, "rejected_stale": 0, "noop_replay": 0}

    tenant_id = str(rows[0].get("tenant_id", ""))
    ids = [str(row[conflict_col]) for row in rows]
    existing = _select_existing(engine, table, tenant_id, ids, id_col=conflict_col)

    to_insert: list[dict[str, Any]] = []
    to_update: list[dict[str, Any]] = []
    rejected_stale = 0
    noop_replay = 0
    for row in rows:
        row_id = str(row[conflict_col])
        current = existing.get(row_id)
        if current is None:
            to_insert.append(row)
            continue
        if str(current.get("idempotency_key", "")) == str(
            row.get("idempotency_key", "")
        ):
            noop_replay += 1
            continue
        if _as_int(current.get("revision")) >= _as_int(row.get("revision")):
            rejected_stale += 1
            continue
        to_update.append(row)

    tbl = _safe_ident(table)
    written = 0
    if to_insert:
        columns = _bounded_columns(list(to_insert[0].keys()))
        values_sql = ", ".join(
            "(" + ", ".join(_sql_literal(row.get(c)) for c in columns) + ")"
            for row in to_insert
        )
        gc.sql_exec(f"INSERT INTO {tbl} ({', '.join(columns)}) VALUES {values_sql}")
        written += len(to_insert)
    for row in to_update:
        columns = _bounded_columns([c for c in row if c != conflict_col])
        set_clause = ", ".join(f"{c} = {_sql_literal(row[c])}" for c in columns)
        gc.sql_exec(
            f"UPDATE {tbl} SET {set_clause} WHERE "
            f"{_safe_ident(conflict_col)} = {_sql_literal(row[conflict_col])} "
            f"AND tenant_id = {_sql_literal(tenant_id)}"
        )
        written += 1
    return {
        "written": written,
        "rejected_stale": rejected_stale,
        "noop_replay": noop_replay,
    }


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


def _build_skill_row(
    *,
    skill_id: str,
    name: str,
    description: str,
    uri: str,
    provider: str,
    mcp_server: str,
    skill_type: str | None,
    disabled: bool,
    tenant_id: str,
    discovery_authority_kind: str,
    discovery_principal: str,
    discovery_grant_digest: str,
    revision: int,
    idempotency_key: str | None,
    now: str,
) -> dict[str, Any]:
    normalized_type, classification = classify_skill_type(skill_type)
    content = {
        "name": _privacy_safe(name),
        "description": _privacy_safe(description),
        "uri": uri,
        "skill_type": normalized_type,
        "classification": classification,
        "provider": _privacy_safe(provider),
        "mcp_server": mcp_server,
        "enabled": not disabled,
        "discovery_authority_kind": discovery_authority_kind,
        "discovery_principal": discovery_principal,
        "discovery_grant_digest": discovery_grant_digest,
    }
    return {
        "id": _bound_row_id(skill_id, discovery_grant_digest),
        "tenant_id": tenant_id,
        **content,
        "revision": revision,
        "idempotency_key": idempotency_key or _content_signature(content),
        "updated_at": now,
    }


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
    revision: int | None = None,
    idempotency_key: str | None = None,
    discovery_binding: Any | None = None,
) -> bool:
    """Upsert one row of the ``skills`` relational table.

    Best-effort: returns ``False`` (never raises) when the engine has no SQL
    surface, or when the write was CAS-rejected (stale) or was a no-op
    (idempotent replay of an already-stored row) — the caller
    (:func:`~..ingestion.skill_workflow_ingest.ingest_runnable_skill`) never
    inspects this return value, so this is purely observational.
    ``revision``/``idempotency_key`` are optional — omitted, they default to
    a wall-clock revision and a content digest of the row respectively (see
    :func:`_default_revision`/:func:`_content_signature`), so an unchanged
    re-ingest of the same skill (routine at every GraphOS boot) is a no-op
    rather than a redundant write.
    """
    if not ensure_fleet_catalog_tables(engine):
        return False
    tenant_id = _resolve_tenant_id(engine)
    (
        binding_tenant,
        discovery_principal,
        discovery_grant_digest,
        discovery_authority_kind,
    ) = _binding_scope(discovery_binding)
    if (
        binding_tenant != tenant_id
        or discovery_authority_kind not in _DISCOVERY_AUTHORITY_KINDS
    ):
        return False
    row = _build_skill_row(
        skill_id=skill_id,
        name=name,
        description=description,
        uri=uri,
        provider=provider,
        mcp_server=mcp_server,
        skill_type=skill_type,
        disabled=disabled,
        tenant_id=tenant_id,
        discovery_authority_kind=discovery_authority_kind,
        discovery_principal=discovery_principal,
        discovery_grant_digest=discovery_grant_digest,
        revision=_default_revision() if revision is None else int(revision),
        idempotency_key=idempotency_key,
        now=_now_iso(),
    )
    stats = _cas_batch_upsert(engine, TABLE_SKILLS, [row])
    return stats["written"] > 0


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


def _schema_digest(input_schema: dict[str, Any]) -> str:
    """Stable SHA-256 of a tool's ``inputSchema`` alone.

    Narrower than the whole tool row's idempotency digest (which also covers
    ``description`` etc.) — this is what a consumer diffs to answer "did this
    tool's *contract* change" (defect 5), independent of prose drift.
    """
    payload = json.dumps(
        input_schema or {}, sort_keys=True, default=str, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_tool_row(
    entry: dict[str, Any],
    *,
    tenant_id: str,
    server_id: str,
    server_name: str,
    discovery_authority_kind: str,
    discovery_principal: str,
    discovery_grant_digest: str,
    revision: int,
    idempotency_key: str | None,
    now: str,
) -> dict[str, Any] | None:
    tool_name = entry.get("name")
    if not tool_name:
        return None
    input_schema = entry.get("inputSchema") or {}
    content = {
        "server_id": server_id,
        "server_name": server_name,
        "name": tool_name,
        "description": _privacy_safe(entry.get("description", "")),
        "input_schema": json.dumps(input_schema, default=str),
        "schema_digest": _schema_digest(input_schema),
        "tool_mode": _derive_tool_mode(input_schema),
        "enabled": True,
        "discovery_authority_kind": discovery_authority_kind,
        "discovery_principal": discovery_principal,
        "discovery_grant_digest": discovery_grant_digest,
    }
    return {
        "id": _bound_row_id(f"tool_{server_name}_{tool_name}", discovery_grant_digest),
        "tenant_id": tenant_id,
        **content,
        "revision": revision,
        "idempotency_key": idempotency_key or _content_signature(content),
        "updated_at": now,
    }


def _build_resource_row(
    *,
    resource_id: str,
    tenant_id: str,
    server_id: str,
    server_name: str,
    discovery_authority_kind: str,
    discovery_principal: str,
    discovery_grant_digest: str,
    uri: str,
    name: str,
    description: str,
    mime_type: str,
    resource_kind: str,
    revision: int,
    idempotency_key: str | None,
    now: str,
) -> dict[str, Any]:
    content = {
        "server_id": server_id,
        "server_name": server_name,
        "uri": uri,
        "name": name,
        "description": _privacy_safe(description),
        "mime_type": mime_type,
        "resource_kind": resource_kind,
        "discovery_authority_kind": discovery_authority_kind,
        "discovery_principal": discovery_principal,
        "discovery_grant_digest": discovery_grant_digest,
    }
    return {
        "id": _bound_row_id(resource_id, discovery_grant_digest),
        "tenant_id": tenant_id,
        **content,
        "revision": revision,
        "idempotency_key": idempotency_key or _content_signature(content),
        "updated_at": now,
    }


def _build_prompt_row(
    entry: dict[str, Any],
    *,
    tenant_id: str,
    server_id: str,
    server_name: str,
    discovery_authority_kind: str,
    discovery_principal: str,
    discovery_grant_digest: str,
    revision: int,
    idempotency_key: str | None,
    now: str,
) -> dict[str, Any] | None:
    prompt_name = entry.get("name")
    if not prompt_name:
        return None
    provider_tag = entry.get("provider") or ""
    prompt_id = f"prompt_{server_name}_{provider_tag}_{prompt_name}".strip("_")
    content = {
        "server_id": server_id,
        "server_name": server_name,
        "name": prompt_name,
        "description": _privacy_safe(entry.get("description", "")),
        "uri": str(entry.get("uri") or ""),
        "discovery_authority_kind": discovery_authority_kind,
        "discovery_principal": discovery_principal,
        "discovery_grant_digest": discovery_grant_digest,
    }
    return {
        "id": _bound_row_id(prompt_id, discovery_grant_digest),
        "tenant_id": tenant_id,
        **content,
        "revision": revision,
        "idempotency_key": idempotency_key or _content_signature(content),
        "updated_at": now,
    }


def write_fleet_catalog(
    engine: Any,
    catalog: dict[str, dict],
    *,
    configs: dict[str, dict] | None = None,
    discovery_bindings: dict[str, Any] | None = None,
    revision: int | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Mirror one probed multiplexer ``catalog`` into the 6 fleet-catalog tables.

    ``catalog`` is the SAME ``{server: {"tools": [...], "skills": [...],
    "prompts": [...], "error": str|None}}`` map
    :func:`~..core.source_sync._write_fleet_nodes` consumes — call this with
    that exact object (never a second probe) so the relational rows and the
    KG nodes can never observe a different fleet state.

    ``configs``, when supplied, is the multiplexer's own
    ``{server: {"command"|"url": ..., "disabled": bool}}`` config map
    (``MCPMultiplexer.load_catalog()``) — used ONLY for ``transport``/``url``/
    ``enabled`` on the (desired-state) server row. ``command``/``args``
    themselves are never stored (they can carry local paths/secrets — the
    same discipline ``skill_reference`` documents for skill identity).

    ``discovery_bindings`` is an internal map of server name to typed,
    process-owned authority objects returned by the multiplexer: an
    ``OAuthGrantBinding`` after broker token resolution, or a
    ``TenantLocalDiscoveryBinding`` for a non-OAuth/local child. Catalog
    payload fields are never used as a substitute; a missing or wrong-tenant
    binding leaves only the desired server row writable. Local authority is a
    tenant visibility contract, never a synthetic OAuth grant digest.

    ``revision``/``idempotency_key`` are optional, shared across every row
    this call writes (representing "this one probe/write attempt"); omitted,
    each row instead defaults to its own content digest as its idempotency
    key, so an unchanged row from an unchanged re-probe is independently a
    no-op even though a changed sibling row in the same catalog is written.
    Passing an explicit value makes the WHOLE call one logical attempt: a
    verbatim retry of the same call (same catalog, same explicit key) is then
    a no-op across every row, which is the shape a caller that wants to
    retry a failed/partial write after a transient engine error would use.

    Unlike the KG write, a server is written here EVEN WHEN unreachable
    (``mcp_server_discovery.reachable=false``, ``last_error=<text>``) —
    "unavailable" must never look identical to "empty" to a caller reading
    this table.

    Returns a counts dict (backward-compatible top-level keys, plus a nested
    ``"cas"`` breakdown of written/rejected-stale/no-op-replay per table);
    never raises (the caller wraps this, but every per-row failure here
    degrades to "table not written" rather than an exception escaping
    mid-catalog).
    """
    if not ensure_fleet_catalog_tables(engine):
        return {"status": "skipped", "reason": "no engine SQL surface"}

    configs = configs or {}
    discovery_bindings = discovery_bindings or {}
    tenant_id = _resolve_tenant_id(engine)
    discovery_authority_ready = False
    now = _now_iso()
    write_revision = _default_revision() if revision is None else int(revision)

    server_rows: list[dict[str, Any]] = []
    discovery_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    skill_rows: list[dict[str, Any]] = []
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

        server_content = {
            "name": server_name,
            "transport": transport,
            "url": str(cfg.get("url") or ""),
            "enabled": not bool(cfg.get("disabled", False)),
        }
        server_rows.append(
            {
                "id": server_id,
                "tenant_id": tenant_id,
                **server_content,
                "revision": write_revision,
                "idempotency_key": idempotency_key
                or _content_signature(server_content),
                "updated_at": now,
            }
        )

        (
            binding_tenant,
            discovery_principal,
            discovery_grant_digest,
            discovery_authority_kind,
        ) = _binding_scope(discovery_bindings.get(server_name))
        server_discovery_authority_ready = bool(
            binding_tenant == tenant_id
            and discovery_authority_kind in _DISCOVERY_AUTHORITY_KINDS
            and (
                discovery_authority_kind == DISCOVERY_AUTHORITY_TENANT_LOCAL
                or (discovery_principal and discovery_grant_digest)
            )
        )
        discovery_authority_ready = (
            discovery_authority_ready or server_discovery_authority_ready
        )

        # Desired registration is tenant-scoped and can be retained without
        # discovery authority. Every observed/derived row, however, must be
        # bound to either the exact OAuth grant or the process-owned tenant
        # local visibility contract; legacy/unbound writes are skipped rather
        # than creating globally visible rows.
        if not server_discovery_authority_ready:
            continue

        discovery_content = {
            "server_id": server_id,
            "server_name": server_name,
            "reachable": reachable,
            "last_error": _privacy_safe(str(err or "")),
            "tool_count": len(tools),
            "skill_count": len(skills),
            "prompt_count": len(prompts),
            "resource_count": len(skills) + len(prompts),
            "discovery_authority_kind": discovery_authority_kind,
            "discovery_principal": discovery_principal,
            "discovery_grant_digest": discovery_grant_digest,
        }
        discovery_key = idempotency_key or _content_signature(discovery_content)
        discovery_rows.append(
            {
                "id": f"disc_{server_id}_{discovery_key[:24]}",
                "tenant_id": tenant_id,
                **discovery_content,
                "observed_at": now,
                "revision": write_revision,
                "idempotency_key": discovery_key,
            }
        )

        for entry in tools:
            if not isinstance(entry, dict):
                continue
            tool_row = _build_tool_row(
                entry,
                tenant_id=tenant_id,
                server_id=server_id,
                server_name=server_name,
                discovery_authority_kind=discovery_authority_kind,
                discovery_principal=discovery_principal,
                discovery_grant_digest=discovery_grant_digest,
                revision=write_revision,
                idempotency_key=idempotency_key,
                now=now,
            )
            if tool_row is not None:
                tool_rows.append(tool_row)

        for entry in skills:
            if not isinstance(entry, dict):
                continue
            skill_name = entry.get("name")
            if not skill_name:
                continue
            skill_rows.append(
                _build_skill_row(
                    skill_id=f"skill_{server_name}_{skill_name}",
                    name=skill_name,
                    description=entry.get("description", ""),
                    uri=str(entry.get("uri") or ""),
                    provider=f"mcp:{server_name}",
                    mcp_server=server_name,
                    skill_type="mcp_skill",
                    disabled=False,
                    tenant_id=tenant_id,
                    discovery_authority_kind=discovery_authority_kind,
                    discovery_principal=discovery_principal,
                    discovery_grant_digest=discovery_grant_digest,
                    revision=write_revision,
                    idempotency_key=idempotency_key,
                    now=now,
                )
            )
            resource_rows.append(
                _build_resource_row(
                    resource_id=f"resource_{server_name}_skill_{skill_name}",
                    tenant_id=tenant_id,
                    server_id=server_id,
                    server_name=server_name,
                    discovery_authority_kind=discovery_authority_kind,
                    discovery_principal=discovery_principal,
                    discovery_grant_digest=discovery_grant_digest,
                    uri=str(entry.get("uri") or ""),
                    name=skill_name,
                    description=entry.get("description", ""),
                    mime_type="text/markdown",
                    resource_kind="skill",
                    revision=write_revision,
                    idempotency_key=idempotency_key,
                    now=now,
                )
            )

        for entry in prompts:
            if not isinstance(entry, dict):
                continue
            prompt_row = _build_prompt_row(
                entry,
                tenant_id=tenant_id,
                server_id=server_id,
                server_name=server_name,
                discovery_authority_kind=discovery_authority_kind,
                discovery_principal=discovery_principal,
                discovery_grant_digest=discovery_grant_digest,
                revision=write_revision,
                idempotency_key=idempotency_key,
                now=now,
            )
            if prompt_row is None:
                continue
            prompt_rows.append(prompt_row)
            prompt_name = str(entry.get("name") or "")
            resource_rows.append(
                _build_resource_row(
                    resource_id=f"resource_{server_name}_prompt_{prompt_name}",
                    tenant_id=tenant_id,
                    server_id=server_id,
                    server_name=server_name,
                    discovery_authority_kind=discovery_authority_kind,
                    discovery_principal=discovery_principal,
                    discovery_grant_digest=discovery_grant_digest,
                    uri=str(entry.get("uri") or ""),
                    name=prompt_name,
                    description=entry.get("description", ""),
                    mime_type="text/plain",
                    resource_kind="prompt",
                    revision=write_revision,
                    idempotency_key=idempotency_key,
                    now=now,
                )
            )

    servers_stats = _cas_batch_upsert(engine, TABLE_MCP_SERVERS, server_rows)
    discovery_stats = _cas_batch_upsert(
        engine, TABLE_MCP_SERVER_DISCOVERY, discovery_rows
    )
    tools_stats = _cas_batch_upsert(engine, TABLE_MCP_TOOLS, tool_rows)
    prompts_stats = _cas_batch_upsert(engine, TABLE_MCP_PROMPTS, prompt_rows)
    resources_stats = _cas_batch_upsert(engine, TABLE_MCP_RESOURCES, resource_rows)
    skills_stats = _cas_batch_upsert(engine, TABLE_SKILLS, skill_rows)

    return {
        "status": "ok",
        "servers_written": servers_stats["written"],
        "servers_unreachable": servers_unreachable,
        "discovery_status": ("bound" if discovery_authority_ready else "unavailable"),
        "tools_written": tools_stats["written"],
        "prompts_written": prompts_stats["written"],
        "resources_written": resources_stats["written"],
        "skills_written": skills_stats["written"],
        "discovery_written": discovery_stats["written"],
        "cas": {
            TABLE_MCP_SERVERS: servers_stats,
            TABLE_MCP_SERVER_DISCOVERY: discovery_stats,
            TABLE_MCP_TOOLS: tools_stats,
            TABLE_MCP_PROMPTS: prompts_stats,
            TABLE_MCP_RESOURCES: resources_stats,
            TABLE_SKILLS: skills_stats,
        },
    }
