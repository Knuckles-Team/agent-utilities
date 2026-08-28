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
  ``FOREIGN KEY``, and no working ``ON CONFLICT`` at all today (every form
  raises the same duplicate-key error a bare ``INSERT`` does — measured live
  2026-08-25; a parallel track is adding those), so this is NOT expressed
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
* ``skill_classification_overrides`` (not one of the 6 migrated tables --
  see its own constant comment) — id (the skill's UNBOUND base id),
  tenant_id, skill_type, set_by, revision, idempotency_key, updated_at.
  CONCEPT:AU-KG.ingest.skill-classification-writeback: an operator's
  classification choice, made durable through
  :func:`write_skill_classification_override` and consulted by
  :func:`write_skill_row` on every write so it survives the next
  ``fleet-tool-schema-sync`` re-derive from the source SKILL.md's frontmatter.
  It exists because every deployed profile mounts the ``universal-skills``
  source tree **read-only** (NFS export, empirically confirmed unwritable
  even where the k8s manifest's mount flag does not say so) -- an operator
  classification therefore cannot always be written back to the SKILL.md
  itself, but MUST still survive a re-sync. See
  :func:`~..ingestion.skill_classification.reclassify_skill` for the
  capability that writes both this table and, best-effort, the source file.

Desired server rows reuse the exact KG node-id convention
(``mcp_server_<name>``). Discovery-derived rows retain that logical prefix but
append the immutable grant digest (for example,
``tool_<server>_<name>__<grant_digest>``), so each principal/grant snapshot is
separate while its relationship to the KG object remains obvious.

No secret values, ever: ``command``/``args`` are deliberately never stored
(they can carry local paths and secrets) and no column added here can carry a
credential, token, or endpoint auth material — the same discipline
:func:`write_fleet_catalog`'s docstring already documented, unchanged.

**NE-052 / AU-CATALOG — migrating an already-deployed pre-NE-007 store.**
``CREATE TABLE IF NOT EXISTS`` is a silent no-op against a table that
already exists under the OLD (pre-NE-007) shape — 5 tables, no
``mcp_server_discovery``, no ``tenant_id``/``revision``/``idempotency_key``/
``schema_digest``/``discovery_*`` columns anywhere. Every CAS write built in
this module assumes those columns exist, so an un-migrated deployment fails
at INSERT/UPDATE time referencing columns the live table does not have.
:func:`_claim_and_migrate` (invoked by :func:`ensure_fleet_catalog_tables`)
closes that gap:

* **Detection is column-based, not version-based.** The pre-NE-007 store has
  no version marker at all, so every call re-introspects each table's real
  columns via ``information_schema.columns`` (:func:`_existing_table_columns`)
  rather than trusting a cached/assumed generation.
* **A forward-only, checksummed ledger** (table
  ``fleet_catalog_schema_migrations``) records one append-only row per
  applied migration step (:data:`_MIGRATION_COLUMN_STEPS`, each identified
  by a stable id and a content checksum of the columns it adds) plus one
  singleton row (id ``"schema_state"``) tracking the store's current
  version/checksum — so a later start can cheaply confirm "already current"
  without repeating the migration.
* **Ordered, idempotent ``ALTER TABLE ADD COLUMN`` steps** bring an old store
  up to the frozen NE-007/current shape, with an explicit, narrow one-time
  backfill of ``tenant_id``/``revision``/``idempotency_key``/
  ``schema_digest`` for pre-existing rows (never the ``discovery_*``
  columns — those stay unbound/NULL for legacy rows exactly as this module
  already documented, since there is no way to retroactively know who ran a
  pre-NE-007 probe). Existing rows and their other columns are never
  dropped or rewritten.
* ``tenant_id`` backfill uses the reserved sentinel
  :data:`LEGACY_TENANT_SENTINEL` — never a caller/session tenant — because a
  pre-existing row predates per-tenant scoping and guessing a real tenant
  would silently mis-attribute historical data.
* **Fail closed.** A store whose columns don't correspond to any known
  schema generation (:func:`_detect_diverged_schema`), or whose ledger
  records a migration id this code version does not recognize, raises
  :class:`FleetCatalogSchemaDivergedError` / :class:`FleetCatalogSchemaTooNewError`
  — the one deliberate exception to this module's usual "never raises,
  always best-effort" contract, because silently limping forward against an
  unverified schema is exactly the defect this closes.
* **Concurrency** is a claim on the ledger's singleton lock row, written
  read-then-write via :func:`_ledger_put` (the deployed engine's SQL tier
  ignores an ``ON CONFLICT`` clause entirely — measured live 2026-08-25;
  see that function). The claim writes a fresh token and re-reads it: a
  process that finds a different claimant lost the race, performs no DDL,
  and returns ``False`` for that attempt rather than racing the winner — a
  genuine no-op, not an error — picking up the now-migrated schema on its
  next call.
* **Verification**: after applying every needed step, the winner
  re-introspects every table and asserts the columns now match the frozen
  current shape (:data:`_CURRENT_SCHEMA_COLUMNS`) before recording the
  ledger as ``"complete"`` and returning success.
"""

import hashlib
import json
import logging
import uuid
from collections.abc import Iterator, Mapping
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
# Operator-set classification override (CONCEPT:AU-KG.ingest.skill-classification-writeback).
# Deliberately NOT one of the 6 NE-007/NE-052-migrated tables above -- it is a
# brand-new table with no legacy shape to migrate FROM, so it is created via
# its own trivial ``CREATE TABLE IF NOT EXISTS`` (see
# ``_ensure_skill_classification_overrides_table``) rather than being folded
# into ``_claim_and_migrate``'s ledger-tracked, column-diff-checked set --
# adding it there would require extending ``_CURRENT_SCHEMA_COLUMNS``/
# ``_detect_diverged_schema`` for a table that was never at risk of the
# pre-NE-007 legacy shape those exist to detect.
TABLE_SKILL_CLASSIFICATION_OVERRIDES = "skill_classification_overrides"

DISCOVERY_AUTHORITY_OAUTH_GRANT: Literal["oauth_grant"] = "oauth_grant"
DISCOVERY_AUTHORITY_TENANT_LOCAL: Literal["tenant_local"] = "tenant_local"
_DISCOVERY_AUTHORITY_KINDS = frozenset(
    {DISCOVERY_AUTHORITY_OAUTH_GRANT, DISCOVERY_AUTHORITY_TENANT_LOCAL}
)


class FleetCatalogMigrationError(RuntimeError):
    """Base for a fleet-catalog schema state this code refuses to serve.

    Raised only by :func:`_claim_and_migrate` / :func:`ensure_fleet_catalog_tables`
    — the one deliberate exception to this module's usual "best-effort, never
    raises" contract (see the module docstring's NE-052 section). A store in
    one of these states must not be silently written to: doing so is exactly
    the class of defect (a write against columns the code cannot verify)
    this hardening closes.
    """


class FleetCatalogSchemaDivergedError(FleetCatalogMigrationError):
    """The store's actual columns match no known schema generation.

    Raised when a table's introspected column set is neither the frozen
    pre-NE-007 legacy shape, the current shape, nor any valid point on the
    ordered forward-only migration path between them (see
    :func:`_detect_diverged_schema`) — i.e. the table was hand-modified or
    partially/out-of-order migrated by something other than this module.
    """


class FleetCatalogSchemaTooNewError(FleetCatalogMigrationError):
    """The migration ledger records a migration this code version does not know.

    Raised when ``fleet_catalog_schema_migrations`` already names a
    ``migration_id`` outside this module's own :data:`_MIGRATION_COLUMN_STEPS`
    — the store was migrated forward by a newer revision of this module and
    this (older) code cannot safely verify or extend that schema.
    """


# Reserved sentinel for ``tenant_id`` on a row that predates per-tenant
# scoping (a pre-NE-007 row, migrated forward by :func:`_claim_and_migrate`).
# Deliberately NOT a real tenant, and never derived from ambient/session
# authority: a caller's/session's tenant is who is running the migration,
# not who originally owned the un-scoped historical row, and guessing the
# latter would silently mis-attribute data no verified authority ever
# claimed. Chosen to be obviously synthetic (never collides with a real
# tenant id, which this codebase always resolves from verified session/actor
# identity, never a literal containing this reserved prefix).
LEGACY_TENANT_SENTINEL = "__legacy_pre_tenant_scope__"


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
    updated_at TEXT NOT NULL,
    acl_classification TEXT,
    acl_owner_id TEXT,
    acl_shared_scope TEXT
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
    updated_at TEXT NOT NULL,
    kg_node_id TEXT,
    acl_classification TEXT,
    acl_owner_id TEXT,
    acl_shared_scope TEXT
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
    updated_at TEXT NOT NULL,
    kg_node_id TEXT,
    acl_classification TEXT,
    acl_owner_id TEXT,
    acl_shared_scope TEXT
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


_UNSTAMPED_ACL: dict[str, Any] = {
    "acl_classification": None,
    "acl_owner_id": None,
    "acl_shared_scope": None,
}


def _stamped_acl_fields(label: str) -> dict[str, Any]:
    """Best-effort ACL stamp for a catalog row, from the SAME policy the KG
    node write for ``label`` uses.

    CONCEPT:AU-KG.ingest.fleet-catalog-acl-projection. Calls
    ``tenant_sharing.stamp_ownership``/``stamp_classification`` directly
    (ambient :func:`~...security.brain_context.current_actor`, exactly the
    call every ``IntelligenceGraphEngine``/``GraphComputeEngine`` node-write
    seam makes — see ``engine.py``'s ``_upsert_node``) on a throwaway
    ``dict``, rather than re-deriving that policy's PUBLIC/CONFIDENTIAL and
    org/private rules a second time here. This is deliberate: a
    SQL-authoritative ACL projection that could drift from the KG's own
    would be worse than one that simply declines to answer, and calling the
    identical function is the only way to guarantee it never can.

    Returns :data:`_UNSTAMPED_ACL` (every field ``None``) when no verified
    actor is bound in the ambient context (``PermissionError`` from
    ``stamp_ownership``) or the stamp otherwise fails — the row this feeds
    is still written (this module's writes are never blocked by ACL
    metadata being unavailable), it simply carries no SQL-authoritative ACL
    yet. ``secured_reads.catalog_acl_rows`` treats a NULL
    ``acl_classification`` as "SQL has no opinion" and falls back to the
    existing Cypher hydration path for that id — never as "unrestricted".
    """
    from .tenant_sharing import stamp_classification, stamp_ownership

    props: dict[str, Any] = {}
    try:
        stamp_ownership(props)
        stamp_classification(props, label)
    except Exception:  # noqa: BLE001 — best-effort; the catalog write itself must never fail because of this
        return dict(_UNSTAMPED_ACL)
    return {
        "acl_classification": props.get("classification"),
        "acl_owner_id": props.get("_owner_id"),
        "acl_shared_scope": props.get("_shared_scope"),
    }


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


# ---------------------------------------------------------------------------
# NE-052 / AU-CATALOG: versioned migration ledger for an already-deployed
# (pre-NE-007 or partially-migrated) store. See the module docstring's
# "NE-052 / AU-CATALOG" section for the overall design.
# ---------------------------------------------------------------------------

_MIGRATION_LEDGER = "fleet_catalog_schema_migrations"

_LEDGER_DDL = f"""CREATE TABLE IF NOT EXISTS {_MIGRATION_LEDGER} (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    claimant TEXT NOT NULL,
    claimed_at TEXT NOT NULL,
    version BIGINT NOT NULL,
    migration_id TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TEXT NOT NULL
)"""

_LOCK_ROW_ID = "schema_state"
_CURRENT_MARKER = "current"
_SCHEMA_VERSION_CURRENT = 1

# The 5 tables that existed pre-NE-007 and can therefore carry legacy rows
# needing a one-time backfill. ``mcp_server_discovery`` is a brand-new
# append-only table introduced BY NE-007 — it never has legacy rows.
_STEP1_TABLES: tuple[str, ...] = (
    TABLE_MCP_SERVERS,
    TABLE_MCP_TOOLS,
    TABLE_MCP_PROMPTS,
    TABLE_MCP_RESOURCES,
    TABLE_SKILLS,
)

# NE-0XX / AU-CATALOG-ACL: the columns ``secured_reads._durable_access_rows``
# needs to answer an ACL projection straight from SQL (``classification``,
# ``_owner_id``, ``_shared_scope`` — ``external_access`` is deliberately NOT
# added: it is a source-connector-only descriptor and a fleet/first-party
# catalog row never carries one, so there is nothing genuine to store).
# ``skills`` already has an unrelated ``classification`` column (the
# skill_type DISPLAY LABEL, e.g. "Atomic Skill" — see the module docstring's
# schema section), so the ACL columns are named with an ``acl_`` prefix on
# every table for one consistent, collision-free name across all three.
# ``mcp_tools``/``skills`` also gain ``kg_node_id`` — the bare KG node id
# (``tool_<server>_<name>`` / ``skill_<server>_<name>`` / ``skill:<slug>``,
# with no discovery-grant-digest suffix), because their own ``id`` primary
# key is the immutable-per-snapshot ``_bound_row_id`` (base id + the
# discovery grant's digest, see module docstring "Desired server rows reuse
# the exact KG node-id convention" section) and therefore does NOT equal the
# KG node's own id the way ``mcp_servers.id`` already does. Reads key off
# ``kg_node_id``, never off ``id``, for those two tables.
_ACL_PROJECTION_MIGRATION: dict[str, tuple[str, ...]] = {
    TABLE_MCP_SERVERS: ("acl_classification", "acl_owner_id", "acl_shared_scope"),
    TABLE_MCP_TOOLS: (
        "kg_node_id",
        "acl_classification",
        "acl_owner_id",
        "acl_shared_scope",
    ),
    TABLE_SKILLS: (
        "kg_node_id",
        "acl_classification",
        "acl_owner_id",
        "acl_shared_scope",
    ),
}

# Ordered, forward-only migration steps. Each entry is
# ``(migration_id, {table: (new_column, ...)})``. Applied in order; a step
# already fully present (every listed column already exists) is skipped.
# Step 3 reuses ``_DISCOVERY_BINDING_MIGRATION`` verbatim — this is the SAME
# mechanism the pre-NE-052 code already used for those 3 columns, now simply
# tracked as one named, checksummed, ledgered step instead of a standalone
# loop, per "no second write path". Step 4 reuses ``_ACL_PROJECTION_MIGRATION``
# the same way.
_MIGRATION_COLUMN_STEPS: tuple[tuple[str, dict[str, tuple[str, ...]]], ...] = (
    (
        "0001_tenant_revision_idempotency",
        {
            table: ("tenant_id", "revision", "idempotency_key")
            for table in _STEP1_TABLES
        },
    ),
    ("0002_tool_schema_digest", {TABLE_MCP_TOOLS: ("schema_digest",)}),
    ("0003_discovery_binding_columns", dict(_DISCOVERY_BINDING_MIGRATION)),
    ("0004_acl_projection_columns", dict(_ACL_PROJECTION_MIGRATION)),
)

_KNOWN_MIGRATION_IDS = frozenset(mid for mid, _cols in _MIGRATION_COLUMN_STEPS)

# The exact pre-NE-007 column set per table (copied from commit 1f96b7bce,
# the last revision before the NE-007 hardening), used only to recognize a
# genuinely legacy store as a KNOWN, valid starting point — never to create
# it (this code only ever adds columns going forward).
_LEGACY_SCHEMA_COLUMNS: dict[str, frozenset[str]] = {
    TABLE_MCP_SERVERS: frozenset(
        {
            "id",
            "name",
            "transport",
            "url",
            "enabled",
            "reachable",
            "last_probe_at",
            "last_error",
            "tool_count",
            "skill_count",
            "prompt_count",
            "resource_count",
            "updated_at",
        }
    ),
    TABLE_MCP_TOOLS: frozenset(
        {
            "id",
            "server_id",
            "server_name",
            "name",
            "description",
            "input_schema",
            "tool_mode",
            "enabled",
            "updated_at",
        }
    ),
    TABLE_MCP_PROMPTS: frozenset(
        {"id", "server_id", "server_name", "name", "description", "uri", "updated_at"}
    ),
    TABLE_MCP_RESOURCES: frozenset(
        {
            "id",
            "server_id",
            "server_name",
            "uri",
            "name",
            "description",
            "mime_type",
            "resource_kind",
            "updated_at",
        }
    ),
    TABLE_SKILLS: frozenset(
        {
            "id",
            "name",
            "description",
            "uri",
            "skill_type",
            "classification",
            "provider",
            "mcp_server",
            "enabled",
            "updated_at",
        }
    ),
    # mcp_server_discovery did not exist pre-NE-007 — no legacy baseline.
}


def _parse_ddl_columns(ddl: str) -> frozenset[str]:
    """Column names declared by one ``CREATE TABLE (...)`` DDL string.

    Derives the "current" expected schema straight from the frozen
    :data:`_DDL` text rather than a hand-maintained parallel list, so the two
    can never drift apart. Safe for this module's DDL specifically: no
    column definition here contains a literal comma (no ``DEFAULT '...,...'``
    etc.), so a plain top-level split is exact.
    """
    body = ddl[ddl.index("(") + 1 : ddl.rindex(")")]
    columns: set[str] = set()
    for part in body.split(","):
        token = part.strip().split()
        if token:
            columns.add(token[0])
    return frozenset(columns)


_CURRENT_SCHEMA_COLUMNS: dict[str, frozenset[str]] = {
    table: _parse_ddl_columns(ddl) for table, ddl in _DDL.items()
}


def _table_schema_digest(columns: dict[str, set[str]]) -> str:
    payload = {table: sorted(cols) for table, cols in columns.items()}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _step_checksum(migration_id: str, table_columns: dict[str, tuple[str, ...]]) -> str:
    payload = {migration_id: {t: sorted(c) for t, c in table_columns.items()}}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _steps_needed(current_columns: dict[str, set[str]]) -> list[str]:
    """Which ordered migration steps still have a missing column, in order."""
    needed = []
    for migration_id, table_columns in _MIGRATION_COLUMN_STEPS:
        if any(
            set(columns) - current_columns.get(table, set())
            for table, columns in table_columns.items()
        ):
            needed.append(migration_id)
    return needed


def _is_reachable_state(
    table: str, current: set[str], legacy: frozenset[str], expected: frozenset[str]
) -> bool:
    """Is ``current`` the legacy shape, the current shape, or a valid
    in-between point on the ordered forward-only migration path — never a
    step applied out of order or only partially.

    A store reaches an in-between point one of TWO ways, and both are valid:

    * **Migrated up** from the pre-NE-007 ``legacy`` shape — this code never
      drops a column, so such a store keeps every legacy column (including
      the observed-discovery ones ``mcp_servers`` no longer declares) plus
      the columns each applied step added. Walk the steps forward from
      ``legacy``.
    * **Created fresh** by an earlier post-legacy code version — its
      ``CREATE TABLE`` was the then-current DDL, so it has today's
      ``expected`` shape MINUS every column a LATER step introduced, and it
      never had the retired legacy columns at all. Peel the steps back in
      reverse from ``expected``.

    Modelling only the first (the original bug) mis-classified every
    fresh-created store as diverged the moment a new column step landed —
    measured live 2026-08-25 on the graph-os catalog, whose ``mcp_servers``
    was created at step ``0003`` and so matched neither ``legacy`` nor any
    forward accumulation. That raised :class:`FleetCatalogSchemaDivergedError`
    out of :func:`ensure_fleet_catalog_tables`, which
    :func:`~.source_sync._write_fleet_relational` caught and degraded to
    ``{"status": "error"}`` — silently skipping the ENTIRE relational
    catalog write on every sync while the KG node write succeeded.
    """
    if current == legacy or current == expected:
        return True
    accumulated = set(legacy)
    for _migration_id, table_columns in _MIGRATION_COLUMN_STEPS:
        columns = table_columns.get(table)
        if not columns:
            continue
        accumulated |= set(columns)
        if current == accumulated:
            return True
    remaining = set(expected)
    for _migration_id, table_columns in reversed(_MIGRATION_COLUMN_STEPS):
        columns = table_columns.get(table)
        if not columns:
            continue
        remaining -= set(columns)
        if current == remaining:
            return True
    return False


def _detect_diverged_schema(current_columns: dict[str, set[str]]) -> str | None:
    """Return a human-readable reason if a table's columns match no known
    schema generation, else ``None``."""
    for table in _DDL:
        current = current_columns.get(table, set())
        if not current:
            continue  # table does not exist yet -- CREATE TABLE establishes it fresh
        expected = _CURRENT_SCHEMA_COLUMNS[table]
        legacy = _LEGACY_SCHEMA_COLUMNS.get(table, frozenset())
        unexpected = current - (expected | legacy)
        if unexpected:
            return (
                f"table {table!r} has unrecognized column(s) {sorted(unexpected)} "
                "not part of any known fleet-catalog schema generation"
            )
        if not _is_reachable_state(table, current, legacy, expected):
            return (
                f"table {table!r} column set {sorted(current)} does not "
                "correspond to any known point on the forward-only "
                "migration path"
            )
    return None


def _backfill_tenant_id_set(
    row: Mapping[str, Any], added_columns: list[str]
) -> str | None:
    if "tenant_id" in added_columns and not row.get("tenant_id"):
        return f"tenant_id = {_sql_literal(LEGACY_TENANT_SENTINEL)}"
    return None


def _backfill_revision_set(
    row: Mapping[str, Any], added_columns: list[str]
) -> str | None:
    if "revision" in added_columns and not row.get("revision"):
        return f"revision = {_sql_literal(0)}"
    return None


def _backfill_idempotency_key_set(
    row: Mapping[str, Any], added_columns: list[str], row_id: Any
) -> str | None:
    if "idempotency_key" in added_columns and not row.get("idempotency_key"):
        return f"idempotency_key = {_sql_literal(f'legacy-migration-{row_id}')}"
    return None


def _backfill_schema_digest_set(
    table: str, row: Mapping[str, Any], added_columns: list[str]
) -> str | None:
    if not (
        table == TABLE_MCP_TOOLS
        and "schema_digest" in added_columns
        and not row.get("schema_digest")
    ):
        return None
    raw_schema = row.get("input_schema")
    try:
        parsed_schema = (
            json.loads(raw_schema) if isinstance(raw_schema, str) and raw_schema else {}
        )
    except (TypeError, ValueError):
        parsed_schema = {}
    if not isinstance(parsed_schema, dict):
        parsed_schema = {}
    return f"schema_digest = {_sql_literal(_schema_digest(parsed_schema))}"


def _backfill_kg_node_id_set(
    table: str, row: Mapping[str, Any], added_columns: list[str], row_id: Any
) -> str | None:
    if not (
        table in (TABLE_MCP_TOOLS, TABLE_SKILLS)
        and "kg_node_id" in added_columns
        and not row.get("kg_node_id")
    ):
        return None
    # Deterministic reconstruction, not a guess: every row's ``id`` is
    # EITHER the bare KG node id verbatim (a genuinely pre-NE-007 row,
    # written before ``_bound_row_id`` ever appended a discovery-grant
    # suffix) OR that same bare id with ``__<digest-or-"tenant_local">``
    # appended (see :func:`_bound_row_id`) -- and the exact digest this row
    # was bound with is itself already stored in ``discovery_grant_digest``
    # (backfilled/left-NULL identically to every other discovery-binding
    # column). Stripping that exact, known suffix when present, and leaving
    # ``id`` unchanged when it is not, recovers the true KG node id in both
    # cases with no placeholder value.
    digest = str(row.get("discovery_grant_digest") or "")
    suffix = f"__{digest or DISCOVERY_AUTHORITY_TENANT_LOCAL}"
    base_id = str(row_id)
    if base_id.endswith(suffix):
        base_id = base_id[: -len(suffix)]
    return f"kg_node_id = {_sql_literal(base_id)}"


def _backfill_row_set_parts(
    table: str, row: Mapping[str, Any], added_columns: list[str], row_id: Any
) -> list[str]:
    parts = [
        _backfill_tenant_id_set(row, added_columns),
        _backfill_revision_set(row, added_columns),
        _backfill_idempotency_key_set(row, added_columns, row_id),
        _backfill_schema_digest_set(table, row, added_columns),
        _backfill_kg_node_id_set(table, row, added_columns, row_id),
    ]
    return [part for part in parts if part is not None]


def _backfill_legacy_rows(gc: Any, table: str, added_columns: list[str]) -> None:
    """One-time backfill of newly-added columns for a table's pre-existing rows.

    Per-row ``UPDATE`` (never a mass unscoped one — the engine's own SQL
    tier refuses an ``UPDATE``/``DELETE`` with no ``WHERE`` clause) is a
    deliberate, narrow exception to this module's "batch, never per-element"
    rule for steady-state writes: this runs at most once ever per legacy
    row (gated by the migration ledger), not on every ingest cycle.
    """
    rows = gc.sql_exec(f"SELECT * FROM {_safe_ident(table)}")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        row_id = row.get("id")
        if row_id is None:
            continue
        set_parts = _backfill_row_set_parts(table, row, added_columns, row_id)
        if not set_parts:
            continue
        gc.sql_exec(
            f"UPDATE {_safe_ident(table)} SET {', '.join(set_parts)} "
            f"WHERE id = {_sql_literal(row_id)}"
        )


def _add_missing_columns(
    gc: Any, table: str, columns: set[str], existing: set[str]
) -> list[str]:
    newly_added: list[str] = []
    for column in columns:
        if column in existing:
            continue
        col_type = "BIGINT" if column == "revision" else "TEXT"
        gc.sql_exec(
            f"ALTER TABLE {_safe_ident(table)} ADD COLUMN "
            f"{_safe_ident(column)} {col_type}"
        )
        existing.add(column)
        newly_added.append(column)
    return newly_added


def _apply_step_for_table(
    gc: Any,
    migration_id: str,
    table: str,
    columns: set[str],
    current_columns: dict[str, set[str]],
) -> None:
    existing = current_columns.setdefault(table, set())
    newly_added = _add_missing_columns(gc, table, columns, existing)
    if not newly_added:
        return
    # Discovery-binding columns are deliberately left NULL/unbound for
    # legacy rows (see module docstring / _DISCOVERY_BINDING_MIGRATION)
    # -- every OTHER step's newly-added columns get a real backfill.
    # (Step 4's ``acl_classification``/``acl_owner_id``/``acl_shared_scope``
    # are the SAME kind of deliberately-left-NULL case -- there is no
    # verified actor context to recover for a pre-existing row, so
    # :func:`_backfill_legacy_rows` only reconstructs that step's
    # ``kg_node_id`` and leaves the ACL columns unset; a NULL
    # ``acl_classification`` is exactly what tells a reader "SQL has no
    # opinion for this row", never "unrestricted".)
    if migration_id != "0003_discovery_binding_columns":
        _backfill_legacy_rows(gc, table, newly_added)


def _apply_step(
    gc: Any, migration_id: str, current_columns: dict[str, set[str]]
) -> None:
    table_columns = next(
        cols for mid, cols in _MIGRATION_COLUMN_STEPS if mid == migration_id
    )
    for table, columns in table_columns.items():
        _apply_step_for_table(gc, migration_id, table, columns, current_columns)
    checksum = _step_checksum(migration_id, table_columns)
    _ledger_put(
        gc,
        "step__" + migration_id,
        {
            "status": "applied",
            "claimant": "",
            "claimed_at": "",
            "version": 0,
            "migration_id": migration_id,
            "checksum": checksum,
            "applied_at": _now_iso(),
        },
        overwrite=False,
    )


_LEDGER_COLUMNS: tuple[str, ...] = (
    "status",
    "claimant",
    "claimed_at",
    "version",
    "migration_id",
    "checksum",
    "applied_at",
)


def _ledger_put(
    gc: Any, row_id: str, values: dict[str, Any], *, overwrite: bool
) -> None:
    """Write one migration-ledger row WITHOUT an ``ON CONFLICT`` clause.

    Measured live 2026-08-25 against the deployed engine: its SQL tier
    IGNORES ``ON CONFLICT`` entirely — ``DO NOTHING`` and ``DO UPDATE``
    both raise the same bare "duplicate key value violates unique
    constraint" error a plain ``INSERT`` does. Every ledger write used one,
    so a store that had already recorded a completed migration could never
    claim the lock for the NEXT step: the claim ``INSERT`` raised straight
    out of :func:`ensure_fleet_catalog_tables`, and
    :func:`~.source_sync._write_fleet_relational` degraded that to "the
    relational catalog was not written" on every sync.

    So the ledger is written the way :func:`_cas_batch_upsert` already
    writes every catalog row on this tier — read first, then ``INSERT`` a
    new id or ``UPDATE`` an existing one. ``overwrite=False`` reproduces
    ``DO NOTHING`` (an existing row is left exactly as it is);
    ``overwrite=True`` reproduces ``DO UPDATE``.
    """
    if _read_ledger_row(gc, row_id) is not None:
        if not overwrite:
            return
        assignments = ", ".join(
            f"{_safe_ident(column)} = {_sql_literal(values[column])}"
            for column in _LEDGER_COLUMNS
        )
        gc.sql_exec(
            f"UPDATE {_MIGRATION_LEDGER} SET {assignments} "
            f"WHERE id = {_sql_literal(row_id)}"
        )
        return
    columns = ", ".join(("id", *_LEDGER_COLUMNS))
    literals = ", ".join(
        _sql_literal(value)
        for value in (row_id, *(values[column] for column in _LEDGER_COLUMNS))
    )
    gc.sql_exec(f"INSERT INTO {_MIGRATION_LEDGER} ({columns}) VALUES ({literals})")


def _read_ledger_row(gc: Any, row_id: str) -> dict[str, Any] | None:
    try:
        rows = gc.sql_exec(
            f"SELECT * FROM {_MIGRATION_LEDGER} WHERE id = {_sql_literal(row_id)}"
        )
    except Exception:  # noqa: BLE001 - best-effort ledger read, caller decides fallback
        logger.debug("fleet catalog migration ledger read failed for %s", row_id)
        return None
    if not isinstance(rows, list) or not rows:
        return None
    row = rows[0]
    return dict(row) if isinstance(row, Mapping) else None


def _finalize_ledger(gc: Any, columns: dict[str, set[str]], *, claimant: str) -> None:
    digest = _table_schema_digest(columns)
    now = _now_iso()
    _ledger_put(
        gc,
        _LOCK_ROW_ID,
        {
            "status": "complete",
            "claimant": claimant,
            "claimed_at": now,
            "version": _SCHEMA_VERSION_CURRENT,
            "migration_id": _CURRENT_MARKER,
            "checksum": digest,
            "applied_at": now,
        },
        overwrite=True,
    )


# How long a migration claim stays valid. The claim is a LEASE, not a
# permanent lock: a migrator that dies mid-step (measured live 2026-08-25 —
# the graph-os container was OOM-killed while applying step 0004) leaves its
# ``migrating`` row behind forever, and treating that as a live claim wedges
# the store permanently, with a HALF-APPLIED schema and every subsequent
# fleet-catalog write skipped. Generous enough that a genuinely running
# migration is never stolen (a step is a handful of ``ALTER TABLE``s plus a
# bounded backfill), short enough that a crash self-heals on the next sync.
_MIGRATION_CLAIM_LEASE_SEC = 900.0


def _claim_is_live(lock_row: dict[str, Any]) -> bool:
    """Is this ``migrating`` ledger row a claim another process still holds?

    A claim whose ``claimed_at`` is older than
    :data:`_MIGRATION_CLAIM_LEASE_SEC`, or whose timestamp cannot be read at
    all, is treated as ABANDONED and may be taken over — being wedged
    forever behind a dead migrator is strictly worse than the bounded risk
    of two migrators overlapping, which the re-read after the claim already
    detects and which every step is independently idempotent against
    (``_steps_needed`` re-derives what is outstanding from the store's real
    columns on every attempt).
    """
    claimed_at = str(lock_row.get("claimed_at") or "")
    try:
        claimed = datetime.fromisoformat(claimed_at)
    except ValueError:
        return False
    if claimed.tzinfo is None:
        claimed = claimed.replace(tzinfo=UTC)
    return (datetime.now(UTC) - claimed).total_seconds() < _MIGRATION_CLAIM_LEASE_SEC


def _claim_and_migrate(engine: Any) -> bool:
    """Detect, migrate (if needed), verify, and record the fleet-catalog schema.

    See the module docstring's "NE-052 / AU-CATALOG" section. Returns
    ``False`` for the same best-effort reasons :func:`ensure_fleet_catalog_tables`
    always has (no SQL surface, a transient read failure, losing a
    concurrent migration claim) — never raises for those. Raises
    :class:`FleetCatalogSchemaDivergedError` / :class:`FleetCatalogSchemaTooNewError`
    when the store's schema cannot be safely verified — a deliberate
    fail-closed exception to this module's usual contract.
    """
    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec"):
        return False

    current_columns = _collect_current_columns(gc)
    if current_columns is None:
        return False

    diverged_reason = _detect_diverged_schema(current_columns)
    if diverged_reason:
        raise FleetCatalogSchemaDivergedError(diverged_reason)

    lock_row = _read_ledger_row(gc, _LOCK_ROW_ID)
    _reject_unknown_recorded_migration(lock_row)

    needed = _steps_needed(current_columns)
    if not needed:
        if _ledger_needs_finalization(lock_row):
            _finalize_ledger(gc, current_columns, claimant="")
        return True

    if not _claim_is_available(lock_row):
        return False

    token, already_complete = _acquire_migration_claim(gc)
    if token is None:
        return already_complete

    for migration_id in needed:
        _apply_step(gc, migration_id, current_columns)

    verified_columns = _verify_post_migration_columns(gc)
    _finalize_ledger(gc, verified_columns, claimant=token)
    return True


def _collect_current_columns(gc: Any) -> dict[str, set[str]] | None:
    """DDL-ensure every table + ledger, then read back each table's actual
    columns. Returns ``None`` (caller returns ``False``) if any table's
    columns could not be read."""
    for ddl in _DDL.values():
        gc.sql_exec(ddl)
    gc.sql_exec(_LEDGER_DDL)

    current_columns: dict[str, set[str]] = {}
    for table in _DDL:
        cols = _existing_table_columns(gc, table)
        if cols is None:
            return None
        current_columns[table] = cols
    return current_columns


def _reject_unknown_recorded_migration(lock_row: dict[str, Any] | None) -> None:
    if lock_row is None:
        return
    recorded = str(lock_row.get("migration_id") or "")
    if (
        recorded
        and recorded != _CURRENT_MARKER
        and recorded not in _KNOWN_MIGRATION_IDS
    ):
        raise FleetCatalogSchemaTooNewError(
            f"fleet catalog migration ledger records unknown migration "
            f"{recorded!r}; this code version cannot verify or extend "
            "that schema"
        )


def _ledger_needs_finalization(lock_row: dict[str, Any] | None) -> bool:
    return lock_row is None or lock_row.get("status") != "complete"


def _claim_is_available(lock_row: dict[str, Any] | None) -> bool:
    """``False`` when another process holds a LIVE ``migrating`` claim
    (caller must not proceed); ``True`` either when there is no conflicting
    claim, or when a prior claim has expired and may be taken over."""
    if lock_row is None or str(lock_row.get("status")) != "migrating":
        return True
    if _claim_is_live(lock_row):
        # Another process holds a LIVE claim — never take that over.
        logger.info(
            "fleet catalog schema migration already claimed by another "
            "process; skipping this attempt (will retry on the next call)"
        )
        return False
    logger.warning(
        "fleet catalog schema migration claim from %s has expired; taking it over",
        lock_row.get("claimed_at"),
    )
    return True


def _acquire_migration_claim(gc: Any) -> tuple[str | None, bool]:
    """Attempt the ``migrating`` claim. Returns ``(token, False)`` when this
    call won the claim; ``(None, True)`` when it lost the race but another
    process already finished (a no-op success); ``(None, False)`` when it
    lost the race and no one has finished (a no-op non-success, retry on the
    next call)."""
    token = uuid.uuid4().hex
    # ``overwrite=True``, deliberately: a ``schema_state`` row marked
    # ``complete`` records that the schema was current AT THE TIME — it is
    # not a live claim (that case returned above), and it must not veto the
    # NEXT step. ``needed`` above is computed from the store's ACTUAL
    # columns, so reaching this line already means a step is genuinely
    # outstanding. Every real store that has ever finished a migration
    # carries such a row, so the old ``DO NOTHING`` claim could never
    # migrate one: the re-read below took the "someone else already
    # finished" short-circuit and returned success having applied nothing,
    # and the write then failed at INSERT time on the very columns the
    # skipped step adds.
    _ledger_put(
        gc,
        _LOCK_ROW_ID,
        {
            "status": "migrating",
            "claimant": token,
            "claimed_at": _now_iso(),
            "version": 0,
            "migration_id": "",
            "checksum": "",
            "applied_at": "",
        },
        overwrite=True,
    )
    claimed = _read_ledger_row(gc, _LOCK_ROW_ID)
    if claimed is not None and str(claimed.get("claimant")) == token:
        return token, False
    if claimed is not None and claimed.get("status") == "complete":
        return None, True  # someone else already finished -- no-op success
    logger.info(
        "fleet catalog schema migration already claimed by another "
        "process; skipping this attempt (will retry on the next call)"
    )
    return None, False  # lost the race -- a genuine no-op, not an error


def _verify_post_migration_columns(gc: Any) -> dict[str, set[str]]:
    verified_columns: dict[str, set[str]] = {}
    for table in _DDL:
        cols = _existing_table_columns(gc, table)
        if cols is None:
            raise FleetCatalogSchemaDivergedError(
                "post-migration verification could not read back the schema"
            )
        verified_columns[table] = cols
    if _steps_needed(verified_columns):
        raise FleetCatalogSchemaDivergedError(
            "post-migration verification failed: schema still does not "
            "match the expected current shape"
        )
    return verified_columns


def ensure_fleet_catalog_tables(engine: Any) -> bool:
    """Ensure all 6 fleet-catalog tables exist AND are migrated to the
    current (NE-007/NE-052) shape, once per store per process.

    Returns ``False`` (never raises) when the engine has no SQL surface, a
    schema read failed transiently, or a concurrent migration was lost to
    another process — the same graceful degrade :mod:`~.table_ingest` uses,
    so a caller that also writes KG nodes is never blocked by this not being
    ready yet (e.g. in a unit test with a bare fake engine). The one
    exception: raises :class:`FleetCatalogMigrationError` when the store's
    schema cannot be safely verified/migrated (unknown/newer or
    diverged/hand-modified) — see the module docstring's "NE-052 /
    AU-CATALOG" section; that case must never be swallowed into a silent
    skip.
    """
    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec"):
        return False
    key = id(gc)
    if key in _ensured_stores:
        return True
    ok = _claim_and_migrate(engine)
    if ok:
        _ensured_stores.add(key)
    return ok


# Rows per SQL statement. Batched writing is required (the
# ``check-no-per-element-ingest-loop`` gate, and the module docstring's
# "batched, never per-element" contract), but a batch of UNBOUNDED size is a
# different failure: the live fleet probes ~9,600 tools, and rendering all of
# them into one ``INSERT ... VALUES`` (each carrying a full ``input_schema``
# JSON blob) built a multi-megabyte statement that had to be held in memory
# by this process, serialized to the engine, and parsed there all at once.
# Measured live 2026-08-25: doing that inside the graph-os container
# OOM-killed it (10Gi limit). Chunking keeps the write batched — ~20
# statements for the whole fleet's tools instead of ~9,600 — while bounding
# peak statement size.
_MAX_ROWS_PER_STATEMENT = 500


def _chunks(items: list[Any], size: int = _MAX_ROWS_PER_STATEMENT) -> Iterator[list]:
    """Split ``items`` into consecutive lists of at most ``size`` entries."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _select_existing(
    engine: Any,
    table: str,
    tenant_id: str,
    ids: list[str],
    *,
    id_col: str = "id",
) -> dict[str, dict[str, Any]]:
    """Batched ``SELECT``s for every id in ``ids`` — never one per row.

    Returns ``{id: row}`` for whatever currently exists (scoped to
    ``tenant_id``, so a row from another tenant can never be read back as
    "existing" here even if an id collided). Issued in chunks of
    :data:`_MAX_ROWS_PER_STATEMENT` ids — see that constant. Best-effort: an
    engine with no read surface, or a query failure, degrades to "nothing
    exists yet" (every row in the caller's batch is then treated as new)
    rather than raising — consistent with this module never blocking the
    write path it supports.
    """
    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec") or not ids:
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for chunk in _chunks(ids):
        chunk_rows = _select_existing_chunk(gc, table, id_col, tenant_id, chunk)
        if chunk_rows is None:
            return {}
        existing.update(chunk_rows)
    return existing


def _select_existing_chunk(
    gc: Any, table: str, id_col: str, tenant_id: str, chunk: list[str]
) -> dict[str, dict[str, Any]] | None:
    """Rows for one id-chunk, or ``None`` on a query failure -- the caller
    degrades to "nothing exists yet" for the WHOLE batch on any chunk
    failure, matching the original single-return-point behavior."""
    tbl = _safe_ident(table)
    col = _safe_ident(id_col)
    id_list = ", ".join(_sql_literal(row_id) for row_id in chunk)
    stmt = (
        f"SELECT * FROM {tbl} WHERE tenant_id = {_sql_literal(tenant_id)} "
        f"AND {col} IN ({id_list})"
    )
    try:
        rows = gc.sql_exec(stmt)
    except Exception:  # noqa: BLE001 — CAS read is best-effort
        logger.debug("fleet catalog CAS read failed for %s", table)
        return None
    chunk_rows: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        if isinstance(row, dict) and row.get(id_col) is not None:
            chunk_rows[str(row[id_col])] = row
    return chunk_rows


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _collect_acl_rows(
    rows: Any,
    id_col: str,
    tenant_id: str,
    result: dict[str, dict[str, Any]],
    best_revision: dict[str, int],
) -> None:
    """Fold one table's SQL rows into ``result``, keeping the newest per id.

    A ``mcp_tools``/``skills`` row's ``kg_node_id`` is NOT unique across the
    whole table by itself — every distinct discovery-grant snapshot of the
    same logical tool/skill is its own immutable row (module docstring,
    "Discovery-derived rows... one row per distinct observation") — so more
    than one row can legitimately share a ``kg_node_id``. The row with the
    greatest ``revision`` is the most recently observed one and is treated
    as the current answer; an older sibling is superseded, not merged.
    ``mcp_servers`` never has this collision (its ``id`` already is the KG
    node id, one row per server), so it simply always "wins" with
    ``best_revision`` starting empty for it.

    A row whose ``acl_classification`` is empty/NULL is skipped entirely —
    that is a legacy/un-stamped catalog row (see the ACL-projection
    migration's backfill posture): the caller must treat that id as "SQL has
    no opinion", identical to the id not being in the catalog at all, never
    as "unrestricted".
    """
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        node_id = row.get(id_col)
        if not isinstance(node_id, str) or not node_id:
            continue
        classification = row.get("acl_classification")
        if not classification:
            continue
        revision = _as_int(row.get("revision"))
        if node_id in best_revision and revision <= best_revision[node_id]:
            continue
        best_revision[node_id] = revision
        result[node_id] = {
            "tenant_id": tenant_id,
            "classification": classification,
            "external_access": None,
            "owner_id": row.get("acl_owner_id"),
            "shared_scope": row.get("acl_shared_scope"),
        }


def catalog_acl_rows(
    engine: Any, node_ids: list[str], tenant_id: str
) -> dict[str, dict[str, Any]]:
    """SQL-authoritative ACL projection for fleet-catalog node ids.

    CONCEPT:AU-KG.ingest.fleet-catalog-acl-projection — the SQL half of the
    fix ``secured_reads._durable_access_rows`` was deferred on: the two
    unlabeled Cypher full scans per tool the production incident measured
    are, for a fleet ``Tool``/``MCPServer``/``Skill`` node, now answerable
    from an indexed ``SELECT`` against ``mcp_servers``/``mcp_tools``/
    ``skills`` instead. Looks ``node_ids`` up against ``mcp_servers`` (by
    ``id``, which already equals the KG node id) and ``mcp_tools``/
    ``skills`` (by ``kg_node_id`` — their own ``id`` is a
    discovery-grant-suffixed row identity, NOT the KG node id; see
    :data:`_ACL_PROJECTION_MIGRATION`), scoped to ``tenant_id`` so a
    cross-tenant catalog row can never answer for this caller.

    Returns ``{node_id: {"tenant_id", "classification", "external_access":
    None, "owner_id", "shared_scope"}}`` — the exact shape
    ``secured_reads._durable_access_rows`` already builds from its Cypher
    rows — but ONLY for a ``node_id`` whose matched catalog row carries a
    non-empty ``acl_classification`` (see :func:`_collect_acl_rows`).
    ``external_access`` is always ``None``: a fleet/first-party catalog row
    is never source-connector-sourced, so there is no genuine value to
    report, never a placeholder.

    An id absent from every table, OR present but with no stamped ACL yet
    (a legacy row, or a write whose ``_stamped_acl_fields`` call found no
    verified actor), is simply OMITTED from the returned dict — the caller
    MUST fall back to the Cypher hydration path for that id; this function
    never returns a partial/guessed answer for an id it cannot fully back.

    Best-effort like every other read in this module: an engine with no SQL
    surface, an unmigrated/unreadable schema, or a query failure all return
    ``{}`` (nothing resolved via SQL — a pure "no fast path today", never a
    grant) so the caller's existing Cypher fallback is completely
    unaffected. Never raises.
    """
    gc = _graph_compute(engine)
    if _acl_lookup_unusable(gc, node_ids, tenant_id):
        return {}
    try:
        if not ensure_fleet_catalog_tables(engine):
            return {}
    except FleetCatalogMigrationError:
        # A store this code cannot safely verify/migrate must not be read
        # from either -- identical fail-closed posture to the write side.
        return {}

    ids = _resolve_acl_query_ids(node_ids)
    if not ids:
        return {}

    result = _collect_catalog_acl_query_results(gc, ids, tenant_id)
    if result is None:
        return {}

    # Defence-in-depth: only ever answer for an id actually asked about, and
    # never let a duplicate/short-circuited id sneak in even if a future
    # change to the SELECTs above widened the WHERE clause.
    return {node_id: row for node_id, row in result.items() if node_id in ids}


def _acl_lookup_unusable(gc: Any, node_ids: list[str], tenant_id: str) -> bool:
    return gc is None or not hasattr(gc, "sql_exec") or not node_ids or not tenant_id


def _resolve_acl_query_ids(node_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(str(node_id) for node_id in node_ids if node_id))


def _collect_catalog_acl_query_results(
    gc: Any, ids: list[str], tenant_id: str
) -> dict[str, dict[str, Any]] | None:
    """Query mcp_servers/mcp_tools/skills and fold into one ACL-row dict, or
    ``None`` on any query failure (caller degrades to ``{}``)."""
    id_list = ", ".join(_sql_literal(node_id) for node_id in ids)
    result: dict[str, dict[str, Any]] = {}
    try:
        # ``SELECT *`` -- the exact query shape :func:`_select_existing`
        # already uses for this engine's SQL tier (no per-column projection
        # support proven there); the columns actually used are picked out of
        # the returned row mapping by :func:`_collect_acl_rows`.
        server_rows = gc.sql_exec(
            f"SELECT * FROM {_safe_ident(TABLE_MCP_SERVERS)} "
            f"WHERE tenant_id = {_sql_literal(tenant_id)} AND id IN ({id_list})"
        )
        _collect_acl_rows(server_rows, "id", tenant_id, result, {})

        for table in (TABLE_MCP_TOOLS, TABLE_SKILLS):
            rows = gc.sql_exec(
                f"SELECT * FROM {_safe_ident(table)} "
                f"WHERE tenant_id = {_sql_literal(tenant_id)} "
                f"AND kg_node_id IN ({id_list})"
            )
            _collect_acl_rows(rows, "kg_node_id", tenant_id, result, {})
    except Exception:  # noqa: BLE001 — SQL ACL lookup is a best-effort fast path
        logger.debug("fleet catalog ACL SQL lookup failed; caller falls back to Cypher")
        return None
    return result


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

    to_insert, to_update, rejected_stale, noop_replay = _classify_cas_rows(
        rows, existing, conflict_col
    )

    tbl = _safe_ident(table)
    written = _cas_batch_insert(gc, tbl, to_insert)
    written += _cas_batch_update(gc, tbl, to_update, conflict_col, tenant_id)
    return {
        "written": written,
        "rejected_stale": rejected_stale,
        "noop_replay": noop_replay,
    }


def _classify_cas_row(
    row: dict[str, Any], existing: dict[str, dict[str, Any]], conflict_col: str
) -> str:
    """One of ``"insert"``/``"noop"``/``"stale"``/``"update"`` for ``row``
    against its existing catalog state (see :func:`_cas_batch_upsert`'s
    docstring for the exact rules)."""
    row_id = str(row[conflict_col])
    current = existing.get(row_id)
    if current is None:
        return "insert"
    if str(current.get("idempotency_key", "")) == str(row.get("idempotency_key", "")):
        return "noop"
    if _as_int(current.get("revision")) >= _as_int(row.get("revision")):
        return "stale"
    return "update"


def _classify_cas_rows(
    rows: list[dict[str, Any]],
    existing: dict[str, dict[str, Any]],
    conflict_col: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    to_insert: list[dict[str, Any]] = []
    to_update: list[dict[str, Any]] = []
    rejected_stale = 0
    noop_replay = 0
    for row in rows:
        kind = _classify_cas_row(row, existing, conflict_col)
        if kind == "insert":
            to_insert.append(row)
        elif kind == "noop":
            noop_replay += 1
        elif kind == "stale":
            rejected_stale += 1
        else:
            to_update.append(row)
    return to_insert, to_update, rejected_stale, noop_replay


def _cas_batch_insert(gc: Any, tbl: str, to_insert: list[dict[str, Any]]) -> int:
    if not to_insert:
        return 0
    written = 0
    columns = _bounded_columns(list(to_insert[0].keys()))
    for chunk in _chunks(to_insert):
        values_sql = ", ".join(
            "(" + ", ".join(_sql_literal(row.get(c)) for c in columns) + ")"
            for row in chunk
        )
        gc.sql_exec(f"INSERT INTO {tbl} ({', '.join(columns)}) VALUES {values_sql}")
        written += len(chunk)
    return written


def _cas_batch_update(
    gc: Any,
    tbl: str,
    to_update: list[dict[str, Any]],
    conflict_col: str,
    tenant_id: str,
) -> int:
    written = 0
    for row in to_update:
        columns = _bounded_columns([c for c in row if c != conflict_col])
        set_clause = ", ".join(f"{c} = {_sql_literal(row[c])}" for c in columns)
        gc.sql_exec(
            f"UPDATE {tbl} SET {set_clause} WHERE "
            f"{_safe_ident(conflict_col)} = {_sql_literal(row[conflict_col])} "
            f"AND tenant_id = {_sql_literal(tenant_id)}"
        )
        written += 1
    return written


_SKILL_CLASSIFICATION_OVERRIDES_DDL = """CREATE TABLE IF NOT EXISTS skill_classification_overrides (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    skill_type TEXT NOT NULL,
    set_by TEXT NOT NULL,
    revision BIGINT NOT NULL,
    idempotency_key TEXT NOT NULL,
    updated_at TEXT NOT NULL
)"""

# One-time-per-store DDL cache, mirroring ``_ensured_stores`` but kept
# separate: this table's readiness is independent of the 6-table NE-052
# migration ledger (see the ``TABLE_SKILL_CLASSIFICATION_OVERRIDES`` comment).
_ensured_override_stores: set[int] = set()


def _ensure_skill_classification_overrides_table(engine: Any) -> bool:
    """Idempotently create the override table. Best-effort, never raises.

    No migration ledger, no legacy shape -- this table did not exist before
    this feature, so there is nothing to migrate FROM. A fresh
    ``CREATE TABLE IF NOT EXISTS`` is the whole contract.
    """
    gc = _graph_compute(engine)
    if gc is None or not hasattr(gc, "sql_exec"):
        return False
    key = id(gc)
    if key in _ensured_override_stores:
        return True
    try:
        gc.sql_exec(_SKILL_CLASSIFICATION_OVERRIDES_DDL)
    except Exception as exc:  # noqa: BLE001 -- best-effort, mirrors sibling ensure fns
        logger.warning(
            "skill_classification_overrides table creation failed (%s)",
            type(exc).__name__,
        )
        return False
    _ensured_override_stores.add(key)
    return True


def read_skill_classification_override(
    engine: Any, *, skill_id: str, tenant_id: str
) -> str | None:
    """Return the operator-assigned ``skill_type`` override for ``skill_id``, if any.

    ``skill_id`` is the UNBOUND base id (e.g. ``"skill:<slug>"``, never the
    ``__<grant-digest>``-suffixed catalog row id) -- the same stable key
    :func:`write_skill_classification_override` stores under, so an override
    survives every re-ingest of that skill regardless of which discovery
    binding wrote the surrounding row. Returns ``None`` (never raises) when
    the table is unavailable, unreadable, or genuinely has no override --
    callers treat "no override" the same as "read failed": fall back to the
    ingester's own declared value.
    """
    if not _ensure_skill_classification_overrides_table(engine):
        return None
    existing = _select_existing(
        engine, TABLE_SKILL_CLASSIFICATION_OVERRIDES, tenant_id, [skill_id]
    )
    row = existing.get(skill_id)
    if row is None:
        return None
    value = str(row.get("skill_type") or "").strip().lower()
    return value or None


def write_skill_classification_override(
    engine: Any,
    *,
    skill_id: str,
    skill_type: str,
    principal: str,
    revision: int | None = None,
) -> bool:
    """Durably record an operator's classification choice for ``skill_id``.

    This is the mechanism that lets a classification survive the next
    ``fleet-tool-schema-sync`` re-derive: :func:`write_skill_row` consults
    this override (via :func:`read_skill_classification_override`) BEFORE
    calling :func:`classify_skill_type` on whatever the corpus/caller
    declared, so every future re-ingest of this skill resolves to the
    override until it is changed again here. Stored in the engine's own SQL
    catalog store (always writable by this process, unlike the NFS-mounted,
    read-only ``universal-skills`` source tree in every deployed profile) --
    never the filesystem.

    Returns ``True`` only when the write is provably durable: a fresh row
    written, an existing row updated, or an identical replay of the same
    value already on record (idempotent no-op). Returns ``False`` on any
    failure or a stale-revision rejection -- callers MUST NOT report success
    on ``False`` (fail-closed).
    """
    if not _ensure_skill_classification_overrides_table(engine):
        return False
    tenant_id = _resolve_tenant_id(engine)
    if not tenant_id:
        return False
    content = {"skill_type": skill_type, "set_by": principal}
    row = {
        "id": skill_id,
        "tenant_id": tenant_id,
        "skill_type": skill_type,
        "set_by": principal,
        "revision": _default_revision() if revision is None else int(revision),
        "idempotency_key": _content_signature(content),
        "updated_at": _now_iso(),
    }
    try:
        stats = _cas_batch_upsert(engine, TABLE_SKILL_CLASSIFICATION_OVERRIDES, [row])
    except Exception as exc:  # noqa: BLE001 -- fail closed, never raise into the caller
        logger.warning(
            "skill classification override write failed (%s)", type(exc).__name__
        )
        return False
    return stats["written"] > 0 or stats["noop_replay"] > 0


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
    acl: dict[str, Any] | None = None,
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
        # ``kg_node_id`` is ``skill_id`` verbatim — the bare KG ``Skill``
        # node id, never suffixed with the discovery-grant digest the way
        # this row's own ``id`` (below) is. See the ACL-projection migration
        # comment (:data:`_ACL_PROJECTION_MIGRATION`) for why the two must
        # differ and why a reader needs both.
        "kg_node_id": skill_id,
        **(acl if acl is not None else _stamped_acl_fields("Skill")),
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
    surface, when the write was CAS-rejected (stale), or when it was a no-op
    (idempotent replay of an already-stored row, identical content) — the
    caller (:func:`~..ingestion.skill_workflow_ingest.ingest_runnable_skill`)
    never inspects this return value in the routine ingest path, so this is
    purely observational there. :func:`~..ingestion.skill_classification.reclassify_skill`
    DOES need to tell "genuinely refreshed" apart from "already correct" vs.
    "failed" — it re-reads the row afterward rather than trusting this
    boolean alone, since a no-op replay (content already matches) and a
    failure both return ``False`` here.
    ``revision``/``idempotency_key`` are optional — omitted, they default to
    a wall-clock revision and a content digest of the row respectively (see
    :func:`_default_revision`/:func:`_content_signature`), so an unchanged
    re-ingest of the same skill (routine at every GraphOS boot) is a no-op
    rather than a redundant write.

    **Override precedence** (CONCEPT:AU-KG.ingest.skill-classification-writeback):
    before ``skill_type`` is normalized, :func:`read_skill_classification_override`
    is consulted for this ``skill_id``. When an operator has classified this
    skill through the write-back capability, that choice wins over whatever
    the caller (frontmatter parse, MCP harvest, etc.) declared -- this is
    what makes an operator classification survive the next
    ``fleet-tool-schema-sync`` re-derive instead of being silently reverted.
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
    override = read_skill_classification_override(
        engine, skill_id=skill_id, tenant_id=tenant_id
    )
    row = _build_skill_row(
        skill_id=skill_id,
        name=name,
        description=description,
        uri=uri,
        provider=provider,
        mcp_server=mcp_server,
        skill_type=override or skill_type,
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


def get_skill_row(engine: Any, *, skill_id: str) -> dict[str, Any] | None:
    """Read one CURRENT ``skills`` row by its catalog (bound) id.

    Tenant-scoped to this process's own verified authority, matching every
    other read in this module. Returns ``None`` (never raises) when the
    catalog is unavailable or the id does not exist for this tenant --
    callers must treat that as "cannot reclassify, unknown skill", never as
    an empty-but-valid row.

    Public (unlike ``_select_existing``) because
    :mod:`~..ingestion.skill_classification` needs to preserve a row's
    existing description/uri/provider/mcp_server/enabled when only its
    classification is changing, and reaching into this module's private CAS
    internals from another module would break the module-boundary
    convention every other cross-module caller here already follows (see
    this module's own docstring on ``write_fleet_catalog`` being the
    catalog's one writer).
    """
    if not ensure_fleet_catalog_tables(engine):
        return None
    tenant_id = _resolve_tenant_id(engine)
    if not tenant_id:
        return None
    existing = _select_existing(engine, TABLE_SKILLS, tenant_id, [skill_id])
    return existing.get(skill_id)


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
    acl: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    tool_name = entry.get("name")
    if not tool_name:
        return None
    input_schema = entry.get("inputSchema") or {}
    kg_node_id = f"tool_{server_name}_{tool_name}"
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
        # Bare KG ``Tool`` node id (``source_sync._write_fleet_nodes``'s own
        # ``tool_node_id``) -- NOT this row's own ``id`` below, which is
        # suffixed with the discovery grant digest. See
        # :data:`_ACL_PROJECTION_MIGRATION`.
        "kg_node_id": kg_node_id,
        **(acl if acl is not None else _stamped_acl_fields("Tool")),
    }
    return {
        "id": _bound_row_id(kg_node_id, discovery_grant_digest),
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

    # Resolved ONCE for the whole batch, not per row: every row this call
    # writes shares the same write-time ambient actor (one probe/sync
    # attempt), so this mirrors the KG node write's own per-label stamp
    # (``tenant_sharing.stamp_ownership``/``stamp_classification``) without
    # re-deriving it 1-per-row. See :func:`_stamped_acl_fields`.
    acl_server = _stamped_acl_fields("MCPServer")
    acl_tool = _stamped_acl_fields("Tool")
    acl_skill = _stamped_acl_fields("Skill")

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
            **acl_server,
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
        #
        # BUG-PE-056 — the ONE exception, and it is not an authority
        # loosening: a server whose probe FAILED never gets a binding
        # (``MCPMultiplexer._bind_local_discovery_bindings`` mints one only
        # for ``info["error"] is None``), so it used to fall out here with no
        # discovery row at all — making "unavailable" indistinguishable from
        # "empty" to the dashboard, the exact confusion this function's own
        # docstring promises never to create. A failure observation exposes
        # NO discovered capability (an errored probe has no tools/skills/
        # prompts), so recording it needs no grant — only the verified tenant
        # scope :func:`_resolve_tenant_id` already established. Record it
        # under the process-owned tenant-local visibility contract, with the
        # same empty principal/grant fields a local child's successful probe
        # carries, and fall through to the ``continue`` below so no derived
        # row is ever written for it.
        if not server_discovery_authority_ready and not reachable and tenant_id:
            discovery_authority_kind = DISCOVERY_AUTHORITY_TENANT_LOCAL
            discovery_principal = ""
            discovery_grant_digest = ""
            unreachable_content = {
                "server_id": server_id,
                "server_name": server_name,
                "reachable": False,
                "last_error": _privacy_safe(str(err or "")),
                "tool_count": 0,
                "skill_count": 0,
                "prompt_count": 0,
                "resource_count": 0,
                "discovery_authority_kind": discovery_authority_kind,
                "discovery_principal": discovery_principal,
                "discovery_grant_digest": discovery_grant_digest,
            }
            unreachable_key = idempotency_key or _content_signature(unreachable_content)
            discovery_rows.append(
                {
                    "id": f"disc_{server_id}_{unreachable_key[:24]}",
                    "tenant_id": tenant_id,
                    **unreachable_content,
                    "observed_at": now,
                    "revision": write_revision,
                    "idempotency_key": unreachable_key,
                }
            )
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
                acl=acl_tool,
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
                    acl=acl_skill,
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
