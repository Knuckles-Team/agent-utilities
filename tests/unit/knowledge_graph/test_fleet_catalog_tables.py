"""Fleet-catalog relational hardening (CONCEPT:AU-KG.ingest.fleet-catalog-relational-tables).

Exercises the NE-007/AU-CATALOG hardening of ``fleet_catalog_tables``:
explicit per-row ``tenant_id`` resolved only from verified ambient authority,
compare-and-set (revision fencing + idempotency-key dedup) applied in
application code over the engine's SQL tier (which has no composite PK / FK /
conditional upsert), the desired-registration (``mcp_servers``) vs.
observed-discovery (``mcp_server_discovery``, append-only) split, the tool
``schema_digest``, and that the batched write path issues one statement per
table rather than one per row.

``_FakeGraphCompute`` below is a small in-memory SQL emulator (not just a
statement recorder like ``table_ingest``'s fake) because CAS behavior is only
observable by actually round-tripping a ``SELECT`` against previously
``INSERT``/``UPDATE``-ed state — it parses exactly the statement shapes this
module generates (see ``_cas_batch_upsert``/``_select_existing``), not
arbitrary SQL.
"""

from __future__ import annotations

import contextvars
import re
from datetime import UTC, datetime, timedelta

import pytest

from agent_utilities.knowledge_graph.core import fleet_catalog_tables as fct
from agent_utilities.knowledge_graph.core.discovery_authority import OAuthGrantBinding
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    current_session,
    suspend_session,
    use_session,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

pytestmark = pytest.mark.concept("AU-KG.ingest.fleet-catalog-relational-tables")


# ---------------------------------------------------------------------------
# A tiny SQL emulator: enough to round-trip CREATE/SELECT/INSERT/UPDATE in
# exactly the shapes _cas_batch_upsert / _select_existing generate.
# ---------------------------------------------------------------------------


def _split_top(s: str) -> list[str]:
    """Split on top-level commas outside single-quoted string literals."""
    parts: list[str] = []
    buf = ""
    in_quote = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "'":
            if in_quote and i + 1 < len(s) and s[i + 1] == "'":
                buf += "''"
                i += 2
                continue
            in_quote = not in_quote
            buf += ch
            i += 1
            continue
        if ch == "," and not in_quote:
            parts.append(buf)
            buf = ""
            i += 1
            continue
        buf += ch
        i += 1
    parts.append(buf)
    return [p.strip() for p in parts]


def _parse_literal(tok: str):
    tok = tok.strip()
    if tok == "NULL":
        return None
    if tok == "TRUE":
        return True
    if tok == "FALSE":
        return False
    if tok.startswith("'") and tok.endswith("'"):
        return tok[1:-1].replace("''", "'")
    try:
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return tok


def _extract_value_rows(values_clause: str) -> list[str]:
    """Extract each parenthesized row's inner content from '(a,b),(c,d)'."""
    rows: list[str] = []
    buf = ""
    depth = 0
    in_quote = False
    i = 0
    while i < len(values_clause):
        ch = values_clause[i]
        if ch == "'":
            if in_quote and i + 1 < len(values_clause) and values_clause[i + 1] == "'":
                buf += "''"
                i += 2
                continue
            in_quote = not in_quote
            if depth >= 1:
                buf += ch
            i += 1
            continue
        if not in_quote and ch == "(":
            depth += 1
            if depth == 1:
                i += 1
                continue
        if not in_quote and ch == ")":
            depth -= 1
            if depth == 0:
                rows.append(buf)
                buf = ""
                i += 1
                continue
        if depth >= 1:
            buf += ch
        i += 1
    return rows


class _FakeGraphCompute:
    """In-memory SQL emulator, now schema-aware.

    Beyond the original CAS-shaped statements, this fake also has to model
    enough of the real engine's ``information_schema.columns`` / ``ALTER
    TABLE ADD COLUMN`` behavior for the
    migration path (``fleet_catalog_tables._claim_and_migrate``) to be
    meaningfully exercised:

    * ``self.columns`` tracks each table's REAL column set, parsed from the
      ``CREATE TABLE`` DDL text and extended by ``ALTER TABLE ADD COLUMN`` —
      this is what makes "already-current store issues no DDL" and "old
      store fails at INSERT time" both actually observable in a test,
      instead of the previous column-blind fake accepting any row shape.
    * An ``INSERT``/``UPDATE`` naming a column absent from the tracked
      column set raises — reproducing the real engine rejecting a write
      against an undeclared column, which is exactly NE-052's reported
      defect against a legacy pre-tenant-scope store.
    * ``INSERT ... ON CONFLICT ...`` is REFUSED, matching the deployed
      engine, whose SQL tier ignores the clause and raises a duplicate-key
      error for every form of it.
    """

    def __init__(self) -> None:
        self.statements: list[str] = []
        self.tables: dict[str, dict[str, dict]] = {}
        self.columns: dict[str, set[str]] = {}

    def _row_columns(self, table: str, cols: list[str]) -> None:
        known = self.columns.setdefault(table, set())
        if not known:
            known.update(cols)
            return
        unknown = [c for c in cols if c not in known]
        if unknown:
            raise RuntimeError(
                f"column(s) {unknown} do not exist on table {table!r} "
                f"(known columns: {sorted(known)})"
            )

    def sql_exec(self, statement: str):
        self.statements.append(statement)
        head = statement.strip().split(None, 1)[0].upper()

        if head == "CREATE":
            m = re.match(
                r"CREATE TABLE IF NOT EXISTS (\w+) \((.*)\)\s*$", statement, re.DOTALL
            )
            if m:
                table, body = m.groups()
                if table in self.tables:
                    # Real ``IF NOT EXISTS`` semantics: a full no-op against
                    # an already-existing table, including its columns —
                    # this is exactly the silent-no-op behavior NE-052
                    # exists to work around, so the fake must reproduce it
                    # rather than "helpfully" merging in the new DDL's
                    # columns.
                    return {"ok": True}
                self.tables[table] = {}
                known = self.columns.setdefault(table, set())
                for part in body.split(","):
                    tok = part.strip().split()
                    if tok:
                        known.add(tok[0])
            return {"ok": True}

        if head == "ALTER":
            m = re.match(r"ALTER TABLE (\w+) ADD COLUMN (\w+)", statement)
            assert m, f"unrecognized ALTER: {statement}"
            table, column = m.groups()
            known = self.columns.setdefault(table, set())
            if column in known:
                raise RuntimeError(
                    f"duplicate ADD COLUMN {column!r} on {table!r} — the "
                    "production migration must check existence first"
                )
            known.add(column)
            return {"ok": True}

        if head == "SELECT":
            m = re.match(
                r"SELECT column_name FROM information_schema\.columns "
                r"WHERE table_schema = 'public' AND table_name = '(\w+)'$",
                statement,
            )
            if m:
                table = m.group(1)
                return [
                    {"column_name": c} for c in sorted(self.columns.get(table, set()))
                ]

            m = re.match(
                r"SELECT \* FROM (\w+) WHERE tenant_id = (.+?) AND (\w+) IN \((.*)\)$",
                statement,
                re.DOTALL,
            )
            if m:
                table, tenant_lit, col, in_list = m.groups()
                tenant_id = _parse_literal(tenant_lit)
                ids = [_parse_literal(t) for t in _split_top(in_list)]
                store = self.tables.get(table, {})
                return [
                    dict(row)
                    for row in store.values()
                    if row.get("tenant_id") == tenant_id and row.get(col) in ids
                ]

            m = re.match(r"SELECT \* FROM (\w+) WHERE id = (.+)$", statement, re.DOTALL)
            if m:
                table, id_lit = m.groups()
                row_id = _parse_literal(id_lit.strip())
                row = self.tables.get(table, {}).get(str(row_id))
                return [dict(row)] if row else []

            m = re.match(r"SELECT \* FROM (\w+)\s*$", statement)
            if m:
                table = m.group(1)
                return [dict(row) for row in self.tables.get(table, {}).values()]

            return []

        if head == "INSERT":
            # The REAL engine ignores an ``ON CONFLICT`` clause entirely —
            # every form raises the same duplicate-key error a bare INSERT
            # does (measured live 2026-08-25 against the deployed engine).
            # This fake used to IMPLEMENT the clause, which is precisely why
            # the migration-ledger claim's dependence on it went unnoticed
            # until it broke every fleet-catalog write in production. A fake
            # that is more capable than the thing it stands in for cannot
            # catch that class of bug, so emitting the clause is now a hard
            # failure here.
            assert " ON CONFLICT " not in statement, (
                "the engine's SQL tier does not support ON CONFLICT; use a "
                f"read-then-write instead: {statement}"
            )
            m = re.match(
                r"INSERT INTO (\w+) \((.*?)\) VALUES (.*)$", statement, re.DOTALL
            )
            assert m, f"unrecognized INSERT: {statement}"
            table, cols_str, values_str = m.groups()
            cols = [c.strip() for c in cols_str.split(",")]
            self._row_columns(table, cols)
            store = self.tables.setdefault(table, {})

            for row_str in _extract_value_rows(values_str):
                vals = [_parse_literal(t) for t in _split_top(row_str)]
                row = dict(zip(cols, vals, strict=True))
                # A bare duplicate-id INSERT stays a tolerant overwrite: the
                # tenant-collision test writes the same id for two tenants
                # and depends on it. Only the ON CONFLICT assertion above
                # was tightened.
                store[str(row.get("id"))] = row
            return {"ok": True}

        if head == "UPDATE":
            m = re.match(
                r"UPDATE (\w+) SET (.*) WHERE (\w+) = (.+?) AND tenant_id = (.+)$",
                statement,
                re.DOTALL,
            )
            if m:
                table, set_str, id_col, id_lit, tenant_lit = m.groups()
                store = self.tables.setdefault(table, {})
                row_id = _parse_literal(id_lit)
                existing = dict(store.get(str(row_id), {}))
                self._row_columns(
                    table, [p.partition("=")[0].strip() for p in _split_top(set_str)]
                )
                for pair in _split_top(set_str):
                    col, _, lit = pair.partition("=")
                    existing[col.strip()] = _parse_literal(lit.strip())
                store[str(row_id)] = existing
                return {"ok": True}

            m2 = re.match(
                r"UPDATE (\w+) SET (.*) WHERE (\w+) = (.+)$", statement, re.DOTALL
            )
            assert m2, f"unrecognized UPDATE: {statement}"
            table, set_str, id_col, id_lit = m2.groups()
            store = self.tables.setdefault(table, {})
            row_id = _parse_literal(id_lit.strip())
            existing = dict(store.get(str(row_id), {}))
            self._row_columns(
                table, [p.partition("=")[0].strip() for p in _split_top(set_str)]
            )
            for pair in _split_top(set_str):
                col, _, lit = pair.partition("=")
                existing[col.strip()] = _parse_literal(lit.strip())
            store[str(row_id)] = existing
            return {"ok": True}

        raise AssertionError(f"unrecognized statement: {statement}")


class _FakeEngine:
    def __init__(self) -> None:
        self.graph_compute = _FakeGraphCompute()


@pytest.fixture(autouse=True)
def _reset_ddl_cache():
    """``_ensured_stores`` is a module-level cache keyed by ``id(graph_compute)``
    — a fresh fake engine each test gets a fresh id, but clear it anyway so
    test order/id-reuse can never leak DDL-issued state across tests."""
    fct._ensured_stores.clear()
    yield
    fct._ensured_stores.clear()


def _session(tenant: str, actor_id: str = "probe-actor") -> GraphSession:
    actor = ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id=tenant,
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=tenant,
        scopes=frozenset({"kg:read", "kg:write"}),
        graph="g",
        policy_version="v1",
        audience="test",
    )


def _server_catalog(*, error: str | None = None, description: str = "d") -> dict:
    return {
        "srv": {
            "error": error,
            "tools": [
                {
                    "name": "t1",
                    "description": description,
                    "inputSchema": {"properties": {"x": {"type": "string"}}},
                }
            ],
            "skills": [],
            "prompts": [],
        }
    }


def _test_discovery_binding() -> OAuthGrantBinding | None:
    """Supply an explicit broker-shaped binding to the writer test seam."""

    session = current_session()
    if session is None:
        return None
    return OAuthGrantBinding(
        tenant_id=session.tenant,
        principal_id=session.actor.actor_id,
        provider_id="test-provider",
        resource_url="https://test-provider.example/mcp",
        audience="https://test-provider.example/mcp",
        granted_scopes=("mcp:read",),
        key_version=1,
        grant_revision=f"test-{session.tenant}-{session.actor.actor_id}",
    )


def _test_local_discovery_binding() -> fct.TenantLocalDiscoveryBinding | None:
    session = current_session()
    if session is None:
        return None
    return fct.TenantLocalDiscoveryBinding(tenant_id=session.tenant)


def _write_fleet_catalog(eng, catalog, **kwargs):
    binding = _test_discovery_binding()
    bindings = {server: binding for server in catalog} if binding is not None else None
    return fct.write_fleet_catalog(
        eng,
        catalog,
        discovery_bindings=bindings,
        **kwargs,
    )


def _write_local_fleet_catalog(eng, catalog, **kwargs):
    binding = _test_local_discovery_binding()
    bindings = {server: binding for server in catalog} if binding is not None else None
    return fct.write_fleet_catalog(
        eng,
        catalog,
        discovery_bindings=bindings,
        **kwargs,
    )


def _write_skill_row(eng, **kwargs):
    return fct.write_skill_row(
        eng,
        discovery_binding=_test_discovery_binding(),
        **kwargs,
    )


def _rows_with_prefix(table: str, prefix: str, eng: _FakeEngine) -> list[dict]:
    return [
        row
        for row_id, row in eng.graph_compute.tables[table].items()
        if row_id == prefix or row_id.startswith(f"{prefix}__")
    ]


def _one_row(table: str, prefix: str, eng: _FakeEngine) -> dict:
    rows = _rows_with_prefix(table, prefix, eng)
    assert len(rows) == 1
    return rows[0]


# ---------------------------------------------------------------------------
# tenant_id: resolved from verified session, never caller-supplied
# ---------------------------------------------------------------------------


def test_tenant_id_comes_from_ambient_session_not_a_kwarg():
    eng = _FakeEngine()
    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        _write_fleet_catalog(eng, _server_catalog())
    row = eng.graph_compute.tables["mcp_servers"]["mcp_server_srv"]
    assert row["tenant_id"] == "tenant-a"
    # write_fleet_catalog/write_skill_row accept no tenant_id parameter at all —
    # there is no injection point for a caller-claimed tenant.
    assert "tenant_id" not in (fct.write_fleet_catalog.__kwdefaults__ or {})
    assert "tenant_id" not in (fct.write_skill_row.__kwdefaults__ or {})


def test_two_tenants_rows_never_mix():
    eng = _FakeEngine()
    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        _write_fleet_catalog(eng, _server_catalog(description="a-desc"))
    with use_actor(_session("tenant-b").actor), use_session(_session("tenant-b")):
        _write_fleet_catalog(eng, _server_catalog(description="b-desc"))

    # Both tenants wrote the SAME logical id ("mcp_server_srv") — the fake
    # store is a single flat dict keyed by id, so a real leak would show up
    # as tenant B's write silently clobbering tenant A's row's tenant_id, or
    # a CAS check reading across tenants and wrongly rejecting/no-op-ing.
    servers = eng.graph_compute.tables["mcp_servers"]
    assert servers["mcp_server_srv"]["tenant_id"] == "tenant-b"  # last write wins here
    # But the CAS existence check is tenant-scoped: tenant B's write was
    # treated as brand-new (not a stale/duplicate of tenant A's), proven by
    # both writes landing (not one being rejected as a stale replay).
    tools = _rows_with_prefix("mcp_tools", "tool_srv_t1", eng)
    assert {row["tenant_id"] for row in tools} == {"tenant-a", "tenant-b"}
    # No row anywhere carries a foreign tenant_id string leaking into content.
    for table in eng.graph_compute.tables.values():
        for row in table.values():
            assert row.get("tenant_id") in ("tenant-a", "tenant-b", None, "")


def test_write_skill_row_tenant_id_from_session():
    eng = _FakeEngine()
    with use_actor(_session("tenant-x").actor), use_session(_session("tenant-x")):
        ok = _write_skill_row(eng, skill_id="skill:foo", name="foo", description="d")
    assert ok is True
    row = _one_row("skills", "skill:foo", eng)
    assert row["tenant_id"] == "tenant-x"


# ---------------------------------------------------------------------------
# CAS: stale revision rejected, replayed idempotency key is a no-op
# ---------------------------------------------------------------------------


def test_stale_revision_write_is_rejected_and_leaves_fresh_row_intact():
    eng = _FakeEngine()
    with use_actor(_session("t").actor), use_session(_session("t")):
        _write_skill_row(
            eng,
            skill_id="skill:s",
            name="s",
            description="fresh",
            revision=10,
            idempotency_key="fresh-write",
        )
        ok = _write_skill_row(
            eng,
            skill_id="skill:s",
            name="s",
            description="STALE — must not land",
            revision=5,
            idempotency_key="stale-write",
        )
    assert ok is False
    row = _one_row("skills", "skill:s", eng)
    assert row["description"] == "fresh"
    assert row["revision"] == 10


def test_replayed_write_with_same_idempotency_key_is_a_noop():
    eng = _FakeEngine()
    with use_actor(_session("t").actor), use_session(_session("t")):
        first = _write_skill_row(
            eng,
            skill_id="skill:s",
            name="s",
            description="v1",
            revision=1,
            idempotency_key="attempt-1",
        )
        # A retried write of the exact same logical attempt: same content,
        # same idempotency key, a HIGHER revision (as a real retry would
        # carry, since revision often defaults to wall-clock) — must still
        # be recognized as a replay and be a no-op, not applied as a change.
        replay = _write_skill_row(
            eng,
            skill_id="skill:s",
            name="s",
            description="v1",
            revision=2,
            idempotency_key="attempt-1",
        )
    assert first is True
    assert replay is False
    row = _one_row("skills", "skill:s", eng)
    assert row["revision"] == 1  # untouched by the no-op replay
    # Only ONE INSERT/UPDATE was ever issued for this id (the replay issued
    # no write statement at all, only the CAS read).
    write_stmts = [
        s
        for s in eng.graph_compute.statements
        if s.startswith("INSERT INTO skills") or s.startswith("UPDATE skills")
    ]
    assert len(write_stmts) == 1


def test_changed_content_after_existing_row_is_applied_via_update():
    eng = _FakeEngine()
    with use_actor(_session("t").actor), use_session(_session("t")):
        _write_skill_row(
            eng,
            skill_id="skill:s",
            name="s",
            description="v1",
            revision=1,
            idempotency_key="attempt-1",
        )
        ok = _write_skill_row(
            eng,
            skill_id="skill:s",
            name="s",
            description="v2 — genuinely changed",
            revision=2,
            idempotency_key="attempt-2",
        )
    assert ok is True
    row = _one_row("skills", "skill:s", eng)
    assert row["description"] == "v2 — genuinely changed"
    assert row["revision"] == 2


# ---------------------------------------------------------------------------
# Desired (mcp_servers) vs. observed (mcp_server_discovery) split
# ---------------------------------------------------------------------------


def test_discovery_observation_does_not_change_desired_enabled_state():
    eng = _FakeEngine()
    configs = {"srv": {"url": "http://x", "disabled": False}}
    with use_actor(_session("t").actor), use_session(_session("t")):
        _write_fleet_catalog(eng, _server_catalog(), configs=configs)
        # A later probe observes the server as UNREACHABLE — this is pure
        # discovery information and must never flip the desired `enabled`
        # row, which only the (unrelated) config-derived desired state owns.
        _write_fleet_catalog(
            eng, _server_catalog(error="connection refused"), configs=configs
        )

    server_row = eng.graph_compute.tables["mcp_servers"]["mcp_server_srv"]
    assert server_row["enabled"] is True  # untouched by the unreachable probe
    assert "reachable" not in server_row  # desired row never carries observed fields

    discovery_rows = list(eng.graph_compute.tables["mcp_server_discovery"].values())
    reachable_states = {row["server_id"]: row["reachable"] for row in discovery_rows}
    # Both a reachable=True and a reachable=False observation were recorded
    # as SEPARATE rows (append-only), not one row mutated in place.
    assert any(r["reachable"] is True for r in discovery_rows)
    assert any(r["reachable"] is False for r in discovery_rows)
    assert len(discovery_rows) == 2
    assert reachable_states  # sanity: at least one server observed


def test_unreachable_server_still_gets_an_honest_discovery_row():
    eng = _FakeEngine()
    with use_actor(_session("t").actor), use_session(_session("t")):
        result = _write_fleet_catalog(
            eng, _server_catalog(error="econnrefused: no route to host")
        )
    assert result["servers_unreachable"] == 1
    assert result["discovery_written"] == 1
    discovery_rows = list(eng.graph_compute.tables["mcp_server_discovery"].values())
    assert len(discovery_rows) == 1
    row = discovery_rows[0]
    assert row["reachable"] is False
    assert "econnrefused" in row["last_error"]
    # The server row itself still exists too — never omitted just because
    # the probe failed.
    assert "mcp_server_srv" in eng.graph_compute.tables["mcp_servers"]


def test_discovery_binds_its_discovery_principal():
    eng = _FakeEngine()
    with (
        use_actor(_session("t", actor_id="probe-runner-9").actor),
        use_session(_session("t", actor_id="probe-runner-9")),
    ):
        _write_fleet_catalog(eng, _server_catalog())
    row = next(iter(eng.graph_compute.tables["mcp_server_discovery"].values()))
    assert row["discovery_principal"] == "probe-runner-9"
    assert row["discovery_grant_digest"]


def test_discovery_grant_is_derived_and_caller_payload_cannot_spoof_subject():
    eng = _FakeEngine()
    catalog = _server_catalog()
    catalog["srv"]["discovery_principal"] = "spoofed-subject"
    catalog["srv"]["discovery_grant_digest"] = "spoofed-grant"
    session = _session("t", actor_id="verified-probe")
    with use_actor(session.actor), use_session(session):
        expected_binding = _test_discovery_binding()
        assert expected_binding is not None
        expected_principal = expected_binding.principal_id
        expected_grant = expected_binding.fingerprint
        _write_fleet_catalog(eng, catalog)

    discovery = next(iter(eng.graph_compute.tables["mcp_server_discovery"].values()))
    assert discovery["discovery_principal"] == expected_principal
    assert discovery["discovery_grant_digest"] == expected_grant
    assert discovery["discovery_principal"] != "spoofed-subject"
    assert discovery["discovery_grant_digest"] != "spoofed-grant"


def test_two_principals_get_disjoint_immutable_discovery_snapshots():
    eng = _FakeEngine()
    first = _session("t", actor_id="probe-a")
    second = _session("t", actor_id="probe-b")
    with use_actor(first.actor), use_session(first):
        _write_fleet_catalog(eng, _server_catalog())
    with use_actor(second.actor), use_session(second):
        _write_fleet_catalog(eng, _server_catalog())

    tools = _rows_with_prefix("mcp_tools", "tool_srv_t1", eng)
    assert {row["discovery_principal"] for row in tools} == {
        "probe-a",
        "probe-b",
    }
    assert len({row["discovery_grant_digest"] for row in tools}) == 2
    assert len({row["id"] for row in tools}) == 2


def test_unbound_discovery_skips_derived_rows_but_keeps_desired_registration():
    eng = _FakeEngine()
    with suspend_session():
        result = _write_fleet_catalog(eng, _server_catalog())

    assert result["servers_written"] == 1
    assert result["discovery_written"] == 0
    assert result["tools_written"] == 0
    assert result["discovery_status"] == "unavailable"
    assert not eng.graph_compute.tables["mcp_server_discovery"]
    assert not eng.graph_compute.tables["mcp_tools"]


def test_non_oauth_local_binding_populates_derived_rows_without_grant_digest():
    eng = _FakeEngine()
    session = _session("tenant-local")
    with use_actor(session.actor), use_session(session):
        result = _write_local_fleet_catalog(eng, _server_catalog())

    assert result["discovery_status"] == "bound"
    assert result["discovery_written"] == 1
    discovery = next(iter(eng.graph_compute.tables["mcp_server_discovery"].values()))
    tool = next(iter(eng.graph_compute.tables["mcp_tools"].values()))
    assert discovery["discovery_authority_kind"] == fct.DISCOVERY_AUTHORITY_TENANT_LOCAL
    assert tool["discovery_authority_kind"] == fct.DISCOVERY_AUTHORITY_TENANT_LOCAL
    assert discovery["discovery_principal"] == ""
    assert discovery["discovery_grant_digest"] == ""
    assert tool["discovery_grant_digest"] == ""


def test_untyped_local_authority_payload_cannot_populate_derived_rows():
    eng = _FakeEngine()
    session = _session("tenant-local")
    with use_actor(session.actor), use_session(session):
        result = fct.write_fleet_catalog(
            eng,
            _server_catalog(),
            discovery_bindings={"srv": {"tenant_id": "tenant-local"}},
        )

    assert result["discovery_status"] == "unavailable"
    assert not eng.graph_compute.tables["mcp_server_discovery"]
    assert not eng.graph_compute.tables["mcp_tools"]


def test_binding_from_wrong_tenant_cannot_authorize_derived_rows():
    eng = _FakeEngine()
    wrong_tenant = OAuthGrantBinding(
        tenant_id="tenant-other",
        principal_id="probe-actor",
        provider_id="test-provider",
        resource_url="https://test-provider.example/mcp",
        audience="https://test-provider.example/mcp",
        granted_scopes=("mcp:read",),
        key_version=1,
        grant_revision="wrong-tenant",
    )
    session = _session("tenant-a")
    with use_actor(session.actor), use_session(session):
        result = fct.write_fleet_catalog(
            eng,
            _server_catalog(),
            discovery_bindings={"srv": wrong_tenant},
        )
    assert result["discovery_status"] == "unavailable"
    assert not eng.graph_compute.tables["mcp_tools"]


# ---------------------------------------------------------------------------
# schema_digest: changed input_schema -> different digest, unchanged -> same
# ---------------------------------------------------------------------------


def test_changed_input_schema_produces_a_different_digest():
    a = fct._schema_digest({"properties": {"x": {"type": "string"}}})
    b = fct._schema_digest({"properties": {"x": {"type": "integer"}}})
    assert a != b


def test_unchanged_input_schema_produces_the_same_digest():
    schema = {"properties": {"x": {"type": "string"}}, "required": ["x"]}
    a = fct._schema_digest(schema)
    b = fct._schema_digest(dict(schema))  # structurally identical, different object
    assert a == b


def test_tool_row_schema_digest_reflects_a_changed_contract():
    eng = _FakeEngine()
    catalog_v1 = _server_catalog()
    catalog_v2 = {
        "srv": {
            "error": None,
            "tools": [
                {
                    "name": "t1",
                    "description": "d",
                    "inputSchema": {"properties": {"x": {"type": "integer"}}},
                }
            ],
            "skills": [],
            "prompts": [],
        }
    }
    with use_actor(_session("t").actor), use_session(_session("t")):
        _write_fleet_catalog(eng, catalog_v1, revision=1, idempotency_key="a1")
        _write_fleet_catalog(eng, catalog_v2, revision=2, idempotency_key="a2")
    row = _one_row("mcp_tools", "tool_srv_t1", eng)
    assert row["schema_digest"] == fct._schema_digest(
        {"properties": {"x": {"type": "integer"}}}
    )


# ---------------------------------------------------------------------------
# Batching: one statement per table, not one per row
# ---------------------------------------------------------------------------


def test_batched_write_issues_one_insert_per_table_not_one_per_row():
    eng = _FakeEngine()
    catalog = {
        f"srv{i}": {
            "error": None,
            "tools": [
                {"name": f"tool{j}", "description": "d", "inputSchema": {}}
                for j in range(5)
            ],
            "skills": [],
            "prompts": [],
        }
        for i in range(4)
    }
    with use_actor(_session("t").actor), use_session(_session("t")):
        result = _write_fleet_catalog(eng, catalog)

    assert result["tools_written"] == 20  # 4 servers * 5 tools
    assert result["servers_written"] == 4

    insert_stmts = [
        s for s in eng.graph_compute.statements if s.startswith("INSERT INTO mcp_tools")
    ]
    # 20 new tool rows, written via new ids only (no pre-existing rows) —
    # exactly ONE batched INSERT statement, not 20.
    assert len(insert_stmts) == 1

    server_inserts = [
        s
        for s in eng.graph_compute.statements
        if s.startswith("INSERT INTO mcp_servers")
    ]
    assert len(server_inserts) == 1

    discovery_inserts = [
        s
        for s in eng.graph_compute.statements
        if s.startswith("INSERT INTO mcp_server_discovery")
    ]
    assert len(discovery_inserts) == 1


def test_ensure_fleet_catalog_tables_ddl_issued_once_per_store():
    eng = _FakeEngine()
    with use_actor(_session("t").actor), use_session(_session("t")):
        _write_fleet_catalog(eng, _server_catalog())
        _write_fleet_catalog(eng, _server_catalog())
    create_stmts = [
        s for s in eng.graph_compute.statements if s.startswith("CREATE TABLE")
    ]
    # once, not once per write call: the 6 catalog tables + the migration
    # ledger table itself.
    assert len(create_stmts) == len(fct._DDL) + 1
    migration_stmts = [
        s for s in eng.graph_compute.statements if s.startswith("ALTER TABLE")
    ]
    # A genuinely FRESH store's CREATE TABLE already carries every current
    # column (including the discovery-binding ones) — no ALTER is needed at
    # all. (Already-current store -> no-op, no DDL issued.)
    assert migration_stmts == []
    assert all("ADD COLUMN IF NOT EXISTS" not in s for s in migration_stmts)


def test_already_current_store_second_process_start_is_a_pure_noop():
    """A second, independent introspection of an already-migrated store
    (simulating a fresh process restart, bypassing the in-process
    ``_ensured_stores`` cache) issues no CREATE/ALTER/backfill at all —
    only the cheap ledger + information_schema reads."""
    eng = _FakeEngine()
    with use_actor(_session("t").actor), use_session(_session("t")):
        assert fct.ensure_fleet_catalog_tables(eng) is True
    before = len(eng.graph_compute.statements)

    ok = fct._claim_and_migrate(eng)  # bypasses the process-level cache
    assert ok is True
    after_stmts = eng.graph_compute.statements[before:]
    assert not any(
        s.startswith("CREATE TABLE") and "IF NOT EXISTS" not in s for s in after_stmts
    )
    ddl_stmts = [
        s
        for s in after_stmts
        if s.startswith("ALTER TABLE")
        or s.startswith("INSERT INTO mcp_")
        or s.startswith("UPDATE mcp_")
        or s.startswith("INSERT INTO skills")
        or s.startswith("UPDATE skills")
    ]
    assert ddl_stmts == []


# ---------------------------------------------------------------------------
# NE-052 / AU-CATALOG: fail-closed schema states + concurrency
# ---------------------------------------------------------------------------


def test_unknown_newer_ledger_migration_id_refuses_to_serve():
    eng = _FakeEngine()
    gc = eng.graph_compute
    for ddl in fct._DDL.values():
        gc.sql_exec(ddl)
    gc.sql_exec(fct._LEDGER_DDL)
    # Simulate a store migrated forward by a NEWER revision of this module —
    # its ledger names a migration_id this code has never heard of.
    gc.tables[fct._MIGRATION_LEDGER][fct._LOCK_ROW_ID] = {
        "id": fct._LOCK_ROW_ID,
        "status": "complete",
        "claimant": "",
        "claimed_at": "2030-01-01T00:00:00+00:00",
        "version": 99,
        "migration_id": "0099_from_the_future",
        "checksum": "deadbeef",
        "applied_at": "2030-01-01T00:00:00+00:00",
    }
    with pytest.raises(fct.FleetCatalogSchemaTooNewError):
        fct._claim_and_migrate(eng)


def test_diverged_hand_modified_schema_refuses_to_serve():
    eng = _FakeEngine()
    gc = eng.graph_compute
    for ddl in fct._DDL.values():
        gc.sql_exec(ddl)
    # Hand-inject a column that belongs to no known fleet-catalog schema
    # generation at all.
    gc.columns[fct.TABLE_MCP_SERVERS].add("mystery_column")
    with pytest.raises(fct.FleetCatalogSchemaDivergedError):
        fct._claim_and_migrate(eng)


def test_concurrent_migration_claim_is_exactly_once_via_ledger_cas():
    """Two independent calls against the SAME legacy-shaped store (modeling
    two processes racing to migrate it) apply the DDL/backfill exactly
    once — the loser performs no DDL and returns a benign no-op, never an
    error, never a duplicate/partial migration."""
    eng = _FakeEngine()
    _seed_legacy_store(eng)

    first = fct._claim_and_migrate(eng)
    assert first is True
    alter_after_first = [
        s for s in eng.graph_compute.statements if s.startswith("ALTER TABLE")
    ]
    assert alter_after_first  # the winner actually migrated something

    before_second = len(eng.graph_compute.statements)
    second = fct._claim_and_migrate(eng)
    assert second is True
    after_second = eng.graph_compute.statements[before_second:]
    assert not any(s.startswith("ALTER TABLE") for s in after_second)
    assert not any(s.startswith("UPDATE mcp_servers") for s in after_second)

    ledger_row = eng.graph_compute.tables[fct._MIGRATION_LEDGER][fct._LOCK_ROW_ID]
    assert ledger_row["status"] == "complete"


def test_losing_claim_while_another_process_still_migrating_is_a_noop_not_an_error():
    eng = _FakeEngine()
    _seed_legacy_store(eng)
    gc = eng.graph_compute
    gc.sql_exec(fct._LEDGER_DDL)
    # Another process already claimed the migration and has not finished.
    # The claim is a LEASE (`_MIGRATION_CLAIM_LEASE_SEC`), so a LIVE claim
    # must be stamped now — a fixed past date would model an ABANDONED one.
    gc.tables[fct._MIGRATION_LEDGER][fct._LOCK_ROW_ID] = {
        "id": fct._LOCK_ROW_ID,
        "status": "migrating",
        "claimant": "other-process-token",
        "claimed_at": datetime.now(UTC).isoformat(),
        "version": 0,
        "migration_id": "",
        "checksum": "",
        "applied_at": "",
    }
    result = fct._claim_and_migrate(eng)
    assert result is False  # lost the race -- benign no-op, not an exception
    assert not any(s.startswith("ALTER TABLE") for s in eng.graph_compute.statements)
    # The other process's in-progress claim was not clobbered.
    assert (
        gc.tables[fct._MIGRATION_LEDGER][fct._LOCK_ROW_ID]["claimant"]
        == "other-process-token"
    )


# ---------------------------------------------------------------------------
# No secret values, honest states
# ---------------------------------------------------------------------------


def test_no_command_or_args_columns_anywhere():
    for ddl in fct._DDL.values():
        assert "command" not in ddl.lower()
        assert " args " not in ddl.lower()


def test_no_engine_sql_surface_degrades_gracefully():
    class _NoSqlEngine:
        pass

    result = _write_fleet_catalog(_NoSqlEngine(), _server_catalog())
    assert result["status"] == "skipped"
    ok = _write_skill_row(_NoSqlEngine(), skill_id="skill:s", name="s")
    assert ok is False


# ---------------------------------------------------------------------------
# NE-052 / AU-CATALOG: migration path for a store deployed BEFORE the NE-007
# hardening (no tenant_id/revision/idempotency_key/schema_digest anywhere,
# and no ``mcp_server_discovery`` table at all) — reproduces the reported
# defect (``ensure_fleet_catalog_tables`` only ever issued ``CREATE TABLE IF
# NOT EXISTS``, a silent no-op against an already-existing old table) and
# proves the fix. The exact pre-NE-007 DDL below is copied verbatim from
# commit 1f96b7bce (the last revision before the NE-007 hardening landed).
# ---------------------------------------------------------------------------

_LEGACY_DDL: dict[str, str] = {
    "mcp_servers": """CREATE TABLE IF NOT EXISTS mcp_servers (
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
    "mcp_tools": """CREATE TABLE IF NOT EXISTS mcp_tools (
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
    "mcp_prompts": """CREATE TABLE IF NOT EXISTS mcp_prompts (
    id TEXT PRIMARY KEY,
    server_id TEXT NOT NULL,
    server_name TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    uri TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
    "mcp_resources": """CREATE TABLE IF NOT EXISTS mcp_resources (
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
    "skills": """CREATE TABLE IF NOT EXISTS skills (
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


def _seed_legacy_store(eng: _FakeEngine) -> None:
    """Stand up a pre-NE-007 store: old DDL + real pre-existing rows.

    Rows are written directly into the fake's table dict (not through
    ``sql_exec``) because the legacy DDL has no ``tenant_id`` column for a
    real ``INSERT`` to name — exactly the shape a genuinely old deployment
    would have on disk.
    """
    gc = eng.graph_compute
    for ddl in _LEGACY_DDL.values():
        gc.sql_exec(ddl)
    gc.tables["mcp_servers"]["mcp_server_legacy"] = {
        "id": "mcp_server_legacy",
        "name": "legacy",
        "transport": "stdio",
        "url": "",
        "enabled": True,
        "reachable": True,
        "last_probe_at": "2024-01-01T00:00:00+00:00",
        "last_error": "",
        "tool_count": 1,
        "skill_count": 0,
        "prompt_count": 0,
        "resource_count": 0,
        "updated_at": "2024-01-01T00:00:00+00:00",
    }
    gc.tables["mcp_tools"]["tool_legacy_t1"] = {
        "id": "tool_legacy_t1",
        "server_id": "mcp_server_legacy",
        "server_name": "legacy",
        "name": "t1",
        "description": "an old tool",
        "input_schema": '{"properties": {"x": {"type": "string"}}}',
        "tool_mode": "verbose",
        "enabled": True,
        "updated_at": "2024-01-01T00:00:00+00:00",
    }
    gc.tables["skills"]["skill_legacy_s1"] = {
        "id": "skill_legacy_s1",
        "name": "s1",
        "description": "an old skill",
        "uri": "skill://legacy/s1",
        "skill_type": "mcp_skill",
        "classification": "MCP Skill",
        "provider": "mcp:legacy",
        "mcp_server": "legacy",
        "enabled": True,
        "updated_at": "2024-01-01T00:00:00+00:00",
    }


def test_old_schema_store_migrates_preserves_rows_and_write_then_succeeds():
    eng = _FakeEngine()
    _seed_legacy_store(eng)

    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        ok = fct.ensure_fleet_catalog_tables(eng)
        assert ok is True

        # Pre-existing rows survived (not dropped/recreated) and were
        # backfilled with the reserved legacy tenant sentinel — never a
        # guessed real tenant.
        server_row = eng.graph_compute.tables["mcp_servers"]["mcp_server_legacy"]
        assert server_row["name"] == "legacy"  # original data intact
        assert server_row["tenant_id"] == fct.LEGACY_TENANT_SENTINEL
        assert server_row["tenant_id"] != "tenant-a"  # never a guessed real tenant
        assert isinstance(server_row["revision"], int)
        assert server_row["idempotency_key"]

        tool_row = eng.graph_compute.tables["mcp_tools"]["tool_legacy_t1"]
        assert tool_row["description"] == "an old tool"
        assert tool_row["tenant_id"] == fct.LEGACY_TENANT_SENTINEL
        # schema_digest was backfilled from the row's own stored input_schema.
        assert tool_row["schema_digest"] == fct._schema_digest(
            {"properties": {"x": {"type": "string"}}}
        )

        skill_row = eng.graph_compute.tables["skills"]["skill_legacy_s1"]
        assert skill_row["tenant_id"] == fct.LEGACY_TENANT_SENTINEL

        # The regression: a write against this now-migrated store must
        # succeed (previously failed at INSERT time referencing columns
        # that did not exist on the old table).
        result = _write_fleet_catalog(eng, _server_catalog())
        assert result["status"] == "ok"
        assert result["servers_written"] >= 1

        ok2 = _write_skill_row(
            eng, skill_id="skill:new", name="new", description="fresh"
        )
        assert ok2 is True
    new_skill = _one_row("skills", "skill:new", eng)
    assert new_skill["tenant_id"] == "tenant-a"


# ---------------------------------------------------------------------------
# Skill classification override (CONCEPT:AU-KG.ingest.skill-classification-writeback)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_override_ddl_cache():
    fct._ensured_override_stores.clear()
    yield
    fct._ensured_override_stores.clear()


def test_classification_override_round_trips():
    eng = _FakeEngine()
    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        ok = fct.write_skill_classification_override(
            eng, skill_id="skill:foo", skill_type="workflow", principal="operator-1"
        )
        assert ok is True
        value = fct.read_skill_classification_override(
            eng, skill_id="skill:foo", tenant_id="tenant-a"
        )
    assert value == "workflow"


def test_classification_override_absent_returns_none():
    eng = _FakeEngine()
    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        value = fct.read_skill_classification_override(
            eng, skill_id="skill:never-set", tenant_id="tenant-a"
        )
    assert value is None


def test_classification_override_is_tenant_scoped():
    eng = _FakeEngine()
    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        fct.write_skill_classification_override(
            eng, skill_id="skill:foo", skill_type="workflow", principal="operator-1"
        )
    with use_actor(_session("tenant-b").actor), use_session(_session("tenant-b")):
        value = fct.read_skill_classification_override(
            eng, skill_id="skill:foo", tenant_id="tenant-b"
        )
    assert value is None  # tenant-b never set an override for this id


def test_write_skill_row_honors_an_existing_override_over_the_caller_value():
    """The mechanism that makes a classification survive a re-sync: once an
    override is on record, EVERY future write_skill_row call for that skill
    resolves to the override, no matter what skill_type the caller (a
    simulated fleet-tool-schema-sync re-derive from frontmatter) passes.
    """
    eng = _FakeEngine()
    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        _write_skill_row(
            eng, skill_id="skill:foo", name="foo", description="d", skill_type="mystery"
        )
        row = _one_row("skills", "skill:foo", eng)
        assert row["skill_type"] == "mystery"

        fct.write_skill_classification_override(
            eng, skill_id="skill:foo", skill_type="workflow", principal="operator-1"
        )

        # Simulates the hourly sync re-deriving from the (unchanged, still
        # "mystery") on-disk frontmatter and writing it straight through --
        # the override must win. No explicit revision: it must default to a
        # fresh wall-clock value higher than the first write's, so this is a
        # genuine "changed content" update, not a stale-revision rejection.
        _write_skill_row(
            eng,
            skill_id="skill:foo",
            name="foo",
            description="d",
            skill_type="mystery",
            idempotency_key="resync-1",
        )
    row = _one_row("skills", "skill:foo", eng)
    assert row["skill_type"] == "workflow"
    assert row["classification"] == "Workflow"


def test_get_skill_row_returns_none_for_unknown_id():
    eng = _FakeEngine()
    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        row = fct.get_skill_row(eng, skill_id="skill:does-not-exist")
    assert row is None


def test_get_skill_row_returns_the_current_row():
    eng = _FakeEngine()
    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        _write_skill_row(eng, skill_id="skill:foo", name="foo", description="d")
        bound_id = _one_row("skills", "skill:foo", eng)["id"]
        row = fct.get_skill_row(eng, skill_id=bound_id)
    assert row is not None
    assert row["name"] == "foo"


# ---------------------------------------------------------------------------
# NE-0XX / AU-CATALOG-ACL: the ACL-projection migration step (0004) and the
# writer-side ACL stamp it backs -- CONCEPT:AU-KG.ingest.fleet-catalog-acl-projection.
# ---------------------------------------------------------------------------

# The exact pre-ACL-projection shape of the 3 affected tables, AS AN
# ALTER-TABLE-MIGRATED STORE WOULD ACTUALLY CARRY IT -- i.e. this module's
# own ``_DDL`` as it stood immediately before this change (steps 0001-0003
# applied, step 0004 not yet invented). This is the REALISTIC production
# case the task description calls out: ``fleet-tool-schema-sync`` already
# ran under the prior code version, so these tables are genuinely deployed
# with real rows, just missing the new ACL columns.
#
# ``mcp_servers`` is deliberately NOT the clean 9-column shape a genuinely
# FRESH ``CREATE TABLE`` produces under the pre-ACL ``_DDL`` text: this
# module's migration mechanism is additive-only (``ALTER TABLE ADD COLUMN``
# never drops a column), so a store that reached "current" via the real
# ledgered migration path (:data:`fct._MIGRATION_COLUMN_STEPS`, step
# ``0001_tenant_revision_idempotency``, the only step that ever touches
# ``mcp_servers``) still carries its pre-NE-007 observed-discovery columns
# (``reachable``/``last_probe_at``/... -- see :data:`fct._LEGACY_SCHEMA_COLUMNS`)
# forever, since those moved to the new ``mcp_server_discovery`` table
# without ever being physically dropped from an already-migrated
# ``mcp_servers``. That 16-column shape (legacy 13 + step 1's 3) is what
# :func:`fct._is_reachable_state` actually recognizes as "already migrated,
# pre-ACL" for this table.
_PRE_ACL_DDL: dict[str, str] = {
    fct.TABLE_MCP_SERVERS: """CREATE TABLE IF NOT EXISTS mcp_servers (
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
    updated_at TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    revision BIGINT NOT NULL,
    idempotency_key TEXT NOT NULL
)""",
    fct.TABLE_MCP_TOOLS: """CREATE TABLE IF NOT EXISTS mcp_tools (
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
    fct.TABLE_SKILLS: """CREATE TABLE IF NOT EXISTS skills (
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


def _seed_step3_store(eng: _FakeEngine) -> None:
    """Stand up a store already fully migrated through step 0003 -- exactly
    the shape this module shipped in before the ACL-projection change, with
    real pre-existing rows (a desired server, a discovery-bound tool, a
    discovery-bound skill)."""
    gc = eng.graph_compute
    for ddl in _PRE_ACL_DDL.values():
        gc.sql_exec(ddl)
    gc.tables[fct.TABLE_MCP_SERVERS]["mcp_server_srv"] = {
        "id": "mcp_server_srv",
        "tenant_id": "tenant-a",
        "name": "srv",
        "transport": "stdio",
        "url": "",
        "enabled": True,
        "reachable": True,
        "last_probe_at": "2026-01-01T00:00:00+00:00",
        "last_error": "",
        "tool_count": 1,
        "skill_count": 0,
        "prompt_count": 0,
        "resource_count": 0,
        "revision": 1,
        "idempotency_key": "seed-server",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    gc.tables[fct.TABLE_MCP_TOOLS]["tool_srv_t1__grant-digest-1"] = {
        "id": "tool_srv_t1__grant-digest-1",
        "tenant_id": "tenant-a",
        "server_id": "mcp_server_srv",
        "server_name": "srv",
        "name": "t1",
        "description": "an existing tool",
        "input_schema": '{"properties": {}}',
        "schema_digest": fct._schema_digest({"properties": {}}),
        "tool_mode": "verbose",
        "enabled": True,
        "discovery_authority_kind": fct.DISCOVERY_AUTHORITY_OAUTH_GRANT,
        "discovery_principal": "principal:probe",
        "discovery_grant_digest": "grant-digest-1",
        "revision": 1,
        "idempotency_key": "seed-tool",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    gc.tables[fct.TABLE_SKILLS]["skill_srv_s1__tenant_local"] = {
        "id": "skill_srv_s1__tenant_local",
        "tenant_id": "tenant-a",
        "name": "s1",
        "description": "an existing skill",
        "uri": "skill://srv/s1",
        "skill_type": "mcp_skill",
        "classification": "MCP Skill",
        "provider": "mcp:srv",
        "mcp_server": "srv",
        "enabled": True,
        "discovery_authority_kind": fct.DISCOVERY_AUTHORITY_TENANT_LOCAL,
        "discovery_principal": "",
        "discovery_grant_digest": "",
        "revision": 1,
        "idempotency_key": "seed-skill",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }


def test_acl_projection_migration_adds_columns_to_an_already_deployed_step3_store():
    """The case that actually matters in production: these tables are
    ALREADY DEPLOYED at the pre-ACL (0001-0003) shape from
    ``fleet-tool-schema-sync`` runs before this change -- ``CREATE TABLE IF
    NOT EXISTS`` is a silent no-op against them (module docstring), so only
    the ledgered ``ALTER TABLE`` path can bring them forward. Proves step
    0004 does exactly that, without touching -- let alone dropping -- the
    pre-existing rows or their other columns."""
    eng = _FakeEngine()
    _seed_step3_store(eng)
    gc = eng.graph_compute

    assert "acl_classification" not in gc.columns[fct.TABLE_MCP_SERVERS]
    assert "kg_node_id" not in gc.columns[fct.TABLE_MCP_TOOLS]

    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        ok = fct.ensure_fleet_catalog_tables(eng)
    assert ok is True

    for table in (fct.TABLE_MCP_SERVERS, fct.TABLE_MCP_TOOLS, fct.TABLE_SKILLS):
        assert {"acl_classification", "acl_owner_id", "acl_shared_scope"} <= gc.columns[
            table
        ]
    assert "kg_node_id" in gc.columns[fct.TABLE_MCP_TOOLS]
    assert "kg_node_id" in gc.columns[fct.TABLE_SKILLS]
    # mcp_servers never needed kg_node_id -- its own `id` already IS the KG
    # node id (a server row is never suffixed with a discovery-grant digest).
    assert "kg_node_id" not in gc.columns[fct.TABLE_MCP_SERVERS]

    # Pre-existing data survived untouched.
    server_row = gc.tables[fct.TABLE_MCP_SERVERS]["mcp_server_srv"]
    assert server_row["name"] == "srv"
    assert server_row["revision"] == 1

    # kg_node_id was deterministically reconstructed by stripping the exact
    # recorded discovery_grant_digest suffix off the row's own `id`.
    tool_row = gc.tables[fct.TABLE_MCP_TOOLS]["tool_srv_t1__grant-digest-1"]
    assert tool_row["kg_node_id"] == "tool_srv_t1"
    skill_row = gc.tables[fct.TABLE_SKILLS]["skill_srv_s1__tenant_local"]
    assert skill_row["kg_node_id"] == "skill_srv_s1"

    # ACL columns are deliberately left NULL for a pre-existing row -- there
    # is no verified actor context to recover retroactively. NULL is what
    # tells a reader "SQL has no opinion", never "unrestricted".
    assert not tool_row.get("acl_classification")
    assert not tool_row.get("acl_owner_id")
    assert not skill_row.get("acl_classification")
    assert not server_row.get("acl_classification")

    ledger_row = gc.tables[fct._MIGRATION_LEDGER]["step__0004_acl_projection_columns"]
    assert ledger_row["status"] == "applied"
    assert ledger_row["checksum"] == fct._step_checksum(
        "0004_acl_projection_columns", dict(fct._ACL_PROJECTION_MIGRATION)
    )

    # A write against this now-migrated store succeeds (the regression this
    # closes: an already-deployed store must not fail at INSERT time
    # referencing columns it lacked before this step ran).
    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        result = _write_fleet_catalog(eng, _server_catalog())
    assert result["status"] == "ok"


# The OTHER way a store legitimately arrives at the pre-ACL shape, and the
# one the comment above ``_PRE_ACL_DDL`` wrongly assumed could not happen in
# production: a store CREATED FRESH by the pre-ACL code version. Its
# ``CREATE TABLE`` was the then-current DDL text, so ``mcp_servers`` is the
# clean 9-column shape -- it never had the retired pre-NE-007
# observed-discovery columns at all, and so matches neither
# ``_LEGACY_SCHEMA_COLUMNS`` nor any forward accumulation from it.
_FRESH_PRE_ACL_DDL: dict[str, str] = {
    fct.TABLE_MCP_SERVERS: """CREATE TABLE IF NOT EXISTS mcp_servers (
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
    fct.TABLE_MCP_TOOLS: _PRE_ACL_DDL[fct.TABLE_MCP_TOOLS],
    fct.TABLE_SKILLS: _PRE_ACL_DDL[fct.TABLE_SKILLS],
}


def test_fresh_created_pre_acl_store_is_migrated_not_reported_diverged():
    """ROOT-CAUSE REGRESSION (measured live 2026-08-25 on graph-os).

    ``_is_reachable_state`` modelled only "migrated up from the pre-NE-007
    legacy shape", so a store whose ``mcp_servers`` was CREATED FRESH at
    step 0003 -- today's shape minus step 0004's ACL columns, and without
    the retired legacy observed-discovery columns -- matched nothing and was
    declared diverged. ``ensure_fleet_catalog_tables`` then raised,
    ``source_sync._write_fleet_relational`` caught it, and the ENTIRE
    relational catalog write was silently skipped on every sync while the KG
    node write succeeded -- the dashboard kept reading a months-stale
    catalog. The fix must recognize this shape and migrate it forward.
    """
    eng = _FakeEngine()
    gc = eng.graph_compute
    for ddl in _FRESH_PRE_ACL_DDL.values():
        gc.sql_exec(ddl)
    for ddl in (
        fct._DDL[fct.TABLE_MCP_SERVER_DISCOVERY],
        fct._DDL[fct.TABLE_MCP_PROMPTS],
        fct._DDL[fct.TABLE_MCP_RESOURCES],
    ):
        gc.sql_exec(ddl)

    assert gc.columns[fct.TABLE_MCP_SERVERS] == {
        "id",
        "tenant_id",
        "name",
        "transport",
        "url",
        "enabled",
        "revision",
        "idempotency_key",
        "updated_at",
    }

    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        ok = fct.ensure_fleet_catalog_tables(eng)
    assert ok is True
    assert {"acl_classification", "acl_owner_id", "acl_shared_scope"} <= gc.columns[
        fct.TABLE_MCP_SERVERS
    ]

    # And the write that was being skipped now actually lands.
    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        result = _write_fleet_catalog(eng, _server_catalog())
    assert result["status"] == "ok"
    assert result["tools_written"] == 1


def test_an_abandoned_migration_claim_is_taken_over_not_waited_on_forever():
    """A migrator that dies mid-step leaves its `migrating` ledger row
    behind. Measured live 2026-08-25: the graph-os container was OOM-killed
    while applying step 0004, leaving a half-applied schema AND a claim no
    process held. Treating that as live wedges the store permanently and
    every fleet-catalog write stays skipped, so the claim is a lease."""
    eng = _FakeEngine()
    _seed_legacy_store(eng)
    gc = eng.graph_compute
    gc.sql_exec(fct._LEDGER_DDL)
    stale = datetime.now(UTC) - timedelta(seconds=fct._MIGRATION_CLAIM_LEASE_SEC + 60)
    gc.tables[fct._MIGRATION_LEDGER][fct._LOCK_ROW_ID] = {
        "id": fct._LOCK_ROW_ID,
        "status": "migrating",
        "claimant": "token-of-a-process-that-died",
        "claimed_at": stale.isoformat(),
        "version": 0,
        "migration_id": "",
        "checksum": "",
        "applied_at": "",
    }
    assert fct._claim_and_migrate(eng) is True
    assert any(s.startswith("ALTER TABLE") for s in gc.statements)
    assert gc.tables[fct._MIGRATION_LEDGER][fct._LOCK_ROW_ID]["status"] == "complete"


def test_an_unreadable_claim_timestamp_is_treated_as_abandoned():
    """Never wedge on a value that cannot be interpreted."""
    assert fct._claim_is_live({"claimed_at": ""}) is False
    assert fct._claim_is_live({"claimed_at": "not-a-timestamp"}) is False
    assert fct._claim_is_live({"claimed_at": datetime.now(UTC).isoformat()}) is True


def test_large_batches_are_chunked_not_one_giant_statement_nor_one_per_row():
    """Batched writing must stay batched, but bounded. The live fleet probes
    ~9,600 tools; rendering all of them into ONE `INSERT ... VALUES` built a
    multi-megabyte statement that OOM-killed the graph-os container
    (measured 2026-08-25). Chunking keeps it a handful of statements, never
    one per row."""
    eng = _FakeEngine()
    n = fct._MAX_ROWS_PER_STATEMENT * 2 + 7
    catalog = {
        "srv": {
            "error": None,
            "tools": [
                {"name": f"t{i}", "description": "d", "inputSchema": {}}
                for i in range(n)
            ],
            "skills": [],
            "prompts": [],
        }
    }
    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        result = _write_fleet_catalog(eng, catalog)
    assert result["tools_written"] == n

    tool_inserts = [
        s
        for s in eng.graph_compute.statements
        if s.startswith(f"INSERT INTO {fct.TABLE_MCP_TOOLS} ")
    ]
    tool_selects = [
        s
        for s in eng.graph_compute.statements
        if s.startswith(f"SELECT * FROM {fct.TABLE_MCP_TOOLS} ")
    ]
    assert len(tool_inserts) == 3
    assert len(tool_selects) == 3
    for statement in tool_inserts:
        assert statement.count("), (") + 1 <= fct._MAX_ROWS_PER_STATEMENT


def test_a_completed_ledger_row_does_not_veto_the_next_migration_step():
    """SECOND ROOT CAUSE (measured live 2026-08-25 on platform/graph-os).

    Every real store that has ever finished a migration carries a
    ``schema_state`` ledger row marked ``complete``. Claiming the lock for
    the NEXT step used ``INSERT ... ON CONFLICT (id) DO NOTHING`` against
    that row — and the engine's SQL tier ignores the clause, raising a bare
    duplicate-key error straight out of ``ensure_fleet_catalog_tables``. Even
    had the clause worked, the claim would have been a silent no-op and the
    re-read would have taken the "someone else already finished"
    short-circuit, returning success having applied nothing — and the write
    would then have failed at INSERT time on the very columns the skipped
    step adds. The claim must take over a completed row and run the step.
    """
    eng = _FakeEngine()
    gc = eng.graph_compute
    for ddl in _FRESH_PRE_ACL_DDL.values():
        gc.sql_exec(ddl)
    for table in (
        fct.TABLE_MCP_SERVER_DISCOVERY,
        fct.TABLE_MCP_PROMPTS,
        fct.TABLE_MCP_RESOURCES,
    ):
        gc.sql_exec(fct._DDL[table])
    gc.sql_exec(fct._LEDGER_DDL)
    gc.tables[fct._MIGRATION_LEDGER] = {
        fct._LOCK_ROW_ID: {
            "id": fct._LOCK_ROW_ID,
            "status": "complete",
            "claimant": "",
            "claimed_at": "2026-08-20T22:52:56+00:00",
            "version": fct._SCHEMA_VERSION_CURRENT,
            "migration_id": fct._CURRENT_MARKER,
            "checksum": "a-digest-from-when-this-shape-WAS-current",
            "applied_at": "2026-08-20T22:52:56+00:00",
        }
    }

    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        assert fct.ensure_fleet_catalog_tables(eng) is True

    assert {"acl_classification", "acl_owner_id", "acl_shared_scope"} <= gc.columns[
        fct.TABLE_MCP_SERVERS
    ]
    assert (
        gc.tables[fct._MIGRATION_LEDGER]["step__0004_acl_projection_columns"]["status"]
        == "applied"
    )
    assert gc.tables[fct._MIGRATION_LEDGER][fct._LOCK_ROW_ID]["status"] == "complete"
    # And no statement reached the engine carrying a clause it cannot honor.
    assert not [s for s in gc.statements if " ON CONFLICT " in s]


def test_unbound_unreachable_server_still_records_a_failure_observation():
    """BUG-PE-056. A failed probe never gets a discovery binding (the
    multiplexer mints one only for ``info["error"] is None``), so it used to
    produce NO ``mcp_server_discovery`` row at all -- making "unavailable"
    indistinguishable from "empty" to the dashboard, which reads
    ``tool_count`` off that row. A failure observation exposes no discovered
    capability, so it is recorded under the process-owned tenant-local
    visibility contract with empty principal/grant fields. Derived rows
    (tools/skills/prompts/resources) stay unwritten, exactly as before."""
    eng = _FakeEngine()
    catalog = _server_catalog(error="econnrefused: no route to host")
    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        result = fct.write_fleet_catalog(eng, catalog, discovery_bindings=None)

    assert result["status"] == "ok"
    assert result["servers_unreachable"] == 1
    assert result["discovery_written"] == 1
    row = next(iter(eng.graph_compute.tables[fct.TABLE_MCP_SERVER_DISCOVERY].values()))
    assert row["reachable"] is False
    assert "econnrefused" in row["last_error"]
    assert row["tenant_id"] == "tenant-a"
    assert row["discovery_authority_kind"] == fct.DISCOVERY_AUTHORITY_TENANT_LOCAL
    assert row["discovery_principal"] == ""
    assert row["discovery_grant_digest"] == ""
    # No unbound capability row was created as a side effect.
    assert not eng.graph_compute.tables[fct.TABLE_MCP_TOOLS]
    assert not eng.graph_compute.tables[fct.TABLE_SKILLS]


def test_unbound_reachable_server_still_writes_no_derived_rows():
    """The authority model is unchanged for a SUCCESSFUL probe with no
    binding: no discovery row, no tool rows. BUG-PE-056's exception is
    scoped strictly to a failure observation."""
    eng = _FakeEngine()
    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        result = fct.write_fleet_catalog(
            eng, _server_catalog(), discovery_bindings=None
        )
    assert result["discovery_written"] == 0
    assert not eng.graph_compute.tables[fct.TABLE_MCP_SERVER_DISCOVERY]
    assert not eng.graph_compute.tables[fct.TABLE_MCP_TOOLS]


def test_diverged_schema_still_detected_with_the_new_acl_columns_present():
    """The divergence guard must still fire on a hand-modified store even
    after the schema's current shape grew to include the ACL-projection
    columns -- growing by 4 legitimate columns must never be read as license
    to also accept an arbitrary 5th, unrecognized one."""
    eng = _FakeEngine()
    gc = eng.graph_compute
    for ddl in fct._DDL.values():
        gc.sql_exec(ddl)
    gc.columns[fct.TABLE_MCP_TOOLS].add("mystery_acl_column")
    with pytest.raises(fct.FleetCatalogSchemaDivergedError):
        fct._claim_and_migrate(eng)


def test_too_new_ledger_still_refused_after_the_acl_step_exists():
    """A ledger recording a migration this code version does not know about
    is still refused exactly as before step 0004 was added -- adding a new
    KNOWN step must never widen what counts as recognized."""
    eng = _FakeEngine()
    gc = eng.graph_compute
    for ddl in fct._DDL.values():
        gc.sql_exec(ddl)
    gc.sql_exec(fct._LEDGER_DDL)
    gc.tables[fct._MIGRATION_LEDGER][fct._LOCK_ROW_ID] = {
        "id": fct._LOCK_ROW_ID,
        "status": "complete",
        "claimant": "",
        "claimed_at": "2030-01-01T00:00:00+00:00",
        "version": 99,
        "migration_id": "0005_from_a_future_release",
        "checksum": "deadbeef",
        "applied_at": "2030-01-01T00:00:00+00:00",
    }
    with pytest.raises(fct.FleetCatalogSchemaTooNewError):
        fct._claim_and_migrate(eng)


def test_write_fleet_catalog_stamps_acl_projection_fields_from_the_write_time_actor():
    """The writer stamps ``acl_classification``/``acl_owner_id``/
    ``acl_shared_scope`` from the SAME policy
    (``tenant_sharing.stamp_ownership``/``stamp_classification``) the
    matching KG node write uses for these labels -- neither "MCPServer" nor
    "Tool" is a PUBLIC_CATALOG_LABEL, so the classification stays
    ``confidential`` and the owner marker still names the writer.

    ``acl_shared_scope`` is ``org``, not ``private``: ``_session`` mints an
    ``ActorType.AUTOMATED_SERVICE`` actor, and ``stamp_ownership`` now scopes
    a service write to the org by ACTOR TYPE rather than by the writer's
    current ``kg:admin`` role (see ``test_tenant_sharing.py``
    ``::test_stamp_ownership_service_is_org_scoped_regardless_of_privilege``).
    The fleet tool catalog is exactly the platform data that change exists
    for -- deciding its durable visibility from a mutable IdP role is what
    orphaned 23,994 rows behind the engine's row-level owner check during a
    two-day role outage. The owner marker is retained as provenance; an
    explicit ``org`` scope is what the row-visibility check reads."""
    eng = _FakeEngine()
    with (
        use_actor(_session("tenant-a", actor_id="sync-actor").actor),
        use_session(_session("tenant-a", actor_id="sync-actor")),
    ):
        _write_fleet_catalog(eng, _server_catalog())
    server_row = eng.graph_compute.tables[fct.TABLE_MCP_SERVERS]["mcp_server_srv"]
    tool_row = _one_row(fct.TABLE_MCP_TOOLS, "tool_srv_t1", eng)

    assert server_row["acl_classification"] == "confidential"
    assert server_row["acl_owner_id"] == "sync-actor"
    assert server_row["acl_shared_scope"] == "org"
    assert tool_row["acl_classification"] == "confidential"
    assert tool_row["acl_owner_id"] == "sync-actor"
    assert tool_row["acl_shared_scope"] == "org"
    assert tool_row["kg_node_id"] == "tool_srv_t1"


def test_write_fleet_catalog_leaves_acl_projection_null_without_a_verified_actor():
    """No ambient actor bound at write time must not crash the catalog write
    -- it simply carries no SQL-authoritative ACL yet, exactly like a legacy
    pre-migration row. The suite-wide ``isolate_graph_compute_engine``
    fixture binds a real ambient actor for every test by default, so this
    runs in a genuinely fresh :class:`contextvars.Context` (the same
    technique ``test_secured_reads.test_missing_identity_fails_closed``
    uses) to actually get an unbound one, rather than a session with no
    actor at all (not otherwise reachable through this suite)."""
    eng = _FakeEngine()

    def _write_without_an_actor():
        with use_session(_session("tenant-a")):
            # `use_session` alone does not bind `current_actor()`.
            return fct.write_fleet_catalog(
                eng,
                _server_catalog(),
                discovery_bindings={
                    "srv": fct.TenantLocalDiscoveryBinding(tenant_id="tenant-a")
                },
            )

    result = contextvars.Context().run(_write_without_an_actor)
    assert result["status"] == "ok"
    tool_row = _one_row(fct.TABLE_MCP_TOOLS, "tool_srv_t1", eng)
    assert tool_row.get("acl_classification") is None
    assert tool_row.get("acl_owner_id") is None
