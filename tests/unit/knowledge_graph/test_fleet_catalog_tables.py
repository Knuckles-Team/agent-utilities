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

import re

import pytest

from agent_utilities.knowledge_graph.core import fleet_catalog_tables as fct
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    current_session,
    suspend_session,
    use_session,
)
from agent_utilities.mcp.remote_oauth_broker import OAuthGrantBinding
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
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.tables: dict[str, dict[str, dict]] = {}

    def sql_exec(self, statement: str):
        self.statements.append(statement)
        head = statement.strip().split(None, 1)[0].upper()

        if head == "CREATE":
            m = re.match(r"CREATE TABLE IF NOT EXISTS (\w+)", statement)
            if m:
                self.tables.setdefault(m.group(1), {})
            return {"ok": True}

        if head == "ALTER":
            # The production migration is idempotent; this fake only needs to
            # record that the additive legacy-column step was issued.
            return {"ok": True}

        if head == "SELECT":
            m = re.match(
                r"SELECT \* FROM (\w+) WHERE tenant_id = (.+?) AND (\w+) IN \((.*)\)$",
                statement,
                re.DOTALL,
            )
            if not m:
                return []
            table, tenant_lit, col, in_list = m.groups()
            tenant_id = _parse_literal(tenant_lit)
            ids = [_parse_literal(t) for t in _split_top(in_list)]
            store = self.tables.get(table, {})
            return [
                dict(row)
                for row in store.values()
                if row.get("tenant_id") == tenant_id and row.get(col) in ids
            ]

        if head == "INSERT":
            m = re.match(
                r"INSERT INTO (\w+) \((.*?)\) VALUES (.*)$", statement, re.DOTALL
            )
            assert m, f"unrecognized INSERT: {statement}"
            table, cols_str, values_str = m.groups()
            cols = [c.strip() for c in cols_str.split(",")]
            store = self.tables.setdefault(table, {})
            for row_str in _extract_value_rows(values_str):
                vals = [_parse_literal(t) for t in _split_top(row_str)]
                row = dict(zip(cols, vals, strict=True))
                store[str(row.get("id"))] = row
            return {"ok": True}

        if head == "UPDATE":
            m = re.match(
                r"UPDATE (\w+) SET (.*) WHERE (\w+) = (.+?) AND tenant_id = (.+)$",
                statement,
                re.DOTALL,
            )
            assert m, f"unrecognized UPDATE: {statement}"
            table, set_str, id_col, id_lit, tenant_lit = m.groups()
            store = self.tables.setdefault(table, {})
            row_id = _parse_literal(id_lit)
            existing = dict(store.get(str(row_id), {}))
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
    assert len(create_stmts) == len(fct._DDL)  # once, not once per write call
    migration_stmts = [
        s for s in eng.graph_compute.statements if s.startswith("ALTER TABLE")
    ]
    assert len(migration_stmts) == sum(
        len(columns) for columns in fct._DISCOVERY_BINDING_MIGRATION.values()
    )
    assert all("ADD COLUMN IF NOT EXISTS" not in s for s in migration_stmts)


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
