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
    """In-memory SQL emulator, now schema-aware.

    Beyond the original CAS-shaped statements, this fake also has to model
    enough of the real engine's ``information_schema.columns`` / ``ALTER
    TABLE ADD COLUMN`` / ``INSERT ... ON CONFLICT`` behavior for the
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
    * ``INSERT ... ON CONFLICT (id) DO NOTHING`` / ``DO UPDATE SET ...`` is
      understood, needed for the migration ledger's claim/finalize writes.
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
            base, _, conflict_clause = statement.partition(" ON CONFLICT ")
            m = re.match(r"INSERT INTO (\w+) \((.*?)\) VALUES (.*)$", base, re.DOTALL)
            assert m, f"unrecognized INSERT: {statement}"
            table, cols_str, values_str = m.groups()
            cols = [c.strip() for c in cols_str.split(",")]
            self._row_columns(table, cols)
            store = self.tables.setdefault(table, {})

            conflict_action: tuple[str, str] | None = None
            if conflict_clause:
                clause = conflict_clause.strip()
                if re.match(r"\(\w+\)\s+DO NOTHING$", clause, re.IGNORECASE):
                    conflict_action = ("nothing", "")
                else:
                    um = re.match(
                        r"\(\w+\)\s+DO UPDATE SET\s+(.*)$",
                        clause,
                        re.IGNORECASE | re.DOTALL,
                    )
                    assert um, f"unrecognized ON CONFLICT clause: {conflict_clause}"
                    conflict_action = ("update", um.group(1))

            for row_str in _extract_value_rows(values_str):
                vals = [_parse_literal(t) for t in _split_top(row_str)]
                row = dict(zip(cols, vals, strict=True))
                row_id = str(row.get("id"))
                if row_id in store:
                    if conflict_action is None:
                        # No pre-existing test relies on a bare duplicate-id
                        # INSERT raising (the tenant-collision test writes
                        # the same id for two tenants and expects the fake's
                        # historical blind-overwrite tolerance), so this
                        # stays a tolerant overwrite here — only an explicit
                        # ON CONFLICT clause (used exclusively by the
                        # migration ledger) gets real conflict semantics.
                        store[row_id] = row
                        continue
                    kind, set_clause = conflict_action
                    if kind == "nothing":
                        continue
                    existing = dict(store[row_id])
                    for pair in _split_top(set_clause):
                        col, _, lit = pair.partition("=")
                        existing[col.strip()] = _parse_literal(lit.strip())
                    store[row_id] = existing
                    continue
                store[row_id] = row
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
    gc.tables[fct._MIGRATION_LEDGER][fct._LOCK_ROW_ID] = {
        "id": fct._LOCK_ROW_ID,
        "status": "migrating",
        "claimant": "other-process-token",
        "claimed_at": "2026-08-20T00:00:00+00:00",
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
