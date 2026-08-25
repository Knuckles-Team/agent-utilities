"""Focused tenant, cursor, bound, and degraded-read registry contracts."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from agent_utilities.gateway import registry_api
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    current_session,
    suspend_session,
    use_session,
)
from agent_utilities.mcp.remote_oauth_broker import OAuthGrantBinding
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

# --- A small, faithful SQL WHERE-clause interpreter for the fake engine ---
#
# The production code (`registry_api._build_where`/`_keyset_predicate`) now
# pushes the tenant/authorization/query/keyset predicate into the SQL text
# itself, so a test double that just string-searches for a few markers (the
# pre-pushdown approach) would no longer prove anything: it wouldn't catch a
# predicate that is present in the statement but wrong. This tokenizes and
# evaluates the WHERE clause for real, against each candidate row, so the
# pagination/authorization/filter tests below exercise the actual predicate
# the route builds rather than trusting the route's own bookkeeping.

_TOKEN_RE = re.compile(
    r"\s+|\(|\)|,|'(?:[^']|'')*'|>=|<=|>|<|=|[A-Za-z_][A-Za-z0-9_]*|[0-9]+"
)


def _tokenize_where(where: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    for match in _TOKEN_RE.finditer(where):
        text = match.group(0)
        if text.isspace():
            continue
        if text in ("(", ")", ","):
            tokens.append((text, text))
        elif text.startswith("'"):
            tokens.append(("literal", text[1:-1].replace("''", "'")))
        elif text in (">", "<", "=", ">=", "<="):
            tokens.append(("op", text))
        elif text[0].isdigit():
            tokens.append(("num", text))
        else:
            tokens.append(("word", text))
    return tokens


class _WhereParser:
    """Recursive-descent parser over the small predicate grammar the route
    generates: AND/OR of `col = 'lit'`, `col IN ('a','b')`,
    `strpos(LOWER(col), LOWER('lit')) > 0`, and `LOWER(col) {=,>} LOWER('lit')`,
    with arbitrary parenthesized nesting."""

    def __init__(self, tokens: list[tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str | None, str | None]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def take(self) -> tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, kind: str, value: str | None = None) -> tuple[str, str]:
        tok = self.take()
        assert tok[0] == kind and (value is None or tok[1].upper() == value), tok
        return tok

    def parse_or(self):
        node = self.parse_and()
        while self.peek() == ("word", "OR"):
            self.take()
            node = ("or", node, self.parse_and())
        return node

    def parse_and(self):
        node = self.parse_factor()
        while self.peek() == ("word", "AND"):
            self.take()
            node = ("and", node, self.parse_factor())
        return node

    def parse_factor(self):
        if self.peek()[0] == "(":
            self.take()
            node = self.parse_or()
            self.expect(")")
            return node
        return self.parse_predicate()

    def parse_predicate(self):
        kind, value = self.take()
        assert kind == "word", (kind, value)
        upper = value.upper()
        if upper == "STRPOS":
            self.expect("(")
            self.expect("word", "LOWER")
            self.expect("(")
            _, col = self.expect("word")
            self.expect(")")
            self.expect(",")
            self.expect("word", "LOWER")
            self.expect("(")
            _, lit = self.expect("literal")
            self.expect(")")
            self.expect(")")
            self.expect("op", ">")
            self.expect("num", "0")
            return ("strpos", col, lit)
        if upper == "LOWER":
            self.expect("(")
            _, col = self.expect("word")
            self.expect(")")
            _, op = self.expect("op")
            self.expect("word", "LOWER")
            self.expect("(")
            _, lit = self.expect("literal")
            self.expect(")")
            return ("lower_cmp", op, col, lit)
        if upper == "FALSE":
            return ("false",)
        col = value
        peek_kind, peek_val = self.peek()
        if peek_kind == "op":
            self.take()
            op = peek_val
        elif peek_kind == "word" and (peek_val or "").upper() == "IN":
            self.take()
            op = "IN"
        else:  # pragma: no cover - defensive; every generated predicate matches
            raise AssertionError((peek_kind, peek_val))
        if op == "IN":
            self.expect("(")
            _, lit = self.expect("literal")
            lits = [lit]
            while self.peek() == (",", ","):
                self.take()
                _, lit = self.expect("literal")
                lits.append(lit)
            self.expect(")")
            return ("in", col, lits)
        _, lit = self.expect("literal")
        return ("cmp", op, col, lit)


def _eval_where_node(node: tuple, row: dict[str, Any]) -> bool:
    kind = node[0]
    if kind == "and":
        return _eval_where_node(node[1], row) and _eval_where_node(node[2], row)
    if kind == "or":
        return _eval_where_node(node[1], row) or _eval_where_node(node[2], row)
    if kind == "false":
        return False
    if kind == "cmp":
        _, op, col, lit = node
        val = str(row.get(col) if row.get(col) is not None else "")
        return val == lit if op == "=" else val > lit
    if kind == "in":
        _, col, lits = node
        val = str(row.get(col) if row.get(col) is not None else "")
        return val in lits
    if kind == "strpos":
        _, col, lit = node
        val = str(row.get(col) if row.get(col) is not None else "")
        return lit.lower() in val.lower()
    if kind == "lower_cmp":
        _, op, col, lit = node
        val = str(row.get(col) if row.get(col) is not None else "").lower()
        target = lit.lower()
        return val == target if op == "=" else val > target
    raise AssertionError(node)  # pragma: no cover - defensive


def _eval_where(where: str, row: dict[str, Any]) -> bool:
    parser = _WhereParser(_tokenize_where(where))
    node = parser.parse_or()
    return _eval_where_node(node, row)


_STATEMENT_RE = re.compile(
    r"^SELECT (?P<cols>.+?) FROM (?P<table>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?: WHERE (?P<where>.+?))?"
    r"(?: ORDER BY (?P<order>.+?))?"
    r"(?: LIMIT (?P<limit>\d+))?$"
)


class _FakeGraphCompute:
    """A real (if minimal) SQL engine for `mcp_*`-shaped tables: it actually
    evaluates WHERE, ORDER BY, and LIMIT rather than trusting the caller —
    so a test asserting on its output is exercising the production SQL text,
    not restating the route's own logic."""

    def __init__(self, rows: dict[str, list[dict[str, Any]]] | None = None):
        self.rows = rows or {}
        self.statements: list[str] = []
        self.fail = False
        self.return_raw = False
        # Simulates a broken engine that ignores the requested LIMIT — used
        # to prove the route's own bound-exceeded guard fires.
        self.ignore_limit = False

    def sql_exec(self, statement: str):
        self.statements.append(statement)
        if self.fail:
            raise OSError("catalog backend unavailable")
        match = _STATEMENT_RE.match(statement)
        assert match, statement
        table = match.group("table")
        cols = match.group("cols")
        where = match.group("where")
        order = match.group("order")
        limit = match.group("limit")
        rows_all = self.rows.get(table, [])
        is_count = cols.strip().upper().startswith("COUNT(")
        if self.return_raw:
            # `return_raw` models a misbehaving/degraded engine projection
            # that ignores the WHERE predicate entirely. A COUNT(*) query
            # still returns a validly-shaped (if unfiltered) aggregate row,
            # so the per-row scope-validation path (not the count-shape
            # check) is what a malformed *page* actually exercises.
            if is_count:
                return [{"row_count": len(rows_all)}]
            return list(rows_all)
        matched = [row for row in rows_all if where is None or _eval_where(where, row)]
        if is_count:
            return [{"row_count": len(matched)}]
        if order:
            order_col = order.split(",")[0].strip()
            if order_col.upper().startswith("LOWER("):
                order_col = order_col[len("LOWER(") : -1]
            matched = sorted(
                matched,
                key=lambda row: (
                    str(row.get(order_col) or "").lower(),
                    str(row.get("id") or ""),
                ),
            )
        if limit is not None and not self.ignore_limit:
            matched = matched[: int(limit)]
        return matched


class _FakeEngine:
    def __init__(self, rows: dict[str, list[dict[str, Any]]] | None = None):
        self.graph_compute = _FakeGraphCompute(rows)


class _AuthorityMiddleware:
    def __init__(self, app, actor: ActorContext, session: GraphSession):
        self.app = app
        self.actor = actor
        self.session = session

    async def __call__(self, scope, receive, send):
        with use_actor(self.actor), use_session(self.session):
            await self.app(scope, receive, send)


def _authority_app(
    monkeypatch,
    *,
    engine: _FakeEngine,
    actor_id: str = "actor-a",
    tenant_id: str = "tenant-a",
):
    actor = ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("registry:read",),
        tenant_id=tenant_id,
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=tenant_id,
        scopes=frozenset({"kg:read"}),
        graph=tenant_id,
        policy_version="test",
        audience="test",
    )
    monkeypatch.setattr(registry_api, "_get_catalog_engine", lambda: engine)
    monkeypatch.setattr(
        registry_api,
        "_resolve_current_discovery_grants",
        lambda actor: (_grant_digest(actor.actor_id, actor.tenant_id),),
    )
    app = FastAPI()
    registry_api.register_registry_routes(app, prefix="/api")
    return TestClient(_AuthorityMiddleware(app, actor, session))


def _grant_digest(actor_id: str, tenant_id: str = "tenant-a") -> str:
    return OAuthGrantBinding(
        tenant_id=tenant_id,
        principal_id=actor_id,
        provider_id="acme",
        resource_url="https://protected-mcp.example.com/mcp",
        audience="https://protected-mcp.example.com/mcp",
        granted_scopes=("mcp:read",),
        key_version=1,
        grant_revision=f"registry-test-{tenant_id}-{actor_id}",
    ).fingerprint


def _rows() -> dict[str, list[dict[str, Any]]]:
    grant_a = _grant_digest("actor-a")
    grant_b = _grant_digest("actor-b")
    return {
        "mcp_servers": [
            {
                "id": "mcp_server_alpha",
                "tenant_id": "tenant-a",
                "name": "alpha",
                "transport": "http",
                "url": "https://user:password@example.test/api?token=secret",
                "enabled": True,
            },
            {
                "id": "mcp_server_beta",
                "tenant_id": "tenant-a",
                "name": "beta",
                "transport": "stdio",
                "url": "https://example.test/mcp/secret-token",
                "enabled": False,
            },
            {
                "id": "mcp_server_alpha",
                "tenant_id": "tenant-b",
                "name": "foreign",
                "transport": "http",
                "url": "https://foreign.test",
                "enabled": True,
            },
        ],
        "mcp_server_discovery": [
            {
                "id": "disc_alpha",
                "tenant_id": "tenant-a",
                "server_id": "mcp_server_alpha",
                "server_name": "alpha",
                "reachable": False,
                "last_error": "/private/path/token=secret",
                "tool_count": 1,
                "skill_count": 0,
                "prompt_count": 0,
                "resource_count": 0,
                "observed_at": "2026-08-18T00:00:00Z",
                "discovery_authority_kind": "oauth_grant",
                "discovery_principal": "actor-a",
                "discovery_grant_digest": grant_a,
            },
            {
                "id": "disc_other",
                "tenant_id": "tenant-a",
                "server_id": "mcp_server_alpha",
                "server_name": "alpha",
                "reachable": True,
                "last_error": "",
                "tool_count": 2,
                "skill_count": 0,
                "prompt_count": 0,
                "resource_count": 0,
                "observed_at": "2026-08-18T00:00:00Z",
                "discovery_authority_kind": "oauth_grant",
                "discovery_principal": "actor-b",
                "discovery_grant_digest": grant_b,
            },
        ],
        "mcp_tools": [
            {
                "id": "tool_alpha_a",
                "tenant_id": "tenant-a",
                "server_id": "mcp_server_alpha",
                "server_name": "alpha",
                "name": "only-a",
                "description": "",
                "schema_digest": "a",
                "tool_mode": "verbose",
                "enabled": True,
                "discovery_authority_kind": "oauth_grant",
                "discovery_principal": "actor-a",
                "discovery_grant_digest": grant_a,
            },
            {
                "id": "tool_alpha_b",
                "tenant_id": "tenant-a",
                "server_id": "mcp_server_alpha",
                "server_name": "alpha",
                "name": "only-b",
                "description": "",
                "schema_digest": "b",
                "tool_mode": "verbose",
                "enabled": True,
                "discovery_authority_kind": "oauth_grant",
                "discovery_principal": "actor-b",
                "discovery_grant_digest": grant_b,
            },
        ],
    }


def _rows_with_local() -> dict[str, list[dict[str, Any]]]:
    rows = _rows()
    rows["mcp_server_discovery"].append(
        {
            **rows["mcp_server_discovery"][0],
            "id": "disc_local",
            "server_name": "local-alpha",
            "server_id": "mcp_server_local-alpha",
            "discovery_authority_kind": "tenant_local",
            "discovery_principal": "",
            "discovery_grant_digest": "",
        }
    )
    rows["mcp_tools"].append(
        {
            **rows["mcp_tools"][0],
            "id": "tool_local",
            "server_name": "local-alpha",
            "server_id": "mcp_server_local-alpha",
            "name": "local-tool",
            "discovery_authority_kind": "tenant_local",
            "discovery_principal": "",
            "discovery_grant_digest": "",
        }
    )
    return rows


def _registry_routes(app: Any) -> list[tuple[str, set[str]]]:
    """Flatten FastAPI/Starlette route wrappers for a stable contract check."""

    found: list[tuple[str, set[str]]] = []
    pending = list(getattr(app, "routes", ()))
    while pending:
        route = pending.pop()
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if isinstance(path, str) and "registry" in path:
            found.append((path, set(methods or ())))
        original = getattr(route, "original_router", None)
        if original is not None:
            pending.extend(getattr(original, "routes", ()))
        pending.extend(getattr(route, "routes", ()) or ())
    return found


def test_registry_reads_native_catalog_with_tenant_and_principal_predicate(monkeypatch):
    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers", params={"q": "alpha"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    assert body["items"][0]["name"] == "alpha"
    assert (
        body["items"][0]["url"] == ""
    )  # userinfo/query credentials are never returned
    # Both the count and the page read push the tenant/authorization AND the
    # `q` filter into the SQL text itself (filter pushdown) — the last two
    # statements are the count then the page.
    statements = engine.graph_compute.statements[-2:]
    for statement in statements:
        assert "tenant_id = 'tenant-a'" in statement
        assert "strpos(LOWER(name), LOWER('alpha')) > 0" in statement
    assert "LIMIT" in statements[-1]  # page read carries a LIMIT
    assert "tenant-b" not in response.text


def test_catalog_sql_runs_under_the_fixed_catalog_service_identity(monkeypatch):
    """D-catalog-503-human root-cause regression.

    The engine's native ``Method::Sql`` RPC opens an OWNER-SCOPED catalog
    keyed by whichever verified actor issues the call (see
    ``_catalog_service_session``'s docstring) -- so if the RPC ran under the
    REQUESTING actor's own session, a distinct actor from whoever wrote the
    fleet catalog tables would always get "table not found", exactly the
    live 503 this fix closes. Prove ``_require_sql_exec``'s wrapper actually
    swaps the ambient session for the duration of the SQL call rather than
    handing the caller's own actor through: the SAME query text still gets
    the caller's tenant/principal predicate (authorization is unaffected),
    but the identity ``sql_exec`` observes as ambient must differ from the
    HTTP caller's own actor id.
    """
    engine = _FakeEngine(_rows())
    observed_actor_ids: list[str] = []
    real_sql_exec = engine.graph_compute.sql_exec

    def _recording_sql_exec(statement: str):
        session = current_session()
        observed_actor_ids.append(session.actor.actor_id if session else "")
        return real_sql_exec(statement)

    monkeypatch.setattr(engine.graph_compute, "sql_exec", _recording_sql_exec)
    client = _authority_app(monkeypatch, engine=engine, actor_id="actor-a")

    response = client.get("/api/registry/servers")

    assert response.status_code == 200, response.text
    assert observed_actor_ids, "sql_exec was never called"
    assert all(actor_id != "actor-a" for actor_id in observed_actor_ids), (
        "the catalog SQL RPC ran under the caller's own session "
        f"({observed_actor_ids!r}) instead of the fixed catalog-service "
        "identity -- this is the exact owner-scoped-catalog 503 the fix "
        "closes"
    )
    # The row-level WHERE predicate is still the REAL caller's tenant/actor
    # -- only the RPC-executing identity changed, never the authorization
    # boundary.
    assert "tenant_id = 'tenant-a'" in engine.graph_compute.statements[-1]


def test_discovery_is_principal_scoped_and_error_is_classified(monkeypatch):
    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/discoveries")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    assert body["items"][0]["last_error"] == "unavailable"
    assert "actor-b" not in response.text
    assert "discovery_principal = 'actor-a'" in engine.graph_compute.statements[-1]


def test_derived_catalog_is_disjoint_by_principal_and_grant(monkeypatch):
    engine = _FakeEngine(_rows())
    actor_a = _authority_app(monkeypatch, engine=engine, actor_id="actor-a")
    response_a = actor_a.get("/api/registry/tools")
    assert response_a.status_code == 200, response_a.text
    assert [item["name"] for item in response_a.json()["items"]] == ["only-a"]
    assert "discovery_grant_digest IN (" in engine.graph_compute.statements[-1]

    actor_b = _authority_app(monkeypatch, engine=engine, actor_id="actor-b")
    response_b = actor_b.get("/api/registry/tools")
    assert response_b.status_code == 200, response_b.text
    assert [item["name"] for item in response_b.json()["items"]] == ["only-b"]
    assert "only-a" not in response_b.text


def test_local_discovery_is_tenant_readable_without_oauth_grant(monkeypatch):
    engine = _FakeEngine(_rows_with_local())
    client = _authority_app(monkeypatch, engine=engine)
    monkeypatch.setattr(
        registry_api, "_resolve_current_discovery_grants", lambda _actor: ()
    )

    response = client.get("/api/registry/tools")

    assert response.status_code == 200, response.text
    assert [item["name"] for item in response.json()["items"]] == ["local-tool"]
    statement = engine.graph_compute.statements[-1]
    assert "discovery_authority_kind = 'tenant_local'" in statement
    assert "discovery_grant_digest IN (" not in statement


def test_local_visibility_does_not_broaden_oauth_principal_isolation(monkeypatch):
    rows = _rows_with_local()
    engine = _FakeEngine(rows)
    client_a = _authority_app(monkeypatch, engine=engine, actor_id="actor-a")
    response_a = client_a.get("/api/registry/tools")
    assert response_a.status_code == 200, response_a.text
    assert {item["name"] for item in response_a.json()["items"]} == {
        "local-tool",
        "only-a",
    }

    client_b = _authority_app(monkeypatch, engine=engine, actor_id="actor-b")
    response_b = client_b.get("/api/registry/tools")
    assert response_b.status_code == 200, response_b.text
    assert {item["name"] for item in response_b.json()["items"]} == {
        "local-tool",
        "only-b",
    }
    assert "only-a" not in response_b.text


# --- Generic per-kind fleet-public / user-scoped coverage -----------------
#
# The two tests above (`test_local_discovery_is_tenant_readable_without_oauth_grant`,
# `test_local_visibility_does_not_broaden_oauth_principal_isolation`) prove the
# public-fleet-row / private-OAuth-row design only for `tools`. Every kind that
# carries the `discovery_*` columns (`discoveries`, `tools`, `prompts`,
# `resources`, `skills`) shares the identical `_build_where` predicate, so the
# same two properties must hold for each of them individually — this is the
# harness/fleet-catalog tenant-public directive (AUTHZ LANE A): a `kg:read`
# principal with no discovery grant must see every fleet-ingested (tenant_local)
# row, and cross-principal isolation of OAuth-bound discovery rows must not be
# broadened by that public visibility.

_DISCOVERY_SCOPED_KINDS = ("discoveries", "tools", "prompts", "resources", "skills")


def _generic_row(
    kind: str, *, suffix: str, authority: str, principal: str, grant: str
) -> dict[str, Any]:
    """Build one row shaped for ``kind``'s table, using only real columns."""

    spec = registry_api._KIND_SPECS[kind]
    template = {
        "id": f"{kind}_{suffix}",
        "tenant_id": "tenant-a",
        "server_id": "mcp_server_alpha",
        "server_name": f"{kind}-{suffix}",
        "name": f"{kind}-{suffix}",
        "description": "",
        "uri": "",
        "skill_type": "",
        "classification": "",
        "provider": "",
        "mcp_server": "",
        "enabled": True,
        "transport": "http",
        "url": "",
        "schema_digest": "",
        "tool_mode": "verbose",
        "mime_type": "",
        "resource_kind": "",
        "reachable": True,
        "last_error": "",
        "tool_count": 0,
        "skill_count": 0,
        "prompt_count": 0,
        "resource_count": 0,
        "observed_at": "2026-08-18T00:00:00Z",
        "discovery_authority_kind": authority,
        "discovery_principal": principal,
        "discovery_grant_digest": grant,
    }
    return {column: template[column] for column in spec.columns}


def _fleet_and_user_scoped_rows(kind: str) -> tuple[str, list[dict[str, Any]]]:
    """One tenant_local (fleet) row plus one oauth_grant row per test actor."""

    spec = registry_api._KIND_SPECS[kind]
    grant_a = _grant_digest("actor-a")
    grant_b = _grant_digest("actor-b")
    rows = [
        _generic_row(
            kind, suffix="local", authority="tenant_local", principal="", grant=""
        ),
        _generic_row(
            kind,
            suffix="a",
            authority="oauth_grant",
            principal="actor-a",
            grant=grant_a,
        ),
        _generic_row(
            kind,
            suffix="b",
            authority="oauth_grant",
            principal="actor-b",
            grant=grant_b,
        ),
    ]
    return spec.table, rows


@pytest.mark.parametrize("kind", _DISCOVERY_SCOPED_KINDS)
def test_fleet_rows_are_tenant_public_for_every_discovery_scoped_kind(
    monkeypatch, kind
):
    """kg:read with NO discovery grants still sees every fleet (tenant_local)
    row — the bug this lane fixes, proven for each affected kind, not just
    ``tools``."""

    table, rows = _fleet_and_user_scoped_rows(kind)
    engine = _FakeEngine({table: rows})
    client = _authority_app(monkeypatch, engine=engine)
    monkeypatch.setattr(
        registry_api, "_resolve_current_discovery_grants", lambda _actor: ()
    )

    response = client.get(f"/api/registry/{kind}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    names = {item.get("name") or item.get("server_name") for item in body["items"]}
    assert names == {f"{kind}-local"}
    statement = engine.graph_compute.statements[-1]
    assert "discovery_authority_kind = 'tenant_local'" in statement
    assert "discovery_grant_digest IN (" not in statement
    assert "actor-a" not in response.text
    assert "actor-b" not in response.text


@pytest.mark.parametrize("kind", _DISCOVERY_SCOPED_KINDS)
def test_cross_principal_isolation_holds_alongside_public_fleet_rows(monkeypatch, kind):
    """The load-bearing security test: making fleet rows tenant-public must
    NOT leak one principal's OAuth-bound discovery rows to another principal,
    for every affected kind — not just ``tools``."""

    table, rows = _fleet_and_user_scoped_rows(kind)
    engine = _FakeEngine({table: rows})

    client_a = _authority_app(monkeypatch, engine=engine, actor_id="actor-a")
    response_a = client_a.get(f"/api/registry/{kind}")
    assert response_a.status_code == 200, response_a.text
    names_a = {
        item.get("name") or item.get("server_name")
        for item in response_a.json()["items"]
    }
    assert names_a == {f"{kind}-local", f"{kind}-a"}
    assert f"{kind}-b" not in response_a.text

    client_b = _authority_app(monkeypatch, engine=engine, actor_id="actor-b")
    response_b = client_b.get(f"/api/registry/{kind}")
    assert response_b.status_code == 200, response_b.text
    names_b = {
        item.get("name") or item.get("server_name")
        for item in response_b.json()["items"]
    }
    assert names_b == {f"{kind}-local", f"{kind}-b"}
    assert f"{kind}-a" not in response_b.text


def test_servers_kind_has_no_discovery_predicate_regression(monkeypatch):
    """Regression guard: ``servers`` had no ``principal``/``grant`` columns
    before this lane and must not gain one — it was already tenant-public."""

    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers")

    assert response.status_code == 200, response.text
    statement = engine.graph_compute.statements[-1]
    assert "discovery_" not in statement
    assert "principal" not in statement
    assert "grant" not in statement


def test_registry_predicate_accepts_only_current_grants_for_one_principal(monkeypatch):
    rows = _rows()
    grant_a = _grant_digest("actor-a")
    grant_refresh = _grant_digest("actor-a-refresh")
    rows["mcp_tools"].append(
        {
            **rows["mcp_tools"][0],
            "id": "tool_alpha_refresh",
            "name": "only-a-refresh",
            "discovery_grant_digest": grant_refresh,
        }
    )
    engine = _FakeEngine(rows)
    client = _authority_app(monkeypatch, engine=engine)
    monkeypatch.setattr(
        registry_api,
        "_resolve_current_discovery_grants",
        lambda _actor: (grant_a, grant_refresh),
    )

    response = client.get("/api/registry/tools")

    assert response.status_code == 200, response.text
    assert {item["name"] for item in response.json()["items"]} == {
        "only-a",
        "only-a-refresh",
    }
    statement = engine.graph_compute.statements[-1]
    assert "discovery_grant_digest IN (" in statement


def test_cursor_is_stable_scope_bound_and_tamper_evident(monkeypatch):
    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)

    first = client.get("/api/registry/servers", params={"limit": 1})
    assert first.status_code == 200
    cursor = first.json()["next_cursor"]
    assert cursor

    second = client.get("/api/registry/servers", params={"limit": 1, "cursor": cursor})
    assert second.status_code == 200
    assert [item["name"] for item in second.json()["items"]] == ["beta"]
    assert second.json()["items"][0]["url"] == "https://example.test"
    assert "secret-token" not in second.text

    tampered = client.get(
        "/api/registry/servers", params={"limit": 1, "cursor": cursor + "x"}
    )
    assert tampered.status_code == 400
    assert tampered.json()["detail"] == "invalid registry cursor"


def test_cursor_is_bound_to_actor_tenant_filter_and_kind(monkeypatch):
    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)
    first = client.get("/api/registry/servers", params={"limit": 1})
    cursor = first.json()["next_cursor"]
    assert cursor

    actor_client = _authority_app(monkeypatch, engine=engine, actor_id="actor-b")
    assert (
        actor_client.get(
            "/api/registry/servers", params={"limit": 1, "cursor": cursor}
        ).status_code
        == 400
    )
    tenant_client = _authority_app(
        monkeypatch, engine=engine, actor_id="actor-a", tenant_id="tenant-b"
    )
    assert (
        tenant_client.get(
            "/api/registry/servers", params={"limit": 1, "cursor": cursor}
        ).status_code
        == 400
    )
    assert (
        client.get(
            "/api/registry/servers",
            params={"limit": 1, "q": "alpha", "cursor": cursor},
        ).status_code
        == 400
    )
    assert (
        client.get(
            "/api/registry/tools", params={"limit": 1, "cursor": cursor}
        ).status_code
        == 400
    )


def test_expired_cursor_is_rejected(monkeypatch):
    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)
    monkeypatch.setattr(registry_api, "_CURSOR_TTL_SECONDS", -1.0)
    expired = registry_api._cursor_token(
        kind="servers",
        query="",
        after=("alpha", "mcp_server_alpha"),
        tenant="tenant-a",
        principal="actor-a",
        grant_digest="",
    )

    response = client.get(
        "/api/registry/servers", params={"limit": 1, "cursor": expired}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid registry cursor"


def test_cursor_round_trip_covers_every_row_exactly_once(monkeypatch):
    source = _rows()["mcp_servers"][0]
    rows = [
        {**source, "id": f"mcp_server_{index:03d}", "name": f"server-{index:03d}"}
        for index in range(37)
    ]
    engine = _FakeEngine({"mcp_servers": rows})
    client = _authority_app(monkeypatch, engine=engine)

    seen_ids: list[str] = []
    cursor = None
    for _ in range(100):
        params: dict[str, Any] = {"limit": 5}
        if cursor:
            params["cursor"] = cursor
        response = client.get("/api/registry/servers", params=params)
        assert response.status_code == 200, response.text
        body = response.json()
        seen_ids.extend(item["id"] for item in body["items"])
        cursor = body["next_cursor"]
        if not cursor:
            break
    else:  # pragma: no cover - failure path only
        raise AssertionError("pagination did not terminate")

    assert len(seen_ids) == len(set(seen_ids)) == 37  # no duplicates, no gaps


def test_authorization_holds_across_paginated_boundaries(monkeypatch):
    """Interleave actor-a's and actor-b's rows in sort order and paginate one
    row at a time (forcing a page boundary between every pair) — actor-a's
    reader must see exactly its own rows, never a neighbor's, at any boundary."""
    rows = _rows()
    grant_a = _grant_digest("actor-a")
    grant_b = _grant_digest("actor-b")
    rows["mcp_tools"] = [
        {
            **rows["mcp_tools"][0],
            "id": "tool_aa",
            "name": "aa-tool",
            "discovery_grant_digest": grant_a,
        },
        {
            **rows["mcp_tools"][1],
            "id": "tool_ab",
            "name": "ab-tool",
            "discovery_grant_digest": grant_b,
        },
        {
            **rows["mcp_tools"][0],
            "id": "tool_ac",
            "name": "ac-tool",
            "discovery_grant_digest": grant_a,
        },
        {
            **rows["mcp_tools"][1],
            "id": "tool_ad",
            "name": "ad-tool",
            "discovery_grant_digest": grant_b,
        },
    ]
    engine = _FakeEngine(rows)
    client = _authority_app(monkeypatch, engine=engine, actor_id="actor-a")

    seen: list[str] = []
    cursor = None
    for _ in range(10):
        params: dict[str, Any] = {"limit": 1}
        if cursor:
            params["cursor"] = cursor
        response = client.get("/api/registry/tools", params=params)
        assert response.status_code == 200, response.text
        body = response.json()
        seen.extend(item["name"] for item in body["items"])
        cursor = body["next_cursor"]
        if not cursor:
            break

    assert seen == ["aa-tool", "ac-tool"]
    assert "ab-tool" not in seen
    assert "ad-tool" not in seen


def test_filter_is_literal_and_bounds_are_enforced(monkeypatch):
    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)

    injection_like = client.get("/api/registry/servers", params={"q": "' OR 1=1 --"})
    assert injection_like.status_code == 200
    assert injection_like.json()["items"] == []
    assert client.get("/api/registry/servers", params={"limit": 0}).status_code == 422
    assert client.get("/api/registry/servers", params={"limit": 101}).status_code == 422


def test_catalog_failure_is_explicit_unavailable(monkeypatch):
    engine = _FakeEngine(_rows())
    engine.graph_compute.fail = True
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "catalog_unavailable",
    }


def test_get_kind_catalog_failure_is_explicit_unavailable(monkeypatch):
    """`_get_kind` (the single-item handler) fails closed identically to
    `_list_kind`: bare `JSONResponse(..., status_code=503)` served over the
    declared `RegistryItemEnvelope[Any]` response_model. This pins the exact
    wire behavior (status code + JSON body) of `_get_kind`'s unavailable
    path -- lines 946/952/963 -- as a regression guard for the return-type
    annotation widening to `RegistryItemEnvelope[Any] | JSONResponse`
    (type-hygiene only; FastAPI already honors a returned Response object
    over response_model, so this body/status must be byte-identical to
    before that annotation change)."""
    engine = _FakeEngine(_rows())
    engine.graph_compute.fail = True
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers/mcp_server_alpha")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "catalog_unavailable",
    }


@pytest.mark.parametrize("malformation", ["missing_tenant", "wrong_tenant"])
def test_malformed_catalog_scope_is_explicitly_unavailable(monkeypatch, malformation):
    engine = _FakeEngine(_rows())
    engine.graph_compute.return_raw = True
    row = engine.graph_compute.rows["mcp_servers"][0]
    if malformation == "missing_tenant":
        row.pop("tenant_id")
    else:
        row["tenant_id"] = "tenant-b"
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "catalog_unavailable",
    }


def test_legacy_unbound_derived_row_is_not_relabelled_public(monkeypatch):
    rows = {
        "mcp_tools": [
            {
                "id": "legacy-tool",
                "tenant_id": "tenant-a",
                "server_id": "mcp_server_alpha",
                "server_name": "alpha",
                "name": "legacy",
                "description": "",
                "schema_digest": "legacy",
                "tool_mode": "verbose",
                "enabled": True,
                "discovery_principal": "",
                # The legacy row intentionally has no grant digest.
            }
        ]
    }
    engine = _FakeEngine(rows)
    engine.graph_compute.return_raw = True
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/tools")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "catalog_unavailable",
    }


@pytest.mark.parametrize(
    ("authority_kind", "principal", "grant"),
    [
        ("unknown", "", ""),
        ("tenant_local", "spoofed", ""),
        ("tenant_local", "", "spoofed-grant"),
    ],
)
def test_malformed_local_authority_is_unavailable(
    monkeypatch, authority_kind, principal, grant
):
    rows = _rows_with_local()
    rows["mcp_tools"] = [
        {
            **rows["mcp_tools"][1],
            "discovery_authority_kind": authority_kind,
            "discovery_principal": principal,
            "discovery_grant_digest": grant,
        }
    ]
    engine = _FakeEngine(rows)
    engine.graph_compute.return_raw = True
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/tools")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "catalog_unavailable",
    }


def test_invalid_catalog_model_field_is_explicitly_unavailable(monkeypatch):
    engine = _FakeEngine(_rows())
    engine.graph_compute.rows["mcp_servers"][0]["enabled"] = "not-a-boolean"
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "catalog_unavailable",
    }


def test_a_huge_table_is_paged_not_materialized(monkeypatch):
    """The defect this branch fixes: a table far past `_MAX_CATALOG_ROWS`
    (10_000) used to force-fetch `_MAX_CATALOG_ROWS + 1` rows on every
    request and fail closed. With LIMIT/COUNT pushdown, the same table is
    served as an ordinary small page — the row count no longer determines
    whether the read is safe."""
    source = _rows()["mcp_servers"][0]
    rows = [{**source, "id": f"mcp_server_{index:05d}"} for index in range(10_001)]
    engine = _FakeEngine({"mcp_servers": rows})
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers", params={"limit": 5})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 10_001
    assert len(body["items"]) == 5
    assert body["next_cursor"]
    page_statement = engine.graph_compute.statements[-1]
    assert "LIMIT 6" in page_statement  # limit + 1, never the table size


def test_engine_ignoring_limit_is_explicitly_unavailable(monkeypatch):
    """Equivalent guard for the new design: if the engine ever returns more
    rows than the LIMIT it was given (a broken/misconfigured projection),
    the route fails closed instead of silently serving an oversized page."""
    source = _rows()["mcp_servers"][0]
    rows = [{**source, "id": f"mcp_server_{index}"} for index in range(5)]
    engine = _FakeEngine({"mcp_servers": rows})
    engine.graph_compute.ignore_limit = True
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers", params={"limit": 1})

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "catalog_unavailable",
    }


def test_missing_item_is_privacy_safe_and_no_write_verbs_are_mounted(monkeypatch):
    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers/does-not-exist")
    assert response.status_code == 404
    assert response.json()["detail"] == "registry item not found"

    routes = _registry_routes(client.app.app)
    assert routes
    assert all(methods == {"GET"} for _, methods in routes)


def test_missing_graph_session_is_denied(monkeypatch):
    monkeypatch.setattr(
        registry_api, "_get_catalog_engine", lambda: _FakeEngine(_rows())
    )
    app = FastAPI()
    registry_api.register_registry_routes(app, prefix="/api")

    with suspend_session():
        response = TestClient(app).get("/api/registry/servers")

    assert response.status_code == 403
    assert response.json()["detail"] == "registry access denied"


# --- Event-loop offload (DEFECT A) ---
#
# `_list_kind` used to call `_authorized_count`/`_authorized_page` as plain
# synchronous calls with no `run_in_executor`/`to_thread` offload anywhere,
# so a slow engine RPC stalled the *entire* ASGI process, not just the one
# request, and had no deadline at all. These tests exercise the fix
# (`registry_api._offload_catalog_call`, `asyncio.wait_for(asyncio.to_thread(...))`)
# directly against a fake engine whose `sql_exec` blocks synchronously.


def _direct_authority(actor_id: str = "actor-a", tenant_id: str = "tenant-a"):
    """Build a bare actor/session pair for a direct (non-TestClient) coroutine
    call, mirroring `_authority_app`'s middleware setup without the ASGI
    plumbing -- needed so a test can `await` `_list_kind` concurrently with
    another coroutine on the *same* event loop and observe whether the loop
    stayed responsive."""

    actor = ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("registry:read",),
        tenant_id=tenant_id,
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=tenant_id,
        scopes=frozenset({"kg:read"}),
        graph=tenant_id,
        policy_version="test",
        audience="test",
    )
    return actor, session


async def test_list_kind_offloads_the_blocking_engine_call_off_the_event_loop(
    monkeypatch,
):
    """A concurrently scheduled coroutine must keep making progress *while*
    `_list_kind`'s catalog SQL call is in flight, proving the blocking engine
    RPC no longer runs inline on the event loop.

    Before the fix, `_authorized_count`/`_authorized_page` ran synchronously
    with no `await` in between, so once the event loop picked `_list_kind` to
    run it would not yield back until both catalog calls finished -- the
    heartbeat coroutine below would then show one large gap spanning the
    whole blocking window instead of steady small gaps throughout it.
    """

    engine = _FakeEngine(_rows())
    real_sql_exec = engine.graph_compute.sql_exec

    def slow_sql_exec(statement: str):
        time.sleep(0.3)
        return real_sql_exec(statement)

    engine.graph_compute.sql_exec = slow_sql_exec

    actor, session = _direct_authority()
    monkeypatch.setattr(registry_api, "_get_catalog_engine", lambda: engine)
    monkeypatch.setattr(
        registry_api,
        "_resolve_current_discovery_grants",
        lambda actor: (_grant_digest(actor.actor_id, actor.tenant_id),),
    )

    request = Request({"type": "http", "query_string": b"", "headers": []})
    ticks: list[float] = []

    async def heartbeat() -> None:
        start = time.monotonic()
        for _ in range(60):
            await asyncio.sleep(0.02)
            ticks.append(time.monotonic() - start)

    with use_actor(actor), use_session(session):
        result, _ = await asyncio.gather(
            registry_api._list_kind(
                request, kind="servers", model=registry_api.RegistryServer
            ),
            heartbeat(),
        )

    assert isinstance(result, registry_api.RegistryPage)
    assert result.status == "ok"
    # `_list_kind` makes two sequential offloaded calls (count, then page) of
    # 0.3s each -- ~0.6s of blocking work in total. The heartbeat must have
    # kept ticking at its own ~0.02s cadence throughout that window: no gap
    # between consecutive ticks anywhere near the 0.3s a single blocked call
    # would produce if it still ran inline on the loop.
    assert len(ticks) >= 20, ticks
    gaps = [b - a for a, b in zip(ticks, ticks[1:], strict=False)]
    assert max(gaps) < 0.2, gaps


def test_list_kind_deadline_breach_fails_closed_and_bounded(monkeypatch):
    """A genuinely hung engine call must fail this one request quickly and
    cleanly -- not hang the request (or the ASGI process) indefinitely.
    Regression guard for DEFECT A's "no deadline bound at all" finding."""

    engine = _FakeEngine(_rows())
    real_sql_exec = engine.graph_compute.sql_exec

    def hung_sql_exec(statement: str):
        time.sleep(2.0)
        return real_sql_exec(statement)

    engine.graph_compute.sql_exec = hung_sql_exec
    monkeypatch.setattr(registry_api, "_CATALOG_READ_TIMEOUT_S", 0.2)
    client = _authority_app(monkeypatch, engine=engine)

    # `time.sleep(2.0)` runs in a worker thread that `wait_for` abandons
    # (rather than kills) once the deadline fires -- the OS thread keeps
    # running in the background regardless, it just no longer holds up the
    # request. Used outside a `with` block, Starlette's TestClient tears its
    # portal down (and drains that stray executor thread) *after every
    # single call*, which would fold that background 2s into this test's
    # wall-clock measurement and defeat the point of the assertion. Keeping
    # the portal open across the call measures what actually matters: how
    # long the request itself took to come back.
    with client:
        start = time.monotonic()
        response = client.get("/api/registry/servers")
        elapsed = time.monotonic() - start

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "catalog_unavailable",
    }
    # Bounded by the (patched) deadline, not by the 2s hang.
    assert elapsed < 1.5, elapsed


def test_success_path_response_shape_is_unchanged_by_the_offload(monkeypatch):
    """Pin the exact 200 envelope for a known page so the offload change is
    provably byte-identical on the success path: same rows, same shape."""

    engine = _FakeEngine(_rows())
    client = _authority_app(monkeypatch, engine=engine)

    response = client.get("/api/registry/servers", params={"q": "alpha"})

    assert response.status_code == 200, response.text
    assert response.json() == {
        "status": "ok",
        "kind": "servers",
        "items": [
            {
                "id": "mcp_server_alpha",
                "name": "alpha",
                "transport": "http",
                "url": "",
                "enabled": True,
            }
        ],
        "count": 1,
        "next_cursor": None,
    }
