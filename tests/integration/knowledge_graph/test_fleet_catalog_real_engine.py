"""NE-055 -- fleet-catalog SQL against the real epistemic-graph engine.

The unit coverage for ``fleet_catalog_tables`` uses ``_FakeGraphCompute`` to
exercise the application CAS policy.  This module deliberately uses the
repository's ``engine_graph`` fixture instead: it provisions the exact
epistemic-graph server artifact and sends the production DDL/DML through the
authenticated ``GraphComputeEngine.sql_exec`` path.

The test is intentionally narrow.  It verifies the six production tables and
their declared columns, round-trips one complete catalog, and checks replay,
stale-write rejection, and a newer CAS update from durable rows.  If the
approved real-engine artifact is unavailable, ``engine_graph`` skips; there is
no emulator or in-process fallback in this proof.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import pytest

from agent_utilities.knowledge_graph.core import fleet_catalog_tables as fct
from agent_utilities.knowledge_graph.core.session import current_session

pytestmark = [
    pytest.mark.integration,
    pytest.mark.engine,
    pytest.mark.concept("AU-KG.ingest.fleet-catalog-relational-tables"),
]


def _declared_columns(ddl: str) -> set[str]:
    """Extract the production declaration's column names independently."""

    body = ddl[ddl.index("(") + 1 : ddl.rfind(")")]
    columns: set[str] = set()
    for line in body.splitlines():
        token = line.strip().rstrip(",")
        if token:
            columns.add(token.split(None, 1)[0])
    return columns


def _rows(gc: Any, statement: str) -> list[dict[str, Any]]:
    """Execute a real SQL read and reject non-row/fallback result shapes."""

    result = gc.sql_exec(statement)
    assert isinstance(result, list), (
        "the real SQL surface must return row lists for catalog reads; "
        f"got {type(result).__name__}"
    )
    rows: list[dict[str, Any]] = []
    for row in result:
        assert isinstance(row, Mapping), (
            "the real SQL surface must return mapping rows; "
            f"got {type(row).__name__}"
        )
        rows.append(dict(row))
    return rows


def _column_names(gc: Any, table: str) -> set[str]:
    """Read one table's live schema from epistemic-graph's catalog view."""

    statement = (
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = "
        f"{fct._sql_literal(table)}"
    )
    return {str(row["column_name"]) for row in _rows(gc, statement)}


def _row(gc: Any, table: str, tenant_id: str, row_id: str) -> dict[str, Any]:
    rows = _rows(
        gc,
        f"SELECT * FROM {fct._safe_ident(table)} WHERE tenant_id = "
        f"{fct._sql_literal(tenant_id)} AND id IN ({fct._sql_literal(row_id)})",
    )
    assert len(rows) == 1, f"expected one durable {table}.{row_id}, got {rows!r}"
    return rows[0]


def _catalog(server_name: str, *, skill_description: str = "first version") -> dict:
    return {
        server_name: {
            "error": None,
            "tools": [
                {
                    "name": "catalog-tool",
                    "description": "A real catalog tool",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                }
            ],
            "skills": [
                {
                    "name": "catalog-skill",
                    "description": skill_description,
                    "uri": "skill://ne055/catalog-skill",
                }
            ],
            "prompts": [
                {
                    "name": "catalog-prompt",
                    "description": "A real catalog prompt",
                    "uri": "prompt://ne055/catalog-prompt",
                }
            ],
        }
    }


@pytest.fixture()
def fleet_catalog_engine(engine_graph: Any) -> Any:
    """Bind the production catalog writer to the live graph SQL authority."""

    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    return IntelligenceGraphEngine(
        backend=EpistemicGraphBackend(graph_name=engine_graph.graph_name),
        defer_background_start=True,
    )


def test_fleet_catalog_ddl_dml_and_cas_round_trip_on_real_engine(
    fleet_catalog_engine: Any,
) -> None:
    """Exercise production fleet-catalog SQL against durable epistemic-graph."""

    session = current_session()
    assert session is not None, "engine_graph must provide a verified GraphSession"
    tenant_id = str(session.tenant)
    binding = fct.TenantLocalDiscoveryBinding(tenant_id=tenant_id)
    gc = fleet_catalog_engine.graph_compute

    server_name = f"ne055-{uuid4().hex[:12]}"
    catalog = _catalog(server_name)
    configs = {
        server_name: {
            "url": "https://catalog.example.invalid/mcp",
            "disabled": False,
        }
    }
    bindings = {server_name: binding}
    attempt = "ne055-real-attempt-1"
    revision = 100

    first = fct.write_fleet_catalog(
        fleet_catalog_engine,
        catalog,
        configs=configs,
        discovery_bindings=bindings,
        revision=revision,
        idempotency_key=attempt,
    )
    assert first["status"] == "ok"
    assert first["servers_written"] == 1
    assert first["discovery_written"] == 1
    assert first["tools_written"] == 1
    assert first["prompts_written"] == 1
    assert first["resources_written"] == 2
    assert first["skills_written"] == 1
    assert first["discovery_status"] == "bound"

    # The generated DDL is the contract under test.  Read every declaration
    # back through information_schema rather than trusting the fake emulator
    # or merely asserting that CREATE returned a status object.
    for table, ddl in fct._DDL.items():
        declared = _declared_columns(ddl)
        actual = _column_names(gc, table)
        assert declared <= actual, (
            f"real epistemic-graph schema for {table} is missing declared "
            f"columns: {sorted(declared - actual)}"
        )

    server_id = f"mcp_server_{server_name}"
    discovery_id = f"disc_{server_id}_{attempt[:24]}"
    tool_id = fct._bound_row_id(f"tool_{server_name}_catalog-tool", "")
    skill_id = fct._bound_row_id(f"skill_{server_name}_catalog-skill", "")
    prompt_id = fct._bound_row_id(
        f"prompt_{server_name}__catalog-prompt", ""
    )
    resource_skill_id = fct._bound_row_id(
        f"resource_{server_name}_skill_catalog-skill", ""
    )
    resource_prompt_id = fct._bound_row_id(
        f"resource_{server_name}_prompt_catalog-prompt", ""
    )

    # Read back one row from every table.  These are real SQL rows emitted by
    # the production writer, not an in-memory table owned by this test.
    assert (
        _row(gc, fct.TABLE_MCP_SERVERS, tenant_id, server_id)["revision"]
        == revision
    )
    assert _row(
        gc, fct.TABLE_MCP_SERVER_DISCOVERY, tenant_id, discovery_id
    )["reachable"] is True
    assert (
        _row(gc, fct.TABLE_MCP_TOOLS, tenant_id, tool_id)["name"]
        == "catalog-tool"
    )
    assert (
        _row(gc, fct.TABLE_MCP_PROMPTS, tenant_id, prompt_id)["name"]
        == "catalog-prompt"
    )
    assert _row(
        gc, fct.TABLE_MCP_RESOURCES, tenant_id, resource_skill_id
    )["resource_kind"] == "skill"
    assert _row(
        gc, fct.TABLE_MCP_RESOURCES, tenant_id, resource_prompt_id
    )["resource_kind"] == "prompt"
    assert _row(gc, fct.TABLE_SKILLS, tenant_id, skill_id)["revision"] == revision

    # A retry may carry a newly computed revision, but the same explicit key
    # identifies the same logical write.  Every existing row must be a durable
    # no-op and the original revision must remain visible after the retry.
    replay = fct.write_fleet_catalog(
        fleet_catalog_engine,
        catalog,
        configs=configs,
        discovery_bindings=bindings,
        revision=revision + 1,
        idempotency_key=attempt,
    )
    assert replay["servers_written"] == 0
    assert replay["cas"][fct.TABLE_MCP_SERVERS] == {
        "written": 0,
        "rejected_stale": 0,
        "noop_replay": 1,
    }
    assert replay["cas"][fct.TABLE_MCP_SERVER_DISCOVERY]["noop_replay"] == 1
    assert replay["cas"][fct.TABLE_MCP_TOOLS]["noop_replay"] == 1
    assert replay["cas"][fct.TABLE_MCP_PROMPTS]["noop_replay"] == 1
    assert replay["cas"][fct.TABLE_MCP_RESOURCES]["noop_replay"] == 2
    assert replay["cas"][fct.TABLE_SKILLS]["noop_replay"] == 1
    assert (
        _row(gc, fct.TABLE_MCP_SERVERS, tenant_id, server_id)["revision"]
        == revision
    )

    # Exercise the same production CAS seam with an existing skill row: a
    # stale revision cannot clobber the durable value, while a newer revision
    # and a new key can update it through the real engine's UPDATE path.
    stale = fct.write_skill_row(
        fleet_catalog_engine,
        skill_id=f"skill_{server_name}_catalog-skill",
        name="catalog-skill",
        description="stale write must be rejected",
        uri="skill://ne055/catalog-skill",
        provider=f"mcp:{server_name}",
        mcp_server=server_name,
        skill_type="mcp_skill",
        revision=revision - 1,
        idempotency_key="ne055-stale",
        discovery_binding=binding,
    )
    assert stale is False
    stale_row = _row(gc, fct.TABLE_SKILLS, tenant_id, skill_id)
    assert stale_row["description"] == "first version"
    assert stale_row["revision"] == revision

    updated = fct.write_skill_row(
        fleet_catalog_engine,
        skill_id=f"skill_{server_name}_catalog-skill",
        name="catalog-skill",
        description="new durable version",
        uri="skill://ne055/catalog-skill",
        provider=f"mcp:{server_name}",
        mcp_server=server_name,
        skill_type="mcp_skill",
        revision=revision + 1,
        idempotency_key="ne055-update",
        discovery_binding=binding,
    )
    assert updated is True
    updated_row = _row(gc, fct.TABLE_SKILLS, tenant_id, skill_id)
    assert updated_row["description"] == "new durable version"
    assert updated_row["revision"] == revision + 1
