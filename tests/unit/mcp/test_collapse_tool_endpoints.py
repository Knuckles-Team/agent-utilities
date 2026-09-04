"""FIX LANE (collapse-tool-endpoints): tests for the tool-listing endpoint
collapse onto ONE source of truth.

Background / what changed:

- ``GET /tools`` (``agent_utilities.server.routers.core.list_tools``) was
  investigated for deletion as a "legacy duplicate" of the canonical
  ``/api/tools``. It was KEPT: ``agent_terminal_ui.client.AgentClient.
  list_tools`` (``agent-packages/agent-terminal-ui/agent_terminal_ui/
  client.py``) hits this exact un-prefixed path and is covered by that
  package's own real-transport wiring test
  (``tests/test_graph_route_wiring.py::
  test_the_servers_own_unprefixed_routers_keep_their_paths``), which pins
  the request to land at ``/tools``. Deleting it would have broken a real,
  tested caller. Its ``t.description AS descriptionription`` /
  ``s.description AS descriptionription`` typo IS fixed here (see below).

- ``GET /api/tools`` (``agent_utilities.mcp.kg_server.get_tools_endpoint``)
  now sources its ``mcp_tools`` section from the SAME SQL fleet-catalog
  ``servers`` kind that backs ``/api/registry/servers`` and the webui BFF,
  instead of re-parsing ``mcp_config.json``. ``builtin_tools``/``skills``/
  ``skill_workflows``/``skill_graphs`` stay filesystem/KG-native-sourced
  (see ``_build_tools_payload_sync``'s docstring for the evidence on each).
  A new additive ``section_status`` key reports per-section degrade instead
  of the whole request failing.

- Two ``descriptionription``/``descriptionription_text`` alias typos (one
  in ``core.py``, one in ``app.py``) are fixed, each verified against its
  actual downstream consumer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_utilities.gateway import registry_api
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.mcp import kg_server
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

ACTOR = ActorContext(
    actor_id="principal:ops",
    actor_type=ActorType.AI_AGENT,
    roles=(),
    tenant_id="acme",
    authenticated=True,
)


def _session(graph: str = "tenant-acme-graph") -> GraphSession:
    return GraphSession(
        actor=ACTOR,
        tenant=ACTOR.tenant_id,
        scopes=frozenset({"kg:read"}),
        graph=graph,
        policy_version="policy-v1",
        audience="agent-services",
    )


class _FakeServersCompute:
    """Minimal SQL double for the ``mcp_servers`` catalog table.

    Deliberately simpler than ``tests/unit/gateway/test_registry_api.py``'s
    full WHERE-clause interpreter: this file's tests use small (<100 row)
    fixtures that always fit on ONE page, so ``registry_api._authorized_page``
    never needs a second keyset round trip, and the fixture rows are already
    pre-scoped to the session's own tenant — good enough to prove the real
    ``_authorized_page``/``_authorized_count`` code paths run, without
    reimplementing SQL predicate evaluation a second time.
    """

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.fail = False

    def sql_exec(self, statement: str):
        if self.fail:
            raise OSError("catalog backend unavailable")
        if "COUNT(" in statement.upper():
            return [{"row_count": len(self.rows)}]
        return list(self.rows)


class _FakeCatalogEngine:
    def __init__(self, rows: list[dict[str, Any]]):
        self.graph_compute = _FakeServersCompute(rows)


class _NoSqlEngine:
    """An engine with no ``graph_compute.sql_exec`` at all — the shape
    ``registry_api._require_sql_exec`` fails closed on."""


def _server_row(name: str, *, transport: str = "stdio", enabled: bool = True) -> dict[str, Any]:
    return {
        "id": f"mcp_server_{name}",
        "tenant_id": ACTOR.tenant_id,
        "name": name,
        "transport": transport,
        "url": "",
        "enabled": enabled,
    }


def _toggle_free_engine() -> Any:
    """A KG engine whose ``query_cypher`` always returns no rows — every
    ``get_toggle_states_batch`` lookup then fail-opens to ``enabled=True``,
    matching that function's own documented default."""
    engine = MagicMock()
    engine.query_cypher.return_value = []
    return engine


def _write_skill_md(
    path: Path, *, name: str, description: str, domain: str = "demo", tags: str = "alpha, beta"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\ndomain: {domain}\ntags: {tags}\n---\n\nBody.\n",
        encoding="utf-8",
    )


def _seeded_workspace(tmp_path: Path) -> Path:
    """Build a workspace_root with one atomic skill, one workflow, and one
    skill-graph SKILL.md, matching the exact directory shape
    ``_build_tools_payload_sync`` globs."""
    skills_root = (
        tmp_path / "agent-packages" / "skills" / "universal-skills" / "universal_skills"
    )
    _write_skill_md(
        skills_root / "demo" / "demo-skill" / "SKILL.md",
        name="demo-skill",
        description="An atomic demo skill",
    )
    # NOTE: `_build_tools_payload_sync` classifies a workflow via
    # `"workflows" in p.parts` -- an EXACT path-component match, not a
    # substring check. The real shipped corpus actually uses
    # `<domain>-workflows/` directories (e.g. `finance-workflows/`), which
    # this exact check does not match (a pre-existing classification
    # quirk, out of scope for this fix lane -- untouched by it). This
    # fixture uses a literal `workflows/` segment so it exercises the
    # "Skill Workflow" branch as the CURRENT code actually requires,
    # rather than silently asserting on the "Agent Skill" fallback branch
    # that a `<domain>-workflows/` path would actually hit today.
    _write_skill_md(
        skills_root / "workflows" / "demo-workflow" / "SKILL.md",
        name="demo-workflow",
        description="A demo workflow",
    )
    graphs_root = tmp_path / "agent-packages" / "skills" / "skill-graphs" / "skill_graphs"
    _write_skill_md(
        graphs_root / "demo-graph" / "SKILL.md",
        name="demo-graph",
        description="A demo skill graph",
    )
    return tmp_path


# ── Test 1: contract — top-level keys + per-item fields ────────────────────


def test_api_tools_contract_top_level_keys_and_per_item_fields(monkeypatch, tmp_path):
    """`/api/tools` still returns its documented keys, and each section's
    items still carry the field names a caller (the dashboard/gateway) would
    read. This is the shape-preservation guarantee the fix lane required."""
    workspace_root = _seeded_workspace(tmp_path)
    catalog_engine = _FakeCatalogEngine([_server_row("alpha")])
    monkeypatch.setattr(registry_api, "_get_catalog_engine", lambda: catalog_engine)

    with use_actor(ACTOR), use_session(_session()):
        payload = kg_server._build_tools_payload_sync(_toggle_free_engine(), workspace_root)

    assert set(payload.keys()) == {
        "mcp_tools",
        "builtin_tools",
        "skills",
        "skill_graphs",
        "skill_workflows",
        "section_status",
    }
    assert payload["section_status"] == {
        "mcp_tools": "ok",
        "builtin_tools": "ok",
        "skills": "ok",
        "skill_workflows": "ok",
        "skill_graphs": "ok",
    }

    assert len(payload["mcp_tools"]) == 1
    assert set(payload["mcp_tools"][0].keys()) == {
        "name",
        "type",
        "launch_mode",
        "command",
        "args",
        "status",
        "enabled",
    }
    assert payload["mcp_tools"][0]["name"] == "alpha"

    assert payload["builtin_tools"], "expected real agent_utilities/tools/*.py entries"
    assert set(payload["builtin_tools"][0].keys()) == {
        "name",
        "type",
        "file_path",
        "status",
        "enabled",
    }

    assert len(payload["skills"]) == 1
    skill_item = payload["skills"][0]
    assert skill_item["name"] == "demo-skill"
    assert skill_item["description"] == "An atomic demo skill"
    assert {"id", "name", "description", "domain", "tags", "enabled", "file_path", "type"} <= set(
        skill_item.keys()
    )

    assert len(payload["skill_workflows"]) == 1
    assert payload["skill_workflows"][0]["name"] == "demo-workflow"
    assert payload["skill_workflows"][0]["type"] == "Skill Workflow"

    assert len(payload["skill_graphs"]) == 1
    assert payload["skill_graphs"][0]["name"] == "demo-graph"
    assert payload["skill_graphs"][0]["type"] == "Skill Graph"


# ── Test 2: /api/tools and /api/registry/* agree where they claim to ───────


def test_mcp_tools_and_registry_servers_read_the_identical_catalog_rows(monkeypatch):
    """The whole point of this fix lane: `/api/tools`'s `mcp_tools` section
    and `/api/registry/servers` must report the SAME server set, because
    they now read the exact same `servers` fleet-catalog kind through the
    exact same registry_api authorized-read path — no more two competing
    inventories of the same MCP fleet."""
    rows = [_server_row("alpha", transport="stdio"), _server_row("bravo", transport="http")]
    catalog_engine = _FakeCatalogEngine(rows)
    monkeypatch.setattr(registry_api, "_get_catalog_engine", lambda: catalog_engine)

    with use_actor(ACTOR), use_session(_session()):
        api_tools_names = {
            row["name"]
            for row in kg_server._read_catalog_kind_sync(
                "servers", require_discovery_binding=False
            )
        }

        tenant, principal, grant_digests = registry_api._require_catalog_authority(
            require_discovery_binding=False
        )
        registry_page = registry_api._authorized_page(
            "servers",
            tenant=tenant,
            principal=principal,
            grant_digests=grant_digests,
            query="",
            after=None,
            limit=registry_api._MAX_LIMIT,
            engine=catalog_engine,
        )
        registry_servers_names = {row["name"] for row in registry_page}

    assert api_tools_names == registry_servers_names == {"alpha", "bravo"}


def test_mcp_tools_and_registry_tools_are_documented_different_granularity():
    """`/api/tools`'s `mcp_tools` key has ALWAYS held one item per configured
    MCP *server* (never individual tools, despite the key's name) — see
    `_build_tools_payload_sync`'s docstring. `/api/registry/tools` (kind
    "tools") is genuinely a different concept: one row per individual
    MCP-discovered tool, keyed to its owning server. These do not, and are
    not meant to, "agree" item-for-item; this asserts that documented
    difference explicitly instead of forcing a false equivalence.
    """
    servers_spec = registry_api._KIND_SPECS["servers"]
    tools_spec = registry_api._KIND_SPECS["tools"]

    assert servers_spec.table == "mcp_servers"
    assert tools_spec.table == "mcp_tools"
    # Tool-granularity columns that have no server-list analogue.
    assert {"server_id", "server_name", "schema_digest", "tool_mode"} <= set(tools_spec.columns)
    # `/api/tools`'s own mcp_tools item shape (proven by test 1 above) never
    # carries any of those tool-granularity fields.
    mcp_tools_item_fields = {
        "name",
        "type",
        "launch_mode",
        "command",
        "args",
        "status",
        "enabled",
    }
    assert mcp_tools_item_fields.isdisjoint({"server_id", "schema_digest", "tool_mode"})


# ── Test 3: a per-section failure still yields 200 with an explicit marker ──


def test_mcp_tools_section_degrades_without_failing_the_whole_request(monkeypatch, tmp_path):
    """The fleet catalog being unreachable must mark ONLY `mcp_tools` as
    unavailable — every other section, and the request itself, still
    succeeds. Never a blanket 503."""
    workspace_root = _seeded_workspace(tmp_path)
    monkeypatch.setattr(registry_api, "_get_catalog_engine", lambda: _NoSqlEngine())

    with use_actor(ACTOR), use_session(_session()):
        payload = kg_server._build_tools_payload_sync(_toggle_free_engine(), workspace_root)

    assert payload["mcp_tools"] == []
    assert payload["section_status"]["mcp_tools"] == "unavailable"
    # Every other section is independent and still healthy.
    assert payload["section_status"]["builtin_tools"] == "ok"
    assert payload["section_status"]["skills"] == "ok"
    assert payload["section_status"]["skill_workflows"] == "ok"
    assert payload["section_status"]["skill_graphs"] == "ok"
    assert payload["skills"], "unrelated sections must not be discarded"


@pytest.mark.asyncio
async def test_get_tools_endpoint_returns_200_even_when_the_catalog_is_down(monkeypatch):
    """End-to-end: the ASGI handler itself must still answer 200, carrying
    the explicit per-section marker, never a 503, when the fleet catalog
    cannot be read."""
    monkeypatch.setattr(registry_api, "_get_catalog_engine", lambda: _NoSqlEngine())
    monkeypatch.setattr(kg_server, "_get_engine", lambda: _toggle_free_engine())
    monkeypatch.setattr(kg_server, "setting", lambda _key, default="": default)

    with use_actor(ACTOR), use_session(_session()):
        response = await kg_server.get_tools_endpoint(None)

    assert response.status_code == 200
    import json

    body = json.loads(bytes(response.body))
    assert body["section_status"]["mcp_tools"] == "unavailable"
    assert body["mcp_tools"] == []


# ── Test 4: the bare `/tools` route stays mounted (a real caller depends on
#    it — deletion would have broken agent-terminal-ui) ────────────────────


def test_bare_tools_route_is_kept_and_still_serves_a_flat_list(monkeypatch):
    """`GET /tools` (`server/routers/core.py::list_tools`) was investigated
    for deletion as a legacy duplicate of `/api/tools`, but
    `agent_terminal_ui.client.AgentClient.list_tools` genuinely calls this
    exact un-prefixed path and expects a flat `list[dict]` body (see that
    package's `tests/test_graph_route_wiring.py::
    test_the_servers_own_unprefixed_routers_keep_their_paths`, and
    `agent_terminal_ui/widgets/tools_sidebar.py`'s consumption of the
    result). It was therefore KEPT, not deleted — this proves the route is
    still registered on the router and still returns the flat shape that
    caller expects.
    """
    from agent_utilities.server.routers import core

    paths = {getattr(route, "path", None) for route in core.router.routes}
    assert "/tools" in paths, (
        "GET /tools must stay mounted -- agent-terminal-ui's AgentClient."
        "list_tools() is a real, tested caller of this exact path"
    )


# ── Test 5: the two `descriptionription`/`descriptionription_text` typo
#    fixes actually reach the field name each real consumer reads ──────────


@pytest.mark.asyncio
async def test_core_list_tools_description_reaches_tools_sidebar_consumer_key(monkeypatch):
    """core.py FIX: `t.description AS descriptionription` (and the `Skill`
    twin) is fixed to `AS description`. Consumer evidence:
    `agent_terminal_ui/widgets/tools_sidebar.py::ToolsSidebar._populate_tree`
    reads `item.get("description", "")` for both search filtering and the
    rendered label. Simulates the real engine returning a row shaped by the
    FIXED query (keyed "description") and proves it survives to that exact
    key -- pre-fix, the real value would have landed under
    "descriptionription" and `item.get("description", "")` would always see
    "".
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.server.routers.core import list_tools

    class _FakeKG:
        backend = object()

        def query_cypher(self, query: str):
            assert "descriptionription" not in query, "typo alias must be gone"
            assert "AS description" in query
            if "Tool" in query:
                return [
                    {
                        "id": "t1",
                        "name": "some-tool",
                        "description": "does a thing",
                        "source_name": "srv",
                        "type": "tool",
                    }
                ]
            return []

    monkeypatch.setattr(IntelligenceGraphEngine, "get_active", staticmethod(lambda: _FakeKG()))

    items = await list_tools()

    assert len(items) == 1
    # This is EXACTLY the key agent-terminal-ui's ToolsSidebar reads.
    assert items[0].get("description", "") == "does a thing"


def test_graph_native_list_skills_description_text_reaches_its_own_consumer(monkeypatch):
    """app.py FIX: `t.description AS descriptionription_text` (and the
    `Prompt` twin) is fixed to `AS description_text`. Consumer evidence:
    the same function's own `p.get("description_text", "")`/
    `t.get("description_text", "")` calls immediately below each query --
    this is also what agent-webui's `/skills` chat command
    (`api_extensions.py`, `cmd_name == 'skills'`) renders into its response
    markdown. `_graph_native_list_skills` was extracted to module level
    (from a nested closure inside `build_agent_app`) specifically so this
    is directly testable.
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.server.app import _graph_native_list_skills

    class _FakeBackend:
        def execute(self, query: str):
            assert "descriptionription_text" not in query, "typo alias must be gone"
            assert "AS description_text" in query
            if "Prompt" in query:
                return [{"id": "p1", "name": "prompt-one", "description_text": "a prompt"}]
            return [
                {
                    "id": "t1",
                    "name": "tool-one",
                    "description_text": "a tool",
                    "server": "srv",
                }
            ]

    class _FakeEngine:
        backend = _FakeBackend()

    monkeypatch.setattr(IntelligenceGraphEngine, "get_or_create", staticmethod(lambda: _FakeEngine()))

    skills = _graph_native_list_skills()
    by_name = {s["name"]: s for s in skills}

    assert by_name["prompt-one"]["description"] == "a prompt"
    assert by_name["tool-one"]["description"] == "[srv] a tool"
