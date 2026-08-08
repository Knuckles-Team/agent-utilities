"""Fleet skills become runnable only when genuinely dispatchable.

CONCEPT:AU-ECO.mcp.cross-process-skill-harvest — the promotion half of the
cross-process harvest: a skill body pulled off a fleet MCP child is promoted to
a runnable ``CallableResource`` through the existing ``ingest_runnable_skill``
primitive, and a skill that fails ANY precondition is not promoted at all but
records WHICH precondition it failed.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.ingestion.fleet_skill_harvest import (
    RUNNABLE_PRECONDITIONS,
    evaluate_skill_dispatchable,
    promote_harvested_skills,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor


class _Engine:
    """Records typed writes the way the real engine's authority writer does."""

    def __init__(self, existing: dict[str, dict] | None = None) -> None:
        self.nodes: dict[str, dict] = dict(existing or {})
        self.edges: list[tuple[str, str, str]] = []
        self.backend = self

    def _upsert_node(self, node_type: str, node_id: str, properties: dict) -> None:
        self.nodes[node_id] = {"type": node_type, **properties}

    def link_nodes(self, source: str, target: str, relationship: str, **_kw) -> None:
        self.edges.append((source, target, relationship))

    def execute(self, query: str, params: dict | None = None):
        params = params or {}
        node = self.nodes.get(params.get("rid", ""))
        if node is None:
            return []
        return [
            {
                "provider_ref": node.get("provider_ref", ""),
                "mcp_server": node.get("mcp_server", ""),
            }
        ]


@pytest.fixture()
def authority():
    actor = ActorContext(
        actor_id="harvest-test",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id="tenant_test",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant="tenant_test",
        scopes=frozenset({"kg:read", "kg:write"}),
        graph="g",
        policy_version="v1",
        audience="test",
    )
    with use_actor(actor), use_session(session):
        yield


def _entry(name: str, **overrides) -> dict:
    entry = {
        "name": name,
        "uri": f"skill://{name}/SKILL.md",
        "description": f"{name} description",
        "instructions": f"# {name}\n\nDo the thing.",
    }
    entry.update(overrides)
    return entry


def _server(*entries: dict, error: str | None = None, tools: bool = True) -> dict:
    return {
        "error": error,
        "tools": [{"name": "t", "description": "d", "inputSchema": {}}]
        if tools
        else [],
        "skills": list(entries),
    }


def test_dispatchable_skill_is_promoted_to_a_runnable_resource(authority):
    engine = _Engine()
    report = promote_harvested_skills(
        engine, {"servicenow-mcp": _server(_entry("servicenow-incident-management"))}
    )

    assert report["promoted"] == 1
    assert report["blocked"] == 0
    assert report["errors"] == 0
    resource = engine.nodes["resource:skill:servicenow-incident-management"]
    assert resource["type"] == "CallableResource"
    assert resource["resource_type"] == "AGENT_SKILL"
    assert resource["runnable_bound"] is True
    # The runnable resource records WHICH child serves it, so the dispatcher can
    # bind the toolset the skill's instructions direct it to call.
    assert resource["mcp_server"] == "servicenow-mcp"
    assert resource["system_prompt"].strip().startswith("# servicenow-incident")
    assert (
        "mcp_server_servicenow-mcp",
        "resource:skill:servicenow-incident-management",
        "SERVES",
    ) in engine.edges


@pytest.mark.parametrize(
    ("catalog_kwargs", "entry_kwargs", "expected"),
    [
        ({"error": "connection refused"}, {}, "server_reachable"),
        (
            {},
            {"instructions": "", "harvest_error": "server served an empty skill body"},
            "skill_body_served",
        ),
        ({"tools": False}, {}, "server_exposes_tools"),
    ],
)
def test_a_skill_that_is_not_dispatchable_fails_closed_with_a_named_precondition(
    authority, catalog_kwargs, entry_kwargs, expected
):
    """Present-but-not-dispatchable must never become runnable."""
    engine = _Engine()
    report = promote_harvested_skills(
        engine,
        {"child-mcp": _server(_entry("some-skill", **entry_kwargs), **catalog_kwargs)},
    )

    assert report["promoted"] == 0
    assert report["blocked"] == 1
    assert expected in report["blocked_detail"]["child-mcp/some-skill"]
    # Fail CLOSED: no runnable resource is written for a blocked skill.
    assert "resource:skill:some-skill" not in engine.nodes
    assert expected in RUNNABLE_PRECONDITIONS


def test_a_locally_installed_skill_is_never_overwritten_by_a_fleet_child(authority):
    """A co-installed provider outranks whatever a child chooses to serve."""
    engine = _Engine(
        {
            "resource:skill:graph-query-and-explanation": {
                "type": "CallableResource",
                "provider_ref": "provider://agent_utilities",
                "mcp_server": "",
                "system_prompt": "the locally installed body",
            }
        }
    )

    report = promote_harvested_skills(
        engine, {"rogue-mcp": _server(_entry("graph-query-and-explanation"))}
    )

    assert report["promoted"] == 0
    assert (
        report["blocked_detail"]["rogue-mcp/graph-query-and-explanation"]
        == "no_local_provider_conflict"
    )
    assert (
        engine.nodes["resource:skill:graph-query-and-explanation"]["system_prompt"]
        == "the locally installed body"
    )


def test_re_harvesting_the_same_child_is_a_refresh_not_a_conflict(authority):
    """The SAME child may update a skill it already owns."""
    engine = _Engine()
    catalog = {"servicenow-mcp": _server(_entry("servicenow-cmdb"))}
    assert promote_harvested_skills(engine, catalog)["promoted"] == 1

    catalog["servicenow-mcp"]["skills"][0]["instructions"] = "# updated body"
    report = promote_harvested_skills(engine, catalog)

    assert report["promoted"] == 1
    assert report["blocked"] == 0
    assert (
        engine.nodes["resource:skill:servicenow-cmdb"]["system_prompt"]
        == "# updated body"
    )


def test_preconditions_are_reported_in_order(authority):
    """The FIRST unmet precondition is the one reported."""
    engine = _Engine()
    # Unreachable AND body-less AND tool-less: reachability is reported, because
    # a child that never answered cannot also be said to have served no body.
    assert (
        evaluate_skill_dispatchable(
            engine,
            "child-mcp",
            {"error": "boom", "tools": [], "skills": []},
            _entry("s", instructions=""),
            resource_id="resource:skill:s",
        )
        == "server_reachable"
    )
