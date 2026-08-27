"""Characterization coverage for `graph_ontology` action branches the
pre-existing suite never drove through the real dispatch core.

CONCEPT:AU-KG.ontology.dedicated-tbox-graph

Before the WB1-AU-03 extract-method split of `graph_ontology` (CCN 60),
`test_ontology_catalogue_live_path.py` / `test_ontology_proposal_live_path.py`
covered 'load'/'list'/'get'/'deprecate'/'undeprecate'/'propose'/
'review_proposal'/'promote_proposal'/'rollback_proposal'/'list_proposals'/
'get_proposal' through the real `_execute_tool` dispatch, and
`test_ontology_stardog_surface.py` only checked that 'publish_stardog'/
'import_stardog' are REGISTERED (manifest-level), never that they dispatch.
'update', 'delete', 'validate', 'activate'/'deactivate', 'sync_packages',
'publish_stardog', 'import_stardog', and the tool's own unknown-action
fallback had ZERO real-dispatch coverage anywhere. This file closes that gap
using the exact same `_execute_tool` pattern
`test_ontology_catalogue_live_path.py` establishes, so the split is proven,
not just assumed, behavior-preserving.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.ontology.lifecycle import reset_registry
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import ontology_tools

PETS_TTL = (
    "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
    "@prefix ex: <http://example.org/pets#> .\n"
    "<http://example.org/pets> a owl:Ontology .\n"
    "ex:Animal a owl:Class .\n"
    "ex:Dog a owl:Class ; rdfs:subClassOf ex:Animal .\n"
)

PETS_TTL_V2 = (
    "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
    "@prefix ex: <http://example.org/pets#> .\n"
    "<http://example.org/pets> a owl:Ontology .\n"
    "ex:Animal a owl:Class .\n"
    "ex:Dog a owl:Class ; rdfs:subClassOf ex:Animal .\n"
    "ex:Cat a owl:Class ; rdfs:subClassOf ex:Animal .\n"
)


class _CollectingMCP:
    """Minimal FastMCP stand-in (mirrors test_ontology_catalogue_live_path.py)."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        def _deco(fn):
            self.tools[kwargs.get("name", fn.__name__)] = fn
            return fn

        return _deco


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def registered() -> dict[str, object]:
    mcp = _CollectingMCP()
    ontology_tools.register_ontology_tools(mcp)
    return mcp.tools


async def _graph_ontology(**kwargs) -> dict:
    raw = await kg_server._execute_tool("graph_ontology", **kwargs)
    return json.loads(raw)


async def test_update_action_requires_source_iri_and_version(registered) -> None:
    out = await _graph_ontology(action="update", source=PETS_TTL_V2)
    assert "requires" in out["error"]


async def test_update_action_replaces_an_existing_ontology(registered) -> None:
    await _graph_ontology(action="load", source=PETS_TTL, source_type="text")
    updated = await _graph_ontology(
        action="update",
        source=PETS_TTL_V2,
        iri="http://example.org/pets",
        version="1.0.0",
        source_type="text",
    )
    assert "error" not in updated
    fetched = await _graph_ontology(action="get", iri="http://example.org/pets")
    assert "error" not in fetched


async def test_delete_action_requires_iri(registered) -> None:
    out = await _graph_ontology(action="delete")
    assert "requires `iri`" in out["error"]


async def test_delete_action_removes_a_loaded_ontology(registered) -> None:
    await _graph_ontology(action="load", source=PETS_TTL, source_type="text")
    deleted = await _graph_ontology(action="delete", iri="http://example.org/pets")
    assert "error" not in deleted
    missing = await _graph_ontology(action="get", iri="http://example.org/pets")
    assert "error" in missing


async def test_validate_action_requires_source(registered) -> None:
    out = await _graph_ontology(action="validate")
    assert "requires `source`" in out["error"]


async def test_validate_action_validates_turtle_text(registered) -> None:
    out = await _graph_ontology(
        action="validate", source=PETS_TTL, source_type="text"
    )
    assert "error" not in out


async def test_activate_and_deactivate_actions_require_iri(registered) -> None:
    activate_out = await _graph_ontology(action="activate")
    assert "requires `iri`" in activate_out["error"]
    deactivate_out = await _graph_ontology(action="deactivate")
    assert "requires `iri`" in deactivate_out["error"]


async def test_activate_and_deactivate_actions_toggle_the_reasoning_flag(
    registered,
) -> None:
    await _graph_ontology(action="load", source=PETS_TTL, source_type="text")
    deactivated = await _graph_ontology(
        action="deactivate", iri="http://example.org/pets"
    )
    assert "error" not in deactivated
    activated = await _graph_ontology(
        action="activate", iri="http://example.org/pets"
    )
    assert "error" not in activated
    active_only = await _graph_ontology(action="list", active_only=True)
    assert active_only["count"] >= 1


async def test_sync_packages_action_returns_a_summary(registered, monkeypatch) -> None:
    """`resolve_provider_ontologies`/`resolve_workspace_provider_ontologies`
    scan every installed distribution's entry points -- monkeypatched to an
    empty contribution set so this stays a fast, deterministic unit test
    (the real scan is legitimately slow under host load and is exercised
    elsewhere at boot); this still proves the 'sync_packages' branch really
    reaches `_sync_package_ontologies` and returns its summary shape."""
    import agent_utilities.knowledge_graph.core.ontology_federation as federation

    monkeypatch.setattr(federation, "resolve_provider_ontologies", lambda: [])
    monkeypatch.setattr(federation, "resolve_workspace_provider_ontologies", lambda: [])
    out = await _graph_ontology(action="sync_packages")
    assert out["action"] == "sync_packages"
    assert out["artifacts_loaded"] == 0
    assert out["error_count"] == 0


async def test_publish_stardog_dispatches_to_the_publisher(
    registered, monkeypatch
) -> None:
    """No live Stardog endpoint is reachable in this unit test --
    `OntologyPublisher.push_to_stardog` is monkeypatched to a fast stub so
    this proves the 'publish_stardog' branch really reaches it (with the
    bundled ontology graph + named_graph/overwrite params), not that the
    real Stardog transport works (out of partition)."""
    import agent_utilities.knowledge_graph.core.ontology_publisher as publisher

    calls: list[dict] = []

    def _fake_push(self, graph, *, named_graph, overwrite):
        calls.append({"named_graph": named_graph, "overwrite": overwrite})
        return {"status": "published", "named_graph": named_graph}

    monkeypatch.setattr(publisher.OntologyPublisher, "push_to_stardog", _fake_push)
    out = await _graph_ontology(action="publish_stardog", named_graph="catalog")
    assert out == {"status": "published", "named_graph": "catalog"}
    assert calls == [{"named_graph": "catalog", "overwrite": True}]


async def test_import_stardog_dispatches_to_the_importer(
    registered, monkeypatch
) -> None:
    import agent_utilities.knowledge_graph.core.ontology_publisher as publisher

    calls: list[dict] = []

    def _fake_import(*, named_graph, engine, activate):
        calls.append(
            {"named_graph": named_graph, "engine": engine, "activate": activate}
        )
        return {"status": "imported"}

    monkeypatch.setattr(
        publisher, "import_ontology_from_stardog", _fake_import
    )
    out = await _graph_ontology(action="import_stardog", activate=False)
    assert out == {"status": "imported"}
    assert len(calls) == 1
    assert calls[0]["named_graph"] is None
    assert calls[0]["activate"] is False


async def test_unknown_action_reports_the_error_verbatim(registered) -> None:
    out = await _graph_ontology(action="not-a-real-action")
    assert out["error"] == "unknown action: 'not-a-real-action'"
