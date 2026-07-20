"""Live-path tests for the hosted-ontology import/export + catalogue surface
(Ontology-Playground coverage rows #23 and #4).

Exercises the REAL ``graph_ontology`` MCP tool, registered by
``register_ontology_tools`` and dispatched through the REAL ``_execute_tool``
(the same single-source-of-truth dispatcher the REST granular routes'
``_call()`` helper and every MCP client use) — not a mock of either. This
fails if the wiring from the tool body to
``OntologyLifecycle.load``/``list_ontologies`` ever breaks.

Deliberately goes through ``_execute_tool`` rather than calling the registered
closure directly: the tool's params are declared ``name: T = Field(default=...)``,
and ``_execute_tool`` is what resolves an OMITTED param's raw ``FieldInfo``
default down to its real value (documented in ``kg_server._execute_tool``) —
calling the closure directly would bind any omitted kwarg to the literal
``FieldInfo`` object instead of ``""``/``False``. No live engine required — the
tool degrades to ``engine=None`` exactly like ``OntologyLifecycle(engine=None)``
in ``test_lifecycle.py``.
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

FINANCE_TTL = (
    "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
    "<http://example.org/finance> a owl:Ontology .\n"
    "<http://example.org/finance#Invoice> a owl:Class .\n"
)


class _CollectingMCP:
    """Minimal FastMCP stand-in that captures every ``@mcp.tool``-registered function.

    ``register_ontology_tools`` ALSO assigns directly into the module-level
    ``kg_server.REGISTERED_TOOLS`` (the real dispatch table), so registering
    against this fake is enough to make ``_execute_tool`` reach the real tool.
    """

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
    """Register the real ontology tools (populates kg_server.REGISTERED_TOOLS)."""
    mcp = _CollectingMCP()
    ontology_tools.register_ontology_tools(mcp)
    return mcp.tools


async def _graph_ontology(**kwargs) -> dict:
    raw = await kg_server._execute_tool("graph_ontology", **kwargs)
    return json.loads(raw)


def test_graph_ontology_registered(registered):
    assert "graph_ontology" in registered


# ── Import / export round-trip through the real tool (row #23) ──────────────


async def test_load_then_get_serialize_round_trips_turtle(registered):
    load_payload = await _graph_ontology(action="load", source=PETS_TTL, source_type="text")
    assert load_payload["status"] == "ok"
    iri = load_payload["ontology"]["iri"]
    assert iri == "http://example.org/pets"

    export_payload = await _graph_ontology(action="get", iri=iri, serialize=True)
    turtle = export_payload["ontology"]["turtle"]
    assert "ex:Dog" in turtle or "pets#Dog" in turtle


async def test_load_stores_category_and_tags_through_real_tool(registered):
    payload = await _graph_ontology(
        action="load",
        source=PETS_TTL,
        source_type="text",
        category="animals",
        tags_json=json.dumps(["demo", "pets"]),
    )
    assert payload["ontology"]["category"] == "animals"
    assert payload["ontology"]["tags"] == ["demo", "pets"]


async def test_load_tolerates_malformed_tags_json(registered):
    """A cosmetic-metadata parse failure degrades to an empty tag list, never a crash."""
    payload = await _graph_ontology(
        action="load", source=PETS_TTL, source_type="text", tags_json="not json"
    )
    assert payload["status"] == "ok"
    assert payload["ontology"]["tags"] == []


# ── Catalogue browse/search/filter (row #4) ──────────────────────────────────


async def test_list_action_default_is_unfiltered_despite_source_type_default_auto(
    registered,
):
    """Regression guard: the tool's `source_type` Field defaults to 'auto' (the
    load/validate parse-hint sentinel). A plain action='list' call must NOT
    silently filter to source_type=='auto' — it must return every hosted
    ontology, exactly like before the catalogue filters existed."""
    await _graph_ontology(action="load", source=PETS_TTL, source_type="text")
    await _graph_ontology(action="load", source=FINANCE_TTL, source_type="text")

    payload = await _graph_ontology(action="list")
    assert payload["count"] == 2


async def test_list_action_filters_by_category_tag_and_search(registered):
    await _graph_ontology(
        action="load",
        source=PETS_TTL,
        source_type="text",
        category="animals",
        tags_json=json.dumps(["demo"]),
    )
    await _graph_ontology(
        action="load",
        source=FINANCE_TTL,
        source_type="text",
        category="finance",
        tags_json=json.dumps(["draft"]),
    )

    by_category = await _graph_ontology(action="list", category="finance")
    assert by_category["count"] == 1
    assert by_category["ontologies"][0]["iri"] == "http://example.org/finance"

    by_tag = await _graph_ontology(action="list", tag="demo")
    assert by_tag["count"] == 1
    assert by_tag["ontologies"][0]["iri"] == "http://example.org/pets"

    by_search = await _graph_ontology(action="list", search="pets")
    assert by_search["count"] == 1

    by_explicit_source_type = await _graph_ontology(action="list", source_type="text")
    assert by_explicit_source_type["count"] == 2  # both were loaded as source_type='text'

    no_match = await _graph_ontology(action="list", category="no-such-category")
    assert no_match["count"] == 0
