"""Live-path tests for the ontology-evolution governed-proposal actions on the
REAL ``graph_ontology`` MCP tool (CONCEPT:AU-KG.ontology.evolution-governed-loop,
program item 7.5) — 'propose'/'list_proposals'/'get_proposal'/'review_proposal'/
'promote_proposal'/'rollback_proposal'.

Exercises the REAL tool dispatched through the REAL ``_execute_tool``, the same
pattern ``test_ontology_catalogue_live_path.py`` uses — this fails if the
wiring from the tool body to ``knowledge_graph.ontology.evolution`` ever
breaks. No live engine required.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.ontology import evolution
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


class _CollectingMCP:
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
    evolution.reset_proposal_registry()
    yield
    reset_registry()
    evolution.reset_proposal_registry()


@pytest.fixture
def registered() -> dict[str, object]:
    mcp = _CollectingMCP()
    ontology_tools.register_ontology_tools(mcp)
    return mcp.tools


async def _graph_ontology(**kwargs) -> dict:
    raw = await kg_server._execute_tool("graph_ontology", **kwargs)
    return json.loads(raw)


async def test_propose_through_real_tool_never_hosts_directly(registered):
    payload = await _graph_ontology(
        action="propose",
        source=PETS_TTL,
        source_type="text",
        iri="http://example.org/pets",
    )
    assert payload["status"] == "ok"
    proposal = payload["proposal"]
    assert proposal["classification"]["kind"] == "additive"
    assert proposal["status"] == "pending_review"

    listing = await _graph_ontology(action="list")
    assert listing["count"] == 0  # propose never auto-merges


async def test_list_and_get_proposal_through_real_tool(registered):
    proposed = await _graph_ontology(
        action="propose",
        source=PETS_TTL,
        source_type="text",
        iri="http://example.org/pets",
    )
    proposal_id = proposed["proposal"]["proposal_id"]

    listed = await _graph_ontology(action="list_proposals")
    assert any(p["proposal_id"] == proposal_id for p in listed["proposals"])

    fetched = await _graph_ontology(action="get_proposal", proposal_id=proposal_id)
    assert fetched["proposal"]["proposal_id"] == proposal_id

    missing = await _graph_ontology(action="get_proposal", proposal_id="nope")
    assert "error" in missing


async def test_review_then_promote_is_held_by_default_through_real_tool(registered):
    """SAFETY-CRITICAL: through the real MCP surface, promotion is STILL gated
    even after an approving review -- nothing auto-merges."""
    proposed = await _graph_ontology(
        action="propose",
        source=PETS_TTL,
        source_type="text",
        iri="http://example.org/pets",
    )
    proposal_id = proposed["proposal"]["proposal_id"]

    reviewed = await _graph_ontology(
        action="review_proposal",
        proposal_id=proposal_id,
        approve=True,
        reviewer="alice",
    )
    assert reviewed["proposal"]["status"] == "approved"

    promoted = await _graph_ontology(action="promote_proposal", proposal_id=proposal_id)
    assert promoted["status"] == "held"

    listing = await _graph_ontology(action="list")
    assert listing["count"] == 0


async def test_rollback_unpromoted_proposal_errors_through_real_tool(registered):
    proposed = await _graph_ontology(
        action="propose",
        source=PETS_TTL,
        source_type="text",
        iri="http://example.org/pets",
    )
    proposal_id = proposed["proposal"]["proposal_id"]
    rolled_back = await _graph_ontology(
        action="rollback_proposal", proposal_id=proposal_id
    )
    assert "error" in rolled_back


async def test_propose_requires_source_and_iri(registered):
    missing_source = await _graph_ontology(
        action="propose", iri="http://example.org/pets"
    )
    assert "error" in missing_source
    missing_iri = await _graph_ontology(
        action="propose", source=PETS_TTL, source_type="text"
    )
    assert "error" in missing_iri
