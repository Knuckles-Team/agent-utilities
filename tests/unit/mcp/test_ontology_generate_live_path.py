"""Live-path test for the from-scratch ontology generator (coverage row #13).

CONCEPT:AU-KG.ontology.standalone-generation — exercises the REAL
``ontology_derive(action="generate")`` MCP tool function as registered by
``register_ontology_tools`` (not a mock of the tool itself), so this fails if
the wiring from the tool body to
``schema_discovery.generate_standalone_ontology``/``ontology_generation_report``
ever breaks. Mirrors the ``_CollectingMCP`` stand-in pattern used by
``tests/unit/test_engine_surface_tools.py``. Only the LLM completion function
is faked (no network call in tests).
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.mcp.tools import ontology_tools


class _CollectingMCP:
    """Minimal FastMCP stand-in that captures every ``@mcp.tool``-registered function.

    ``register_ontology_tools`` registers tools with varying decorator call
    shapes (``@mcp.tool(name=..., description=..., tags=...)`` and a bare
    ``@mcp.tool()`` for a nested quant sub-registrar), so ``tool()`` accepts
    ``*args, **kwargs`` and falls back to the wrapped function's ``__name__``.
    """

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        def _deco(fn):
            self.tools[kwargs.get("name", fn.__name__)] = fn
            return fn

        return _deco


@pytest.fixture
def tools() -> dict[str, object]:
    """Register the real ontology tools onto a collecting MCP and return them by name."""
    mcp = _CollectingMCP()
    ontology_tools.register_ontology_tools(mcp)
    return mcp.tools


def _fake_llm(prompt: str) -> str:
    return (
        '{"entity_types":[{"name":"VetClinic","description":"a veterinary clinic"}],'
        '"relation_types":[{"name":"treats","domain":"VetClinic","range":"Animal"}]}'
    )


def test_ontology_derive_registered(tools):
    assert "ontology_derive" in tools


def test_generate_action_returns_standalone_interface_linktype_proposal(
    monkeypatch, tools
):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.cards.make_lite_llm_fn",
        lambda: _fake_llm,
    )
    ontology_derive = tools["ontology_derive"]
    raw = ontology_derive(
        action="generate",
        sample_text="Sample business-scenario text about a veterinary clinic",
        object_type="veterinary",
    )
    payload = json.loads(raw)

    assert payload["domain_hint"] == "veterinary"
    interface_names = {i["name"] for i in payload["interfaces"]}
    link_names = {lk["name"] for lk in payload["link_types"]}
    assert "VetClinic" in interface_names
    assert "treats" in link_names
    assert payload["counts"] == {"interfaces": 1, "link_types": 1}
    # Reviewable Turtle proposal, gated (never auto-applied) — same
    # RESERVE-PENDING convention as discover_extensions' ttl_proposal.
    assert ":VetClinic a owl:Class" in payload["ttl_proposal"]
    assert "RESERVE-PENDING" in payload["ttl_proposal"]


def test_generate_action_is_a_complete_proposal_not_a_diff(monkeypatch, tools):
    """Unlike 'discover_extensions', 'generate' never filters against the live ontology."""

    def fake_llm(prompt: str) -> str:
        # "Person" would classify "covered" in discover_extensions' live-ontology
        # diff; 'generate' must still propose it in full (empty base).
        return '{"entity_types":[{"name":"Person","description":"a person"}],"relation_types":[]}'

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.cards.make_lite_llm_fn",
        lambda: fake_llm,
    )
    ontology_derive = tools["ontology_derive"]
    raw = ontology_derive(action="generate", sample_text="x", object_type="document")
    payload = json.loads(raw)
    assert any(i["name"] == "Person" for i in payload["interfaces"])


def test_generate_action_never_auto_applies(monkeypatch, tools):
    """The proposal is returned only — nothing writes to the live ontology system."""
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.cards.make_lite_llm_fn",
        lambda: _fake_llm,
    )

    def _boom():
        raise AssertionError("action='generate' must never touch the live ontology system")

    monkeypatch.setattr(
        "agent_utilities.mcp.kg_server._ontology_system", lambda: _boom()
    )
    ontology_derive = tools["ontology_derive"]
    raw = ontology_derive(
        action="generate", sample_text="text", object_type="document"
    )
    payload = json.loads(raw)
    assert "error" not in payload
