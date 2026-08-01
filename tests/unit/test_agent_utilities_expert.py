#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-ORCH.dispatch.builtin-agent-templates — the agent-utilities-expert is a well-formed, registry-loadable
prompt that is wired into a dispatchable AgentTemplate.

These are LIVE-PATH tests (Wire-First): they exercise the real prompt-loading
path and the real seeding/resolution path, not just the data file.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_utilities.agent.registry_builder import (
    _BUILTIN_AGENT_TEMPLATES,
    seed_builtin_agent_templates,
)
from agent_utilities.core.config import load_specialized_prompts
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.orchestration.agent_runner import (
    _build_execution_config,
    _is_bound_template_agent,
    _resolve_agent_from_kg,
    _resolve_toolset_ids,
)
from agent_utilities.prompting.structured import (
    StructuredPrompt,
    validate_canonical,
)

EXPERT = "agent-utilities-expert"
_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "agent_utilities"
    / "prompts"
    / f"{EXPERT}.json"
)


def _isolated_engine() -> IntelligenceGraphEngine:
    """Build an ``IntelligenceGraphEngine`` bound to THIS test's isolated graph.

    A bare ``IntelligenceGraphEngine(db_path=":memory:")`` constructs a bare
    ``EpistemicGraphBackend()``, which independently resolves the ambient
    tenant's SHARED default graph (``resolve_routing_graph(None)``) rather
    than the per-test isolated graph the autouse
    ``isolate_graph_compute_engine`` fixture provisions (that redirect only
    intercepts a literal ``None``/``"__commons__"``/``"__secrets__"``
    ``graph_name`` passed straight to ``GraphComputeEngine.__init__``, which
    a bare ``EpistemicGraphBackend()`` bypasses). Construct the isolated
    ``GraphComputeEngine`` first and rebind the backend to it before
    constructing the engine — the same idiom already established for this
    exact defect (D-OTR-2/D-OTR-3).
    """
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.graph_compute import (
        GraphComputeEngine,
    )

    compute = GraphComputeEngine(backend_type="rust")
    backend = EpistemicGraphBackend()
    backend._graph = compute
    return IntelligenceGraphEngine(backend=backend)


def test_expert_prompt_is_canonical_and_loadable() -> None:
    """The prompt file is canonical-valid and renders a real persona body."""
    data = json.loads(_PROMPT_PATH.read_text(encoding="utf-8"))
    assert validate_canonical(data) == []

    prompt = StructuredPrompt.model_validate(data)
    body = prompt.render()
    # Grounded coverage of the required expertise areas.
    for marker in (
        "5 pillars",
        "epistemic-graph",
        "worktree",
        ".specify/specs",
        "code_context",
        "graph_loops",
        "source_sync",
        "SpecProposal",
    ):
        assert marker in body, f"expert prompt missing expertise marker: {marker!r}"


def test_expert_loads_from_registry() -> None:
    """``load_specialized_prompts`` resolves the expert from the prompt registry."""
    rendered = load_specialized_prompts(EXPERT)
    assert isinstance(rendered, str)
    # StructuredPrompt.render() emits the instructions/rules body, not the raw
    # identity.role label (test_expert_prompt_is_canonical_and_loadable already
    # covers this exact contract with the same marker) -- assert on rendered
    # content that only the real expert persona (not a generic placeholder)
    # would produce.
    assert "epistemic-graph" in rendered


def test_expert_is_a_dispatchable_agent_template(monkeypatch, tmp_path: Path) -> None:
    """Seeding registers a resolvable AgentTemplate bound to the prompt + local model."""
    # Live GraphOS deliberately has no graph-os child in MCP_CONFIG: self tools
    # must bind in-process, while the remaining declared fleet toolsets stay HTTP.
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    name: {"url": f"https://{name}.example.test/mcp"}
                    for name in (
                        "repository-manager-mcp",
                        "data-science-mcp",
                        "scholarx-mcp",
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MCP_CONFIG", str(config_path))
    monkeypatch.delenv("FLEET_MCP_URL_TEMPLATE", raising=False)
    tmpl = next(t for t in _BUILTIN_AGENT_TEMPLATES if t["name"] == EXPERT)
    assert tmpl["system_prompt_id"] == f"prompt:{EXPERT}"
    assert tmpl["model_preference"].startswith("qwen/")
    assert "graph-os" in tmpl["toolset_ids"]
    assert "repository-manager-mcp" in tmpl["toolset_ids"]

    engine = _isolated_engine()
    if engine.backend is None:  # pragma: no cover - backend-less env
        return
    engine.backend.create_schema()

    # Ingest the prompt node the template binds to, then seed the template.
    data = json.loads(_PROMPT_PATH.read_text(encoding="utf-8"))
    body = StructuredPrompt.model_validate(data).render()
    engine._upsert_node(
        "Prompt",
        f"prompt:{EXPERT}",
        {"id": f"prompt:{EXPERT}", "name": EXPERT, "system_prompt": body},
    )

    seeded = seed_builtin_agent_templates(engine)
    assert seeded >= 1

    # The orchestrator's resolution path discovers it as a dispatchable agent and
    # recovers its persona via the USES_PROMPT-linked Prompt node.
    meta = _resolve_agent_from_kg(engine, EXPERT)
    assert meta["type"] == "agent_template"
    assert "graph-os" in meta["capabilities"]
    # Same StructuredPrompt.render() contract as test_expert_loads_from_registry:
    # the rendered body carries the instructions/rules text, not the raw
    # identity.role label.
    assert "epistemic-graph" in meta["system_prompt"]

    # CONCEPT:AU-ORCH.adapter.transport-toolset-factory — resolution surfaces the persona AND the toolset_ids;
    # _build_execution_config must turn those toolset_ids into LIVE MCP toolsets so
    # the dispatched expert can query graph-os and ground its answer (the fix that
    # stops the prompt-only hallucination). Assert the binding + the routing
    # predicate that sends it down the direct grounding loop.
    #
    # This hermetic test environment has no configured chat_models, so
    # _configured_model_for_class would raise "configured standard model
    # class is unavailable" — same reason
    # test_agent_template_rejects_explicit_undeclared_tool_allow_list (below)
    # already stubs it out. Only the id/routing metadata matters here, not
    # which real model gets selected.
    monkeypatch.setattr(
        "agent_utilities.orchestration.agent_runner._configured_model_for_class",
        lambda _model_class: SimpleNamespace(
            id="synthetic-model",
            provider="test",
            base_url=None,
            api_key_ref=None,
        ),
    )
    config = _build_execution_config(
        engine,
        EXPERT,
        meta,
        recent_mementos=[],
        allowed_tools=["graph_analyze"],
    )
    bound = config.get("mcp_toolsets") or []
    assert len(bound) == len(meta["capabilities"]), (
        "every declared toolset_id must bind to a live toolset"
    )
    assert _is_bound_template_agent(meta, config), (
        "a bound AgentTemplate must route to the direct grounding loop, not the planner"
    )
    # The persona (not the bare 'Specialized agent' placeholder) drives the run.
    assert "epistemic-graph" in config["tag_prompts"][EXPERT]
    native = next(toolset for toolset in bound if toolset.id == "graph-os")
    assert native.metadata == {"graphos_native": True}
    assert list(native.tools) == ["graph_analyze"]


def test_agent_template_rejects_explicit_undeclared_tool_allow_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller grants may narrow an AgentTemplate, never add capabilities."""
    meta = {
        "type": "agent_template",
        "capabilities": ["graph-os"],
        "tools": [{"name": "graph_analyze"}],
    }
    monkeypatch.setattr(
        "agent_utilities.orchestration.agent_runner._configured_model_for_class",
        lambda _model_class: SimpleNamespace(
            id="synthetic-model",
            provider="test",
            base_url=None,
            api_key_ref=None,
        ),
    )

    with pytest.raises(PermissionError, match="undeclared tool"):
        _build_execution_config(
            None,
            "synthetic-template",
            meta,
            recent_mementos=[],
            allowed_tools=["graph_query"],
        )


def test_expert_binds_graphos_natively_without_mcp_config_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The resident expert never resolves its declared graph-os toolset over HTTP."""
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "repository-manager-mcp": {
                        "url": "https://repository-manager.example.test/mcp"
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MCP_CONFIG", str(config_path))
    monkeypatch.delenv("FLEET_MCP_URL_TEMPLATE", raising=False)
    monkeypatch.setattr(
        "agent_utilities.orchestration.agent_runner._configured_model_for_class",
        lambda _model_class: SimpleNamespace(
            id="synthetic-model",
            provider="test",
            base_url=None,
            api_key_ref=None,
        ),
    )
    template = next(item for item in _BUILTIN_AGENT_TEMPLATES if item["name"] == EXPERT)
    meta = {
        "type": "agent_template",
        "capabilities": ["graph-os", "repository-manager-mcp"],
        "tools": [],
        "system_prompt": "Synthetic expert prompt",
    }
    assert set(meta["capabilities"]).issubset(template["toolset_ids"])

    config = _build_execution_config(
        object(),
        EXPERT,
        meta,
        recent_mementos=[],
        allowed_tools=["graph_analyze"],
    )

    native = next(
        toolset for toolset in config["mcp_toolsets"] if toolset.id == "graph-os"
    )
    assert native.metadata == {"graphos_native": True}
    assert list(native.tools) == ["graph_analyze"]
    assert len(config["mcp_toolsets"]) == 2


def test_resolve_toolset_ids_binds_live_toolsets(monkeypatch) -> None:
    """``_resolve_toolset_ids`` turns a list of fleet server ids into live toolsets.

    CONCEPT:AU-ORCH.adapter.transport-toolset-factory — the binding seam. With no ``:Server`` node present it falls
    back to the fleet served-URL convention (the same resolution the focused-tools
    path uses), binding one callable ``MCPToolset`` per id.
    """
    monkeypatch.setenv("FLEET_MCP_URL_TEMPLATE", "https://{server}.example.test/mcp")
    engine = _isolated_engine()
    ids = ["repository-manager-mcp", "data-science-mcp"]
    toolsets = _resolve_toolset_ids(engine, ids)
    assert len(toolsets) == len(ids)
    # Each is a real callable toolset (supports tool filtering — the least-privilege
    # contract _execute_single_server relies on), not a prompt string.
    for ts in toolsets:
        assert hasattr(ts, "filtered"), "bound object must be a real MCPToolset"

    # An empty id is skipped (no phantom toolset).
    assert _resolve_toolset_ids(engine, ["", None]) == []  # type: ignore[list-item]
