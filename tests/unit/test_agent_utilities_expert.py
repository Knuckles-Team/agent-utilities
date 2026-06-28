#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:ORCH-1.100 — the agent-utilities-expert is a well-formed, registry-loadable
prompt that is wired into a dispatchable AgentTemplate.

These are LIVE-PATH tests (Wire-First): they exercise the real prompt-loading
path and the real seeding/resolution path, not just the data file.
"""

import json
from pathlib import Path

from agent_utilities.agent.registry_builder import (
    _BUILTIN_AGENT_TEMPLATES,
    seed_builtin_agent_templates,
)
from agent_utilities.core.config import load_specialized_prompts
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.orchestration.agent_runner import _resolve_agent_from_kg
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


def test_expert_prompt_is_canonical_and_loadable() -> None:
    """The prompt file is canonical-valid and renders a real persona body."""
    data = json.loads(_PROMPT_PATH.read_text(encoding="utf-8"))
    assert validate_canonical(data, strict=True) == []

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
    assert "Agent Utilities Ecosystem Expert" in rendered


def test_expert_is_a_dispatchable_agent_template() -> None:
    """Seeding registers a resolvable AgentTemplate bound to the prompt + local model."""
    tmpl = next(t for t in _BUILTIN_AGENT_TEMPLATES if t["name"] == EXPERT)
    assert tmpl["system_prompt_id"] == f"prompt:{EXPERT}"
    assert tmpl["model_preference"].startswith("qwen/")
    assert "graph-os" in tmpl["toolset_ids"]
    assert "repository-manager-mcp" in tmpl["toolset_ids"]

    engine = IntelligenceGraphEngine(db_path=":memory:")
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
    assert "Agent Utilities Ecosystem Expert" in meta["system_prompt"]
