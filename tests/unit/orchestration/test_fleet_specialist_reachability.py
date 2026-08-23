"""CONCEPT:AU-ORCH.dispatch.fleet-specialist-reachability — a thin entrypoint persona
(e.g. the agent-webui default ``webui-assistant``) must not accidentally starve the
router of real fleet specialists.

Production incident this closes: ``_build_execution_config`` seeded ``tag_prompts``
with a bare ``{agent_name: "Specialized agent: <name>"}`` placeholder for ANY agent
that resolved with no real persona and no capabilities (the exact
``_unresolved_agent_meta()`` shape a name like ``webui-assistant`` — never a real KG
AgentTemplate — resolves to on a miss). That single-entry dict is truthy, so
``graph/_router_impl.py``'s ``router_step`` — which treats an EMPTY ``deps.tag_prompts``
as "load the full fleet registry" (``get_discovery_registry()``) — never took that
fallback branch. The router's own free-text specialist proposals (e.g.
"agent-utilities-expert") then matched nothing against the one-entry registry, and the
turn died with an empty plan (production symptom: ``Available: ['webui-assistant']``).

These are wiring-adjacent unit tests on ``_build_execution_config`` itself (the seam
that owns ``tag_prompts`` construction) — they drive the REAL function, not a mock of
it, and assert the exact shape ``graph/_router_impl.py`` reads off ``deps.tag_prompts``.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def _synthetic_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A minimal configured model so ``_build_execution_config`` doesn't need real LLM config."""
    from agent_utilities.core.config import ChatModelConfig, config

    monkeypatch.setattr(
        config,
        "chat_models",
        [
            ChatModelConfig(
                id="synthetic-standard", provider="openai", intelligence_level="normal"
            )
        ],
    )


def test_thin_unresolved_agent_leaves_tag_prompts_empty(
    _synthetic_chat_model: None,
) -> None:
    """A miss-shaped agent_meta (no system_prompt, no capabilities — the shape a
    KG-unresolved name like 'webui-assistant' gets) must NOT seed a single bare
    placeholder entry: that would be the exact truthy-one-entry-dict bug. An EMPTY
    ``tag_prompts`` is what lets the router's own registry-widening fallback fire.
    """
    from agent_utilities.orchestration.agent_runner import _build_execution_config

    meta: dict[str, Any] = {
        "type": "unknown",
        "capabilities": [],
        "tools": [],
        "system_prompt": "",
    }

    config = _build_execution_config(
        None,
        "webui-assistant",
        meta,
        execution_profile="chat",
        recent_mementos=[],
    )

    assert config["tag_prompts"] == {}
    assert "webui-assistant" not in config["tag_prompts"]


def test_resolved_persona_still_claims_its_own_domain_slot(
    _synthetic_chat_model: None,
) -> None:
    """A REAL resolved persona (a seeded AgentTemplate system_prompt) must still drive
    the run under its own name — this fix must not regress the seeded-agent-template
    path (CONCEPT:AU-ORCH.dispatch.seeded-agent-template)."""
    from agent_utilities.orchestration.agent_runner import _build_execution_config

    meta: dict[str, Any] = {
        "type": "agent_template",
        "capabilities": [],
        "tools": [],
        "system_prompt": "You are the epistemic-graph expert.",
    }

    config = _build_execution_config(
        None,
        "agent-utilities-expert",
        meta,
        execution_profile="chat",
        recent_mementos=[],
    )

    assert config["tag_prompts"]["agent-utilities-expert"] == (
        "You are the epistemic-graph expert."
    )


def test_agent_with_real_capabilities_still_populates_tag_prompts(
    _synthetic_chat_model: None,
) -> None:
    """An agent with actual bound capabilities (even with no resolved persona text)
    has something real to offer the router and must still claim its domain slot plus
    each capability tag — only the FULLY-empty (persona AND capabilities) case is
    left for the router's registry fallback."""
    from agent_utilities.orchestration.agent_runner import _build_execution_config

    meta: dict[str, Any] = {
        "type": "unknown",
        "capabilities": ["github-mcp"],
        "tools": [],
        "system_prompt": "",
    }

    config = _build_execution_config(
        None,
        "messaging-assistant",
        meta,
        execution_profile="chat",
        recent_mementos=[],
    )

    assert "messaging-assistant" in config["tag_prompts"]
    assert config["tag_prompts"]["github-mcp"] == "Capability: github-mcp"
