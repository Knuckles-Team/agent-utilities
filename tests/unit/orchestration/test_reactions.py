"""Tests for the core reaction output + governance (CONCEPT:AU-ECO.reactions.emitted-alongside-reply/4.80).

Reactions are a first-class orchestrator output, not a messaging-only feature: any agent
turn can emit an ``AgentReaction``, decided by the core ``decide_reaction`` heuristic and
governed by the one ``EmoteRegistry``. These tests exercise the core type, the registry
governance gate, and the instinctive decision with a mocked model.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.orchestration.reactions import (
    AgentReaction,
    EmoteRegistry,
    decide_reaction,
)


# ── AgentReaction (CONCEPT:AU-ECO.reactions.emitted-alongside-reply) ─────────────────────────────────────────
def test_agent_reaction_normalizes_and_validates() -> None:
    r = AgentReaction(emote="  👍 ", target_message_id="100", intensity=2.0)
    assert r.emote == "👍"
    assert r.is_valid() is True
    # intensity clamped to [0, 1]
    assert r.intensity == 1.0
    assert r.target_message_id == "100"


def test_agent_reaction_empty_is_invalid() -> None:
    assert AgentReaction(emote="").is_valid() is False
    assert AgentReaction(emote="   ").is_valid() is False


def test_agent_reaction_roundtrip_dict() -> None:
    r = AgentReaction(emote="🎉", target_message_id="7", intensity=0.5)
    d = r.to_dict()
    assert d == {"emote": "🎉", "target_message_id": "7", "intensity": 0.5}
    back = AgentReaction.from_dict(d)
    assert back == r


def test_agent_reaction_dict_omits_optional_nones() -> None:
    assert AgentReaction(emote="👀").to_dict() == {"emote": "👀"}


# ── EmoteRegistry + governance (CONCEPT:AU-ECO.reactions.one-emote-registry-governance) ────────────────────────────
def test_registry_default_menu_and_known() -> None:
    reg = EmoteRegistry()
    assert "👍" in reg.available()
    assert reg.is_known("👍") is True
    assert reg.is_known("🦄") is False


def test_registry_allows_in_menu_when_no_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cosmetic reactions fail-open when the governance gate is unavailable."""
    reg = EmoteRegistry()

    # Force the ActionPolicy import path to raise -> fail-open allow for in-menu emote.
    import agent_utilities.orchestration.action_policy as ap

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("no policy")

    monkeypatch.setattr(ap, "get_action_policy", _boom)
    assert reg.allows("👍") is True
    # Unknown emote is never allowed, regardless of governance.
    assert reg.allows("🦄") is False


def test_registry_denies_when_policy_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = EmoteRegistry()
    import agent_utilities.orchestration.action_policy as ap

    class _Decision:
        decision = "deny"

    class _Policy:
        def decide(self, _req: Any) -> Any:
            return _Decision()

    monkeypatch.setattr(ap, "get_action_policy", lambda _e=None: _Policy())
    assert reg.allows("👍") is False


def test_registry_allows_when_policy_queues(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-deny decision (e.g. queue_approval) still permits a cosmetic emote."""
    reg = EmoteRegistry()
    import agent_utilities.orchestration.action_policy as ap

    class _Decision:
        decision = "queue_approval"

    class _Policy:
        def decide(self, _req: Any) -> Any:
            return _Decision()

    monkeypatch.setattr(ap, "get_action_policy", lambda _e=None: _Policy())
    assert reg.allows("👍") is True


# ── decide_reaction (CONCEPT:AU-ECO.reactions.emitted-alongside-reply) ───────────────────────────────────────
class _FakeResult:
    def __init__(self, out: str) -> None:
        self.output = out


class _FakeAgent:
    """Stands in for pydantic_ai.Agent — returns a canned emoji."""

    _canned = "👍"

    def __init__(self, *_a: Any, **_k: Any) -> None:
        pass

    async def run(self, _content: str) -> _FakeResult:
        return _FakeResult(self._canned)


@pytest.fixture()
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import pydantic_ai

    monkeypatch.setattr(pydantic_ai, "Agent", _FakeAgent)
    monkeypatch.setattr(
        "agent_utilities.core.model_factory.create_model", lambda *a, **k: object()
    )
    # Fresh, policy-free registry so the in-menu emote is allowed (fail-open).
    EmoteRegistry._instance = None


@pytest.mark.asyncio
async def test_decide_reaction_emits_for_in_menu(
    fake_model: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_FakeAgent, "_canned", "👍")
    reaction = await decide_reaction("please look into this", target_message_id="55")
    assert isinstance(reaction, AgentReaction)
    assert reaction.emote == "👍"
    assert reaction.target_message_id == "55"


@pytest.mark.asyncio
async def test_decide_reaction_none_word(
    fake_model: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_FakeAgent, "_canned", "NONE")
    assert await decide_reaction("ok") is None


@pytest.mark.asyncio
async def test_decide_reaction_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REACTIONS", "0")
    assert await decide_reaction("great job!") is None


@pytest.mark.asyncio
async def test_decide_reaction_messaging_alias_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy MESSAGING_REACTIONS opt-out still disables the now-core decision."""
    monkeypatch.setenv("MESSAGING_REACTIONS", "0")
    assert await decide_reaction("great job!") is None


@pytest.mark.asyncio
async def test_decide_reaction_empty_content(fake_model: None) -> None:
    assert await decide_reaction("") is None
