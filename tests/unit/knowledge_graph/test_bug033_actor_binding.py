"""BUG-033/BUG-039 regression tests.

Three write paths (``chat_persistence.save_chat_to_disk``,
``conversation_ingestion.ingest_conversations_to_kg``,
``messaging.enrichment.enrich_conversation``) used to write ``Thread``/
``Message``/``Concept`` nodes through the raw engine with whatever ambient
actor happened to be bound -- often none. The downstream ownership stamp
(``tenant_sharing.stamp_ownership``) then raised ``PermissionError``, which
was swallowed at five chokepoints, so the write proceeded anyway and the
node landed completely unowned -- and ``filter_visible`` treats unowned as
visible to everyone (a separate, NOT-in-scope change; see BUG-033/BD-033).

This file proves, with a known-bad input at each layer:

1. ``security.request_identity.system_write_session`` never returns an
   unauthenticated identity: it prefers the caller's own ambient
   ``GraphSession`` and otherwise mints (once, cached) this process's own
   verified system identity through the existing minting convention.
2. Each of the three named write paths binds a REAL actor before writing --
   even when the calling context starts with NO ambient actor at all (the
   isolated ``contextvars.Context()`` construction the rest of this test
   suite already uses to simulate a genuinely actor-free background caller,
   e.g. the cron scheduler's bare asyncio loop).
3. The five ``stamp_ownership`` chokepoints (covered in more detail in
   ``test_acl_registration_chokepoints.py``) now fail closed rather than
   silently proceeding unstamped.

Revert the fix (restore the ``try/except PermissionError: pass`` swallow at
the five chokepoints, and drop the ``system_write_session`` binding from the
three write paths) and every test in this file that asserts a bound actor or
a raised ``PermissionError`` fails; every test that currently expects a
raise instead observes the write silently succeed with governance properties
missing. That before/after contrast is the property this file exists to
prove.
"""

from __future__ import annotations

import contextvars
from unittest.mock import MagicMock

import pytest

from agent_utilities.security.brain_context import IdentityRequiredError, current_actor


@pytest.fixture(autouse=True)
def _reset_system_write_session_cache(monkeypatch):
    """Each test starts with no cached fallback identity.

    ``security.request_identity`` caches the minted process identity at
    module scope (mint once, reuse) -- reset it per test so a cache hit in
    one test can never mask a genuine minting failure in another.
    """
    import agent_utilities.security.request_identity as ri

    monkeypatch.setattr(ri, "_system_write_session", None)
    yield
    monkeypatch.setattr(ri, "_system_write_session", None)


# ---------------------------------------------------------------------------
# system_write_session()
# ---------------------------------------------------------------------------


def test_system_write_session_prefers_the_ambient_session():
    """A caller that already has a verified session bound keeps its OWN
    identity -- the fallback minting path must never override a real
    caller's authority with a generic system one."""
    from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext
    from agent_utilities.security.request_identity import system_write_session

    actor = ActorContext(
        actor_id="writer:alice",
        actor_type=ActorType.HUMAN,
        roles=(),
        tenant_id="tenant-a",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant="tenant-a",
        scopes=frozenset({"kg:write"}),
        graph="tenant-a-graph",
        policy_version="v1",
        audience="test-audience",
    )
    with use_session(session):
        resolved = system_write_session()

    assert resolved is session
    assert resolved.actor.actor_id == "writer:alice"


def _force_local_process_authority(monkeypatch) -> None:
    """Force the deterministic, no-network ``mint_local_process_session`` path.

    The test session's own ``_session_engine`` autouse fixture (conftest.py)
    exports ``GRAPH_SERVICE_ENDPOINTS`` for the whole run when a real engine
    binary is available, which makes ``local_process_authority_enabled``
    environment-dependent here. Tests that want to exercise the *mint*
    behaviour deterministically (not "whichever branch this sandbox happens
    to take") pin it explicitly; ``test_system_write_session_never_returns_
    unauthenticated`` below instead exercises the OTHER branch on purpose.
    """
    from agent_utilities.security import request_identity as ri

    monkeypatch.setattr(ri, "local_process_authority_enabled", lambda _config: True)


def test_system_write_session_mints_a_real_actor_with_no_ambient_session(monkeypatch):
    """BUG-033's actual gap: no ambient session at all (a bare background
    caller). ``system_write_session`` must mint a REAL, authenticated actor
    -- never return an unauthenticated/empty identity."""
    _force_local_process_authority(monkeypatch)
    from agent_utilities.security.request_identity import system_write_session

    def isolated():
        return system_write_session()

    session = contextvars.Context().run(isolated)

    assert session.actor.authenticated is True
    assert session.actor.actor_id
    assert session.tenant
    assert session.audience
    assert session.policy_version


def test_system_write_session_mint_is_cached_across_calls(monkeypatch):
    """The fallback identity is minted ONCE per process and reused -- not
    re-minted (and not re-validated over the network) on every write."""
    _force_local_process_authority(monkeypatch)
    from agent_utilities.security.request_identity import system_write_session

    def isolated():
        first = system_write_session()
        second = system_write_session()
        return first, second

    first, second = contextvars.Context().run(isolated)
    assert first is second


def test_system_write_session_never_returns_unauthenticated(monkeypatch):
    """A genuine minting failure (external identity configured but
    unreachable) must propagate loudly -- never degrade to an
    unauthenticated/synthesized session that would defeat the whole point of
    this fix."""
    from agent_utilities.core.config import config
    from agent_utilities.security import request_identity as ri

    monkeypatch.setattr(config, "deployment_profile", "single-node-prod")
    monkeypatch.setattr(config, "graph_service_endpoints", ["https://example.invalid"])
    monkeypatch.setattr(config, "kg_auth_token_ref", None)
    monkeypatch.setattr(config, "kg_identity_oauth2", None)

    def isolated():
        with pytest.raises(RuntimeError):
            ri.system_write_session()

    contextvars.Context().run(isolated)


# ---------------------------------------------------------------------------
# chat_persistence.save_chat_to_disk
# ---------------------------------------------------------------------------


def test_save_chat_to_disk_binds_a_real_actor_with_no_ambient_session(monkeypatch):
    """Simulates the cron scheduler's bare background loop (BUG-033's
    reproduced gap): no actor/session bound anywhere in the calling context.
    The Message-node write must still observe a REAL, authenticated actor,
    and the Thread-node raw-Cypher write must carry the governance stamp."""
    _force_local_process_authority(monkeypatch)
    import agent_utilities.core.chat_persistence as cp
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    captured_actors = []

    def _capture_add_node(*_a, **_kw):
        captured_actors.append(current_actor())
        return None

    fake_engine.add_node.side_effect = _capture_add_node
    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", staticmethod(lambda: fake_engine)
    )

    def isolated():
        cp.save_chat_to_disk(
            "chat-isolated", [{"role": "user", "content": "hello there"}]
        )

    contextvars.Context().run(isolated)

    # The Message-node write ran under a real, authenticated actor.
    assert len(captured_actors) == 1
    assert captured_actors[0].authenticated is True
    assert captured_actors[0].actor_id

    # The Thread-node raw MERGE carries the governance stamp (this write
    # bypasses the generic ``_upsert_node`` seam entirely, so it is stamped
    # explicitly in ``save_chat_to_disk`` itself).
    thread_call = fake_engine.backend.execute.call_args_list[0]
    _query, params = thread_call.args
    assert params.get("classification")
    assert params.get("tenant_id")


def test_save_chat_to_disk_fails_rather_than_write_unowned_when_minting_fails(
    monkeypatch,
):
    """If this process's system identity genuinely cannot be minted (e.g. a
    misconfigured external identity), the write must fail loudly -- it must
    NOT fall through to an unowned write. Nothing reaches the backend."""
    import agent_utilities.core.chat_persistence as cp
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", staticmethod(lambda: fake_engine)
    )

    from agent_utilities.security import request_identity as ri

    def _boom(*_a, **_kw):
        raise RuntimeError("identity acquisition unavailable")

    monkeypatch.setattr(ri, "local_process_authority_enabled", lambda _config: False)
    monkeypatch.setattr(ri, "acquire_process_identity_token", _boom)

    def isolated():
        with pytest.raises(RuntimeError):
            cp.save_chat_to_disk("chat-fail", [{"role": "user", "content": "hi"}])

    contextvars.Context().run(isolated)
    fake_engine.backend.execute.assert_not_called()
    fake_engine.add_node.assert_not_called()


# ---------------------------------------------------------------------------
# conversation_ingestion.ingest_conversations_to_kg
# ---------------------------------------------------------------------------


def test_ingest_conversations_binds_a_real_actor_with_no_ambient_session(monkeypatch):
    """Simulates a bare CLI/script invocation of multi-IDE chat-log
    ingestion with nothing ambient. Every Thread/Message write must observe
    a real, authenticated actor."""
    _force_local_process_authority(monkeypatch)
    from agent_utilities.knowledge_graph.core import conversation_ingestion as ci
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    captured_actors = []

    def _capture(*_a, **_kw):
        captured_actors.append(current_actor())
        return None

    fake_engine.add_node.side_effect = _capture
    fake_engine.link_nodes.side_effect = _capture
    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", staticmethod(lambda: fake_engine)
    )

    conversations = [
        {
            "id": "claude:conv-1",
            "source": "claude",
            "title": "test conversation",
            "timestamp": "2026-01-01T00:00:00",
            "messages": [{"role": "user", "content": "hello"}],
            "path": "/tmp/conv-1.jsonl",
        }
    ]

    def isolated():
        return ci.ingest_conversations_to_kg(
            conversations=conversations, extract_concepts=False
        )

    result = contextvars.Context().run(isolated)

    assert result["total_ingested"] == 1
    # One Thread add_node + one Message add_node + one link_nodes = 3 calls,
    # every one observing a real bound actor.
    assert len(captured_actors) == 3
    assert all(a.authenticated for a in captured_actors)
    assert all(a.actor_id for a in captured_actors)


def test_ingest_conversations_fails_rather_than_ingest_unowned_when_minting_fails(
    monkeypatch,
):
    from agent_utilities.knowledge_graph.core import conversation_ingestion as ci
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", staticmethod(lambda: fake_engine)
    )

    from agent_utilities.security import request_identity as ri

    def _boom(*_a, **_kw):
        raise RuntimeError("identity acquisition unavailable")

    monkeypatch.setattr(ri, "local_process_authority_enabled", lambda _config: False)
    monkeypatch.setattr(ri, "acquire_process_identity_token", _boom)

    conversations = [
        {
            "id": "claude:conv-2",
            "source": "claude",
            "title": "t",
            "timestamp": "2026-01-01T00:00:00",
            "messages": [{"role": "user", "content": "hello"}],
            "path": "/tmp/conv-2.jsonl",
        }
    ]

    def isolated():
        with pytest.raises(RuntimeError):
            ci.ingest_conversations_to_kg(
                conversations=conversations, extract_concepts=False
            )

    contextvars.Context().run(isolated)
    fake_engine.add_node.assert_not_called()


# ---------------------------------------------------------------------------
# messaging.enrichment.enrich_conversation
# ---------------------------------------------------------------------------


def test_enrich_conversation_binds_a_real_actor_with_no_ambient_session(monkeypatch):
    """Simulates a messaging co-service turn with nothing ambient (e.g. the
    context never got threaded through to this particular executor). The
    chat-turn Thread node write must observe a real, authenticated actor."""
    _force_local_process_authority(monkeypatch)
    from agent_utilities.messaging import enrichment as ez

    monkeypatch.setattr(ez, "_enabled", lambda: True)
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.cards.make_lite_llm_fn",
        lambda: (lambda _prompt: "{}"),
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.extractors.text.extract_text_concepts",
        lambda *_a, **_kw: ([], []),
    )
    monkeypatch.setattr(ez, "_surface_intents", lambda *_a, **_kw: None)

    fake_engine = MagicMock()
    captured_actors = []

    def _capture(*_a, **_kw):
        captured_actors.append(current_actor())
        return None

    fake_engine.add_node.side_effect = _capture
    fake_engine.link_nodes.side_effect = _capture

    def isolated():
        return ez.enrich_conversation(
            fake_engine, "hello world", platform="telegram", channel_id="chan-1"
        )

    contextvars.Context().run(isolated)

    # No concepts extracted, but the chat-turn Thread node write still
    # happens first and must be under a real actor.
    assert len(captured_actors) == 1
    assert captured_actors[0].authenticated is True
    assert captured_actors[0].actor_id


def test_enrich_conversation_no_actor_leak_when_minting_fails(monkeypatch):
    """A genuine minting failure must not silently skip enrichment with an
    unauthenticated identity; ``current_actor()`` must never observe a fake
    identity, and the caller sees a real exception (not a swallowed no-op)."""
    from agent_utilities.messaging import enrichment as ez

    monkeypatch.setattr(ez, "_enabled", lambda: True)

    fake_engine = MagicMock()

    from agent_utilities.security import request_identity as ri

    def _boom(*_a, **_kw):
        raise RuntimeError("identity acquisition unavailable")

    monkeypatch.setattr(ri, "local_process_authority_enabled", lambda _config: False)
    monkeypatch.setattr(ri, "acquire_process_identity_token", _boom)

    def isolated():
        with pytest.raises(RuntimeError):
            ez.enrich_conversation(
                fake_engine, "hello world", platform="telegram", channel_id="chan-2"
            )

    contextvars.Context().run(isolated)
    fake_engine.add_node.assert_not_called()


# ---------------------------------------------------------------------------
# Sanity: current_actor() genuinely raises with nothing bound (baseline for
# the "isolated contextvars.Context()" technique used throughout this file).
# ---------------------------------------------------------------------------


def test_isolated_context_has_no_ambient_actor_baseline():
    def isolated():
        with pytest.raises(IdentityRequiredError):
            current_actor()

    contextvars.Context().run(isolated)
