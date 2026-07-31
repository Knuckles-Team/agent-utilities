"""Logprob-weighted G-Eval (CONCEPT:AU-AHE.harness.ahe-2) — CI-safe with a mocked endpoint.

Asserts the probability-weighted score math over the score token's top-logprobs and that
the chain-of-thought rubric is generated once and cached. Live discrimination
(good>>bad, continuous) is validated separately against vLLM.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from agent_utilities.core.contextual_model import (
    _EmptyEvidenceSource,
    use_context_compiler_engine,
)
from agent_utilities.harness import g_eval as ge
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


def _verified_session() -> GraphSession:
    """A minimal verified ambient GraphSession -- compiled_chat_completion (which
    _complete routes every call through) requires one; see
    tests/unit/test_graph_session.py for the established construction pattern."""
    actor = ActorContext(
        actor_id="principal:g-eval-test",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:read",),
        tenant_id="tenant-a",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant="tenant-a",
        scopes=frozenset({"kg:read"}),
        graph="tenant-a-graph",
        policy_version="policy-1",
        audience="agent-services",
    )


def _logprob_choice(top: dict[str, float], content: str = ""):
    """Build a fake chat-completion choice with one score token + top_logprobs."""
    cands = [SimpleNamespace(token=t, logprob=lp) for t, lp in top.items()]
    tok = SimpleNamespace(top_logprobs=cands)
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        logprobs=SimpleNamespace(content=[tok]),
    )


class _FakeClient:
    def __init__(self, choice):
        self._choice = choice
        self.calls = 0

        class _Chat:
            def __init__(self, outer):
                self.completions = SimpleNamespace(create=outer._create)

        self.chat = _Chat(self)

    def _create(self, **kw):
        self.calls += 1
        # rubric call (no logprobs requested) returns text; score call returns the choice.
        if not kw.get("logprobs"):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="1. step"))]
            )
        return SimpleNamespace(choices=[self._choice])


def test_logprob_weighted_score(monkeypatch):
    # token "5" p≈0.9, "1" p≈0.1 → weighted ≈ (5*0.9 + 1*0.1)/1.0 = 4.6 → /5 = 0.92
    lp5, lp1 = math.log(0.9), math.log(0.1)
    client = _FakeClient(_logprob_choice({"5": lp5, "1": lp1}, content="5"))
    monkeypatch.setattr(ge, "_live_endpoint", lambda: (client, "fake-model"))
    ge._rubric.cache_clear()
    # _complete routes every call through compiled_chat_completion, which requires
    # a verified ambient GraphSession (agent_utilities/knowledge_graph/core/session.py)
    # AND a configured ContextCompiler engine (graph_session_required() is an
    # always-True invariant, not a feature switch -- compile_model_context fails
    # closed with no engine at all rather than silently degrading).
    with (
        use_session(_verified_session()),
        use_context_compiler_engine(_EmptyEvidenceSource()),
    ):
        score, reason = ge.GEval("t", "c").score("q", "a")
    assert abs(score - 0.92) < 0.02
    assert "logprob-weighted" in reason


def test_rubric_is_cached(monkeypatch):
    client = _FakeClient(_logprob_choice({"3": math.log(0.99)}, content="3"))
    monkeypatch.setattr(ge, "_live_endpoint", lambda: (client, "fake-model"))
    ge._rubric.cache_clear()
    g = ge.GEval("task-x", "criteria-y")
    with (
        use_session(_verified_session()),
        use_context_compiler_engine(_EmptyEvidenceSource()),
    ):
        g.score("q1", "a1")
        n_after_first = client.calls
        g.score("q2", "a2")
    # second score reuses the cached rubric → only ONE extra call (the score call), not two.
    assert client.calls == n_after_first + 1


def test_degrades_without_endpoint(monkeypatch):
    monkeypatch.setattr(ge, "_live_endpoint", lambda: None)
    score, reason = ge.GEval("t", "c").score("q", "a")
    assert score == 0.0 and "unavailable" in reason
