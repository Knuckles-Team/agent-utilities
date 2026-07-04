"""Logprob-weighted G-Eval (CONCEPT:AU-AHE.harness.ahe-2) — CI-safe with a mocked endpoint.

Asserts the probability-weighted score math over the score token's top-logprobs and that
the chain-of-thought rubric is generated once and cached. Live discrimination
(good>>bad, continuous) is validated separately against vLLM.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from agent_utilities.harness import g_eval as ge


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
    score, reason = ge.GEval("t", "c").score("q", "a")
    assert abs(score - 0.92) < 0.02
    assert "logprob-weighted" in reason


def test_rubric_is_cached(monkeypatch):
    client = _FakeClient(_logprob_choice({"3": math.log(0.99)}, content="3"))
    monkeypatch.setattr(ge, "_live_endpoint", lambda: (client, "fake-model"))
    ge._rubric.cache_clear()
    g = ge.GEval("task-x", "criteria-y")
    g.score("q1", "a1")
    n_after_first = client.calls
    g.score("q2", "a2")
    # second score reuses the cached rubric → only ONE extra call (the score call), not two.
    assert client.calls == n_after_first + 1


def test_degrades_without_endpoint(monkeypatch):
    monkeypatch.setattr(ge, "_live_endpoint", lambda: None)
    score, reason = ge.GEval("t", "c").score("q", "a")
    assert score == 0.0 and "unavailable" in reason
