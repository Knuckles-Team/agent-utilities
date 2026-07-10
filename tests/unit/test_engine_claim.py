"""``:AgentTask`` claim backend switch (CONCEPT:AU-OS.state.cognitive-scheduler-preemption, C3/Phase 3b, D13).

Covers ``AGENT_CLAIM_BACKEND`` resolution (kg default, engine opt-in, fail-safe on
an unrecognized value), the engine-native claim path's feature detection (a live
result is used exclusively; a degraded/unreachable one falls back to the KG path),
and — the load-bearing safety property — that a single claim NEVER exercises both
backends for the same task.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.orchestration import engine_claim

# ---------------------------------------------------------------------------
# resolve_claim_backend
# ---------------------------------------------------------------------------


def test_default_backend_is_kg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_CLAIM_BACKEND", raising=False)
    assert engine_claim.resolve_claim_backend() == engine_claim.AGENT_CLAIM_BACKEND_KG


def test_backend_resolves_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_CLAIM_BACKEND", "engine")
    assert (
        engine_claim.resolve_claim_backend() == engine_claim.AGENT_CLAIM_BACKEND_ENGINE
    )


def test_explicit_backend_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_CLAIM_BACKEND", "engine")
    assert (
        engine_claim.resolve_claim_backend("kg") == engine_claim.AGENT_CLAIM_BACKEND_KG
    )


def test_unknown_backend_value_fails_safe_to_kg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_CLAIM_BACKEND", "bogus")
    assert engine_claim.resolve_claim_backend() == engine_claim.AGENT_CLAIM_BACKEND_KG


# ---------------------------------------------------------------------------
# claim_agent_task — backend dispatch + never-both safety
# ---------------------------------------------------------------------------


def test_kg_backend_delegates_to_kg_claim_and_never_probes_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kg_calls: list[str] = []
    monkeypatch.setattr(
        engine_claim,
        "_claim_agent_task_kg",
        lambda engine, task_id, **kw: kg_calls.append(task_id)
        or {"task_id": task_id, "lease_id": "lease:kg:1"},
    )

    def _fail_if_engine_invoked(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("engine claim must never be probed on the kg backend")

    monkeypatch.setattr(engine_claim, "_try_engine_claim", _fail_if_engine_invoked)

    result = engine_claim.claim_agent_task(
        object(), "task-1", backend=engine_claim.AGENT_CLAIM_BACKEND_KG
    )
    assert result == {"task_id": "task-1", "lease_id": "lease:kg:1"}
    assert kg_calls == ["task-1"]


def test_engine_backend_uses_live_result_exclusively_never_touches_kg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_utilities.mcp.tools.engine_surface_tools as engine_surface_tools

    def fake_invoke(*, surface, action, graph, candidates, params):
        assert surface == "tasks"
        assert action == "claim_next"
        return json.dumps(
            {
                "surface": surface,
                "action": action,
                "result": {
                    "claimed": True,
                    "lease_id": "lease:engine:1",
                    "dag_id": "dag-1",
                    "checkpoint_id": None,
                    "depends_on_task_ids": [],
                },
            }
        )

    monkeypatch.setattr(engine_surface_tools, "_invoke", fake_invoke)

    def _fail_if_kg_invoked(*a, **k):  # pragma: no cover - must never run
        raise AssertionError(
            "kg claim must never be attempted when the engine claim is live"
        )

    monkeypatch.setattr(engine_claim, "_claim_agent_task_kg", _fail_if_kg_invoked)

    result = engine_claim.claim_agent_task(
        object(), "task-1", backend=engine_claim.AGENT_CLAIM_BACKEND_ENGINE
    )
    assert result == {
        "task_id": "task-1",
        "lease_id": "lease:engine:1",
        "dag_id": "dag-1",
        "checkpoint_id": None,
        "depends_on_task_ids": [],
    }


def test_engine_backend_falls_back_to_kg_when_engine_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_utilities.mcp.tools.engine_surface_tools as engine_surface_tools

    def fake_invoke(*, surface, action, graph, candidates, params):
        return json.dumps(
            {
                "surface": surface,
                "action": action,
                "degraded": True,
                "error": "no matching client method",
            }
        )

    monkeypatch.setattr(engine_surface_tools, "_invoke", fake_invoke)

    kg_calls: list[str] = []
    monkeypatch.setattr(
        engine_claim,
        "_claim_agent_task_kg",
        lambda engine, task_id, **kw: kg_calls.append(task_id)
        or {"task_id": task_id, "lease_id": "lease:kg:fallback"},
    )

    result = engine_claim.claim_agent_task(
        object(), "task-1", backend=engine_claim.AGENT_CLAIM_BACKEND_ENGINE
    )
    assert result == {"task_id": "task-1", "lease_id": "lease:kg:fallback"}
    # Exactly ONE kg claim attempt — never both backends racing for the same task.
    assert kg_calls == ["task-1"]


def test_engine_backend_falls_back_to_kg_when_engine_says_not_claimed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine surface IS live but explicitly reports 'not claimed' (e.g. a
    live lease elsewhere) — still exactly one path taken, never a double claim."""
    import agent_utilities.mcp.tools.engine_surface_tools as engine_surface_tools

    monkeypatch.setattr(
        engine_surface_tools,
        "_invoke",
        lambda **kw: json.dumps(
            {"surface": "tasks", "action": "claim_next", "result": {"claimed": False}}
        ),
    )
    kg_calls: list[str] = []
    monkeypatch.setattr(
        engine_claim,
        "_claim_agent_task_kg",
        lambda engine, task_id, **kw: kg_calls.append(task_id) or None,
    )

    result = engine_claim.claim_agent_task(
        object(), "task-1", backend=engine_claim.AGENT_CLAIM_BACKEND_ENGINE
    )
    assert result is None
    assert kg_calls == ["task-1"]


def test_never_run_both_backends_on_one_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Combined safety assertion: whichever backend is selected, the OTHER
    backend's underlying primitive is invoked at most zero times per claim."""
    import agent_utilities.mcp.tools.engine_surface_tools as engine_surface_tools

    engine_invocations = {"n": 0}
    kg_invocations = {"n": 0}

    def fake_invoke(**kw):
        engine_invocations["n"] += 1
        return json.dumps(
            {
                "surface": "tasks",
                "action": "claim_next",
                "result": {"claimed": True, "lease_id": "lease:engine:1"},
            }
        )

    def fake_kg(engine, task_id, **kw):
        kg_invocations["n"] += 1
        return {"task_id": task_id, "lease_id": "lease:kg:1"}

    monkeypatch.setattr(engine_surface_tools, "_invoke", fake_invoke)
    monkeypatch.setattr(engine_claim, "_claim_agent_task_kg", fake_kg)

    # engine backend, live result — engine wins, kg untouched.
    engine_claim.claim_agent_task(
        object(), "task-1", backend=engine_claim.AGENT_CLAIM_BACKEND_ENGINE
    )
    assert engine_invocations["n"] == 1
    assert kg_invocations["n"] == 0

    # kg backend — kg wins, engine untouched.
    engine_claim.claim_agent_task(
        object(), "task-2", backend=engine_claim.AGENT_CLAIM_BACKEND_KG
    )
    assert engine_invocations["n"] == 1  # unchanged
    assert kg_invocations["n"] == 1
