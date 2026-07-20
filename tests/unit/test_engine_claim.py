"""``:AgentTask`` claim backend resolution (CONCEPT:AU-OS.state.cognitive-scheduler-preemption, C3/Phase 3b, D13).

Covers ``AGENT_CLAIM_BACKEND`` resolution (``workitem`` is the sole backend and
default; fail-safe on an unrecognized value) and that :func:`claim_agent_task`
routes every claim through the WorkItem bridge
(:func:`~agent_utilities.orchestration.work_item.claim_agent_task_via_work_item`).

No-Legacy history (report §2 Claims seam / §9 #4): the ``kg`` KG-``:AgentLease``
backend and the ``engine`` namespace-probing backend (``_CLAIM_NEXT_CANDIDATES``,
``_try_engine_claim``) are both deleted, not shimmed — see ``engine_claim.py``'s
module docstring for why. This file no longer exercises either.
"""

from __future__ import annotations

import pytest

from agent_utilities.orchestration import engine_claim

# ---------------------------------------------------------------------------
# resolve_claim_backend
# ---------------------------------------------------------------------------


def test_default_backend_is_workitem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_CLAIM_BACKEND", raising=False)
    assert (
        engine_claim.resolve_claim_backend()
        == engine_claim.AGENT_CLAIM_BACKEND_WORKITEM
    )


def test_backend_resolves_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_CLAIM_BACKEND", "workitem")
    assert (
        engine_claim.resolve_claim_backend()
        == engine_claim.AGENT_CLAIM_BACKEND_WORKITEM
    )


def test_explicit_backend_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_CLAIM_BACKEND", "bogus")
    assert (
        engine_claim.resolve_claim_backend("workitem")
        == engine_claim.AGENT_CLAIM_BACKEND_WORKITEM
    )


def test_unknown_backend_value_fails_safe_to_workitem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_CLAIM_BACKEND", "kg")  # the retired backend name
    assert (
        engine_claim.resolve_claim_backend()
        == engine_claim.AGENT_CLAIM_BACKEND_WORKITEM
    )


# ---------------------------------------------------------------------------
# claim_agent_task — routes to the WorkItem bridge, nothing else
# ---------------------------------------------------------------------------


def test_claim_agent_task_delegates_to_work_item_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.orchestration import work_item

    bridge_calls: list[dict] = []
    monkeypatch.setattr(
        work_item,
        "claim_agent_task_via_work_item",
        lambda engine, task_id, **kw: (
            bridge_calls.append({"task_id": task_id, **kw})
            or {"task_id": task_id, "lease_id": "lease:workitem:1", "fence_token": 1}
        ),
    )

    result = engine_claim.claim_agent_task(object(), "task-1")
    assert result == {
        "task_id": "task-1",
        "lease_id": "lease:workitem:1",
        "fence_token": 1,
    }
    assert len(bridge_calls) == 1
    assert bridge_calls[0]["task_id"] == "task-1"
    assert bridge_calls[0]["claim_ttl_s"] == work_item.DEFAULT_LEASE_TTL_S


def test_claim_agent_task_honors_explicit_token_now_and_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.orchestration import work_item

    bridge_calls: list[dict] = []
    monkeypatch.setattr(
        work_item,
        "claim_agent_task_via_work_item",
        lambda engine, task_id, **kw: bridge_calls.append(kw) or None,
    )

    engine_claim.claim_agent_task(
        object(), "task-2", token="hostA:1", now=1000.0, claim_ttl_s=60.0
    )
    assert bridge_calls == [{"token": "hostA:1", "now": 1000.0, "claim_ttl_s": 60.0}]


def test_claim_agent_task_unrecognized_backend_override_still_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bogus explicit ``backend=`` never silently disables claiming — it's
    logged and the sole (workitem) backend is used anyway."""
    from agent_utilities.orchestration import work_item

    monkeypatch.setattr(
        work_item,
        "claim_agent_task_via_work_item",
        lambda engine, task_id, **kw: {"task_id": task_id, "lease_id": "lease:1"},
    )

    result = engine_claim.claim_agent_task(object(), "task-3", backend="engine")
    assert result == {"task_id": "task-3", "lease_id": "lease:1"}
