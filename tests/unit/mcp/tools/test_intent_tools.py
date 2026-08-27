"""CA-28: the `manage` lakehouse-maintenance status surface, and a regression
proof that a tool-hint pin can never bypass the `act`-verb approval gate.

The approval-gate positive/negative proof for a PINNED, non-destructive-but-
``approval_class``-gated tool under ``act`` already exists end to end in
``tests/unit/test_intent_surface.py::
test_human_approval_class_routes_to_exact_tool_even_when_not_destructive`` —
this module adds the CA-28-owned regression (a distinct fixture tool, kept in
CA-28's own test file per the lane's acceptance-gate command) rather than
duplicating that suite, plus new coverage for ``_manage_lifecycle``'s
``lakehouse_status`` action this lane adds.
"""

from __future__ import annotations

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import intent_tools
from tests.unit.test_intent_surface import _install_test_capability


@pytest.fixture(autouse=True)
def _fresh_candidate_cache():
    """Same reset contract as ``tests/unit/test_intent_surface.py`` — required
    whenever a test monkeypatches ``REGISTERED_TOOLS``/CPDs, so the resolver
    rebuilds its candidate table against the hermetic fixture rather than a
    stale cross-test cache."""
    kg_server.ensure_tools_registered()
    intent_tools._CANDIDATES_CACHE = None
    intent_tools._ACTIONS_BY_TOOL_CACHE = None
    intent_tools._OUTCOME_ROUTER = None
    intent_tools._REWARD_EPOCH = 0
    intent_tools._RESOLUTION_CACHE.clear()
    intent_tools._PREVIEW_PLAN_CACHE.clear()
    yield
    intent_tools._CANDIDATES_CACHE = None
    intent_tools._ACTIONS_BY_TOOL_CACHE = None
    intent_tools._OUTCOME_ROUTER = None
    intent_tools._REWARD_EPOCH = 0
    intent_tools._RESOLUTION_CACHE.clear()
    intent_tools._PREVIEW_PLAN_CACHE.clear()


# ── `manage(action="lakehouse_status")` — CA-28's read-only status surface ──


@pytest.mark.asyncio
async def test_lakehouse_status_index_rebuild_is_available_and_documents_how_to_call():
    """CA-24 already landed a REAL working index rebuild (`graph_ingest`
    `action=opensearch_reindex`); the status surface must point at it rather
    than re-implement it (design non-goal: 'not a new subsystem')."""
    result = await intent_tools._manage_lifecycle(
        mcp=None, intent="lakehouse status", hints={"action": "lakehouse_status"}
    )
    assert result is not None
    assert result["executed"] is True
    status = result["status"]
    assert status["index_rebuild"]["status"] == "available"
    assert status["index_rebuild"]["owner"] == "CA-24"
    assert "opensearch_reindex" in status["index_rebuild"]["how_to_call"]


@pytest.mark.asyncio
async def test_lakehouse_status_degrades_typed_not_raises_when_dependency_unlanded():
    """CA-21's Debezium-consumer lag function and CA-26's policy-bundle-epoch
    function have not landed yet — the status read must degrade to a typed
    'unavailable' entry, never raise, so a read-only status call can never
    fail just because a dependency lane is mid-flight."""
    result = await intent_tools._manage_lifecycle(
        mcp=None, intent="lakehouse status", hints={"action": "lakehouse_status"}
    )
    status = result["status"]
    assert status["cdc_lag"]["status"] == "unavailable"
    assert status["cdc_lag"]["owner"] == "CA-21"
    assert status["policy_bundle_epoch"]["status"] == "unavailable"
    assert status["policy_bundle_epoch"]["owner"] == "CA-26"


@pytest.mark.asyncio
async def test_lakehouse_status_is_read_only_and_needs_no_preview_plan_ref():
    """Unlike load/unload/reclaim (which require a preview -> plan_ref ->
    execute round trip), a pure status read must not require one — it never
    mutates anything, so ``execute`` defaulting to False must still return
    the real status, not a preview stub."""
    result = await intent_tools._manage_lifecycle(
        mcp=None,
        intent="lakehouse status",
        hints={"action": "lakehouse_status"},
        execute=False,
    )
    assert result["executed"] is True
    assert "plan" not in result


@pytest.mark.asyncio
async def test_manage_action_outside_status_or_reclaim_falls_through_to_resolver():
    """Unrecognized ``action`` hints must still fall through to the normal
    capability resolver (``None`` return), unchanged from before this lane."""
    result = await intent_tools._manage_lifecycle(
        mcp=None, intent="configure something", hints={"action": "not_a_real_action"}
    )
    assert result is None


# ── act-verb approval gate: a tool-hint pin never bypasses approval (CA-28-owned regression) ──


@pytest.mark.asyncio
async def test_ca28_pinned_act_on_approval_gated_tool_is_still_denied_without_approval(
    monkeypatch,
):
    """Acceptance gate #4: hints_json={"tool": "<requires-approval-tool>"}
    selects WHICH candidate, never bypasses ``approval_required`` — the SAME
    ``intent_tools.py`` gate (baseline §7) that governs every existing
    Action must govern any new CA-29-registered fleet Action identically."""
    called = False

    async def fake_lakehouse_action(**_kw) -> str:
        nonlocal called
        called = True
        return "ok"

    _install_test_capability(
        monkeypatch,
        "fake_lakehouse_trigger_action",
        fake_lakehouse_action,
        verbs=("act",),
        one_line="Trigger the synthetic CA-28 lakehouse-maintenance action.",
        mutates=True,
        idempotent=True,
        approval_class="human_approval_required",
    )
    hints = {"tool": "fake_lakehouse_trigger_action"}
    intent = "trigger the synthetic CA-28 lakehouse-maintenance action"

    preview = await intent_tools.dispatch_intent("act", intent, hints=hints)
    plan = preview["routing"]["plan"]
    assert plan["approval"]["required"] is True
    assert plan["approval"]["route"] == "exact_tool"

    result = await intent_tools.dispatch_intent(
        "act",
        intent,
        hints={**hints, "plan_ref": plan["plan_ref"]},
        execute=True,
    )
    assert result["executed"] is False
    assert result["approval_required"] is True
    assert result["required_load_tools"] == ["fake_lakehouse_trigger_action"]
    assert called is False
