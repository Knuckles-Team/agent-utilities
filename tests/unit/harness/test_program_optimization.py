"""Native program/policy optimization subsystem — metric, registry, driver, sweep.

Covers CONCEPT:AU-AHE.optimization.real-optimization-metric (metric), AHE-3.40
(registry/driver/dispatch), AHE-3.44 (extraction), AHE-3.45 (concept-match/routing),
AHE-3.46 (scheduled sweep + promotion gate). Few-shot demo refinement and program
compilation now run inside the native ``eg-program`` engine job (WS-I moved the DSPy
Python implementation onto the native path) — there is no Python-callable equivalent
left to unit-test; here we test the wiring, the real metric, and the self-supervised
metrics that remain in Python — all offline-deterministic.
"""

from __future__ import annotations

import logging

import pytest

from agent_utilities.harness.program_optimization import (
    OPTIMIZABLE_TARGETS,
    get_target,
    graded_score,
    make_optimization_metric,
    run_component_optimization,
)

# ── AHE-3.39 — the real metric ───────────────────────────────────────────────


def test_graded_score_is_graded_not_exact():
    assert graded_score(
        "the cat sat on the mat", "the cat sat on the mat"
    ) == pytest.approx(1.0)
    near = graded_score("the cat sat on the mat", "the cat sat on a mat")
    assert 0.0 < near < 1.0  # graded — exact-match would be 0
    assert graded_score("alpha beta", "totally different words") < near


def test_metric_bool_and_reward_blend():
    class E:
        response = "deploy the service to staging"

    class P:
        response = "deploy the service to staging"

    m_bool = make_optimization_metric(return_bool=True)
    assert m_bool(E(), P()) is True

    # reward blend pulls a perfect text score down toward a low reward
    m_blend = make_optimization_metric(reward_fn=lambda ex: 0.0, reward_weight=0.5)
    assert m_blend(E(), P()) == pytest.approx(0.5, abs=0.01)


# ── AHE-3.40 — target registry ───────────────────────────────────────────────


def test_registry_has_three_builtin_targets():
    assert set(OPTIMIZABLE_TARGETS) == {"system_prompt", "tool_description", "skill"}
    assert get_target("tool_description").kg_label == "EvolvedToolDescriptionNode"
    assert get_target("skill").load_text({"sop": "do X then Y"}) == "do X then Y"
    assert get_target("nonexistent") is None


def test_system_prompt_target_reads_blueprint_identity_and_instructions():
    t = get_target("system_prompt")
    text = t.load_text({"identity": {"role": "planner"}, "instructions": "be terse"})
    assert "planner" in text and "be terse" in text


# ── AHE-3.44 — self-supervised extraction metric ─────────────────────────────


def _onehot(text: str) -> list[float]:
    v = [0.0] * 4096
    v[hash((text or "").lower().strip()) % 4096] = 1.0
    return v


def test_extraction_quality_rewards_clean_over_messy():
    from agent_utilities.knowledge_graph.extraction.extraction_optimizer import (
        extraction_quality,
    )

    clean = [
        {"subject": "Acme", "predicate": "makes", "object": "Widgets"},
        {"subject": "Acme", "predicate": "located_in", "object": "Ohio"},
        {"subject": "Bob", "predicate": "works_at", "object": "Acme"},
    ]
    messy = [
        {"subject": "Acme", "predicate": "makes", "object": "Widgets"},
        {"subject": "Acme", "predicate": "makes", "object": "Widgets"},  # duplicate
        {"subject": "acme", "predicate": "located_in", "object": "Ohio"},  # fragmented
    ]
    cq = extraction_quality(clean, embed_fn=_onehot)
    mq = extraction_quality(messy, embed_fn=_onehot)
    assert cq["score"] == pytest.approx(1.0)
    assert mq["score"] < cq["score"]
    assert extraction_quality([], embed_fn=_onehot)["score"] == 0.0


def test_canonical_consistency_penalizes_fragmentation():
    from agent_utilities.knowledge_graph.extraction.extraction_optimizer import (
        canonical_consistency,
    )

    assert canonical_consistency([{"subject": "Acme", "object": "Ohio"}]) == 1.0
    frag = canonical_consistency(
        [{"subject": "Acme", "object": "x"}, {"subject": "acme", "object": "x"}]
    )
    assert frag < 1.0


# ── AHE-3.45 — policy metrics ────────────────────────────────────────────────


def test_classification_accuracy_and_routing_success():
    from agent_utilities.harness.policy_optimization import (
        classification_accuracy,
        routing_success_rate,
    )

    assert classification_accuracy(
        [True, False, True], [True, True, True]
    ) == pytest.approx(2 / 3)
    assert classification_accuracy([], []) == 0.0
    assert routing_success_rate(
        [{"success": True}, {"success": False}, {"success": True}]
    ) == pytest.approx(2 / 3)


# ── AHE-3.40 — the optimize-component surface dispatch ───────────────────────


def test_run_component_optimization_dispatch():
    assert "error" in run_component_optimization("bogus")
    # The native ``eg-program`` job is the sole backend now (no separate
    # registry-vs-self-supervised code path): with no engine/active native
    # authority, every real target fails closed the same uniform way.
    for target in ("system_prompt", "tool_description", "extraction", "routing"):
        report = run_component_optimization(target)
        assert report["target"] == target
        assert report["status"] == "error"
        assert report["error_code"] == "native_unavailable"


# ── AHE-3.46 — scheduled optimization sweep (the daemon-tick twin) ───────────


def test_should_promote_gate():
    from agent_utilities.harness.program_optimization import should_promote

    assert should_promote(0.7, 0.8) is True
    assert should_promote(0.7, 0.7) is True  # ties promote at min_delta=0
    assert should_promote(0.7, 0.72, min_delta=0.05) is False
    assert should_promote(0.7, 0.8, min_delta=0.05) is True


def test_gather_optimization_data_best_effort():
    from agent_utilities.harness.program_optimization import gather_optimization_data

    # no engine / no query_cypher → empty (degrades, never raises)
    assert gather_optimization_data(None, "extraction") == {}

    class FakeEngine:
        def query_cypher(self, cypher):
            if "Document" in cypher:
                return [{"content": "doc one"}, {"content": "doc two"}]
            if "Concept" in cypher:
                return [
                    {"concept": "RAG", "article": "retrieval augmented..."},
                    {"concept": "RLHF", "article": "reward modeling..."},
                ]
            if "ExecutionTrace" in cypher:
                return [{"task_text": "t", "primitive_used": "direct", "success": True}]
            return []

    eng = FakeEngine()
    assert len(gather_optimization_data(eng, "extraction")["documents"]) == 2
    pairs = gather_optimization_data(eng, "concept_match")["labeled_pairs"]
    assert any(rel for *_, rel in pairs) and any(not rel for *_, rel in pairs)
    assert len(gather_optimization_data(eng, "routing")["traces"]) == 1


def test_gather_degrades_when_query_raises():
    from agent_utilities.harness.program_optimization import gather_optimization_data

    class BadEngine:
        def query_cypher(self, cypher):
            raise RuntimeError("backend down")

    assert gather_optimization_data(BadEngine(), "extraction") == {"documents": []}


def test_run_optimization_sweep_is_propose_only():
    from agent_utilities.harness.program_optimization import (
        SCHEDULABLE_TARGETS,
        run_optimization_sweep,
    )

    rep = run_optimization_sweep(None)
    assert rep["propose_only"] is True
    assert set(rep["targets"]) == set(SCHEDULABLE_TARGETS)
    # no engine/data/LLM → nothing optimized, but the sweep completes cleanly
    assert rep["optimized"] == []


def test_daemon_tick_calls_sweep(monkeypatch):
    from agent_utilities.harness import program_optimization
    from agent_utilities.knowledge_graph.core import engine_tasks

    called = {}

    def fake_sweep(engine, targets=None):
        called["engine"] = engine
        return {"targets": {}, "optimized": ["extraction"], "propose_only": True}

    monkeypatch.setattr(program_optimization, "run_optimization_sweep", fake_sweep)

    sentinel = object()
    # invoke the unbound tick with a sentinel self — it must dispatch to the sweep
    engine_tasks.TaskManagerMixin._tick_optimize_components(sentinel)
    assert called["engine"] is sentinel


# ── U-103/U-135 — idle-vs-failure classification and bounded retry ──────────


class _MustNotCallEngine:
    """A native authority present (so the capability check passes) that fails the
    test outright if actually invoked — used to prove a no-data sweep never
    reaches the native optimizer."""

    def optimize_program(self, request):
        pytest.fail("native optimizer must not be invoked when there is no data")


class _FailingNativeEngine:
    """Has real governed data for ``extraction`` (via ``query_cypher``) but its
    native optimizer always raises — a transient native execution failure."""

    def __init__(self) -> None:
        self.calls = 0

    def query_cypher(self, cypher):
        if "Document" in cypher:
            return [{"content": "governed doc one"}]
        return []

    def optimize_program(self, request):
        self.calls += 1
        raise RuntimeError("transient native failure")


def test_run_component_optimization_treats_empty_corpus_as_idle():
    report = run_component_optimization("extraction", {}, engine=_MustNotCallEngine())

    assert report["status"] == "no_data"
    assert "error_code" not in report


def test_repeated_idle_sweep_produces_no_warnings_and_never_promotes(caplog):
    from agent_utilities.harness.program_optimization import (
        reset_target_backoff,
        run_optimization_sweep,
    )

    reset_target_backoff()
    caplog.set_level(logging.INFO)

    engine = _MustNotCallEngine()  # no query_cypher → every target degrades to {}
    for _ in range(5):
        report = run_optimization_sweep(engine)
        assert report["failed"] == []
        assert report["optimized"] == []  # no data ⇒ never a promoted change

    warnings_and_errors = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings_and_errors == []


def test_repeated_execution_failures_get_bounded_backoff_and_no_immediate_retry(
    monkeypatch,
):
    from agent_utilities.harness import program_optimization as po

    po.reset_target_backoff()
    fake_now = {"t": 0.0}
    monkeypatch.setattr(po.time, "monotonic", lambda: fake_now["t"])
    monkeypatch.setattr(po._JITTER_RNG, "uniform", lambda a, b: 0.0)  # deterministic

    engine = _FailingNativeEngine()

    # Tick 1 (t=0): the target is actually attempted and fails.
    report1 = po.run_optimization_sweep(engine, targets=["extraction"])
    assert report1["failed"] == ["extraction"]
    assert report1["optimized"] == []
    assert engine.calls == 1

    # Tick 2, same instant: bounded backoff must suppress the immediate retry —
    # zero additional native calls, and the target is not reported as a fresh
    # failure (no warning amplification for a deferred target).
    report2 = po.run_optimization_sweep(engine, targets=["extraction"])
    assert engine.calls == 1
    assert report2["failed"] == []
    assert report2.get("deferred") == ["extraction"]

    # Advance past the first backoff window — the retry is now allowed, and it
    # fails again, growing (bounded) backoff rather than resetting it.
    fake_now["t"] = po._BACKOFF_BASE_S
    report3 = po.run_optimization_sweep(engine, targets=["extraction"])
    assert engine.calls == 2
    assert report3["failed"] == ["extraction"]
    assert report3["optimized"] == []


def test_backoff_delay_is_bounded_and_jittered():
    from agent_utilities.harness.program_optimization import (
        _BACKOFF_MAX_S,
        _backoff_delay_s,
    )

    samples = {round(_backoff_delay_s(1), 6) for _ in range(20)}
    assert len(samples) > 1  # jitter varies the delay across calls
    assert all(0.0 <= value <= _BACKOFF_MAX_S * 1.5 for value in samples)

    # Growth saturates at the cap instead of growing unboundedly.
    huge = _backoff_delay_s(1000)
    assert huge <= _BACKOFF_MAX_S * 1.5
