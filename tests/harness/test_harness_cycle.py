"""The propose–probe–compose–persist–reflect harness evolution cycle."""

from __future__ import annotations

import dataclasses

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from agent_utilities import host_ports as hp
from agent_utilities.graph import RunBudget, RunPlan, RunRequest
from agent_utilities.harness.harness_cycle import (
    ABSTAIN_REASONS,
    ContrastAnswer,
    ContrastDelta,
    ContrastRequest,
    CycleOutcome,
    CycleRefusal,
    HarnessCandidate,
    ProbeTrace,
    compose,
    conditions_digest,
    measure_contrasts,
    persist,
    probe_candidates,
    propose,
    reflect,
    run_cycle_tick,
)


def _candidate() -> HarnessCandidate:
    return HarnessCandidate(
        candidate_id="c-1",
        task_ref="task-1",
        component_kind="system_prompt",
        component_version_ref="eg:component/system_prompt/3",
        seed="seed-1",
        model_ref="model-1",
        budget_ref="budget-1",
    )


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.1")
def test_model_round_trips() -> None:
    candidate = _candidate()
    assert HarnessCandidate(**dataclasses.asdict(candidate)) == candidate

    outcome = CycleOutcome(
        task_ref="task-1",
        k=2,
        adopted_version_ref=None,
        abstain_reason="budget_below_contrast",
        contrast_ref=None,
        lesson_ref=None,
    )
    assert CycleOutcome(**dataclasses.asdict(outcome)) == outcome

    adopted = CycleOutcome(
        task_ref="task-1",
        k=2,
        adopted_version_ref="eg:component/system_prompt/3",
        abstain_reason=None,
        contrast_ref="eg:contrast/1",
        lesson_ref="eg:lesson/1",
    )
    assert CycleOutcome(**dataclasses.asdict(adopted)) == adopted

    for code in ABSTAIN_REASONS:
        abstained = CycleOutcome(
            task_ref="task-1",
            k=1,
            adopted_version_ref=None,
            abstain_reason=code,
            contrast_ref=None,
            lesson_ref=None,
        )
        assert CycleOutcome(**dataclasses.asdict(abstained)) == abstained

    with pytest.raises(ValueError, match="open refusal code"):
        CycleOutcome(
            task_ref="task-1",
            k=2,
            adopted_version_ref=None,
            abstain_reason="because",
            contrast_ref=None,
            lesson_ref=None,
        )
    with pytest.raises(ValueError, match="adopts nothing"):
        CycleOutcome(
            task_ref="task-1",
            k=2,
            adopted_version_ref="eg:component/system_prompt/3",
            abstain_reason="compose_abstention",
            contrast_ref=None,
            lesson_ref=None,
        )


def _probe_candidates() -> list[HarnessCandidate]:
    return [
        HarnessCandidate(
            candidate_id="c-0",
            task_ref="task-1",
            component_kind="system_prompt",
            component_version_ref="v-a",
            seed="seed-1",
            model_ref="model-1",
            budget_ref="budget-1",
        ),
        HarnessCandidate(
            candidate_id="c-1",
            task_ref="task-1",
            component_kind="guardrails",
            component_version_ref="v-b",
            seed="seed-1",
            model_ref="model-1",
            budget_ref="budget-1",
        ),
    ]


@pytest.mark.parametrize("field", ["seed", "model_ref", "budget_ref"])
async def test_probe_rejects_inconsistent_candidates_before_any_run(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unexpected_run(*args, **kwargs):
        raise AssertionError("a mismatched candidate must not start a probe")

    monkeypatch.setattr(
        "agent_utilities.harness.harness_cycle.execute_graph", unexpected_run
    )
    candidates = _probe_candidates()
    candidates[1] = dataclasses.replace(candidates[1], **{field: "different"})

    with pytest.raises(CycleRefusal, match="probe_trace_missing"):
        await probe_candidates(
            graph=object(),
            config=None,
            task="classify",
            seed="seed-1",
            model_ref="model-1",
            budget_ref="budget-1",
            run_id="run-1",
            current_version_ref="v-0",
            candidates=candidates,
        )


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.3")
async def test_probe_shares_conditions_and_records_trace() -> None:
    def model(messages, info):
        return ModelResponse(parts=[TextPart("done")])

    hp.register_run_decision_provider(
        lambda request: RunPlan(model=FunctionModel(model), steps=(request.task,))
    )

    traces = await probe_candidates(
        graph=object(),
        config=None,
        task="classify",
        seed="seed-1",
        model_ref="model-1",
        budget_ref="budget-1",
        run_id="run-1",
        current_version_ref="",
        candidates=_probe_candidates(),
    )

    assert [trace.candidate_id for trace in traces] == ["c-0", "c-1"]
    assert [trace.trace_id for trace in traces] == [
        "run-1:probe:c-0",
        "run-1:probe:c-1",
    ]
    assert {trace.conditions_digest for trace in traces} == {
        conditions_digest(
            task="classify", seed="seed-1", model_ref="model-1", budget_ref="budget-1"
        )
    }
    for trace in traces:
        assert trace.status == "completed"
        assert trace.error is None
        assert trace.steps == 1


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.3")
async def test_probe_records_current_harness_baseline() -> None:
    def model(messages, info):
        return ModelResponse(parts=[TextPart("done")])

    hp.register_run_decision_provider(
        lambda request: RunPlan(model=FunctionModel(model), steps=(request.task,))
    )

    traces = await probe_candidates(
        graph=object(),
        config=None,
        task="classify",
        seed="seed-1",
        model_ref="model-1",
        budget_ref="budget-1",
        run_id="run-1",
        current_version_ref="v-0",
        candidates=_probe_candidates(),
    )

    assert [trace.candidate_id for trace in traces] == [
        "current",
        "c-0",
        "c-1",
    ]
    assert traces[0].trace_id == "run-1:probe:current"
    assert {trace.conditions_digest for trace in traces} == {
        conditions_digest(
            task="classify", seed="seed-1", model_ref="model-1", budget_ref="budget-1"
        )
    }


def _trace(
    candidate_id: str, *, status: str = "completed", tool_calls: int = 0
) -> ProbeTrace:
    return ProbeTrace(
        candidate_id=candidate_id,
        trace_id=f"run-1:probe:{candidate_id}",
        conditions_digest="digest-1",
        status=status,
        steps=1,
        usage={"requests": 1, "tool_calls": tool_calls, "total_tokens": 10},
        error=None if status == "completed" else "boom",
    )


def _fixture_contrasts(*, baseline, candidates, traces):
    """Synthetic measurements for isolated downstream orchestration tests."""
    return tuple(
        ContrastDelta(
            candidate_id=candidate.candidate_id,
            run_a_trace_id=baseline.trace_id,
            run_b_trace_id=traces[candidate.candidate_id].trace_id,
            retry_delta=0,
            tool_selection_delta=0,
            eval_score_delta=0,
            cost_delta=0,
        )
        for candidate in candidates
    )


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.12")
@pytest.mark.parametrize("has_baseline", [False, True])
async def test_compose_refuses_unverified_measurements_before_decide(
    has_baseline,
) -> None:
    candidates = _probe_candidates()
    traces = {
        candidate.candidate_id: _trace(candidate.candidate_id, tool_calls=100)
        for candidate in candidates
    }
    baseline = _trace("current") if has_baseline else None
    calls: list[ContrastRequest] = []
    with pytest.raises(CycleRefusal, match="measurement_unavailable"):
        measure_contrasts(baseline=baseline, candidates=candidates, traces=traces)
    with pytest.raises(CycleRefusal, match="measurement_unavailable"):
        await compose(
            task_ref="task-1",
            task_class="classification",
            conditions_digest_value="digest-1",
            candidates=candidates,
            baseline=baseline,
            traces=traces,
            decide=calls.append,
        )
    assert calls == []


async def test_compose_delegates_to_engine(monkeypatch) -> None:
    import agent_utilities.harness.harness_cycle as cycle

    monkeypatch.setattr(cycle, "measure_contrasts", _fixture_contrasts)
    candidates = _probe_candidates()
    baseline = _trace("current")
    traces = {
        candidate.candidate_id: _trace(candidate.candidate_id)
        for candidate in candidates
    }
    calls: list[ContrastRequest] = []

    def decide(request: ContrastRequest):
        calls.append(request)
        return {
            "adopted_version_ref": candidates[1].component_version_ref,
            "abstain_reason": None,
            "adopted_component_kind": "guardrails",
            "adopted_content": "new guardrails",
        }

    answer = await compose(
        task_ref="task-1",
        task_class="classification",
        conditions_digest_value="digest-1",
        candidates=candidates,
        baseline=baseline,
        traces=traces,
        decide=decide,
    )

    assert len(calls) == 1
    request = calls[0]
    assert request.task_ref == "task-1"
    assert request.task_class == "classification"
    assert request.conditions_digest == "digest-1"
    # Passed through in candidate order; the cycle ranks nothing.
    assert request.candidate_version_refs == (
        candidates[0].component_version_ref,
        candidates[1].component_version_ref,
    )
    assert request.probe_trace_ids == (
        "run-1:probe:c-0",
        "run-1:probe:c-1",
    )
    assert [delta.candidate_id for delta in request.contrasts] == ["c-0", "c-1"]
    for delta in request.contrasts:
        assert delta.run_a_trace_id == baseline.trace_id
        assert delta.run_b_trace_id in request.probe_trace_ids
    # The engine's answer stands, even for a later candidate.
    assert answer.adopted_version_ref == candidates[1].component_version_ref
    assert answer.abstain_reason is None


async def test_compose_abstains_when_engine_abstains(monkeypatch) -> None:
    import agent_utilities.harness.harness_cycle as cycle

    monkeypatch.setattr(cycle, "measure_contrasts", _fixture_contrasts)
    candidates = _probe_candidates()
    calls: list[ContrastRequest] = []

    async def decide(request: ContrastRequest):
        calls.append(request)
        return {"adopted_version_ref": None, "abstain_reason": "no_improvement"}

    with pytest.raises(CycleRefusal, match="compose_abstention"):
        await compose(
            task_ref="task-1",
            task_class="classification",
            conditions_digest_value="digest-1",
            candidates=candidates,
            baseline=_trace("current"),
            traces={
                candidate.candidate_id: _trace(candidate.candidate_id)
                for candidate in candidates
            },
            decide=decide,
        )
    assert len(calls) == 1


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.12")
@pytest.mark.parametrize("baseline_trace", ["", "baseline"])
def test_persist_refuses_unverified_contrast_before_writes_or_switch(
    tmp_path, baseline_trace
) -> None:
    component = tmp_path / "harness" / "system_prompt.md"
    component.parent.mkdir(parents=True)
    component.write_text("current prompt", encoding="utf-8")
    writes: list[object] = []
    contrast = ContrastDelta(
        candidate_id="c-0",
        run_a_trace_id=baseline_trace,
        run_b_trace_id="candidate",
        retry_delta=-1,
        tool_selection_delta=1,
        eval_score_delta=1,
        cost_delta=-1,
    )
    answer = ContrastAnswer(
        adopted_version_ref="v-a",
        abstain_reason=None,
        adopted_component_kind="system_prompt",
        adopted_content="changed",
    )
    with pytest.raises(CycleRefusal, match="measurement_unavailable"):
        persist(
            task_ref="task-1",
            task_class="classification",
            conditions_digest_value="digest-1",
            adopted=_probe_candidates()[0],
            contrast=contrast,
            answer=answer,
            write_envelope=writes.append,
            root=tmp_path,
        )
    assert writes == []
    assert component.read_text(encoding="utf-8") == "current prompt"


def _library() -> list[dict[str, str]]:
    return [
        {"component_kind": "system_prompt", "component_version_ref": "v-a"},
        {"component_kind": "guardrails", "component_version_ref": "v-b"},
    ]


def _tick_request() -> RunRequest:
    return RunRequest(
        task="classify",
        run_id="task-1",
        metadata={
            "harness_seed": "seed-1",
            "harness_current_version_ref": "v-0",
        },
    )


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.12")
@pytest.mark.parametrize("has_baseline", [False, True])
async def test_tick_refuses_without_owner_measurements_before_decide_or_persist(
    tmp_path, events, has_baseline
) -> None:
    def model(messages, info):
        return ModelResponse(parts=[TextPart("done")])

    hp.register_run_decision_provider(
        lambda request: RunPlan(
            model=FunctionModel(model),
            steps=(request.task,),
            budget=RunBudget(candidate_slots=2),
        )
    )
    metadata = {"harness_seed": "seed-1"}
    if has_baseline:
        metadata["harness_current_version_ref"] = "v-0"
    request = RunRequest(task="classify", run_id="task-1", metadata=metadata)
    calls: list[object] = []
    writes: list[object] = []
    component = tmp_path / "harness" / "system_prompt.md"
    component.parent.mkdir(parents=True)
    component.write_text("current prompt", encoding="utf-8")

    outcome = await run_cycle_tick(
        request,
        RunPlan(model=FunctionModel(model), budget=RunBudget(candidate_slots=2)),
        task_class="classification",
        graph=object(),
        library_query=lambda _class: _library(),
        decide=calls.append,
        write_envelope=writes.append,
        root=tmp_path,
    )
    assert outcome.abstain_reason == "measurement_unavailable"
    assert outcome.adopted_version_ref is None
    assert outcome.contrast_ref is None and outcome.lesson_ref is None
    assert calls == [] and writes == []
    assert component.read_text(encoding="utf-8") == "current prompt"
    cycle_events = [
        event for event in events if event.get("type") == "harness_cycle_outcome"
    ]
    assert len(cycle_events) == 1
    assert cycle_events[0]["abstain_reason"] == "measurement_unavailable"
    assert cycle_events[0]["evidence_run_ids"] == []


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.6")
def test_reflect_reports_adoption_or_abstention(events) -> None:
    adopted = CycleOutcome(
        task_ref="task-1",
        k=2,
        adopted_version_ref="v-a",
        abstain_reason=None,
        contrast_ref="ref-contrast",
        lesson_ref="ref-lesson",
    )
    reflect(adopted, evidence_run_ids=("baseline", "candidate"))
    assert events[-1]["lesson_ref"] == "ref-lesson"
    assert events[-1]["evidence_run_ids"] == ["baseline", "candidate"]
    abstained = CycleOutcome(
        task_ref="task-1",
        k=2,
        adopted_version_ref=None,
        abstain_reason="measurement_unavailable",
        contrast_ref=None,
        lesson_ref=None,
    )
    reflect(abstained, evidence_run_ids=("baseline", "candidate"))
    assert events[-1]["lesson_ref"] is None
    assert events[-1]["evidence_run_ids"] == []


async def test_refusals_keep_harness(tmp_path, monkeypatch) -> None:
    import agent_utilities.harness.harness_cycle as cycle

    def model(messages, info):
        return ModelResponse(parts=[TextPart("done")])

    def plan(candidate_slots):
        return RunPlan(
            model=FunctionModel(model),
            steps=("task",),
            budget=RunBudget(candidate_slots=candidate_slots),
        )

    def registered_provider(request):
        return RunPlan(
            model=FunctionModel(model),
            steps=(request.task,),
            budget=RunBudget(candidate_slots=2),
        )

    hp.register_run_decision_provider(registered_provider)

    component = tmp_path / "harness" / "system_prompt.md"
    component.parent.mkdir(parents=True)
    component.write_text("current prompt", encoding="utf-8")
    written: list[str] = []
    queried: list[str] = []

    def library(task_class):
        queried.append(task_class)
        return _library()

    def write_envelope(envelope):
        written.append(str(envelope["kind"]))
        return f"ref-{envelope['kind']}"

    def abstain(request):
        return {"adopted_version_ref": None, "abstain_reason": "no_improvement"}

    # budget_below_contrast: an allocation under two candidate slots.
    for slots in (1, 0, None):
        outcome = await run_cycle_tick(
            _tick_request(),
            plan(slots),
            task_class="classification",
            graph=object(),
            library_query=library,
            decide=lambda request: {
                "adopted_version_ref": "v-a",
                "abstain_reason": None,
                "adopted_component_kind": "system_prompt",
                "adopted_content": "adopted prompt",
            },
            write_envelope=write_envelope,
            root=tmp_path,
        )
        assert outcome.abstain_reason == "budget_below_contrast"
        assert outcome.k == (slots or 0)
        assert outcome.adopted_version_ref is None
        assert outcome.contrast_ref is None and outcome.lesson_ref is None
    assert queried == [] and written == []

    # no_candidates: an empty library answer.
    outcome = await run_cycle_tick(
        _tick_request(),
        plan(2),
        task_class="classification",
        graph=object(),
        library_query=lambda _class: [],
        decide=lambda request: {
            "adopted_version_ref": "v-a",
            "abstain_reason": None,
            "adopted_component_kind": "system_prompt",
            "adopted_content": "adopted prompt",
        },
        write_envelope=write_envelope,
        root=tmp_path,
    )
    assert outcome.abstain_reason == "no_candidates"
    assert outcome.k == 2
    assert written == []

    # probe_trace_missing: a lost probe trace.
    async def lost_probe(*args, **kwargs):
        raise RuntimeError("trace lost")

    monkeypatch.setattr(cycle, "execute_graph", lost_probe)
    outcome = await run_cycle_tick(
        _tick_request(),
        plan(2),
        task_class="classification",
        graph=object(),
        library_query=library,
        decide=lambda request: {
            "adopted_version_ref": "v-a",
            "abstain_reason": None,
            "adopted_component_kind": "system_prompt",
            "adopted_content": "adopted prompt",
        },
        write_envelope=write_envelope,
        root=tmp_path,
    )
    assert outcome.abstain_reason == "probe_trace_missing"
    assert written == []

    # compose_abstention: the engine abstains, and no adoption persists.
    monkeypatch.undo()
    monkeypatch.setattr(cycle, "measure_contrasts", _fixture_contrasts)
    outcome = await run_cycle_tick(
        _tick_request(),
        plan(2),
        task_class="classification",
        graph=object(),
        library_query=library,
        decide=abstain,
        write_envelope=write_envelope,
        root=tmp_path,
    )
    assert outcome.abstain_reason == "compose_abstention"
    assert outcome.adopted_version_ref is None
    assert written == []

    # Every refusal kept the current harness.
    assert component.read_text(encoding="utf-8") == "current prompt"


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.8")
async def test_port_fails_closed_without_provider() -> None:
    from agent_utilities.harness.harness_cycle import drive_cycle_tick

    with pytest.raises(RuntimeError, match="arbora"):
        await drive_cycle_tick(
            RunRequest(task="classify"),
            RunPlan(model="m-1"),
            task_class="classification",
            graph=object(),
        )


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.8")
async def test_drive_delegates_to_registered_provider() -> None:
    from agent_utilities.harness.harness_cycle import drive_cycle_tick

    outcome = CycleOutcome(
        task_ref="task-1",
        k=2,
        adopted_version_ref="v-a",
        abstain_reason=None,
        contrast_ref="ref-harness_contrast",
        lesson_ref="ref-harness_lesson",
    )
    seen: list[tuple[object, object]] = []

    async def provider(request, plan, **kwargs):
        seen.append((request, kwargs))
        return outcome

    hp.register_harness_cycle_provider(provider)
    assert (
        await drive_cycle_tick(
            RunRequest(task="classify"),
            RunPlan(model="m-1"),
            task_class="classification",
            graph=object(),
        )
        is outcome
    )
    assert seen[0][0].task == "classify"
    assert seen[0][1]["task_class"] == "classification"

    hp.register_harness_cycle_provider(lambda *args, **kwargs: "not an outcome")
    with pytest.raises(TypeError, match="CycleOutcome"):
        await drive_cycle_tick(
            RunRequest(task="classify"),
            RunPlan(model="m-1"),
            task_class="classification",
            graph=object(),
        )


def test_cycle_refusal_rejects_open_code() -> None:
    for code in ABSTAIN_REASONS:
        assert CycleRefusal(code).code == code
    with pytest.raises(ValueError, match="open refusal code"):
        CycleRefusal("because")


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.2")
@pytest.mark.parametrize(
    ("returned", "k", "expected_refs"),
    [
        (
            [
                {"component_kind": "system_prompt", "component_version_ref": "v-a"},
                {"component_kind": "guardrails", "component_version_ref": "v-b"},
            ],
            2,
            ("v-a", "v-b"),
        ),
        (
            [
                {"component_kind": "system_prompt", "component_version_ref": "v-a"},
                {"component_kind": "guardrails", "component_version_ref": "v-b"},
                {"component_kind": "memory", "component_version_ref": "v-c"},
            ],
            2,
            ("v-a", "v-b"),
        ),
    ],
)
def test_propose_queries_lesson_library(returned, k, expected_refs) -> None:
    queried: list[str] = []

    def library(task_class: str) -> list[dict[str, str]]:
        queried.append(task_class)
        return [dict(entry) for entry in returned]

    candidates = propose(
        task_ref="task-1",
        task_class="classification",
        library=library,
        seed="seed-1",
        model_ref="model-1",
        budget_ref="budget-1",
        k=k,
    )

    assert queried == ["classification"]
    assert tuple(c.component_version_ref for c in candidates) == expected_refs
    for candidate in candidates:
        assert candidate.task_ref == "task-1"
        assert candidate.seed == "seed-1"
        assert candidate.model_ref == "model-1"
        assert candidate.budget_ref == "budget-1"
        with pytest.raises(dataclasses.FrozenInstanceError):
            candidate.seed = "other"  # type: ignore[misc]

    # The cycle stores nothing: a rerun re-queries the library and stores no
    # answer of its own (no write, no cache).
    again = propose(
        task_ref="task-1",
        task_class="classification",
        library=library,
        seed="seed-1",
        model_ref="model-1",
        budget_ref="budget-1",
        k=k,
    )
    assert again == candidates
    assert queried == ["classification", "classification"]

    with pytest.raises(CycleRefusal, match="no_candidates"):
        propose(
            task_ref="task-1",
            task_class="classification",
            library=lambda _class: [],
            seed="seed-1",
            model_ref="model-1",
            budget_ref="budget-1",
            k=k,
        )
