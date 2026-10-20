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
    persist,
    probe_candidates,
    propose,
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


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.4")
async def test_compose_delegates_to_engine() -> None:
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


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.4")
async def test_compose_abstains_when_engine_abstains() -> None:
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
            baseline=None,
            traces={
                candidate.candidate_id: _trace(candidate.candidate_id)
                for candidate in candidates
            },
            decide=decide,
        )
    assert len(calls) == 1


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.5")
def test_persist_envelopes_switch_harness(tmp_path, monkeypatch) -> None:
    import agent_utilities.harness.harness_cycle as cycle

    component = tmp_path / "harness" / "system_prompt.md"
    component.parent.mkdir(parents=True)
    component.write_text("current prompt", encoding="utf-8")

    envelopes: list[dict[str, object]] = []

    def write_envelope(envelope: dict[str, object]) -> str:
        envelopes.append(envelope)
        return f"ref-{envelope['kind']}"

    calls: list[tuple[list, object]] = []
    real_apply = cycle.apply_edits

    def counting_apply(edits: list, root, **kwargs) -> object:
        calls.append((list(edits), root))
        return real_apply(edits, root, **kwargs)

    monkeypatch.setattr(cycle, "apply_edits", counting_apply)

    adopted = _probe_candidates()[0]
    contrast = ContrastDelta(
        candidate_id="c-0",
        run_a_trace_id="run-1:probe:current",
        run_b_trace_id="run-1:probe:c-0",
        retry_delta=0,
        tool_selection_delta=0,
        eval_score_delta=1,
        cost_delta=0,
    )
    answer = ContrastAnswer(
        adopted_version_ref="v-a",
        abstain_reason=None,
        adopted_component_kind="system_prompt",
        adopted_content="adopted prompt",
    )

    version_ref, contrast_ref, lesson_ref = persist(
        task_ref="task-1",
        task_class="classification",
        conditions_digest_value="digest-1",
        adopted=adopted,
        contrast=contrast,
        answer=answer,
        write_envelope=write_envelope,
        root=tmp_path,
    )

    assert [envelope["kind"] for envelope in envelopes] == [
        "harness_component_version",
        "harness_contrast",
        "harness_lesson",
    ]
    assert len(calls) == 1
    assert version_ref == "v-a"
    assert contrast_ref == "ref-harness_contrast"
    assert lesson_ref == "ref-harness_lesson"
    assert component.read_text(encoding="utf-8") == "adopted prompt\n"
    assert [
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
    ] == ["harness/system_prompt.md"]
    assert envelopes[0]["fact"] == {
        "task_ref": "task-1",
        "component_kind": "system_prompt",
        "component_version_ref": "v-a",
        "evidence_run_id": "run-1:probe:c-0",
    }
    assert envelopes[1]["fact"]["conditions_digest"] == "digest-1"
    assert envelopes[2]["fact"]["evidence_run_ids"] == [
        "run-1:probe:current",
        "run-1:probe:c-0",
    ]


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


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.6")
async def test_reflect_submits_lesson_on_adoption(tmp_path, events) -> None:
    def model(messages, info):
        return ModelResponse(parts=[TextPart("done")])

    hp.register_run_decision_provider(
        lambda request: RunPlan(
            model=FunctionModel(model),
            steps=(request.task,),
            budget=RunBudget(candidate_slots=2),
        )
    )

    def decide(request: ContrastRequest):
        return {
            "adopted_version_ref": request.candidate_version_refs[0],
            "abstain_reason": None,
            "adopted_component_kind": "system_prompt",
            "adopted_content": "adopted prompt",
        }

    written: list[str] = []

    def write_envelope(envelope: dict[str, object]) -> str:
        written.append(str(envelope["kind"]))
        return f"ref-{envelope['kind']}"

    component = tmp_path / "harness" / "system_prompt.md"
    component.parent.mkdir(parents=True)
    component.write_text("current prompt", encoding="utf-8")

    outcome = await run_cycle_tick(
        _tick_request(),
        RunPlan(
            model=FunctionModel(model),
            budget=RunBudget(candidate_slots=2),
        ),
        task_class="classification",
        graph=object(),
        library_query=lambda _class: _library(),
        decide=decide,
        write_envelope=write_envelope,
        root=tmp_path,
    )

    assert outcome.adopted_version_ref == "v-a"
    assert outcome.abstain_reason is None
    assert outcome.lesson_ref == "ref-harness_lesson"
    assert outcome.contrast_ref == "ref-harness_contrast"
    assert written == [
        "harness_component_version",
        "harness_contrast",
        "harness_lesson",
    ]
    assert component.read_text(encoding="utf-8") == "adopted prompt\n"

    cycle_events = [
        event for event in events if event.get("type") == "harness_cycle_outcome"
    ]
    assert len(cycle_events) == 1
    reported = cycle_events[0]
    # One lesson on adoption, citing the evidence runs by trace id.
    assert reported["lesson_ref"] == "ref-harness_lesson"
    assert reported["evidence_run_ids"] == [
        "task-1:probe:current",
        "task-1:probe:task-1:candidate:0",
    ]
    assert reported["abstain_reason"] is None

    # None on abstention.
    def abstain(request: ContrastRequest):
        return {"adopted_version_ref": None, "abstain_reason": "no_improvement"}

    written.clear()
    events.clear()
    outcome = await run_cycle_tick(
        _tick_request(),
        RunPlan(
            model=FunctionModel(model),
            budget=RunBudget(candidate_slots=2),
        ),
        task_class="classification",
        graph=object(),
        library_query=lambda _class: _library(),
        decide=abstain,
        write_envelope=write_envelope,
        root=tmp_path,
    )
    assert outcome.abstain_reason == "compose_abstention"
    assert outcome.adopted_version_ref is None
    assert outcome.lesson_ref is None
    assert written == []
    assert component.read_text(encoding="utf-8") == "adopted prompt\n"
    cycle_events = [
        event for event in events if event.get("type") == "harness_cycle_outcome"
    ]
    assert len(cycle_events) == 1
    assert cycle_events[0]["lesson_ref"] is None
    assert cycle_events[0]["evidence_run_ids"] == []


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
