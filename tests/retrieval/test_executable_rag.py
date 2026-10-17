#!/usr/bin/python
"""Tests for the executable multi-hop RAG interpreter + HybridRetriever wiring.

CONCEPT:AU-KG.retrieval.memory-first-retrieval
"""


import pytest

from agent_utilities.knowledge_graph.retrieval.executable_rag import (
    ExecutableRagProgram,
    PlanStep,
    RagResult,
    StepOp,
    build_linear_plan,
    parse_executable_plan,
)

pytestmark = pytest.mark.concept("AU-KG.retrieval.memory-first-retrieval")


def _docs(*ids):
    return [{"id": i, "content": f"content for {i}"} for i in ids]


# --- plan building ---------------------------------------------------------


def test_build_linear_plan_shape():
    plan = build_linear_plan(["q1", "q2"], question="Q", mode="vector", top_k=4)
    assert [s.op for s in plan] == [StepOp.RETRIEVE, StepOp.RETRIEVE, StepOp.ANSWER]
    assert plan[0].top_k == 4 and plan[0].mode == "vector"


# --- interpreter: happy path -----------------------------------------------


def test_run_collects_evidence_and_answers():
    retrieve = lambda q, mode, k: _docs(f"{q}-d1", f"{q}-d2")  # noqa: E731
    answer = lambda q, ev: f"answer from {len(ev)} docs"  # noqa: E731
    prog = ExecutableRagProgram(retrieve, answer)
    result = prog.run(build_linear_plan(["a", "b"], question="Q"), question="Q")
    assert isinstance(result, RagResult)
    assert result.success is True
    assert len(result.evidence_ids) == 4
    assert result.trace[-1].op == StepOp.ANSWER
    assert result.trace[-1].n_results == 4


# --- execution-driven adaptive retrieval (boost k) -------------------------


def test_adaptive_retrieval_boosts_topk_until_evidence():
    calls = []

    def retrieve(q, mode, k):
        calls.append(k)
        return _docs("d1") if k >= 8 else []  # only returns once k boosted to >=8

    prog = ExecutableRagProgram(
        retrieve,
        lambda q, ev: "ok",
        min_evidence=1,
        max_repairs=3,
        repair_topk_factor=2,
    )
    plan = [PlanStep(op=StepOp.RETRIEVE, query="x", mode="vector", top_k=2)]
    result = prog.run(plan)
    assert calls == [2, 4, 8]  # boosted 2→4→8
    assert result.trace[0].repaired is True
    assert "boost_k" in result.trace[0].repair_reason


# --- mode fallback (vector → grep) -----------------------------------------


def test_mode_fallback_switches_when_primary_empty():
    def retrieve(q, mode, k):
        return _docs("g1") if mode == "grep" else []

    prog = ExecutableRagProgram(
        retrieve, lambda q, ev: "ok", max_repairs=1, fallback_modes=["grep"]
    )
    plan = [PlanStep(op=StepOp.RETRIEVE, query="x", mode="vector", top_k=3)]
    result = prog.run(plan)
    assert result.trace[0].mode == "grep"
    assert "fallback_mode=grep" in result.trace[0].repair_reason


# --- compiler-grounded self-repair on insufficient answer ------------------


def test_insufficient_answer_triggers_re_retrieve():
    state = {"calls": 0}

    def retrieve(q, mode, k):
        state["calls"] += 1
        # first retrieve returns 1 doc; the repair re-retrieve returns more
        return _docs("d1") if state["calls"] == 1 else _docs("d2", "d3")

    answers = iter(["insufficient", "now I can answer"])

    prog = ExecutableRagProgram(retrieve, lambda q, ev: next(answers))
    plan = build_linear_plan(["x"], question="Q")
    result = prog.run(plan, question="Q")
    assert result.answer == "now I can answer"
    assert result.success is True
    assert result.trace[-1].repaired is True
    assert "re_retrieve" in result.trace[-1].repair_reason


# --- data-flow variable substitution ---------------------------------------


def test_variable_substitution_chains_steps():
    seen_queries = []

    def retrieve(q, mode, k):
        seen_queries.append(q)
        return _docs("d1")

    prog = ExecutableRagProgram(retrieve, lambda q, ev: "draft-answer")
    plan = [
        PlanStep(op=StepOp.RETRIEVE, query="{{question}}", out_var="r0"),
        PlanStep(op=StepOp.ANSWER, query="summarize", out_var="ans"),
        # second retrieve references the prior answer string var
        PlanStep(op=StepOp.RETRIEVE, query="follow up on {{ans}}", out_var="r1"),
    ]
    prog.run(plan, question="What is X?")
    assert seen_queries[0] == "What is X?"  # {{question}} resolved
    assert seen_queries[1] == "follow up on draft-answer"  # {{ans}} resolved


# --- live path: HybridRetriever.retrieve_executable ------------------------


# --- LLM plan synthesizer (parse-or-fallback) ------------------------------


def test_parse_executable_plan_parses_multistep_json():
    raw = (
        'noise {"steps": ['
        '{"op": "retrieve", "query": "sub a", "mode": "grep", "top_k": 7, "out_var": "r0"},'
        '{"op": "retrieve", "query": "sub b", "mode": "vector", "out_var": "r1"},'
        '{"op": "answer", "query": "Q"}]} trailing'
    )
    plan = parse_executable_plan(raw, question="Q", mode="vector", top_k=5)
    assert [s.op for s in plan] == [StepOp.RETRIEVE, StepOp.RETRIEVE, StepOp.ANSWER]
    assert plan[0].mode == "grep" and plan[0].top_k == 7
    assert plan[1].top_k == 5  # default filled in


def test_parse_executable_plan_appends_missing_answer():
    raw = '{"steps": [{"op": "retrieve", "query": "x"}]}'
    plan = parse_executable_plan(raw, question="Q")
    assert plan[-1].op == StepOp.ANSWER


def test_parse_executable_plan_falls_back_on_garbage():
    plan = parse_executable_plan("not json at all", question="Q", top_k=4)
    # Degrades to a linear plan over the question.
    assert [s.op for s in plan] == [StepOp.RETRIEVE, StepOp.ANSWER]
    assert plan[0].query == "Q"


def test_parse_executable_plan_falls_back_when_no_retrieve():
    raw = '{"steps": [{"op": "answer", "query": "Q"}]}'
    plan = parse_executable_plan(raw, question="Q", subqueries=["s1"])
    assert any(s.op == StepOp.RETRIEVE for s in plan)
