#!/usr/bin/python
from __future__ import annotations

"""Unit tests for the eval-set optimizer + compounding loop (CONCEPT:ORCH-1.55).

No LLM: a deterministic substring-match judge and stub system functions exercise
the seed -> score -> harvest -> re-optimize compounding loop.
"""

from agent_utilities.rlm.eval_set_optimizer import (
    EvalCase,
    EvalResult,
    EvalSet,
    EvalSetOptimizer,
)


# ── deterministic fake judge ────────────────────────────────────────────────


def substring_judge(system_output: str, case: EvalCase) -> EvalResult:
    """Score 1.0 if the expected text is a substring of the output, else 0.0."""
    hit = case.expected in system_output
    return EvalResult(case_id=case.case_id, score=1.0 if hit else 0.0, passed=hit)


def make_optimizer(refine_fn=None) -> EvalSetOptimizer:
    return EvalSetOptimizer(EvalSet(), substring_judge, refine_fn=refine_fn)


# ── EvalSet basics + dedup ──────────────────────────────────────────────────


def test_eval_set_add_and_dedup_by_case_id():
    s = EvalSet()
    s.add(EvalCase(case_id="a", input="q1", expected="x"))
    s.add(EvalCase(case_id="a", input="DIFFERENT", expected="y"))  # dedup no-op
    s.add(EvalCase(case_id="b", input="q2", expected="z"))
    assert len(s) == 2
    assert [c.case_id for c in s.cases()] == ["a", "b"]
    # original 'a' survived; the duplicate was ignored
    assert s.cases()[0].input == "q1"


# ── seed_from_annotations ───────────────────────────────────────────────────


def test_seed_from_annotations_counts_and_dedup():
    opt = make_optimizer()
    annotations = [
        ("seed1", "what is 2+2", "4"),
        ("seed2", "capital of france", "Paris"),
    ]
    added = opt.seed_from_annotations(annotations)
    assert added == 2
    assert len(opt.eval_set) == 2
    assert all(c.source == "seed" for c in opt.eval_set.cases())
    # re-seeding adds nothing (dedup by case_id)
    assert opt.seed_from_annotations(annotations) == 0
    assert len(opt.eval_set) == 2


# ── score_system pass_rate + failures ───────────────────────────────────────


def test_score_system_pass_rate_and_failures():
    opt = make_optimizer()
    opt.seed_from_annotations(
        [("c1", "q1", "alpha"), ("c2", "q2", "beta"), ("c3", "q3", "gamma")]
    )

    def system(inp: str) -> str:
        # answers q1 and q2 correctly, misses q3
        return {"q1": "alpha", "q2": "beta", "q3": "WRONG"}[inp]

    report = opt.score_system(system)
    assert report["pass_rate"] == 2 / 3
    assert report["mean_score"] == 2 / 3
    assert len(report["results"]) == 3
    assert len(report["failures"]) == 1
    failed_case, failed_output = report["failures"][0]
    assert failed_case.case_id == "c3"
    assert failed_output == "WRONG"


def test_score_system_is_weight_aware():
    opt = make_optimizer()
    opt.eval_set.add(EvalCase(case_id="heavy", input="q1", expected="ok", weight=3.0))
    opt.eval_set.add(EvalCase(case_id="light", input="q2", expected="ok", weight=1.0))

    def system(inp: str) -> str:
        return "ok" if inp == "q1" else "no"  # heavy passes, light fails

    report = opt.score_system(system)
    # weighted: 3 of 4 weight passed
    assert report["pass_rate"] == 0.75


# ── harvest_failure ─────────────────────────────────────────────────────────


def test_harvest_failure_adds_production_failure_case():
    opt = make_optimizer()
    case = EvalCase(case_id="c1", input="q1", expected="alpha")
    opt.eval_set.add(case)
    harvested = opt.harvest_failure("WRONG", case)
    assert harvested.source == "production_failure"
    assert harvested.case_id == "prodfail:c1"
    assert harvested in opt.eval_set
    assert len(opt.eval_set) == 2


def test_harvest_failure_uses_refine_fn_when_provided():
    def refiner(case: EvalCase, system_output: str) -> EvalCase:
        # sharpen: re-anchor expected to exactly what was missed, mark refined
        return EvalCase(
            case_id=case.case_id,
            input=case.input,
            expected=f"{case.expected}|sharpened",
            source="refined",
            weight=case.weight,
        )

    opt = make_optimizer(refine_fn=refiner)
    case = EvalCase(case_id="c1", input="q1", expected="alpha")
    harvested = opt.harvest_failure("WRONG", case)
    assert harvested.source == "refined"
    assert harvested.expected == "alpha|sharpened"
    assert harvested in opt.eval_set


# ── optimize_round grows the set by the number of failures ──────────────────


def test_optimize_round_grows_eval_set_by_failure_count():
    opt = make_optimizer()
    opt.seed_from_annotations(
        [("c1", "q1", "alpha"), ("c2", "q2", "beta"), ("c3", "q3", "gamma")]
    )

    def system(inp: str) -> str:
        return {"q1": "alpha", "q2": "WRONG", "q3": "WRONG"}[inp]

    size_before = len(opt.eval_set)
    report = opt.optimize_round(system)
    assert report["pass_rate_before"] == 1 / 3
    assert report["n_new_evals"] == 2  # two failures harvested
    assert report["eval_set_size"] == size_before + 2
    assert len(report["failures"]) == 2


# ── compounding loop: pass-rate rises while eval set grows monotonically ─────


def test_compounding_loop_monotonic_growth_and_rising_pass_rate():
    opt = make_optimizer()
    opt.seed_from_annotations(
        [("c1", "q1", "alpha"), ("c2", "q2", "beta"), ("c3", "q3", "gamma")]
    )

    # v1 answers 1/3; v2 answers 2/3; v3 answers all seeds (and the harvested
    # prodfail cases share the same expected text, so v3 satisfies them too).
    def v1(inp: str) -> str:
        return {"q1": "alpha", "q2": "WRONG", "q3": "WRONG"}[inp]

    def v2(inp: str) -> str:
        return {"q1": "alpha", "q2": "beta", "q3": "WRONG"}[inp]

    def v3(inp: str) -> str:
        return {"q1": "alpha", "q2": "beta", "q3": "gamma"}[inp]

    reports = opt.compounding_loop([v1, v2, v3])
    assert len(reports) == 3

    sizes = [r["eval_set_size"] for r in reports]
    # the eval set never shrinks (the compounding-IP property)
    assert sizes == sorted(sizes)
    # it strictly grew while there were failures to harvest
    assert sizes[0] > 3

    pass_rates = [r["pass_rate_before"] for r in reports]
    # later, better systems clear a higher bar despite the tougher (grown) set
    assert pass_rates[0] < pass_rates[1] < pass_rates[2]
    # the final, fully-correct version passes everything
    assert pass_rates[2] == 1.0
    # final round added nothing because there were no failures
    assert reports[2]["n_new_evals"] == 0


# ── determinism ─────────────────────────────────────────────────────────────


def test_scoring_is_deterministic():
    def build():
        o = make_optimizer()
        o.seed_from_annotations([("c1", "q1", "alpha"), ("c2", "q2", "beta")])
        return o

    def system(inp: str) -> str:
        return {"q1": "alpha", "q2": "WRONG"}[inp]

    r1 = build().score_system(system)
    r2 = build().score_system(system)
    assert r1["pass_rate"] == r2["pass_rate"]
    assert r1["mean_score"] == r2["mean_score"]
    assert [x.case_id for x, _ in r1["failures"]] == [
        x.case_id for x, _ in r2["failures"]
    ]
