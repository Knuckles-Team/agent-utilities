#!/usr/bin/python
from __future__ import annotations

"""Executable multi-hop RAG — a program over retrieve()/answer() primitives.

CONCEPT:KG-2.12 — Memory-First Retrieval (executable program mode)

Distilled from the PyRAG research (`.specify/specs/research-evolution-20260606/`
plan b2-03): instead of free-form NL self-reflection, multi-hop RAG is expressed
as an **executable plan** — a sequence of typed ``retrieve``/``answer`` steps with
explicit data-flow variables — run by a deterministic interpreter that gives two
training-free, grounded loops:

* **execution-driven adaptive retrieval** — a step that returns too little
  evidence transparently boosts ``top_k`` and re-runs (and can fall back to
  another retrieval *mode*, e.g. vector → grep), the "compiler signal" replacing
  an LLM self-judgement;
* **inspectable execution trace** — every step records its query, mode, k,
  result count, and whether it was repaired, for provenance/debugging.

The interpreter is injected with a ``retrieve_fn(query, mode, top_k) -> list`` and
an ``answer_fn(query, evidence) -> str`` so it is fully deterministic and testable
with no model; an LLM plan-synthesizer / answerer can be supplied later behind the
same seams. The retrieval spine for the stack — one trace dispatches per step to
grep (direct-corpus), vector, graph, or lossless-expand.

Concept: executable-rag
"""

import re
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

RetrieveFn = Callable[[str, str, int], list[dict[str, Any]]]
AnswerFn = Callable[[str, list[dict[str, Any]]], str]

_VAR = re.compile(r"\{\{(\w+)\}\}")
# Markers an answer_fn may emit to signal it lacked evidence (drives repair).
_INSUFFICIENT = (
    "insufficient",
    "i don't know",
    "cannot answer",
    "no evidence",
    "unknown",
)


class StepOp(StrEnum):
    RETRIEVE = "retrieve"
    ANSWER = "answer"


class PlanStep(BaseModel):
    """One step of an executable RAG plan."""

    op: StepOp
    query: str  # may reference prior string vars / {{question}}
    mode: str = "vector"  # retrieval mode for RETRIEVE steps
    top_k: int = 5
    out_var: str = ""  # bind this step's result to a variable


class StepTrace(BaseModel):
    """Inspectable record of one executed step."""

    op: StepOp
    query: str
    mode: str = ""
    top_k: int = 0
    n_results: int = 0
    repaired: bool = False
    repair_reason: str = ""


class RagResult(BaseModel):
    """Result of running an executable RAG plan."""

    answer: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    trace: list[StepTrace] = Field(default_factory=list)
    success: bool = False


def _is_insufficient(answer: str) -> bool:
    low = (answer or "").strip().lower()
    return (not low) or any(m in low for m in _INSUFFICIENT)


def build_linear_plan(
    subqueries: list[str],
    *,
    question: str = "",
    mode: str = "vector",
    top_k: int = 5,
) -> list[PlanStep]:
    """Build a decompose→retrieve*→answer plan from sub-queries.

    One ``retrieve`` step per sub-query (accumulating evidence) followed by a
    single ``answer`` step over the original question.
    """
    plan = [
        PlanStep(op=StepOp.RETRIEVE, query=sq, mode=mode, top_k=top_k, out_var=f"r{i}")
        for i, sq in enumerate(subqueries)
    ]
    plan.append(PlanStep(op=StepOp.ANSWER, query=question or "{{question}}"))
    return plan


def parse_executable_plan(
    raw: str,
    *,
    question: str = "",
    subqueries: list[str] | None = None,
    mode: str = "vector",
    top_k: int = 5,
) -> list[PlanStep]:
    """Parse an LLM-synthesized plan (``{"steps":[...]}``) into ``PlanStep``\\ s.

    The optional LLM plan-synthesizer counterpart to :func:`build_linear_plan`: an
    LLM can emit a richer, non-linear plan (interleaving retrieve modes, branching
    sub-queries) which this parses into the same typed plan the deterministic
    interpreter runs. **Always succeeds** — any malformed / empty / partial output
    degrades to :func:`build_linear_plan`, so the live path never breaks on a
    planner failure (mirrors the HyDE planner's parse-or-fallback contract).
    """
    fallback = lambda: build_linear_plan(  # noqa: E731
        subqueries or ([question] if question else [""]),
        question=question,
        mode=mode,
        top_k=top_k,
    )
    import json as _json

    try:
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not match:
            return fallback()
        steps_raw = _json.loads(match.group()).get("steps", [])
        plan: list[PlanStep] = []
        for s in steps_raw:
            if not isinstance(s, dict):
                continue
            op_val = str(s.get("op", "")).lower()
            if op_val not in (StepOp.RETRIEVE.value, StepOp.ANSWER.value):
                continue
            plan.append(
                PlanStep(
                    op=StepOp(op_val),
                    query=str(s.get("query", question) or question),
                    mode=str(s.get("mode", mode) or mode),
                    top_k=int(s.get("top_k", top_k) or top_k),
                    out_var=str(s.get("out_var", "")),
                )
            )
        # A usable plan must retrieve at least once and end by answering.
        if not any(p.op is StepOp.RETRIEVE for p in plan):
            return fallback()
        if not plan or plan[-1].op is not StepOp.ANSWER:
            plan.append(PlanStep(op=StepOp.ANSWER, query=question or "{{question}}"))
        return plan
    except (ValueError, TypeError, KeyError):
        return fallback()


class ExecutableRagProgram:
    """Deterministic interpreter for an executable RAG plan (CONCEPT:KG-2.12)."""

    def __init__(
        self,
        retrieve_fn: RetrieveFn,
        answer_fn: AnswerFn,
        *,
        min_evidence: int = 1,
        max_repairs: int = 2,
        repair_topk_factor: int = 2,
        fallback_modes: list[str] | None = None,
    ) -> None:
        self.retrieve_fn = retrieve_fn
        self.answer_fn = answer_fn
        self.min_evidence = min_evidence
        self.max_repairs = max_repairs
        self.repair_topk_factor = max(2, repair_topk_factor)
        self.fallback_modes = fallback_modes or []

    def _resolve(self, template: str, variables: dict[str, Any], question: str) -> str:
        def sub(m: re.Match[str]) -> str:
            key = m.group(1)
            if key == "question":
                return question
            val = variables.get(key, "")
            return val if isinstance(val, str) else ""

        return _VAR.sub(sub, template)

    def _retrieve_with_repair(
        self, query: str, mode: str, top_k: int
    ) -> tuple[list[dict[str, Any]], StepTrace]:
        results = self.retrieve_fn(query, mode, top_k)
        repaired = False
        reason = ""
        k = top_k
        attempts = 0
        # execution-driven adaptive retrieval: boost k while under-evidenced
        while len(results) < self.min_evidence and attempts < self.max_repairs:
            k *= self.repair_topk_factor
            attempts += 1
            repaired = True
            reason = "insufficient_evidence:boost_k"
            results = self.retrieve_fn(query, mode, k)
        # mode fallback: still empty → try alternate retrieval modes once each
        if len(results) < self.min_evidence:
            for alt in self.fallback_modes:
                if alt == mode:
                    continue
                alt_results = self.retrieve_fn(query, alt, top_k)
                if len(alt_results) >= self.min_evidence:
                    results = alt_results
                    repaired = True
                    reason = f"insufficient_evidence:fallback_mode={alt}"
                    mode = alt
                    k = top_k
                    break
        return results, StepTrace(
            op=StepOp.RETRIEVE,
            query=query,
            mode=mode,
            top_k=k,
            n_results=len(results),
            repaired=repaired,
            repair_reason=reason,
        )

    def run(self, plan: list[PlanStep], question: str = "") -> RagResult:
        """Execute ``plan`` and return the answer, evidence, and trace."""
        variables: dict[str, Any] = {}
        evidence: list[dict[str, Any]] = []
        trace: list[StepTrace] = []
        last_answer = ""
        last_retrieve: tuple[str, str, int] | None = None

        for step in plan:
            q = self._resolve(step.query, variables, question)

            if step.op == StepOp.RETRIEVE:
                results, st = self._retrieve_with_repair(q, step.mode, step.top_k)
                if step.out_var:
                    variables[step.out_var] = results
                evidence.extend(results)
                last_retrieve = (q, st.mode, st.top_k)
                trace.append(st)

            else:  # ANSWER
                answer = self.answer_fn(q, evidence)
                repaired = False
                reason = ""
                # compiler-grounded self-repair: an insufficient answer re-runs the
                # implicated retrieve at boosted k, then re-answers once.
                if _is_insufficient(answer) and last_retrieve is not None:
                    rq, rmode, rk = last_retrieve
                    boosted = self.retrieve_fn(rq, rmode, rk * self.repair_topk_factor)
                    if len(boosted) > 0:
                        evidence.extend(boosted)
                        answer = self.answer_fn(q, evidence)
                        repaired = True
                        reason = "insufficient_answer:re_retrieve"
                last_answer = answer
                if step.out_var:
                    variables[step.out_var] = answer
                trace.append(
                    StepTrace(
                        op=StepOp.ANSWER,
                        query=q,
                        n_results=len(evidence),
                        repaired=repaired,
                        repair_reason=reason,
                    )
                )

        # de-dup evidence ids preserving order
        seen: set[str] = set()
        ev_ids: list[str] = []
        for e in evidence:
            eid = str(e.get("id", ""))
            if eid and eid not in seen:
                seen.add(eid)
                ev_ids.append(eid)

        return RagResult(
            answer=last_answer,
            evidence_ids=ev_ids,
            trace=trace,
            success=not _is_insufficient(last_answer),
        )
