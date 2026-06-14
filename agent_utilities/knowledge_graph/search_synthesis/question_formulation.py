#!/usr/bin/python
"""Question formulation and adversarial refinement.

CONCEPT:KG-2.72 — question formulation + adversarial refinement

The last two FORT-Searcher stages (arXiv:2606.12087, §3.1.3–3.1.4). Formulation
renders a selected answer-bearing subgraph as a natural-language question while
**withholding intermediate names** (so downstream queries are not executable from
the start) and keeping clues that are jointly identifying yet individually
generic. Refinement then runs the four shortcut detectors as an adversary and
repairs the draft until no risk trips — pruning redundant co-covered clues,
generalizing over-selective clues, and withholding exposed constants — or reports
the residual risk so the caller can discard an unrepairable draft.

This mirrors FORT's loop but runs against the provenance-rich
:class:`EvidenceGraph` and reuses the live :class:`RetrievalQualityGate` failure
modes as an auxiliary reject signal when available.

Concept: question-formulation
"""

from __future__ import annotations

import copy

from .models import (
    SINGLE_CLUE_POOL_FLOOR,
    EvidenceFact,
    EvidenceGraph,
    SearchTask,
)
from .shortcut_risks import PriorProbe, diagnose

# When a required clue is too selective to drop, generalize it to this pool size
# (FORT "low-specificity clue formulation"): the clue stays but is reformulated to
# leave a larger candidate pool, so it no longer identifies the answer alone.
_GENERIC_POOL = SINGLE_CLUE_POOL_FLOOR + 1_000


def _render_clue(fact: EvidenceFact, withhold: set[str]) -> str:
    text = fact.clue
    for name in fact.referenced_names:
        if name in withhold:
            text = text.replace(name, fact.referring_expr)
    return text


def formulate(
    eg: EvidenceGraph,
    *,
    selected_ids: set[str] | None = None,
    withhold: set[str] | None = None,
    max_per_source: int = 1,
    probe: PriorProbe | None = None,
) -> SearchTask:
    """Render selected clues as a question and diagnose its shortcut risks.

    Names in ``withhold`` are replaced by each clue's generic referring
    expression. The returned :class:`SearchTask` carries the
    :class:`RiskReport` so callers can accept/reject without re-diagnosing.
    """
    withhold = withhold or set()
    facts = [f for f in eg.facts if selected_ids is None or f.id in selected_ids]
    clauses = [_render_clue(f, withhold) for f in facts]
    question = (
        "Identify the entity satisfying all of the following constraints: "
        + "; ".join(clauses)
        + "."
    )
    rendered = EvidenceGraph(
        answer_id=eg.answer_id,
        answer_aliases=eg.answer_aliases,
        facts=facts,
        root_popularity=eg.root_popularity,
    )
    report = diagnose(rendered, question, max_per_source=max_per_source, probe=probe)
    difficulty = sum(1 for f in facts if f.required)
    return SearchTask(
        question=question,
        answer_id=eg.answer_id,
        answer_aliases=eg.answer_aliases,
        evidence_fact_ids=[f.id for f in facts],
        difficulty=difficulty,
        risk_report=report,
    )


def _repair(
    working: EvidenceGraph,
    selected: set[str],
    withhold: set[str],
    task: SearchTask,
) -> bool:
    """Apply one round of shortcut repairs in place. Returns True if anything changed.

    Repairs (FORT §3.1.4): withhold exposed intermediate names, drop redundant
    co-covered / over-selective clues, and generalize required-but-over-selective
    clues. A required clue is never dropped (it would break well-posedness).
    """
    changed = False
    report = task.risk_report

    # 1. exposed constants → withhold the leaked intermediate names; drop any
    #    redundant clue that leaks a gold-answer alias (cannot be withheld).
    exposed = report.by_risk("exposed_constants")
    if exposed and exposed.tripped:
        answer_low = {a.lower() for a in working.answer_aliases}
        for name in exposed.offenders:
            if name.lower() in answer_low:
                for f in list(working.facts):
                    if (
                        f.id in selected
                        and not f.required
                        and name.lower() in f.clue.lower()
                    ):
                        selected.discard(f.id)
                        changed = True
            elif name not in withhold:
                withhold.add(name)
                changed = True

    # 2. evidence co-coverage → drop redundant clues sharing an over-covered source,
    #    keeping one (prefer a derived, required clue).
    cocov = report.by_risk("evidence_co_coverage")
    if cocov and cocov.tripped:
        by_source: dict[str, list[EvidenceFact]] = {}
        for fid in list(selected):
            f = working.fact(fid)
            by_source.setdefault(f.source_document_id, []).append(f)
        for group in by_source.values():
            if len(group) <= 1:
                continue
            keep = sorted(group, key=lambda f: (f.required, f.derived), reverse=True)[0]
            for f in group:
                if f.id != keep.id and not f.required:
                    selected.discard(f.id)
                    changed = True

    # 3. single-clue selectivity → drop redundant over-selective clues; generalize
    #    required ones to a larger candidate pool.
    sel = report.by_risk("single_clue_selectivity")
    if sel and sel.tripped:
        for fid in list(sel.offenders):
            if fid not in selected:
                continue
            f = working.fact(fid)
            if f.required:
                if f.standalone_pool < _GENERIC_POOL:
                    f.standalone_pool = _GENERIC_POOL
                    changed = True
            else:
                selected.discard(fid)
                changed = True

    return changed


def refine(
    eg: EvidenceGraph,
    *,
    max_iters: int = 6,
    max_per_source: int = 1,
    probe: PriorProbe | None = None,
) -> SearchTask:
    """Adversarially refine a draft until no shortcut trips (or risk is irreparable).

    Returns the final :class:`SearchTask`; ``task.risk_report.clear`` tells the
    caller whether to accept it. Prior-knowledge binding is not structurally
    repairable here (it needs a different seed entity) — it surfaces in the
    report so the caller can discard and re-seed.
    """
    working = copy.deepcopy(eg)
    selected = {f.id for f in working.facts}
    withhold: set[str] = set()

    task = formulate(
        working,
        selected_ids=selected,
        withhold=withhold,
        max_per_source=max_per_source,
        probe=probe,
    )
    for _ in range(max_iters):
        if task.risk_report.clear:
            return task
        if not _repair(working, selected, withhold, task):
            break  # nothing left to repair — return best effort with residual risk
        task = formulate(
            working,
            selected_ids=selected,
            withhold=withhold,
            max_per_source=max_per_source,
            probe=probe,
        )
    return task
