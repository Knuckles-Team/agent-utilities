#!/usr/bin/python
"""The four shortcut-risk detectors, as queries over an evidence graph.

CONCEPT:AU-KG.retrieval.formulate-adversarially-refine — shortcut-risk control

FORT-Searcher (arXiv:2606.12087, §2.3) formalizes four *actionable shortcut
risks* that let a multi-constraint search task collapse to a cheaper identifying
route, so the synthesized question never forces real evidence acquisition:

* **single-clue selectivity** — one clue alone narrows the candidate pool
  ``s(P) = |Ans(P)|`` to the answer (route-level, eq 7).
* **evidence co-coverage** — several clues are verified by one retrieved source,
  collapsing the separate-retrieval effort ``M_ev(P)`` (route-level, eq 8).
* **exposed constants** — an exact answer/intermediate name on the question
  surface makes downstream queries immediately executable, cutting the
  dependency depth ``dep(P)`` (route-level, eq 9).
* **prior-knowledge binding** — the solver names the answer from parametric
  memory before retrieval anchors it, raising the solver-side cost reduction
  ``U_π0`` (solver-level, eq 11).

Unlike FORT (heuristics over freshly scraped pages) these run as deterministic
checks over the *provenance-rich* :class:`EvidenceGraph`, so co-coverage is an
exact source-sharing test, not an estimate. Each detector returns a
:class:`RiskFinding`; :func:`diagnose` bundles all four into a
:class:`RiskReport`.

Concept: shortcut-risks
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from .models import (
    SINGLE_CLUE_POOL_FLOOR,
    EvidenceGraph,
    RiskFinding,
    RiskReport,
)

# A closed-book probe answers a question from parametric knowledge only (no
# retrieval). Returns the model's answer string; the detector checks whether it
# already names the gold answer. ``None`` (the CPU default) skips the probe.
PriorProbe = Callable[[str], str]


def single_clue_selectivity(
    eg: EvidenceGraph, *, floor: int = SINGLE_CLUE_POOL_FLOOR
) -> RiskFinding:
    """Flag clues whose standalone candidate pool ``s({c})`` already identifies y*.

    A clue with ``standalone_pool <= floor`` lets the solver reach the answer
    from that single clue, skipping the rest of the intended evidence. Severity
    scales with how many selected clues are over-selective.
    """
    offenders = [f.id for f in eg.facts if f.standalone_pool <= floor]
    n = len(eg.facts) or 1
    score = round(len(offenders) / n, 6)
    return RiskFinding(
        risk="single_clue_selectivity",
        score=score,
        tripped=bool(offenders),
        offenders=offenders,
        detail=(
            f"{len(offenders)} clue(s) with standalone pool <= {floor} identify the "
            "answer alone"
            if offenders
            else "no clue is identifying in isolation"
        ),
    )


def evidence_co_coverage(eg: EvidenceGraph, *, max_per_source: int = 1) -> RiskFinding:
    """Flag clues co-covered by a single evidence source (small ``M_ev``).

    Groups selected clues by ``source_document_id``; any source verifying more
    than ``max_per_source`` clues compresses several intended evidence-acquisition
    steps into one retrieval. Offenders are the clues sharing an over-covered
    source.
    """
    by_source: dict[str, list[str]] = defaultdict(list)
    for f in eg.facts:
        by_source[f.source_document_id].append(f.id)
    offenders: list[str] = []
    worst = 0
    for ids in by_source.values():
        if len(ids) > max_per_source:
            offenders.extend(ids)
            worst = max(worst, len(ids))
    n = len(eg.facts) or 1
    score = round(len(offenders) / n, 6)
    return RiskFinding(
        risk="evidence_co_coverage",
        score=score,
        tripped=bool(offenders),
        offenders=offenders,
        detail=(
            f"a single source covers {worst} clues (> {max_per_source})"
            if offenders
            else "every clue is verified by a distinct source"
        ),
    )


def exposed_constants(eg: EvidenceGraph, question: str) -> RiskFinding:
    """Flag answer/intermediate names exposed on the question surface (small ``dep``).

    An exact gold-answer alias on the surface is a degenerate leak; an exposed
    *intermediate* name makes a downstream query executable from the start,
    shortening the serial dependency depth. Detection is literal containment
    against the rendered question text.
    """
    low = question.lower()
    leaked_answer = [a for a in eg.answer_aliases if a and a.lower() in low]
    leaked_inter: list[str] = []
    for f in eg.facts:
        for name in f.referenced_names:
            if name and name.lower() in low:
                leaked_inter.append(name)
    offenders = sorted(set(leaked_answer) | set(leaked_inter))
    # An exposed answer alias is maximally severe; intermediate exposure is graded.
    if leaked_answer:
        score = 1.0
    else:
        score = round(min(1.0, len(leaked_inter) / (len(eg.facts) or 1)), 6)
    return RiskFinding(
        risk="exposed_constants",
        score=score,
        tripped=bool(offenders),
        offenders=offenders,
        detail=(
            (f"answer alias(es) exposed: {leaked_answer}; " if leaked_answer else "")
            + (
                f"intermediate name(s) exposed: {sorted(set(leaked_inter))}"
                if leaked_inter
                else ""
            )
        ).strip()
        or "no exact constants exposed on the question surface",
    )


def prior_knowledge_binding(
    eg: EvidenceGraph,
    question: str,
    *,
    probe: PriorProbe | None = None,
    popularity_threshold: float = 0.7,
) -> RiskFinding:
    """Flag tasks a solver can answer from prior knowledge before retrieval.

    Two conservative signals: a high ``root_popularity`` (a well-known seed/answer
    is easy to bind from parametric memory), and — when an ``InferenceBackend``
    closed-book ``probe`` is supplied — the probe naming the gold answer with no
    evidence. The CPU default (no probe, popularity 0) yields no risk, so smokes
    pass without an inference server.
    """
    tripped = False
    offenders: list[str] = []
    score = 0.0
    detail_parts: list[str] = []

    if eg.root_popularity >= popularity_threshold:
        tripped = True
        score = max(score, round(eg.root_popularity, 6))
        offenders.append(eg.answer_id)
        detail_parts.append(
            f"root/answer popularity {eg.root_popularity:.2f} >= {popularity_threshold}"
        )

    if probe is not None:
        guess = (probe(question) or "").lower()
        if any(a and a.lower() in guess for a in eg.answer_aliases):
            tripped = True
            score = 1.0
            if eg.answer_id not in offenders:
                offenders.append(eg.answer_id)
            detail_parts.append("closed-book probe named the gold answer")

    return RiskFinding(
        risk="prior_knowledge_binding",
        score=score,
        tripped=tripped,
        offenders=offenders,
        detail="; ".join(detail_parts) or "no prior-binding signal",
    )


def diagnose(
    eg: EvidenceGraph,
    question: str,
    *,
    max_per_source: int = 1,
    probe: PriorProbe | None = None,
) -> RiskReport:
    """Run all four detectors over a (graph, rendered-question) pair."""
    return RiskReport(
        findings=[
            single_clue_selectivity(eg),
            evidence_co_coverage(eg, max_per_source=max_per_source),
            exposed_constants(eg, question),
            prior_knowledge_binding(eg, question, probe=probe),
        ]
    )
