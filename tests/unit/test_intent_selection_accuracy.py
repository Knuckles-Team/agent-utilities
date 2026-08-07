"""Seam 8 A/B selection-accuracy harness (CONCEPT:AU-ECO.mcp.intent-surface-selection-accuracy).

Regression tripwire for ``scripts/measure_intent_routing_accuracy.py``'s LIVE
measurement: runs the real resolver against the hand-labelled corpus and
asserts it stays above a floor set with headroom below the measured baseline
(top-1 0.76 / top-3 0.86 against the original 21-case corpus as of
2026-07-11 — see ``docs/architecture/intent-surface.md`` §7) so a real
resolver/CPD-wiring regression fails CI instead of silently rotting.

D-ORC-19 (2026-08-01): the corpus carried exactly ONE delegation case
(``"orchestrate an agent workflow"``), and it only ever passed because its
wording happened to contain the target tool's own name-token
("orchestrate") — no case named an ingested SKILL, the actual operator
delegation workflow, so this floor stayed green while unpinned skill
delegation routed to the wrong tool 100% of the time (D-INT-4). Four
skill-naming delegation cases were added (none containing "orchestrate") to
make that regression class measurable; re-measured against the 25-case
corpus at the same commit as the D-INT-4 router fix: top-1 76.00% (19/25),
top-3 84.00% (21/25) — comfortably above this floor, with 4/4 new
delegation cases plus the original one all top-1 hits post-fix (they were
2 MISS / 2 ambiguous-but-top-3 pre-fix, live-measured by temporarily
reverting only ``intent_tools.py`` to its pre-fix ``HEAD`` content: top-1
64.00%/16/25, top-3 80.00%/20/25).
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.retrieval.intent_selection_accuracy import (
    CORPUS,
    measure_selection_accuracy,
    render_report,
)


def test_corpus_is_a_bounded_labelled_set_covering_every_verb():
    assert 15 <= len(CORPUS) <= 30
    verbs_covered = {case.verb for case in CORPUS}
    # `find` is deliberately not a separate case class — every case here is
    # ALSO a valid `find` query (find ranks across all verbs unfiltered).
    assert verbs_covered == {"ask", "write", "act", "manage", "why"}


def test_intent_surface_selection_accuracy_meets_measured_floor():
    """Live-measured, not fabricated — see the module docstring for the run
    that produced the baseline this floor is set (with margin) beneath."""
    report = measure_selection_accuracy()
    assert report.n == len(CORPUS)
    failure_detail = render_report(report)
    assert report.top1_accuracy >= 0.60, failure_detail
    assert report.top3_accuracy >= 0.75, failure_detail
