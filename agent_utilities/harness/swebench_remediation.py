"""CONCEPT:AHE-3.23 — SWE-failure-driven remediation (the surpass lever).

OpenHands' SWE-bench score is a static number. Here, every *unresolved* instance becomes a
``FailureRecord`` → clustered → filed as a ``failure_gap`` Concept via the single shared
AHE-3.18 path (:func:`file_gap_topic`). The golden loop's ``unresolved_topics()`` intake then
picks those gaps up unchanged and drives a remediation cycle; promotion is gated by a
SWE-specific regression check that **re-runs the exact failed instance in isolation** and only
passes when it now resolves. Because the workspace mirrored every action to the KG grounded on
the ``Code`` symbols it mutated (KG-2.64), the failure is attributable, not an opaque log.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agent_utilities.knowledge_graph.adaptation.failure_analyzer import (
    FailureRecord,
    cluster_failures,
    file_gap_topic,
)

from .swebench_corpus import SweBenchInstance
from .swebench_harness import InstanceResult, evaluate_instance

logger = logging.getLogger(__name__)


def build_failure_records(results: list[InstanceResult]) -> list[FailureRecord]:
    """One FailureRecord per unresolved instance (resolved ones are not failures)."""
    records: list[FailureRecord] = []
    for r in results:
        if r.resolved:
            continue
        detail = (
            r.error
            or f"FAIL_TO_PASS {r.fail_to_pass_passed}/{r.fail_to_pass_total}, "
            f"PASS_TO_PASS {r.pass_to_pass_passed}/{r.pass_to_pass_total}"
        )
        records.append(
            FailureRecord(
                kind="low_score",
                name=r.instance_id,
                detail=detail,
                anomaly_type="swebench_unresolved",
                trace_id=r.trace_run_id or None,
                value=0.0,
                baseline=1.0,
            )
        )
    return records


def remediate(
    results: list[InstanceResult],
    engine: Any,
    *,
    golden_loop: Any = None,
    run_cycle: bool = False,
    max_topics: int = 5,
) -> dict[str, Any]:
    """File failure-gap Concepts for every unresolved instance; optionally run a golden cycle.

    Returns ``{patterns, gaps, cycle}``. ``gaps`` are the dicts the golden loop intake consumes;
    when ``run_cycle`` is set and a ``golden_loop`` controller is supplied, one cycle is driven
    with those gaps as explicit topics.
    """
    records = build_failure_records(results)
    patterns = cluster_failures(records)
    gaps: list[dict[str, Any]] = []
    for pattern in patterns:
        gap = file_gap_topic(engine, pattern, source="swebench")
        if gap:
            gaps.append(gap)

    cycle: dict[str, Any] | None = None
    if run_cycle and golden_loop is not None and gaps:
        try:
            cycle = golden_loop.run_one_cycle(topics=gaps, max_topics=max_topics)
        except Exception as exc:  # noqa: BLE001 - remediation report must not raise
            logger.warning("golden-loop remediation cycle failed: %s", exc)
            cycle = {"error": str(exc)}

    return {
        "unresolved": len(records),
        "patterns": len(patterns),
        "gaps": gaps,
        "cycle": cycle,
    }


def make_swebench_regression_gate(
    instance: SweBenchInstance,
    *,
    workspace_factory: Callable[[SweBenchInstance], Any],
    solve: Any = None,
) -> Callable[[], Awaitable[bool]]:
    """Build an async gate that re-runs *this exact instance* and returns True iff now resolved.

    Stronger than the generic volatility guard (``FailureAnalyzer.make_regression_check``): a
    remediation for a SWE-bench failure is promoted (AHE-3.14 governed merge) only when the
    previously-failing instance actually re-resolves on a fresh workspace.
    """

    async def gate() -> bool:
        ws = workspace_factory(instance)
        await ws.start()
        try:
            result = await evaluate_instance(instance, workspace=ws, solve=solve)
            return result.resolved
        finally:
            await ws.stop()

    return gate
