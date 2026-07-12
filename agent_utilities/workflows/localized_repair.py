"""Failure-localized, region-preserving repair — Atomic Task Graph paper idea #1
(arXiv:2607.01942, ``reports/paper-analysis-2607.01942.md`` §4 Rank 1;
``reports/autonomous-sdlc-loop-design.md`` §5.2/G15).

CONCEPT:AU-ORCH.execution.workflow-lifecycle-management. On a step/CI failure, the
naive options are "blind-retry the whole run" or "retry everything after the
failure in wall-clock order" — both discard validated work. ATG's one novel
contribution is computing the ACTUALLY-invalidated region from the DAG's own
structure: walk ``TRANSITION_TO`` edges FORWARD from the failed node to find its
transitive descendants (the region a fix can possibly change), and leave
everything else — upstream steps and sibling branches that never depended on the
failed node — untouched and reusable (their ``:RunTrace``/``:ToolCall`` results
stay valid).

:func:`localized_repair_region` is the reusable primitive both
:meth:`agent_utilities.workflows.runner.WorkflowRunner.resume_localized` (a
mid-workflow step failure) and :mod:`agent_utilities.observability.ci_recycle`
(a CI pipeline/job failure) call — one DAG-walk implementation, two callers, no
duplicated forward-walk logic.

Best-effort + engine-guarded: a missing engine or a node with no outgoing
``TRANSITION_TO`` edges degrades to an empty invalidated region (nothing to
repair beyond the failed node itself) rather than raising.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger("agent_utilities.workflows.localized_repair")

# The DAG edge the lifecycle spine / skill-workflow executor materializes step
# ordering on (``skill_workflow_ingest.py``, ``workflow_store.py``).
DEFAULT_EDGE_TYPE = "TRANSITION_TO"


def _rel(props: Any) -> str:
    if isinstance(props, dict):
        return str(props.get("rel_type") or props.get("type") or "")
    return ""


def _out_targets(engine: Any, node_id: str, edge_type: str) -> list[str]:
    """The targets of ``node_id``'s outgoing ``edge_type`` edges, best-effort."""
    if engine is None or not node_id:
        return []
    try:
        raw = engine.out_edges(node_id, data=True) or []
    except Exception as exc:  # noqa: BLE001 — read is best-effort
        logger.debug("localized_repair: out_edges(%s) failed: %s", node_id, exc)
        return []
    return [tgt for _src, tgt, props in raw if _rel(props) == edge_type]


def localized_repair_region(
    failed_node: str,
    *,
    engine: Any | None = None,
    edge_type: str = DEFAULT_EDGE_TYPE,
    all_nodes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute the minimal invalidated region downstream of ``failed_node``.

    Walks ``edge_type`` edges FORWARD (BFS) from ``failed_node`` and returns
    every transitively-reachable node id — the region a repair must re-execute.
    ``failed_node`` itself is included in ``invalidated`` (it must re-run too).

    When ``all_nodes`` is supplied (the full id set of the owning workflow/DAG),
    ``preserved`` is populated with every node NOT in the invalidated region —
    the validated upstream + sibling-branch steps whose prior
    ``:RunTrace``/``:ToolCall`` results a repair must leave untouched. Without
    ``all_nodes`` (the caller doesn't know/care about the full DAG — e.g. a bare
    CI job failure with no modeled sibling steps), ``preserved`` is ``[]``.

    Best-effort: a missing engine or a node with no outgoing ``edge_type`` edges
    yields ``invalidated == [failed_node]`` — never raises.
    """
    invalidated: set[str] = {failed_node} if failed_node else set()
    if engine is not None and failed_node:
        frontier = [failed_node]
        seen = {failed_node}
        while frontier:
            cur = frontier.pop()
            for nxt in _out_targets(engine, cur, edge_type):
                if nxt in seen:
                    continue
                seen.add(nxt)
                invalidated.add(nxt)
                frontier.append(nxt)

    preserved: list[str] = []
    if all_nodes is not None:
        preserved = sorted(set(all_nodes) - invalidated)

    return {
        "failed": failed_node,
        "invalidated": sorted(invalidated),
        "preserved": preserved,
    }
