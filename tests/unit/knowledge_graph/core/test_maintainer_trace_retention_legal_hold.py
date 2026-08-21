"""GOC-09: trace/tool-call/outcome retention with legal-hold protection
(CONCEPT:AU-KG.audit.trace-retention-legal-hold).

Known-bad proof: a ``RunTrace`` older than the retention window with
``is_permanent`` unset/False is deleted by ``prune_expired_traces``; a
BYTE-COMPARABLE ``RunTrace`` (same age, same everything else) marked
``is_permanent=True`` survives the exact same sweep. This is exercised against a
minimal Cypher-shaped fake backend that actually evaluates the sweep's
``WHERE n.timestamp < $cutoff AND (n.is_permanent IS NULL OR n.is_permanent =
False)`` predicate over synthetic rows — not merely asserting the query string
contains the right substrings — so the proof is about real delete/survive
behavior, not query shape.

The sweep issues TWO statements per batch, and the fake models both: a READ
that selects one bounded batch of ids, then a WRITE that deletes exactly those
ids. It used to be one statement (``... WITH n LIMIT $batch_size DETACH DELETE
n``), which is outside the engine's native Cypher write subset — ``WITH``
between a MATCH and a write clause is rejected at the wire boundary, so
retention never ran at all against a native backend. The fake enforces the
split by REFUSING any query that is not one of the two expected shapes, so a
regression to the unsupported form fails here rather than only in production.
"""

from __future__ import annotations

import re
from typing import Any

from agent_utilities.knowledge_graph.core.maintainer import (
    _TRACE_RETENTION_BATCH_SIZE,
    GraphMaintainer,
)
from agent_utilities.observability.trace_ontology import (
    OUTCOME_NODE_LABEL,
    TOOL_CALL_NODE_LABEL,
    TRACE_NODE_LABEL,
)

_SELECT_RE = re.compile(
    r"MATCH \(n:(?P<label>\w+)\)\s*"
    r"WHERE n\.timestamp < \$cutoff\s*"
    r"AND \(n\.is_permanent IS NULL OR n\.is_permanent = False\)\s*"
    r"RETURN n\.id AS id\s*"
    r"LIMIT \$batch_size",
)

_DELETE_RE = re.compile(
    r"MATCH \(n:(?P<label>\w+)\)\s*"
    r"WHERE n\.id IN \$ids\s*"
    r"DETACH DELETE n\s*"
    r"RETURN count\(n\) AS deleted_count",
)

#: The shape this sweep must never regress to: `WITH` between a MATCH and a
#: write clause is outside the engine's native Cypher write subset.
_UNSUPPORTED_WITH_RE = re.compile(r"WITH\s+n\s+LIMIT")


class _FakeRetentionBackend:
    """Stores nodes by id and actually evaluates the sweep's WHERE clause.

    Real (not string-matched) semantics for exactly this module's own retention
    query shape: a node whose ``timestamp`` is before the cutoff AND whose
    ``is_permanent`` is missing/False is removed; everything else survives.
    """

    def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
        self.nodes = dict(nodes)
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def execute(self, query: str, params: dict[str, Any] | None = None) -> Any:
        params = params or {}
        self.queries.append((query, params))
        if _UNSUPPORTED_WITH_RE.search(query):
            raise AssertionError(
                "query uses `WITH` between MATCH and a write clause, which the "
                f"engine's native Cypher write subset rejects: {query!r}"
            )
        selecting = _SELECT_RE.search(query)
        if selecting is not None:
            label = selecting.group("label")
            cutoff = params["cutoff"]
            return [
                {"id": node_id}
                for node_id, props in self.nodes.items()
                if props.get("node_type_label") == label
                and props.get("timestamp", "") < cutoff
                and not props.get("is_permanent", False)
            ][: params["batch_size"]]
        deleting = _DELETE_RE.search(query)
        if deleting is None:
            raise AssertionError(f"unexpected query shape: {query!r}")
        label = deleting.group("label")
        # Deletes ONLY the ids it was handed, and only if they still match the
        # label -- so a sweep that selected under one label and deleted under
        # another could not pass unnoticed.
        victims = [
            node_id
            for node_id in params["ids"]
            if node_id in self.nodes
            and self.nodes[node_id].get("node_type_label") == label
        ]
        for node_id in victims:
            del self.nodes[node_id]
        return [{"deleted_count": len(victims)}]


class _FakeEngine:
    def __init__(self, backend: _FakeRetentionBackend) -> None:
        self.backend = backend


def test_held_trace_survives_the_exact_sweep_that_deletes_a_comparable_unheld_trace():
    """Known-bad proof: two byte-comparable RunTrace nodes diverge ONLY on
    is_permanent; the sweep deletes the un-held one and leaves the held one intact."""
    nodes = {
        "trace:held": {
            "node_type_label": TRACE_NODE_LABEL,
            "timestamp": "2020-01-01T00:00:00Z",  # ancient — well past any retention window
            "is_permanent": True,
        },
        "trace:unheld": {
            "node_type_label": TRACE_NODE_LABEL,
            "timestamp": "2020-01-01T00:00:00Z",  # same age
            "is_permanent": False,
        },
    }
    backend = _FakeRetentionBackend(nodes)
    maintainer = GraphMaintainer(_FakeEngine(backend))

    swept = maintainer.prune_expired_traces(retention_days=90)

    assert (
        swept == 3
    )  # one query issued per label (RunTrace/ToolCall/OutcomeEvaluation)
    assert "trace:held" in backend.nodes, "legal-hold node must survive the sweep"
    assert "trace:unheld" not in backend.nodes, (
        "the comparable un-held node must be deleted by the same sweep"
    )


def test_recent_trace_survives_regardless_of_hold_status():
    """Known-good regression: a trace inside the retention window is untouched either
    way — the sweep is age-gated, not a blanket is_permanent=False purge."""
    nodes = {
        "trace:recent": {
            "node_type_label": TRACE_NODE_LABEL,
            "timestamp": "2099-01-01T00:00:00Z",  # far in the future relative to "now"
            "is_permanent": False,
        },
    }
    backend = _FakeRetentionBackend(nodes)
    maintainer = GraphMaintainer(_FakeEngine(backend))

    maintainer.prune_expired_traces(retention_days=90)

    assert "trace:recent" in backend.nodes, "a recent, un-held node is not swept"


def test_sweep_covers_all_three_trace_ontology_labels():
    """Every one of the canonical trace ontology's node types is swept, not just
    RunTrace — a caller relying on retention for ToolCall/OutcomeEvaluation must not
    be silently unprotected."""
    nodes = {
        "call:unheld": {
            "node_type_label": TOOL_CALL_NODE_LABEL,
            "timestamp": "2020-01-01T00:00:00Z",
            "is_permanent": False,
        },
        "outcome:unheld": {
            "node_type_label": OUTCOME_NODE_LABEL,
            "timestamp": "2020-01-01T00:00:00Z",
            "is_permanent": False,
        },
    }
    backend = _FakeRetentionBackend(nodes)
    maintainer = GraphMaintainer(_FakeEngine(backend))

    maintainer.prune_expired_traces(retention_days=90)

    assert backend.nodes == {}
    labels_queried = {
        match.group("label")
        for q, _ in backend.queries
        for match in (_SELECT_RE.search(q) or _DELETE_RE.search(q),)
        if match is not None
    }
    assert labels_queried == {
        TRACE_NODE_LABEL,
        TOOL_CALL_NODE_LABEL,
        OUTCOME_NODE_LABEL,
    }


def test_prune_expired_traces_is_a_noop_without_a_backend():
    class _NoBackendEngine:
        backend = None

    maintainer = GraphMaintainer(_NoBackendEngine())
    assert maintainer.prune_expired_traces(retention_days=90) == 0


def test_trace_retention_deletes_in_bounded_batches_with_exact_count():
    nodes = {
        f"trace:{index}": {
            "node_type_label": TRACE_NODE_LABEL,
            "timestamp": "2020-01-01T00:00:00Z",
            "is_permanent": False,
        }
        for index in range(_TRACE_RETENTION_BATCH_SIZE + 5)
    }
    backend = _FakeRetentionBackend(nodes)
    maintainer = GraphMaintainer(_FakeEngine(backend))

    maintainer.prune_expired_traces(retention_days=90)

    assert backend.nodes == {}
    assert maintainer.last_trace_retention_report == {
        "labels": 3,
        "deleted_rows": _TRACE_RETENTION_BATCH_SIZE + 5,
        "truncated": False,
    }
    # Two batches, each a select + a delete: the first select returns a full
    # batch (so the loop continues), the second returns the remaining 5 (short,
    # so it stops).
    trace_queries = [
        query
        for query, _params in backend.queries
        if f"(n:{TRACE_NODE_LABEL})" in query
    ]
    assert len(trace_queries) == 4
    assert sum(1 for q in trace_queries if _SELECT_RE.search(q)) == 2
    assert sum(1 for q in trace_queries if _DELETE_RE.search(q)) == 2


def test_trace_retention_rejects_dangerous_or_unbounded_windows():
    backend = _FakeRetentionBackend({})
    maintainer = GraphMaintainer(_FakeEngine(backend))

    for invalid in (True, 0, -1, 3_651, 90.0):
        try:
            maintainer.prune_expired_traces(retention_days=invalid)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"retention_days={invalid!r} must fail closed")

    assert backend.queries == []
