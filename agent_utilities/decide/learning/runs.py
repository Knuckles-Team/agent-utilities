"""The live retrieval path's half of retrieval learning (EH-394, EH-395).

``HybridRetriever.plan_and_retrieve`` asks EG for the plan (a template, or a
proven path of the retriever's task class), runs it, and notes what it
returned; ``record_answer_usage`` -- the generation step's report of which
recalled units the answer used -- then attests the run's outcome to EG,
joined to the committed plan record. An independent evaluator's verdict on
that record (a different principal; EG enforces it) is what turns the outcome
into labels: hard negatives, path support, adapter training data. AU holds
nothing learned: the only state here is the short correlation between a query
and its pending run, bounded, until the answer reports its citations.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.decide.consumers.retrieval import PATH_PREFIX, choose_retrieval
from agent_utilities.decide.learning.ops import (
    outcome_op,
    paths_op,
    q16,
    result_of,
    space_identity,
)
from agent_utilities.decide.learning.session import current_session

logger = logging.getLogger(__name__)

#: Pending runs one retriever keeps before the oldest is forgotten.
MAX_PENDING_RUNS = 256
#: The attribute a retriever keeps its pending runs under.
_LEDGER_ATTR = "_retrieval_run_ledger"


@dataclass(frozen=True, slots=True)
class PathScope:
    """What a retriever's proven paths are keyed by: the task class it serves
    and the composed schema identity of the graph it reads."""

    task_class: str
    composed_digest: str


@dataclass
class PendingRun:
    """One executed retrieval, until its answer reports what it cited."""

    record_id: str
    path: Mapping[str, Any] | None = None
    returned: list[tuple[str, str | None]] = field(default_factory=list)
    query: dict[str, Any] | None = None


class RunLedger:
    """Bounded, insertion-ordered pending runs by query text."""

    def __init__(self, limit: int = MAX_PENDING_RUNS) -> None:
        self._runs: OrderedDict[str, PendingRun] = OrderedDict()
        self._limit = limit

    def put(self, query: str, run: PendingRun) -> None:
        self._runs[query] = run
        self._runs.move_to_end(query)
        while len(self._runs) > self._limit:
            self._runs.popitem(last=False)

    def get(self, query: str) -> PendingRun | None:
        return self._runs.get(query)

    def pop(self, query: str) -> PendingRun | None:
        return self._runs.pop(query, None)


def ledger_of(retriever: Any) -> RunLedger:
    ledger = getattr(retriever, _LEDGER_ATTR, None)
    if not isinstance(ledger, RunLedger):
        ledger = RunLedger()
        setattr(retriever, _LEDGER_ATTR, ledger)
    return ledger


def proven_paths(scope: PathScope | None) -> list[Mapping[str, Any]]:
    """The task class's proven paths under its schema identity (EG's rows)."""
    session = current_session()
    if scope is None or session is None:
        return []
    op = paths_op(session.tenant, scope.task_class, scope.composed_digest)
    try:
        rows = result_of(session.send(op), "paths").get("rows") or []
    except Exception as exc:
        logger.warning("proven retrieval paths unavailable: %s", exc)
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def plan_retrieval(retriever: Any, query: str, mode: str) -> str:
    """The HyDE mode to run for ``query``; a chosen proven path is kept on the
    pending run (and runs first, see :func:`first_pass`)."""
    paths = proven_paths(getattr(retriever, "path_scope", None))
    choice = choose_retrieval(query, mode, paths)
    by_id = {PATH_PREFIX + str(p["template_digest"]): p for p in paths}
    chosen = by_id.get(choice.option_id) if choice.decided else None
    if choice.decided and choice.logged and choice.record_id:
        template = None if chosen is None else chosen.get("template")
        ledger_of(retriever).put(query, PendingRun(choice.record_id, template))
    return "standard" if chosen is not None else choice.option_id


def _rank_ops(
    rank: str, query: str, vector: list[float] | None
) -> list[dict[str, Any]]:
    vector_leg = [{"Rank": {"query": vector}}] if vector is not None else []
    text_leg = [{"RankText": {"query": query}}]
    table: dict[str, list[dict[str, Any]]] = {
        "unranked": [],
        "vector": vector_leg,
        "text": text_leg,
        "fuse_rrf": [{"FuseRrf": {"branches": [vector_leg, text_leg], "k": 60.0}}]
        if vector_leg
        else text_leg,
    }
    return table.get(rank, [])


def path_plan(
    template: Mapping[str, Any], query: str, vector: list[float] | None, k: int
) -> list[dict[str, Any]]:
    """A proven path template bound to one query: a unified plan (RLS applies
    when the engine runs it)."""
    ops: list[dict[str, Any]] = [{"Scan": {"label": str(template["anchor_class"])}}]
    for edge in template.get("edges") or []:
        ops.append(
            {
                "Traverse": {
                    "rel": str(edge["relationship"]),
                    "min": int(edge["min_hops"]),
                    "max": int(edge["max_hops"]),
                }
            }
        )
    ops.extend(_rank_ops(str(template.get("rank") or "unranked"), query, vector))
    ops.append({"Limit": {"k": int(k)}})
    return ops


def _embed(retriever: Any, query: str) -> list[float] | None:
    model = getattr(retriever, "embed_model", None)
    if model is None:
        return None
    try:
        return [float(x) for x in model.get_text_embedding(query)]
    except Exception as exc:
        logger.warning("query embedding for retrieval learning failed: %s", exc)
        return None


def _plan_rows(graph: Any, plan: list[dict[str, Any]]) -> list[Mapping[str, Any]]:
    try:
        rows = graph.query_unified(plan) or []
    except Exception as exc:
        logger.warning("proven path plan failed; the plan's template runs: %s", exc)
        return []
    return [row for row in rows if row.get("id") is not None]


def _hydrated(
    retriever: Any, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    ids = [str(row["id"]) for row in rows]
    props = retriever._batch_node_properties(ids) if ids else {}
    return [
        {**dict(props.get(i) or {}), "id": i, "_score": float(row.get("score") or 0.0)}
        for i, row in zip(ids, rows, strict=True)
    ]


def run_path(
    retriever: Any, template: Mapping[str, Any], query: str, k: int
) -> list[dict[str, Any]]:
    """Execute a proven path through the engine's unified plan."""
    graph = getattr(getattr(retriever, "engine", None), "graph", None)
    if graph is None:
        return []
    plan = path_plan(template, query, _embed(retriever, query), k)
    return _hydrated(retriever, _plan_rows(graph, plan))


def first_pass(
    retriever: Any,
    query: str,
    k: int,
    template_pass: Callable[[], list[list[dict[str, Any]]]],
) -> list[list[dict[str, Any]]]:
    """The first retrieval pass: the chosen proven path when it returns
    anything, the chosen template's pass otherwise."""
    run = ledger_of(retriever).get(query)
    if run is not None and run.path is not None:
        nodes = run_path(retriever, run.path, query, k)
        if nodes:
            return [nodes]
    return template_pass()


def _query_vector(retriever: Any, query: str) -> dict[str, Any] | None:
    model = getattr(retriever, "embed_model", None)
    vector = _embed(retriever, query)
    name = getattr(model, "model_name", None)
    if not vector or not name:
        return None
    return {"space_digest": space_identity(str(name), len(vector)), "q16": q16(vector)}


#: The node fields whose text the admission classifier reads, in priority order.
_TEXT_FIELDS = ("content", "text", "description", "summary", "name")


def content_class_of(node: Mapping[str, Any]) -> str:
    """The unit's ingestion content class (EH-269's deterministic classifier),
    the label EG's per-class usage and AU's admission feedback key on."""
    from agent_utilities.knowledge_graph.ingestion.embedding_admission import (
        classify_unit,
    )

    declared = node.get("content_class")
    if declared:
        return str(declared)
    text = next((str(node[k]) for k in _TEXT_FIELDS if node.get(k)), "")
    row = dict(node)
    verdict = classify_unit(
        connector=str(node.get("connector") or ""), row=row, text=text
    )
    return verdict.content_class.value


def note_returned(
    retriever: Any, query: str, nodes: Sequence[Mapping[str, Any]]
) -> None:
    """Record what the pending run of ``query`` returned, in rank order."""
    run = ledger_of(retriever).get(query)
    if run is None:
        return
    run.returned = [
        (str(n["id"]), content_class_of(n)) for n in nodes if n.get("id") is not None
    ]
    run.query = _query_vector(retriever, query)


def attest_citations(retriever: Any, query: str, used_ids: Sequence[str]) -> bool:
    """Attest the pending run of ``query``: what it returned and what the
    answer cited. ``False`` when there is nothing to attest or EG refused."""
    run = ledger_of(retriever).pop(query)
    session = current_session()
    if run is None or session is None:
        return False
    op = outcome_op(
        session.tenant,
        run.record_id,
        run.returned,
        used_ids,
        query=run.query,
        path=run.path,
    )
    try:
        session.send(op)
    except Exception as exc:
        logger.warning("retrieval outcome %s not recorded: %s", run.record_id, exc)
        return False
    return True


__all__ = [
    "MAX_PENDING_RUNS",
    "PathScope",
    "PendingRun",
    "RunLedger",
    "attest_citations",
    "content_class_of",
    "first_pass",
    "ledger_of",
    "note_returned",
    "path_plan",
    "plan_retrieval",
    "proven_paths",
    "run_path",
]
