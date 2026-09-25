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

import json
import logging
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.decide.consumers.retrieval import PATH_PREFIX, choose_retrieval
from agent_utilities.decide.learning.ops import (
    literal,
    outcome_op,
    q16,
    space_identity,
)
from agent_utilities.decide.learning.run_scope import (
    current_skill_ref,
    note_retrieval,
)
from agent_utilities.decide.learning.session import current_session

logger = logging.getLogger(__name__)

#: Pending runs one retriever keeps before the oldest is forgotten.
MAX_PENDING_RUNS = 256
#: The attribute a retriever keeps its pending runs under.
_LEDGER_ATTR = "_retrieval_run_ledger"
#: The native task a retriever serves unless it declares its own
#: ``task_class`` (memory-first retrieval answers research questions).
DEFAULT_TASK_CLASS = "eg:task/research"
#: How long a graph's composed schema identity is reused before it is re-read.
SCHEMA_TTL_S = 60.0
#: The version of the plan policy a seeded template records.
PLAN_POLICY_VERSION = "au.retrieval.plan/1"


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
    #: What the run's proven paths were keyed by (a seeded template's key).
    scope: PathScope | None = None
    #: Each returned unit's schema class (its ``type``/``label``).
    classes: dict[str, str] = field(default_factory=dict)
    #: The skill the run executed under (topology -> skill, EH-394).
    skill_ref: str | None = None


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


_PATHS_SQL = (
    "SELECT template_digest, successes, failures, template_json "
    "FROM decision_proven_paths WHERE task_class = {task} AND composed_digest = {schema} "
    "ORDER BY successes DESC, failures, template_digest"
)


def _path_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "template_digest": row["template_digest"],
        "successes": int(row.get("successes") or 0),
        "failures": int(row.get("failures") or 0),
        "template": json.loads(str(row["template_json"])),
    }


def proven_paths(scope: PathScope | None) -> list[Mapping[str, Any]]:
    """The task class's proven paths under its schema identity (EG's
    ``decision_proven_paths`` relation, as this caller may see it)."""
    session = current_session()
    if scope is None or session is None:
        return []
    sql = _PATHS_SQL.format(
        task=literal(scope.task_class), schema=literal(scope.composed_digest)
    )
    try:
        return [_path_row(row) for row in session.query(sql)]
    except Exception as exc:
        logger.warning("proven retrieval paths unavailable: %s", exc)
        return []


#: graph name -> (composed schema digest, when it was read).
_SCHEMA_DIGESTS: dict[str, tuple[str, float]] = {}


def composed_digest_of(
    graph: Any, *, now: Callable[[], float] = time.monotonic
) -> str | None:
    """The graph's committed composed GraphSchema identity (EG
    ``GraphSchemaList``), read at most once per :data:`SCHEMA_TTL_S`."""
    name = str(getattr(graph, "graph_name", "") or "")
    cached = _SCHEMA_DIGESTS.get(name)
    if cached is not None and now() - cached[1] < SCHEMA_TTL_S:
        return cached[0]
    try:
        digest = str(graph.graph_schema_list().composed_digest or "")
    except Exception as exc:
        logger.warning("composed schema of %s unreadable: %s", name, exc)
        return None
    _SCHEMA_DIGESTS[name] = (digest, now())
    return digest or None


def path_scope_of(retriever: Any) -> PathScope | None:
    """What the retriever's proven paths are keyed by: its declared
    ``path_scope``, else its task class (``task_class`` or
    :data:`DEFAULT_TASK_CLASS`) under its graph's composed schema. ``None``
    with no session (nothing is asked) or no readable schema."""
    declared = getattr(retriever, "path_scope", None)
    if isinstance(declared, PathScope):
        return declared
    graph = getattr(getattr(retriever, "engine", None), "graph", None)
    if current_session() is None or graph is None:
        return None
    digest = composed_digest_of(graph)
    task = str(getattr(retriever, "task_class", None) or DEFAULT_TASK_CLASS)
    return None if digest is None else PathScope(task, digest)


def plan_retrieval(retriever: Any, query: str, mode: str) -> str:
    """The HyDE mode to run for ``query``; a chosen proven path is kept on the
    pending run (and runs first, see :func:`first_pass`)."""
    scope = path_scope_of(retriever)
    paths = proven_paths(scope)
    choice = choose_retrieval(query, mode, paths)
    by_id = {PATH_PREFIX + str(p["template_digest"]): p for p in paths}
    chosen = by_id.get(choice.option_id) if choice.decided else None
    if choice.decided and choice.logged and choice.record_id:
        template = None if chosen is None else chosen.get("template")
        run = PendingRun(
            choice.record_id, template, scope=scope, skill_ref=current_skill_ref()
        )
        ledger_of(retriever).put(query, run)
        note_retrieval(retriever, query)
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
TEXT_FIELDS = ("content", "text", "description", "summary", "name")


def unit_text(node: Mapping[str, Any]) -> str:
    """A unit's text as ingestion embedded and classified it."""
    return next((str(node[k]) for k in TEXT_FIELDS if node.get(k)), "")


def content_class_of(node: Mapping[str, Any]) -> str:
    """The unit's ingestion content class (EH-269's deterministic classifier),
    the label EG's per-class usage and AU's admission feedback key on."""
    from agent_utilities.knowledge_graph.ingestion.embedding_admission import (
        classify_unit,
    )

    declared = node.get("content_class")
    if declared:
        return str(declared)
    text = unit_text(node)
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
    kept = [n for n in nodes if n.get("id") is not None]
    run.returned = [(str(n["id"]), content_class_of(n)) for n in kept]
    run.classes = {str(n["id"]): cls for n in kept if (cls := _schema_class(n))}
    run.query = _query_vector(retriever, query)


def _schema_class(node: Mapping[str, Any]) -> str | None:
    value = node.get("type") or node.get("label")
    return str(value) if value else None


def seeded_template(run: PendingRun, cited: Sequence[str]) -> dict[str, Any] | None:
    """The typed plan template a template-planned run executed, so a judged
    success can become a proven path (EH-394): anchored on the schema class
    most of its CITED units share (else its returned units'), ranked the way
    the hybrid retriever ranks (vector + text fused when it had a vector), under
    the run's task class and composed schema, with the skill it ran under."""
    classes = [run.classes[i] for i in cited if i in run.classes] or list(
        run.classes.values()
    )
    if run.scope is None or not classes:
        return None
    anchor = sorted(Counter(classes).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    vector = run.query is not None
    return {
        "task_class": run.scope.task_class,
        "composed_digest": run.scope.composed_digest,
        "policy_version": PLAN_POLICY_VERSION,
        "anchor_class": anchor,
        "edges": [],
        "rank": "fuse_rrf" if vector else "text",
        "slots": ["query_text", "query_vector"] if vector else ["query_text"],
        "skill_ref": run.skill_ref,
    }


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
        path=run.path or seeded_template(run, [str(i) for i in used_ids]),
    )
    try:
        session.send(op)
    except Exception as exc:
        logger.warning("retrieval outcome %s not recorded: %s", run.record_id, exc)
        return False
    return True


def _named(unit_id: str, text: str) -> bool:
    return len(unit_id) >= 3 and unit_id in text


def attest_answer(retrievals: Sequence[tuple[Any, str]], answer: Any) -> int:
    """Attest each pending retrieval of a finished run: the answer cites the
    returned units it names (by id). Returns how many were attested."""
    text = answer if isinstance(answer, str) else json.dumps(answer, default=str)
    attested = 0
    for retriever, query in dict.fromkeys(retrievals):
        run = ledger_of(retriever).get(query)
        if run is None:
            continue
        cited = [unit for unit, _ in run.returned if _named(unit, text)]
        retriever.record_answer_usage(cited, query=query)
        attested += 1
    return attested


__all__ = [
    "DEFAULT_TASK_CLASS",
    "MAX_PENDING_RUNS",
    "TEXT_FIELDS",
    "PathScope",
    "PendingRun",
    "RunLedger",
    "attest_answer",
    "attest_citations",
    "composed_digest_of",
    "content_class_of",
    "first_pass",
    "ledger_of",
    "note_returned",
    "path_plan",
    "path_scope_of",
    "plan_retrieval",
    "proven_paths",
    "run_path",
    "seeded_template",
    "unit_text",
]
