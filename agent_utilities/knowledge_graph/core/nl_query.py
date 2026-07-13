#!/usr/bin/python
from __future__ import annotations

"""Natural-language → query (CONCEPT:AU-KG.ingest.mirror-inbound).

An LLM→query layer: given a natural-language question plus the live KG / table schema
context, the model emits an executable query in one of the engine's query dialects
(``cypher`` over the property graph, ``sql`` over the KG + user tables, or ``sparql``
over the RDF projection), and we execute it through the matching engine surface.

Kept grounded:

* the prompt is anchored to the *actual* schema (node labels + user tables pulled
  live from the engine), so the model emits queries over things that exist;
* the result always carries the generated query back (auditability) plus citations
  (the node / source ids the answer rows touch), so it is verifiable, not a black box.

Reuses ``core.model_factory.create_model`` (reasoning_effort defaults to ``"none"`` —
fast, content-bearing completions) exactly like the other one-shot LLM call sites.
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_DIALECTS = ("cypher", "sql", "sparql")

#: The engine's SQL surface projects the property graph as two schema-on-read
#: DataFusion tables (``eg-query::sql::providers::infer_nodes``/``infer_edges``).
#: A node's cypher LABEL is NOT a distinct queryable column — it is whatever value
#: lives in the node's own ``type`` property (the engine's Cypher label-index/
#: ``labels(n)`` matcher checks ``type`` first, then ``node_type``, then ``label``
#: as scalars, then a ``labels`` array). Every observed node/edge property becomes
#: its own SQL column, so a bare ``label`` column often DOES exist (as an ordinary,
#: usually-empty property) — filtering on it silently returns 0 rows instead of
#: erroring, which is exactly the bug this grounds against.
_SQL_SCHEMA_NOTE = (
    "SQL table `nodes`: fixed `id` (Utf8 node id) + `props` (Binary raw blob) + "
    "one column per node property (schema-on-read, so columns vary by graph). "
    "The node's CYPHER LABEL / kind is NOT its own column — it is the `type` "
    "property (fallback `node_type`, then `label`). ALWAYS filter label/kind "
    "questions with `WHERE type = '<Label>'`. Do NOT use a `label` column for "
    "this — a same-named `label` property often exists but is unrelated/empty, "
    "so `WHERE label = ...` silently returns 0 rows instead of erroring.\n"
    "SQL table `edges`: FIXED columns only — `src` (source node id), `dst` "
    "(target node id), `rel` (relationship/edge-type string), `props` (Binary "
    "raw blob). There is NO `source_node_id`/`target_node_id`/`type` column on "
    "edges — use `src`/`dst`/`rel`."
)

_SYSTEM_PROMPT = (
    "You translate a natural-language question into a single read-only query against "
    "a Knowledge Graph engine. You may use one of these dialects:\n"
    "  - cypher: read-only Cypher over the property graph (MATCH ... RETURN ...).\n"
    "  - sql:    read-only SQL over the KG + user tables (SELECT ... FROM nodes / "
    "FROM <table> ...). See the SQL schema note in the prompt for the real "
    "`nodes`/`edges` columns (label questions filter on `type`, not `label`; "
    "edges use `src`/`dst`/`rel`, not invented column names).\n"
    "  - sparql: SPARQL 1.1 SELECT/ASK over the RDF projection.\n"
    "Rules: emit ONLY a single query, never a mutation (no CREATE/MERGE/DELETE/"
    "INSERT/DROP/UPDATE). Prefer cypher unless the question is clearly relational "
    "(then sql) or clearly about RDF/ontology triples (then sparql). Ground every "
    "label / table / column you reference in the provided schema — never invent a "
    "column name. "
    'Respond with ONLY a JSON object: {"dialect": "...", "query": "..."}.'
)


def build_schema_context(engine: Any, *, max_labels: int = 60) -> dict[str, Any]:
    """Collect a compact, live schema snapshot to ground the model.

    Pulls distinct node labels (via Cypher) and user SQL tables (via the SQL surface),
    plus a fixed ``sql_columns`` note describing the REAL `nodes`/`edges` SQL columns
    and the cypher-label-to-SQL-column mapping (CONCEPT:AU-KG.query.ask-gateway-rest-twin
    grounding fix — see ``_SQL_SCHEMA_NOTE``). Every step is best-effort — a cold /
    partial engine yields an empty section, never an error.

    Label probing reads ``n.type``/``n.node_type``/``n.label`` directly rather than the
    Cypher ``labels(n)`` function: the engine's Cypher executor does not implement
    ``labels(n)`` as a callable expression (it always evaluates to null), so a query
    built around it silently returns no labels at all. Reading the scalar properties
    the engine's own label index keys on (mirroring ``node_has_label`` in
    ``eg-query::cypher::exec``) is the grounded equivalent. Dedup is done client-side
    because the engine's ``DISTINCT`` does not reliably collapse duplicates for this
    query shape either.
    """
    labels: list[str] = []
    try:
        rows = engine.query_cypher(
            "MATCH (n) RETURN n.type AS t, n.node_type AS nt, n.label AS lb "
            f"LIMIT {max(max_labels * 20, 2000)}"
        )
        seen: dict[str, None] = {}
        for row in rows or []:
            for key in ("t", "nt", "lb"):
                val = row.get(key)
                if isinstance(val, str) and val:
                    seen.setdefault(val, None)
                    break
        labels = list(seen)[:max_labels]
    except Exception as exc:  # noqa: BLE001 — schema is advisory
        logger.debug("schema label probe failed: %s", exc)

    tables: list[str] = []
    try:
        from .table_ingest import list_tables

        tables = list_tables(engine)
    except Exception as exc:  # noqa: BLE001
        logger.debug("schema table probe failed: %s", exc)

    return {
        "node_labels": labels,
        "tables": tables,
        "sql_columns": _SQL_SCHEMA_NOTE,
    }


def _parse_llm_query(output: str) -> dict[str, str]:
    """Extract ``{dialect, query}`` from the model output (tolerant of prose/fences)."""
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in model output: {output[:200]!r}")
    obj = json.loads(match.group(0))
    dialect = str(obj.get("dialect", "")).strip().lower()
    query = str(obj.get("query", "")).strip()
    if dialect not in _DIALECTS:
        raise ValueError(f"unsupported dialect {dialect!r} (want one of {_DIALECTS})")
    if not query:
        raise ValueError("model returned an empty query")
    return {"dialect": dialect, "query": query}


def _is_mutation(query: str) -> bool:
    """True if the query contains a write keyword (defence-in-depth)."""
    no_lit = re.sub(r"'[^']*'|\"[^\"]*\"", "", query)
    return bool(
        re.search(
            r"\b(CREATE|MERGE|DELETE|REMOVE|DROP|SET|INSERT|UPDATE|LOAD)\b",
            no_lit,
            re.IGNORECASE,
        )
    )


def _execute(
    engine: Any, dialect: str, query: str, *, include_epistemic: bool = False
) -> list[dict[str, Any]]:
    """Run the generated query through the matching engine surface.

    ``include_epistemic`` (CONCEPT:AU-KB-CURRENCY) only applies to the ``cypher``
    dialect — ``engine.sql``/``engine.sparql`` have no epistemic-envelope
    parameter, so it is silently not passed for those (never raises).
    """
    if dialect == "sql":
        return engine.sql(query)
    if dialect == "sparql":
        return engine.sparql(query)
    if include_epistemic:
        # Only pass the new kwarg when actually requested — keeps the default
        # call shape byte-identical for any `query_cypher` implementation
        # (real or test double) that predates this parameter.
        return engine.query_cypher(query, include_epistemic=True)
    return engine.query_cypher(query)


def _citations(rows: list[dict[str, Any]], limit: int = 25) -> list[str]:
    """Pull provenance ids out of result rows (node id / source_uri / iri)."""
    out: list[str] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for key in ("id", "node_id", "source_uri", "iri", "uri", "name"):
            val = row.get(key)
            if isinstance(val, str) and val and val not in out:
                out.append(val)
                break
        if len(out) >= limit:
            break
    return out


def nl_to_query(
    engine: Any,
    question: str,
    *,
    dialect: str = "auto",
    execute: bool = True,
    limit: int = 50,
    include_epistemic: bool = False,
) -> dict[str, Any]:
    """Translate ``question`` to a query, execute it, and return a grounded answer.

    Args:
        engine: the live KG engine (provides ``query_cypher`` / ``sql`` / ``sparql``).
        question: the natural-language question.
        dialect: ``auto`` (let the model choose) or pin to ``cypher``/``sql``/``sparql``.
        execute: when False, return only the generated query (dry-run / preview).
        limit: max result rows returned.
        include_epistemic: Opt-in (CONCEPT:AU-KB-CURRENCY). Only takes effect
            when the model (or a forced ``dialect``) resolves to ``cypher`` —
            see :func:`_execute`. When true and honored, ``results`` holds
            :class:`~agent_utilities.knowledge_graph.core.epistemic_row.EpistemicRow`
            instances instead of plain dicts, and ``citations`` degrades to
            ``[]`` (the id/source-uri extraction only recognizes plain-dict
            rows) rather than raising.

    Returns ``{question, dialect, generated_query, results, row_count, citations,
    schema}`` — or ``{error: ...}`` on failure.
    """
    from pydantic_ai import Agent

    from agent_utilities.core.model_factory import create_model

    if not question or not question.strip():
        return {"error": "empty question"}

    schema = build_schema_context(engine)
    forced = dialect.strip().lower() if dialect and dialect != "auto" else ""
    if forced and forced not in _DIALECTS:
        return {"error": f"unsupported dialect {forced!r} (want one of {_DIALECTS})"}

    prompt = (
        f"Question: {question}\n\n"
        f"Schema (node labels): {', '.join(schema['node_labels']) or '(unknown)'}\n"
        f"Schema (SQL tables): {', '.join(schema['tables']) or '(none)'}\n"
        f"Schema (SQL columns): {schema['sql_columns']}\n"
    )
    if forced:
        prompt += f"\nYou MUST use the '{forced}' dialect.\n"

    try:
        from agent_utilities.core.event_loop import run_sync_isolated

        model = create_model(role="generator")
        agent = Agent(model=model, system_prompt=_SYSTEM_PROMPT)
        # BUG-2 (kg-exhaustive-smoke.md): ``agent.run_sync`` spins its own event
        # loop and raises "This event loop is already running" when called (as
        # every real graph_ask/ask_data MCP/REST call is) from inside the
        # gateway's already-running loop. Sibling call site to the fixed
        # ``nl_planner.AuNlPlanner._default_run`` — same worker-thread guard.
        result = run_sync_isolated(lambda: agent.run_sync(prompt))
        parsed = _parse_llm_query(result.output)
    except Exception as exc:  # noqa: BLE001 — LLM / parse failure is reported, not raised
        return {"error": f"nl->query generation failed: {exc}", "schema": schema}

    if forced and parsed["dialect"] != forced:
        parsed["dialect"] = forced

    out: dict[str, Any] = {
        "question": question,
        "dialect": parsed["dialect"],
        "generated_query": parsed["query"],
        "schema": schema,
    }

    if _is_mutation(parsed["query"]):
        out["error"] = "generated query is a mutation; refused (read-only surface)"
        return out

    if not execute:
        return out

    try:
        rows = _execute(
            engine,
            parsed["dialect"],
            parsed["query"],
            include_epistemic=include_epistemic,
        )
        rows = list(rows or [])[:limit]
        out["results"] = rows
        out["row_count"] = len(rows)
        out["citations"] = _citations(rows)
    except Exception as exc:  # noqa: BLE001 — execution error reported with the query
        out["error"] = f"query execution failed: {exc}"
    return out
