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

_SYSTEM_PROMPT = (
    "You translate a natural-language question into a single read-only query against "
    "a Knowledge Graph engine. You may use one of these dialects:\n"
    "  - cypher: read-only Cypher over the property graph (MATCH ... RETURN ...).\n"
    "  - sql:    read-only SQL over the KG + user tables (SELECT ... FROM nodes / "
    "FROM <table> ...). The node table is `nodes`.\n"
    "  - sparql: SPARQL 1.1 SELECT/ASK over the RDF projection.\n"
    "Rules: emit ONLY a single query, never a mutation (no CREATE/MERGE/DELETE/"
    "INSERT/DROP/UPDATE). Prefer cypher unless the question is clearly relational "
    "(then sql) or clearly about RDF/ontology triples (then sparql). Ground every "
    "label / table you reference in the provided schema. "
    'Respond with ONLY a JSON object: {"dialect": "...", "query": "..."}.'
)


def build_schema_context(engine: Any, *, max_labels: int = 60) -> dict[str, Any]:
    """Collect a compact, live schema snapshot to ground the model.

    Pulls distinct node labels (via Cypher) and user SQL tables (via the SQL surface).
    Every step is best-effort — a cold / partial engine yields an empty section, never
    an error.
    """
    labels: list[str] = []
    try:
        rows = engine.query_cypher(
            "MATCH (n) RETURN DISTINCT labels(n) AS labels LIMIT 200"
        )
        seen: dict[str, None] = {}
        for row in rows or []:
            for lbl in row.get("labels") or []:
                if isinstance(lbl, str):
                    seen.setdefault(lbl, None)
        labels = list(seen)[:max_labels]
    except Exception as exc:  # noqa: BLE001 — schema is advisory
        logger.debug("schema label probe failed: %s", exc)

    tables: list[str] = []
    try:
        from .table_ingest import list_tables

        tables = list_tables(engine)
    except Exception as exc:  # noqa: BLE001
        logger.debug("schema table probe failed: %s", exc)

    return {"node_labels": labels, "tables": tables}


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


def _execute(engine: Any, dialect: str, query: str) -> list[dict[str, Any]]:
    """Run the generated query through the matching engine surface."""
    if dialect == "sql":
        return engine.sql(query)
    if dialect == "sparql":
        return engine.sparql(query)
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
) -> dict[str, Any]:
    """Translate ``question`` to a query, execute it, and return a grounded answer.

    Args:
        engine: the live KG engine (provides ``query_cypher`` / ``sql`` / ``sparql``).
        question: the natural-language question.
        dialect: ``auto`` (let the model choose) or pin to ``cypher``/``sql``/``sparql``.
        execute: when False, return only the generated query (dry-run / preview).
        limit: max result rows returned.

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
    )
    if forced:
        prompt += f"\nYou MUST use the '{forced}' dialect.\n"

    try:
        model = create_model(role="generator")
        agent = Agent(model=model, system_prompt=_SYSTEM_PROMPT)
        result = agent.run_sync(prompt)
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
        rows = _execute(engine, parsed["dialect"], parsed["query"])
        rows = list(rows or [])[:limit]
        out["results"] = rows
        out["row_count"] = len(rows)
        out["citations"] = _citations(rows)
    except Exception as exc:  # noqa: BLE001 — execution error reported with the query
        out["error"] = f"query execution failed: {exc}"
    return out
