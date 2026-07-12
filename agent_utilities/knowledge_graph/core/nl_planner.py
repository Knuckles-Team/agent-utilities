#!/usr/bin/python
from __future__ import annotations

"""AU-as-engine NL→query planner (CONCEPT:AU-KG.query.ask-gateway-rest-twin).

The **agent-utilities half** of the epistemic-graph NL→query dual-mode. The engine
(EG-078/080) defines an ``NlPlanner`` seam: a natural-language string is turned into an
executable query STRING that runs through the engine's deterministic pipeline
(``eg_plan::uql::parse`` → the fused executor). The engine ships a *standalone* planner
(``UreqNlPlanner``) that POSTs to an OpenAI-compatible endpoint over its own pure-Rust
``ureq`` client — that path is complete.

This module is the **"LLM opt-out from AU"** path: when agent-utilities drives the engine,
the planning step runs on the SAME fleet LLM the rest of AU uses
(``core.model_factory.create_model`` — the local vLLM / configured provider) instead of the
engine's standalone ureq client. AU generates the query with its configured model, then
executes it through the EXISTING AU→engine query surfaces
(:meth:`~...engine_query.QueryMixin.uql` / ``sql`` / ``sparql`` / ``query_cypher``) — the
engine still runs the query through its deterministic executor, so the result is grounded
and verifiable, exactly like the engine-standalone seam. No new transport is introduced.

Kept additive + configurable:

* falls back to a clean error (never a crash) when no LLM is configured — see
  :func:`is_llm_configured`;
* reuses the KG-2.266 helpers (live schema grounding, mutation guard, citations);
* adds ``uql`` — the engine's native cross-modal language — as the preferred dialect,
  matching what the engine's own ``NlPlanner`` produces.
"""

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from .nl_query import _citations, _is_mutation, build_schema_context

logger = logging.getLogger(__name__)

#: Query dialects the AU planner may emit. ``uql`` is preferred — it is the engine's
#: native unified cross-modal language and the target the engine's ``NlPlanner`` seam
#: produces — but cypher/sql/sparql are accepted so the planner degrades to a familiar
#: dialect when the question maps cleanly onto one.
_DIALECTS = ("uql", "cypher", "sql", "sparql")

_SYSTEM_PROMPT = (
    "You are the query planner for a Knowledge Graph engine. Translate a "
    "natural-language request into a SINGLE read-only query the engine can execute. "
    "Choose one dialect:\n"
    "  - uql:    the engine's native cross-modal Unified Query Language (PREFER this). "
    "Pipeline form: MATCH (:Label) [WHERE prop > n AND ...] |> TRAVERSE -[:REL]->{1,2} "
    "|> RANK BY ~[1.0, 0.0, 0.0, 0.0] |> LIMIT k. Use it for graph traversal + "
    "filtering + vector ranking in one query.\n"
    "  - cypher: read-only Cypher over the property graph (MATCH ... RETURN ...).\n"
    "  - sql:    read-only SQL over the KG (SELECT ... FROM nodes ...). See the SQL "
    "schema note in the prompt for the real `nodes`/`edges` columns (label questions "
    "filter on `type`, not `label`; edges use `src`/`dst`/`rel`, not invented column "
    "names).\n"
    "  - sparql: SPARQL 1.1 SELECT/ASK over the RDF projection.\n"
    "Rules: emit ONLY one query, NEVER a mutation (no CREATE/MERGE/DELETE/INSERT/DROP/"
    "SET/UPDATE). Ground every label / table / column you reference in the provided "
    "schema — never invent a column name. "
    'Respond with ONLY a JSON object: {"dialect": "...", "query": "..."}.'
)


def is_llm_configured() -> bool:
    """True when agent-utilities has an LLM endpoint the fleet planner can reach.

    CONCEPT:AU-KG.query.ask-gateway-rest-twin — the clean-fallback gate. Returns ``False`` when nothing usable is
    configured (no OpenAI-compatible base URL, no provider API key, no model registry),
    so :func:`nl_query` can report a clear error instead of attempting a doomed model
    call. Best-effort and never raises — a config-load hiccup degrades to "not
    configured".
    """
    try:
        from agent_utilities.core.config import config

        # A configured fleet chat model (config.json ``chat_models`` — e.g. the local
        # vLLM at ``http://vllm.arpa/v1``) IS a usable planner endpoint: ``create_model``
        # routes an unmapped role to ``config.default_chat_model`` (see model_factory).
        # This is the SAME model delegation already uses, so recognize it here instead of
        # forcing the operator to also set the OPENAI_BASE_URL env var (config is the
        # single source of truth).
        if getattr(config, "chat_models", None):
            return True
        if getattr(config, "openai_base_url", None):
            return True
        for key in (
            "openai_api_key",
            "anthropic_api_key",
            "groq_api_key",
            "mistral_api_key",
            "gemini_api_key",
            "deepseek_api_key",
        ):
            if getattr(config, key, None):
                return True
        if getattr(config, "model_registry_path", None):
            return True
    except Exception as exc:  # noqa: BLE001 — a config hiccup means "not configured"
        logger.debug("is_llm_configured probe failed: %s", exc)
    return False


def _parse_plan(output: str) -> dict[str, str]:
    """Extract ``{dialect, query}`` from raw model output (tolerant of prose/fences)."""
    match = re.search(r"\{.*\}", output or "", re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in model output: {(output or '')[:200]!r}")
    obj = json.loads(match.group(0))
    dialect = str(obj.get("dialect", "")).strip().lower()
    query = str(obj.get("query", "")).strip()
    if dialect not in _DIALECTS:
        raise ValueError(f"unsupported dialect {dialect!r} (want one of {_DIALECTS})")
    if not query:
        raise ValueError("model returned an empty query")
    return {"dialect": dialect, "query": query}


class AuNlPlanner:
    """CONCEPT:AU-KG.query.ask-gateway-rest-twin — agent-utilities' configured fleet LLM AS the engine's NL planner.

    Mirrors the engine's EG-078 ``NlPlanner`` trait: :meth:`plan` turns a
    natural-language request (+ a schema hint) into an executable query STRING. The
    difference from the engine's standalone ``UreqNlPlanner`` is the model client — this
    runs on the SAME fleet LLM the rest of agent-utilities uses
    (``core.model_factory.create_model``, role ``planner``), so a caller that drives the
    engine from AU gets NL→query on the fleet model instead of the engine's own ureq
    client.

    ``run`` is an optional injection seam ``(prompt, system_prompt) -> raw_model_text`` —
    default ``None`` builds the AU LLM lazily on first use. Tests (and any caller that
    wants to substitute a model) pass their own ``run``.
    """

    def __init__(
        self,
        *,
        run: Callable[[str, str], str] | None = None,
        role: str = "planner",
    ) -> None:
        self._run = run
        self._role = role

    def _default_run(self, prompt: str, system_prompt: str) -> str:
        """Call the AU-configured fleet LLM once and return its raw text output."""
        import asyncio

        from pydantic_ai import Agent

        from agent_utilities.core.model_factory import create_model

        model = create_model(role=self._role)
        agent = Agent(model=model, system_prompt=system_prompt)

        def _call() -> str:
            return str(agent.run_sync(prompt).output)

        # ``nl_query`` is a SYNC entrypoint but the MCP/gateway dispatch calls it from
        # inside a running event loop. ``agent.run_sync`` spins its own loop and raises
        # "This event loop is already running" when one is already active on this thread.
        # Detect that and run the sync call on a worker thread (which has no running loop).
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return _call()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_call).result()

    def plan(
        self,
        text: str,
        *,
        schema_hint: str = "",
        dialect: str = "auto",
    ) -> dict[str, str]:
        """Turn ``text`` into ``{dialect, query}`` via the fleet LLM (returns a query STRING).

        ``schema_hint`` is a compact description of the live schema (node labels / SQL
        tables) so the model grounds its query in things that exist. ``dialect`` may pin
        the output to one of :data:`_DIALECTS` (``auto`` lets the model choose, preferring
        ``uql``). Raises on an unparseable / empty / mutation-shaped model response so the
        caller reports the failure rather than executing junk.
        """
        forced = dialect.strip().lower() if dialect and dialect != "auto" else ""
        if forced and forced not in _DIALECTS:
            raise ValueError(
                f"unsupported dialect {forced!r} (want one of {_DIALECTS})"
            )
        prompt = f"Request: {text}\n\n{schema_hint}".rstrip()
        if forced:
            prompt += f"\n\nYou MUST use the '{forced}' dialect.\n"
        raw = (self._run or self._default_run)(prompt, _SYSTEM_PROMPT)
        parsed = _parse_plan(raw)
        if forced and parsed["dialect"] != forced:
            parsed["dialect"] = forced
        return parsed


def _render_schema(schema: dict[str, Any], extra_hint: str = "") -> str:
    """Render the live schema snapshot (+ any caller hint) into a compact prompt block."""
    lines = [
        f"Schema (node labels): {', '.join(schema.get('node_labels') or []) or '(unknown)'}",
        f"Schema (SQL tables): {', '.join(schema.get('tables') or []) or '(none)'}",
        f"Schema (SQL columns): {schema.get('sql_columns') or '(unknown)'}",
    ]
    if extra_hint:
        lines.append(f"Hint: {extra_hint}")
    return "\n".join(lines)


def _execute(engine: Any, dialect: str, query: str) -> list[dict[str, Any]]:
    """Run the generated query through the matching AU→engine surface."""
    if dialect == "uql":
        uql_fn = getattr(engine, "uql", None)
        if not callable(uql_fn):
            raise RuntimeError(
                "engine has no UQL surface (build the server with the 'query' feature)"
            )
        return list(uql_fn(query) or [])
    if dialect == "sql":
        return engine.sql(query)
    if dialect == "sparql":
        return engine.sparql(query)
    return engine.query_cypher(query)


def nl_query(
    engine: Any,
    text: str,
    *,
    dialect: str = "auto",
    schema_hint: str = "",
    execute: bool = True,
    limit: int = 50,
    planner: AuNlPlanner | None = None,
) -> dict[str, Any]:
    """NL→query with agent-utilities' fleet LLM as the engine's planner (CONCEPT:AU-KG.query.ask-gateway-rest-twin).

    (1) grounds a live schema snapshot from ``engine``; (2) has the AU-configured fleet
    LLM (:class:`AuNlPlanner`) translate ``text`` into an executable query STRING in the
    best dialect (uql/cypher/sql/sparql); (3) submits that query to the engine through the
    EXISTING AU→engine surface (:func:`_execute` → ``engine.uql``/``sql``/``sparql``/
    ``query_cypher``), so the engine runs it through its own deterministic executor.

    Additive + configurable: when no LLM is configured (and no ``planner`` injected) it
    returns a clean error instead of crashing (:func:`is_llm_configured`). A generated
    mutation is refused (read-only surface). ``execute=False`` previews the query without
    running it. Pass a ``planner`` to substitute the model (tests / a caller-owned model).

    Returns ``{request, dialect, generated_query, planner, schema, results, row_count,
    citations}`` — or a ``{..., error}`` on planning / execution failure.
    """
    if not text or not text.strip():
        return {"error": "empty request"}

    forced = dialect.strip().lower() if dialect and dialect != "auto" else ""
    if forced and forced not in _DIALECTS:
        return {"error": f"unsupported dialect {forced!r} (want one of {_DIALECTS})"}

    if planner is None:
        if not is_llm_configured():
            return {
                "error": (
                    "nl->query planning unavailable: no LLM configured. Set "
                    "OPENAI_BASE_URL (the fleet vLLM), a provider API key, or a model "
                    "registry to enable the agent-utilities NL planner."
                )
            }
        planner = AuNlPlanner()

    schema = build_schema_context(engine)
    try:
        parsed = planner.plan(
            text,
            schema_hint=_render_schema(schema, schema_hint),
            dialect=dialect,
        )
    except Exception as exc:  # noqa: BLE001 — planning failure is reported, not raised
        return {"error": f"nl->query planning failed: {exc}", "schema": schema}

    out: dict[str, Any] = {
        "request": text,
        "dialect": parsed["dialect"],
        "generated_query": parsed["query"],
        "planner": "agent-utilities-fleet-llm",
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
