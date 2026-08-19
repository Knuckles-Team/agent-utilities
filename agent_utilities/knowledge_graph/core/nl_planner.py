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

# Keep the client-side contract explicitly versioned with the dependency-free
# ``eg-plan::uql::parser`` grammar.  UQL v1's predicate production accepts one
# bare identifier for a property (``name = 'x'``); it does not have Cypher's
# ``node.name`` or a nested ``props.name`` expression.  The engine remains the
# final parser/authority.  This small boundary prevents a known model-shaped
# spelling from reaching it and gives callers a stable evidence/error version.
UQL_GRAMMAR_VERSION = "eg-plan.uql.v1"

_UQL_IDENT_START = re.compile(r"[A-Za-z_]")
_UQL_IDENT_CONTINUE = re.compile(r"[A-Za-z0-9_]")


class UqlPlanError(ValueError):
    """A generated UQL query cannot satisfy the versioned engine grammar.

    This is deliberately distinct from an engine execution failure: callers can
    feed it into a bounded corrective-planning attempt without ever treating the
    invalid query as an empty result.
    """

    code = "uql_grammar_invalid"

    def __init__(self, message: str, *, query: str, at: int | None = None) -> None:
        self.query = query
        self.at = at
        self.grammar_version = UQL_GRAMMAR_VERSION
        super().__init__(
            f"{message} (grammar {UQL_GRAMMAR_VERSION})"
            + (f" at byte {at}" if at is not None else "")
        )


def _is_uql_ident_start(char: str) -> bool:
    return bool(char) and _UQL_IDENT_START.fullmatch(char) is not None


def _is_uql_ident_continue(char: str) -> bool:
    return bool(char) and _UQL_IDENT_CONTINUE.fullmatch(char) is not None


def canonicalize_uql_query(
    query: str,
) -> tuple[str, list[dict[str, str]]]:
    """Compile model-shaped UQL property references to the v1 surface syntax.

    The native parser's ``pred = prop comparison value`` production consumes a
    single identifier.  Models commonly borrow a property-graph JSON spelling
    and emit ``props.name`` for a code-context request; the dot is not a UQL
    token and the engine rejects it before producing any evidence.  ``props`` is
    the only wrapper we lower automatically: it is an unambiguous serialization
    wrapper, not a query alias.  Other dotted references (``node.name``,
    ``n.name`` and so on) fail closed so a bounded caller can replan them rather
    than silently changing their meaning.

    Strings are copied verbatim, including dots in source names and URLs.  This
    helper intentionally validates only the property-reference seam; the engine
    remains authoritative for the complete UQL grammar and execution semantics.
    """
    if not isinstance(query, str):
        raise UqlPlanError("UQL query must be text", query=str(query))

    out: list[str] = []
    corrections: list[dict[str, str]] = []
    i = 0
    n = len(query)
    quote: str | None = None

    while i < n:
        char = query[i]

        if quote is not None:
            out.append(char)
            if char == quote:
                # UQL's lexer accepts doubled quotes inside either quote style.
                if i + 1 < n and query[i + 1] == quote:
                    out.append(query[i + 1])
                    i += 2
                    continue
                quote = None
            elif char == "\\" and quote == '"' and i + 1 < n:
                # Match the engine lexer for the two supported double-quoted
                # escapes; copying the escaped byte is sufficient here.
                out.append(query[i + 1])
                i += 2
                continue
            i += 1
            continue

        if char in ("'", '"'):
            quote = char
            out.append(char)
            i += 1
            continue

        if char == "<":
            # The v1 lexer also admits a whitespace-free angle-bracketed IRI
            # (for example the target of `REASON <http://ex/Device>`).  Copy it
            # as one opaque token so its path dots are not mistaken for a
            # property qualifier; a numeric comparison (`year < 2024`) has no
            # closing IRI shape and falls through unchanged.
            close = query.find(">", i + 1)
            if close != -1:
                body = query[i + 1 : close]
                if body and ":" in body and not any(c.isspace() for c in body):
                    out.append(query[i : close + 1])
                    i = close + 1
                    continue

        if _is_uql_ident_start(char):
            start = i
            i += 1
            while i < n and _is_uql_ident_continue(query[i]):
                i += 1
            word = query[start:i]

            # A dotted identifier is the exact cross-seam defect this contract
            # owns.  Lower only the known model wrapper; reject every other
            # qualifier instead of inventing alias semantics for UQL v1.
            if (
                i + 1 < n
                and query[i] == "."
                and _is_uql_ident_start(query[i + 1])
            ):
                dot_at = i
                property_start = i + 1
                i = property_start + 1
                while i < n and _is_uql_ident_continue(query[i]):
                    i += 1
                prop = query[property_start:i]
                dotted = f"{word}.{prop}"
                if word.casefold() != "props":
                    raise UqlPlanError(
                        "UQL v1 WHERE properties must be bare identifiers; "
                        f"dotted reference {dotted!r} is not parseable",
                        query=query,
                        at=dot_at,
                    )
                out.append(prop)
                corrections.append(
                    {
                        "kind": "property_reference",
                        "from": dotted,
                        "to": prop,
                        "reason": "UQL v1 predicates use bare property identifiers",
                    }
                )
                continue

            out.append(word)
            continue

        if char == ".":
            # Preserve the two forms that are valid outside identifiers in the
            # v1 lexer: hop ranges (`1..2`) and decimal numbers (`2.0`, `.5`).
            if i + 1 < n and query[i + 1] == ".":
                out.extend((".", "."))
                i += 2
                continue
            if (i + 1 < n and query[i + 1].isdigit()) or (
                i > 0
                and query[i - 1].isdigit()
                and (i + 1 == n or query[i + 1] != ".")
            ):
                out.append(char)
                i += 1
                continue
            raise UqlPlanError(
                "UQL v1 does not define a standalone `.` token in a property "
                "expression; use one bare identifier",
                query=query,
                at=i,
            )

        out.append(char)
        i += 1

    return "".join(out), corrections


def _normalize_plan(parsed: dict[str, Any]) -> dict[str, Any]:
    """Apply the AU UQL boundary even when a caller injects another planner."""
    normalized = dict(parsed)
    if normalized.get("dialect") != "uql":
        normalized.setdefault("corrections", [])
        return normalized

    query = str(normalized.get("query", "")).strip()
    if not query:
        raise UqlPlanError("UQL query must not be empty", query=query)
    normalized["query"], corrections = canonicalize_uql_query(query)
    normalized["grammar_version"] = UQL_GRAMMAR_VERSION
    normalized["corrections"] = list(normalized.get("corrections") or []) + corrections
    return normalized

_SYSTEM_PROMPT = (
    "You are the query planner for a Knowledge Graph engine. Translate a "
    "natural-language request into a SINGLE read-only query the engine can execute. "
    "Choose one dialect:\n"
    f"  - uql:    the engine's native cross-modal Unified Query Language (PREFER this; "
    f"grammar {UQL_GRAMMAR_VERSION}). "
    "Pipeline form: MATCH (:Label) [WHERE prop > n AND ...] |> TRAVERSE -[:REL]->{1,2} "
    "|> RANK BY ~[1.0, 0.0, 0.0, 0.0] |> LIMIT k. Use it for graph traversal + "
    "filtering + vector ranking in one query.\n"
    "    UQL WHERE properties are ONE bare identifier only (for example "
    "`name = 'build_code_context'`); never emit `props.name`, `node.name`, "
    "an alias-qualified name, or bracket access. For a code-context request, "
    "use `MATCH (:Code) WHERE name = '<symbol>' |> LIMIT k`.\n"
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
        # vLLM at ``http://vllm.example/v1``) IS a usable planner endpoint: ``create_model``
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
        from agent_utilities.core.contextual_model import create_context_agent
        from agent_utilities.core.event_loop import run_sync_isolated
        from agent_utilities.core.model_factory import create_model

        model = create_model(role=self._role)
        agent = create_context_agent(model=model, system_prompt=system_prompt)

        # ``nl_query`` is a SYNC entrypoint but the MCP/gateway dispatch calls it from
        # inside a running event loop. ``agent.run_sync`` spins its own loop and raises
        # "This event loop is already running" when one is already active on this thread.
        # :func:`run_sync_isolated` detects that and runs the sync call on a worker
        # thread (which has no running loop) — the shared form of this exact guard,
        # also used by the sibling ``nl_query.nl_to_query`` / ``data_analyst``
        # call sites (BUG-2, kg-exhaustive-smoke.md).
        return str(run_sync_isolated(lambda: agent.run_sync(prompt)).output)

    def plan(
        self,
        text: str,
        *,
        schema_hint: str = "",
        dialect: str = "auto",
    ) -> dict[str, Any]:
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
        return _normalize_plan(parsed)


def _render_schema(schema: dict[str, Any], extra_hint: str = "") -> str:
    """Render the live schema snapshot (+ any caller hint) into a compact prompt block."""
    lines = [
        f"Schema (node labels): {', '.join(schema.get('node_labels') or []) or '(unknown)'}",
        f"Schema (SQL tables): {', '.join(schema.get('tables') or []) or '(none)'}",
        f"Schema (SQL columns): {schema.get('sql_columns') or '(unknown)'}",
        "Schema (UQL grammar): "
        f"{UQL_GRAMMAR_VERSION}; WHERE property references are bare identifiers "
        "only (for example `name`, never `props.name`).",
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
    max_corrections: int = 1,
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
    citations}`` — or a ``{..., error}`` on planning / execution failure.  Invalid
    UQL is never represented as an empty result: the direct path performs at most
    ``max_corrections`` bounded replans (default one), retaining an attempt trace
    and the grammar version in the evidence shape before returning a clean error.
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
    correction_budget = max(0, int(max_corrections))
    schema_hint_text = _render_schema(schema, schema_hint)
    attempts: list[dict[str, Any]] = []
    plan_text = text
    last_out: dict[str, Any] | None = None
    last_error = ""

    for attempt in range(1 + correction_budget):
        try:
            parsed = planner.plan(
                plan_text,
                schema_hint=schema_hint_text,
                dialect=dialect,
            )
            # Keep the guard at the execution boundary as well as in
            # ``AuNlPlanner``: injected planners are testable seams, not a way
            # to bypass the grammar contract before a query reaches GraphOS.
            parsed = _normalize_plan(parsed)
        except Exception as exc:  # noqa: BLE001 — bounded planning failure
            last_error = f"nl->query planning failed: {exc}"
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "phase": "planning",
                    "query": getattr(exc, "query", None),
                    "error_code": getattr(exc, "code", "planner_error"),
                    "error": last_error,
                    "grammar_version": UQL_GRAMMAR_VERSION,
                }
            )
            if attempt < correction_budget:
                previous = attempts[-1]
                plan_text = (
                    f"{text}\n\nThe previous generated query failed the "
                    f"{UQL_GRAMMAR_VERSION} contract and must be corrected. "
                    f"Error: {previous['error']}\n"
                    "Return one parseable, read-only query; never return an "
                    "empty answer as a substitute."
                )
            continue

        parsed_query = str(parsed["query"])
        plan_evidence = {
            "grammar_version": parsed.get("grammar_version")
            or (UQL_GRAMMAR_VERSION if parsed["dialect"] == "uql" else None),
            "dialect": parsed["dialect"],
            "query": parsed_query,
            "corrections": list(parsed.get("corrections") or []),
            "bounded": True,
        }
        out: dict[str, Any] = {
            "request": text,
            "dialect": parsed["dialect"],
            "generated_query": parsed_query,
            "planner": "agent-utilities-fleet-llm",
            "schema": schema,
            "plan": plan_evidence,
            "attempts": attempts,
        }
        last_out = out

        if _is_mutation(parsed_query):
            # Mutations are a hard refusal, never a self-correction candidate.
            out["error"] = "generated query is a mutation; refused (read-only surface)"
            return out

        if not execute:
            return out

        try:
            rows = _execute(engine, parsed["dialect"], parsed_query)
            rows = list(rows or [])[:limit]
            out["results"] = rows
            out["row_count"] = len(rows)
            out["citations"] = _citations(rows)
            return out
        except Exception as exc:  # noqa: BLE001 — bounded execution correction
            last_error = f"query execution failed: {exc}"
            step = {
                "attempt": attempt + 1,
                "phase": "execution",
                "dialect": parsed["dialect"],
                "query": parsed_query,
                "error": last_error,
                "grammar_version": plan_evidence["grammar_version"],
            }
            attempts.append(step)
            if attempt < correction_budget:
                plan_text = (
                    f"{text}\n\nYour previous {parsed['dialect']} query failed "
                    "and must be corrected.\n"
                    f"Previous query: {parsed_query}\nError: {last_error}\n"
                    "Generate one corrected read-only query grounded in the schema. "
                    "Do not turn a query error into an empty answer."
                )

    # Every bounded attempt failed.  Preserve the final candidate and trace so
    # EvidenceBundle can report a planner/engine error instead of a false empty
    # result, while keeping the response shape useful to the operator.
    if last_out is None:
        last_out = {
            "request": text,
            "planner": "agent-utilities-fleet-llm",
            "schema": schema,
            "plan": {"grammar_version": UQL_GRAMMAR_VERSION, "bounded": True},
        }
    last_out["attempts"] = attempts
    last_out["error"] = last_error or "nl->query planning failed"
    return last_out
