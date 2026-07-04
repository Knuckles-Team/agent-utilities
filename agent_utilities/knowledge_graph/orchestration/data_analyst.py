#!/usr/bin/python
from __future__ import annotations

"""DB-GPT-style text2sql data-analysis agent loop (CONCEPT:AU-KG.query.data-gateway-rest-twin).

The **agent** half of NL data analysis over the epistemic-graph engine. Where
:mod:`~agent_utilities.knowledge_graph.core.nl_planner` (CONCEPT:AU-KG.query.ask-gateway-rest-twin) is a *single*
NL→query→execute shot, this module wraps that seam in a bounded, DB-GPT-style ReAct loop
so a single natural-language *question* is answered end-to-end:

    schema-link  →  plan (NL→query via KG-2.305)  →  execute  →  inspect
                        ↑                                 │
                        └──── self-correct on error ──────┘   (bounded)
                                                          │
                                                          ▼
                                            synthesize NL answer

Steps (mirrors DB-GPT's data-analysis agent):

1. **schema-link** — pull the live schema snapshot (node labels + SQL tables) and pick the
   subset most relevant to the question (deterministic token overlap — needs no LLM), so the
   planner is grounded on the *right* tables, not the whole catalogue.
2. **plan** — reuse the KG-2.305 :class:`~...nl_planner.AuNlPlanner` (the AU fleet LLM as the
   engine's NL planner) to turn the question + linked schema into an executable query STRING.
3. **execute** — run it through the existing AU→engine surface
   (:func:`~...nl_planner._execute` → ``uql``/``sql``/``sparql``/``query_cypher``), so the
   engine's own deterministic executor runs the query (grounded + verifiable).
4. **self-correct** — on a planning-parse or execution error, feed the failing query + the
   error text back to the planner for a *bounded* number of retries (``max_corrections``),
   then give up with a clean error carrying the full attempt trace.
5. **synthesize** — turn the winning rows into a natural-language answer (the AU fleet LLM,
   injectable), degrading to a deterministic summary when no LLM is configured.

Kept additive + configurable, exactly like KG-2.305: when nothing is configured (and no
``planner`` is injected) it returns a **clean error**, never a crash — see
:func:`~...nl_planner.is_llm_configured`. Mutations are refused (read-only surface). All the
LLM seams (``planner`` for query generation, ``synthesize`` for the final answer) are
injectable so the loop is fully exercisable with a mock LLM + a mock engine.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

from ..core import nl_planner, nl_query
from ..core.nl_planner import AuNlPlanner  # re-exported for type annotations

logger = logging.getLogger(__name__)

#: Prompt for the (optional) answer-synthesis LLM call.
_ANSWER_SYSTEM_PROMPT = (
    "You are a data analyst. Given a user's question and the JSON rows returned by a "
    "database query, write a concise, direct natural-language answer to the question, "
    "grounded ONLY in the rows. State counts / values explicitly. If the rows are empty, "
    "say no matching data was found. Do not invent data that is not in the rows."
)

#: Tokens too generic to be useful for schema-linking (question stop-words).
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "how",
        "many",
        "much",
        "what",
        "which",
        "who",
        "are",
        "was",
        "were",
        "with",
        "from",
        "that",
        "this",
        "count",
        "list",
        "show",
        "give",
        "all",
        "get",
        "find",
        "per",
        "each",
        "have",
        "has",
        "does",
        "did",
        "into",
        "over",
        "than",
        "then",
        "top",
        "most",
        "least",
        "average",
        "avg",
        "sum",
        "total",
        "number",
        "there",
        "their",
        "them",
        "about",
    }
)


def _tokens(text: str) -> set[str]:
    """Lowercased alphanumeric word-stems (≥3 chars, non-stopword) for overlap scoring."""
    out: set[str] = set()
    word = ""
    for ch in (text or "").lower():
        if ch.isalnum():
            word += ch
        else:
            if len(word) >= 3 and word not in _STOPWORDS:
                out.add(word)
                # singular/plural fold so "orders" matches table "order"
                if word.endswith("s"):
                    out.add(word[:-1])
            word = ""
    if len(word) >= 3 and word not in _STOPWORDS:
        out.add(word)
        if word.endswith("s"):
            out.add(word[:-1])
    return out


def _relevance(name: str, q_tokens: set[str]) -> int:
    """Overlap score between a schema name and the question tokens (CONCEPT:AU-KG.query.data-gateway-rest-twin)."""
    n_tokens = _tokens(name)
    if not n_tokens:
        return 0
    score = len(n_tokens & q_tokens)
    # Substring match (e.g. question token "customer" vs table "customers") — cheap boost.
    low = name.lower()
    for tok in q_tokens:
        if tok in low or low in tok:
            score += 1
    return score


def schema_link(
    question: str, schema: dict[str, Any], *, max_each: int = 8
) -> dict[str, list[str]]:
    """Pick the schema labels / tables most relevant to ``question`` (CONCEPT:AU-KG.query.data-gateway-rest-twin).

    Deterministic token-overlap scoring — needs no LLM, so schema-linking works in the
    clean-fallback path too. Returns ``{"tables": [...], "node_labels": [...]}`` limited to
    the top ``max_each`` of each. When nothing scores (a question that shares no vocabulary
    with the schema) it falls back to *all* names, so the planner is never starved.
    """
    q_tokens = _tokens(question)
    out: dict[str, list[str]] = {}
    for kind in ("tables", "node_labels"):
        names = [n for n in (schema.get(kind) or []) if isinstance(n, str)]
        scored = [(n, _relevance(n, q_tokens)) for n in names]
        hits = [n for n, s in sorted(scored, key=lambda x: -x[1]) if s > 0]
        out[kind] = (hits or names)[:max_each]
    return out


def _render_linked_schema(linked: dict[str, list[str]], extra_hint: str = "") -> str:
    """Compact prompt block emphasising the schema-linked tables/labels for the planner."""
    lines = [
        "Relevant node labels: "
        + (", ".join(linked.get("node_labels") or []) or "(unknown)"),
        "Relevant SQL tables: " + (", ".join(linked.get("tables") or []) or "(none)"),
    ]
    if extra_hint:
        lines.append(f"Hint: {extra_hint}")
    return "\n".join(lines)


def _fallback_answer(question: str, rows: list[dict[str, Any]]) -> str:
    """Deterministic NL answer when no synthesis LLM is available (clean fallback)."""
    n = len(rows)
    if n == 0:
        return f"No matching data was found for: {question}"
    first = rows[0]
    if isinstance(first, dict):
        preview = ", ".join(
            f"{k}={first[k]}" for k in list(first)[:4] if first.get(k) is not None
        )
    else:  # pragma: no cover - rows are dicts from the engine surfaces
        preview = str(first)
    plural = "s" if n != 1 else ""
    return f"Found {n} result row{plural} for: {question}. Top row: {preview}."


class DataAnalystAgent:
    """DB-GPT-style text2sql data-analysis agent (CONCEPT:AU-KG.query.data-gateway-rest-twin).

    A bounded ReAct-ish loop over the engine: schema-link the question, generate a query
    with the KG-2.305 planner, execute it, self-correct on error (bounded), then synthesize
    a natural-language answer. Every LLM seam is injectable so the loop runs against a mock
    LLM + mock engine:

    * ``planner`` — the KG-2.305 :class:`~...nl_planner.AuNlPlanner` used to generate (and
      re-generate, on correction) the query. ``None`` builds the AU fleet planner lazily,
      gated by :func:`~...nl_planner.is_llm_configured` (clean-fallback contract).
    * ``synthesize`` — ``(question, rows) -> str`` for the final NL answer. ``None`` uses the
      AU fleet LLM when configured, else the deterministic :func:`_fallback_answer`.
    * ``max_corrections`` — how many extra self-correction attempts after the first plan
      (``0`` disables the loop; default ``2``, i.e. up to 3 total attempts).
    """

    def __init__(
        self,
        engine: Any,
        *,
        planner: AuNlPlanner | None = None,
        synthesize: Callable[[str, list[dict[str, Any]]], str] | None = None,
        max_corrections: int = 2,
        limit: int = 50,
    ) -> None:
        self._engine = engine
        self._planner = planner
        self._synthesize = synthesize
        self._max_corrections = max(0, int(max_corrections))
        self._limit = max(1, int(limit))

    # -- synthesis ---------------------------------------------------------
    def _default_synthesize(self, question: str, rows: list[dict[str, Any]]) -> str:
        """Synthesize the NL answer via the AU fleet LLM; deterministic fallback if none."""
        if not nl_planner.is_llm_configured():
            return _fallback_answer(question, rows)
        try:
            from pydantic_ai import Agent

            from agent_utilities.core.model_factory import create_model

            model = create_model(role="generator")
            agent = Agent(model=model, system_prompt=_ANSWER_SYSTEM_PROMPT)
            payload = json.dumps(rows[:20], default=str)
            prompt = f"Question: {question}\n\nRows (JSON): {payload}"
            return str(agent.run_sync(prompt).output).strip() or _fallback_answer(
                question, rows
            )
        except Exception as exc:  # noqa: BLE001 — synthesis failure degrades gracefully
            logger.debug("answer synthesis failed, using fallback: %s", exc)
            return _fallback_answer(question, rows)

    # -- the loop ----------------------------------------------------------
    def analyze(self, question: str, *, dialect: str = "auto") -> dict[str, Any]:
        """Answer ``question`` end-to-end via the bounded loop (CONCEPT:AU-KG.query.data-gateway-rest-twin).

        Returns ``{question, answer, dialect, query, results, row_count, citations,
        linked_schema, attempts}`` on success — where ``attempts`` is the full trace of
        every (query, error) the loop tried — or ``{..., error}`` when planning is
        unavailable (no LLM) or every attempt (incl. corrections) failed.
        """
        if not question or not question.strip():
            return {"error": "empty question"}

        planner = self._planner
        if planner is None:
            if not nl_planner.is_llm_configured():
                return {
                    "error": (
                        "data analysis unavailable: no LLM configured. Set "
                        "OPENAI_BASE_URL (the fleet vLLM), a provider API key, or a model "
                        "registry to enable the agent-utilities data-analyst agent."
                    )
                }
            planner = nl_planner.AuNlPlanner()

        # 1. schema-link
        schema = nl_planner.build_schema_context(self._engine)
        linked = schema_link(question, schema)
        base_hint = _render_linked_schema(linked)

        attempts: list[dict[str, Any]] = []
        last_error: str | None = None

        # 2-4. plan → execute → (bounded) self-correct
        for attempt in range(1 + self._max_corrections):
            hint = base_hint
            plan_text = question
            if last_error and attempts:
                prev = attempts[-1]
                # Feed the failing query + error back so the planner can repair it.
                plan_text = (
                    f"{question}\n\nYour previous {prev.get('dialect', '')} query "
                    f"FAILED and must be corrected.\nPrevious query: "
                    f"{prev.get('query', '')}\nError: {last_error}\n"
                    "Generate a corrected read-only query grounded in the schema above."
                )

            try:
                parsed = planner.plan(plan_text, schema_hint=hint, dialect=dialect)
            except Exception as exc:  # noqa: BLE001 — planning error is correctable
                last_error = f"planning failed: {exc}"
                attempts.append({"attempt": attempt + 1, "error": last_error})
                continue

            step: dict[str, Any] = {
                "attempt": attempt + 1,
                "dialect": parsed["dialect"],
                "query": parsed["query"],
            }

            if nl_query._is_mutation(parsed["query"]):
                # A mutation is a hard refusal, not a correctable error.
                step["error"] = (
                    "generated query is a mutation; refused (read-only surface)"
                )
                attempts.append(step)
                return {
                    "question": question,
                    "error": step["error"],
                    "dialect": parsed["dialect"],
                    "query": parsed["query"],
                    "linked_schema": linked,
                    "attempts": attempts,
                }

            try:
                rows = list(
                    nl_planner._execute(
                        self._engine, parsed["dialect"], parsed["query"]
                    )
                    or []
                )
            except Exception as exc:  # noqa: BLE001 — execution error → self-correct
                last_error = f"execution failed: {exc}"
                step["error"] = last_error
                attempts.append(step)
                continue

            # 5. success → synthesize
            rows = rows[: self._limit]
            attempts.append(step)
            synth = self._synthesize or self._default_synthesize
            answer = synth(question, rows)
            return {
                "question": question,
                "answer": answer,
                "dialect": parsed["dialect"],
                "query": parsed["query"],
                "results": rows,
                "row_count": len(rows),
                "citations": nl_query._citations(rows),
                "linked_schema": linked,
                "attempts": attempts,
            }

        # every attempt (incl. corrections) failed
        return {
            "question": question,
            "error": (
                f"data analysis failed after {len(attempts)} attempt(s): {last_error}"
            ),
            "linked_schema": linked,
            "attempts": attempts,
        }


def ask_data(
    engine: Any,
    question: str,
    *,
    dialect: str = "auto",
    max_corrections: int = 2,
    limit: int = 50,
    planner: AuNlPlanner | None = None,
    synthesize: Callable[[str, list[dict[str, Any]]], str] | None = None,
) -> dict[str, Any]:
    """One-call DB-GPT-style data-analysis over the engine (CONCEPT:AU-KG.query.data-gateway-rest-twin).

    Thin convenience wrapper around :class:`DataAnalystAgent` — mirrors the shape of
    :func:`~...nl_planner.nl_query` (KG-2.305) but runs the full multi-step loop (schema-link
    → plan → execute → self-correct → synthesize) and returns a synthesized NL ``answer``
    alongside the ``query`` used and the ``results`` rows. Additive + configurable: a clean
    error (never a crash) when no LLM is configured and no ``planner`` is injected.
    """
    return DataAnalystAgent(
        engine,
        planner=planner,
        synthesize=synthesize,
        max_corrections=max_corrections,
        limit=limit,
    ).analyze(question, dialect=dialect)
