"""LLM-planner routing strategy prompt fragments (R9 — CONCEPT:ORCH-1.1).

R9 is the HTN-style subtask-specification + wide-search-orchestration guidance
injected into the router's planning system prompt. Extracted verbatim so the
instruction text is owned in one place; the monolith concatenates the returned
block into its full planner prompt.

R10 (RLM parse + fallback parser) and R11 (agentic/topological detection ->
model escalation) remain orchestration-bound in the router and delegate HTN
decomposition to ``graph.planning.Planner``; they are tracked in the capability
ledger and exercised by the router characterization tests.
"""

from __future__ import annotations

from typing import Any

SUBTASK_AND_WIDESEARCH_INSTRUCTIONS = (
    "### SUBTASK SPECIFICATION (CONCEPT:ORCH-1.1)\n"
    "For EACH step in your plan, include a 'refined_subtask' — a focused, "
    "specific instruction tailored for that specialist. Do NOT just repeat the "
    "user query. Instead, decompose it into a targeted sub-goal. Example: if the "
    "user asks 'build a REST API with auth', the python_programmer step should "
    "get refined_subtask='Implement a FastAPI REST API with JWT authentication "
    "middleware' — not the raw query.\n"
    "You may also specify 'access_list' per step to control which prior step "
    "results are visible. Use ['all'] for full context, specific node_ids for "
    "selective injection, or leave empty for no prior context.\n\n"
    "### WIDE-SEARCH ORCHESTRATION (CONCEPT:ORCH-1.1)\n"
    "If the query requests extracting a large table of data across many entities "
    "(e.g., 'Web2WideSearch' or 'Wide-Search'), you MUST decompose the extraction "
    "into discrete batches. Emit multiple parallel ExecutionSteps assigned to an "
    "extraction specialist (e.g., 'researcher' or 'web_researcher'), each targeting "
    "a specific partition of the data in its 'refined_subtask' (e.g., 'Extract rows "
    "for entities A-D'). Set 'parallel=True' and configure 'access_list' to "
    "share the shared workboard context if necessary.\n\n"
)


def subtask_and_widesearch_instructions() -> str:
    """R9: the HTN subtask-spec + wide-search orchestration prompt block."""
    return SUBTASK_AND_WIDESEARCH_INSTRUCTIONS


COMPLEXITY_KEYWORDS = ("complex", "architect")


def is_complex_query(query: str, num_specialists: int = 0) -> bool:
    """R11 (detection): keyword/size heuristic that a query warrants escalation.

    The router escalates to a stronger reasoning model when the query reads as an
    architectural/complex task or when many specialists are in play. The
    graph-topology and quantitative-reasoning escalation paths remain in the
    router (they depend on live KG state); this is the extracted text heuristic.
    """
    q = (query or "").lower()
    return any(kw in q for kw in COMPLEXITY_KEYWORDS) or num_specialists > 3


def rlm_plan_instruction(query: str) -> str:
    """R10: the RLM (Recursive Language Model) planning instruction."""
    return (
        f"Create a high-level execution plan for the query: {query}\n\n"
        "Use the REPL to analyze available specialists and the project context. "
        "Decompose the goal into steps that can be handled by the specialists. "
        "You MUST output a valid JSON representation of a GraphPlan. "
        "The GraphPlan should have 'steps' (list of {node_id, input_data}) and "
        "'metadata' ({reasoning}). Use FINAL_VAR('plan', <json_string>)."
    )


def parse_rlm_plan(rlm_result: Any, graph_plan_cls: Any) -> Any | None:
    """R10: parse RLM output into a GraphPlan, or ``None`` to trigger the fallback parser.

    Returning ``None`` (rather than raising) lets the router fall through to its
    LLM re-parse agent without duplicating the try/except.
    """
    import json

    try:
        return graph_plan_cls.model_validate(json.loads(rlm_result))
    except Exception:
        return None
