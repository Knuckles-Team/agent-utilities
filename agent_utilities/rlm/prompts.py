"""Model-family-aware RLM REPL system prompt (CONCEPT:ORCH-1.54).

Zhang et al. (2025) report that a single fixed RLM system prompt fails to transfer across model
families (a documented failure mode). This module keeps the shared REPL contract in one place and
appends a small family-specific addendum so the same RLM runtime works on OpenAI, Anthropic, and
Qwen-class models without per-call rewrites.
"""

from __future__ import annotations

from typing import Literal

Family = Literal["openai", "anthropic", "qwen"]

# The shared helper contract (identical across families) — what the REPL exposes.
_BASE = (
    "You are a Recursive Language Model (RLM).\n"
    "You have access to a persistent Python REPL.\n"
    "Your objective is to write python code to analyze the `context` variable, "
    "which contains massive amounts of data.\n\n"
    "AVAILABLE HELPERS:\n"
    "- `await rlm_query(prompt, context, schema=None)`: Spawn a full recursive RLM at the "
    "next depth. Pass `schema=` (a type like `bool`/`int`, a Pydantic model, or a raw "
    "JSON Schema dict e.g. `{'type': 'boolean'}`) to FORCE the sub-agent to return a "
    "validated, typed value instead of free-form prose — route on that value directly.\n"
    "- `await run_parallel_sub_calls(calls)`: Run multiple sub-calls in parallel. "
    "`calls` is a list of `{'prompt': '...', 'context': ..., 'schema': <optional>}`. "
    "Prefer a structured `schema` (e.g. a boolean relevance flag per chunk) so you can "
    "filter on typed results rather than re-reading many text answers.\n"
    "- `await magma_view(query, views=None)`: Retrieve MAGMA orthogonal context "
    "(semantic, temporal, causal, entity).\n"
    "- `await graph_query(cypher, params=None)`: Run a Cypher query against the knowledge graph.\n"
    "- `await ephemeral_graph_query(cypher, namespace, params=None)`: Run a Cypher query against a specific ephemeral memory namespace.\n"
    "- `await owl_query(sparql)`: Run a SPARQL query against the OWL reasoner "
    "for transitive reasoning (wasDerivedFrom chains, SKOS hierarchies, escalation paths).\n"
    "- `await kg_bulk_export(node_type, limit=500)`: Export KG nodes as JSON for bulk analysis.\n"
    "- `await sub_agent_call(prompt, agent_id, input_data)`: Dispatch a task to another specialist.\n"
    "- `FINAL_VAR('result_name', value)`: Explicitly output the final result.\n\n"
    "IMPORTANT: You do NOT have the full context in your window. "
    "Access it programmatically via the `context` variable. "
    "Write Python code inside ```python blocks."
)

# Per-family addenda targeting each family's characteristic RLM failure mode.
_ADDENDA: dict[str, str] = {
    "openai": "",
    "anthropic": (
        "\n\nSTYLE: Do not narrate your plan in prose. Respond with a ```python block first; "
        "keep any explanation to a single short line. Always finish by calling `FINAL_VAR`."
    ),
    "qwen": (
        "\n\nSTYLE: Be terse to conserve output tokens — emit a ```python block immediately, "
        "no long reasoning. Use exactly one code block per turn. You MUST end by calling "
        "`FINAL_VAR('result', value)`; a turn without it is wasted."
    ),
}


def infer_family(model_id: str) -> Family:
    """Infer the prompt family from a (possibly ``provider:``-prefixed) model id."""
    mid = (model_id or "").lower()
    if "claude" in mid or "anthropic" in mid:
        return "anthropic"
    if "qwen" in mid:
        return "qwen"
    # OpenAI, Google/Gemini, and unknowns share the neutral default prompt.
    return "openai"


def build_system_prompt(prompt_family: str, model_id: str) -> str:
    """Build the RLM REPL system prompt for ``model_id`` under the configured ``prompt_family``.

    ``prompt_family='auto'`` infers the family from ``model_id``; any other value pins it.
    """
    family: str = infer_family(model_id) if prompt_family == "auto" else prompt_family
    return _BASE + _ADDENDA.get(family, "")
