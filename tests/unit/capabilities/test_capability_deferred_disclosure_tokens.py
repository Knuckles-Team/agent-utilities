"""Track 1 of the pydantic-ai native-adoption program: measure the real prompt-token
cost of expressing two EXISTING agent-utilities units — the ``graph-query-and-explanation``
skill and the ``graph-orchestration-and-automation`` skill (the fleet's own delegation
playbook, used here as the "agent" unit) — as native ``pydantic_ai.capabilities.Capability``
objects with ``defer_loading=True``, versus the same objects with ``defer_loading=False``.

CONCEPT:AU-ORCH — see ``reports/program/pydantic-ai-native-adoption.md`` Track 1.

Both SKILL.md bodies are read verbatim from disk (100% real, already-shipped instructions
text). The per-tool descriptions attached as function tools are copied verbatim from each
skill's own "Action reference" table — real text, real tool names — but the tool FUNCTIONS
themselves are thin measurement stand-ins (they raise if actually called) since invoking the
real graph-os intent-surface tools requires a live engine; this test only measures the
request-shaping cost (instructions + tool-schema bytes), never executes a tool.

Token counts are produced with ``tiktoken`` (``o200k_base``) as a PROXY — neither Anthropic
nor every provider publishes an offline tokenizer, so absolute counts are not billing-exact
for any one provider. The relative reduction (deferred vs. not) is what this test asserts.

IMPORTANT — what this measures, and what it does NOT: pydantic-ai's ``ToolSearch``/deferred
``Capability`` machinery has TWO distinct code paths (verified by reading
``pydantic_ai.toolsets._tool_search.ToolSearchToolset.get_tools`` and
``pydantic_ai.models.Model._resolve_native_tool_swap``):

1. **Native mode** (Anthropic / OpenAI Responses): the FULL tool schema is still sent on
   every request, with a wire-level ``defer_loading: true`` marker the provider itself
   honors server-side. The local JSON payload is the SAME size deferred or not — the
   savings (if any) happen on the provider's infrastructure and can only be measured via
   that provider's own usage/billing response, not locally.
2. **Local-fallback mode** (any provider profile that does not declare native tool-search
   support — the shape of our own default local-LLM path, which speaks the OpenAI
   chat-completions API, not the OpenAI Responses native-tool-search surface): deferred
   tools are dropped from ``ModelRequestParameters.function_tools`` ENTIRELY until
   discovered (``Model._resolve_native_tool_swap`` rule 3). This IS a real, local,
   measurable reduction in what actually reaches the wire.

This test forces path 2 with an explicit ``ModelProfile(supported_native_tools=set())`` so
the measured numbers are real and reproducible without a live Anthropic/OpenAI account.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("tiktoken")
pydantic_ai_capabilities = pytest.importorskip("pydantic_ai.capabilities")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.capabilities import Capability  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402
from pydantic_ai.profiles import ModelProfile  # noqa: E402

#: No native tool-search support declared — forces pydantic-ai's LOCAL discovery
#: fallback (``Model._resolve_native_tool_swap`` rule 3: an undiscovered deferred
#: tool paired with an unsupported native tool is dropped from the wire entirely).
#: This is the shape of our own default local-LLM path (OpenAI chat-completions
#: API, not OpenAI's Responses native-tool-search surface) — see
#: ``reports/program/pydantic-ai-native-adoption.md`` Track 1.
_NON_NATIVE_PROFILE = ModelProfile(supported_native_tools=set())

_SKILLS_ROOT = Path(__file__).resolve().parents[3] / "agent_utilities" / "skills"


def _read_skill_body(skill_id: str) -> str:
    path = _SKILLS_ROOT / skill_id / "SKILL.md"
    assert path.exists(), f"expected real skill file at {path}"
    return path.read_text(encoding="utf-8")


# Real tool names + descriptions copied verbatim from
# ``agent_utilities/skills/graph-query-and-explanation/SKILL.md``'s Action reference table.
_GRAPH_QUERY_TOOLS: list[tuple[str, str]] = [
    (
        "graph_ask",
        "NL question -> generated query (dialect=auto|cypher|sql|sparql) + rows. "
        "execute=false previews the generated query without running it.",
    ),
    (
        "graph_code",
        "code_context (target=how|usage|impact), cross_repo_usages, call_graph, "
        "similar_code, routes, change_coupling, code_evolution, blast_radius, "
        "code_metrics, arch_report, adr. Query the KG before grepping.",
    ),
    (
        "graph_search",
        "mode: hybrid (default), hyde, deep, concept, analogy, memory, discover, "
        "latent, rerank, adore, hard_negatives, chrono_ids, compiled; top_k, "
        "self_correct, as_of, target (named/all connections).",
    ),
    (
        "graph_query",
        "Explicit read-only Cypher, SQL, SPARQL, or federated query over the "
        "knowledge graph, scoped by the `scope` argument.",
    ),
    (
        "graph_epistemic",
        "why (=explain_belief), status (=epistemic_status), why_not, "
        "what_would_invalidate, what_changed, resolve_conflict.",
    ),
]

# Real tool names + descriptions copied verbatim from
# ``agent_utilities/skills/graph-orchestration-and-automation/SKILL.md``'s Action reference table.
_GRAPH_ORCHESTRATION_TOOLS: list[tuple[str, str]] = [
    (
        "graph_orchestrate",
        "dispatch, swarm (goal -> decompose -> parallel waves -> verify -> "
        "synthesize), execute_agent, execute_workflow/compile_workflow, status, "
        "request_approval/grant_approval, consensus/start_debate, computer_use, "
        "optimize_component, distill_skills, loop_cycle, publish_proposal, "
        "failure_ingest. The fleet execution entrypoint.",
    ),
    (
        "graph_workflows",
        "compile, compile_process, list, execute, execute_dynamic, dispatch, "
        "status, export.",
    ),
    (
        "graph_loops",
        "submit (objective + kind=research|develop|skill), list, run, drive, "
        "cancel, prioritize, state, specs, review, placement_control, gaps, "
        "submit_gap, gap.",
    ),
    (
        "graph_sandbox",
        "status (per-rung availability + pooled warm-parent count + per-rung "
        "reward EMA), reap, warm.",
    ),
    (
        "graph_bus",
        "register/heartbeat/leave/status, roster, send, receive, "
        "subscribe/unsubscribe, ack, dispatch; mesh/federation: "
        "register_hub/list_hubs/federate/federate_in.",
    ),
]


def _measurement_tool(name: str, description: str) -> Any:
    """Build a thin, never-called stand-in function tool for one real fleet tool.

    Only the wire-visible shape (name, docstring/description, `data` parameter) is
    exercised by this test — the function body is unreachable in every scenario
    below, since none of them let the model choose to call a tool.
    """

    def _fn(data: str = "") -> str:
        raise AssertionError(f"measurement stand-in for {name!r} must never be called")

    _fn.__name__ = name
    _fn.__doc__ = description
    return _fn


def _build_capability(
    *, skill_id: str, tool_specs: list[tuple[str, str]], defer_loading: bool
) -> Capability:
    instructions = _read_skill_body(skill_id)
    tools = [_measurement_tool(name, desc) for name, desc in tool_specs]
    return Capability(
        id=skill_id,
        description=f"Real agent-utilities skill: {skill_id}",
        instructions=instructions,
        tools=tools,
        defer_loading=defer_loading,
    )


def _capture_agent_info(capability: Capability) -> AgentInfo:
    captured: dict[str, AgentInfo] = {}

    def _capture(_messages: Any, info: AgentInfo) -> ModelResponse:
        captured["info"] = info
        return ModelResponse(parts=[TextPart(content="ok")])

    agent: Agent[None, str] = Agent(
        FunctionModel(_capture, profile=_NON_NATIVE_PROFILE), capabilities=[capability]
    )
    agent.run_sync("hello")
    return captured["info"]


def _serialized_prompt_bytes(info: AgentInfo) -> str:
    """Everything the FIRST model request would carry for this capability:
    resolved instructions + every function tool's name/description/schema.
    """
    payload = {
        "instructions": info.instructions or "",
        "tools": [
            {
                "name": td.name,
                "description": td.description,
                "parameters_json_schema": td.parameters_json_schema,
            }
            for td in info.function_tools
        ],
    }
    return json.dumps(payload, sort_keys=True)


def _token_count(text: str) -> int:
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


@pytest.mark.parametrize(
    ("skill_id", "tool_specs"),
    [
        ("graph-query-and-explanation", _GRAPH_QUERY_TOOLS),
        ("graph-orchestration-and-automation", _GRAPH_ORCHESTRATION_TOOLS),
    ],
)
def test_defer_loading_shrinks_first_turn_prompt(
    skill_id: str, tool_specs: list[tuple[str, str]]
) -> None:
    baseline_capability = _build_capability(
        skill_id=skill_id, tool_specs=tool_specs, defer_loading=False
    )
    deferred_capability = _build_capability(
        skill_id=skill_id, tool_specs=tool_specs, defer_loading=True
    )

    baseline_info = _capture_agent_info(baseline_capability)
    deferred_info = _capture_agent_info(deferred_capability)

    baseline_text = _serialized_prompt_bytes(baseline_info)
    deferred_text = _serialized_prompt_bytes(deferred_info)

    baseline_tokens = _token_count(baseline_text)
    deferred_tokens = _token_count(deferred_text)

    # Real assertion: deferring must not be neutral or worse — the whole point of
    # `defer_loading=True` is a materially smaller first-turn prompt.
    assert deferred_tokens < baseline_tokens
    # Sanity: the deferred capability contributes NO function tools of its own on
    # turn 1 (only the internal loader's `load_capability` tool exists, which this
    # capability-only Agent does carry — but none of THIS capability's tools).
    assert not any(td.name in {n for n, _ in tool_specs} for td in deferred_info.function_tools)

    reduction_pct = round(100 * (1 - deferred_tokens / baseline_tokens), 1)
    print(
        f"\n[{skill_id}] baseline={baseline_tokens} tokens, "
        f"deferred={deferred_tokens} tokens, reduction={reduction_pct}%"
    )
