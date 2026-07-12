"""graph-os MCP analyze SUITE (register_analyze_suite_tools).

``graph_analyze`` had grown to ~30 actions spanning six unrelated domains in one wall of
description — hard for an agent to select against. This splits that surface into a few
COHESIVE, intent-scoped tools, each with a short scannable description:

* ``graph_code``     — code intelligence (call graph, similar code, routes, impact, …)
* ``graph_research`` — assimilation / research pipeline
* ``graph_evaluate`` — eval, harness gates, world-model, forecasting
* ``graph_explain``  — the universal context plane (one cited answer per question)
* ``graph_observe``  — KG-native observability/eval analytics (CONCEPT:AU-KG.ingest.observability-queries-opik-cannot)

The code/research/evaluate/explain tools are THIN facades: they carry only a focused
description and delegate to the SAME proven action core via ``_execute_tool`` (no logic
duplication, FieldInfo defaults resolved there). ``graph_observe`` routes to the trace
analytics. ``graph_analyze`` remains as the residual ops/structural surface + catch-all.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server


def register_analyze_suite_tools(mcp: Any) -> None:
    """Register the focused analyze-suite tools on the FastMCP server."""

    async def _delegate(
        action: str, query: str, top_k: int, node_id: str, depth: int, target: str
    ) -> Any:
        # One core: every facade routes the SAME action set through _execute_tool, which
        # resolves FieldInfo defaults exactly like an MCP invocation. No second handler.
        return await kg_server._execute_tool(
            "graph_analyze",
            action=action,
            query=query,
            top_k=top_k,
            node_id=node_id,
            depth=depth,
            target=target,
        )

    @mcp.tool(
        name="graph_code",
        description=(
            "Understand a CODEBASE via the ingested code graph — query this before grep. "
            "Actions: 'code_context' (cited how/usage/impact answer for a symbol/area; "
            "target=how|usage|impact, node_id=optional :Code anchor — KG-2.134), "
            "'cross_repo_usages' (every usage of a symbol across the fleet — KG-2.135), "
            "'call_graph' (callers/callees/inherits for node_id; target=callees|callers|"
            "inherits — KG-2.100), 'similar_code' (near-clones of node_id — KG-2.101), "
            "'routes' (HTTP route→handler→service graph — KG-2.102), 'change_coupling' "
            "(files that change together; target=repo path — KG-2.104), 'code_evolution' "
            "(query the ingested commit-history graph; target=file|owners|hotspots|coupled, "
            "query=file/subsystem path — file timelines, ownership, churn hotspots, co-change "
            "— KG-2.283), 'blast_radius' "
            "(transitive impact of node_id; depth — what breaks if I change this), "
            "'code_metrics' (god nodes / communities / bridges — KG-2.210), 'arch_report' "
            "(regenerable architecture report — KG-2.213), 'adr' (architecture decision records)."
        ),
        tags=["graph-os", "code"],
    )
    async def graph_code(
        action: str = Field(
            default="code_context",
            description="code_context | cross_repo_usages | call_graph | similar_code | routes | change_coupling | code_evolution | blast_radius | code_metrics | arch_report | adr",
        ),
        query: str = Field(
            default="", description="Question / area / symbol name / repo path."
        ),
        top_k: int = Field(default=10, description="Result count."),
        node_id: str = Field(
            default="",
            description="Symbol/:Code node id (call_graph, similar_code, blast_radius).",
        ),
        depth: int = Field(default=2, description="Traversal depth (blast_radius)."),
        target: str = Field(
            default="",
            description="how|usage|impact (code_context) or callees|callers|inherits (call_graph).",
        ),
    ) -> str:
        """Code intelligence over the ingested, resolved code graph."""
        return await _delegate(action, query, top_k, node_id, depth, target)

    @mcp.tool(
        name="graph_research",
        description=(
            "Run the research/assimilation pipeline. Actions: 'synthesize' (synthesize "
            "knowledge from a source), 'deep_extract' (deep entity/relation extraction), "
            "'background_research' (spawn background research), 'relevance_sweep' (sweep "
            "ingested content for relevance), 'research_ingest' (ingest a research artifact), "
            "'evolve_variants' (evolve solution variants), 'track_citations' (citation graph), "
            "'spawn_background' (spawn a background analysis job). query=source/topic; jobs "
            "return a job_id to poll with graph_ingest(action='status')."
        ),
        tags=["graph-os", "research"],
    )
    async def graph_research(
        action: str = Field(
            default="synthesize",
            description="synthesize | deep_extract | background_research | relevance_sweep | research_ingest | evolve_variants | track_citations | spawn_background",
        ),
        query: str = Field(default="", description="Source / topic / artifact."),
        top_k: int = Field(default=10, description="Complexity budget / result count."),
        node_id: str = Field(default="", description="Optional node id."),
        depth: int = Field(default=2, description="Optional depth."),
        target: str = Field(default="", description="Optional target."),
    ) -> str:
        """Research assimilation + knowledge-synthesis pipeline."""
        return await _delegate(action, query, top_k, node_id, depth, target)

    @mcp.tool(
        name="graph_evaluate",
        description=(
            "Evaluate agents/harnesses and reason over learned world models. Actions: "
            "'evaluate' / 'evaluate_alpha' (score outputs), 'evaluate_harness' (run the "
            "harness eval), 'guard_corpus' (reliability/eval corpus gate), 'harness_gate' "
            "(formal concentration/no-regression SHACL gate — AHE-3.53), 'check_constraints', "
            "'specialize' (one SAI specialization cycle + superhuman cert — AHE-3.29), "
            "'world_model_rollout' (forward-simulate the world model — KG-2.73b), "
            "'latent_efficiency_benchmark' (AHE-3.48), 'evolve_model', 'forecast', 'causal', "
            "'invariant'."
        ),
        tags=["graph-os", "evaluate"],
    )
    async def graph_evaluate(
        action: str = Field(
            default="evaluate",
            description="evaluate | evaluate_alpha | evaluate_harness | guard_corpus | harness_gate | check_constraints | specialize | world_model_rollout | latent_efficiency_benchmark | evolve_model | forecast | causal | invariant",
        ),
        query: str = Field(
            default="",
            description="Subject of the evaluation (JSON / id / start state).",
        ),
        top_k: int = Field(default=10, description="Steps / result count."),
        node_id: str = Field(default="", description="Optional node id."),
        depth: int = Field(default=2, description="Optional depth."),
        target: str = Field(default="", description="Optional target."),
    ) -> str:
        """Evaluation, harness gates, world-model rollouts, forecasting/causal analysis."""
        return await _delegate(action, query, top_k, node_id, depth, target)

    @mcp.tool(
        name="graph_explain",
        description=(
            "The UNIVERSAL context plane (CONCEPT:AU-KG.retrieval.route-question-its-domain): route a question to its domain "
            "provider and return ONE grounded, cited answer. action='explain' with "
            "target='domain:intent' (e.g. 'ops:why', 'code:usage', 'deploy:status', "
            "'entity:health') — or a bare intent with the domain inferred, or target='domains' "
            "to list providers. action='context' returns a synthesized context bundle. "
            "Domains: code, ops (live task-queue), deploy (is my change live — KG-2.138), "
            "entity/tickets/deploys/process (KG-2.139), capability (Capability Power "
            "Descriptor for a graph-os tool by id, e.g. 'capability:graph_query', or "
            "'capability:list' for the browsable index — Seam 8 Phase 1, "
            "CONCEPT:AU-KG.retrieval.capability-power-descriptor)."
        ),
        tags=["graph-os", "explain"],
    )
    async def graph_explain(
        action: str = Field(default="explain", description="explain | context"),
        query: str = Field(default="", description="The question."),
        top_k: int = Field(default=10, description="Result count."),
        node_id: str = Field(default="", description="Optional anchor node id."),
        depth: int = Field(default=2, description="Optional depth."),
        target: str = Field(
            default="", description="'domain:intent' | bare intent | 'domains'."
        ),
    ) -> str:
        """One grounded, cited answer per question, routed to the right domain provider."""
        return await _delegate(action, query, top_k, node_id, depth, target)

    @mcp.tool(
        name="graph_observe",
        description=(
            "Reason over the KG-native observability subgraph — traces, online-scores, "
            "assertion verdicts, generations, prompt versions — queries an opaque trace "
            "store can't do (CONCEPT:AU-KG.ingest.observability-queries-opik-cannot). Actions: 'trace_rootcause' (FAILED "
            "assertions + low scores joined to their trace's agent, grouped; query=agent/"
            "capability filter), 'prompt_regression' (mean score per prompt version — which "
            "regressed), 'failure_cluster' (failing traces clustered by the failed assertion "
            "— systemic breaks across agents)."
        ),
        tags=["graph-os", "observe", "eval"],
    )
    async def graph_observe(
        action: str = Field(
            default="trace_rootcause",
            description="trace_rootcause | prompt_regression | failure_cluster",
        ),
        query: str = Field(
            default="",
            description="Optional agent/capability filter (trace_rootcause).",
        ),
        top_k: int = Field(default=20, description="Max rows/clusters."),
    ) -> str:
        """Observability + eval analytics over the trace/score subgraph (KG-2.257)."""
        engine = kg_server._get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        from agent_utilities.harness import trace_analytics as ta

        backend = getattr(engine, "backend", None)
        try:
            if action == "trace_rootcause":
                out = ta.trace_rootcause(backend, capability=query, top_k=top_k)
            elif action == "prompt_regression":
                out = ta.prompt_regression(backend, top_k=top_k)
            elif action == "failure_cluster":
                out = ta.failure_cluster(backend, top_k=top_k)
            else:
                return f"Error: Unknown observe action '{action}'"
            return json.dumps(out, indent=2, default=str)
        except Exception as e:
            return f"Observe error: {e}"

    for _name, _fn in [
        ("graph_code", graph_code),
        ("graph_research", graph_research),
        ("graph_evaluate", graph_evaluate),
        ("graph_explain", graph_explain),
        ("graph_observe", graph_observe),
    ]:
        kg_server.REGISTERED_TOOLS[_name] = _fn
