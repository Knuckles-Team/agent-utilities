export const meta = {
  name: 'surface-gap-handlers',
  description: 'Design MCP/REST action handlers for the 26 unexposed capability modules (parallel research+codegen; main loop integrates)',
  phases: [{ title: 'Design handlers' }],
}

// Each entry: the unexposed module + the host action-routed tool it should be
// wired into. quant_mcp_tools is handled structurally by the main loop (it
// registers a whole tool), so it is excluded here.
const MODULES = [
  // harness self-evolution → graph_analyze
  ['agent_utilities/harness/agentic_evolution_engine.py', 'graph_analyze', 'evolve_variants'],
  ['agent_utilities/harness/background_spawner.py', 'graph_analyze', 'spawn_background'],
  ['agent_utilities/harness/citation_tracker.py', 'graph_analyze', 'track_citations'],
  ['agent_utilities/harness/constraint_engine.py', 'graph_analyze', 'check_constraints'],
  ['agent_utilities/harness/corpus_collapse_guard.py', 'graph_analyze', 'guard_corpus'],
  ['agent_utilities/harness/evaluation_engine.py', 'graph_analyze', 'evaluate_harness'],
  ['agent_utilities/harness/evolve_agent.py', 'graph_analyze', 'evolve_agent'],
  ['agent_utilities/harness/recursive_distill.py', 'graph_analyze', 'recursive_distill'],
  ['agent_utilities/harness/search_distillation.py', 'graph_analyze', 'distill_search'],
  // orchestration mixins → graph_orchestrate
  ['agent_utilities/knowledge_graph/orchestration/engine_enterprise.py', 'graph_orchestrate', 'enterprise_op'],
  ['agent_utilities/knowledge_graph/orchestration/engine_finance.py', 'graph_orchestrate', 'finance_op'],
  ['agent_utilities/knowledge_graph/orchestration/engine_ml_rlm.py', 'graph_orchestrate', 'ml_rlm_op'],
  // kb / maintenance / workflows → graph_analyze
  ['agent_utilities/knowledge_graph/kb/entity_claim_extractor.py', 'graph_analyze', 'extract_claims'],
  ['agent_utilities/knowledge_graph/kb/link_inference.py', 'graph_analyze', 'infer_links'],
  ['agent_utilities/knowledge_graph/kb/x_workflows.py', 'graph_analyze', 'x_workflow'],
  ['agent_utilities/knowledge_graph/maintenance/document_cleanup.py', 'graph_analyze', 'cleanup_documents'],
  ['agent_utilities/workflows/epistemic_sync.py', 'graph_analyze', 'epistemic_sync'],
  ['agent_utilities/workflows/skill_picker.py', 'graph_analyze', 'pick_skill'],
  // finance engines (regime_detector is already imported by the quant tool) → graph_analyze
  ['agent_utilities/domains/finance/banking.py', 'graph_analyze', 'quant_banking'],
  ['agent_utilities/domains/finance/cross_market_arb.py', 'graph_analyze', 'quant_arb'],
  ['agent_utilities/domains/finance/crypto_connector.py', 'graph_analyze', 'quant_crypto'],
  ['agent_utilities/domains/finance/exchange_bridge.py', 'graph_analyze', 'quant_exchange'],
  ['agent_utilities/domains/finance/microstructure.py', 'graph_analyze', 'quant_microstructure'],
  ['agent_utilities/domains/finance/strategy_engine.py', 'graph_analyze', 'quant_strategy'],
  ['agent_utilities/domains/finance/regime_detector.py', 'graph_analyze', 'quant_regime'],
  ['agent_utilities/domains/hr/workforce_manager.py', 'graph_analyze', 'workforce_plan'],
]

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['module', 'action_name', 'host_tool', 'snippet', 'entry', 'usable'],
  properties: {
    module: { type: 'string' },
    action_name: { type: 'string' },
    host_tool: { type: 'string', enum: ['graph_analyze', 'graph_orchestrate'] },
    snippet: {
      type: 'string',
      description: 'Complete `elif action == "<action_name>":` block ready to paste into the tool dispatch chain.',
    },
    entry: { type: 'string', description: 'The ClassName.method or function the snippet invokes.' },
    usable: { type: 'boolean', description: 'true if the snippet performs a real, correct invocation (not a stub).' },
    notes: { type: 'string' },
  },
}

const CONVENTIONS = `
You are wiring ONE capability module into the graph-os MCP server
(agent_utilities/mcp/kg_server.py) so it becomes reachable on BOTH the MCP and
REST surfaces. You will NOT edit any file — you return a structured spec and the
main loop integrates it.

The host tool is an async function with an \`action\` param and an \`if action ==\`
dispatch chain. Adding a new \`elif\` branch exposes a new action on both surfaces
automatically (the REST route passes \`action\` through). Your job: read the module,
pick its single most useful public entry point, and write the elif branch.

In scope inside the dispatch you may use:
- graph_analyze (async): \`engine\` (already resolved, may be None), \`query: str\`
  (use as the primary text/path/id/ticker input), \`top_k: int\`, \`node_id: str\`,
  \`target: str\`, \`depth: int\`. \`json\` is imported in the function.
- graph_orchestrate (async): call \`engine = _get_engine()\` at the top of your body;
  \`task: str\` (primary input), \`agent_name: str\`, \`max_steps: int\`.

Rules:
1. Lazy-import the module INSIDE the branch (never top-level).
2. Instantiate the entry class (pass \`engine\` if its constructor takes one) and
   call its most representative public method, deriving args from the available
   params (\`query\`/\`task\` is the main input). For a module-level function, call it.
3. Return a \`str\`. For dict/list results: \`return json.dumps(result, default=str)\`.
   (graph_orchestrate: \`import json as _json\` then \`_json.dumps\`.)
4. If \`engine\` is required and None, return a short "Error: ..." string.
5. If the entry genuinely needs inputs you cannot supply from the params, still
   make a best-effort correct call using \`query\`/\`task\` and sensible defaults, and
   set usable=false with a note — do NOT invent fake data or write a pure stub.
6. The \`elif\` line must be indented 12 spaces; the body 16 spaces (matching the
   existing chain). action_name must be exactly the one you were given.

Reference (an existing branch in graph_analyze):
            elif action == "research_ingest":
                from agent_utilities.knowledge_graph.research.research_intelligence_engine import (
                    ResearchIntelligenceEngine,
                )
                if not query:
                    return "Error: research_ingest needs a URL/paper id in \`query\`."
                rie = ResearchIntelligenceEngine(engine)
                return await rie.ingest_url(query)

Return ONLY the structured spec.`

phase('Design handlers')
const specs = await parallel(
  MODULES.map(([module, host, action]) => () =>
    agent(
      `${CONVENTIONS}\n\nMODULE TO WIRE: ${module}\nHOST TOOL: ${host}\nREQUIRED action_name: "${action}"\n\nRead the module at that path (it's under /home/apps/workspace/agent-packages/agent-utilities/). Identify the best public entry point and produce the elif branch.`,
      { label: `wire:${action}`, phase: 'Design handlers', schema: SCHEMA, agentType: 'Explore' },
    ),
  ),
)

return specs.filter(Boolean)
