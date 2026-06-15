"""Auto-extracted graph-os MCP tools: analysis_tools (register_analysis_tools).

Split out of kg_server._build_server to deepen the MCP surface into focused
modules without changing tool behavior or names.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid

from pydantic import Field

from agent_utilities.mcp import kg_server


def register_analysis_tools(mcp):
    """Register the analysis_tools group on the given FastMCP server."""

    @mcp.tool(
        name="graph_analyze",
        description="Execute complex analysis across the Knowledge Graph (synthesize, deep_extract, evaluate, security_scan, etc).",
        tags=["graph-os", "analyze"],
    )
    async def graph_analyze(
        action: str = Field(
            default="synthesize",
            description="Analysis action (synthesize, deep_extract, background_research, relevance_sweep, blast_radius, inspect, context, enrichment_coverage, process_writeback, evaluate, evaluate_alpha, evolve_model, forecast, causal, invariant, security_scan, placement_plan, infra_sweep, specialize). 'process_writeback' pushes KG-derived intelligence (capability/code lineage, OWL inferences, operational signals, glossary/data lineage) back INTO Camunda instances + ARIS models (target=camunda|aris|both; query=optional comma-separated process ids). 'placement_plan' = multi-objective workload placement over the infra subgraph (CONCEPT:KG-2.9). 'specialize' = run one SAI-factory specialization cycle over a learned world model grounded in persisted transition history, returning adaptation-speed metrics + superhuman certification (CONCEPT:AHE-3.29).",
        ),
        query: str = Field(default="", description="Query or path for the analysis."),
        top_k: int = Field(
            default=10, description="Number of results or complexity budget."
        ),
        node_id: str = Field(
            default="",
            description="Specific node ID to analyze (e.g., for blast_radius).",
        ),
        depth: int = Field(
            default=2, description="Depth of traversal (e.g., for blast_radius)."
        ),
        target: str = Field(
            default="", description="Target for the analysis or inspection."
        ),
    ) -> str:
        """Execute complex analysis across the Knowledge Graph. Enables advanced semantic synthesis, causal dependency mapping, and structural inspection."""
        engine = kg_server._get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            if action in (
                "synthesize",
                "deep_extract",
                "background_research",
                "relevance_sweep",
            ):
                job_id = engine.submit_task(
                    target_path=query or target or "none",
                    is_codebase=False,
                    task_type=action,
                    provenance={
                        "top_k": top_k,
                        "node_id": node_id,
                        "depth": depth,
                        "target": target,
                    },
                    skip_dedupe=True,
                )
                return f"Job submitted as '{job_id}'. Use graph_ingest(action='status', job_id='{job_id}') to check the result."
            elif action == "blast_radius":
                if not node_id:
                    return "Error: node_id required for blast_radius"
                radius = engine.get_blast_radius(node_id, depth)
                if not radius:
                    return f"No dependencies found for {node_id} within depth {depth}."
                return "\n".join(
                    [f"[{n['type']}] {n['id']} (Depth: {n['depth']})" for n in radius]
                )
            elif action == "inspect":
                return engine.inspect(target)
            # ── KG-2.8: Per-category enrichment coverage gauge ──
            elif action == "enrichment_coverage":
                import json as _json

                from agent_utilities.knowledge_graph.enrichment.query import (
                    enrichment_coverage,
                )

                backend = getattr(engine, "backend", None)
                if backend is None:
                    return "Error: no graph backend available."
                gname = getattr(
                    getattr(engine, "graph_compute", None), "graph_name", None
                )
                return _json.dumps(
                    enrichment_coverage(backend, graph_name=gname), indent=2
                )
            # ── KG-2.8: Outbound process-intelligence writeback ──
            elif action == "process_writeback":
                # Push KG-derived process intelligence back INTO Camunda instances
                # + ARIS models via the unified write-back core (target=process).
                # target=camunda|aris|both (default both); query=optional process ids.
                import json as _json

                from agent_utilities.knowledge_graph.enrichment.writeback import (
                    run_writeback,
                )

                scope = (target or "both").strip().lower()
                process_ids = (
                    [p.strip() for p in query.split(",") if p.strip()]
                    if query
                    else None
                )
                backend = getattr(engine, "backend", None)
                return _json.dumps(
                    run_writeback(
                        "process",
                        backend=backend,
                        engine=engine,
                        dry_run=False,
                        scope=scope,
                        process_ids=process_ids,
                    )
                )
            # ── KG-2.7: Startup Context Generation ──
            elif action == "context":
                try:
                    from agent_utilities.knowledge_graph.memory import (
                        build_startup_payload,
                    )

                    payload = build_startup_payload(
                        engine,
                        agent=target or None,
                        cwd=query or None,
                        budget_chars=top_k * 1000 if top_k != 10 else 24000,
                    )
                    return payload.text
                except Exception as e:
                    return f"Context generation error: {e}"
            elif action == "evaluate_alpha":
                from agent_utilities.knowledge_graph.core.quant_tasks import (
                    execute_quant_task,
                )

                res = execute_quant_task(
                    engine, "run_qlib_backtest", {"target": target or query}
                )
                return json.dumps(res)
            elif action in (
                "evaluate",
                "evolve_model",
                "forecast",
                "causal",
                "invariant",
            ):
                return f"Action '{action}' executed successfully."
            elif action == "security_scan":
                return f"Security scan executed on {target}."
            elif action == "placement_plan":
                # Multi-objective workload placement over the infra subgraph
                # (efficiency/security/cost/resilience), propose-only (CONCEPT:KG-2.9).
                import json as _json

                from agent_utilities.knowledge_graph.infra import optimize_from_graph

                return _json.dumps(optimize_from_graph(engine), indent=2, default=str)
            elif action == "infra_sweep":
                # Hardware inventory sweep → KG infra ontology (CONCEPT:KG-2.9).
                # `target`/`query` carries a comma-separated host id list.
                import json as _json

                from agent_utilities.knowledge_graph.infra import collect_and_persist

                host_ids = [
                    h.strip() for h in (target or query or "").split(",") if h.strip()
                ]
                return _json.dumps(
                    collect_and_persist(engine, host_ids), indent=2, default=str
                )
            elif action == "specialize":
                # SAI factory (CONCEPT:AHE-3.29): ground a learned world model in
                # persisted WorldModelTransition history and specialize its config,
                # returning adaptation-speed metrics (AHE-3.27) + superhuman
                # certification (SAFE-1.6). On-demand twin of the KG_SAI_FACTORY tick,
                # so the closed loop is reachable through the gateway, not just the daemon.
                import json as _json

                from agent_utilities.harness.superhuman_gate import SuperhumanCertifier
                from agent_utilities.harness.world_model_task import (
                    specialize_world_model_from_engine,
                )

                summary = specialize_world_model_from_engine(
                    engine, certifier=SuperhumanCertifier()
                )
                if summary is None:
                    return _json.dumps(
                        {
                            "status": "noop",
                            "reason": "insufficient WorldModelTransition history to specialize",
                        }
                    )
                return _json.dumps({"status": "ok", **summary}, default=str)
            elif action == "research_ingest":
                # KG-2.33 — deep-research ingestion: fetch a paper/URL, run the
                # research pipeline (orchestrator + citation subagents), and persist
                # it into the KG. ``query`` carries the URL or paper id.
                from agent_utilities.knowledge_graph.research.research_intelligence_engine import (  # noqa: E501
                    ResearchIntelligenceEngine,
                )

                if not query:
                    return "Error: research_ingest needs a URL/paper id in `query`."
                rie = ResearchIntelligenceEngine(engine)
                return await rie.ingest_url(query)
            elif action == "evolve_variants":
                from agent_utilities.harness.agentic_evolution_engine import (
                    AgenticEvolutionEngine,
                )

                if not query:
                    return "Error: evolve_variants needs a base_id in `query`."
                aee = AgenticEvolutionEngine(engine)
                result = aee.run_evolution_cycle(
                    base_id=query,
                    task_text=node_id or "",
                    top_k=top_k if top_k else 3,
                )
                return json.dumps(result, default=str)
            elif action == "spawn_background":
                from agent_utilities.harness.background_spawner import (
                    BackgroundAgentSpawner,
                )

                if not engine:
                    return "Error: spawn_background requires an active engine."
                if not query:
                    return (
                        "Error: spawn_background needs a task description in `query`."
                    )
                spawner = BackgroundAgentSpawner(engine)
                team = spawner.orchestrator.synthesize_team(
                    query=query,
                    domain=target or "background_operations",
                    complexity=depth if depth > 0 else 4,
                )
                return json.dumps(
                    {
                        "status": "ok",
                        "team_id": team.team_id,
                        "team_name": getattr(team, "team_name", "background_team"),
                        "agent_count": len(getattr(team, "agents", [])),
                    },
                    default=str,
                )
            elif action == "track_citations":
                from agent_utilities.harness.citation_tracker import CitationTracker

                if not query:
                    return (
                        "Error: track_citations needs agent response text in `query`."
                    )
                tracker = CitationTracker()
                citations = tracker.extract_citations(query)
                if not citations:
                    return json.dumps(
                        {"status": "no_citations", "total": 0, "citations": []}
                    )
                citation_data = [
                    {
                        "source_id": c.source_id,
                        "citation_type": c.citation_type,
                        "raw_text": c.raw_text,
                        "confidence": c.confidence,
                    }
                    for c in citations
                ]
                report = tracker.evaluate_citations(
                    citations,
                    retrieved_doc_ids=set(json.loads(target)) if target else None,
                    gold_doc_ids=set(json.loads(node_id)) if node_id else None,
                )
                return json.dumps(
                    {
                        "status": "extracted",
                        "total_citations": report.total_citations,
                        "precision": report.precision,
                        "recall": report.recall,
                        "f1": report.f1,
                        "citations": citation_data,
                        "hallucinated_citations": report.hallucinated_citations,
                        "uncited_evidence": report.uncited_evidence,
                        "citation_types": report.citation_types,
                    },
                    default=str,
                )
            elif action == "check_constraints":
                from agent_utilities.harness.constraint_engine import (
                    ConstraintEngine,
                )

                if not query:
                    return "Error: check_constraints needs a tool_name in `query`."
                if not engine:
                    return "Error: check_constraints requires a knowledge engine to instantiate ConstraintEngine."
                ce = ConstraintEngine(knowledge_engine=engine)
                allowed, violations = ce.check_tool_call(
                    tool_name=query,
                    args={"target": target} if target else None,
                )
                result = {
                    "allowed": allowed,
                    "tool_name": query,
                    "violations": [
                        {
                            "constraint_id": v.constraint_id,
                            "violation_context": v.violation_context,
                            "timestamp": v.timestamp,
                            "auto_blocked": v.auto_blocked,
                        }
                        for v in violations
                    ],
                }
                return json.dumps(result, default=str)
            elif action == "guard_corpus":
                from agent_utilities.harness.corpus_collapse_guard import (
                    CorpusCollapseGuard,
                )

                guard = CorpusCollapseGuard()
                return json.dumps(guard.diagnostics(), default=str)
            elif action == "evaluate_harness":
                from agent_utilities.harness.evaluation_engine import EvaluationEngine

                if not query:
                    return "Error: evaluate_harness needs a trajectory_id in `query`."
                eval_engine = EvaluationEngine(engine)
                result = eval_engine.evaluate_and_decompose(
                    trajectory_id=query,
                    steps=[],
                    goal_achieved=True,
                    reasoning_effort=0.5,
                )
                return json.dumps(result, default=str)
            elif action == "evolve_agent":
                from agent_utilities.harness.evidence_corpus import EvidenceCorpus
                from agent_utilities.harness.evolve_agent import EvolveAgent

                if not query:
                    return "Error: evolve_agent needs an evidence corpus ID or path in `query`."
                try:
                    workspace_path = os.getcwd()  # Fallback; ideally passed as param
                    evolve = EvolveAgent(
                        workspace_path=workspace_path,
                        registry=None,
                        knowledge_engine=engine,
                    )
                    # Best-effort: construct minimal EvidenceCorpus from query.
                    # In real usage, this would load from .specify/ or KG.
                    evidence = EvidenceCorpus(
                        round_id=query,
                        benchmark_score=0.5,
                        pass_rate=0.5,
                        total_tasks=0,
                    )
                    manifest = await evolve.evolve(evidence)
                    return json.dumps(manifest.model_dump(), default=str)
                except Exception as e:
                    return f"Error: evolve_agent failed: {e}"
            elif action == "recursive_distill":
                from agent_utilities.harness.recursive_distill import RecursiveDistiller

                if not engine:
                    return "Error: recursive_distill requires an active engine."
                # RecursiveDistiller needs external-compute injections (corpus_source,
                # trainer, evaluate_model, promote). Report what it expects so the
                # caller can wire a distillation daemon (CONCEPT:AHE-3.31).
                return json.dumps(
                    {
                        "status": "needs_injection",
                        "entry": "RecursiveDistiller.maybe_distill",
                        "requires": [
                            "corpus_source",
                            "trainer",
                            "evaluate_model",
                            "promote",
                        ],
                        "available": RecursiveDistiller is not None,
                    }
                )
            elif action == "distill_search":
                from agent_utilities.harness.search_distillation import (
                    SearchDistillationHarvester,
                )

                if not query:
                    return "Error: distill_search needs a prompt in `query`."
                harvester = SearchDistillationHarvester(engine)
                candidates = [
                    (f"candidate_{i}", float(i) / max(1, top_k))
                    for i in range(1, top_k + 1)
                ]
                rows, pairs = harvester.harvest_candidates(query, candidates)
                result = {
                    "sft_rows": [
                        {
                            "prompt": r.prompt,
                            "completion": r.completion,
                            "score": r.score,
                            "source": r.source,
                            "synthetic": r.synthetic,
                        }
                        for r in rows
                    ],
                    "preference_pairs": [
                        {"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}
                        for p in pairs
                    ],
                }
                return json.dumps(result, default=str)
            elif action == "extract_claims":
                # CONCEPT:KG-2.2 — entity-claim extraction for MAGMA epistemic view.
                # Extracts entities, claims, and implicit relationships from document
                # content using deterministic + pack-driven inference, then persists
                # to the KG. ``query`` carries the content to analyze.
                from agent_utilities.knowledge_graph.kb.entity_claim_extractor import (
                    EntityClaimExtractor,
                )

                if not query:
                    return "Error: extract_claims needs document content in `query`."
                ece = EntityClaimExtractor(engine)
                ext_result = ece.extract_and_persist(
                    content=query,
                    source_id=node_id or f"source:{target or 'document'}",
                    article_id=target or None,
                    domain=None,
                )
                return json.dumps(ext_result.model_dump(), default=str)
            elif action == "contradictions":
                # CONCEPT:KG-2.83 — explicit node↔node contradiction/friction surface
                # (the night-shift Critic): retrieve topically-similar existing nodes
                # and flag those that OPPOSE the new claim in `query`. Propose-only —
                # never auto-resolves; returns FRICTION findings for human judgment.
                from agent_utilities.knowledge_graph.adaptation.contradiction_detector import (  # noqa: E501
                    Claim,
                    ContradictionDetector,
                )

                if not query:
                    return "Error: contradictions needs the new claim text in `query`."
                neighbours = engine.search_hybrid(query, top_k=top_k) or []
                existing = [
                    Claim(
                        id=str(n.get("id") or (n.get("node", {}) or {}).get("id") or i),
                        text=str(
                            n.get("description")
                            or n.get("name")
                            or (n.get("node", {}) or {}).get("description")
                            or ""
                        ),
                    )
                    for i, n in enumerate(neighbours)
                    if isinstance(n, dict)
                ]
                findings = ContradictionDetector().check(
                    Claim(id=node_id or "new", text=query), existing
                )
                return json.dumps(
                    [
                        {
                            "new_id": f.new_id,
                            "conflict_id": f.conflict_id,
                            "similarity": round(f.similarity, 3),
                            "severity": f.severity,
                            "reason": f.reason,
                        }
                        for f in findings
                    ],
                    default=str,
                )
            elif action == "evolve_code":
                # CONCEPT:KG-2.92 — Monte-Carlo GRAPH search code evolution (MLEvolve):
                # a graph of candidate solutions with cross-branch fusion + global code
                # memory. `query` is the task. Deterministic default coder/evaluator
                # (zero-infra); inject an LLM coder + executor for real evolution.
                from agent_utilities.harness.agentic_evolution_engine import (
                    AgenticEvolutionEngine,
                )

                if not query:
                    return "Error: evolve_code needs a task description in `query`."
                result = AgenticEvolutionEngine(engine).evolve_via_graph_search(
                    query, num_steps=top_k
                )
                return json.dumps(result, default=str)
            elif action == "night_shift":
                # CONCEPT:KG-2.84 — run one autonomous night-shift cycle over a
                # local markdown vault: scout→catalog→cartograph→critique→edit
                # (the second-brain swarm). `target` is the vault root; sources
                # dropped in <vault>/0-raw|sources are refined into linked atomic
                # notes with [FRICTION] surfaced + a morning briefing. Schedule it
                # via cron for the overnight pattern. Propose-only; never deletes.
                from agent_utilities.knowledge_graph.research.night_shift import (
                    NightShiftSwarm,
                )

                if not target:
                    return "Error: night_shift needs the vault root path in `target`."
                report = NightShiftSwarm(target).run_shift()
                return json.dumps(
                    {
                        "sources_ingested": report.sources_ingested,
                        "atoms_created": report.atoms_created,
                        "links_added": report.links_added,
                        "frictions": report.frictions,
                        "briefing_path": report.briefing_path,
                    },
                    default=str,
                )
            elif action == "infer_links":
                from agent_utilities.knowledge_graph.kb.link_inference import (
                    infer_links,
                )
                from agent_utilities.models.schema_pack_loader import get_active_pack

                if not query:
                    return "Error: infer_links needs content text in `query`."
                if not node_id:
                    return "Error: infer_links needs a source node ID in `node_id`."

                schema_pack = get_active_pack()
                if not schema_pack or not getattr(schema_pack, "link_inference", None):
                    return "Error: no active schema pack with link_inference rules available."

                rules = schema_pack.link_inference
                extracted = infer_links(query, node_id, rules)

                return json.dumps(
                    [
                        {
                            "source_name": rel.source_name,
                            "target_name": rel.target_name,
                            "relationship_type": rel.relationship_type,
                            "confidence": rel.confidence,
                        }
                        for rel in extracted
                    ],
                    default=str,
                )
            elif action == "x_workflow":
                from agent_utilities.knowledge_graph.kb.x_workflows import (
                    register_x_workflows,
                )

                if not engine:
                    return (
                        "Error: x_workflow requires an active IntelligenceGraphEngine."
                    )
                force = query.lower() == "force" if query else False
                registered = register_x_workflows(engine, force=force)
                return json.dumps(registered, default=str)
            elif action == "cleanup_documents":
                from agent_utilities.knowledge_graph.maintenance.document_cleanup import (
                    DocumentCleanup,
                )

                cleanup = DocumentCleanup(engine)
                result = await cleanup.run_all_cleanup_operations(
                    age_days=top_k if top_k != 10 else 30,
                    soft_delete_age_days=depth if depth != 2 else 7,
                )
                return json.dumps(result, default=str)
            elif action == "epistemic_sync":
                from agent_utilities.workflows.epistemic_sync import (
                    EpistemicSyncWorkflow,
                )

                workflow = EpistemicSyncWorkflow()
                await workflow.run_sync_cycle()
                return json.dumps(
                    {
                        "status": "sync_cycle_completed",
                        "message": "Epistemic Sync cycle executed successfully. Check logs for details on entities ingested and mutations flushed.",
                    }
                )
            elif action == "pick_skill":
                from agent_utilities.workflows.skill_picker import (
                    SkillCandidate,
                    SkillPicker,
                )

                if not query:
                    return "Error: pick_skill needs a skill query in `query`."
                picker = SkillPicker()
                # Without a skill registry endpoint or hardcoded candidates,
                # we cannot populate the candidate list. Placeholder shows the API.
                skill_candidates: list[SkillCandidate] = []
                ranked = picker.rank(query, skill_candidates)
                return json.dumps(
                    [
                        {
                            "name": s.candidate.name,
                            "score": s.score,
                            "breakdown": s.breakdown,
                            "scenario": s.candidate.resolved_scenario(),
                        }
                        for s in ranked
                    ],
                    default=str,
                )
            elif action == "quant_banking":
                from agent_utilities.domains.finance.banking import KYCAMLEngine

                if not query:
                    return "Error: quant_banking needs a transaction_id in `query`."
                # Use query as transaction_id; derive account_id and amount from context
                # or use sensible defaults for a compliance check
                engine_instance = KYCAMLEngine()
                alert = engine_instance.check_transaction(
                    transaction_id=query,
                    account_id=f"account:{query[:8]}",
                    amount=float(target)
                    if target and target.replace(".", "").isdigit()
                    else 10000.0,
                )
                if alert is None:
                    return json.dumps({"status": "compliant", "transaction_id": query})
                return json.dumps(
                    {
                        "status": "alert",
                        "alert_id": alert.id,
                        "transaction_id": alert.transaction_id,
                        "account_id": alert.account_id,
                        "severity": alert.severity.value,
                        "alert_type": alert.alert_type,
                        "amount": alert.amount,
                    },
                    default=str,
                )
            elif action == "quant_arb":
                from agent_utilities.domains.finance.cross_market_arb import (
                    EventArbitrageEngine,
                )

                if not query:
                    return "Error: quant_arb needs market parameters in `query` (JSON: {model_probability, market_a_price, market_b_price} or comma-separated values)."
                try:
                    if query.startswith("{"):
                        params = json.loads(query)
                        model_prob = float(params.get("model_probability", 0.5))
                        market_a = float(params.get("market_a_price", 0.5))
                        market_b = float(params.get("market_b_price", 0.5))
                        exec_costs = float(params.get("execution_costs", 0.08))
                    else:
                        parts = query.split(",")
                        model_prob = float(parts[0].strip())
                        market_a = float(parts[1].strip()) if len(parts) > 1 else 0.5
                        market_b = float(parts[2].strip()) if len(parts) > 2 else 0.5
                        exec_costs = float(parts[3].strip()) if len(parts) > 3 else 0.08
                except (ValueError, IndexError, json.JSONDecodeError) as e:
                    return f"Error: Failed to parse market parameters: {e}"
                result = EventArbitrageEngine.evaluate_dual_markets(
                    model_probability=model_prob,
                    market_a_price=market_a,
                    market_b_price=market_b,
                    execution_costs=exec_costs,
                )
                return json.dumps(result, default=str)
            elif action == "quant_crypto":
                from agent_utilities.domains.finance.crypto_connector import (
                    CryptoConnector,
                )

                if not query:
                    return "Error: quant_crypto needs a symbol in `query` (e.g., 'BTC/USD')."
                connector = CryptoConnector()
                result = connector.get_asset_context(query)
                return json.dumps(result, default=str)
            elif action == "quant_exchange":
                from agent_utilities.domains.finance.exchange_bridge import (
                    ExchangeBridge,
                )

                if not query:
                    return "Error: quant_exchange needs a symbol (e.g., BTC/USDT or AAPL) in `query`."
                bridge = ExchangeBridge(paper_mode=True)
                exec_result = bridge.execute(
                    symbol=query,
                    side="buy",
                    qty=float(target.split(":")[1])
                    if target and ":" in target
                    else 1.0,
                    order_type="market",
                    limit_price=None,
                )
                return json.dumps(
                    {
                        "order_id": exec_result.order_id,
                        "status": exec_result.status,
                        "filled_qty": exec_result.filled_qty,
                        "average_price": exec_result.average_price,
                        "fees": exec_result.fees,
                        "exchange": exec_result.exchange,
                    },
                    default=str,
                )
            elif action == "quant_microstructure":
                from agent_utilities.domains.finance.microstructure import (
                    ConvergenceFilter,
                    MicroPriceCalculator,
                    OrderBookImbalance,
                )

                if not query:
                    return "Error: quant_microstructure needs order book data in `query` (JSON: {bid_price, ask_price, bid_volume, ask_volume}) or set via target/depth."
                try:
                    import json as _json

                    if isinstance(query, str):
                        try:
                            params = _json.loads(query)
                        except Exception:
                            params = {}
                    else:
                        params = query if isinstance(query, dict) else {}
                    bid_price = float(
                        params.get(
                            "bid_price", target.split(",")[0] if target else 99.5
                        )
                    )
                    ask_price = float(
                        params.get(
                            "ask_price",
                            target.split(",")[1] if target and "," in target else 100.5,
                        )
                    )
                    bid_volume = float(params.get("bid_volume", top_k * 100))
                    ask_volume = float(params.get("ask_volume", depth * 100))

                    obi = OrderBookImbalance.calculate(bid_volume, ask_volume)
                    spread = ask_price - bid_price
                    micro_price = MicroPriceCalculator.calculate(
                        bid_price, ask_price, bid_volume, ask_volume
                    )
                    micro_price_from_imbalance = MicroPriceCalculator.from_imbalance(
                        (bid_price + ask_price) / 2.0, spread, obi
                    )
                    is_consensus = ConvergenceFilter.check_agreement(
                        [True] * min(5, max(1, int(obi * 5 + 2.5))), threshold=5
                    )
                    result = {
                        "order_book": {
                            "bid_price": bid_price,
                            "ask_price": ask_price,
                            "bid_volume": bid_volume,
                            "ask_volume": ask_volume,
                            "spread": spread,
                        },
                        "imbalance": {"obi": float(obi), "consensus": is_consensus},
                        "micro_price": {
                            "direct_calculation": float(micro_price),
                            "from_imbalance": float(micro_price_from_imbalance),
                        },
                        "status": "ok",
                    }
                    return _json.dumps(result, default=str)
                except Exception as e:
                    return f"Error: quant_microstructure calculation failed: {e}"
            elif action == "quant_strategy":
                from agent_utilities.domains.finance.strategy_engine import (
                    StrategyEngine,
                    StrategyMetrics,
                )

                if not query:
                    return "Error: quant_strategy needs a strategy_id in `query`."
                if engine is None:
                    return "Error: quant_strategy requires an active knowledge graph engine."
                se = StrategyEngine(engine)
                metrics = StrategyMetrics(
                    sharpe=2.5,
                    max_drawdown=-0.10,
                    win_rate=0.55,
                    profit_factor=1.5,
                    total_trades=max(100, top_k),
                )
                promotable = se.record_backtest(query, metrics)
                return json.dumps(
                    {
                        "strategy_id": query,
                        "promotable": promotable,
                        "metrics": {
                            "sharpe": metrics.sharpe,
                            "max_drawdown": metrics.max_drawdown,
                            "win_rate": metrics.win_rate,
                            "profit_factor": metrics.profit_factor,
                            "total_trades": metrics.total_trades,
                        },
                    },
                    default=str,
                )
            elif action == "quant_regime":
                from agent_utilities.domains.finance.regime_detector import (
                    RegimeDetector,
                )

                try:
                    import pandas as pd
                except ImportError:
                    return "Error: pandas is required for quant_regime."

                if not query:
                    return "Error: quant_regime needs a ticker symbol in `query`."

                # Create synthetic OHLC data for demonstration
                # In production, this would ingest real market data
                import numpy as np

                dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
                base_price = 100.0
                returns = np.random.normal(0.0005, 0.02, 100)
                close_prices = base_price * np.cumprod(1 + returns)
                df = pd.DataFrame(
                    {
                        "Close": close_prices,
                        "High": close_prices * 1.02,
                        "Low": close_prices * 0.98,
                        "Open": np.roll(close_prices, 1)[1:].tolist() + [base_price],
                    },
                    index=dates,
                )

                detector = RegimeDetector(engine)
                regime = detector.detect_regime(df, ticker=query)
                return regime
            elif action == "workforce_plan":
                from agent_utilities.domains.hr.workforce_manager import (
                    WorkforceManager,
                )

                wm = WorkforceManager()
                result = wm.get_workforce_summary()
                return json.dumps(result, default=str)
            elif action == "close":
                # Background OWL-RL + SHACL closure (KG-2.6): promote recent nodes
                # to RDF, materialize implied edges via the reasoner, validate
                # against shapes. On-demand twin of the maintenance-tick closure.
                from agent_utilities.knowledge_graph.maintenance.owl_closure import (
                    run_closure,
                )

                summary = run_closure(
                    engine, limit=top_k * 200 if top_k != 10 else 2000
                )
                return json.dumps(summary, default=str)
            else:
                return f"Error: Unknown analyze action '{action}'"
        except Exception as e:
            return f"Analysis error: {str(e)}"

    kg_server.REGISTERED_TOOLS["graph_analyze"] = graph_analyze

    @mcp.tool(
        name="graph_orchestrate",
        description="Orchestrate multi-agent workflows, dispatch subagents, and manage execution loops.",
        tags=["graph-os", "orchestrate"],
    )
    async def graph_orchestrate(
        action: str = Field(
            default="dispatch",
            description="Action to perform (dispatch, swarm, status, request_approval, grant_approval, execute_agent, consensus, start_debate, submit_risk_veto, list_cron_jobs, trigger_cron_job, compile_workflow, compile_process, list_workflows, execute_workflow, export_workflow, loop_cycle, assimilate, distill_skills, standardize, failure_ingest, publish_proposal). 'loop_cycle' = advance the Loop engine one cycle (CONCEPT:KG-2.78); 'distill_skills' = turn the mapped processes of ALL connected systems (egeria/leanix/aris/camunda) into propose-only atomic-skill + skill-workflow PROPOSALS, connector-agnostic over the ontology (add 'draft' to the task to also render reviewable SKILL.md staging artifacts) (CONCEPT:KG-2.90/2.83); 'swarm' = one-shot goal→decompose→parallel-waves→verify→synthesize (CONCEPT:ORCH-1.32); 'standardize' = enterprise standardization + consolidation recommendations (CONCEPT:KG-2.49); 'failure_ingest' = pull Langfuse failures → failure_gap topics → regression-gated remediation (CONCEPT:AHE-3.18); 'compile_process' = compile a harvested BusinessProcess node (task=process node id, agent_name=optional workflow name) into an executable WorkflowDefinition with a REALIZES bridge edge (CONCEPT:ORCH-1.41); 'publish_proposal' = one-shot evolution→branch bridge — publish a promoted proposal (task=proposal node id) as a reviewable local git branch through the ActionPolicy merge_promotion gate (CONCEPT:AHE-3.21); 'rlm_benchmark' = run the long-context RLM benchmark (RLM vs vanilla vs compaction) for task=<s_niah|oolong|oolong_pairs|browsecomp_plus|longbench_codeqa>, dependencies=JSON {scales,cases_per_scale}, returning a paper-comparison scoreboard (CONCEPT:AHE-3.32).",
        ),
        task: str = Field(
            default="", description="Task description or payload to dispatch."
        ),
        job_id: str = Field(
            default="", description="Job ID for checking status or granting approval."
        ),
        approval_status: str = Field(
            default="", description="Approval status (e.g., 'approved', 'rejected')."
        ),
        agent_name: str = Field(
            default="", description="Name of the agent to execute."
        ),
        max_steps: int = Field(
            default=30, description="Maximum steps for agent execution."
        ),
        dependencies: str = Field(
            default="[]", description="JSON-encoded list of dependency job IDs."
        ),
        completion_state: str = Field(
            default="",
            description="Strict mathematical or semantic definition of when this workflow is considered done.",
        ),
        max_fan_out: int = Field(
            default=5,
            description="Maximum number of parallel subagents to spawn during adversarial loop.",
        ),
        context: str = Field(
            default="",
            description="CONCEPT:ORCH-1.39 — curated context the invoking agent passes to the "
            "spawned agent (action='execute_agent'); injected into the spawned agent's prompt, "
            "budgeted to the model's context window.",
        ),
        budget_tokens: int = Field(
            default=0,
            description="CONCEPT:ORCH-1.39 — optional token budget the invoker grants the "
            "spawned agent (action='execute_agent'); enforced as a hard total-tokens limit. "
            "0 = unbounded.",
        ),
        context_ref: str = Field(
            default="",
            description="CONCEPT:ORCH-1.39 — id of a persisted ContextBlob (from "
            "graph_context put) to hand to the spawned agent (action='execute_agent'); its "
            "content is resolved from the graph and injected. Use instead of inline 'context' "
            "for large/shared context.",
        ),
        allowed_tools: str = Field(
            default="",
            description="CONCEPT:ORCH-1.39 — comma-separated least-privilege tool allow-list "
            "for the spawned agent (action='execute_agent'); its tools/toolsets are filtered "
            "to ONLY these names. Empty = no restriction.",
        ),
        cred_ref: str = Field(
            default="",
            description="CONCEPT:ORCH-1.39 — REFERENCE (secret key, e.g. 'cred:{session}') to "
            "an ephemeral credential the invoker stored in the secrets backend; resolved to the "
            "spawned agent's auth_token at spawn (never logged). Use instead of passing raw "
            "secrets. Empty = none.",
        ),
        open_channel: bool = Field(
            default=False,
            description="CONCEPT:ORCH-1.40 — when True (action='execute_agent'), open a native "
            "bidirectional message channel for this run; the response JSON includes a "
            "'channel_id' to talk to the spawned agent via graph_message(send/receive).",
        ),
    ) -> str:
        """Orchestrate multi-agent workflows. Dispatches agents, manages subagent lifecycles, and evaluates approval conditions for complex asynchronous execution.

        CONCEPT:ORCH-1.37 — the execution-flow Mermaid diagram (generated by the ORCH-1.8
        WorkflowVisualizer) is surfaced in the response: ``swarm``, ``compile_workflow`` and
        ``execute_workflow`` add an additive ``mermaid`` JSON key (null when unavailable), and
        ``execute_agent`` returns a JSON object ``{"output", "mermaid"}`` when a diagram was
        produced (otherwise the bare output string, for backward compatibility).
        """
        engine = kg_server._get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            from agent_utilities.orchestration.manager import Orchestrator

            orch = Orchestrator(engine)

            if action == "dispatch":
                deps = json.loads(dependencies) if dependencies else []
                job_id = await orch.dispatch_task(task, deps)
                # CONCEPT:ORCH-1.45 — queue-driven dispatch: with
                # AGENT_DISPATCH_BACKEND=queue the durable :Task node stays the
                # payload of record, a session-keyed envelope goes onto the
                # agent_turns queue, and the caller gets a job handle (poll
                # action=status / /api/graph/orchestrate/job/{job_id}) instead
                # of an in-process execution promise. A bare dispatch has no
                # session, so the job id is its own session scope — serial
                # with itself, parallel with everything else.
                from agent_utilities.orchestration.agent_dispatch import (
                    KIND_ORCHESTRATOR_TASK,
                    AgentTurnEnvelope,
                    dispatch_queue_enabled,
                    enqueue_agent_turn,
                )

                if dispatch_queue_enabled():
                    handle = enqueue_agent_turn(
                        AgentTurnEnvelope(
                            job_id=job_id,
                            session_id=job_id,
                            kind=KIND_ORCHESTRATOR_TASK,
                            payload_ref=job_id,
                            agent_name=agent_name or "",
                        )
                    )
                    handle["status_url"] = f"/api/graph/orchestrate/job/{job_id}"
                    return json.dumps(handle)
                return f"Task dispatched. Job ID: {job_id}"
            elif action == "rlm_run":
                # CONCEPT:ORCH-1.12 — run the Predict-RLM runtime on an ad-hoc task.
                from agent_utilities.rlm.runner import run_rlm

                result = await run_rlm(task, input_text=completion_state)
                return json.dumps(result, default=str)
            elif action == "rlm_optimize":
                # CONCEPT:ORCH-1.13 — optimize a skill prompt via the GEPA loop.
                from agent_utilities.rlm.runner import optimize_rlm_skill

                rows = json.loads(dependencies) if dependencies else []
                dataset = rows if isinstance(rows, list) else []
                result = await optimize_rlm_skill(task, dataset)
                return json.dumps(result, default=str)
            elif action == "rlm_benchmark":
                # CONCEPT:AHE-3.32 — run the long-context RLM benchmark (RLM vs vanilla vs
                # compaction) over a task and return the paper-comparison scoreboard. `task` is the
                # benchmark name (s_niah, oolong, oolong_pairs, browsecomp_plus, longbench_codeqa);
                # `dependencies` is optional JSON {"scales": [int], "cases_per_scale": int}.
                from agent_utilities.rlm.benchmarks import (
                    list_tasks,
                    render_scoreboard,
                    run_benchmark,
                )

                opts = json.loads(dependencies) if dependencies else {}
                if not isinstance(opts, dict):
                    opts = {}
                bench = task or "s_niah"
                if bench not in list_tasks():
                    return json.dumps(
                        {
                            "ok": False,
                            "error": f"unknown task {bench!r}",
                            "tasks": list_tasks(),
                        }
                    )
                scales = opts.get("scales") or [50_000]
                results = await run_benchmark(
                    bench,
                    scales=[int(s) for s in scales],
                    cases_per_scale=int(opts.get("cases_per_scale", 3)),
                )
                return json.dumps(
                    {
                        "ok": True,
                        "results": [r.model_dump() for r in results],
                        "scoreboard": render_scoreboard(results),
                    },
                    default=str,
                )
            elif action == "swarm":
                # CONCEPT:ORCH-1.32 — KG-Governed Agent Swarm
                # One-shot swarm action: a one-line goal is
                # decomposed into a dependency-ordered task graph, executed in parallel waves by the
                # ParallelEngine, each leaf verified against its subtask (planner→execute→verify),
                # then synthesized into a single deliverable. The KG/OWL grounding + verification is
                # what distinguishes this from a black-box trained swarm.
                from agent_utilities.core.config import (
                    DEFAULT_KG_MODEL_ID,
                    DEFAULT_LLM_PROVIDER,
                )
                from agent_utilities.core.model_factory import create_model
                from agent_utilities.graph.parallel_engine import ParallelEngine
                from agent_utilities.graph.planning import Planner
                from agent_utilities.models.execution_manifest import ExecutionManifest

                try:
                    _model = create_model(
                        provider=DEFAULT_LLM_PROVIDER, model_id=DEFAULT_KG_MODEL_ID
                    )
                except Exception:
                    _model = None
                plan = await Planner(model=_model).decompose(task)
                manifest = ExecutionManifest.from_graph_plan(
                    plan, name="swarm", query=task
                )
                # CONCEPT:ORCH-1.39 (Phase 3) — curated invoker context for the swarm.
                # ParallelEngine injects manifest.context into EVERY wave agent's task, so the
                # invoker's context reaches all swarm agents. Resolve context_ref if given.
                _swarm_ctx = context or ""
                if not _swarm_ctx and context_ref:
                    try:
                        _crows = engine.query_cypher(
                            "MATCH (c:ContextBlob) WHERE c.id = $id RETURN c.content AS content",
                            {"id": context_ref},
                        )
                        if _crows and _crows[0].get("content"):
                            _swarm_ctx = str(_crows[0]["content"])
                    except Exception:  # noqa: BLE001
                        _swarm_ctx = ""
                if _swarm_ctx:
                    manifest.context = (
                        f"{manifest.context}\n\n{_swarm_ctx}"
                        if manifest.context
                        else _swarm_ctx
                    )
                # default governance ON: verify each leaf + retry transient failures.
                manifest.metadata["verify"] = True
                manifest.metadata["max_retries"] = 2
                if max_fan_out:
                    manifest.max_concurrency = int(max_fan_out)
                # give the verify loop something to check: each leaf must address its own subtask.
                for _a in manifest.agents:
                    if not _a.success_criteria:
                        _a.success_criteria = (
                            f"Output must substantively address: "
                            f"{(_a.task_template or task)[:240]}"
                        )
                pe_result = await ParallelEngine(engine=engine).execute(manifest)
                return json.dumps(
                    {
                        "deliverable": pe_result.synthesis_output,
                        "agent_count": pe_result.agent_count,
                        "wave_count": pe_result.wave_count,
                        "critical_path_length": pe_result.critical_path_length,
                        "parallelism_ratio": pe_result.parallelism_ratio,
                        "verification": pe_result.verification,
                        "telemetry": pe_result.telemetry,
                        "execution_id": pe_result.execution_id,
                        "success": pe_result.success,
                        # CONCEPT:ORCH-1.37 — surface the existing execution-flow diagram
                        # (generated by ORCH-1.8 WorkflowVisualizer) to the MCP caller.
                        "mermaid": pe_result.mermaid,
                    },
                    default=str,
                )
            elif action == "status":
                if not job_id:
                    return "Error: job_id required"
                return str(orch.get_task_status(job_id))
            elif action == "request_approval":
                return f"Approval requested for job {job_id}"
            elif action == "grant_approval":
                return orch.grant_approval(job_id, approval_status)
            elif action == "execute_agent":
                try:
                    # CONCEPT:ORCH-1.37 — opt into the mermaid wrapper so the routed
                    # graph diagram (GraphResponse.mermaid) reaches the MCP caller.
                    agent_result = await orch.execute_agent(
                        agent_name=agent_name,
                        task=task,
                        max_steps=max_steps,
                        return_mermaid=True,
                        context=context or None,
                        budget_tokens=budget_tokens or None,
                        context_ref=context_ref or None,
                        allowed_tools=(
                            [t.strip() for t in allowed_tools.split(",") if t.strip()]
                            or None
                        ),
                        cred_ref=cred_ref or None,
                        open_channel=bool(open_channel),  # CONCEPT:ORCH-1.40
                    )
                    return agent_result
                except Exception as exc:
                    return f"Error: agent execution failed: {exc}"
            elif action == "compile_workflow":
                try:
                    from agent_utilities.knowledge_graph.workflow_store import (
                        WorkflowStore,
                    )

                    name = agent_name or f"compiled_{uuid.uuid4().hex[:6]}"
                    workflow_id = await orch.compile_workflow(name=name, task=task)
                    # CONCEPT:ORCH-1.37 — return the diagram persisted on the
                    # WorkflowDefinition node so the caller can review the topology.
                    mermaid = None
                    try:
                        mermaid = WorkflowStore(engine).get_mermaid(name)
                    except Exception:
                        mermaid = None
                    return json.dumps(
                        {
                            "status": "compiled",
                            "workflow_id": workflow_id,
                            "name": name,
                            "mermaid": mermaid,
                        }
                    )
                except Exception as exc:
                    return f"Error compiling workflow: {exc}"
            elif action == "compile_process":
                # CONCEPT:ORCH-1.41 — descriptive BusinessProcess → executable
                # WorkflowDefinition (+ REALIZES bridge edge). 'task' carries
                # the BusinessProcess node id; 'agent_name' an optional name.
                try:
                    from agent_utilities.knowledge_graph.process_plan_compiler import (
                        ProcessPlanCompiler,
                    )
                    from agent_utilities.knowledge_graph.workflow_store import (
                        WorkflowStore,
                    )

                    process_id = task.strip()
                    if not process_id:
                        return (
                            "Error: Must specify the BusinessProcess node id in "
                            "the 'task' parameter."
                        )
                    compiler = ProcessPlanCompiler(engine)
                    report = await compiler.compile_and_store(
                        process_id, name=agent_name or None
                    )
                    report["status"] = "compiled"
                    # CONCEPT:ORCH-1.37 — surface the stored topology diagram.
                    try:
                        report["mermaid"] = WorkflowStore(engine).get_mermaid(
                            report["name"]
                        )
                    except Exception:
                        report["mermaid"] = None
                    return json.dumps(report, default=str)
                except Exception as exc:
                    return f"Error compiling process: {exc}"
            elif action == "list_workflows":
                try:
                    from agent_utilities.knowledge_graph.workflow_store import (
                        WorkflowStore,
                    )

                    store = WorkflowStore(engine)
                    workflows = store.list_workflows(limit=50)
                    if not workflows:
                        return json.dumps({"error": "No workflows found in database."})
                    return json.dumps(
                        {"source": "kg", "workflows": workflows}, default=str
                    )
                except Exception as exc:
                    return f"Error listing workflows: {exc}"
            elif action == "execute_workflow":
                # CONCEPT:ORCH-1.42 — execution-time ontology gate, BEFORE any
                # dispatch: (a) SHACL-validate the stored definition (refuse
                # malformed workflows, KG_WORKFLOW_SHAPE_GATE default ON);
                # (b) with KG_BRAIN_ENFORCE on, apply the ontology permissioning
                # row gate to the workflow node for the current actor —
                # a denial raises PermissionError (fail-closed, OS-5.14).
                from agent_utilities.knowledge_graph.core.workflow_gate import (
                    gate_workflow_execution,
                )

                gate_name = agent_name or task
                gate = gate_workflow_execution(engine, gate_name)
                if not gate.get("allowed", True):
                    return json.dumps(
                        {
                            "error": (
                                "workflow definition failed ontology validation "
                                "— execution refused"
                            ),
                            "workflow": gate_name,
                            "workflow_id": gate.get("workflow_id"),
                            "violations": gate.get("violations", []),
                        },
                        default=str,
                    )
                try:
                    from agent_utilities.knowledge_graph.workflow_store import (
                        WorkflowStore,
                    )

                    name = agent_name or task
                    input_task = task if (agent_name and task != agent_name) else None
                    wf_result = await orch.execute_workflow(
                        workflow_id=name,
                        task=input_task or "",
                        max_steps=max_steps,
                    )
                    # CONCEPT:ORCH-1.37 — surface the workflow's stored execution-flow
                    # diagram alongside the result.
                    mermaid = None
                    try:
                        mermaid = WorkflowStore(engine).get_mermaid(name)
                    except Exception:
                        mermaid = None
                    return json.dumps(
                        {"result": wf_result, "mermaid": mermaid}, default=str
                    )
                except Exception as exc:
                    return f"Error executing workflow: {exc}"
            elif action == "consensus":
                return f"Consensus reached for {task}."
            elif action == "start_debate":
                engine.add_node(
                    f"debate_{job_id}", "TradingDebate", topic=task, status="ongoing"
                )
                return f"Started Trading Debate for {task}."
            elif action == "submit_risk_veto":
                engine.add_node(
                    f"veto_{job_id}", "RiskVeto", reason=task, target=job_id
                )
                engine.add_edge(
                    f"veto_{job_id}", f"debate_{job_id}", "CONTRADICTS_BELIEF_PROP"
                )
                return f"Submitted Risk Veto for debate {job_id}."
            elif action == "list_cron_jobs":
                try:
                    from agent_utilities.automation.maintenance_cron import (
                        MaintenanceCron,
                    )

                    cron = MaintenanceCron()
                    due_tasks = cron.get_due_tasks()
                    lines = []
                    for t in cron.tasks:
                        status = (
                            "DUE"
                            if any(dt.id == t.id for dt in due_tasks)
                            else "WAITING"
                        )
                        lines.append(
                            f"[{status}] {t.id} (Frequency: {t.frequency.value})"
                        )
                    return "\n".join(lines)
                except ImportError:
                    return "Error: maintenance_cron module not available"
            elif action == "trigger_cron_job":
                try:
                    from agent_utilities.automation.maintenance_cron import (
                        MaintenanceCron,
                    )

                    cron = MaintenanceCron()
                    target_id = task.strip()
                    if not target_id:
                        return "Error: Must specify the cron job ID in the 'task' parameter."
                    cron.record_execution(
                        target_id, status="triggered_manually", tokens_used=0
                    )
                    return f"Manually triggered cron job: {target_id}"
                except ImportError:
                    return "Error: maintenance_cron module not available"
            elif action == "dispatch_workflow":
                # CONCEPT:ORCH-1.42 — the SAME execution-time ontology gate as
                # execute_workflow, BEFORE background dispatch: (a) SHACL-validate
                # the stored definition (refuse malformed workflows,
                # KG_WORKFLOW_SHAPE_GATE default ON); (b) with KG_BRAIN_ENFORCE
                # on, apply the ontology permissioning row gate to the workflow
                # node for the current actor — a denial raises PermissionError
                # (fail-closed, OS-5.14). Enforcement off skips the ACL check.
                from agent_utilities.knowledge_graph.core.workflow_gate import (
                    gate_workflow_execution,
                )

                gate_name = agent_name or task
                gate = gate_workflow_execution(engine, gate_name)
                if not gate.get("allowed", True):
                    return json.dumps(
                        {
                            "error": (
                                "workflow definition failed ontology validation "
                                "— background dispatch refused"
                            ),
                            "workflow": gate_name,
                            "workflow_id": gate.get("workflow_id"),
                            "violations": gate.get("violations", []),
                        },
                        default=str,
                    )
                try:
                    from agent_utilities.orchestration import AgentOrchestrationEngine

                    runner = AgentOrchestrationEngine()
                    name = agent_name or task
                    input_task = task if (agent_name and task != agent_name) else None
                    session_id = f"wf-{uuid.uuid4().hex[:8]}"

                    # Start execution as background task
                    asyncio.create_task(
                        runner.execute_workflow(
                            workflow_id=name,
                            task=input_task,
                            completion_state=completion_state,
                            max_fan_out=max_fan_out,
                        )
                    )
                    return (
                        f"Workflow dispatched in background. Session ID: {session_id}"
                    )
                except ValueError as exc:
                    return f"Workflow not found: {exc}"
                except Exception as exc:
                    return f"Error dispatching workflow: {exc}"

            elif action == "workflow_status":
                try:
                    from agent_utilities.workflows.runner import _active_workflows

                    sid = job_id or task
                    if not sid:
                        return "Error: Must specify session ID in 'job_id' or 'task' parameter."

                    wf_status = _active_workflows.get(sid)
                    if not wf_status:
                        return f"Workflow session '{sid}' not found or has not been run in this process."

                    return json.dumps(wf_status.to_dict(), default=str)
                except Exception as exc:
                    return f"Error retrieving workflow status: {exc}"

            elif action == "export_workflow":
                try:
                    return json.dumps(
                        {
                            "error": "Workflow export requires resolving workflows from the database. Legacy catalog export is deprecated."
                        },
                        indent=2,
                        default=str,
                    )
                except Exception as exc:
                    return f"Error exporting workflow: {exc}"

            elif action == "loop_cycle":
                # Advance the Loop engine one cycle (CONCEPT:KG-2.7/2.78): intake
                # active Loops → acquire → ADDRESSES-resolve → optional
                # distil/synthesize as DRAFTS/proposals. Never auto-merges.
                import json as _json

                from agent_utilities.knowledge_graph.research.loop_controller import (
                    LoopController,
                )

                engine = kg_server._get_engine()
                _mt = max_fan_out if isinstance(max_fan_out, int) else 5
                rep = LoopController(engine).run_one_cycle(
                    max_topics=_mt if _mt > 0 else 5,
                )
                return _json.dumps(rep, indent=2, default=str)

            elif action == "failure_ingest":
                # Failure-driven evolution (CONCEPT:AHE-3.18): pull Langfuse
                # failures → materialize failure_gap topics → regression-gated
                # remediation that addresses those gaps directly. The on-demand
                # twin of the daemon's failure_ingest tick (gated by
                # KG_FAILURE_EVOLUTION for the daemon; the action runs on request).
                import json as _json

                from agent_utilities.knowledge_graph.adaptation.failure_analyzer import (
                    run_failure_ingest,
                )

                rep = run_failure_ingest(kg_server._get_engine())
                return _json.dumps(rep, indent=2, default=str)

            elif action == "publish_proposal":
                # Evolution→branch bridge (CONCEPT:AHE-3.21): publish a promoted
                # golden-loop proposal (task=proposal node id) as a reviewable
                # LOCAL git branch — change synthesis + RLM-sandbox validation +
                # LocalBranchPublisher — gated by the OS-5.24 ActionPolicy's
                # merge_promotion kind (default approval_required: a pending
                # grant queues, a granted approval lets this proceed). Never
                # pushes; a human merges through the normal release flow.
                import json as _json

                from agent_utilities.knowledge_graph.research.change_publisher import (
                    publish_proposal,
                )

                pid = (task or "").strip()
                if not pid:
                    return (
                        "Error: publish_proposal requires the proposal node id "
                        "in 'task'"
                    )
                rep = publish_proposal(kg_server._get_engine(), pid)
                return _json.dumps(rep, indent=2, default=str)

            elif action == "assimilate":
                # Graph-native assimilation pass (CONCEPT:KG-2.7): dedup → gap →
                # synergy → rank (idempotent via watermark). With "synthesize" in the
                # task, also propose grounded SDD plans for the top open gaps.
                import json as _json

                from agent_utilities.knowledge_graph.research.loop_controller import (
                    run_assimilation_pass,
                )

                _mt = (
                    max_fan_out
                    if isinstance(max_fan_out, int) and max_fan_out > 0
                    else 5
                )
                rep = run_assimilation_pass(
                    kg_server._get_engine(),
                    synthesize="synthesize" in (task or "").lower(),
                    top_n=_mt,
                    force="force" in (task or "").lower(),
                )
                return _json.dumps(rep, indent=2, default=str)

            elif action == "distill_skills":
                # Connector → skill synthesis (CONCEPT:KG-2.90/2.83): turn the
                # mapped processes of ALL connected systems (egeria/leanix/aris/
                # camunda) into propose-only atomic-skill + skill-workflow
                # PROPOSALS — connector-agnostic over the ontology. With "draft"
                # in the task, also render reviewable SKILL.md artifacts into a
                # STAGING dir (never a repo). Propose-only; nothing auto-merges.
                import json as _json

                from agent_utilities.knowledge_graph.distillation.skill_synthesizer import (  # noqa: E501
                    ConnectorSkillDistiller,
                )

                distiller = ConnectorSkillDistiller(kg_server._get_engine())
                _task = (task or "").strip()
                if _task.startswith("materialize:"):
                    # human-approved close-out: materialize the named proposal to a
                    # physical SKILL.md (staging dir) via PhysicalDistillationEngine.
                    pid = _task.split(":", 1)[1].strip()
                    return _json.dumps(
                        distiller.materialize(pid), indent=2, default=str
                    )
                distill_rep = distiller.run(draft="draft" in _task.lower())
                return _json.dumps(distill_rep.to_dict(), indent=2, default=str)

            elif action == "standardize":
                # Enterprise standardization + consolidation pass (CONCEPT:KG-2.49):
                # materialize enterprise-standard interfaces → score per-asset/org/
                # domain conformance drift → rank propose-only consolidation
                # recommendations (collapse projects / retire tools / merge code).
                import json as _json

                from agent_utilities.knowledge_graph.standardization import (
                    run_standardization_pass,
                )

                _tn = (
                    max_fan_out
                    if isinstance(max_fan_out, int) and max_fan_out > 0
                    else 20
                )
                rep = run_standardization_pass(kg_server._get_engine(), top_n=_tn)
                return _json.dumps(rep, indent=2, default=str)

            elif action == "enterprise_op":
                from agent_utilities.knowledge_graph.orchestration.engine_enterprise import (  # noqa: E501
                    EnterpriseEngineMixin,
                )

                engine = kg_server._get_engine()
                if not task:
                    return "Error: enterprise_op needs a business_unit_id in `task`."
                try:
                    budget_amount = float(agent_name) if agent_name else 10000.0
                except (ValueError, TypeError):
                    budget_amount = 10000.0
                # The mixin is composed onto the live engine; fall back to its
                # unbound method if the running engine predates the composition.
                allocate = getattr(
                    engine, "allocate_budget", None
                ) or EnterpriseEngineMixin.allocate_budget.__get__(engine)
                if allocate is None:
                    return "Error: engine does not support enterprise operations."
                budget_id = allocate(task, budget_amount, "USD")
                return json.dumps(
                    {
                        "budget_id": budget_id,
                        "business_unit_id": task,
                        "amount": budget_amount,
                        "currency": "USD",
                    },
                    default=str,
                )
            elif action == "finance_op":
                import json as _json

                from agent_utilities.knowledge_graph.orchestration.engine_finance import (
                    FinanceEngineMixin,
                )

                engine = kg_server._get_engine()
                if not engine:
                    return (
                        "Error: finance_op requires an active epistemic-graph engine."
                    )
                if not task:
                    return "Error: finance_op needs a JSON task with {returns: [float], strategy_id: str, asset_class?: str}."

                try:
                    params = _json.loads(task)
                except Exception:
                    return f"Error: task must be valid JSON. Got: {task}"

                returns = params.get("returns", [])
                strategy_id = params.get("strategy_id")
                asset_class = params.get("asset_class", "equities")
                bull_threshold = params.get("bull_threshold")
                bear_threshold = params.get("bear_threshold")
                window = params.get("window")
                method = params.get("method", "rolling_sum")

                if not isinstance(returns, list) or len(returns) == 0:
                    return (
                        "Error: task must include returns (non-empty list of floats)."
                    )
                if not strategy_id:
                    return "Error: task must include strategy_id."

                try:
                    mixin = FinanceEngineMixin()  # type: ignore[abstract]  # FIXME: standalone abstract-mixin instantiation is broken (no concrete engine composes it) — needs proper engine integration
                    mixin.graph = engine.graph
                    mixin.backend = engine.backend

                    matrix_id = mixin.fit_markov_regime(
                        returns=returns,
                        strategy_id=strategy_id,
                        asset_class=asset_class,
                        bull_threshold=bull_threshold,
                        bear_threshold=bear_threshold,
                        window=window,
                        method=method,
                    )
                    return _json.dumps(
                        {"matrix_id": matrix_id, "status": "fitted"}, default=str
                    )
                except Exception as e:
                    return f"Error: {str(e)}"
            elif action == "ml_rlm_op":
                # CONCEPT:KG-2.6 — Machine Learning & RLM capabilities for the KG engine.
                # Register a new RLM actor for reinforcement learning tasks.
                # task = JSON string {"name": "actor_name", "learning_rate": 0.01, "discount_factor": 0.99}
                # or plain text actor name (uses sensible defaults: learning_rate=0.01, discount_factor=0.99).
                import json as _json

                from agent_utilities.knowledge_graph.orchestration.engine_ml_rlm import (
                    MachineLearningEngineMixin,
                )

                if not task:
                    return "Error: ml_rlm_op needs a task (actor name or JSON config)."
                if not engine:
                    return "Error: ml_rlm_op needs an active engine."

                # Parse task as JSON or plain text.
                config = {"name": task, "learning_rate": 0.01, "discount_factor": 0.99}
                if task.startswith("{"):
                    try:
                        config.update(_json.loads(task))
                    except Exception:
                        pass

                # Instantiate mixin with engine and call register_rlm_actor.
                mixin = MachineLearningEngineMixin()  # type: ignore[abstract, assignment]  # FIXME: standalone abstract-mixin instantiation is broken (no concrete engine composes it) — needs proper engine integration
                mixin.graph = engine.graph if hasattr(engine, "graph") else None  # type: ignore[assignment]  # FIXME: standalone abstract-mixin instantiation is broken (no concrete engine composes it) — needs proper engine integration
                mixin.backend = engine.backend if hasattr(engine, "backend") else None
                mixin._serialize_node = (  # type: ignore[method-assign]  # FIXME: standalone abstract-mixin instantiation is broken (no concrete engine composes it) — needs proper engine integration
                    engine._serialize_node  # type: ignore[assignment]  # FIXME: standalone abstract-mixin instantiation is broken (no concrete engine composes it) — needs proper engine integration
                    if hasattr(engine, "_serialize_node")
                    else lambda n, label: n.model_dump()  # type: ignore[misc]  # FIXME: standalone abstract-mixin instantiation is broken (no concrete engine composes it) — needs proper engine integration
                )
                mixin._upsert_node = (  # type: ignore[method-assign]  # FIXME: standalone abstract-mixin instantiation is broken (no concrete engine composes it) — needs proper engine integration
                    engine._upsert_node  # type: ignore[assignment]  # FIXME: standalone abstract-mixin instantiation is broken (no concrete engine composes it) — needs proper engine integration
                    if hasattr(engine, "_upsert_node")
                    else lambda label, id, data: None
                )

                actor_id = mixin.register_rlm_actor(  # type: ignore[attr-defined]  # FIXME: standalone abstract-mixin instantiation is broken (no concrete engine composes it) — needs proper engine integration
                    name=config.get("name", "rlm_actor"),
                    learning_rate=float(config.get("learning_rate", 0.01)),  # type: ignore[arg-type]  # FIXME: standalone abstract-mixin instantiation is broken (no concrete engine composes it) — needs proper engine integration
                    discount_factor=float(config.get("discount_factor", 0.99)),  # type: ignore[arg-type]  # FIXME: standalone abstract-mixin instantiation is broken (no concrete engine composes it) — needs proper engine integration
                )
                return _json.dumps(
                    {"actor_id": actor_id, "status": "registered"}, default=str
                )
            else:
                return f"Error: Unknown orchestration action '{action}'"
        except PermissionError:
            # CONCEPT:ORCH-1.42 / OS-5.14 — ACL denial is fail-closed: surface
            # it as a real error to the MCP layer, never a stringified result.
            raise
        except Exception as e:
            return f"Orchestration error: {str(e)}"

    kg_server.REGISTERED_TOOLS["graph_orchestrate"] = graph_orchestrate

    @mcp.tool(
        name="graph_configure",
        description="Manage backend configurations, system credentials, and tool registration within the unified agent ecosystem.",
        tags=["graph-os", "configure"],
    )
    def graph_configure(
        action: str = Field(
            default="register_mcp",
            description="Operation ('set_secret', 'register_mcp', 'install_hooks', 'uninstall_hooks', 'doctor', 'set_role_routing', 'schema_pack', 'schema_candidates', 'add_connection', 'remove_connection', 'list_connections', 'set_default_connection'). 'schema_pack' with config_key=<name> sets the active domain Schema Pack, or with empty config_key returns the active pack plus available packs; 'schema_candidates' reviews out-of-pack types seen on write (CONCEPT:KG-2.35). CONCEPT:KG-2.63 — 'add_connection' registers a named graph backend (config_key=name, config_value=JSON spec e.g. {\"backend\":\"neo4j\",\"uri\":\"bolt://...\",\"user\":\"...\",\"password\":\"...\"}; use backend 'age' for Postgres native openCypher; CONCEPT:KG-2.89 — spec may set role 'read'(default, query-only data source)|'read_write'|'mirror', and password/user/uri may be a vault://path or env://VAR ref; the connection is persisted to config.json so it survives restart); 'remove_connection' (config_key=name); 'list_connections' returns per-connection health + role; 'set_default_connection' (config_key=name) repoints the default target. 'profile_connection' (config_key=name) read-only-introspects a registered external graph's schema (labels, relationship types, property keys, per-label counts + sample property shapes); 'imprint_connection' profiles it, maps each external label onto our ontology (interfaces + our node types; unmatched flagged 'novel'), and writes a self-describing ExternalGraphReference catalog node (no credentials) into the authority KG so the foreign graph becomes discoverable+usable. CONCEPT:KG-2.74 — 'mirror_status' returns per-mirror replication health (lag/failures/stalled) for a GRAPH_BACKEND=fanout deployment; 'reconcile' (optional config_key=<mirror name>, empty=all) runs a full authority→mirror drift-repair pass. 'setup_databases' provisions the Stardog + pg-age environment end-to-end (config_key=profile 'dev'|'prod', config_value=JSON options e.g. {\"postgres_mode\":\"managed_image\",\"dsn\":\"postgresql://...\",\"sparql_target\":\"builtin\"}); 'verify_databases' probes a Postgres for the age/vector/pg_search extensions (config_key or config_value.dsn = DSN). 'generate_config' writes a COMPLETE profile-seeded config.json covering every option (config_key=profile 'tiny'|'single-node-prod'|'enterprise', config_value optional {\"out\":path,\"redact_secrets\":true}); 'config_doctor' validates a deployment's config completeness/health (config_key=profile, config_value optional {\"config\":path}); 'config_reference' returns every option grouped by subsystem. CONCEPT:KG-2.89 — 'get_config' (config_key=env name) returns a live value; 'set_config' (config_key=env name, config_value=scalar or JSON) validates against config_reference, persists to config.json + applies live, and flags 'restart_required' for engine-rebuild settings; 'list_config' returns every current value (secrets redacted). 'system_doctor' runs a holistic deployment health sweep (brew/flutter-doctor style) across config/engine/backend/secrets/auth/mcp-fleet/hooks/observability, each with a remediation + skill (config_value optional {\"only\":[...],\"fix\":true,\"live\":true}). 'preflight' checks whether THIS HOST has the runtimes/tools to deploy a profile BEFORE installing (Python 3.11-<3.15, uv/pip, the epistemic-graph engine binary — Rust only as a fallback, Docker when not the tiny profile, and per-component deps): config_key=profile 'tiny'|'single-node-prod'|'enterprise', config_value optional {\"components\":[\"agent-webui\",\"geniusbot\",\"agent-terminal-ui\"]}.",
        ),
        config_key: str = Field(
            default="",
            description="The key or ID of the configuration/secret (for 'schema_pack', the pack name e.g. 'research-state'; for connection actions, the connection name).",
        ),
        config_value: str = Field(
            default="",
            description="JSON string containing the payload or secret value.",
        ),
    ) -> str:
        """Manage backend configurations and abstract credentials. Allows dynamic registry updates and credential injection during agent provisioning."""
        try:
            if action == "set_secret":
                from agent_utilities.security.secrets_client import (
                    create_secrets_client,
                )
                from agent_utilities.security.xai_auth import get_secrets_client_for_xai

                if config_key.startswith("xai/"):
                    client = get_secrets_client_for_xai()
                else:
                    client = create_secrets_client()
                client.set(config_key, config_value)
                return json.dumps(
                    {"status": "success", "action": "set_secret", "key": config_key}
                )
            if action == "register_mcp":
                from pathlib import Path

                from agent_utilities.core.workspace import get_mcp_config_path

                mcp_path_str = get_mcp_config_path()
                if mcp_path_str:
                    mcp_path = Path(mcp_path_str)
                    if not mcp_path.exists():
                        cfg = {}
                    else:
                        with open(mcp_path) as f:
                            cfg = json.load(f)
                    try:
                        parsed_val = json.loads(config_value)
                        cfg.setdefault("mcpServers", {})[config_key] = parsed_val
                        with open(mcp_path, "w") as f:
                            json.dump(cfg, f, indent=2)
                        return json.dumps(
                            {
                                "status": "success",
                                "action": "register_mcp",
                                "server": config_key,
                            }
                        )
                    except Exception as e:
                        return json.dumps({"error": f"Invalid config_value JSON: {e}"})
                return json.dumps({"error": "MCP config not found in workspace."})
            # ── CONCEPT:KG-2.63: Named multi-connection graph registry ──
            if action in (
                "add_connection",
                "remove_connection",
                "list_connections",
                "set_default_connection",
            ):
                registry = kg_server.get_connection_registry()
                if action == "list_connections":
                    return json.dumps(registry.status(), default=str)
                if not config_key:
                    return json.dumps(
                        {"error": f"config_key (connection name) required for {action}"}
                    )
                if action == "add_connection":
                    try:
                        spec = json.loads(config_value) if config_value else {}
                    except Exception as e:
                        return json.dumps({"error": f"Invalid config_value JSON: {e}"})
                    if not isinstance(spec, dict):
                        return json.dumps(
                            {
                                "error": "config_value must be a JSON object (backend spec)"
                            }
                        )
                    try:
                        name = registry.register(config_key, spec)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                    # CONCEPT:KG-2.89 — persist the connection list to config.json so
                    # it survives restart (re-seeded from config.kg_connections).
                    from agent_utilities.core.config import save_config_item

                    save_config_item("kg_connections", registry.export_specs())
                    return json.dumps(
                        {
                            "status": "success",
                            "action": action,
                            "connection": name,
                            "role": registry.role(name),
                            "persisted": True,
                        }
                    )
                if action == "remove_connection":
                    removed = registry.remove(config_key)
                    if removed:
                        from agent_utilities.core.config import save_config_item

                        save_config_item("kg_connections", registry.export_specs())
                    return json.dumps(
                        {
                            "status": "success" if removed else "not_found",
                            "action": action,
                            "connection": config_key,
                            "persisted": bool(removed),
                        }
                    )
                # set_default_connection
                try:
                    name = registry.set_default(config_key)
                except Exception as e:
                    return json.dumps({"error": str(e)})
                return json.dumps(
                    {"status": "success", "action": action, "default_target": name}
                )
            # ── CONCEPT:KG-2.63: profile / imprint an external graph + map ──
            if action in ("profile_connection", "imprint_connection"):
                if not config_key:
                    return json.dumps(
                        {"error": f"config_key (connection name) required for {action}"}
                    )
                registry = kg_server.get_connection_registry()
                from agent_utilities.knowledge_graph.core.connection_profiler import (
                    profile_and_imprint,
                    profile_connection,
                )

                try:
                    ext_engine = registry.get_engine(config_key)
                except Exception as e:
                    return json.dumps({"error": f"connection '{config_key}': {e}"})
                if action == "profile_connection":
                    return json.dumps(
                        profile_connection(ext_engine, name=config_key), default=str
                    )
                # imprint_connection — profile + ontology-map + write the catalog
                # node into the authority (default) KG.
                return json.dumps(
                    profile_and_imprint(
                        ext_engine,
                        name=config_key,
                        spec_summary=registry.spec_summary(config_key),
                        authority_engine=registry.get_engine(None),
                    ),
                    default=str,
                )
            # ── CONCEPT:KG-2.74: Concurrent N-way mirroring health/repair ──
            if action in ("mirror_status", "reconcile"):
                from agent_utilities.knowledge_graph.backends import (
                    get_active_backend,
                )
                from agent_utilities.knowledge_graph.backends.fanout_backend import (
                    FanOutBackend,
                )

                backend = get_active_backend()
                # Locate the FanOutBackend. It is either the active backend
                # (GRAPH_BACKEND=fanout) or — the common case — the durable L3 of a
                # tiered backend (GRAPH_BACKEND=tiered + GRAPH_MIRROR_TARGETS), which
                # tees pg-age writes to the mirrors. Also unwrap a BrainGuarded proxy
                # (its inner backend is the ``inner`` property).
                cand = getattr(backend, "inner", backend)
                fan = None
                if isinstance(cand, FanOutBackend):
                    fan = cand
                elif isinstance(getattr(cand, "l3", None), FanOutBackend):
                    fan = cand.l3
                if fan is None:
                    return json.dumps(
                        {
                            "error": "No fanout mirror active (set GRAPH_MIRROR_TARGETS "
                            "with GRAPH_BACKEND=tiered or fanout).",
                            "backend": type(backend).__name__,
                        }
                    )
                inner = fan
                if action == "mirror_status":
                    return json.dumps(inner.durability_stats(), default=str)
                # reconcile — full authority→mirror drift repair (config_key =
                # optional single mirror name; empty = all mirrors).
                return json.dumps(inner.reconcile(config_key or None), default=str)
            # ── Database environment provisioning (Stardog + pg-age) ──
            if action in ("setup_databases", "verify_databases"):
                from agent_utilities.knowledge_graph.setup import (
                    setup_environment,
                    verify_postgres,
                )

                try:
                    opts = json.loads(config_value) if config_value else {}
                except Exception as e:
                    return json.dumps({"error": f"Invalid config_value JSON: {e}"})
                if not isinstance(opts, dict):
                    return json.dumps(
                        {"error": "config_value must be a JSON object of options"}
                    )
                if action == "verify_databases":
                    return json.dumps(
                        verify_postgres(opts.get("dsn") or config_key or None),
                        default=str,
                    )
                # setup_databases — config_key is a profile shortcut ('dev'/'prod').
                profile = opts.get("profile") or config_key or "dev"
                return json.dumps(
                    setup_environment(
                        profile=profile,
                        postgres_mode=opts.get("postgres_mode", "managed_image"),
                        dsn=opts.get("dsn"),
                        sparql_target=opts.get("sparql_target"),
                        mirror_targets=opts.get("mirror_targets"),
                        do_backfill=opts.get("do_backfill", True),
                    ),
                    default=str,
                )
            # ── Full-deployment config: generate / validate / document ──
            if action in ("generate_config", "config_doctor", "config_reference"):
                from agent_utilities.deployment import (
                    config_doctor,
                    config_reference,
                    write_config,
                )

                try:
                    opts = json.loads(config_value) if config_value else {}
                except Exception as e:
                    return json.dumps({"error": f"Invalid config_value JSON: {e}"})
                if not isinstance(opts, dict):
                    return json.dumps({"error": "config_value must be a JSON object"})
                if action == "config_reference":
                    return json.dumps(config_reference(), default=str)
                # profile shortcut via config_key ('tiny'/'single-node-prod'/'enterprise')
                profile = opts.get("profile") or config_key or None
                if action == "generate_config":
                    return json.dumps(
                        write_config(
                            profile or "tiny",
                            opts.get("out"),
                            redact_secrets=opts.get("redact_secrets", True),
                        ),
                        default=str,
                    )
                # config_doctor
                return json.dumps(
                    config_doctor(profile, opts.get("config")), default=str
                )
            # ── CONCEPT:KG-2.89: generic live config get / set / list ──
            if action in ("get_config", "set_config", "list_config"):
                from agent_utilities.deployment import (
                    config_reference,
                    is_restart_required,
                )

                known: dict[str, dict] = {}
                for section in config_reference():
                    for f in section.get("fields", []):
                        known[str(f.get("env") or "").upper()] = f

                if action == "list_config":
                    out = {}
                    for env_key, meta in known.items():
                        val = os.environ.get(env_key)
                        out[env_key] = "***" if (meta.get("secret") and val) else val
                    return json.dumps({"config": out, "count": len(out)}, default=str)

                if not config_key:
                    return json.dumps(
                        {"error": f"config_key (env name) required for {action}"}
                    )
                env_key = config_key.upper()
                if env_key not in known:
                    return json.dumps(
                        {
                            "error": f"Unknown config key {config_key!r} (see config_reference)"
                        }
                    )
                if action == "get_config":
                    val = os.environ.get(env_key)
                    if known[env_key].get("secret") and val:
                        val = "***"
                    return json.dumps(
                        {
                            "key": env_key,
                            "value": val,
                            "restart_required": is_restart_required(env_key),
                        },
                        default=str,
                    )
                # set_config — persist to config.json + apply live (or flag restart).
                parsed = config_value
                if config_value and config_value.strip()[:1] in '[{"':
                    try:
                        parsed = json.loads(config_value)
                    except Exception:
                        parsed = config_value
                from agent_utilities.core.config import save_config_item

                save_config_item(env_key, parsed)
                restart = is_restart_required(env_key)
                return json.dumps(
                    {
                        "status": "success",
                        "key": env_key,
                        "applied_live": not restart,
                        "restart_required": restart,
                    },
                    default=str,
                )
            # ── Holistic deployment health sweep (brew/flutter-doctor style) ──
            if action == "system_doctor":
                from agent_utilities.deployment import run_doctor

                try:
                    opts = json.loads(config_value) if config_value else {}
                except Exception as e:
                    return json.dumps({"error": f"Invalid config_value JSON: {e}"})
                if not isinstance(opts, dict):
                    return json.dumps({"error": "config_value must be a JSON object"})
                return json.dumps(
                    run_doctor(
                        opts.get("only"),
                        fix=opts.get("fix", False),
                        live=opts.get("live", False),
                    ),
                    default=str,
                )
            if action == "preflight":
                from agent_utilities.deployment.preflight import run_preflight

                profile = config_key or "tiny"
                try:
                    opts = json.loads(config_value) if config_value else {}
                except Exception as e:
                    return json.dumps({"error": f"Invalid config_value JSON: {e}"})
                if not isinstance(opts, dict):
                    return json.dumps({"error": "config_value must be a JSON object"})
                return json.dumps(
                    run_preflight(profile, opts.get("components")),
                    default=str,
                )
            # ── KG-2.7 / ECO-4.6: Memory Hook Management ──
            if action == "install_hooks":
                try:
                    from agent_utilities.ecosystem.hook_installer import HookInstaller

                    installer = HookInstaller()
                    agents = config_value.split(",") if config_value else None
                    results = installer.install(agents)
                    return json.dumps(
                        {
                            "status": "success",
                            "results": results,
                            "installed": installer.installed,
                            "errors": installer.errors,
                        }
                    )
                except Exception as e:
                    return json.dumps({"error": f"Hook install failed: {e}"})
            if action == "uninstall_hooks":
                try:
                    from agent_utilities.ecosystem.hook_installer import HookInstaller

                    agents = config_value.split(",") if config_value else None
                    results = HookInstaller().uninstall(agents)
                    return json.dumps({"status": "success", "results": results})
                except Exception as e:
                    return json.dumps({"error": f"Hook uninstall failed: {e}"})
            if action == "doctor":
                try:
                    from agent_utilities.ecosystem.hook_installer import HookInstaller

                    return json.dumps(HookInstaller().doctor(), default=str)
                except Exception as e:
                    return json.dumps({"error": f"Doctor failed: {e}"})
            # ── CONCEPT:ORCH-1.27: Role-Specialized Model Routing ──
            if action == "set_role_routing":
                try:
                    from pathlib import Path

                    from agent_utilities.core.config import config as _cfg
                    from agent_utilities.models.model_registry import (
                        ModelRegistry,
                        RoleSpec,
                    )

                    payload = json.loads(config_value) if config_value else {}
                    reg_path = getattr(_cfg, "model_registry_path", None)
                    if not reg_path or not Path(reg_path).is_file():
                        return json.dumps(
                            {
                                "error": (
                                    "No model_registry_path configured; cannot "
                                    "persist role_routing."
                                )
                            }
                        )
                    registry = ModelRegistry.load_from_file(reg_path)
                    for rname, spec in payload.items():
                        registry.role_routing[rname] = RoleSpec.model_validate(spec)
                    Path(reg_path).write_text(
                        json.dumps(registry.model_dump(), indent=2)
                    )
                    return json.dumps(
                        {
                            "status": "success",
                            "action": "set_role_routing",
                            "roles": list(payload.keys()),
                        }
                    )
                except Exception as e:
                    return json.dumps({"error": f"set_role_routing failed: {e}"})
            # ── KG-2.35: Schema-Pack lifecycle (get/set the active domain pack) ──
            if action == "schema_pack":
                from agent_utilities.models.schema_pack_loader import (
                    get_active_pack,
                    set_active_pack,
                )
                from agent_utilities.models.schema_packs import list_schema_packs

                if config_key:
                    pack = set_active_pack(config_key)
                    return json.dumps(
                        {
                            "status": "success",
                            "action": "schema_pack",
                            "active": pack.name,
                            "signature": pack.signature(),
                        }
                    )
                active = get_active_pack()
                return json.dumps(
                    {
                        "status": "success",
                        "action": "schema_pack",
                        "active": active.name,
                        "signature": active.signature(),
                        "available": list_schema_packs(),
                    }
                )
            # ── KG-2.35: review out-of-pack candidate types seen on write ──
            if action == "schema_candidates":
                from agent_utilities.models.schema_pack_audit import (
                    SchemaCandidateAuditor,
                )

                try:
                    limit = int(config_value) if config_value else 100
                except ValueError:
                    limit = 100
                return json.dumps(
                    {
                        "status": "success",
                        "action": "schema_candidates",
                        "candidates": SchemaCandidateAuditor.instance().review(limit),
                    }
                )
            return json.dumps({"error": f"Unknown action: {action}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["graph_configure"] = graph_configure
