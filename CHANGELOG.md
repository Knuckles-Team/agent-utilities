# Changelog

All notable changes to agent-utilities will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- **CONCEPT:KG-2.6: Deterministic Mark-and-Sweep Garbage Collection** — Implemented temporal heartbeat mechanism (`last_seen_timestamp`) to atomically purge stale `:Code` and `:Article` nodes from the Knowledge Graph during ingestion runs, maintaining 1:1 codebase parity natively without MD5 overhead.
- **CONCEPT:ORCH-1.4: Dynamic Subgraph Orchestrator** — Dynamically synthesizes subgraph transition logic from the Knowledge Graph on the fly without using predefined templates. Uses KG-2.41 Formal Graph Theory.
- **CONCEPT:AHE-3.4: Long-Running Background Context Spawner** — Polling module that checks background tasks via KG state and autonomously spawns specialized sub-agents based on context shifts and impact scores.
- **CONCEPT:ECO-4.0: Dynamic Tool Assignment Orchestration** — Matches tool ontology to agent tasks dynamically at runtime based on task context and KG embeddings.
- **CONCEPT:KG-2.6: Ontological Team Sharing** — Serializes dynamically created and successful subgraphs into OWL/Turtle ontology formats so they can be exported/imported as shareable team compositions.
- **Multi-Domain Architectural Pattern** — Transitioned `agent-utilities` to a Multi-Domain Expert System supporting modular expansion into `finance`, `medical`, `law`, and `science`. Domain integrations leverage Vectorized Topological Memory and the core Knowledge Graph, with heavy domain-specific dependencies optionally loaded via tags (e.g., `agent-utilities[finance]`).
- **Quantitative Finance Framework** — Production-grade, KG-native financial framework designed for global asset classes (Crypto, Equities, Forex, Derivatives). Features:
  - **Stationary Feature Engineering**: ADF stationarity testing across multi-asset OHLCV market data.
  - **Topological TradingLSTM**: Neural network architectures designed to process sequence data and networkx topological market regimes.
  - **Execution & Evaluation**: Walk-forward validation loops preventing lookahead bias, automated Kelly Criterion position sizing, and Kolmogorov-Smirnov regime shift detection.
- **FIBO & Quant Ontology Alignment** — Extended `ontology.ttl` with `DomainEntity`, `ScientificEntity`, `LegalEntity`, and specialized finance classes (`FinancialInstrument`, `TradingStrategy`, `StationaryFeature`, `LSTMNetwork`, `MarketRegime`, `ExecutionSignal`, `KellySizing`).
- **CONCEPT:ORCH-1.2: Ontological Fallback Chains** — Uses the Knowledge Graph to find fallback models dynamically rather than relying on static lists during rate limits or server errors.
- **CONCEPT:KG-2.9: Vectorized Context-Window Filtering** — Semantically prunes non-relevant subgraph context before swapping models on token overflow, ensuring only contextually distant nodes are dropped. New module: `agent_utilities/graph/context_filter.py`.
- **CONCEPT:AHE-3.5: KG-Native Agentic Task Detection** — Evaluates topological complexity via KG subgraphs to route dense API toolchains to complex models automatically.
- **CONCEPT:AHE-3.5: Topological Reasoning Detection** — Maps user queries to `MathematicalFoundationNode` or quantitative financial concepts to trigger reasoning models natively.
- **CONCEPT:OS-5.0: Topological Session Persistence** — Pins the model for multi-turn conversations directly to the SessionNode to avoid jarring mid-thread model bouncing.
- **CONCEPT:ECO-4.3**: Graph-Native Durable Execution Engine for fault-tolerant multi-leg algorithmic trading.
- **CONCEPT:ECO-4.3**: Secure Jupyter Sandbox with State Machine Invariant checks for code generation.
- **CONCEPT:AHE-3.5**: AgentSpecs Catalog Generator for shareable OWL-driven agent blueprints.
- **CONCEPT:KG-2.6**: Latent Topology RAG (Latte) for hierarchical routing.
- **CONCEPT:KG-2.5**: Single-Shot SIRA for retrieval sparsity optimization.
- **CONCEPT:KG-2.5**: VOI Budget Controller for scaling law enforcement.
- **CONCEPT:KG-2.3**: Cognitive Trap Defense for topological vulnerability mitigation.
- **CONCEPT:KG-2.3**: Experience Alignment for natively managed few-shot adaptation.
- Domain-driven knowledge_graph refactoring with zero-stub parity.
- **CONCEPT:ECO-4.3: Graph-Native Durable Execution** — Fault-tolerant, resumable state execution by persisting graph execution traces natively into the Knowledge Graph (LadybugDB) for high-assurance multi-leg trading. New module: `agent_utilities/orchestration/durable_execution.py`.
- **CONCEPT:ECO-4.3: Secure Jupyter Sandbox** — Isolated code generation sandbox with Vectorized Topology AST validation and State Machine Invariant checks (MCS Ch 6). Blocks unsafe OS commands. New modules: `agent_utilities/tools/jupyter_adapter.py`, `agent_utilities/tools/sandbox_executor.py`.
- **CONCEPT:AHE-3.5: OWL-Driven AgentSpecs** — Compiles dynamic agent topologies into exportable JSON AgentSpec catalogs, strongly typed by OWL ontologies for reproducibility. New module: `agent_utilities/core/agentspec_catalog.py`.
- **CONCEPT:KG-2.6: Research Intelligence Sub-Agent** — Isolated research context with citation graph traversal (Semantic Scholar API), doom-loop detection, and KG persistence. Adapted from ml-intern's research_tool.py sub-agent pattern. New module: `agent_utilities/knowledge_graph/orchestration/research_subagent.py`.
- **CONCEPT:KG-2.5: Spectral Cluster Navigator** — Tuning-free spectral clustering using normalized Laplacian eigengap heuristics for automatic k-selection. OWL-integrated via `skos:Concept` alignment with financial regime detection extension. Adapted from contextplus. New module: `agent_utilities/knowledge_graph/core/spectral_navigator.py`.
- **CONCEPT:KG-2.5: Symbol Blast Radius Analyzer** — Regex-based symbol usage tracking across Python codebases with definition-line exclusion and KG integration. Adapted from contextplus. New module: `agent_utilities/knowledge_graph/core/blast_radius.py`.
- **CONCEPT:KG-2.3: Auto-Similarity Memory Graph** — Auto-creates `SIMILAR_TO` edges between KG memory nodes when cosine similarity ≥ threshold (default 0.72). Exponential decay scoring with stale edge pruning and hub control. Adapted from contextplus. New module: `agent_utilities/knowledge_graph/memory/auto_similarity.py`.
- **CONCEPT:KG-2.3: Hybrid Search Index** — Weighted semantic+keyword search scoring (72%/28% default) with CamelCase/snake_case token splitting and phrase boost. Uses existing embedding infrastructure. Adapted from contextplus. New module: `agent_utilities/knowledge_graph/retrieval/hybrid_search_scorer.py`.
- **CONCEPT:OS-5.0: Enhanced Doom-Loop Detector** — Pattern-aware doom-loop detection with result-aware tool call signatures, repeating sequence detection, and corrective prompt generation. Adapted from ml-intern. New module: `agent_utilities/security/doom_loop_detector.py`.
- **CONCEPT:KG-2.3: RAG-KG Unification** — Collapses separate RAG vector index into KG-native retrieval using three acceleration layers: similarity-edge shortcuts (O(degree) vs O(N)), spectral cluster scoping, and hybrid semantic+keyword scoring. New module: `agent_utilities/knowledge_graph/retrieval/unified_rag_kg.py`.
- **CONCEPT:KG-2.6: Research Orchestration Integration** — Connects ResearchSubagent to ResearchPipelineRunner and UnifiedRAGKGRetriever for automated 7-phase daily research cycles. MCP-compatible for tool registration. New module: `agent_utilities/knowledge_graph/orchestration/research_orchestrator.py`.
- **CONCEPT:KG-2.6: Graph Distillation Migration** — Migrates standard RAG retrieval to pre-computed SimilarityEdgeNode shortcuts for O(degree) retrieval. Manages distillation index lifecycle: batch creation, incremental updates, stale edge pruning, and coverage health monitoring. New module: `agent_utilities/knowledge_graph/retrieval/graph_distillation.py`.
- 10 new `RegistryNodeType` entries, 8 new `RegistryEdgeType` entries, and 11 new Pydantic models for comparative analysis and RAG-KG unification.
- 55 new tests in `test_comparative_analysis.py` and `test_rag_kg_unification.py` covering all 9 new modules.
- Updated Concept Galaxy to 88 concepts (from 79).
- **CONCEPT:KG-2.6: Formal Graph Theory Primitives** — Mathematically rigorous graph operations from MCS (MIT 6.042J): DAG critical path analysis (O(V+E) makespan), k-connectivity certificates (Whitney's theorem), Euler tour serialization, chromatic scheduling (conflict-free parallelism), personalized PageRank (random walk with restart), adjacency matrix power path counting (A^k theorem), and curated MCS seed taxonomy (20 mathematical foundation nodes for KG preloading). New module: `agent_utilities/knowledge_graph/core/graph_theory_primitives.py`.
- **CONCEPT:KG-2.6: Embedding Alignment Diagnostics** — Multi-layer embedding quality analysis from MINER (arXiv:2605.06460v1): Centered Kernel Alignment (CKA) for structural space comparison, Alignment Ratio (AR) diagnostics, adaptive sparse fusion with neuron-level masking and cross-layer weighting, continuous embedding health monitoring via SVD effective dimensionality and CKA drift detection. New module: `agent_utilities/knowledge_graph/retrieval/embedding_diagnostics.py`.
- **CONCEPT:KG-2.6: Structural Causal Reasoning Engine** — Explicit causal chain modeling from MedCausalX (arXiv:2603.23085v1): Structural Causal Models (SCMs) with do-calculus interventions (graph mutilation), d-separation conditional independence testing (nx.is_d_separator), causal verification protocol for reasoning chains, counterfactual generation (proximity-sorted ancestor queries), spuriousness detection via conditional d-separation, and trajectory-level causal alignment scoring. New module: `agent_utilities/knowledge_graph/core/causal_reasoning.py`.
- **CONCEPT:KG-2.6: Latent Space Anti-Collapse Regularizer** — Formal anti-collapse guarantees from LeWorldModel (arXiv:2603.19312v2): SIGReg normality testing via random 1D projections with Jarque-Bera statistics, SVD-based collapse detection (effective dimensionality ratio), diversity-preserving EWC consolidation (extends AHE-3.6 with participation ratio constraints), embedding diversity metrics (isotropy, participation ratio, Shannon entropy), and predictive consistency scoring for agent trajectory validation. New module: `agent_utilities/knowledge_graph/memory/latent_space_regularizer.py`.
- **CONCEPT:KG-2.6: Probabilistic Knowledge Graph Reasoning** — Probabilistic reasoning from MCS Ch 17–21: Bayesian belief propagation with loopy BP over graph topology (odds-form updates with exponential distance decay), random walk exploration with surprise scoring (frequency × distance), Law of Total Probability aggregation (anti-Simpson's Paradox multi-source combination), Birthday Paradox collision detection (n ≈ 1.2√d threshold), and d-separation conditional independence testing. New module: `agent_utilities/knowledge_graph/core/probabilistic_reasoning.py`.
- **CONCEPT:KG-2.6: Optimal Execution Engine** — Mathematical optimal execution from Oxford HFT (Drissi, 2024): Almgren-Chriss discrete (hyperbolic scheduling via κ = arccosh(...)), Almgren-Chriss continuous (HJB-based smooth trajectories), Cartea-Jaimungal (running inventory penalty with Riccati ODE), Avellaneda-Stoikov market making (reservation price + optimal spread), cointegration pairs trading (OU process parameter estimation with AR(1) regression), and signal-adaptive execution (tanh-gated urgency adjustment). New module: `agent_utilities/knowledge_graph/core/optimal_execution.py`.
- **CONCEPT:KG-2.6: Formal Relations Engine** — Mathematical relation properties (Reflexive, Symmetric, Transitive closures) and Equivalence Classes from MCS Ch 4 for zero-shot entity resolution. New module: `agent_utilities/knowledge_graph/core/formal_relations.py`.
- **CONCEPT:KG-2.6: State Machine Invariant Engine** — Deterministic Finite Automata (DFA) abstractions and provable state invariants from MCS Ch 6 to prevent infinite loops. New module: `agent_utilities/knowledge_graph/core/state_machines.py`.
- **CONCEPT:KG-2.6: Markov Transition Forecasting** — Markov Chain transition matrices over agent interaction traces (Vectorized Topologies) from MCS Ch 21 to predict statistical failure nodes via stationary distribution. New module: `agent_utilities/knowledge_graph/core/markov_transitions.py`.
- 7 new `RegistryNodeType` entries (`MATH_FOUNDATION`, `CRITICAL_PATH_RESULT`, `CAUSAL_FACTOR`, `CAUSAL_MODEL`, `EXECUTION_PLAN`, `MARKET_MAKING_QUOTE`, `PAIRS_TRADE_SIGNAL`), 9 new `RegistryEdgeType` entries for KG-2.41–KG-2.46.
- 116 new tests across 6 test suites (`test_graph_theory_primitives.py`, `test_embedding_diagnostics.py`, `test_causal_reasoning.py`, `test_latent_space_regularizer.py`, `test_probabilistic_reasoning.py`, `test_optimal_execution.py`) plus 3 new suites for formal relations, state machines, and Markov transitions.
- New documentation: `docs/mathematical_foundations.md` — comprehensive 300+ line glossary of all mathematical, probabilistic, and financial engineering terms.
- Updated Concept Galaxy to 97 concepts (from 94).
- **CONCEPT:KG-2.6: Alpha Factor Library** — 20 battle-tested alpha factors (momentum, mean-reversion, volatility, volume, value) with IC/IR analysis for factor selection. Sourced from Qlib Alpha158. New module: `agent_utilities/domains/finance/alpha_factors.py`.
- **CONCEPT:KG-2.6: Risk Management Engine** — Risk-first guard pipeline with VaR (Historical/Parametric/Monte Carlo), stress testing (5 predefined scenarios), and pre-trade validation. Sourced from AutoHedge/OpenAlice. New module: `agent_utilities/domains/finance/risk_manager.py`.
- **CONCEPT:KG-2.6: Portfolio Optimization Suite** — Mean-Variance (Markowitz), Risk Parity, and Black-Litterman portfolio optimization with KG-backed allocation tracking. New module: `agent_utilities/domains/finance/portfolio_optimizer.py`.
- **CONCEPT:KG-2.6: Versioned Order System ("Trading-as-Git")** — Immutable order snapshots with version chains, atomic commit promotion, and pre-commit guard hooks. Sourced from OpenAlice. New module: `agent_utilities/domains/finance/versioned_orders.py`.
- **CONCEPT:KG-2.6: Market Data Abstraction Layer** — Protocol-based data provider system with auto-fallback chains, OHLCV normalization, synthetic GBM data generator, and KG data provenance. New module: `agent_utilities/domains/finance/market_data.py`.
- **CONCEPT:KG-2.6: x402 AI Payment Protocol** — First-class HTTP 402 challenge-response handler for autonomous AI agent payments with budget guards, daily/monthly spend limits, and manual approval thresholds. New module: `agent_utilities/domains/finance/payments.py`.
- **CONCEPT:KG-2.6: Profit Attribution Engine** — P&L decomposition into alpha/beta/residual via OLS regression, plus Sharpe, Sortino, Calmar, Information Ratio, win rate, and profit factor analytics. Sourced from Qlib. New module: `agent_utilities/domains/finance/profit_attribution.py`.
- **CONCEPT:KG-2.6: Universal Real-Time Streaming** — Domain-agnostic pub/sub message bus with wildcard topic matching, sequence numbering, message history, and WebSocket transport adapter. New module: `agent_utilities/domains/finance/streaming.py`.
- Extended `ontology.ttl` with 11 new OWL classes (`AlphaFactor`, `RiskLimit`, `VaREstimate`, `PortfolioAllocation`, `OrderVersion`, `OrderCommitRecord`, `MarketDataSource`, `PaymentProofEntity`, `PaymentBudget`, `ProfitAttribution`, `StreamChannel`) and 7 new object properties (`hasAlphaFactor`, `hasRiskLimit`, `hasAllocation`, `hasOrderVersion`, `paidVia`, `attributedBy`, `streamsTo`).
- Added `[finance-kronos]` optional extra for GPU-accelerated foundation model inference.
- 123 new tests across 8 test suites (`test_alpha_factors.py`, `test_risk_manager.py`, `test_portfolio_optimizer.py`, `test_versioned_orders.py`, `test_market_data.py`, `test_payments.py`, `test_profit_attribution.py`, `test_streaming.py`).
- Updated Concept Galaxy to 105 concepts (from 97).
- **CONCEPT:KG-2.6: Kronos Foundation Model Forecaster** — K-line candlestick tokenizer (1200-token vocabulary) and autoregressive transition matrix predictor for time series forecasting. New module: `agent_utilities/domains/finance/kronos_forecaster.py`.
- **CONCEPT:KG-2.6: Multi-Agent Trading Swarm** — 8 specialized agent roles (Director, Quant, Risk, Execution, Indicator, Pattern, Trend, Sentiment) with weighted consensus aggregation and risk manager veto power. Sourced from QuantAgent/AutoHedge. New module: `agent_utilities/domains/finance/trading_swarm.py`.
- **CONCEPT:KG-2.6: Visual Technical Analysis Engine** — Chart pattern detection (double top/bottom, breakout), support/resistance levels via local extrema clustering, and trend analysis via linear regression. Sourced from QuantAgent. New module: `agent_utilities/domains/finance/visual_ta.py`.
- **CONCEPT:KG-2.6: Real-Time Market Feeds** — Finance-specific StreamBus adapter with tick-to-bar aggregation, multi-symbol topic routing, and price alert triggers. Sourced from FinceptTerminal. New module: `agent_utilities/domains/finance/market_feeds.py`.
- **CONCEPT:KG-2.6: Multi-Platform Strategy Export** — Code generation engine producing Pine Script v6 (TradingView), MQL5 (MetaTrader 5), and TDX (通达信) from universal StrategySpec. Sourced from Vibe-Trading. New module: `agent_utilities/domains/finance/strategy_export.py`.
- **CONCEPT:KG-2.6: Research Autopilot** — Automated hypothesis → backtest → report loop with configurable pass criteria (Sharpe, drawdown, win rate, profit factor). Sourced from Vibe-Trading. New module: `agent_utilities/domains/finance/research_autopilot.py`.
- **CONCEPT:KG-2.6: Strategy Sharing System** — Strategy cards with metadata, configuration presets, registry search, fork capability, and community marketplace. Sourced from Vibe-Trading. New module: `agent_utilities/domains/finance/strategy_sharing.py`.
- Extended `ontology.ttl` with 7 new OWL classes (`KronosModel`, `TradingSwarmEntity`, `ChartPattern`, `MarketFeed`, `StrategyExport`, `ResearchHypothesis`, `StrategyCardEntity`) and 6 new object properties (`forecasts`, `swarmDecidedBy`, `detectedPattern`, `exportedAs`, `testedHypothesis`, `sharedAs`).
- 95 new tests across 7 test suites (`test_kronos_forecaster.py`, `test_trading_swarm.py`, `test_visual_ta.py`, `test_market_feeds.py`, `test_strategy_export.py`, `test_research_autopilot.py`, `test_strategy_sharing.py`).
- Updated Concept Galaxy to 112 concepts (from 105).


#### Added
- **CONCEPT:ORCH-1.4: Learned Agent Routing** — Jointly optimizes decomposition depth, worker choice, and inference budget from execution traces. Three routing policies: `RuleBasedPolicy` (keyword pattern matching to primitives), `TraceLearnedPolicy` (softmax scoring from historical `ExecutionTrace` with exponential moving average quality/success tracking), and `CostAwareRouter` (Pareto-optimal cost/accuracy wrapping any policy with budget filtering). Based on Uno-Orchestra research (arXiv:2605.05007v1, relevance score 31.2). New module: `agent_utilities/graph/routing_policy.py`.
- **CONCEPT:KG-2.2: Elastic Context Operators** — 5 atomic operators for elastic context orchestration: `SKIP` (exclude irrelevant messages), `COMPRESS` (replace messages with summary), `ROLLBACK` (revert to checkpoint), `SNIPPET` (extract focused evidence from verbose content), `DELETE` (permanent removal). Compress is expressively complete (any operation expressible as compression) while specialized operators reduce generation cost and hallucination risk. Extends `ContextCompactor` (KG-2.10) with `ElasticContextManager`, `ContextCheckpoint`, and `OperatorResult`. Based on LongSeeker's Context-ReAct paradigm (arXiv:2605.05191v1, relevance score 25.5). Extended module: `agent_utilities/knowledge_graph/context_compactor.py`.
- **CONCEPT:KG-2.2: Multi-Timescale Memory Dynamics** — Three-tier `TimescaleMemoryStore` with timescale-aware exponential decay: Working (5min half-life, promotes at 3+ accesses), Episodic (4hr half-life, promotes at 5+ accesses), Semantic (30-day half-life, permanent). Consolidation engine promotes high-activation memories up tiers. Content-hash deduplication, keyword-scored retrieval with activation weighting, and configurable decay floor pruning. Based on Continual Knowledge Updating (arXiv:2605.05097v1, relevance score 11.2). New module: `agent_utilities/knowledge_graph/timescale_memory.py`.
- **CONCEPT:KG-2.3: Versioned KG Mutations** — Git-like transactional mutation semantics for Knowledge Graph evolution. `KGTransaction` (batches add_node/update_node/delete_node/add_edge/delete_edge mutations), `KGCommit` (atomic application with rollback data and parent-commit chaining), `KGVersionEngine` (commit/rollback/diff with full history), `KGDiff` (structural diff between graph versions: nodes_added/removed/modified, edges_added/removed). Based on Evolving Idea Graphs (arXiv:2605.04922v1, relevance score 11.2). New module: `agent_utilities/knowledge_graph/kg_versioning.py`.
- **CONCEPT:ECO-4.0: Dynamic Skill Evolution** — On-the-fly skill creation and consolidation to avoid catastrophic forgetting. `SkillNeologismDetector` (identifies when existing skills don't cover a new capability via Jaccard keyword similarity below configurable threshold), `SkillFactory` (creates new `SkillNode` instances from detected gaps or execution traces with provenance tracking), `SkillMerger` (detects overlapping skills and consolidates them, combining keywords, patterns, and confidence scores). Based on Skill Neologisms (arXiv:2605.04970v1, relevance score 11.9). New module: `agent_utilities/knowledge_graph/skill_evolver.py`.
- **CONCEPT:OS-5.3: Jailbreak Robustness Hardening** — Extends Prompt Injection Scanner (OS-5.4) with 4-category jailbreak attack taxonomy from SoK research. Template-based (DAN/Developer Mode, AIM persona, UCAR unrestricted, Grandma exploit), optimization-based (GCG adversarial suffix detection, token smuggling via encoding), LLM-based (context boundary confusion with `[/INST]`/`[/SYS]` markers, multi-turn escalation), manual (role-play/hypothetical framing, false authority claims). 12 new `ThreatPattern` entries, `JailbreakCategory` enum. Based on SoK: Robustness against Jailbreak Attacks (arXiv:2605.05058v1, relevance score 16.2). Extended module: `agent_utilities/security/prompt_scanner.py`.
- 32 new unit tests in `test_research_enhancements.py` covering all 6 enhancements (routing policies, elastic operators, skill evolution, timescale memory, KG versioning, jailbreak patterns).
- Updated Concept Galaxy to 79 concepts (from 73).

### Added
- **CONCEPT:AHE-3.3: Agent-Interpretable Model Evolver** — Autoresearch loop that evolves scikit-learn-compatible model classes optimized for dual objectives: predictive accuracy and LLM readability via `__str__()`. Features Pareto frontier tracking with O(n²) dominance checking, reward decomposition integration (AHE-3.10), display strategy auto-selection, and KG-native evolutionary lineage via `EVOLVED_MODEL` transitive edges. Actual model fitting delegated to `data-science-mcp` via MCP tool calls. Based on Microsoft Research's Agentic-iModels (arXiv:2605.03808). New module: `agent_utilities/harness/imodel_evolver.py`.
- **CONCEPT:AHE-3.3: LLM-Graded Interpretability Tests** — 6-category, 200-test protocol measuring whether an LLM can simulate model behavior from `__str__()` alone. Categories: feature attribution (32), point simulation (43), sensitivity analysis (32), counterfactual (32), confidence calibration (32), data attribution (29). Includes numerical tolerance grading, reward hacking detection, and EvalRunner (AHE-3.12) integration. Results persist as `InterpretabilityTestNode` in the KG. Based on arXiv:2605.03808. New module: `agent_utilities/harness/interpretability_tests.py`.
- **CONCEPT:KG-2.8: Topological Graph Visualization** — Scalable WebGL-based Knowledge Graph visualization engine using Sigma.js and ForceAtlas2 physics for the `agent-webui`. Implements intelligent mass assignment and radial clustering for high-mass structural nodes to prevent graph spaghetti at 100K+ scale. Provides full interactive CRUD capabilities via React overlay UIs.
- **CONCEPT:KG-2.6: Model Display Optimization** — Display-predict decoupling engine optimizing model `__str__()` for agent consumption independently of `predict()` logic. 5 strategies: `linear_collapse`, `piecewise_table`, `symbolic_equation`, `coefficient_summary`, `adaptive` (SmartAdditive pattern with per-feature R² gating). Includes linearization with R² threshold, hinge basis collapse, and bounded complexity budgets (`DisplayComplexityBudget`). Results persist as `ModelDisplayNode` with `DISPLAY_OF` edges. Based on arXiv:2605.03808. New module: `agent_utilities/knowledge_graph/model_display.py`.
- **Agentic-iModels KG Models** — New Pydantic models: `IModelNode`, `InterpretabilityTestNode`, `ModelDisplayNode`, `IModelCandidate`, `DisplayComplexityBudget`, `ParetoPoint`. 3 new `RegistryNodeType` entries (`IMODEL`, `INTERPRETABILITY_TEST`, `MODEL_DISPLAY`) and 4 new `RegistryEdgeType` entries (`EVOLVED_MODEL`, `TESTED_INTERPRETABILITY`, `DISPLAY_OF`, `PARETO_DOMINATES`). New module: `agent_utilities/models/imodel.py`.
- **Agentic-iModels OWL Ontology** — Extended `ontology.ttl` with 4 OWL classes (`IModel`, `InterpretabilityTest`, `ModelDisplay`, `ParetoFrontierEntry`), transitive `evolvedModel` property for model lineage inference, and 3 datatype properties. Updated `owl_bridge.py` with 7 new promotable types.
- 70 new unit tests across `test_imodel_evolver.py` (25), `test_interpretability_tests.py` (24), and `test_model_display.py` (21).
- **CONCEPT:KG-2.5: Topological Analogy Engine** — Leverages exact subgraph isomorphism (networkx VF2) and vectorized embeddings (`EncPI`) to find analogous subgraphs across different domains, enabling structural pattern matching and cross-domain innovation extraction within the Knowledge Graph. New module: `agent_utilities/knowledge_graph/analogy_engine.py`.
- **CONCEPT:KG-2.2: OWL-Driven Semantic Subsumption** — Enables hierarchy-aware zero-shot ontology alignment. Automatically computes topological embedding cosine similarity against OWL class prototypes to infer and inject newly discovered concepts directly into the correct class lineage. New module: `agent_utilities/knowledge_graph/semantic_subsumption.py`.
- **Finance Schema Pack Enhancements** — Integrated abstractions from OpenAlice (Trading-as-Git, `VERSIONED_TRADE_COMMIT`, `EXECUTION_GUARD`, `UNIFIED_TRADING_ACCOUNT`) and Kronos (`TIME_SERIES_FORECAST`) into `FinanceSchemaPack`. Added corresponding `RegistryNodeType` and `RegistryEdgeType` entries in `knowledge_graph.py`.
- **CONCEPT:OS-5.1: Topological Vulnerability Scanner** — Enhances security by scanning execution graphs for structural vulnerabilities (e.g., untrusted data flows, dependency deadlocks) by matching them against known risk subgraphs using the Analogy Engine. New module: `agent_utilities/security/topological_scanner.py`.
- **CONCEPT:KG-2.6: Research Intelligence Pipeline** — Automated end-to-end research ingestion cycle: ScholarX Discovery → 9-domain Relevance Scoring → Tiered Ingestion (full KG + SQLite for relevant papers ≥ 3.0, abstract-only for marginal ≥ 1.0) → OWL Enrichment → Digest Generation. Supports arXiv papers via ScholarX, local files (PDF/HTML/Markdown), and web URLs. KG-backed watchlists via PolicyNodes. Integrated into MaintenanceCron for continuous discovery. New module: `agent_utilities/automation/research_pipeline.py`.
- **CONCEPT:KG-2.6: KG Source Resolver** — Bridges the Knowledge Graph (indexing/discovery layer) to the comparative-analysis skill (analysis layer) by materializing KG-stored documents to filesystem paths with metadata enrichment. Optional — gracefully returns empty when no KG is available. New module: `agent_utilities/knowledge_graph/source_resolver.py`.
- **Research Artifact Generator** — Creates structured LLM artifacts from KG-ingested papers: key contributions extraction, method detection, concept linkage discovery, application mapping to existing CONCEPT IDs, and periodic digest generation with emerging theme detection. New module: `agent_utilities/knowledge_graph/research_artifacts.py`.
- **Comparative Analysis KG Integration** — Updated `discover_projects.py` with `--kg-query` flag enabling KG-backed source resolution for the comparative-analysis skill. KG sources are materialized to `~/.scholarx/analysis/` as enriched markdown.
- **Conceptual Registry Parity** — Formalized 6 existing features into the 5-Pillar ecosystem architecture to ensure full zero-stub tracking:
  - **CONCEPT:ORCH-1.7: SDD Pipeline**
  - **CONCEPT:KG-2.1: Cross-Session Chat Recall**
  - **CONCEPT:KG-2.1: Project-Aware Context**
  - **CONCEPT:AHE-3.2: Agentic Engineering Patterns**
  - **CONCEPT:OS-5.4: Telemetry & Observability**
  - **CONCEPT:OS-5.1: Policy & Prompt Governance**
- Updated Concept Galaxy to 67 concepts (from 61).
- 45 new unit tests across `test_research_pipeline.py`, `test_source_resolver.py`, and `test_research_artifacts.py`.

- **CONCEPT:AHE-3.1: Multi-Strategy EvalRunner** — Multi-strategy evaluation runner with three scoring modes: exact match (Jaccard-normalized), semantic similarity (embedding cosine), and LLM-as-Judge (structured JSON prompt). Composite mode combines all three with configurable weights. Integrates with existing `EvaluationMonitor` for trend tracking and alerting. OWL-promoted via `eval_run` node type and `evaluated_by` edge. Ported from MATE's `eval_runner.py`.
- **CONCEPT:OS-5.4: Token Usage Tracker** — 4-bucket granular token analytics (prompt/response/thoughts/tool_use) with session aggregation, per-agent breakdown, and configurable budget alerting. Includes `record_from_llm_response()` adapter for pydantic-ai integration. OWL-promoted via `token_usage_record` node type. Ported from MATE's `token_usage_service.py` and `token_usage_callback.py`.
- **CONCEPT:OS-5.4: Audit Logger** — Append-only compliance audit trail with 30+ action constants, never-raise semantics, FIFO eviction, configurable retention cleanup, and query filtering by actor/action/resource/session. OWL-promoted via `audit_log` node type and `audited_by` edge. Ported from MATE's `audit_service.py`.
- **CONCEPT:OS-5.3: Guardrail Callback Engine** — Push-based input/output guardrail interception with block/redact/warn/log actions, regex and keyword pattern matching, and PolicyEngine adapter for unified evaluation. OWL-promoted via `guardrail_trigger` node type and `triggered_guardrail` edge. Ported from MATE's `guardrail_callback.py`.
- **CONCEPT:AHE-3.2: Agent Config Versioning** — Immutable agent configuration snapshots with sequential versioning, forward-only rollback (creates new version copying target's config), structured diffs between versions, and SUPERSEDES edge chains for KG-native version traversal. OWL-promoted via `agent_config_version` node type and `config_version_of` edge. Ported from MATE's `AgentConfigVersion` model.
- **OWL Bridge Extension** — Added `token_usage_record`, `audit_log`, `guardrail_trigger`, `agent_config_version`, `eval_run` to `PROMOTABLE_NODE_TYPES`; `audited_by`, `triggered_guardrail`, `config_version_of`, `evaluated_by` to `PROMOTABLE_EDGE_TYPES`.
- **MATE Comparative Analysis** — Functional gap closure between agent-utilities and the MATE framework (Control Room patterns, Eval Framework, Token Analytics, Audit Logging, Guardrail Callbacks, Config Versioning).
- Updated Concept Galaxy to 59 concepts (from 54).
- **CONCEPT:OS-5.1: Prompt Injection Scanner** — Pattern-based prompt injection and command injection scanner with 25+ threat vectors adapted from Goose's `scanner.rs`. Provides `PromptInjectionScanner` with text, tool-call, and conversation scanning modes. Integrates with `PolicyEngine` via `PromptInjectionPolicy` adapter. Security findings persist as `SecurityFindingNode` in the KG, enabling OWL transitive risk propagation via `propagatesRiskTo`.
- **CONCEPT:OS-5.3: Tool Repetition Guard** — Detects infinite tool call loops via consecutive identical call tracking and per-session budgets (configurable via `MAX_TOOL_REPEATS`, `MAX_TOOL_CALLS_PER_SESSION`). Adapted from Goose's `tool_monitor.rs`/`tool_inspection.rs`. Denied repetitions distill into `ExperienceNode` tactical rules (AHE-3.5) for cross-session loop avoidance. `RepetitionPolicy` adapter for `PolicyEngine`.
- **CONCEPT:KG-2.1: Token-Aware Context Compaction** — Intelligent context window management replacing naive truncation. Three strategies adapted from Goose's `context_mgmt/mod.rs`: `summarize_tools` (default), `drop_middle`, `progressive`. Compaction summaries persist as `EpisodeNode` snapshots for cross-session context recall via `MemoryRetriever`. Backward-compatible `compact_messages()` wrapper in `chat_persistence.py`.
- **CONCEPT:ORCH-1.3: Structured Retry Manager** — Shell-based success checks, on-failure hooks, and configurable timeouts for structured retry logic. Adapted from Goose's `retry.rs`. `RetryManager` evaluates `SuccessCheck` commands and manages attempt state. Retry outcomes feed into `TeamConfigNode.record_team_outcome()` reward signaling (AHE-3.3) for routing improvement.
- **Cross-Session Chat Recall** — `search_chat_history()` function in `chat_persistence.py` for keyword-based search across stored `Thread`/`Message` nodes. Adapted from Goose's `ChatHistorySearch` (Rust/SQLite). Uses KG Cypher backend with relevance scoring.
- **OWL Bridge Extension** — Added `security_finding` and `experience` to `PROMOTABLE_NODE_TYPES`; `detected_threat` and `triggered_retry` to `PROMOTABLE_EDGE_TYPES` for transitive risk inference.
- **CONCEPT:KG-2.6: Financial Trading Pipeline** — Added 5 new KG node types (`TradingSignalNode`, `OrderNode`, `PositionNode`, `PortfolioNode`, `StrategyNode`) and 6 edge types for modeling complete trading pipeline lifecycle. OWL-promoted with FIBO alignment for transitive provenance chains (e.g., Strategy → Signal → Order → Position → Portfolio).
- **CONCEPT:ECO-4.3: Market Data Connector Protocol** — Generic `DataConnectorProtocol` with auto-fallback chain and provenance tracking. Includes `DataConnectorRegistry` with prioritized failover, rate-limit awareness, and `DataFetchRecordNode` for immutable audit trails. OWL `fallsBackTo` declared as transitive for automated connector chain inference.
- **CONCEPT:ORCH-1.4: Swarm Preset Template Engine** — YAML-driven declarative multi-agent workflow engine with DAG-based dependency resolution, parallel dispatch identification, and variable substitution. Includes `SwarmPresetEngine` with topological sort, cycle detection, and layer-based execution ordering. KG-persisted via `SwarmPresetNode`, `SwarmRunNode`, `SwarmTaskRecordNode`.
- **CONCEPT:ORCH-1.3: Multi-Level Abstraction Layering** — Planners emit coarse-grained abstraction steps and delegate fine-grained execution to specialist nodes, reducing upfront planning token overhead.
- **Adaptive Model Routing & Reward-Driven Routing** — Included adaptive fast-path model selection (`gpt-4o-mini` fallback) for simple queries. Leverages ACO `pheromone_trails` to down-weight specialists with historically low success rates.
- **CONCEPT:KG-2.6: Risk Scoring Ontology Extension** — Domain-agnostic risk assessment framework with `RiskAssessmentNode`, `RiskFactorNode`, `RiskMitigationNode`. OWL `propagatesRiskTo` declared as transitive property enabling automated upstream risk chain inference via the OWL reasoner.
- **CONCEPT:AHE-3.6: Backtest Evaluation Harness** — Strategy evaluation harness with SQLite storage (separate from KG), walk-forward validation windows, benchmark comparison, and KG integration via `BacktestRunNode` and `BacktestMetricNode`. Connects to `StrategyNode` (KG-2.6) for full provenance chains.
- **CONCEPT:AHE-3.6: Horizon-Aware Task Curriculum** — Progressive horizon scheduling derived from Long-Horizon Training research (Kim et al., ICML 2026). Implements macro-action composition to reduce effective interaction steps, subgoal checkpoints for intermediate credit assignment, and configurable promotion policies (threshold/plateau/adaptive EMA). Integrates with `CognitiveScheduler` and `SwarmManager` for automatic horizon reduction during swarm execution.
- **CONCEPT:AHE-3.1: Decomposed Reward Signals** — Separates step-level reward (local constraint satisfaction) from trajectory-level reward (goal achievement) to prevent penalizing correct intermediate steps in failed trajectories. Implements `RewardDecomposer` engine with `R_total = R_trajectory + α·ΣR_step` formula, distillation insight extraction (correct-in-failures, incorrect-in-successes patterns), and integration with `ExperienceNode` pipeline.
- **Finance Schema Pack Expansion** — Expanded `FinanceSchemaPack` with all trading pipeline, risk scoring, data connector, and backtest types. Added retrieval boosts for `propagates_risk_to` (1.6×), `generated_signal` (1.5×), and `evaluated_strategy` (1.5×).
- **OWL Ontology Extension** — Added 18 new OWL classes and 17 new object/datatype properties to `ontology.ttl` including 3 transitive properties (`fallsBackTo`, `taskDependsOn`, `propagatesRiskTo`).
- **Backward Compatibility** — Added `SelfModelNode` alias for renamed `MemoryRetrieverNode` and `self_model.py` shim module for import compatibility.
- **CONCEPT:KG-2.5: Topological Mincut Partitioning** - Uses NetworkX Louvain detection to dynamically partition the Knowledge Graph into emergent topological clusters. Includes Label Propagation fallback for failed partitioning loops. Stable communities are persisted back to the Cypher backend via `maintenance_cron`, providing hierarchical waypoints for graph traversal.
- **CONCEPT:AHE-3.4: Temporal Drift & EWC Consolidation** - Tracks concept drift across node embeddings via coefficient of variation and cosine distance. Mitigates catastrophic forgetting by applying a lightweight Fisher-proxy Elastic Weight Consolidation (EWC++) when modifying established knowledge graph representations.
- **CONCEPT:AHE-3.5: Heavy Thinking Orchestration** - Two-stage parallel-then-deliberate reasoning pipeline adapted from HEAVYSKILL research. Features include: tiered hybrid complexity gating (heuristic → confidence → LLM fallback), configurable K parallel thinkers (default 4), a Serialized Memory Cache with thinking token pruning and trajectory shuffling, iterative convergence refinement, KG-native `TrajectoryNode` and `DeliberationNode` persistence, EncPI hyperedge mapping, and `WorkspaceAttention.deliberation_score()` for cross-trajectory consensus analysis.
- **CONCEPT:AHE-3.4: Distributed Agentic Evolution** - Transitioned the harness into an open-source hive mind where autonomous agents evolve globally. Features include:
  - **Evolutionary Vector (`genius-agent`)**: A background daemon (`--evolve` flag) that runs `SelfImprovementCycle` indefinitely, creating synthetic tasks and writing new skills to close gaps.
  - **Autonomous PR Generator**: A central `autonomous-contribution` skill in `universal-skills` that formats local KG breakthroughs (TeamConfigs, Skills) into Git branches and upstream PRs.
  - **Community Telemetry & Identity**: `TeamConfigNode` and `CallableResourceNode` metadata now include `origin` (`local` | `community` | `upstream`), deterministic hash identifiers, precise timestamps, and author fields to prevent duplication and establish primacy.
  - **Human-in-the-loop Guardrails**: All autonomously generated skills are explicitly flagged with `Author: Autonomous` in the SKILL.md frontmatter, requiring maintainer approval before centralized CI merge and global distribution.
- **CONCEPT:KG-2.4: Inductive Knowledge Hypergraphs** - Implemented Positional Interaction Encodings (`EncPI`) allowing the `HybridRetriever` to map n-ary hyperedges and perform zero-shot inductive reasoning across novel graph topologies by vectorizing relational intersections.
- **CONCEPT:KG-2.4: Offline/Async Knowledge Compression** — Added `TraceDistiller` to periodically run `ConsolidationEngine` background tasks, abstracting episode-level execution traces into generalized `PreferenceNode` and `PrincipleNode` knowledge points.
- **CONCEPT:AHE-3.4: Memory-Aware Test-Time Scaling** - Integrated batch-parallel trajectory generation into the orchestration planner. Rather than distilling memory from a single sequential failure, the system scales inference across parallel Siblings, extracts reasoning across all paths, and maps them to hyperedges for zero-shot generalization and graph-native topological feedback.
- **CONCEPT:AHE-3.4: Decomposed Context Retrieval** - Modified HybridRetriever to decompose complex queries into abstract technical sub-queries for targeted multi-vector retrieval.
- **CONCEPT:AHE-3.4: Cross-Rollout Critique** - Added contrastive self-correction distillation to the `verifier`. When a failure is followed by a successful retry, the system contrasts the states to distill an action-level tactical fix.
- **CONCEPT:AHE-3.4: Experience Node Architecture** - Introduced `ExperienceNode` schema to store specific `Condition -> Action` tactical insights in the Knowledge Graph for continual learning.
- **CONCEPT:AHE-3.4: HTN & LATS** - Integrated Hierarchical Task Networks (`Task.subtasks`) and `LATSPlanner` (Monte Carlo Tree Search fallback) into the orchestration planner for recursive goal decomposition.
- **CONCEPT:AHE-3.4: Tiered Virtual Context Blocks** - Introduced `VirtualContextBlockNode` for tiered working/episodic memory scaling.
- **CONCEPT:AHE-3.4: Multi-Agent BFT Consensus** - Integrated `execute_bft_consensus` into `A2AClient` for Byzantine Fault Tolerance across agent peers.
- **CONCEPT:ORCH-1.3: Execution Budgets & Cost Governors** - Implemented `ExecutionBudget` and integrated enforcement inside `dispatcher_step` to prevent infinite LLM loops and cap USD costs.
- **CONCEPT:KG-2.1: Quiet-STaR Rationale Persistence** - Created `QuietStarRationaleNode` to persist internal chain-of-thought traces with reward gradients for self-improvement.
- **Context-Aware Entity Representations (CONCEPT:KG-2.2)** — Injects topological graph structure directly into node vector embeddings. Features include:
  - Expands Deep GraphRAG principles by appending multi-hop contexts (up to 2 levels of parents/children) directly into the stringified node description before embedding generation.
  - Automatically fetches and appends OWL-inferred relationships (e.g., transitive subclasses) into the node's context space.
  - Enables "topology-aware" vector semantic searches for free, drastically improving multi-hop accuracy.
  - `ContextualRepresentationBuilder` dynamically controls the depth and breadth of injected structural logic.
  - `re_embed_node` pipeline immediately re-embeds nodes when the OWL bridge downfeeds new inferred facts.
- **Wide-Search Orchestration (CONCEPT:ORCH-1.1)** — Pydantic-native Graph node architecture for orchestrating large-scale extractions. Features include:
  - Automates batch decomposition within the SDD pipeline by instructing orchestrators (planners/routers) to partition large extractions into parallel `ExecutionStep`s (`is_parallel=True`).
  - Implements a hybrid validation strategy inside `join_step` using `WideSearchWorkboard`.
  - Fast-path: Native Pydantic schema validation for expected row counts and schema conformity.
  - Slow-path: `wide_search_joiner` LLM repair node to standardization data, fix schema mismatches, or signal re-plans on fast-path failures.
- **Trace Distillation Error Categorization (CONCEPT:AHE-3.1)** — Categorizes orchestrator (`ORCHESTRATOR_SKILL`) vs worker (`WORKER_SKILL`) failure modes through AHE skill distillation to enable self-evolving updates. RLM prompt updated to support proposing targeted ComponentEdits for orchestration logic vs worker tools.
- **A2A Config File (CONCEPT:ECO-4.1)** — File-based external A2A agent discovery via `a2a_config.json`. Mirrors `mcp_config.json` symmetry. Features include:
  - `secret://`, `env://`, `vault://` URI-based auth token resolution via `SecretsClient.resolve_ref()`.
  - Soft-fail startup: unreachable `.well-known/agent-card.json` endpoints log a warning and skip, never blocking server startup.
  - Periodic background re-fetch (`A2A_REFRESH_INTERVAL`, default 300s) to detect capability changes from remote agents.
  - Full KG ingestion: agent cards are registered as `CallableResource` nodes with embeddings, making them eligible for affinity-based swarm selection.
  - Cache invalidation: bulk ingestion triggers `invalidate_registry_cache()` (CONCEPT:ORCH-1.2) to keep the hot cache synchronized.
- **Unified Specialist Model (CONCEPT:ORCH-1.2)** — Collapses the artificial `prompt` / `mcp` agent type distinction into a single `specialist` type. Any specialist can now host any combination of MCP tools and/or agent skills. A2A agents remain their own type (`a2a`) due to the fundamentally different remote execution protocol. Legacy `agent_type` values are normalized at read time for full backward compatibility.
- New `A2A_CONFIG` and `A2A_REFRESH_INTERVAL` environment variables.
- New module: `agent_utilities/protocols/a2a_config.py` (config loader, auto-discovery, periodic refresh).
- Updated Concept Galaxy to 29 concepts (from 27).
- **Confidence-Gated Model Router (CONCEPT:ORCH-1.2)** — Adaptive model tier routing using runtime confidence signals from specialist consensus. When WorkspaceAttention scores indicate high agreement across specialist outputs, the model tier is automatically downgraded to reduce cost. Low confidence triggers escalation to heavier tiers. Based on the Squeeze Evolve multi-model orchestration framework (Maheswaran et al., 2026). Features include:
  - `ModelRegistry.pick_for_task_adaptive()` with `confidence_signal` and `routing_percentile` parameters.
  - Integration with `pick_specialist_model()` in the graph executor for automatic confidence-gated routing.
  - Composition with CONCEPT:OS-5.2 (Homeostatic Downgrade): budget pressure routes first, then confidence adjusts within the budget-allowed tier range.
  - `routing_confidence_log` on `GraphState` for per-specialist routing decision observability.
  - Soft SelfModel (CONCEPT:KG-2.1) integration: blends 70% runtime confidence + 30% historical proficiency when available; degrades gracefully when absent.
- **Evolutionary Aggregation Engine (CONCEPT:ORCH-1.2)** — Group-level diversity scoring and three-tier aggregation for specialist outputs. Extends WorkspaceAttention (CONCEPT:ORCH-1.2) with group fitness computation (confidence and diversity signals) and routes aggregation to the most cost-effective strategy:
  - `MAJORITY_VOTE`: Free — no LLM call when all specialists agree.
  - `LIGHT_MODEL`: Cheap model synthesis for moderate-confidence groups.
  - `HEAVY_MODEL`: Deep aggregation with reasoning-tier models for low-confidence, high-diversity groups.
  - `ConvergenceMonitor` in the CognitiveScheduler (CONCEPT:OS-5.2) detects diversity collapse and triggers early loop termination.
  - Configurable `population_size` (N=4) and `group_size` (K=2) for fully adjustable evolutionary loops.
- New module: `agent_utilities/graph/evolutionary_aggregation.py`.
- New documentation: `docs/squeeze-evolve-routing.md`.
- `routing_percentile` field on `GraphDeps` (env var: `ROUTING_PERCENTILE`, default 50.0).
- `ROUTING_DECISION` node type and `ROUTED_BY`/`AGGREGATED_FROM` edge types in Knowledge Graph.
- Updated Concept Galaxy to 40 concepts (from 38).
- 46 new unit tests across `test_confidence_routing.py` and `test_evolutionary_aggregation.py`.
- **Schema Packs (CONCEPT:KG-2.2)** — Domain-configurable KG profiles that scope the active node types, edge types, retrieval boosts, and OWL extensions to a specific domain. Inspired by gbrain#587 schema-pack proposal. Features include:
  - `SchemaPack` base model with dual operating modes: `ADDITIVE` (layer on top of core) and `EXCLUSIVE` (only pack + protected core types).
  - Protected core types (memory, episode, person, concept, etc.) that are always active regardless of mode.
  - Per-pack retrieval boost multipliers for domain-specific edge weighting.
  - Schema pack registry with `get_schema_pack()` factory and `register_schema_pack()` for runtime extensions.
  - Four pre-built packs: `core` (default), `research-state`, `biomedical`, `finance`.
  - `SchemaPackNode` for KG persistence of active pack configuration.
  - OWL Bridge integration: `PROMOTABLE_NODE_TYPES` and `PROMOTABLE_EDGE_TYPES` filtered through active pack.
- **Backlink-Density Retrieval Boost (CONCEPT:KG-2.2)** — Logarithmic in-degree-based scoring modifier in `HybridRetriever` that boosts hub entities with many inbound edges. Pack-configurable via `backlink_boost_strategy` (`global`, `context_only`, `disabled`) and `backlink_boost_factor` (default 0.1). Based on gbrain's observed +31% P@5 improvement.
- **KG Eval Capture (CONCEPT:KG-2.2)** — Lightweight regression testing harness for Knowledge Graph retrieval. Records query-result pairs to a separate SQLite database (`eval_log.db`) to prevent KG contamination. Features include:
  - `KGEvalCapture.capture()` — append-only recording of queries, results, scores, and latency.
  - `KGEvalCapture.replay()` — re-runs captured queries and reports Jaccard@k, top-1 stability, and latency delta.
  - `export()` / `purge()` for maintenance. Controlled by `KG_EVAL_CAPTURE` env var (disabled by default).
- New module: `agent_utilities/models/schema_pack.py`.
- New package: `agent_utilities/models/schema_packs/` with 4 pre-built domain profiles.
- New module: `agent_utilities/knowledge_graph/eval_capture.py`.
- `SCHEMA_PACK` node type and `USES_SCHEMA_PACK` edge type in Knowledge Graph.
- Updated Concept Galaxy to 43 concepts (from 40).
- 47 new unit tests across `test_schema_packs.py`, `test_backlink_boost.py`, and `test_eval_capture.py`.
- **Conductor Workflow Specification (CONCEPT:ORCH-1.1)** — Refined natural-language subtask instructions per specialist step. The router/planner now generates a focused `refined_subtask` field on each `ExecutionStep`, crafting targeted sub-goals instead of forwarding the raw user query. Inspired by the RL Conductor's per-step subtask specification (Nielsen et al., ICLR 2026).
- **Execution Visibility Graph (CONCEPT:ORCH-1.1)** — Per-step `access_list` controlling which prior step results are visible to each specialist. `_resolve_access_context()` helper filters `results_registry` before injection. Supports `["all"]`, specific node_ids, or empty for no context sharing.
- **Model Synergy Tracker (CONCEPT:AHE-3.3)** — Per-model-combination EMA success tracking in SelfModel (CONCEPT:KG-2.1). `model_synergies` dict on `SelfModelNode` tracks sorted pipe-delimited model combination keys. `SelfModel.get_best_synergies()` filters by available models for intelligent recombination.
- **Recursive Graph Orchestration (CONCEPT:ORCH-1.1)** — Nested `run_graph()` calls for self-referential test-time scaling. `recursive_orchestrator` specialist spawns inner graph executions with parent context. `RecursiveContext` dataclass and `MAX_RECURSION_DEPTH` env var (default 2) for depth control.
- New module: `agent_utilities/graph/recursive_executor.py`.
- New documentation: `docs/conductor-orchestration.md`.
- Updated Concept Galaxy to 47 concepts (from 43).
- ~60 new unit tests across `test_conductor_workflow.py`, `test_visibility_graph.py`, `test_model_synergy.py`, and `test_recursive_orchestration.py`.
- **Structural Fingerprint Engine (CONCEPT:KG-2.3)** — AST-based signature extraction and three-level change classification (NONE/COSMETIC/STRUCTURAL) for incremental KG updates. Generic capability that avoids costly full re-ingestion when only cosmetic changes occur. Includes `FingerprintManager` for workspace-level scanning and `detect_stale_files()` for git-based staleness detection.
- **Graph Integrity Validator (CONCEPT:KG-2.3)** — Non-blocking 4-tier graph validation inspired by Understand-Anything's `graph-reviewer`. Features include:
  - Tier 1 (Auto-fix): LLM type alias normalization (30+ node aliases, 30+ edge aliases), score clamping, missing name defaults.
  - Tier 2 (Integrity): Dangling edges, missing node types, untyped edges, duplicate IDs.
  - Tier 3 (Quality): Orphan nodes, self-referencing edges, generic descriptions, underscored hub detection.
  - Tier 4 (Fatal): Zero-node graphs, graph fragmentation below 50% threshold.
  - Integrated as 15th pipeline phase (`validate`) with `KGEvalCapture` (CONCEPT:KG-2.2) trend storage.
- **Entity-Claim Extraction / MAGMA Completion (CONCEPT:KG-2.2)** — Two-phase entity-claim extraction that fills the MAGMA epistemic view with real data. Features include:
  - Deterministic Phase 1: Regex-based extraction of citations, wikilinks, and assertion patterns.
  - `ClaimNode` model with confidence scoring, claim types, and epistemic metadata.
  - New edge types: `BUILDS_ON`, `EXEMPLIFIES`, `AUTHORED_BY` (joining existing `CONTRADICTS` and `CITES`).
  - `retrieve_epistemic_view()` fully implemented with real Cypher queries (replacing stub) and NetworkX fallback.
- New module: `agent_utilities/knowledge_graph/fingerprint.py`.
- New module: `agent_utilities/knowledge_graph/graph_validator.py`.
- New module: `agent_utilities/knowledge_graph/kb/entity_claim_extractor.py`.
- New pipeline phase: `validate` (15th phase, runs after `knowledge_base`).
- `CLAIM` node type and `BUILDS_ON`/`EXEMPLIFIES`/`AUTHORED_BY` edge types in Knowledge Graph.
- Updated Concept Galaxy to 54 concepts (from 47).
- 43 new unit tests across `test_graph_validator.py`, `test_entity_claim_extractor.py`, and `test_fingerprint.py`.

### Changed
- `MCPAgent.agent_type` default changed from `"prompt"` to `"specialist"`.
- `DiscoveredSpecialist.source` values unified to `"specialist"` or `"a2a"`.
- `discover_agents()` collapsed from 3 type branches to 2 (specialist + a2a).
- Executor `_execute_agent_package_logic()` simplified from 3 execution paths to 2 (remote A2A vs. unified specialist).
- `build_agent_app()` and `create_agent_server()` accept new `a2a_config` parameter.
- `initialize_graph_from_workspace()` accepts `a2a_config` parameter and syncs A2A agents alongside MCP agents during startup.

## [0.3.0] - 2026-05-02

### Added
- **First Principles Architecture** — Four new foundational concepts (CONCEPT:ORCH-1.2 through CONCEPT:ECO-4.1) that rewire the routing, dispatch, and feedback layers from first principles.
- **Registry Hot Cache (CONCEPT:ORCH-1.2)** — Session-scoped `_RegistryCache` singleton providing O(1) specialist lookups. Replaces full registry scans on every routing call, reducing prompt bloat by filtering to only the top-7 relevant specialists per query.
- **Event-Driven Cache Invalidation** — Cache auto-invalidates on MCP reload (`/mcp/reload`), pipeline completion, Self-Model session updates, and TeamConfig promotions. No stale-cache risk.
- **TeamConfig Promotion (CONCEPT:AHE-3.3)** — `promote_coalition_to_template()` persists successful specialist coalitions as reusable `TeamConfigNode` templates in the Knowledge Graph. `find_matching_team_config()` enables 3-stage hybrid routing: TeamConfig → Self-Model bias → LLM planning.
- **TeamConfig Reward Tracking** — `record_team_outcome()` records success/failure outcomes against team templates, enabling reward-weighted team selection over time.
- **RLM + TeamConfig Synergy** — When a `TeamConfig` is selected and input exceeds size thresholds, RLM capability is auto-attached to specialists via `capability_overrides`.
- **AgentCapability Type System (CONCEPT:ORCH-1.2)** — `AgentCapabilityNode` formalized as a first-class KG node with `auto_activate`, `trigger_conditions`, and `handler_module` fields. Capabilities are auto-activated in the executor based on input constraints (e.g., RLM for large payloads, critic for code).
- **PlannerGraphSkill (CONCEPT:ECO-4.1)** — A2A-native routing entry point via `PlannerGraphSkill` registered in `server/app.py`. When a `graph_bundle` is available, A2A requests bypass LLM orchestration and route directly through the graph planner.
- **Self-Model Feedback Loop** — Post-execution verification in `synthesizer_step` now feeds outcomes back to `SelfModel.update_after_session()` and `record_team_outcome()`, enabling recursive learning.
- **WorkspaceAttention Scoring** — `WorkspaceAttention` (GWT) scores are computed and logged per-specialist during execution for data-driven specialist prioritization.
- **Process Lifecycle Management** — `atexit` and `SIGTERM`/`SIGINT` handlers in `server/__init__.py` ensure all child processes (MCP servers, TUI, background threads) are gracefully killed on server exit. Uses child-only `pgrep` pattern instead of `killpg` to avoid self-termination.
- **USES_PROMPT Edges** — `link_prompt_to_agent()` creates `USES_PROMPT` edges between specialist nodes and their JSON prompt templates for full traceability.
- 33 new unit tests across `test_config_helpers.py`, `test_team_config.py`, and `test_capability_nodes.py`.
- 3 new documentation files: `first-principles.md`, `registry-cache.md`, `process-lifecycle.md`.

### Changed
- Router now performs 3-stage hybrid routing: (1) TeamConfig match → (2) Self-Model proficiency bias → (3) LLM planning fallback.
- Specialist filtering reduced from O(N) full registry scan to O(7) via `get_relevant_specialists()` using query-keyed caching.
- Verification synthesizer now feeds execution outcomes back to both Self-Model and TeamConfig for continuous improvement.
- `RegistryNodeType` enum extended with `TEAM_CONFIG` and `AGENT_CAPABILITY` types.
- `RegistryEdgeType` enum extended with `HAS_CAPABILITY`, `REUSED_TEAM`, and `USES_PROMPT` types.
- Server process cleanup uses targeted `pgrep -P` child enumeration instead of `os.killpg()` to avoid killing the process group (which would terminate test runners).

### Fixed
- mypy `call-arg` error: `SelfModel.update_after_session()` now correctly receives `GraphState` instead of kwargs.
- mypy `assignment` error: `log_file_path` now guards against `None` before assigning to env dict.
- ruff `F401`: Removed unused `contextlib.suppress` import from `server/__init__.py`.
- Bandit `B110` suppressions added for defensive `try/except pass` patterns in cleanup handlers.

## [0.2.41] - 2026-04-29

### Added
- Direct graph execution fast-path in AG-UI endpoint — bypasses LLM tool-call overhead
- `AGUIGraphEmitter` module for translating graph events to AG-UI wire format (0:/2:/8:/9: prefixes)
- `run_graph_iter()` and `execute_graph_iter()` — step-by-step graph execution using `graph.iter()` beta API
- Per-step state snapshots and elicitation hooks in the iter-based execution path
- `GRAPH_DIRECT_EXECUTION` env var (default: `true`) to toggle direct dispatch
- `SecretsClient` with pluggable backends: InMemory (Fernet-encrypted), SQLite (persistent + encrypted), HashiCorp Vault (enterprise)
- URI-style secret references: `vault://`, `env://`, `sqlite://` schemes
- `SECRETS_BACKEND`, `SECRETS_SQLITE_PATH`, `SECRETS_VAULT_URL` configuration
- `secrets_client` field on `GraphDeps` for graph execution credential resolution
- `docs/secrets-auth.md` comprehensive documentation (CONCEPT:OS-5.1)
- Concept marker backfill: CONCEPT:OS-5.0, CONCEPT:ORCH-1.1, CONCEPT:OS-5.2, CONCEPT:AHE-3.0, CONCEPT:ECO-4.0, CONCEPT:OS-5.1 across tests and source
- `auth.py` JWT Bearer token validation using `authlib` + JWKS caching (CONCEPT:OS-5.1)
- Combined auth dependency: accepts API key OR JWT Bearer token (gradual migration)
- `AUTH_JWT_JWKS_URI`, `AUTH_JWT_ISSUER`, `AUTH_JWT_AUDIENCE` configuration
- `ALLOWED_ORIGINS` and `ALLOWED_HOSTS` for configurable CORS/host restriction
- MCP subprocess token forwarding via `AGENT_USER_TOKEN` env injection
- `auth = ["authlib>=1.4.0"]` optional extra in `pyproject.toml`
- 69 new unit tests (30 secrets, 20 auth, 19 emitter/iter)

### Changed
- ACP adapter refactored to use pydantic-acp's `agent_factory` callback for per-session agent creation
- Removed `REQUESTED_MODEL_ID_CTX` workaround from ACP's `run_graph_flow` closure
- Unified execution layer (`graph/unified.py`) now exports `execute_graph_iter` as a first-class entry point
- `cryptography>=44.0.0` added as core dependency for Fernet encryption
- `vault` optional extra added to `pyproject.toml` for `hvac>=2.3.0`
- CORS middleware now reads `ALLOWED_ORIGINS` instead of hardcoded `["*"]`
- TrustedHostMiddleware now reads `ALLOWED_HOSTS` instead of hardcoded `["*"]`

## [0.2.40] - 2026-04-28

### Added
- LLM Council integration with 7 role-based advisor prompts
- Hybrid OWL Reasoning sidecar with HermiT/Stardog inference
- Standard ontology schemas (BFO, Schema.org, PROV-O, Dublin Core, SKOS)
- Concept traceability markers (`@pytest.mark.concept`) for doc-test alignment
- `.env.example` template for developer onboarding
- Project Structure section in AGENTS.md

### Changed
- Restructured test suite into domain-driven subdirectories (core, graph, integration)
- Deprecated `MEMORY.md` in favor of Knowledge Graph native storage
- Bumped pre-commit hooks: ruff 0.15.12, mypy 1.20.2, bandit 1.9.4

### Fixed
- Git merge conflict artifacts in atlassian-agent, langfuse-agent, repository-manager
- Duplicate import errors in protocol adapters
- TOML configuration errors in .bumpversion.cfg
- Broken file references in AGENTS.md

## [0.2.39] - 2026-04-26

### Added
- AG-UI protocol adapter for web and terminal frontends
- Human-in-the-loop tool approval with `ApprovalManager`
- MCP elicitation callback support via `global_elicitation_callback()`

### Changed
- Unified protocol layer: ACP, A2A, MCP, AG-UI all served from single FastAPI server
- Migrated flat-file state (MEMORY.md, USER.md, HEARTBEAT.md) to Knowledge Graph

## [0.2.38] - 2026-04-24

### Added
- 14-phase Unified Intelligence Pipeline
- LadybugDB as default embedded graph backend
- Knowledge Base layer with LLM-maintained wiki
- MAGMA-inspired orthogonal reasoning views (Semantic, Temporal, Causal, Entity)

### Changed
- Replaced Neo4j-only backend with pluggable graph abstraction (LadybugDB, FalkorDB, Neo4j)
