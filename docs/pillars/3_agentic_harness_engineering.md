# Pillar 3: Agentic Harness Engineering

## Overview

The **Agentic Harness Engineering** pillar encapsulates the continuous learning, evaluation, and evolutionary refinement of the agent ecosystem. It moves the system from static, pre-programmed behaviors into an adaptive entity that evaluates its own performance, distills lessons, and evolves new models and strategies autonomously.

## Why We Built This (Rationale)

Autonomous agents typically suffer from a "Groundhog Day" effect:
1. **Lack of Continual Learning**: An agent will make the same mistake across 100 sessions because it has no mechanism to convert an execution trace into generalized wisdom.
2. **Evaluation Blind Spots**: Traditional test suites are boolean. LLMs need graded, multi-strategy rubrics to gauge nuance and reasoning degradation.
3. **Catastrophic Forgetting**: As an agent acquires new skills, it often overwrites or corrupts previously established knowledge.

## How It Works (Implementation)

### Trace Distillation & The Experience Node (AHE-3.1 & AHE-3.5)
After a task completes or fails, the orchestrator initiates **Trace Distillation**. By analyzing the gap between failure and successful retry (Cross-Rollout Critique), the system extracts a `Condition -> Action` tactical insight and persists it as an `ExperienceNode`. This allows the agent to intrinsically "remember" how to avoid specific pitfalls in the future.

### EWC Consolidation & Temporal Drift (AHE-3.6)
To prevent catastrophic forgetting when modifying the Knowledge Graph, we implemented a lightweight **Elastic Weight Consolidation (EWC++)**. The system tracks concept drift across node embeddings via coefficient of variation. When drift exceeds a threshold, EWC applies a penalty to preserve the stability of legacy knowledge.

### Heavy Thinking & Horizon-Aware Curriculum (AHE-3.7 & AHE-3.9)
For complex tasks, **Heavy Thinking Orchestration** spawns multiple parallel thinker agents to explore trajectories before synthesizing a consensus. Simultaneously, the **Horizon-Aware Task Curriculum** uses macro-action composition and subgoal checkpoints to train agents on progressively longer execution horizons without losing focus.

### Agentic-iModels & Interpretability (AHE-3.8 & AHE-3.8)
The **Agent-Interpretable Model Evolver** autonomously evolves scikit-learn compatible models optimized for both predictive accuracy and LLM readability. **LLM-Graded Interpretability Tests** run 200-test protocols to verify the agent can correctly simulate the model's behavior natively.

## Benefits Introduced

- **Self-Healing Knowledge**: The ecosystem autonomously refines its understanding and prevents degradation over time.
- **Explainable Autonomy**: Through the iModels integration, the agents can natively interpret and defend the machine learning models they use.
- **Measurable Evolution**: The Continuous Evaluation Engine (EvalRunner) provides exact Jaccard metrics, cosine semantic tracking, and LLM-as-Judge scores to quantitatively prove the agent is getting smarter.

### Workflow Distillation & Bundle Distribution (ORCH-1.8 × AHE-3.2)
The **Workflow Distillation Hook** closes the evolution feedback loop by automatically promoting successful workflow execution patterns into reusable Workflow+TeamConfig pairs in the Knowledge Graph. When `synthesizer_step` completes a successful execution, an asynchronous background task fires the distillation hook. The hook tracks success counts per canonical workflow pattern (based on agent topology, not task content) and only promotes patterns that exceed the configurable `promotion_threshold` (default: 3 successes). Both the threshold and the `quality_score_minimum` are configurable from `config.json`.

Promoted patterns are persisted as paired `WorkflowDefinition` + `TeamConfigNode` KG entries that package workflows with their proven team compositions.

- **Source Code**: `agent_utilities/workflows/distillation_hook.py`
- **Hot Path**: `synthesizer_step → WorkflowDistillationHook.on_execution_complete()`

## Key Concepts Leveraged
- **AHE-3.1**: Continuous Evaluation Engine
- **AHE-3.5**: Continual Learning & Experience Nodes
- **AHE-3.6**: Continual Learning Engine
- **AHE-3.7**: Heavy Thinking Orchestration
- **AHE-3.9**: Horizon-Aware Task Curriculum
- **AHE-3.8**: Agent-Interpretable Model Evolver

## BrowseComp-Plus Extensions (arXiv:2508.06600)

### Adaptive Reasoning Budget (AHE-3.1)
Continuous 0.0–1.0 float scale for test-time compute scaling. Maps effort level to discrete retrieval parameters (search calls, depth, decomposition). Lightweight heuristic `estimate_query_complexity()` auto-classifies query difficulty.
- **Source**: `agent_utilities/harness/reasoning_effort.py`
- **Hot Path**: `EvaluationEngine.evaluate_and_decompose(reasoning_effort=0.7)`

### Disentangled Evaluation (AHE-3.1)
Separates retriever quality from LLM reasoning quality in evaluation. Returns three independent metric groups: `retriever_metrics` (precision, recall, nDCG, MRR), `reasoning_metrics` (step accuracy, goal achievement), and `citation_metrics` (precision, recall, F1). Enables pinpointing whether failures stem from bad retrieval or bad reasoning.
- **Source**: `agent_utilities/harness/evaluation_engine.py`
- **Hot Path**: `EvaluationEngine.evaluate_disentangled(retrieval_results=..., gold_doc_ids=...)`

### Citation Quality Tracking (AHE-3.1)
Measures citation quality in agent responses. Extracts KG node references (`[KG:id]`), concept IDs (`CONCEPT:X`), external URLs, file paths, and arXiv IDs. Computes precision/recall/F1 against retrieved and gold document sets. Identifies hallucinated citations and uncited evidence.
- **Source**: `agent_utilities/harness/citation_tracker.py`
- **Hot Path**: Lazy-loaded in `EvaluationEngine._lazy_init()` → `CitationTracker.evaluate_citations()`

---

## Evolved Self-Evolution Capabilities (Phase 10 — DSPy-Driven Self-Evolution)

### Physical Knowledge Distillation Engine (AHE-3.9) 🔬
The **Physical Knowledge Distillation Engine** represents a monumental architectural breakthrough in self-evolution. Rather than restricting optimized prompts and tool schemas to dynamic, volatile in-memory Knowledge Graph nodes, the distiller maps semantic components from the graph back into structural, human-readable file system changes. This allows the system to bridge the divide between runtime optimization and permanent code enhancement.
- **Source Code**: `agent_utilities/knowledge_graph/distillation/physical_distiller.py`
- **Hot Path**: `PhysicalDistillationEngine.distill_skill(...)` / `distill_mcp_tool(...)` / `distill_system_prompt(...)`

### Multi-Optimizer Prompt Selection Strategy (AHE-3.10) 🔬
The **Multi-Optimizer Prompt Selection Strategy** ensures that the optimization behavior scales appropriately based on the failure footprint. When optimizing prompt signatures via DSPy, the system dynamically inspects failure cluster scales. For highly localized failures, lightweight bootstrap optimizers (like `BootstrapFewShot`) are used. For widespread systemic regressions, the system employs high-parameter multi-generation optimization (like `MIPROv2`) to perform multi-stage hyperparameter tuning.
- **Source Code**: `agent_utilities/harness/evolve_agent.py`
- **Hot Path**: `EvolveAgent._dspy_optimize_cluster(failure_cluster=...)`

### GitOps Commit & Evolution Boundary Traceability (AHE-3.11) 🔬
Every evolutionary cycle is governed by strict, declarative **GitOps boundaries**. When changes are distilled to the physical file system, a structured, isolated git commit is generated programmatically. This commit is tagged with concept traceability IDs and the source failure cluster ID, linking runtime agent telemetry directly to code version control.
- **Source Code**: `agent_utilities/knowledge_graph/distillation/physical_distiller.py`
- **Hot Path**: `PhysicalDistillationEngine.commit_distilled_changes(...)`

### AHE-3.12 — LongMemEval-S Validation Harness

A FastAPI `/benchmark` surface (`server/routers/benchmark.py`, mounted in `build_agent_app`)
compatible with Quarq's HTTP benchmark runner (`quarqlabs/benchmarks`), used to prove the
memory-first synergy stack (ORCH-1.27 role routing + KG-2.11 bi-temporal + KG-2.12 memory-first
retrieval + KG-2.13 learner) meets or beats Quarq's 98.2% on LongMemEval-S. `POST /benchmark/session`
ingests haystack messages as **episodic** memory into a **frozen, versioned** `EvaluationCorpus`
(reproducible across agent versions — Quarq re-derives FAISS each run); `POST /benchmark/query` runs
the HyDE + self-correcting two-pass pipeline (corpus-scoped), synthesizes via the `generator` role,
and scores via the `judge` role with a deterministic fallback; `GET /benchmark/report/{run_id}`
returns the LongMemEval-style accuracy + per-category breakdown. `scripts/check_longmemeval.py` gates
CI on a frozen-subset floor (default 95%), sharing the exact scoring helpers so gate and live router
never diverge; the full 500-question run is nightly/on-demand. Extends AHE-3.

### AHE-3.1 — In-house training substrate

The harness can fine-tune its own open-weight models end-to-end. The deterministic
reward/data engine (`graph/training_signals.py` + data-science-mcp `training_data.py`)
builds SFT/DPO/GRPO corpora; the gradient trainers (data-science-mcp `trainers/`,
torch/PEFT) consume them; the Rust performance path (epistemic-graph
`datascience/training.rs`) mirrors the loss/optimizer kernels; `eval_hooks` bridge
checkpoints back into the AHE-3.1 reliability suite. Trained checkpoints go live via
the model-registry role seam with no hot-path edit. Full cross-repo design:
[In-House Training Substrate](../architecture/in_house_training_substrate.md).

### AHE-3.0 — Prioritized replay of decisive states

`harness/replay_buffer.PrioritizedReplayBuffer` (inverse-frequency priority,
seed-faithful sampling) is wired into `AgenticEvolutionEngine.run_evolution_cycle`:
each cycle pushes its outcome keyed by base id so rare/decisive bases resurface
preferentially via `sample_replay`. Pairs with MEMO merge-generalize (KG-2.1) for
sample-efficient, weight-free self-evolution (source b4-03). Extends AHE-3.0.
