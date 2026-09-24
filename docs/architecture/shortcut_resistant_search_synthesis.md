# Shortcut-Resistant Search-Task Synthesis

*Concepts: KG-2.70, AU-KG.retrieval.formulate-adversarially-refine, AU-KG.retrieval.question-formulation-adversarial-refinement (agent-utilities), AU-AHE.reward.search-task-corpus (reward spine),
search-task-corpus (data-science-mcp). Distills FORT-Searcher (arXiv:2606.12087).*

## Why

Deep-search agents are bottlenecked by **shortcut-resistant data**: verifiable
questions whose answer stays unavailable until enough evidence is acquired through
search. FORT-Searcher shows that structural complexity alone does not guarantee
*realized* search difficulty — a task can collapse through a cheaper identifying
route via four shortcut risks (single-clue selectivity, evidence co-coverage,
exposed constants, prior-knowledge binding). agent-utilities already owns every
organ FORT needs (a persistent evidence graph, multi-hop retrieval, a reward
spine, and an SFT/DPO/GRPO trainer) but had never connected them to *synthesize*
such tasks. This subsystem closes that gap — and goes past FORT by running over a
live, provenance-rich graph, closing a self-play loop, and minting DPO/GRPO
corpora FORT's SFT-only recipe leaves on the table.

## Pipeline

Starting from an answer entity, `build_evidence_subgraph` (KG-2.70,
`knowledge_graph/search_synthesis/evidence_subgraph.py`) does a bounded
checkout of the epistemic graph into an `EvidenceGraph` workspace
(clues, provenance, selectivity).

`formulate + refine`
(`knowledge_graph/search_synthesis/question_formulation.py`) then runs a
loop: `diagnose`
(`knowledge_graph/search_synthesis/shortcut_risks.py`) scores
`single_clue_selectivity`, `evidence_co_coverage`, `exposed_constants`,
and `prior_knowledge_binding`; `repair` prunes redundant clues,
generalizes required ones, or withholds names — looping until clear.
This produces a `SearchTask` (question, answer, evidence_path,
difficulty, risk_report).

Solver rollouts (`ExecutableRagProgram`/`research_autopilot`) turn that
into trajectories, scored as `realized_difficulty`
(`graph/training_signals.py`): `solving_cost` (Ω̂), `answer_hit_time`
(T̄_hit), and `prior_shortcut_rate` (p̂_prior) feed the search_heavy gate.
A task that is too easy is re-synthesized harder (more hops, stricter
thresholds); accepted tasks and trajectories flow into
`search_task_corpus` (`data_science_mcp/search_task_corpus.py`), which
maps them via `tasks_to_sft`/`trajectories_to_preference_pairs`/
`rollouts_to_grpo` into the unchanged `build_sft_examples`/
`build_preference_pairs`/`build_grpo_groups` builders and on to the
Sft/Dpo/Grpo trainers.
```

## Mapping FORT → this implementation

| FORT shortcut risk | Quantity collapsed | Detector (AU-KG.retrieval.formulate-adversarially-refine) |
|---|---|---|
| single-clue selectivity | `s(P)=|Ans(P)|` (eq 7) | `single_clue_selectivity` — flags clues whose `standalone_pool ≤ floor` |
| evidence co-coverage | `M_ev(P)` (eq 8) | `evidence_co_coverage` — clues sharing one `source_document_id` |
| exposed constants | `dep(P)` (eq 9) | `exposed_constants` — answer/intermediate names on the question surface |
| prior-knowledge binding | `U_π0` (eq 11) | `prior_knowledge_binding` — root popularity + optional closed-book probe |

Trajectory signatures (FORT §2.4, eqs 15/16/18) live on the AHE-3.1 reward spine
as `solving_cost`, `answer_hit_time`, `prior_shortcut_rate`, bundled by
`realized_difficulty(...)` with a `search_heavy` verdict for task gating.

## Entry points (Wire-First)

- **MCP** — `graph_search_synthesis` (`mcp/kg_server.py`): `action=synthesize`
  builds + refines a task around an answer entity; `action=diagnose` scores solver
  trajectories.
- **Autonomous loop** — `GoldenLoopController.run_one_cycle(synthesize_search=True)`
  (`knowledge_graph/research/golden_loop.py`) selects candidate answer entities,
  synthesizes clear tasks, persists `SearchTask` proposal nodes (propose-only) and
  drafts a JSONL corpus under `.specify/specs/search-tasks/`.
- **Training** — data-science-mcp `build_training_dataset` tool kinds
  `search_sft` / `search_dpo` / `search_grpo`; corpora feed the existing trainers
  unchanged. The gradient step is GPU-gated (torch/PEFT); the deterministic data
  path and `Trainer.plan()` handoff run on CPU.

## How we surpass FORT

1. **Live provenance** — co-coverage is an exact `source_document_id` test, not a
   page-scrape heuristic.
2. **Self-play** — the golden loop regenerates harder tasks as the solver improves
   (curriculum-scheduled via `HorizonCurriculum`), vs FORT's one-shot dataset.
3. **DPO + GRPO** — a shortcut trajectory is the perfect DPO `rejected`; rollouts
   are GRPO-rewarded by realized search difficulty. FORT trains SFT-only.

## Extension seams

- `build_evidence_subgraph(..., enrich=...)` — plug an LLM enricher for FORT's
  derived-fact constructors (Table 2) and exact-value fuzzing (Table 5); the
  deterministic default needs no model.
- `prior_knowledge_binding(..., probe=...)` / `synthesize(..., probe=...)` — pass a
  closed-book `InferenceBackend` probe to reject prior-bound tasks.
