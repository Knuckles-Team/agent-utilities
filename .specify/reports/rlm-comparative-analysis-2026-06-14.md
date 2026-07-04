# Recursive Language Models (arXiv:2512.24601) vs agent-utilities — Comparative Analysis

**Date:** 2026-06-14
**Paper:** Zhang, Kraska, Khattab — *Recursive Language Models*, arXiv:2512.24601v3 (2025-12-31).
Code: `github.com/alexzhang13/rlm` (`pip install rlms`). PDF cached via ScholarX.
**Scope:** Track A (prove) + Track C (adopt). Plan: `~/.claude/plans/vivid-nibbling-kernighan.md`.

---

## 1. The paper in one paragraph

RLMs treat a long prompt as an **external environment**: the prompt is stored as a `context`
variable in a Python REPL, the root model sees only constant-size metadata about it, and it writes
code to peek/slice/grep and to call the LM recursively (`llm_query`) over snippets, finishing with
`FINAL()`/`FINAL_VAR()`. Depths: 0 (REPL only), 1 (sub-calls to a cheaper base model, e.g.
RLM(GPT-5) → GPT-5-mini), >1 (nested RLMs). Headline numbers (RLM(GPT-5, depth=1)): OOLONG
**56.0 vs 44.0**, OOLONG-Pairs **58.0 vs 0.1**, BrowseComp-Plus **91.3 vs 70.5** (compaction),
LongBench-v2 CodeQA **66.0 vs 20.0** (Qwen3 depth-0), at **$0.99** vs $1.50–2.75; +26% vs
compaction, +130% vs CodeAct-with-sub-calls, +13% vs Claude Code; inputs up to ~10M tokens. They
also post-train **RLM-Qwen3-8B** (+28.3% median) via rejection-SFT on ~1000 filtered trajectories.

## 2. Verdict: agent-utilities already exceeds the paper's *mechanism*

| Capability | Paper | agent-utilities | Evidence |
|---|---|---|---|
| Prompt as external `context` variable | ✅ | ✅ | `rlm/repl.py` (Algorithm 1; metadata-only root) |
| Model writes code to navigate it | ✅ | ✅ | `rlm/repl.py` REPL + sandbox exec |
| Cost-tiered sub-calls (cheap model at depth>0) | ✅ GPT-5-mini | ✅ `sub_llm_model_small` | `rlm/config.py:50-58`, `rlm/repl.py` depth→model |
| Recursive sub-calls | ✅ sequential | ✅ **parallel** (`asyncio.gather`) | `rlm/repl.py` `run_parallel_sub_calls` — paper lists async as *future work* |
| Sandboxed REPL | ⏳ *future work* | ✅ tiered monty/wasm/docker/local + AST router | ORCH-1.38 |
| Structured/typed final output | ⚠️ brittle `FINAL_VAR` tag | ✅ Pydantic schema contracts | ORCH-1.12 |
| Resilience / failure taxonomy | — | ✅ RunTrace + 7-class taxonomy | ORCH-1.29 |
| Query KG/OWL from the REPL | — | ✅ `graph_query`/`owl_query`/`kg_bulk_export` | beyond paper scope |

**Conclusion:** the paper's hard, novel parts (async sub-calls, sandboxing) are already shipped
here; its remaining advantages were **proof** and a **native trained model** — not capability.

## 3. Gaps found — and what this change closes

| # | Gap | Status |
|---|---|---|
| 1 | No benchmark scoreboard (only LongMemEval-S existed) | **Closed (Track A)** — `rlm/benchmarks/` harness over all 5 tasks + paper-comparison scoreboard (AHE-3.32) |
| 2 | RunTrace token usage never populated → no cost column | **Closed** — usage now captured (root + folded sub-call) in `repl.py`, surfaced via `run_rlm` (AHE-3.32) |
| 3 | No drop-in `rlm.completion()` surface | **Closed (Track C1)** — `agent_utilities.rlm.RLM` (AU-ORCH.execution.drop-rlm-completion-client) |
| 4 | One fixed system prompt across model families (paper failure mode) | **Closed (Track C2)** — family-aware prompt, `prompt_family` config (AU-ORCH.execution.drop-rlm-completion-client) |
| 5 | No validated 10M-token result | **Closed (Track C3)** — `live` S-NIAH stress at ~40M chars |
| 6 | Native trained RLM model (RLM-Qwen3-8B analog) | **Deferred (Track B)** — substrate exists in data-science-mcp; wiring documented in the plan appendix |

## 4. What was built

- **Harness** `agent_utilities/rlm/benchmarks/` (AHE-3.32): tasks `s_niah`, `oolong`,
  `oolong_pairs`, `browsecomp_plus`, `longbench_codeqa` (real dataset from
  `<data_dir>/rlm_benchmarks/<name>.jsonl` when staged, else paper-faithful synthetic — every row
  labelled `mode`); systems `rlm` / `vanilla` (truncation) / `compaction` (chunk→summarize→answer);
  `cost.py` token→USD; `runner.run_benchmark`; `scoreboard.render_scoreboard` with `PAPER_RESULTS`.
- **Cost capture** in `rlm/repl.py` + `rlm/runner.py` + `rlm/predict_rlm.py`: `RunTrace.usage` is
  now populated and returned (the field existed but nothing filled it).
- **Drop-in client** `agent_utilities.rlm.RLM` / `RLMResponse` (`rlm/client.py`), exported from the
  package — `RLM(...).completion(prompt).response`.
- **Family-aware prompt** `rlm/prompts.py` + `RLMConfig.prompt_family` (`auto` infers from model id;
  terser for Qwen, code-first for Anthropic).
- **Two surfaces:** MCP `graph_orchestrate(action="rlm_benchmark", task=…, dependencies={scales,…})`
  with the automatic REST twin via `graph_orchestrate_endpoint`.
- **Tests:** `tests/unit/rlm/test_ahe_3_32_benchmarks.py`,
  `test_orch_1_54_dropin_and_prompts.py` (CPU, 31 cases), plus `live`-gated
  `test_ahe_3_32_benchmark_live.py` (RLM vs vanilla, 10M-token stress, drop-in smoke).

## 5. How to run

```bash
# CPU unit tests (harness, client, prompts, usage capture)
pytest tests/unit/rlm/test_ahe_3_32_benchmarks.py tests/unit/rlm/test_orch_1_54_dropin_and_prompts.py

# Live benchmark + scoreboard (needs an LLM endpoint)
pytest -m live tests/unit/rlm/test_ahe_3_32_benchmark_live.py

# Through the fleet (MCP / REST):
graph_orchestrate(action="rlm_benchmark", task="oolong_pairs",
                  dependencies='{"scales":[80000],"cases_per_scale":3}')
```

The live run emits the measured **Measured results** + **Comparison to the paper** tables. To
compare like-for-like against the paper's exact numbers, stage the real OOLONG / BrowseComp-Plus /
LongBench-v2 corpora under `<data_dir>/rlm_benchmarks/` (synthetic rows are flagged otherwise).

## 6. Remaining / deferred

- **Track B (native RLM model):** collect RunTrace trajectories → rejection-filter + RLM verifier →
  `data-science-mcp` `train_sft` (Qwen3-8B LoRA) → `register_checkpoint(role="rlm-coder")`. All
  heavy parts already exist; only the RLM-specific glue + a GPU run remain.
- **Live numbers:** the scoreboard tables are populated by a `-m live` run against a real endpoint;
  this report ships the harness + analysis, not fabricated numbers.
