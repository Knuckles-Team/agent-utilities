# LongMemEval-S Validation Harness (CONCEPT:AU-AHE.evaluation.longmemeval-validation-harness)

## Overview

The LongMemEval-S Validation Harness is a FastAPI `/benchmark` surface that lets Quarq's HTTP
benchmark runner (`quarqlabs/benchmarks`) drive the agent-utilities memory-first stack
(ORCH-1.27 + KG-2.11/2.12/2.13) against LongMemEval-S and prove it meets or beats Quarq's 98.2%.
Extends **AHE-3** (Agentic Harness Engineering).

## How it works

- **Reproducible corpus.** `POST /benchmark/session` ingests haystack messages as **episodic**
  memory into a **frozen, versioned** `EvaluationCorpus` — reproducible across agent versions
  (Quarq re-derives FAISS each run; this does not).
- **Full pipeline per question.** `POST /benchmark/query` runs HyDE + self-correcting two-pass
  retrieval (corpus-scoped), synthesizes via the `generator` role, and scores via the `judge` role
  with a deterministic pure-Python fallback (`normalize_answer`, `judge_binary`).
- **Report + CI gate.** `GET /benchmark/report/{run_id}` returns accuracy + per-category breakdown
  (`aggregate_report`). `scripts/check_longmemeval.py` gates CI on a frozen-subset floor (default
  95%), sharing the exact scoring helpers so gate and live router never diverge; the full
  500-question run is nightly/on-demand.

## Key files / API

| Piece | Location |
|---|---|
| Router | `server/routers/benchmark.py` (`/benchmark/session|query|report|health`, `normalize_answer`, `judge_binary`, `aggregate_report`); mounted in `server/app.py` |
| CI gate | `scripts/check_longmemeval.py` |

## Wiring (≤3 hops)

`POST /benchmark/query` → router → `engine.search_hybrid(mode="hyde")` → `plan_and_retrieve` (≤3 hops).

## Research provenance

Quarq benchmark runner & eval pipeline — `quarqlabs/benchmarks` (HTTP) + agent-oss article eval pipeline.
