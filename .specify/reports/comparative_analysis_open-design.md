# Comparative Analysis — open-design → agent-utilities

**Date**: 2026-06-05 · **Mode**: Lightweight (code-vs-code, cross-domain feature extraction)
**Source**: `nexu-io/open-design` @ `accbd3b` (`/home/apps/workspace/open-source-libraries/open-design`)
**Target (primary)**: `agent-utilities` (`/home/apps/workspace/agent-packages/agent-utilities`)
**Angle (user-selected)**: cross-domain pattern adaptation — *not* cloning the design domain
**Depth (user-selected)**: comprehensive sweep
**Ledger**: [`innovation_ledger_open-design.json`](innovation_ledger_open-design.json) (19 adapted rows · 6 epics · 6 new concepts)

---

## Executive summary

The two projects live in **different problem spaces**: open-design is a TypeScript web-app + local
daemon that turns 16 PATH-detected coding-agent CLIs into a **design-artifact engine**; agent-utilities
is a headless Python **self-evolving AI framework** (5 pillars, 86 concepts, DSTDD + 1:1:1
traceability). So the value is **not domain overlap** — it is open-design's *architecture and execution-loop
patterns*, which are directly transferable to a framework that orchestrates agents but currently lacks a
backend-agnostic execution substrate and a wired-in quality-gate loop.

**Where each project is stronger today**

| Dimension | open-design | agent-utilities |
|---|---|---|
| Backend-agnostic agent execution (any CLI / any provider) | **Strong** — declarative adapter registry + BYOK proxy | **Absent** — `core/execution/engine.py` is a stub; `model_factory` is pydantic-ai-only |
| Human-in-the-loop + pre-emit quality gates | **Strong** — discovery form, P0 checklist, 5-dim critique | **Partial** — harness exists (AHE-3.1) but no `MultiDimensionalCritique`, no wired pre-emit gate |
| Refreshable, provenance-tracked outputs | **Strong** — Live Artifacts (template+data+provenance) | **Latent** — bi-temporal/evidence-weighted/self-curating KG exists, but no artifact abstraction over it |
| Knowledge graph / semantic memory | minimal | **Strong** — KG-2.x is the spine |
| Self-evolution / continuous evaluation | absent | **Strong** — AU-AHE.optimization.telemetry-optimization evolution + variant pool |
| Multi-agent orchestration / HTN planning | absent (delegates to one CLI) | **Strong** — AU-ORCH.planning.orchestration-overview router/planner/parallel engine |
| Concept traceability / SDD discipline | good (specs/, CONTEXT.md) | **Strong** — DSTDD, concept registry, Wire-First |
| Engineering lifecycle ergonomics | **Strong** — one `tools-dev` CLI, namespaces | weaker — docker-compose + scattered scripts |

**The synergy thesis.** open-design's best patterns are *generic*; agent-utilities can make each one
**strictly superior by routing it through the epistemic KG and the evolution engine**:

- Live Artifacts that re-derive from a **bi-temporal, evidence-weighted, self-curating** KG
  (KG-2.11/2.18/2.19) — "failed refresh preserves prior" falls straight out of bi-temporal valid-time.
- Quality gates whose pass/fail signal **feeds the evolution engine** (AU-AHE.optimization.telemetry-optimization) instead of being prompt-only.
- A provider proxy whose every datum is **provenance-traced** in the graph to the producing CLI/model.

---

## Comparison matrix (transferable capability domains)

| Capability domain | Source maturity | Target today | Adapted target outcome | Epic |
|---|---|---|---|---|
| Multi-CLI execution substrate | A | F (stub) | Drive any CLI through one canonical interface | E1 |
| Provider-normalizing BYOK proxy + SSRF guard | A | F (absent) | One SSRF-safe streaming endpoint for any provider | E1 |
| Interactive mid-turn HITL loop | A | C (approve-only) | Keep stdin open; inject tool_result mid-turn | E2 |
| Sidecar isolation / process stamps | A | C (WASM sandbox only) | Per-run UDS-isolated sidecars | E2 |
| Pre-emit quality gates | A | C (harness, unwired) | Discovery→preflight→5-dim critique before emit | E3 |
| Refreshable provenance artifacts | A | D (latent in KG) | KG-backed Live Artifacts | E4 |
| Eval-scored skill discovery | B+ | C (compiler only) | Self-improving scenario picker | E5 |
| Dev lifecycle CLI + run-scoped tokens | A | D | One CLI + scoped capability tokens | E6 |

(Grades are *capability-transfer* readiness, not project quality — agent-utilities leads on KG,
orchestration and self-evolution, which open-design lacks entirely.)

---

## Top innovations extracted (verified against source code, not marketing)

All 19 rows are `verified` (read in `apps/daemon`, `packages/*`, `specs/`). Highlights:

1. **Declarative multi-CLI adapter registry** (`runtimes/registry.ts`, `defs/*.ts`, `detection.ts`) — CLI
   behavior is data (`RuntimeAgentDef`), auto-detected on PATH, with `streamFormat` dispatch and
   `fallbackModels` graceful degradation. → **ORCH-1.33**.
2. **BYOK provider-normalizing proxy** (`chat-routes.ts:668-1385`) with **DNS-resolved SSRF blocking**
   (`connectionTest.ts` `validateBaseUrlResolved`) — resolves the hostname and rejects private IPs,
   closing the public-DNS→private-IP vector; loopback carve-out keeps local LLMs working. → **ORCH-1.34** + OS-5.3.
3. **Mid-turn tool_result injection** (`server.ts:11900+`, `/api/runs/:id/tool-result`) — keeps stdin
   open and serializes a `tool_result` JSONL line back, making interactive tools work headless. → **AU-ORCH.execution.held-turn-registry-mid**.
4. **Layered pre-emit gate**: Turn-1 discovery form → P0/P1/P2 preflight checklist → **5-dimensional
   self-critique** (score 1–5, fix any <3, re-score) composed by a documented **dominant-layer prompt
   precedence** (`prompts/system.ts:8-29`). → **AU-AHE.harness.pre-emit-quality-gate**.
5. **Live Artifacts** (`specs/2026-04-29-live-artifacts/`) — template + data.json + provenance.json;
   bounded, injection-safe interpolation; refresh re-derives data and **preserves the prior render on
   failure**; per-generation provenance. → **KG-2.24** (refresh from the KG).
6. **Run-scoped tool token** — short-lived, runId/project/endpoint/expiry-bound capability token; daemon
   is the sole policy authority. → **OS-5.11**.
7. **Unified `tools-dev` lifecycle CLI** with `--namespace` isolation and `--json`. → E6.

Supporting rows: stream-format dispatch, 3-tier credential resolution, sidecar process stamps, anti-slop
antipattern registry, scenario skill taxonomy + eval-scored picker, SKILL.md critique-policy override,
SSRF-safe artifact contract, live progress SSE, ubiquitous-language glossary.

---

## What we deliberately do NOT take (avoid headless dead code — Wire-First)

| Item | Decision | Reason |
|---|---|---|
| Sandboxed-**iframe renderer** | Keep the SSRF-safe **contract** + postMessage schema; drop the renderer | Browser rendering is the consuming frontend's job; a renderer in a headless Python pkg is dead code |
| Pinned TodoWrite **UI card** | Ship the **SSE progress event**; drop the card | The card is a UI affordance with no headless home |
| **Windows named-pipe** sidecar IPC | POSIX UDS first; defer Windows parity | Deployment target is Linux daemons/containers; no current Windows consumer |
| The design domain itself (palettes, decks, media) | Not adopted | Different problem space; we take the *patterns*, not the *domain* |

---

## Roadmap → SDD

Six epics on the critical path **E1 → E2 → E3 → E5**, with **E4 branching off after E1** and **E6
parallel**. Lighthouse MVP = **E1 + E4**: *drive any agent CLI through a canonical proxy → write results
into the epistemic KG with full provenance → emit a Live Artifact that re-derives itself from that KG on
demand* — KG × orchestration × self-evolution fused with open-design's UX patterns.

Six new CONCEPT:IDs are proposed (ORCH-1.33/1.34/1.35, AU-AHE.harness.pre-emit-quality-gate, KG-2.24, OS-5.11), each gated by its
own `.specify/design/<id>/design.md` (KG-analysis similarity table proving <70% to the nearest existing
concept, C4, 5-pillar data flow, risk) before any `CONCEPT:` marker lands. Full per-epic SDD lives under
`.specify/design/` and `.specify/specs/`.

**Wiring audit** (run after each epic):
`python scripts/check_wiring.py --entry-points mcp/kg_server.py,server/app.py --max-hops 3` — every new
module must be ≤3 hops from an entry point.
