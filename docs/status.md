# Status — the Codex

> **Generated — do not edit by hand.** Produced by `scripts/build_status_page.py` from `docs/concepts.yaml` and `docs/concept_reservations.yaml`. See "How this page stays honest" at the bottom.

**Honesty first.** This page reflects the repository's actual current state — every number below is computed from `docs/concepts.yaml` and `docs/concept_reservations.yaml` at generation time, never hand-typed. If a claim elsewhere in this repo's docs disagrees with a number here, this page is generated more recently and is the one to trust.

## Concepts by pillar × status

**1230 LIVE** concepts (every entry in `docs/concepts.yaml` — the registry only carries concepts with shipped code) and **0 RESERVED** concept IDs (open, unexpired entries in `docs/concept_reservations.yaml`) across **9 pillars**.

| Pillar | LIVE ✅ | RESERVED | BUILDING 🔶 | ROADMAP 🗺 | RETIRED |
|:------|---:|---:|---:|---:|---:|
| **AU-AHE** — Agentic Harness Engineering | 120 | 0 | 0 | 0 | 0 |
| **AU-ECO** — Ecosystem & Peripherals | 141 | 0 | 0 | 0 | 0 |
| **AU-KG** — Epistemic Knowledge Graph | 524 | 0 | 0 | 0 | 0 |
| **AU-ORCH** — Graph Orchestration | 220 | 0 | 0 | 0 | 0 |
| **AU-OS** — Agent OS Infrastructure | 188 | 0 | 0 | 0 | 0 |
| **EG-AHE** — epistemic-graph: harness-facing concepts | 1 | 0 | 0 | 0 | 0 |
| **EG-KG** — epistemic-graph: knowledge-graph engine concepts | 34 | 0 | 0 | 0 | 0 |
| **EG-ORCH** — epistemic-graph: routing/orchestration concepts | 1 | 0 | 0 | 0 | 0 |
| **EG-OS** — epistemic-graph: deployment concepts | 1 | 0 | 0 | 0 | 0 |
| **Total** | 1230 | 0 | 0 | 0 | 0 |

> `BUILDING`/`ROADMAP`/`RETIRED` are always 0 here: `docs/concepts.yaml`'s registry model does not carry a per-concept partial-build state (its own generator strips reserved/retired IDs out entirely rather than flagging them) — a concept is either `RESERVED` or fully `LIVE`. The three statuses are still defined below because they are part of the one vocabulary this page and epistemic-graph's status page share.

## Status vocabulary

Defined once, here — every other table in this repo's docs (README capability tables, `docs/capabilities.md`) should link to this section instead of restating or omitting it. Existing emoji are the rendering of this vocabulary, not a separate scheme.

| Status | Emoji | Meaning |
|:------|:---:|:------|
| `RESERVED` |  | A concept ID is allocated in the reservations ledger; no code implements it yet. |
| `BUILDING` | 🔶 | Partially implemented; unsupported paths fail honestly rather than silently. |
| `LIVE` | ✅ | Implemented and present on `main`. |
| `ROADMAP` | 🗺 | Designed, no committed date. |
| `RETIRED` |  | Formerly live, intentionally removed. |

Lifecycle: `RESERVED` → `BUILDING` → `LIVE` → (optionally) `RETIRED`, with `ROADMAP` as the not-yet-reserved intent stage.

## Domain ownership

Structural ownership — which doc subtree and which CI/pre-commit gate is authoritative for each pillar. Not named humans.

| Pillar | Owning doc subtree | Primary gate |
|:------|:------|:------|
| **AU-AHE** | `docs/pillars/3_agentic_harness_engineering.md` | `scripts/check_concepts.py` + `scripts/check_eval_corpus.py` |
| **AU-ECO** | `docs/pillars/4_ecosystem_peripherals.md` | `scripts/check_concepts.py` + `scripts/check_skill_name_collision.py` |
| **AU-KG** | `docs/pillars/2_epistemic_knowledge_graph/` | `scripts/check_concepts.py` + `scripts/check_ontology.py` |
| **AU-ORCH** | `docs/pillars/1_graph_orchestration.md` | `scripts/check_concepts.py` + `scripts/check_coupling.py` |
| **AU-OS** | `docs/pillars/5_agent_os_infrastructure.md` | `scripts/check_concepts.py` + `scripts/check_genesis_manifest.py` |
| **EG-AHE** | `docs/architecture/ (epistemic-graph engine integration)` | `scripts/check_concepts.py` (marker registration) + epistemic-graph's `scripts/check_documentation_contract.py` |
| **EG-KG** | `docs/architecture/ (epistemic-graph engine integration)` | `scripts/check_concepts.py` (marker registration) + epistemic-graph's `scripts/check_documentation_contract.py` |
| **EG-ORCH** | `docs/architecture/ (epistemic-graph engine integration)` | `scripts/check_concepts.py` (marker registration) + epistemic-graph's `scripts/check_documentation_contract.py` |
| **EG-OS** | `docs/architecture/ (epistemic-graph engine integration)` | `scripts/check_concepts.py` (marker registration) + epistemic-graph's `scripts/check_documentation_contract.py` |

`EG-*` pillar concepts are markers found in *this* repo's code that tag a capability of the `epistemic-graph` engine this repo drives (cross-repo concept federation, see `docs/concept_coordination.md`); the engine's own implementation and its `EG-P0-1` generated ledger are owned by the `epistemic-graph` repo, whose `docs/status.md` is the authoritative status page for that side.

## How this page stays honest

This page is produced by `scripts/build_status_page.py` from `docs/concepts.yaml` and `docs/concept_reservations.yaml` — never hand-typed. Regenerate it with:

```bash
python scripts/build_status_page.py --write
```

`scripts/check_status_page.py` is wired into `.github/workflows/advisory.yml` (report-only, `continue-on-error: true`, matching this repo's existing approved-vs-enforced convention) and fails loudly — without blocking a release — the moment this file drifts from `docs/concepts.yaml` / `docs/concept_reservations.yaml`. Run it locally with:

```bash
python scripts/check_status_page.py
```
