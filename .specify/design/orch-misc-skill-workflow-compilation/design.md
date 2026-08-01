# Design Document: SKILL.md stays human prose; the executable GraphPlan is compiled from it deterministically, not authored separately or via an LLM

CONCEPT:AU-ORCH.execution.skill-workflow-compilation

> `agent_utilities/workflows/skill_compiler.py` — `SkillCompiler.compile` /
> `compile_from_text`.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-ORCH.execution.workflow-lifecycle-management` | `WorkflowDefinition`/`WorkflowRunner`, the executable target this compiler produces a `GraphPlan` toward — the consumer, not this decision | 0.35 | ORCH |

### Extension Analysis

- **Primary Extension Point**: `SkillCompiler.compile_from_text`'s two regex
  passes (`### Step N:` headers, then numbered-list fallback) and its
  `[skill: ...]` / `[depends_on: ...]` bracket-annotation grammar.
- **Extension Strategy**: augment — a new annotation is a new bracket tag
  parsed the same way; the compiler does not need restructuring to support
  more explicit hints.
- **New Concept Required?**: No.

## Decision — parse SKILL.md prose deterministically with regex + explicit bracket-annotation overrides; no LLM in the compile step, no separate workflow-authoring format

`agent_utilities/workflows/skill_compiler.py:1-8, 34-165`

`SkillCompiler.compile` reads a skill directory's plain-English `SKILL.md`
body and turns it into an executable `GraphPlan` of `ExecutionStep`s via
two deterministic regex passes: first `### Step N: <title>` headers, falling
back to numbered `1. **Title**` lists when no step headers match; and if
neither matches, the entire body becomes a single step
(`refined_subtask=markdown.strip()[:1000]`) rather than failing to compile
at all.

Per-step identity and dependency are extracted with two escape hatches that
are explicitly authoritative over the inferred defaults:

- `[skill: <id>]` in the step title is "the authoritative id; only fall back
  to inferring one from the title text when absent" (`skill_compiler.py:145-147`)
  — inference otherwise splits the title on `:` or slugifies it.
- `[depends_on: a, b]` is parsed out of the title or body before the
  colon-split heuristic runs; the code comment explains why order matters
  here — "This MUST be stripped before the depends_on parsing and the
  colon-split heuristic below: the colon inside `[skill: ...]` otherwise gets
  mistaken for the `Step N: <title>` separator, corrupting the parsed id"
  (`skill_compiler.py:85-90`). Absent an explicit `depends_on`, a step
  defaults to sequential dependency on the immediately preceding parsed step.

**The rejected alternatives, both implicit in what the module's docstring
chose not to do:**

1. **Require skills to be authored directly as structured `GraphPlan`/YAML,
   not prose.** This was rejected in favor of keeping `SKILL.md` prose-first
   and human-readable — a skill author writes normal step-by-step
   instructions, the same document a human or another agent reads to
   understand the skill, and the workflow structure is *derived* from it
   rather than a second artifact to keep in sync. The `references/team.yaml`
   escape hatch exists for the one thing prose genuinely can't express well
   (`TeamConfigBlueprint`), not for step sequencing.
2. **Use an LLM to interpret the prose into steps at compile time.** Rejected
   in favor of a fixed regex grammar: compilation is then deterministic,
   free, and instantaneous — the same `SKILL.md` always compiles to the same
   `GraphPlan` — and the bracket-annotation grammar (`[skill: ...]`,
   `[depends_on: ...]`) gives an author a precise, mechanical way to
   disambiguate exactly the cases a regex parse would otherwise get wrong,
   without paying an LLM call (or its nondeterminism) on every compile.

## Data Flow

1. **ORCH**: `SkillCompiler.compile` is called wherever a `SKILL.md`-defined
   skill needs to run as a `GraphPlan` inside the orchestration engine;
   `skill_reference` (`knowledge_graph/ingestion/skill_workflow_ingest.py`)
   is the ingestion-side counterpart that records the skill in the KG.
2. **KG**: none directly at the compile step itself — the compiled plan's
   steps become executable graph nodes downstream.
3. **AHE**: none directly.
4. **ECO**: none directly.
5. **OS**: none directly.

## Risk Assessment

- **Blast Radius**: `agent_utilities/workflows/skill_compiler.py`.
- **Backward Compatible**: Yes — a `SKILL.md` with no bracket annotations
  still compiles via the inference fallbacks; annotations are additive.
- **Breaking Changes**: None.
- **Known weak point**: the regex grammar is necessarily heuristic — a
  `SKILL.md` written in a step-numbering style the two patterns don't
  recognize silently degrades to the single-step fallback rather than
  erroring, which can produce a `GraphPlan` that under-represents the
  skill's real structure without any signal that compilation "failed."
