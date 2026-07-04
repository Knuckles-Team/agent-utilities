# Agentic Prompts

This directory contains the **packaged base** system-prompt JSON blueprints that
define the "souls" and specialized behaviors of agents in the ecosystem. It is the
first (lowest-precedence) layer of the Knowledge-Graph **prompt library**; the
fleet's agent-packages and an operator XDG overlay contribute the rest (see
*Discovery & layering* below).

## Overview

The `prompts/` module uses **JSON-as-Code** for high-precision task specification.
Moving away from free-form Markdown, these blueprints conform to a standardized
Pydantic schema to ensure consistency, versioning, and programmatic manipulation.

## Canonical schema (CONCEPT:AU-ORCH.routing.resolve-body-single-canonical)

The schema is **owned by the Pydantic model `StructuredPrompt` in
[`../prompting/structured.py`](../prompting/structured.py)** — the single source of
truth. Its JSON Schema is generated to
[`../prompting/prompt.schema.json`](../prompting/prompt.schema.json) by
`scripts/gen_prompt_schema.py` (regenerated, never hand-edited).

Required / standard fields:

- `schema_version` — `"1.0"` (the canonical schema version).
- `task` — stable slug/identifier (e.g. `"python_programmer"`).
- `type` — must be `"prompt"`.
- `source` — provenance / KG namespace (`"agent-utilities:base"` for these; a
  package name like `"gitlab-api"` for fleet-contributed prompts).
- **Body → `instructions.core_directive`** — this is the ONE canonical body
  location. (The legacy flat `content` / `input` keys are migration-only and are
  read by `resolve_body()` for back-compat, never authored in new prompts.)

Optional but standardized: `metadata` (description/topic/tone/style/audience),
`identity` (role/goal/personality), the rest of `instructions`
(responsibilities/capabilities/workflow/quality_checklist/methodology/output_format),
`engineering_rules` (CONCEPT:AU-KG.ingest.engineering-rules), `rules`, `skills` (skill slugs the prompt
expects installed), `tools`, and the composition fields `extends`
(e.g. `"agent-utilities:base"`) + `compose` (`append` | `prepend` | `replace`).

## Authoring, validation & drift control

- **Author / scaffold** a prompt with the **`prompt-builder`** universal skill
  (`build_prompt.py` / `validate_prompt.py`) — a thin front-end over
  `StructuredPrompt`.
- **One validator** — `agent_utilities.prompting.structured.validate_canonical()`
  backs the prompt-builder, the CI gate `scripts/check_prompt_schema.py`, and
  per-package `test_prompt_parity`, so "valid here" == "valid in CI".
- **Body resolution** — `resolve_body()` is the single reader used by the prompt
  builder, the workspace builder, and KG ingestion (this fixed a bug where
  decomposed prompts were read as empty).

## Discovery & layering (the KG prompt library)

`registry_builder.ingest_prompts_to_graph()` ingests prompts in precedence order
(later overrides earlier on the namespaced `PromptNode` id):

1. **packaged base** — these `agent_utilities/prompts/*.json` → `prompt:<name>`
2. **fleet-contributed** — every installed agent-package that declares a
   `[project.entry-points."agent_utilities.prompt_providers"]` → `prompt:<pkg>/<name>`
3. **operator overlay** — `*.json` in `core/paths.prompts_dir()`
   (`~/.config/agent-utilities/prompts/`) — a drop-in override layer.

A package prompt sets `extends: "agent-utilities:base"` + `compose: append` to
inherit one of these base prompts at render time. See
[`../../docs/architecture/modular-prompt-skill-contribution.md`](../../docs/architecture/modular-prompt-skill-contribution.md).

## Key Specialist Prompts

- **Planner (`planner.json`)** — high-level task decomposition and goal setting.
- **Router (`router.json`)** — initial query analysis and topology selection.
- **Python Programmer (`python_programmer.json`)** — Python-development logic.
- **Verifier (`verifier.json`)** — quality gate and result scoring logic.
- **Coordinator (`coordinator.json`)** — multi-agent task management / barrier sync.

## Maintenance

- **Add a specialist** — author it with `prompt-builder`; validate with
  `validate_prompt.py --strict`; the `check_prompt_schema` gate enforces conformance.
- **Optimization** — use the `self_improvement_tools` to evolve these prompts based
  on textual gradients and outcome rewards.
- **One-time migration** — `scripts/migrate_prompts.py` canonicalizes legacy
  blueprints (moves `content`/`input` → `instructions.core_directive`, stamps
  `schema_version`/`source`).
