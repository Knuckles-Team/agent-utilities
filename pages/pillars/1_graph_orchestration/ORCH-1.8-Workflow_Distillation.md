# Workflow Distillation & Skill-as-Workflow (CONCEPT:AU-ORCH.execution.parallel-engine-visualizer)

## Overview

Closes the "distributed agentic evolution" gap by wiring execution traces
into automatic Skill scaffolding. Workflows are now distributed natively as
executable Skills via `SkillCompiler`, abandoning legacy YAML presets.

## Architecture

<div class="admonition architecture" markdown>
<p class="admonition-title">Workflow distillation pipeline, and skill-as-workflow compilation</p>

**Workflow distillation pipeline.** A successful `synthesizer_step`
triggers `WorkflowDistillationHook`, which calls
`_record_success()` on the `DistillationTracker` node. Once the threshold is
met, promotion runs `_scaffold_skill()`, writing to
`universal_skills/workflows/distilled/`.

**Skill-as-workflow.** `SKILL.md` (prose) compiles via
`SkillCompiler.compile()` -> `compile_from_text()` into a `GraphPlan`,
optionally merged with metadata from `references/team.yaml`.
</div>

## Implementation Details

### Distillation Hook
- **Source**: `agent_utilities/workflows/distillation_hook.py`
- **Entry Point**: `WorkflowDistillationHook.on_execution_complete()`
- **Trigger**: Async background task from `synthesizer_step` (does not block user response)
- **Pattern Key**: Canonical hash of agent topology (node_ids + dependency edges)
- **Scaffolding**: Automatically generates a `SKILL.md` and `references/team.yaml` in the `universal_skills/workflows/distilled` directory once the promotion threshold is met.

### Skill Compiler
- **Source**: `agent_utilities/workflows/skill_compiler.py`
- **Execution**: `SkillCompiler.compile(skill_dir)` / `compile_from_text(name, markdown)` natively parse `### Step N:` prose from `SKILL.md` files into executable `GraphPlan` steps. `save()`/`update_markdown()` round-trip a `GraphPlan` back to prose; `register_in_kg()` registers the compiled workflow.
- **Team Composition**: `load_team_config()` checks `references/team.yaml` for optional team metadata; defaults to a general execution graph if absent.

### Domain Presets
- **Location**: `universal_skills/workflows/<domain>/`
- **Migration**: Legacy YAML presets (finance, infra, research) have been migrated into atomic skill directories.

## Cross-Pillar Integration

| Pillar | Integration Point |
|--------|------------------|
| ORCH-1.22 | GraphPlan orchestration |
| ORCH-1.24 | WorkflowCatalog export |
| AHE-3.2 | Evolution Engine awareness |
| AHE-3.3 | TeamConfig promotion |
| KG-2.0 | DistillationTracker nodes |

## Configuration

```json
{
  "distillation": {
    "promotion_threshold": 3,
    "quality_score_minimum": 0.6
  }
}
```

## Documentation Coverage
- **Pillar**: ORCH (cross-pillar: AHE)
- **Tests**: `tests/test_skill_compiler.py`
- **C4 Diagram**: `docs/pillars/architecture_c4.md` → Workflow Distillation Flow
