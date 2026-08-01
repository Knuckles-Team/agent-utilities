# Design Document: One facade over variant-pool management and skill-neologism evolution, not two independent subsystems

CONCEPT:AU-AHE.harness.evolutionary-aggregation

> `agent_utilities/harness/agentic_evolution_engine.py`.

## Decision — `AgenticEvolutionEngine` composes variant-pool management (AHE-3.2) and skill-neologism detection/evolution (ECO-4.1) into one lifecycle

The module docstring (`agentic_evolution_engine.py:1-11`) states the decision
directly: "Provides a single entry point for all evolutionary capabilities" —
variant pool management via `VariantPool` and skill neologism detection/
evolution via `SkillEvolver`, unified into one cycle: `detect_skill_gap() →
create_skill() → register_variant() → evaluate_fitness() →
tournament_select() → promote_winner()`.

**The rejected alternative is what the two subsystems' separate origins
imply: keep variant-pool management and skill evolution as two independent
call paths**, each driven by its own caller with its own notion of when a new
candidate exists and how it gets promoted. That would mean a newly-detected
skill gap and a newly-mutated variant go through two different
evaluate/promote pipelines even though both are, structurally, "a candidate
competing to replace something live." The facade instead makes skill
creation feed directly into the SAME tournament-selection and promotion
machinery that variant-pool management already had, so there is one fitness
evaluation and one promotion decision, not two.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/agentic_evolution_engine.py`,
  its `VariantPool` (`agent_utilities/harness/variant_pool.py`) and
  `SkillEvolver` (`agent_utilities/knowledge_graph/adaptation/skill_evolver.py`)
  dependencies.
- **Backward Compatible**: Yes — the facade composes existing subsystems
  without changing their own public APIs.
- **Known weak point**: unifying the promotion decision means a skill
  candidate and a parameter-mutation variant are evaluated by the same
  fitness function even though "does this skill address a real gap" and
  "does this mutated sampling profile score better" are qualitatively
  different judgments — a fitness function tuned for one can under- or
  over-value the other.
