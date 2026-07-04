# Tasks: Program-Synthesis Reasoner with MDL Prior (KG-2.66)

Wire-First, ordered. Cite real files. Depends on KG-2.65 `Reasoner` seam landing first.

1. **Confirm the seam (no new abstraction).** Verify the KG-2.65 `Reasoner` protocol + registry
   exist (sibling spec `kg-2-65-no-first-class-reasoner-paradigm-abstraction-rea`). Register against
   it; do **not** introduce a parallel paradigm interface.

2. **Add the MDL operator to the existing registry.** In
   `agent_utilities/harness/selection_operators.py`, add a pure, model-free `description_length` /
   `mdl_score` operator (program size in DSL primitives) beside `bradley_terry_scores` /
   `conservative_rating` / `select_top_k`. No network, no new module.

3. **Build the typed search space over RLM tools.** Use `agent_utilities/rlm/skills.py`
   (`Skill`, `merge_skills`, `EnvironmentAdapter`) as the DSL — candidate programs are tool
   compositions; rely on the existing name-collision `ValueError` to keep composition explicit.

4. **Implement the `ProgramSynthesis` Reasoner** (new module under `agent_utilities/rlm/`,
   `CONCEPT:AU-KG.enrichment.multimodal-readers`): enumerate/mutate compositions, validate each via
   `agent_utilities/rlm/sandboxes/router.py` `SandboxRouter` (ORCH-1.38), then select survivors by
   `quality − λ·description_length` through the step-2 operator. Self-register in the KG-2.65
   registry under a `program_synthesis` capability tag.

5. **Wire-First live-path test.** `tests/unit/knowledge_graph/test_kg_2_66_program_synthesis.py`
   (`@pytest.mark.concept(id="KG-2.66")`): drive the registered `Reasoner` end-to-end (compose →
   sandbox-validate → MDL-ranked select), assert the success metric (λ>0 finds a shorter correct
   program than λ=0 at equal pass-rate), plus a unit test of the pure `description_length` operator.

6. **Regenerate + gate.** `scripts/build_concepts_yaml.py`, then `scripts/check_concepts.py` and
   `scripts/check_wiring.py` green; author the per-concept doc (Solomonoff/MDL + arXiv 2606.12683
   provenance in the docstring); `pre-commit run --all-files` clean.
