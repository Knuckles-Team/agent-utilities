# Tasks: Emergent Specialist Discovery (ORCH-1.47)

Wire-first, ordered. Cite the real files; extend, do not rebuild.

1. **Read the seam.** `agent_utilities/graph/team_evolution.py`
   (`TeamEvolutionEngine.evaluate_and_evolve`, the hard-coded `error_recovery_specialist`
   mutation) and the MASS heterogeneity API in `agent_utilities/graph/social_system.py`
   (ORCH-1.32). Confirm the `CoordinationTrace`/`Episode` read shapes in
   `agent_utilities/graph/coordination.py` and `agent_utilities/models/knowledge_graph.py`.

2. **Add niche discovery** to `team_evolution.py`: `discover_specialization_niches(team_id)`
   — cluster recent failing/expensive episode embeddings, score each cluster by failure
   density vs. best-archetype competence, return only niches below a module-constant
   competence threshold (no env flag). `CONCEPT:ORCH-1.47` tag in the docstring.

3. **Replace the hard-coded proposal** in `evaluate_and_evolve`: when a niche qualifies,
   synthesize a niche-derived `proposed_agent` label + role prompt + tool scope (a
   `TeamComposition` role per `agent_utilities/models/knowledge_graph.py`); emit it via the
   **existing** `MutationProposal` / `PROPOSED_MUTATION` MERGE. Delete the constant label and
   its fallback (No-Legacy).

4. **Gate on the objective** using `social_system.py` (ORCH-1.32): mark the proposal
   `promotable` only if it raises archetype entropy AND projected outcome quality vs. the
   pre-proposal baseline; else write `status='rejected'` with reason. The accepted proposal
   continues down the existing AHE-3.18 regression-gated golden-loop path —
   `manifest_generators.py` (ORCH-1.9) consumes the role unchanged.

5. **Test** `tests/unit/graph/test_orch_1_47_emergent_specialist_discovery.py`
   (`@pytest.mark.concept(id="ORCH-1.47")`): AC1–AC4 + a `*_live_path` test asserting the
   niche-derived proposal replaces the old constant and the entropy/quality gate fires.

6. **Register + document:** add the `CONCEPT:ORCH-1.47` marker, run
   `scripts/build_concepts_yaml.py` + `scripts/check_concepts.py`; extend
   `docs/architecture/multi_agent_social_system.md` with the discovery → propose → gate loop.

7. **Quality bar:** `pre-commit run --all-files` fully green; `check_wiring.py` shows the new
   code reached from `evaluate_and_evolve`. Worktree → merge to main locally.
