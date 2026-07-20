# Tasks: Corrigibility + Knowledge-Seeking Objective Primitives (SAFE-1.5)

Wire-first, ordered. Reuse the AU-ORCH.session.durable-session-autonomous-goal goal loop and OS-5.24 ActionPolicy — do not rebuild either.

1. **Confirm the seams** (read-only): `agent_utilities/models/goal.py` (`GoalSpec`, `GoalIteration`,
   `GoalCheckpoint`, `GoalResult`, `GoalStatus`), `agent_utilities/orchestration/durable_execution.py`
   (`DurableExecutionManager.save_checkpoint`), `agent_utilities/orchestration/action_policy.py`
   (`_blast_exceeded`), and the wasm epoch-interrupt pattern in
   `agent_utilities/rlm/sandboxes/wasm_backend.py`.

2. **Add the corrigibility controller** — a `CorrigibilityController` with `request_shutdown()` /
   `should_yield()` (new `orchestration/corrigibility.py`, or fold into `models/goal.py`). Tag
   `# CONCEPT:AU-OS.safety.irreversibility-aversion`. Generalize the wasm epoch-interrupt semantics to the goal loop level.

3. **Wire it into the live loop** — at the top of each `GoalIteration` the loop calls `should_yield()`;
   on yield it persists a `GoalCheckpoint` via `DurableExecutionManager` and returns
   `GoalResult(status=interrupted)`. Add `GoalStatus.INTERRUPTED`. No new iteration / retry / escalation.

4. **Add the opt-in knowledge-seeking objective** — `info_gain_reward(before, after)` (expected
   uncertainty reduction over the KG belief) and a `GoalSpec.objective` field defaulting to the current
   behavior; the reward is computed only when `objective == "knowledge_seeking"`. Tag `# CONCEPT:AU-OS.safety.irreversibility-aversion`.

5. **Route irreversible actions through the existing cap** — when an action is marked `irreversible=True`,
   call the OS-5.24 `ActionPolicy` blast-radius path (`_blast_exceeded`); do not add a new gate.

6. **Tests** — `tests/unit/orchestration/test_safe_1_5_corrigibility_objective.py`
   (`@pytest.mark.concept(id="SAFE-1.5")`): a shutdown-mid-loop test (checkpoint persisted, status
   interrupted, no resist/accelerate), a no-signal backward-compat test, an `info_gain_reward` unit test,
   and an irreversible-action-hits-blast-cap test.

7. **Registry + docs + green** — run `scripts/build_concepts_yaml.py` then `scripts/check_concepts.py`;
   author `docs/architecture/corrigibility_and_knowledge_seeking.md`; drive `pre-commit run --all-files`
   fully green.
