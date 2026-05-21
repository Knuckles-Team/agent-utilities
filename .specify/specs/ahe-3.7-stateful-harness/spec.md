# Spec: Stateful Harness & Sensory Verification (CONCEPT:AHE-3.5)

## Pre-Flight Checklist (Mandatory — DSTDD)

- [x] **KG search completed** — `.specify/design/ahe-3.7-stateful-harness/design.md` exists
- [x] **Extension point identified** — Extends AHE-3.7 (Stateful Harness) & OS-5.3 (Security)
- [x] **C4 diagram created** — showing component interactions in Pillar 3 and Pillar 5
- [x] **No new CONCEPT: tag** — uses existing tags
- [ ] **`code-enhancer` audit** — pending implementation
- [ ] **Design validation passes** — pending verification

## Design Reference

→ [design.md](../../design/ahe-3.7-stateful-harness/design.md)

---

## User Stories

### US-1: Concurrency State Branching
**As a** coordinator agent managing parallel swarms,
**I want** to branch my state space into parallel forks,
**so that** concurrent agent steps can edit their local memory space without cross-contaminating base state.

**Acceptance Criteria:**
- [ ] `BranchMergeStateLocker.fork_state(base_key, branch_name)` copies the base state data, stamps the current version as `base_version`, and persists the fork.
- [ ] `BranchMergeStateLocker.get_branch_state(base_key, branch_name)` retrieves the staged branch state.
- [ ] `BranchMergeStateLocker.update_branch_state(base_key, branch_name, new_data)` updates the branched state successfully.

---

### US-2: Smart State Merging & Conflict Resolution
**As a** state management framework,
**I want** to merge parallel state branches back into the base key while handling concurrency conflicts,
**so that** parallel execution paths safely converge to a single source of truth.

**Acceptance Criteria:**
- [ ] Fast-forward merge succeeds when base version has not changed.
- [ ] Three-way merge performs a recursive dictionary merge on key-level divergence.
- [ ] Custom `resolver` callback is supported to let agents programmatically arbitrate state key collisions.
- [ ] Branch keys are automatically deleted from memory/Redis post-merge.

---

### US-3: Declarative Pre/Post Conditions (Sensors)
**As a** system security guardrail,
**I want** to define declarative pre-conditions and post-conditions for execution nodes,
**so that** agents are blocked from launching under invalid states, and output state corruption is caught immediately.

**Acceptance Criteria:**
- [ ] `ToolContract` registers a callable `pre_condition` and a Pydantic `post_condition_schema` or verifier callable.
- [ ] `ContractValidator.validate_pre()` evaluates pre-conditions.
- [ ] `ContractValidator.validate_post()` validates execution outputs against the contract schema.

---

### US-4: End-to-End Sensory Integration
**As a** graph orchestrator,
**I want** step transitions to validate contracts and fork/merge execution state dynamically during steps,
**so that** the entire graph executes within deterministic, safety-checked harness boundaries.

**Acceptance Criteria:**
- [ ] `expert_executor_step` pre-validates node contracts before executing handlers.
- [ ] `expert_executor_step` post-validates outputs before concluding the transition.
- [ ] Fails/retries steps if contract validations throw errors.

---

## Non-Functional Requirements

- [ ] All existing tests continue to pass (zero regression)
- [ ] Pre-commit hooks pass cleanly
- [ ] Documentation updated to reflect transactional state management
