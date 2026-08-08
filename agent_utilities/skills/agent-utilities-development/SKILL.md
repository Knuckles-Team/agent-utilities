---
name: agent-utilities-development
skill_type: skill
description: >-
  Review or implement a concrete agent-utilities repository change. Use for
  read-only orientation, impact or diff review, concept-aware design, approved
  implementation, tests, wiring, REST and MCP parity, documentation, regression
  gates, and isolated lane delivery through the repository-manager lane and
  merge-queue mechanism. For evidence triage, gap proposals, or skill and prompt
  optimization before implementation approval, use agent-utilities-evolution.
---

# Agent Utilities development

Review a proposed change without mutation, or implement an approved change in an
isolated **lane** and prove its live path and relevant gates.

This repository is worked by dozens of agents and humans concurrently, so
delivery is not "branch, commit, merge." It is the repository-manager lane
mechanism: **`--lane start` → work → gates → `--lane finish` → the queue lands
and prunes it.** The mechanism owns isolation, differential gating, landing, and
pruning; this skill owns what to build and how to prove it.

- Isolation, the traps, preflight, finishing → `repository-manager-lane-lifecycle`
- Landing, gates, conflicts → `repository-manager-merge-and-reconcile`
- Waves across many repos, concurrency sizing → `repository-manager-fleet-scale-operations`

## Workflow

### 1. Read the governing context

- Read every applicable `AGENTS.md` before editing.
- Inspect the owning architecture guide, specification, tests, and public entry
  points.
- Search existing concepts and implementations before adding a new abstraction.
- Preserve unrelated changes in a dirty worktree.

Use `graph-query-and-explanation` for code context and impact when the code graph
is available. Fall back to repository search when it is not.

### 2. Isolate and scope

- Translate the request into observable acceptance checks.
- Identify every owned consumer of any contract that will change.
- For orientation, design, impact, or diff review, inspect the current checkout
  read-only and report findings; stop before creating a branch, worktree, edit,
  commit, or other delivery artifact.
- For an authorized implementation, open a lane:

  ```bash
  repository-manager --lane start --lane-repo agent-utilities \
      --lane-branch lane/<name> --lane-base main
  cd <the worktree it printed>
  eval "$(repository-manager --lane env --lane-path . --lane-shell)"
  ```

  A worktree alone is **not** isolation. `--lane start` also gives you a private
  `CARGO_TARGET_DIR`, `PYTEST_ADDOPTS --basetemp`, `TMPDIR` and
  `PRE_COMMIT_HOME`, and returns a preflight report that *proves* the isolation
  rather than asserting it. Never edit the canonical checkout: a background
  `git reset` there has already destroyed ~20 minutes of a lane's work.
- Identify every owned consumer with `scripts/find_callers.py`, not grep —
  `import x as y`, a bare `from … import x`, and
  `monkeypatch.setattr("pkg.mod.x", …)` all hide callers from a symbol search,
  and all three have been missed here.
- Wire any new control at the **chokepoint**, not one entrypoint. A control
  wired at a single entrypoint was deployed and changed literally nothing
  because six callers bypassed it.
- Avoid compatibility aliases: update consumers atomically and delete the old
  path.

### 3. Implement at the owning seam

- Put shared behavior in the core, leaving transports as thin adapters.
- Keep REST and MCP entry points on the same action or service implementation.
- Make normal enhancements native to the existing flow unless their cost or
  risk requires an explicit control.
- Use Pydantic models for structured boundaries and existing dependency patterns.
- Keep examples synthetic and exclude credentials, private endpoints, personal
  data, and machine-specific paths.

### 4. Prove wiring

Trace from a real entry point to the changed behavior. Add a live-path test that
would fail if the new code were merely importable but never invoked. For dynamic
registration, verify the discovery call as well as the registered object.

Three proof obligations that get skipped, each of which has produced a false
green here:

- **Prove a gate catches a deliberately-introduced known-bad input.** Break the
  thing on purpose, watch the gate refuse it, revert. Three gates were found
  green while enforcing nothing — one crashing, one blind to 2 of 16 patterns it
  claimed to cover, one never discovered by its own runner. A gate that has only
  ever seen good input has not been tested.
- **Run a defect-pinning test against the RESTORED bug and confirm it fails.** A
  lane caught its own test passing against the very bug it claimed to pin;
  another found a gate meta-test that had encoded a bug as correct.
- **A capability is not done because it exists.** Fourteen capabilities here were
  fully built and still tracked as unimplemented, because nothing wired them;
  "built but not wired" is the default failure of this codebase, not an edge
  case.

### 5. Document the behavior

- Update the owning guide and Mermaid diagram.
- Update generated sources rather than hand-editing generated artifacts.
- Keep concept references, code, tests, and docs consistent.
- Update exact skill names and paths in prompts, fixtures, scripts, and docs.

### 6. Validate

★ **`uv run pytest` is poisoned in this repository.** It silently resolves the
**system** interpreter and its stale packages, and produced ~80 phantom failures
that cited this project's own guards; six lanes were burned before it was found.
Always:

```bash
python3 scripts/uv_workspace.py run --all-extras -- pytest <args>
```

Print `sys.executable` and the package count in the same run: **≈726 packages is
the correct environment, ≈44 is the stale one.** A verdict from an unproven
interpreter is not evidence. A worktree-local `.venv` has the same effect (~167
phantom failures) — `--lane doctor` refuses one.

Run the narrow tests first, then every gate touched by the change. Run the full
pre-commit suite before delivery, and take the lease for it — it is LEASE-class
because it can destroy unstaged work:

```bash
agent-utilities lane lease --resource precommit-all-files --operation gate -- \
  pre-commit run --all-files
```

Never `--no-verify`, and never mask a gate to force green: `noqa`, `type:
ignore`, `nosec`, `skip`, and `xfail` appearing in a delivery diff are what a
reviewer greps for first.

★ **Judge every gate DIFFERENTIALLY, against the base ref — never against
absolute green.** `main` is legitimately red. An absolute standard once
deadlocked the queue and stranded 19 branches, and rejected a branch that fixed
21 of 30 failing tests because 9 remained. Compare at the granularity the gate
declares (pytest **node ids**, not counts or files). If the baseline cannot be
produced, refuse rather than allow-all.

★ **Measure the MERGED tree, not the branch tip** — `git merge-tree --write-tree
main HEAD`. Reasoning from `git show <branch>:<path>` misled three people in one
day; one concluded a branch had deleted a guard the merged tree in fact kept.

Inspect the final diff for generated churn, stale names, sensitive data, and
stray files. Commit with a neutral repository identity.

### 7. Deliver through the queue

```bash
repository-manager --lane finish --lane-path . --lane-base main
repository-manager --merge-queue status --repo-path .        # watch, do not babysit
```

`finish` preflights (blocking), then enqueues. **Enqueued is not a to-do item:**
a scheduler drains the queue every ~5 minutes, gates the candidate *as merged*,
fast-forwards under both the lane and canonical guards, and prunes the worktree
and the branch. Do not hand-merge into `main` because the queue feels slow — two
lanes hand-merging is how a resolution gets orphaned. If the candidate is
rejected, or a merge conflicts, follow the decision procedure in
`repository-manager-merge-and-reconcile` (generated-file → regenerate; base moved
→ re-measure; gate red → NEW or pre-existing; textual conflict → read both sides'
**intent**, because a semantic divergence can hide inside one).

★ **A merge to `main` is a live deploy.** Fleet pods `hostPath`-mount the
canonical tree over `site-packages`, so landing and deploying are the same act —
check runtime compatibility against the deployed images, not just your venv.

Use an economy model for inventory, search, mechanical edits, and deterministic
checks. Reserve stronger reasoning for ambiguous design, security review, and
cross-system synthesis.

## Skill changes

When editing bundled skills:

1. ★ **`SKILL.md` frontmatter MUST declare `skill_type: skill`** alongside `name`
   and `description`. This is not cosmetic: the delegation binder can only *run*
   a skill that ingests as a `CallableResource(resource_type='AGENT_SKILL')`, and
   the ingester decides that shape by reading `skill_type:`. A file without it
   becomes a 0-step `WorkflowDefinition`, which can be described but never
   executed — `execute_agent` fails at skill resolution with *"exists as
   ['WorkflowDefinition'] but has NO CallableResource node."* 415 of 625 fleet
   skill nodes were once mis-shaped exactly this way. Check this **first** when a
   delegation "cannot find" a skill that plainly exists on disk;
2. generate `agents/openai.yaml` deterministically;
3. put Graph-OS coverage in `agents/graph-os.yaml`;
4. run the skill validator for every retained skill;
5. test both direct and delegated synthetic tasks — a skill that has only been
   invoked directly has not been shown to be runnable;
6. update the coverage gate and current inventory documentation;
7. do **not** hand-edit `WORKFLOW.md` or a provider's `references/catalog.md`;
   both are generated by `scripts/consolidate_provider_skills.py`. Edit
   `SKILL.md` and regenerate.

## Adding a new platform capability (engine → verb → route → skill)

Adding a capability end-to-end — reachable, documented, and discoverable with no
drift between layers — follows a fixed build order:

1. **Engine crate (Rust)**, when the capability needs native compute: implement it
   in the epistemic-graph engine and expose it as a wire `Method`
   (`crates/eg-types/src/protocol.rs`). The pure-Python `epistemic_graph` client
   mirrors the wire protocol 1:1, so a new method surfaces as a coroutine on a
   sub-client with no client-side hand-editing; that client is the source of truth
   for "what the engine can do."
2. **MCP verb + REST route.** A new engine method is auto-discovered by
   `engine_tools._discover_domains()` (client introspection) and appears under its
   domain's `engine_<domain>` action-routed tool automatically — a brand-new
   *domain* needs an entry in `_DOMAIN_CLASSES`/`_DOMAIN_BLURB`, with its REST twin
   `/engine/<domain>` registered in the same change (`ACTION_TOOL_ROUTES`). For a
   synthesized, agent-facing operation, add a curated `graph_*`/`ontology_*`/
   `object_*` tool and register its REST route in the SAME call so the
   surface-parity gate stays green (see *Two surfaces by default*).
3. **Wrapping skill.** Author or extend the domain skill covering the new verb so
   operators can discover it. The naming/coverage contract and the doctor that
   enforces it are documented in `graph-runtime-and-governance`'s "Coverage
   governance" section — run it as part of closing this out.

Verify the whole chain in one pass:

```bash
R="python3 scripts/uv_workspace.py run --all-extras --"   # never bare `uv run`
$R python -m agent_utilities.mcp.skill_coverage  # verb <-> skill coverage: 0 uncovered, 0 orphans
$R pytest tests/unit/test_gateway_mcp_parity.py  # tool <-> REST-route parity
$R python scripts/gen_graphos_manifest.py        # regenerate the action manifest from the client
```

A new verb shipped without covering documentation shows as **uncovered**; stale
coverage pointing at a removed verb shows as an **orphan** — fix both before merge,
or add the verb to the documented exemption list with a written justification.

## Guardrails

Each of these names its replacement, because a prohibition without one does not
hold. Run `repository-manager --lane doctor --lane-path .` and it will tell you
which of them you are currently violating, with the exact remedy command.

- **Never edit the canonical checkout.** Work in the lane worktree.
- **Never `git stash`.** `refs/stash` is ONE ref shared by every worktree here.
  To read a pristine file while yours is dirty: `git show HEAD:<path>`. To park
  work: a `wip:` commit on your branch, or `agent-utilities lane park`.
- **Never export a shared `CARGO_TARGET_DIR`** — it corrupts concurrent worktree
  builds, it does not merely serialize them. Use `--target-dir ./target-isolated`
  and prune it; `agent-utilities lane bind-cargo` makes the partition structural.
- **Never run with the shared `PRE_COMMIT_HOME`.** pre-commit writes your
  unstaged work to a patch file there and restores it in a `finally:`; a crash
  inside that window loses it. `--lane env` sets a private one.
- **Never `git branch -D`.** Only `-d` — its refusal is the safety mechanism
  telling you the work is not contained in the base.
- **Never hand-edit a generated view** (`docs/concept_reservations.yaml`,
  `reports/PROGRAM.md`, a provider's `WORKFLOW.md`/`catalog.md`). Write your
  fragment or edit the source and regenerate; `lane-guard` refuses a hand-edited
  ledger view.
- **Register writes use `--detail-file`/`--evidence-file`, never `--detail "…"`.**
  Register prose contains backticked identifiers, and inside double quotes bash
  performs command substitution on backticks — silently executing them. This has
  already truncated live entries and triggered an accidental `uv sync` against
  the shared workspace `.venv` (D-ORC-22).
- Do not bypass failing gates or silently accept warnings.
- Do not create a second implementation for another entry point.
- Do not commit secrets, credential files, local inventories, or scratch output.
