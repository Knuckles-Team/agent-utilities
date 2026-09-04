---
name: agent-utilities-development
skill_type: skill
description: >-
  Review or implement a concrete agent-utilities repository change. Use for
  read-only orientation, impact or diff review, concept-aware design, approved
  implementation, tests, wiring, REST and MCP parity, documentation, regression
  gates, and isolated lane delivery through the repository-manager lane and
  merge-queue mechanism. Carries the component inventory by layer, the dependency
  direction and known import SCC, the architecture and anti-sprawl gates, and the
  checklist to run before adding any module, route, constant, or dependency. For
  evidence triage, gap proposals, or skill and prompt optimization before
  implementation approval, use agent-utilities-evolution.
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

## Mandatory guardrails (G1–G12) — READ BEFORE EVERYTHING BELOW

Twelve rules. **Every one was broken in this program, usually by someone who had
just written it down**, so each names the failure that produced it and — the part
that matters — **what actually enforces it in this repo today**. A rule you have to
remember at the moment of writing is not a guardrail. Where nothing enforces one,
that is stated plainly; claiming enforcement that does not exist would be G2
committed inside the guardrail section itself. Numbers and mechanisms are
cross-referenced to the section that owns them, never restated here.

| # | The rule, instantiated for agent-utilities | Enforced today by |
|---|---|---|
| **G1** | **State the universe with every number. MUST.** Broken 12 times in one session. Every count in this skill carries its command for exactly this reason — the "47 packages" figure has three defensible answers (see *Orient first*), and three fleet denominators (55/71/73) were in flight against a reconciled 78. | **Nothing — discipline.** Self-check: paste the command beside the number. |
| **G2** | **A gate's declaration is not its behaviour. Run it. MUST.** Here the declaration that lies is usually a `.pre-commit-config.yaml` **comment**: two still describe `openapi_coverage_baseline.json` / `wire_first_baseline.json` ratchets whose files do not exist and whose flags now exit 2. Read the gate **script's** docstring, never the hook comment. | `guardrail-gate-meta-tests` (pre-push/manual) — a gate that cannot be shown to fail on a deliberately broken fixture in `tests/gates/` (44 test modules) is refused. Self-check: cite a gate's status from its OUTPUT with the invocation quoted. |
| **G3** | **"Green" is meaningless without its universe. MUST.** AU's analogue of EG's feature matrix is the **hook universe vs the package universe**: the `mypy` hook sets `exclude: ^(tests/\|test/\|scripts/\|script/)`, `--ignore-missing-imports`, and exactly **three** `additional_dependencies`; `check-import-safety`'s entry carries **22 `--exclude` module exemptions** (both verified 2026-09-03). Package-wide mypy and the hook therefore report **different numbers, both correct** — a recorded run had the package at 19 and the hook at 5. Same shape for `vulture`, `ruff`, `bandit`, `codespell`. **Scope work from the HOOK's output, never a package-wide count.** | **Nothing checks universe agreement — discipline.** Self-check, in this order: `pre-commit run mypy --files <your changed paths>` (the targeted form; never `--all-files` bare, see G11), then `python3 scripts/uv_workspace.py run --all-extras -- mypy agent_utilities` if you want the package figure — and report which one you are quoting. |
| **G4** | **A fix that passes only its author's own new test has demonstrated nothing. MUST.** Procedure below; it is the guardrail that would have caught two of this session's worst regressions. | The merge queue runs `select_tests` over the changed paths **as merged** — but only *after* you enqueue. Before that: **nothing.** Run it yourself, below. |
| **G5** | **Re-derive the consequential closure on every composition. MUST.** A frozen path set describes the base it was cut against, not the base it lands on. In AU this bites as a base that moved under you: the merge gate always evaluates the candidate **as merged** (`git merge-tree --write-tree` → `commit-tree` → throwaway detached worktree), never as it sat on your branch, precisely because a branch tip is not the tree that lands. A recorded path/edge count is a LOWER BOUND, never a scope. | `governance/merge_queue.py` — gates the merged tree, and **refuses rather than allows** when a baseline cannot be produced. Locally: `git merge-tree --write-tree main HEAD` (see *Validate*). |
| **G6** | **Verify a claim before acting on it — including from a skill, a brief, or a report. MUST.** Of nine strict-review blockers on the AU candidate, **one was factually wrong on inspection** — "the fifth recorded work item in this program to prove stale or false". Lanes that verified were right; lanes that trusted were wrong. This applies to *this file*: it carries dated measurements that rot. | **Nothing — discipline.** Cite `file:line` you personally opened, and prefer `graph_code action=code_context` / `scripts/find_callers.py` over grep (see checklist item 1 for the three call shapes grep misses). |
| **G7** | **Never delete on island evidence alone. MUST NOT.** A *module* island is an import fact; a *symbol* island is a LEAD for a human to trace. `scripts/check_wiring.py --wire-first-report` answers "does anything import or call this?" statically and is **a finder of suspects, not a gate** — blind to decorator/entry-point registration and out-of-repo callers. | `guardrail-removed-symbol-consumers` (pre-push/manual, `scripts/check_removed_symbol_consumers.py`) refuses deleting or renaming a public symbol a fleet repo still imports, and **fails closed** when `scripts/fleet_symbol_consumers.json` is missing, unparseable, or older than 30 days. Regenerate with `python3 scripts/gen_fleet_symbol_consumers.py --update`. |
| **G8** | **No baseline files. Ever. MUST NOT.** A baseline records "whatever is true today" and only grows by omission. Verified 2026-09-03: **none of `scripts/{openapi_coverage,wire_first,surface_parity,env_flag,skill_collision}_baseline.*` exists**, and `check_{concept_governance,event_loop_blocking,liveness,no_per_element_ingest_loop,openapi_coverage,swallowed_errors,surface_parity}.py` all **exit 2** on their retired flag. Thresholds are absolute and driven down deliberately. | Retirement is enforced by the scripts themselves (exit 2). ⚠ **One live ratchet survives:** `scripts/check_skill_name_collision.py:374` still *writes* `scripts/skill_collision_baseline.txt`. It behaves absolutely today for one reason only — **that file does not exist**. Do not run `--update-baseline`: one invocation re-installs the ratchet and grandfathers every collision then present. |
| **G9** | **A gate whose universe is empty must FAIL, not pass. MUST.** Vacuous truth is not coverage. AU's instance was mechanical, not careless: git exports `GIT_DIR`/`GIT_INDEX_FILE` into **every** hook subprocess, under which `git -C <subdir> ls-files` silently re-roots its output — so ~20 copy-pasted gate helpers reconstructed paths that exist nowhere, measured an **empty universe**, and reported a confident "all clear". | `scripts/_git_scan.py` (`tracked_or_walked`, `repo_root_of`) — the one extracted, ambient-env-immune discovery helper, now used by **21** gate scripts and pinned by `tests/unit/scripts/test_git_scan.py`. **Any new gate MUST discover its universe through it**, and must refuse an empty result rather than pass. Self-check — run your gate BOTH ways and diff: `python3 scripts/<gate>.py` vs `GIT_DIR=$PWD/.git GIT_INDEX_FILE=$PWD/.git/index python3 scripts/<gate>.py`. |
| **G10** | **A count's composition matters as much as its size. MUST.** A delta is not progress until you know what it is made of. In AU: a raw-SCC counter moving down is explicitly *not* runtime proof, and a predicate fix that repaired 3 failures **regressed the suite 15 → 27**. The 162-edge "cut budget" is a nominal planning sum with `expected_delta: OPEN`. | **Nothing — discipline.** Break every delta down by kind (blocking vs advisory, new vs pre-existing) before calling it progress. |
| **G11** | **Isolation and staging. MUST NOT.** Never edit the canonical checkout · never harness worktree isolation (it sets `core.bare=true` on the shared common dir) · never `git stash` (`refs/stash` is ONE repo-wide ref) · never `git add -A`/`.` · never bare `pre-commit run --all-files` · never `git branch -D` · never `update-ref` to advance a branch (verify by TREE, `git cat-file -e HEAD:<path>`). **The `## Guardrails` section at the end of this file owns the full list *with each prohibition's replacement command* — read it; this row is only the index.** | `lane-guard` refuses a non-merge commit authored in the canonical checkout, a hand-edited `docs/concept_reservations.yaml`, and a commit made with an off-lane `CARGO_TARGET_DIR`. `check-root-hygiene` refuses undeclared root entries and self-installing ratchet artifacts anywhere in the tracked tree. `repository-manager --lane doctor --lane-path .` tells you which you are violating right now, with the remedy. **`git stash`, `git add -A`, and bare `--all-files` are NOT gated — discipline plus the wrapper.** |
| **G12** | **Fail closed, and never substitute a sentinel for absence. MUST.** A reader that swallows its exception and returns `[]`/`0`/`False` is indistinguishable at the call site from a healthy "nothing found" — five AU safety gates were found doing exactly this against the same KG, so all five stood down together at the moment they existed for. Failure must be a **distinct** value (`None`, or raise), and the caller must deny, defer, or escalate on it. | `check-swallowed-errors` (`scripts/check_swallowed_errors.py`) refuses a handler that discards *why* an operation failed unless it logs the bound exception or carries a justified `# noqa: BLE001 — <reason>`. `guardrail-liveness` catches the sibling shape — a live-surface handler that returns a canned payload while doing no real work. Both are static heuristics, not provers. |

### G4 in practice — the scoped PRE-EXISTING run, before the full run

Twice in one session a plausible fix compiled and passed the fixture written for
it and was still wrong; only a **pre-existing** test caught it. A scoped run that
covers only *new* tests is not evidence. Do this before the full run and before
`--lane finish`:

1. **Let the queue's own selector name the touched area** — do not invent a
   mapping. `select_tests` maps each changed `agent_utilities/<pkg>/<mod>.py` to
   every `tests/**/test_<mod>*.py`, and runs a changed test file as itself:

   ```bash
   # run from the lane worktree root; needs no extras (bare python3 resolves it)
   python3 -c 'import subprocess; from pathlib import Path; \
   from agent_utilities.governance.merge_queue import select_tests; \
   c=subprocess.run(["git","diff","--name-only","main...HEAD"], \
   capture_output=True,text=True,check=True).stdout.split(); \
   print("\n".join(select_tests(Path("."), c)))'
   ```

2. **Subtract the tests your own change added**, so what remains is genuinely
   pre-existing — that subtraction *is* the guardrail:

   ```bash
   git diff --name-only --diff-filter=A main...HEAD -- 'tests/**'
   ```

3. **Run the remainder first**, through the launcher, never bare `uv run`
   (see *Validate* for why that verdict would be worthless):

   ```bash
   python3 scripts/uv_workspace.py run --all-extras -- pytest <the remaining files> -q
   ```

4. **Then** run your new tests, then the touched gates, then the full suite.
5. **Say which pre-existing test would have caught the bug had it been running.**
   If none would have, you have not written the test yet — and if the selector
   returned nothing, that is a finding about test coverage, not a green light.

## Orient first — the component inventory (READ BEFORE ADDING ANYTHING)

Most sprawl here is not written deliberately; it is written by an author who did
not know an owner already existed. `agent_utilities/` is **1,712 tracked `.py`
modules** (`git ls-files agent_utilities | grep -c '\.py$'`) across **47
top-level importable packages** — first-level directories carrying their own
`__init__.py`:

```bash
git ls-files agent_utilities | awk -F/ 'NF==3 && $3=="__init__.py"{print $2}' | sort -u | wc -l   # 47
```

★ **State which definition you used** — three neighbouring ones give three
answers (measured 2026-09-03): **47** first-level dirs with their own
`__init__.py`, **48** first-level dirs containing a tracked `.py` anywhere below
(`agent_chat/` has no top-level `__init__.py`), **51** first-level tracked dirs at all
(`.agent_data/`, `data/`, `images/` hold no Python).

Nobody reads 1,712 modules. So read the *layer* table, then the *size* table
(both in the reference below), then `scripts/find_callers.py` — in that order —
before you create a module.

### Layers and sizes — where an owner already lives

[`references/architecture-reference.md`](references/architecture-reference.md) carries the
six target layers (`contracts -> domain -> ports -> application -> adapters ->
composition`) with the existing packages assigned to each and what each may NOT
import, the measured 62 application→adapter / 85 adapter→adapter violations, the
note that the API gateway and the graph-os MCP surface both live INSIDE this repo
and share one `_execute_tool()` core, the per-package file-count table, and the
heaviest single modules. Read it before you create a module.

## Dependency direction and the known SCC

The measured figures — 1,701 modules / 6,066 raw internal edges, the **847-module**
raw all-import SCC, the eager graph's **zero** non-trivial SCCs, the 59-node package
projection with its one 44-package SCC, and `check_import_cycles.py`'s own
differently-scoped **821-module / 48.5%** measurement (the standing reason the
eager/deferred/`TYPE_CHECKING` split must not be "simplified" away) — are under
"Dependency direction — the measured SCC" in `references/architecture-reference.md`.

**The rules — TARGET direction you are held to in review, NOT a property the
tree has or a gate enforces.** ★ Nothing enforces the layering. `check-import-cycles`
refuses only *eager cycles*, in any direction; `scripts/check_coupling.py`
covers only geniusbot→`agent_utilities` and its hook wraps it in `|| true`. Rules
1 and 2 are measurably violated on `main` today (counts repeated in rules 1 and 2
below, re-measured 2026-09-03). Re-measure before quoting a number — one AST walk over
`git ls-files agent_utilities`, counting top-level and `from` import statements
whose first-level package differs from the importing file's, is what produced
them:

1. **Dependencies should point inward.** Contracts ← domain ← ports ←
   application ← adapters ← composition. **Violated today:** 62
   application→adapter import statements over 14 package pairs.
2. **Adapters should not call each other.** REST does not call MCP; MCP does not
   call REST. Both call the same use-case object. **Violated today:** 85
   adapter→adapter import statements over 19 package pairs, including the
   reciprocal `gateway -> mcp` (19) / `mcp -> gateway` (3).
3. **REST and MCP bind the SAME use-case object** — `_execute_tool()`. The tool
   function carries argument marshalling, never logic.
4. **Production code never imports tests, dev tooling, generated output, or
   deployment composition** (RF-ADR-003). Dev/test edges are inventoried
   separately and may not pull production upward.
5. **One capability, one owner, one implementation** (`plans/refactor/DESIGN.md` "Anti-sprawl
   invariants"). No new package/repository without a cohesive lower-level owner,
   **two or more real live consumers**, and a *measured* net reduction in cycles,
   duplicate implementations, public surface, or edges. Splitting by file count,
   scan score, or aesthetics is rejected.

The named reverse edges the program is cutting — and the intended directions that
must stay one-way — are listed with their measured edge counts under "Named
reverse edges" in `references/architecture-reference.md`. Do not add to them.

`KnowledgeGraph` (`knowledge_graph/facade.py`) is the graph facade;
`Orchestrator.execute_agent` (`orchestration/manager.py:512`) owns
agent/loop/workflow dispatch. ★ `run_agent` is **not** a method on
`Orchestrator` — it is a **module-level function**, `orchestration/agent_runner.py:700`,
which `Orchestrator.execute_agent` imports and calls. (`AGENTS.md` writes them
together as `Orchestrator.execute_agent`/`run_agent`; that shorthand is not a
class API.) Facade and orchestrator have **disjoint** authority — the facade does not dispatch agents, and the
orchestrator does not implement graph persistence. Composition binds exactly one
of each.

## The gates that actually enforce this

`.pre-commit-config.yaml` declares **99 hooks** across 14 repos. These are the
architecture / anti-sprawl subset; each names exactly what it refuses.

Each is tabulated with its stage and exactly what it refuses — which clauses are
absolute, which diff-scoped, which advisory or inert today, and the one surviving
live ratchet — in [`references/gates-reference.md`](references/gates-reference.md).
Per G2, read the row for any gate you are about to cite or trust.

**The merge gate is DIFFERENTIAL, not absolute.** `governance/merge_queue.py`
computes a base-ref baseline and blocks only on a **NEW failure not present on the
base ref**; pre-existing failures are reported and explicitly *not* blocking. It
compares at pytest **node-id** granularity (parsed from `FAILED <nodeid>` /
`ERROR <nodeid>`), caches baselines content-addressed, and — critically — **refuses
rather than allows** when a baseline cannot be produced ("never silently treated as
'no pre-existing failures'"). `main` legitimately carries debt; an absolute
standard once stranded 19 branches and rejected a branch that fixed 21 of 30
failing tests. The candidate is always gated **as merged** (`git merge-tree
--write-tree` → `git commit-tree` → throwaway detached worktree), never as it sat
on the branch.

## Before you add anything — the anti-sprawl checklist

Run all five. Any "no" that you cannot answer is a stop, not a caveat.

1. **Does an owner already exist?** Ask the KG first
   (`graph_code action=code_context`), then `scripts/find_callers.py
   <dotted.symbol>` — **not** grep. A text grep has silently missed real call
   sites three times in this program: `import x as y`, a bare `from … import x`,
   and `monkeypatch.setattr("pkg.mod.x", …)`. `find_callers.py` resolves imports,
   aliases, and attribute chains through `ast` and flags the two dynamic shapes a
   static walk can only surface heuristically.
2. **Is this a second route to an existing capability?** If REST already reaches
   it, MCP must reach the *same* `_execute_tool` action — a new handler beside it
   is the drift `guardrail-surface-parity` exists to catch. If N entrypoints must
   each be edited, the code is in the wrong layer: move it to the core
   orchestrator and let the entrypoints inherit it with **zero** per-surface code.
3. **Is a constant or contract being duplicated across a boundary the consumer
   cannot see changes through?** This is the highest-yield question in the whole
   list. Eight near-identical scope-identity builders were written in one session
   because a brief named a "worked reference" without naming where the shared form
   would live; two crates each declared the same literal kept in step only by a
   comment saying "must stay byte-identical". A constant private to its crate and
   redeclared in a test **will** drift. Export the *shared form*, not the raw
   constants.
4. **Does the new module declare an owning component?** Under RF-ADR-005 an
   architecture seam declares its identity at `architecture/component-registry.yml`
   in the owning repository. ★ **Unverified/not yet present:** AU has no
   `architecture/` directory today (`ls architecture/` → no such file). Until it
   exists, name the owning package and its layer in the module docstring beside the
   `CONCEPT:` tag, and reserve the concept id first
   (`agent-utilities --json concept reserve --id …`).
5. **Where does the weight belong?** Heavy AI/ML → `agents/data-science-mcp`.
   Finance/quant → `emerald-exchange`. Any KG compute, ANN, vector similarity, or
   graph algorithm → the Rust `epistemic-graph` engine. A new ontology class →
   **into the existing domain `.ttl`**, never a per-feature file. A new capability
   → an action on an existing service, or a declarative connector preset — almost
   never a new daemon.

## Failure patterns that are now rules

Twelve incident-derived rules, each with the recorded AU/EG failure behind it —
scanner counters as evidence, a gate's universe vs the package's, the poisoned
`uv run` and its lock rewrite, extraction and mypy, ambient commit identity,
copied reference implementations, file-partitioned lanes, dual-purpose predicates,
unverified blockers, stale hook comments, the fleet denominator — are in
[`references/failure-patterns-reference.md`](references/failure-patterns-reference.md).

## Workflow

### 1. Read the governing context

- Read every applicable `AGENTS.md` before editing.
- Inspect the owning architecture guide, specification, tests, and public entry
  points.
- Search existing concepts and implementations before adding a new abstraction —
  run *Before you add anything* above; it is the shortest path to the owner.
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
- Identify every owned consumer with `scripts/find_callers.py`, not grep — the
  three shapes that hide a caller from a text search, and why, are in *Before you
  add anything* → checklist item 1 above. Same tool, same reason; do not re-derive
  it by grepping.
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
# NEVER run `pre-commit run --all-files` bare: in a shared worktree it stashes
# the WHOLE tree and can destroy your own or another session's unstaged work
# (D-OB-12). The lease and the safe wrapper are TWO separate guards; both apply.
agent-utilities lane lease --resource precommit-all-files --operation gate -- \
  python3 scripts/safe_precommit_all_files.py
```

Never `--no-verify`, and never mask a gate to force green: `noqa`, `type:
ignore`, `nosec`, `skip`, and `xfail` appearing in a delivery diff are what a
reviewer greps for first.

★ **Judge every gate DIFFERENTIALLY, against the base ref — never against
absolute green.** `main` is legitimately red, so a pre-existing failure is not
yours to clear. Compare at the granularity the gate declares (pytest **node
ids**, not counts or files), and if the baseline cannot be produced, refuse
rather than allow-all. The mechanism, the evidence, and how `governance/
merge_queue.py` implements it are stated once, above — see **"The merge gate is
DIFFERENTIAL, not absolute"** at the end of *The gates that actually enforce
this*.

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

★ **Merging is NOT deploying** (CONCEPT:AU-OS.governance.merge-deploy-decoupling).
The fleet NFS-mounts the canonical checkout at `/au` with `PYTHONPATH=/au`, so a
merge **arms** a deploy that fires on the next unplanned restart — it does not
ship one. **Merge freely to `main`**; you **MUST** ship only by an explicit
fast-forward of `refs/heads/deployed` to a SHA the full suite has since passed.
Check with `merge-queue promotion`.

An earlier revision of this skill stated the opposite ("a merge to `main` is a
live deploy", via a `hostPath` mount over `site-packages`). Both the conclusion
and the mechanism were wrong, and following it would make you hesitate to merge
against an explicit MUST while leaving you unable to actually ship. Corrected
against `AGENTS.md` 2026-09-03.

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

The fixed build order — Rust engine `Method` in `epistemic-graph` → the
auto-discovered `engine_<domain>` MCP verb with its REST twin registered in the
same change → the wrapping domain skill — plus the three-command verification pass
(`skill_coverage`, `test_gateway_mcp_parity.py`, `gen_graphos_manifest.py`) and the
uncovered/orphan rule are in
[`references/platform-capability-reference.md`](references/platform-capability-reference.md).

## Guardrails — the G11 replacement catalogue

**This section is the expansion of G11** (*Mandatory guardrails*, above); it is not
a second list. Each entry names its replacement, because a prohibition without one
does not hold. Run `repository-manager --lane doctor --lane-path .` and it will
tell you which of them you are currently violating, with the exact remedy command.

- **Never edit the canonical checkout.** Work in the lane worktree.
- **Never use the harness's worktree-isolation tool** (`Agent(isolation:"worktree")`
  / `EnterWorktree`) on this repo. It writes `core.bare = true` into the **shared**
  `$GIT_COMMON_DIR/config` and never restores it, so every one of the 26+ linked
  worktrees then fails `git status`/`git commit` with *"this operation must be run
  in a work tree"* — invisibly. Upstream defect, closed as not-planned. Use
  `repository-manager --lane start` (or a real `git worktree add`).
- **Never `update-ref` to advance a branch.** It moves the ref without the
  worktree, and the NEXT commit there silently reverts everything in between while
  `git status` reads clean and `--is-ancestor` says yes. Use `git merge --ff-only`,
  then verify by TREE: `git cat-file -e HEAD:<path>` (see *Validate* → measure the
  merged tree).
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
- **Never `git add -A` / `git add .`.** A shared worktree routinely holds handoff
  notes, baseline markers, logs, caches, and another concern's edits. Read `git
  status --short`, then stage an explicit reviewed allowlist (`git add -- path…`,
  `git add -u -- exact/path` for deletions), then re-read `git diff --cached
  --name-status` and `git diff --cached` before committing. `*-NOTES.md`,
  scratchpads, logs, caches, test output, and branch-divergence markers are never
  product artifacts.
- **Regenerate `uv.lock` exactly once, after every `pyproject.toml` in the change
  has frozen.** Regenerating per-edit produces a lock that churns against every
  other lane and an `uv-lock --locked` failure nobody can attribute. Verify the
  lock is untouched by your test runs before you commit.
- Do not bypass failing gates or silently accept warnings.
- Do not create a second implementation for another entry point.
- Do not commit secrets, credential files, local inventories, or scratch output.
