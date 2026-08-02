# Design Document: The merge queue — continuous, serialized, tested-as-merged, and self-healing on generated-file conflicts

CONCEPT:AU-OS.governance.serialized-merge-queue ·
CONCEPT:AU-OS.governance.tiered-merge-gate ·
CONCEPT:AU-OS.governance.merged-tree-verification ·
CONCEPT:AU-OS.governance.cross-branch-duplicate-symbol ·
CONCEPT:AU-OS.governance.merge-deploy-decoupling ·
CONCEPT:AU-OS.governance.test-regression-baseline ·
CONCEPT:AU-OS.governance.merge-queue-regenerate-on-land

> Backfilled under the concept-lineage rule
> (CONCEPT:AU-OS.governance.concept-lineage-parent-doc) — this cluster's
> markers already existed (mostly in `agent_utilities/governance/merge_queue.py`
> and the pre-existing, extensively evidenced `docs/architecture/merge-queue.md`)
> but `.specify/design/**` is the only tree `has_design_doc()` scans, and
> `docs/architecture/` was never mirrored there. This is a short index into
> the seven facets, not a restatement — `docs/architecture/merge-queue.md`
> remains the canonical extended write-up (mermaid diagram, full latency
> arithmetic, the D-WD-1/D-OB-17 case studies) for the first five; this
> document adds the two facets that document does not cover.

## Decision — one continuous, serialized queue, gated cheaply against the tree AS MERGED, with the expensive suite moved off the critical path and merge decoupled from deploy

Work used to land through a bulk reconciliation gate at the end of a wave.
**`docs/architecture/merge-queue.md` §"Why this replaced bulk
reconciliation"** measures the cost this had: two independent `CandidateClaim`
classes and two `Fragment` shapes existed simultaneously because branches
were invisible to each other, ~8 deferred items were worked against a stale
`main`, and worktree count peaked at 79. The fix is not "review harder at the
end" — it is continuous merge through an **automated** adversarial review,
since automating the review is what made batching's only real value (the
review itself) available on every single merge instead of once a wave.

Seven facets of that one design, each with its own marker because each is an
independently rejectable/revertable decision:

1. **`serialized-merge-queue`** — one `reconciliation-merge` LEASE serializes
   landing; the queue itself is an APPEND-ONLY `FragmentStore` (no shared
   mutable file to clobber). `agent_utilities/governance/merge_queue.py:3`.
   Rejected alternative: long-lived branches + bulk review (measured cost
   above).

2. **`tiered-merge-gate`** — a **180 s published budget**
   (`FAST_GATE_BUDGET_SECONDS`, `merge_queue.py:52,119`) for merge-cleanliness,
   duplicate-symbol scan, contract checks, import smoke, and targeted tests;
   the **43-minute full suite runs after landing, on `main`, off the queue**.
   Rejected alternative, with the arithmetic: gating on the full suite gives
   1.4 merges/hour: ten agents merging hourly already exceed that (ρ=7, the
   queue diverges without bound). `docs/architecture/merge-queue.md`
   §"Why the full suite is deliberately after the merge".

3. **`merged-tree-verification`** — the gate never runs against a candidate's
   branch tree; it builds a trial commit purely at the object level
   (`git merge-tree --write-tree` → `git commit-tree`, no working tree, no
   ref move), materializes it into a throwaway detached worktree, gates
   there, then `git merge --ff-only` (`merge_queue.py:43,709`). Rejected
   alternative: testing on the branch as it sits — `D-OB-17` is the proof
   this is wrong: two branches individually import fine, merge with **no
   conflict markers**, and the merged tree `ImportError`s. Nothing that
   looks at a branch (its CI, its pre-commit, a diff review) can see that;
   only building and running the merged tree can.

4. **`cross-branch-duplicate-symbol`** — `duplicate_definitions()`
   (`merge_queue.py:627`) `ast`-parses each candidate's changed `.py` files,
   collects module-level `class`/`def` names **added** relative to the
   candidate's own merge base, and reports any name added by more than one
   in-flight candidate, naming every call site. Rejected alternative: trust
   git's silence — two lanes independently writing a module-level
   `CandidateClaim`/`Fragment` land in **different files**, so git merges
   both without a word and every per-branch test passes; the duplicate only
   exists in the *relation between two candidates*, which only the queue
   itself can see (`docs/architecture/merge-queue.md`
   §"The cross-branch duplicate-symbol scan").

5. **`merge-deploy-decoupling`** — `merge_queue.py:86,2470`. graph-os
   NFS-mounts the canonical `main` checkout read-only
   (`services/graph-os/k8s/graph-os.deployment.yaml`, volume `au-src`), so
   **a merge arms a deploy that fires at a moment nobody chose** (next node
   drain/eviction/OOM/reschedule/`rollout restart`) — not "merge deploys",
   worse: an unplanned, unattributed one. The fix separates `au-src` onto a
   `deployed` ref, fast-forwarded only by explicit promotion after the slow
   tier passes. Rejected alternative: leave `main` as the deploy source and
   rely on merge cadence discipline — survivable at one merge/day, not at
   continuous high-concurrency merge (`docs/architecture/merge-queue.md`
   §"Merge is not deploy").

6. **`test-regression-baseline`** — `compute_test_baseline`
   (`merge_queue.py:183,1371,1791,1811,1886`) permits a failing test id in
   the merged-tree gate run **only when that exact id already fails,
   identically, on the base ref** — never by file, module, name pattern, or
   raw count. Measured, not asserted: judging the merged tree in isolation
   against a not-always-green `main` rejected a real branch that fixed 21 of
   30 pre-existing failures in two files, because the merged tree still
   showed 9 failures — file/module-level gating would have passed that
   branch too eagerly on a DIFFERENT branch that broke 9 NEW tests in an
   already-partially-red file, since "the file was already failing" is not
   "these ids were already failing". Fail-closed on a degraded baseline read
   (timeout/crash/uncollectable), never treated as "no pre-existing
   failures". `docs/architecture/merge-queue.md`
   §"Differential (regression) gating for targeted tests" has the full cache
   design (per-`(base_sha, FILE)`, subset reuse under bisection) and the
   rejected pure-content-hash key (would reuse a baseline computed against a
   stale tree).

7. **`merge-queue-regenerate-on-land`** — `merge_queue.py:2192-2216`.
   `docs/concepts.yaml`, `README.md`, `AGENTS.md`, and `docs/project_structure.md`
   (`GENERATED_FILES`) are purely-derived from source (a `CONCEPT:` marker
   count, a concept table); with dozens of candidates converging on one base,
   nearly every OTHER candidate that also touches one of these — or merely
   adds a `CONCEPT:` marker, changing the count — conflicts on it too, even
   though there is no real disagreement, just two stale copies of a derived
   file. On detecting a conflict confined to `GENERATED_FILES`, the queue
   regenerates them from the ALREADY-MERGED source of truth (the trial tree
   itself, after the two branches' real content is combined) via the same
   `build_concepts_yaml.py` → `gen_docs.py --write` → `gen_agents_md.py`
   chain a lane runs by hand, in that order (`gen_docs.py` reads
   `docs/concepts.yaml`, so it must be regenerated first;
   `gen_agents_md.py` writes both `AGENTS.md` and `docs/project_structure.md`
   last). **The rejected alternative is named directly in the code
   comment**: "never 'pick a side' via `checkout --theirs`" — picking a side
   silently drops whichever candidate's real, non-generated content lost,
   which is exactly the failure this mechanism exists to avoid on a file
   that is never itself a source of truth. Exercised in
   `tests/unit/test_merge_queue.py:1225`.

## Risk Assessment

- **Blast Radius**: `agent_utilities/governance/merge_queue.py`,
  `agent_utilities/cli/__init__.py` (the `merge-queue` subcommands),
  `services/graph-os/k8s/graph-os.deployment.yaml` (facet 5).
- **Backward Compatible**: Yes for 1-4, 6-7 (additive gate/queue mechanics).
  Facet 5 changes graph-os's source-of-truth NFS path and is stated in
  `docs/architecture/merge-queue.md` itself as **not yet flipped in
  production** ("a finished fix currently sits blocked on an operator
  decision: merging it would also ship it").
- **Known weak points**, stated directly in `docs/architecture/merge-queue.md`
  §"What is now impossible, rather than discouraged": the queue can still be
  bypassed by a hand `git merge` in the canonical checkout (closing it needs
  `lane-guard` to require a live `reconciliation-merge` lease); duplicate
  symbols are *detected*, not *prevented*, at authoring time; the fast tier
  is a sample over changed paths, not a proof — a regression in an untouched
  module is caught only by the slow tier after landing, which is the
  deliberate trade for a sub-3-minute queue.
