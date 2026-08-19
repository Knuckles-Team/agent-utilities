"""Continuous merge into ``main`` through one serialized, tiered merge queue.

CONCEPT:AU-OS.governance.serialized-merge-queue

**What this replaces, and why.** Work used to accumulate on long-lived branches and
land in a bulk reconciliation gate at the end of a wave. The batching was never the
value — the *adversarial review* inside it was; batching was how we afforded that
review. Automating the review makes continuous merge strictly better, because
long-lived branches have a measured cost that batching cannot pay back:

* **Duplicates that existed only because branches were invisible to each other** —
  two ``CandidateClaim`` classes, two ``Fragment`` shapes, two independent fixes for
  one tenancy bug. Every lane surveyed correctly; each was right *at the moment it
  looked*. The premise each surveyed against was `main`, and `main` was stale.
* **~8 deferred items worked against a `main` that no longer existed** — a premise
  rots in proportion to how long a branch lives.
* **79 worktrees at peak**, with a prune bug whose blast radius scaled with the
  number of long-lived branches.

But naive continuous merge loses three things the bulk review caught, so this module
reproduces each mechanically rather than dropping it:

* an add/add conflict where **git silently dropped one of two new node classes** →
  :func:`duplicate_definitions`, a cross-branch duplicate-symbol scan run over every
  candidate in flight *together*;
* ``D-OB-17``: two branches that **git auto-merges cleanly into a tree that
  ``ImportError``s** — no conflict markers, broken result → :func:`run_fast_gate`
  tests the candidate **as merged** (:func:`trial_merge` → :func:`materialized`),
  never as it sat on its branch;
* five collisions the gate **combined** rather than letting git pick a side → a
  rejected candidate is handed back to its lane *with the colliding symbols named*,
  which is the input a human or agent needs to combine them.

**Arbitration reuses what exists — no second mechanism.** Serialization is the
existing ``reconciliation-merge`` LEASE (``lane_resources.yaml``); the queue itself
is an APPEND-ONLY :class:`~agent_utilities.governance.lanes.FragmentStore` (one
fragment per lane, folded to one view) living in the repository's shared
``--git-common-dir``, which no checkout resets and no merge can conflict; per-lane
scratch and pytest basetemp come from :func:`~agent_utilities.governance.lanes.partitioned_paths`.
Adding the queue added **zero** new arbitration classes.

**The canonical working tree is never a merge arena.**
CONCEPT:AU-OS.governance.merged-tree-verification — the whole merge is computed as
*objects* (``git merge-tree --write-tree`` → ``git commit-tree``), verified in a
throwaway detached worktree, and only then does ``main`` move — by
**fast-forward only**. There is no code path here that resolves a conflict in the
canonical checkout, and none that leaves a detached HEAD carrying commits (the exact
incident that motivated the ``reconciliation-merge`` lease: 26 commits stranded on a
detached HEAD with no ref, after the same shape had already orphaned an earlier
correct resolution).

**The gate must be cheaper than bypassing it.** CONCEPT:AU-OS.governance.tiered-merge-gate.
A gate whose cost exceeds its value gets bypassed — that is this codebase's most
repeated failure (``D-OP-4``: a pre-commit chain too slow to run, so work went
uncommitted; ``D-KCI-6``: a generator whose safe output looked destructive, so nobody
ran it). So checks are tiered by price and the **full suite is deliberately outside
the queue**:

===========  ===================================================  ==================
Tier         Checks                                               Where
===========  ===================================================  ==================
**fast**     merge-cleanliness (~25 ms), duplicate-symbol scan,    inside the lease,
             repo-invariant contract checks (~1.7 s), import       per batch
             smoke over changed modules, targeted tests over
             changed paths
**slow**     the full unit suite, guardrail gates, live matrix     after landing, on
                                                                  ``main``, batched
===========  ===================================================  ==================

**Zero conflicts is not a safety property.** It is the *expected* signal for a whole
defect class, not evidence of one. :func:`run_contract_checks` exists because a
branch that forked after a fix and then reverted it merges **perfectly cleanly** and
lands the revert (reproduced in a controlled repository; see
``docs/architecture/merge-queue.md``). Git is answering "did two people edit the same
lines" — a question about *text*, asked of the branch — and no answer to it is an
answer about whether the merged tree still upholds an invariant. Only running the
invariant against the merged tree is.

Throughput comes from **optimistic batching** (:func:`integrate_batch`): N candidates
are merged into one rolling trial commit, gated **once**, and landed together; a
failing batch is **bisected**, so a bad candidate costs ``log2(N)`` extra gate runs
rather than serializing every candidate behind a full gate. This is the industry
merge-train pattern, and it is what makes 100 concurrent lanes arithmetically
possible — see ``docs/architecture/merge-queue.md`` for the budget derivation.

**Merge is not deploy.** CONCEPT:AU-OS.governance.merge-deploy-decoupling — see
:func:`promotion_state`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_utilities.governance.lanes import (
    FragmentStore,
    LaneArbitrationError,
    LaneScope,
    guarded_tree_mutation,
    hold_lease,
    lane_scope,
    partitioned_paths,
    tree_has_uncommitted_work,
)

# ---------------------------------------------------------------------------
# The latency budget this design is built to (CONCEPT:AU-OS.governance.tiered-merge-gate)
#
# These are not tuning knobs (see *Configuration discipline*) — they are the
# published contract a lane can rely on when deciding whether to wait for the
# queue or bypass it, so they are named constants with the reasoning attached.
# ---------------------------------------------------------------------------

#: Wall-clock a lane should expect between "my candidate is at the head of the
#: queue" and "it landed or was rejected with a reason". Everything inside the
#: lease is sized to fit here; anything that cannot is Tier-slow by definition.
FAST_GATE_BUDGET_SECONDS = 180

#: Hard ceiling on the targeted-test step **against the merged tree** —
#: candidate-facing, always paid fresh. Exceeding it is not a failure — it is a
#: *deferral* to the post-merge suite, reported as such, because a gate that hangs
#: is a gate that gets bypassed. D-MW-10: the base-ref comparison run this step's
#: verdict depends on has its OWN, larger budget — see
#: BASELINE_TEST_BUDGET_SECONDS below — because it is not on this same latency
#: path once its per-file cache is warm.
TARGETED_TEST_BUDGET_SECONDS = 120

#: Ceiling on the import-smoke step (it imports only changed modules).
IMPORT_SMOKE_BUDGET_SECONDS = 60

#: Repo-invariant contract checks, **discovered in the merged tree** rather than
#: enumerated here. Adding a new invariant is a new script, not a change to this
#: module — and because discovery reads the *merged* tree, a candidate that adds a
#: contract has that contract enforced against itself in the same gate run.
CONTRACT_CHECK_GLOB = "scripts/security/check_*.py"

#: Ceiling for the whole contract-check step **against the merged tree** — the
#: candidate-facing, always-fresh half of the check. Measured cost of the four
#: live scripts is ~1-9 s combined (see docs/architecture/merge-queue.md), so this
#: is still comfortable headroom for the (now 8-script) discovered set, not a
#: tuning knob.
CONTRACT_CHECK_BUDGET_SECONDS = 60

#: Ceiling for computing a contract check's OUTPUT on *base_ref* (D-MW-9's
#: differential contract gating, mirroring compute_test_baseline). Separate from
#: CONTRACT_CHECK_BUDGET_SECONDS and generous, on the same reasoning as
#: BASELINE_TEST_BUDGET_SECONDS below: this run is cached per (base_sha, script)
#: and materializes a whole base tree on a cold miss, so it is not on every
#: candidate's critical path once warm, and contending with other lanes for the
#: shared epistemic-graph daemon makes it slower than its merged-tree twin even
#: at the same selection size.
CONTRACT_BASELINE_BUDGET_SECONDS = 180

#: Above this many selected test files the targeted selection has stopped being
#: "targeted" and is just a slow full run wearing a costume. Defer to Tier-slow.
MAX_TARGETED_TEST_FILES = 40

#: Ceiling for computing (the not-yet-cached portion of) the base-ref test
#: baseline. D-MW-10: this used to share TARGETED_TEST_BUDGET_SECONDS (the
#: merged-tree run's budget, above) and started REFUSING landable branches once
#: ~8 concurrent lanes began contending for the shared epistemic-graph daemon —
#: the base run is no smaller than the merged run it is being compared against,
#: so sharing one budget between two runs with different contention profiles was
#: always going to starve one of them. Kept separate and larger (not shared with
#: CONTRACT_BASELINE_BUDGET_SECONDS above — different subprocess, different cost
#: shape) because, per the per-file cache below, this ceiling is paid at most
#: once per (base_sha, file) — most calls after the first pay nothing.
BASELINE_TEST_BUDGET_SECONDS = 240

#: Where the shared, content-addressed test-baseline cache lives inside the
#: arbitration dir (CONCEPT:AU-OS.governance.test-regression-baseline). D-MW-10:
#: keyed by ``(base commit sha, interpreter)`` ONLY — one entry per file, not one
#: entry per whole selection — so a selection that overlaps a PREVIOUSLY-baselined
#: selection at the same base_sha (the common case during bisection, and across
#: batches that land close together) reuses every already-known file and pays
#: only for the files that are genuinely new to this base_sha. A selection that
#: is a pure subset of an already-baselined selection costs nothing at all: no
#: subprocess, no ``materialized()`` worktree checkout.
BASELINE_CACHE_DIRNAME = "test-baseline-cache"

#: Where the discovered-contract-check output baseline lives — same shared
#: arbitration dir, same per-(base_sha, script) file-level granularity as the
#: test baseline above (D-MW-9).
CONTRACT_BASELINE_CACHE_DIRNAME = "contract-baseline-cache"

#: D-MW-9: contract checks are independent, read-only scripts run against the
#: SAME tree — measured SEQUENTIAL cost for the discovered ``scripts/security/
#: check_*.py`` set (several AST-walking all of agent_utilities/ or tests/,
#: including the fast-tier forwarders that subprocess out to the slower
#: canonical scripts still kept at bare ``scripts/`` — see
#: ``scripts/security/_fast_tier_forward.py``) is dominated by the slowest
#: single script (~30s measured, the ContextCompiler boundary scan). Run
#: concurrently instead — subprocess.run releases the GIL while waiting on the
#: child — so wall time tracks that slowest script rather than the sum. Not a
#: tuning knob so much as "more than the discovered set will realistically
#: ever be at once."
CONTRACT_CHECK_MAX_WORKERS = 8

#: Candidates merged into one trial commit and gated together. Failure bisects, so
#: a bad candidate costs ceil(log2(N)) extra gate runs, not N.
DEFAULT_BATCH_SIZE = 8

#: The LEASE that serializes this queue. Registered in ``lane_resources.yaml`` —
#: deliberately the SAME resource the old bulk reconciliation gate held, so the two
#: can never run concurrently during a transition and no second arbiter exists.
MERGE_LEASE = "reconciliation-merge"

#: D-ORC-20: how long the OLDEST queued candidate may sit with nothing draining
#: it before :func:`queue_report` calls that out. ``enqueue`` returns
#: ``state: queued``, which reads exactly like "handed off" — a lane can
#: reasonably believe the queue's own runner will pick it up. It does not:
#: the only driver is the CLI verb ``agent-utilities merge-queue run``,
#: invoked by hand or by an external scheduler (there is no daemon/cron/timer
#: wired to it in this repo). A candidate that has waited longer than one
#: fast-gate budget (:data:`FAST_GATE_BUDGET_SECONDS`) past its enqueue time
#: has almost certainly not been picked up by a run that happened to be in
#: flight — it is waiting on someone to invoke ``merge-queue run``. Generous
#: on purpose: this warns an operator, it never blocks anything.
STALE_QUEUE_THRESHOLD_SECONDS = FAST_GATE_BUDGET_SECONDS * 5

#: Where queue fragments live inside the shared, unversioned arbitration dir.
QUEUE_DIRNAME = "merge-queue"

QUEUED = "queued"
LANDED = "landed"
REJECTED = "rejected"
WITHDRAWN = "withdrawn"


class MergeQueueError(LaneArbitrationError):
    """A merge-queue operation refused, with the reason a caller must act on."""


# ---------------------------------------------------------------------------
# git plumbing — exit codes are answers here, not malfunctions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GitResult:
    """One git invocation's exit code and streams, with no interpretation applied."""

    code: int
    out: str
    err: str

    @property
    def ok(self) -> bool:
        return self.code == 0


def _run_git(args: list[str], cwd: Path, *, timeout: int = 300) -> GitResult:
    """Run git and hand back the exit code.

    Unlike ``lanes._git`` this never raises on a non-zero exit: ``merge-tree``
    answers "there are conflicts" with exit 1 and ``branch -d`` answers "those
    commits would be orphaned" with exit 1. Turning either into an exception would
    hide the answer we are asking for.
    """
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return GitResult(proc.returncode, proc.stdout.strip(), proc.stderr.strip())


def _require_git(args: list[str], cwd: Path) -> str:
    res = _run_git(args, cwd)
    if not res.ok:
        raise MergeQueueError(f"git {' '.join(args)} failed in {cwd}: {res.err}")
    return res.out


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# The queue — APPEND-ONLY fragments, one per lane, in the shared arbitration dir
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """One branch offered for landing, as its own lane recorded it."""

    branch: str
    lane: str
    base: str = "main"
    worktree: str = ""
    enqueued_at: str = ""
    state: str = QUEUED
    reason: str = ""
    #: D-F6-1: WHEN this specific record was appended — distinct from
    #: ``enqueued_at``, which is the original enqueue time and stays fixed
    #: across every later state transition (see ``_record_state``). Used by
    #: :func:`_resolve_latest_candidate_record` to pick the chronologically
    #: LATEST record for an id across every lane's fragment, instead of
    #: :meth:`FragmentStore.fold`'s default "last in lane-name-sorted
    #: iteration order" — which silently prefers whichever lane's name
    #: sorts alphabetically last, not whichever record was actually written
    #: most recently.
    recorded_at: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Candidate:
        return cls(
            branch=str(record.get("id", "")),
            lane=str(record.get("lane", "")),
            base=str(record.get("base", "main")),
            worktree=str(record.get("worktree", "")),
            enqueued_at=str(record.get("enqueued_at", "")),
            state=str(record.get("state", QUEUED)),
            reason=str(record.get("reason", "")),
            recorded_at=str(record.get("recorded_at", "")),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.branch,
            "lane": self.lane,
            "base": self.base,
            "worktree": self.worktree,
            "enqueued_at": self.enqueued_at,
            "state": self.state,
            "reason": self.reason,
            "recorded_at": self.recorded_at,
        }


def queue_store(path: Path | str | None = None) -> FragmentStore:
    """The append-only record set backing the queue for this repository.

    Lives under the repo's shared ``--git-common-dir``: identical from every
    worktree, never rewritten by a checkout/reset/merge, and not version-controlled
    — so two lanes enqueueing at the same instant write two different files and
    neither can clobber the other, exactly as the APPEND-ONLY class prescribes.
    """
    scope = lane_scope(path)
    return FragmentStore(root=scope.arbitration_dir / QUEUE_DIRNAME, key="id")


def enqueue(
    branch: str = "",
    *,
    base: str = "main",
    worktree: str | Path | None = None,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Offer *branch* (default: this lane's own branch) for landing on *base*.

    Enqueueing is deliberately cheap and non-blocking — it appends one record and
    returns. Nothing is verified here: verification happens once, at the head of
    the queue, against the ``main`` that actually exists then. Verifying at enqueue
    time would re-create the stale-premise problem this whole design exists to kill.

    D-ORC-20: ``state: queued`` in the return value reads like completion — a
    lane can reasonably (and wrongly) read it as "handed off, the queue's own
    runner will land it." There is no such runner: the only driver is
    ``agent-utilities merge-queue run``, invoked by hand or an external
    scheduler. The returned ``note`` says so explicitly, on every call, so a
    caller that never reads this docstring still sees it.
    """
    scope = lane_scope(path)
    branch = branch or _require_git(["rev-parse", "--abbrev-ref", "HEAD"], scope.tree)
    if branch in {"HEAD", base}:
        raise MergeQueueError(
            f"refusing to enqueue {branch!r}: a candidate must be a named branch "
            f"that is not {base!r} (detached HEAD or the base itself is never one)"
        )
    now = _now()
    candidate = Candidate(
        branch=branch,
        lane=scope.lane,
        base=base,
        worktree=str(Path(worktree).resolve()) if worktree else str(scope.tree),
        enqueued_at=now,
        state=QUEUED,
        recorded_at=now,
    )
    store = queue_store(scope.tree)
    store.append(candidate.to_record(), lane=scope.lane)
    depth = len(queued(scope.tree))
    return {
        "enqueued": True,
        **candidate.to_record(),
        "note": (
            "queued != landed. D-MQR-7: this note used to say nothing drives "
            "the queue automatically (D-ORC-20) -- that was true when D-ORC-20 "
            "was open and is false now. `merge-queue-runner.timer` drains this "
            "queue automatically (~every 5 minutes, across agent-packages/, "
            "services/, images/, and the infra roots): it gates the candidate "
            "DIFFERENTIALLY against the base, lands it, and prunes the "
            "worktree/branch. You do not need to run anything yourself — "
            "check `agent-utilities merge-queue status` to see whether it has "
            "drained. Do NOT hand-drain with `merge-queue run`: concurrent "
            "lanes share one reconciliation-merge lease and a manual drain "
            f"races the scheduler for no benefit before {branch!r} lands on "
            f"{base!r}."
        ),
        "queue_depth": depth,
    }


def _record_state(
    candidate: Candidate, state: str, reason: str, path: Path | str
) -> None:
    """Supersede a candidate's record with its terminal state.

    A *new* append, never an edit: the fold collapses records sharing an ``id`` to
    the CHRONOLOGICALLY LATEST one written (D-F6-1 — see
    :func:`_resolve_latest_candidate_record`), so a landed/rejected record
    supersedes the queued one without any writer ever rewriting a file another
    lane also writes — including when the terminal state is recorded from a
    DIFFERENT lane's fragment than the one that originally enqueued it (the
    common shape: enqueue from a candidate's own worktree, land from the
    canonical checkout).

    ``recorded_at`` is stamped FRESH here — unlike ``enqueued_at``, which is
    preserved from the original record so the candidate's true enqueue time
    survives every later transition — because it is specifically WHEN THIS
    record was written, the field the fold resolves ties on.
    """
    scope = lane_scope(path)
    store = queue_store(scope.tree)
    updated = Candidate(
        branch=candidate.branch,
        lane=candidate.lane,
        base=candidate.base,
        worktree=candidate.worktree,
        enqueued_at=candidate.enqueued_at,
        state=state,
        reason=reason,
        recorded_at=_now(),
    )
    store.append(updated.to_record(), lane=scope.lane)


def withdraw(branch: str, *, reason: str = "", path: Path | str | None = None) -> dict:
    """Pull a candidate back out of the queue (its lane changed its mind)."""
    scope = lane_scope(path)
    for candidate in _all_candidates(scope.tree):
        if candidate.branch == branch:
            _record_state(candidate, WITHDRAWN, reason, scope.tree)
            return {"withdrawn": True, "branch": branch, "reason": reason}
    raise MergeQueueError(f"{branch!r} is not in the queue")


def _resolve_latest_candidate_record(
    group: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pick the chronologically LATEST record in *group* by ``recorded_at``
    (D-F6-1), not :meth:`FragmentStore.fold`'s default ``group[-1]``.

    ``fold()`` groups every lane's record for one id together in
    ``self.lanes()``'s (alphabetically SORTED) iteration order, then hands
    the whole group here. ``group[-1]`` alone would therefore silently
    prefer whichever LANE NAME sorts alphabetically last among every lane
    that ever wrote a record for this id — completely unrelated to which
    record was actually written most recently. That is exactly backwards
    when a candidate is enqueued from its own worktree (lane e.g.
    ``lane-foo``) and its terminal state is later recorded from the
    canonical checkout (lane ``canonical``): ``"canonical" < "lane-foo"``
    alphabetically, so the OLDER queued record from ``lane-foo`` would win
    over the newer terminal one from ``canonical`` — reproduced live as a
    candidate that had genuinely landed still reporting ``queued`` forever.

    ISO-8601 ``recorded_at`` timestamps sort correctly as plain strings, so
    ``max(..., key=...)`` is enough. Falls back to ``group[-1]`` — the OLD
    behavior — only when every candidate in a tie is missing/blank
    ``recorded_at`` (a record written before this field existed), so
    existing fragment files degrade gracefully rather than needing a
    migration.
    """
    with_timestamp = [r for r in group if str(r.get("recorded_at", "")).strip()]
    if not with_timestamp:
        return group[-1]
    return max(with_timestamp, key=lambda r: str(r.get("recorded_at", "")))


def _all_candidates(path: Path | str | None = None) -> list[Candidate]:
    return [
        Candidate.from_record(r)
        for r in queue_store(path).fold(resolve=_resolve_latest_candidate_record)
    ]


def queued(path: Path | str | None = None) -> list[Candidate]:
    """Every still-pending candidate, oldest first (FIFO, and fair by construction)."""
    pending = [c for c in _all_candidates(path) if c.state == QUEUED]
    return sorted(pending, key=lambda c: (c.enqueued_at, c.branch))


def _candidate_age_seconds(candidate: Candidate, *, now: datetime) -> float | None:
    """Seconds since *candidate* was enqueued, or ``None`` if unparseable."""
    try:
        enqueued = datetime.fromisoformat(candidate.enqueued_at)
    except ValueError:
        return None
    return (now - enqueued).total_seconds()


def queue_report(path: Path | str | None = None) -> dict[str, Any]:
    """Everything an operator or a waiting lane needs: depth, order, terminal outcomes.

    ``stale_queue_warning`` fires when the OLDEST queued candidate has waited
    past :data:`STALE_QUEUE_THRESHOLD_SECONDS`. D-MQR-7: this used to claim the
    queue "has no automatic runner" (true under D-ORC-20, false since
    ``merge-queue-runner.timer`` started draining every ~5 minutes) — an age
    over the threshold now means either the runner is stuck/erroring on this
    repo (check ``reports/lane-logs/merge-queue-runner.log`` for an ERROR line
    naming it — e.g. a corrupted canonical checkout, D-MQR-5/6) or every batch
    that reached this candidate was refused; it is not evidence that nothing is
    watching the queue.
    """
    everything = _all_candidates(path)
    pending = queued(path)
    now = datetime.now(UTC)
    stale_warning = None
    if pending:
        oldest = pending[0]
        age = _candidate_age_seconds(oldest, now=now)
        if age is not None and age > STALE_QUEUE_THRESHOLD_SECONDS:
            stale_warning = (
                f"{oldest.branch!r} has been queued for {age:.0f}s "
                f"(> {STALE_QUEUE_THRESHOLD_SECONDS}s threshold). "
                "merge-queue-runner.timer normally drains this queue every ~5 "
                "minutes -- this age means either the runner is stuck/erroring "
                "on this repo (check reports/lane-logs/merge-queue-runner.log "
                "for an ERROR line naming it) or every batch that reached this "
                "candidate was refused. It is not evidence that nothing is "
                "watching the queue; if it needs manual attention, `agent-"
                "utilities merge-queue run` will drain it directly."
            )
    return {
        "depth": len(pending),
        "queued": [c.to_record() for c in pending],
        "recent": [c.to_record() for c in everything if c.state in {LANDED, REJECTED}][
            -20:
        ],
        "budget_seconds": FAST_GATE_BUDGET_SECONDS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "lease": MERGE_LEASE,
        "stale_queue_warning": stale_warning,
        "checked_at": _now(),
    }


# ---------------------------------------------------------------------------
# Tier fast — merge cleanliness, computed entirely as objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrialMerge:
    """The result of merging two commits **without touching any working tree**.

    ``git merge-tree --write-tree`` does the real three-way merge (it finds the
    merge base itself), writes the resulting tree into the object database, and
    exits 1 with the conflicted paths when it cannot. Nothing is checked out, no
    ref moves, no index is touched — which is why this is safe to run against the
    canonical checkout's ``main`` while other lanes are mid-work, and why it costs
    milliseconds rather than a checkout.
    """

    ok: bool
    tree: str
    conflicts: list[str] = field(default_factory=list)
    detail: str = ""


def trial_merge(repo: Path, base_ref: str, branch: str) -> TrialMerge:
    """Merge *branch* into *base_ref* as objects only; report conflicts, never write a ref."""
    res = _run_git(
        ["merge-tree", "--write-tree", "--name-only", base_ref, branch], repo
    )
    if res.ok:
        return TrialMerge(ok=True, tree=res.out.splitlines()[0].strip())
    if res.code != 1:
        raise MergeQueueError(
            f"git merge-tree failed against {base_ref}..{branch}: {res.err or res.out}"
        )
    lines = res.out.splitlines()
    tree = lines[0].strip() if lines else ""
    # Sections are blank-line separated: OID, conflicted paths, informational text.
    conflicts = [ln.strip() for ln in lines[1:] if ln.strip()]
    blank = next((i for i, ln in enumerate(lines[1:], 1) if not ln.strip()), None)
    if blank is not None:
        conflicts = [ln.strip() for ln in lines[1:blank] if ln.strip()]
    return TrialMerge(ok=False, tree=tree, conflicts=conflicts, detail=res.out)


def _commit_trial(repo: Path, tree: str, parents: list[str], message: str) -> str:
    """Seal a trial tree into a commit object. Creates an object; moves no ref."""
    args = ["commit-tree", tree]
    for parent in parents:
        args += ["-p", parent]
    args += ["-m", message]
    return _require_git(args, repo)


def changed_paths(repo: Path, base_ref: str, ref: str) -> list[str]:
    """Paths *ref* changes relative to its merge base with *base_ref*.

    Merge-base rather than a plain diff: a branch that is merely behind ``main``
    has not "changed" the files ``main`` moved underneath it, and treating those as
    its own changes would balloon the targeted-test selection with work the branch
    never did.
    """
    merge_base = _require_git(["merge-base", base_ref, ref], repo)
    out = _require_git(["diff", "--name-only", f"{merge_base}..{ref}"], repo)
    return [line for line in out.splitlines() if line]


# ---------------------------------------------------------------------------
# Tier fast — the cross-branch duplicate-symbol scan
# CONCEPT:AU-OS.governance.cross-branch-duplicate-symbol
# ---------------------------------------------------------------------------
def _definitions_in(source: str, path: str) -> dict[str, str]:
    """Top-level class/function names defined in *source*, mapped to ``path:line``.

    Deliberately module-level only. A method named ``run`` on two different classes
    is not a duplicate; a **module-level** ``class CandidateClaim`` defined twice
    across two in-flight branches is exactly the failure this exists to catch, and
    a name-level check finds it whether the branches touched the same file or not —
    which the git-level check structurally cannot.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # A file that does not parse is a real defect, but it is the *gate's* job to
        # report it (import smoke / targeted tests do, with a usable traceback).
        # Silently contributing zero symbols here would be a fail-open read.
        return {}
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            found[node.name] = f"{path}:{node.lineno}"
    return found


def _blob(repo: Path, ref: str, path: str) -> str | None:
    """File content at *ref*, or ``None`` when it does not exist there."""
    res = _run_git(["show", f"{ref}:{path}"], repo)
    return res.out if res.ok else None


def added_definitions(repo: Path, base_ref: str, ref: str) -> dict[str, str]:
    """Module-level names *ref* **adds** that its merge base with *base_ref* lacks.

    "Adds" is the operative word: a name that already exists on the base is shared
    ancestry, not a collision, and reporting it would bury the real signal.
    """
    merge_base = _require_git(["merge-base", base_ref, ref], repo)
    added: dict[str, str] = {}
    for path in changed_paths(repo, base_ref, ref):
        if not path.endswith(".py"):
            continue
        after = _blob(repo, ref, path)
        if after is None:  # deleted on this branch
            continue
        before = _blob(repo, merge_base, path) or ""
        new_names = _definitions_in(after, path)
        old_names = set(_definitions_in(before, path)) if before else set()
        for name, where in new_names.items():
            if name not in old_names:
                added[name] = where
    return added


def duplicate_definitions(
    repo: Path, base_ref: str, refs: Iterable[str]
) -> list[dict[str, Any]]:
    """Module-level names **two different candidates each add**.

    This is the check that makes the queue safer than the bulk gate rather than
    merely faster. Two lanes that never saw each other's branch each wrote a
    ``CandidateClaim``; git merged both without a word, because they were in
    different files. Here they are compared *before* landing, against every other
    candidate in flight — the one moment both are visible at once.
    """
    per_ref = {ref: added_definitions(repo, base_ref, ref) for ref in refs}
    seen: dict[str, list[tuple[str, str]]] = {}
    for ref, names in per_ref.items():
        for name, where in names.items():
            seen.setdefault(name, []).append((ref, where))
    return [
        {
            "symbol": name,
            "added_by": [{"branch": ref, "at": where} for ref, where in sorted(sites)],
        }
        for name, sites in sorted(seen.items())
        if len(sites) > 1
    ]


# ---------------------------------------------------------------------------
# Tier fast — verify the candidate AS MERGED, in a throwaway tree
# CONCEPT:AU-OS.governance.merged-tree-verification
# ---------------------------------------------------------------------------
@contextmanager
def materialized(repo: Path, commit: str, *, scope: LaneScope) -> Iterator[Path]:
    """Check *commit* out into a disposable detached worktree, then remove it.

    ~0.5 s for this repository, which is what makes "test it as merged" affordable
    enough to sit inside the gate. It lives under this lane's **partitioned**
    scratch dir, so two queue runners (or a queue runner and any other lane) can
    never materialize into the same path.
    """
    root = partitioned_paths(scope.tree).scratch_dir / "merge-queue-verify"
    root.mkdir(parents=True, exist_ok=True)
    target = root / commit[:12]
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    _require_git(["worktree", "add", "--detach", str(target), commit], repo)
    try:
        yield target
    finally:
        _run_git(["worktree", "remove", "--force", str(target)], repo)
        shutil.rmtree(target, ignore_errors=True)
        _run_git(["worktree", "prune"], repo)


def _package_roots(tree: Path) -> list[str]:
    return sorted(
        p.name for p in tree.iterdir() if p.is_dir() and (p / "__init__.py").is_file()
    )


def _module_names(tree: Path, paths: Iterable[str]) -> list[str]:
    """Importable module names for changed ``.py`` files inside a package root."""
    roots = _package_roots(tree)
    names: list[str] = []
    for path in paths:
        if not path.endswith(".py") or path.startswith("tests/"):
            continue
        parts = Path(path).parts
        if not parts or parts[0] not in roots:
            continue
        if not (tree / path).is_file():  # deleted by the merge
            continue
        stem = list(parts)
        stem[-1] = Path(parts[-1]).stem
        if stem[-1] == "__init__":
            stem = stem[:-1]
        names.append(".".join(stem))
    return sorted(set(names))


def _interpreter(tree: Path) -> str:
    """The interpreter a verdict from this gate is a statement *about*.

    A green/red result only ever describes the interpreter that produced it, so the
    gate resolves one explicitly (the repo's own ``.venv`` first) and **reports it**
    in every check result rather than inheriting whatever ``python3`` happened to be
    on PATH. That ambient inheritance produced ~80 false "environment-blocked"
    verdicts in one day here.
    """
    for candidate in (
        lane_scope(tree).main_tree / ".venv" / "bin" / "python",
        tree / ".venv" / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)
    import sys

    return sys.executable


def _venv_signature(interpreter: str) -> str:
    """A cheap content signature of *interpreter*'s installed package set.

    D-MQD-1: the baseline caches (:func:`_baseline_cache_path` /
    :func:`_contract_baseline_cache_path`) were keyed on ``(base_sha,
    interpreter)`` where ``interpreter`` is a bare PATH STRING. If the venv
    at that path is rebuilt IN PLACE with different package versions while
    ``base_sha`` stays the same (the path itself doesn't change), the old key
    still resolves and the cache would silently serve a failing-id set
    computed against packages that no longer exist.

    Folding this signature into the key busts the cache on exactly that
    event: it lists every ``*.dist-info``/``*.egg-info`` directory name
    (which encodes ``name-version``) under the interpreter's
    ``site-packages``, sorted and hashed — one directory listing, no
    subprocess, no import of the target environment's own packages. Falls
    back to the interpreter file's own mtime+size when the standard venv
    layout isn't found (a non-standard interpreter, e.g. bare
    ``sys.executable`` on a system without a discoverable site-packages) —
    still busts the cache on a full venv recreation (a fresh venv/binary
    always changes mtime), just not on an in-place `pip install` into that
    unusual layout. Never raises: an unresolvable signature degrades to
    ``"unknown"`` (busts every cache uniformly) rather than failing the gate.
    """
    try:
        venv_root = Path(interpreter).resolve().parent.parent
    except OSError:
        return "unknown"
    dist_infos: list[str] = []
    for site_packages in (
        *sorted(venv_root.glob("lib/python*/site-packages")),
        *sorted(venv_root.glob("Lib/site-packages")),  # Windows layout
    ):
        dist_infos.extend(p.name for p in site_packages.glob("*.dist-info"))
        dist_infos.extend(p.name for p in site_packages.glob("*.egg-info"))
    if dist_infos:
        return hashlib.sha256(
            "\n".join(sorted(dist_infos)).encode("utf-8")
        ).hexdigest()[:16]
    try:
        stat = Path(interpreter).stat()
    except OSError:
        return "unknown"
    return hashlib.sha256(f"{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[
        :16
    ]


def _scripts_dir_signature(scripts_dir: Path) -> str:
    """Content digest of every file under *scripts_dir* — the CANDIDATE's
    ``scripts/`` tree, i.e. the instrument :func:`compute_contract_baseline`
    is about to run against the base-ref tree.

    BUG-176: the contract-baseline cache was keyed on ``(base_sha,
    interpreter)`` alone, on the same reasoning :func:`_venv_signature`
    already applies one level down — but a *third* thing can change the
    baseline's correct value without changing either of those: the CONTENT
    of the check scripts themselves. Two candidates can share a
    ``base_sha`` while one of them repairs a ``check_*.py`` (or a shared
    helper it imports, e.g. ``scripts/_git_scan.py``) — the earlier
    candidate's cached baseline was computed with the OLD script and would
    be silently served to the later one, reintroducing exactly the defect
    this fix closes (a repaired check reads its own repair as a
    regression). Folding a digest of the whole ``scripts/`` tree into the
    cache key means a script edit anywhere in that tree — not just the
    discovered ``check_*.py`` files themselves, since several forward to a
    canonical target outside ``scripts/security/`` (e.g.
    ``check_current_only_contract_gate.py`` → ``scripts/check_current_only_
    contract.py``) or import a shared helper (``scripts/_git_scan.py``) —
    busts the cache. ``__pycache__`` is excluded: compiled bytecode is
    derived, not source, and its mtime-driven churn would defeat caching
    for no correctness benefit. Never raises: an unreadable file is skipped
    rather than failing the gate (mirrors :func:`_venv_signature`'s
    never-raises contract).
    """
    digest = hashlib.sha256()
    if not scripts_dir.is_dir():
        return "absent"
    for path in sorted(scripts_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(scripts_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
        digest.update(b"\0")
    return digest.hexdigest()


def select_tests(tree: Path, paths: Iterable[str]) -> list[str]:
    """Test files worth running for *paths* — targeted, and honest when it can't be.

    A changed test runs itself; a changed source file runs the tests named after it
    anywhere under ``tests/``. Above :data:`MAX_TARGETED_TEST_FILES` the selection
    has stopped being targeted, and the caller defers to the post-merge suite rather
    than pretending a slow run is a fast one.
    """
    selected: set[str] = set()
    tests_root = tree / "tests"
    for path in paths:
        if path.startswith("tests/") and path.endswith(".py"):
            if (tree / path).is_file():
                selected.add(path)
            continue
        if not path.endswith(".py"):
            continue
        stem = Path(path).stem
        if stem in {"__init__", "__main__"}:
            continue
        if tests_root.is_dir():
            for match in tests_root.rglob(f"test_{stem}*.py"):
                selected.add(str(match.relative_to(tree)))
    return sorted(selected)


@dataclass(frozen=True)
class Check:
    """One gate check's verdict, with the evidence a rejected lane needs to act."""

    name: str
    ok: bool
    seconds: float
    detail: str = ""
    deferred: bool = False


@dataclass(frozen=True)
class GateResult:
    """The fast tier's verdict over one trial commit."""

    ok: bool
    checks: list[Check]
    interpreter: str
    seconds: float

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "seconds": round(self.seconds, 2),
            "interpreter": self.interpreter,
            "checks": [
                {
                    "name": c.name,
                    "ok": c.ok,
                    "seconds": round(c.seconds, 2),
                    "deferred": c.deferred,
                    "detail": c.detail[:4000],
                }
                for c in self.checks
            ],
        }


@dataclass(frozen=True)
class _ExecFailure:
    """*argv[0]* could not even be started — distinct from a timeout or a
    non-zero exit, both of which are a verdict FROM the interpreter.

    D-MQI-1: a gate whose interpreter path resolves to something that does not
    exist or is not executable (a stale symlink, a `.venv` deleted out from
    under a cached path, a permissions error) previously raised an uncaught
    ``OSError`` straight out of ``subprocess.run`` — which does not "refuse"
    the candidate, it CRASHES the whole batch (and, via the bisection retry in
    :func:`integrate_batch`, every other candidate sharing that run), which is
    indistinguishable from the queue draining nothing. Carrying the failure as
    a value instead lets every call site turn it into an ordinary, named
    :class:`Check` failure — "refused, never assumed clean" — exactly the
    contract this module already applies to every other unrunnable gate
    (a timeout, an unreadable baseline).
    """

    argv: tuple[str, ...]
    error: str

    def detail(self) -> str:
        return (
            f"could not execute {' '.join(self.argv)}: {self.error} — a gate "
            "that cannot run is refused, never assumed clean"
        )


def _timed_run(argv: list[str], cwd: Path, *, timeout: int, env: dict) -> tuple:
    start = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return None, time.monotonic() - start
    except OSError as exc:
        return _ExecFailure(tuple(argv), str(exc)), time.monotonic() - start
    return proc, time.monotonic() - start


def _timed_run_parallel(
    jobs: dict[str, tuple[list[str], Path, int]], *, env: dict
) -> dict[str, tuple]:
    """:func:`_timed_run` for several independent, read-only jobs at once.

    D-MW-9: contract checks never share state (each is a fresh subprocess
    reading the same tree) so there is no correctness reason to serialize them
    — only the historical accident that the loop was written with one. Threads
    are sufficient (not multiprocessing) because ``subprocess.run`` blocks on
    the child process, releasing the GIL for the wait.
    """
    if not jobs:
        return {}
    with ThreadPoolExecutor(
        max_workers=min(CONTRACT_CHECK_MAX_WORKERS, len(jobs))
    ) as pool:
        futures = {
            key: pool.submit(_timed_run, argv, cwd, timeout=timeout, env=env)
            for key, (argv, cwd, timeout) in jobs.items()
        }
        return {key: fut.result() for key, fut in futures.items()}


def contract_scripts(tree: Path) -> list[Path]:
    """Repo-invariant contract checks present **in the merged tree**.

    Discovered, never enumerated. The set is read from the candidate's merged
    result, so a candidate that *adds* an invariant has it enforced against itself
    in the same gate run, and a candidate that deletes one is visibly running one
    fewer check rather than silently passing.
    """
    return sorted(tree.glob(CONTRACT_CHECK_GLOB))


def contract_baseline(repo: Path, base_ref: str) -> set[str]:
    """Contract-check filenames present on *base_ref* right now.

    The merged tree is compared against this, so **removing** a contract is a
    refusal while a repository that genuinely has none is a genuine pass. Without
    the comparison the check has only one value for "nothing found" and cannot tell
    those apart — the fail-open shape this codebase keeps producing.
    """
    listing = _run_git(
        [
            "ls-tree",
            "-r",
            "--name-only",
            base_ref,
            str(Path(CONTRACT_CHECK_GLOB).parent) + "/",
        ],
        repo,
    )
    if not listing.ok:
        return set()
    pattern = Path(CONTRACT_CHECK_GLOB).name
    return {
        Path(line).name
        for line in listing.out.splitlines()
        if line and Path(line).match(pattern)
    }


@dataclass(frozen=True)
class ScriptBaseline:
    """One contract script's state on *base_ref*: whether it exited clean, and
    (best-effort) the itemized violation lines it printed if it did not.

    Two granularities are tracked because contract scripts report failure two
    different ways in this codebase — some print one line per violation
    (``check_current_only_contract.py``, ``check_swallowed_errors.py``, the
    boundary gates), some print a single static message regardless of *why*
    they failed (``check_sbom_licenses.py``'s ``{"error": "LicenseAuditError"}``,
    a bare ``sys.exit(1)`` with no output at all). Line-diffing alone would
    silently treat the second class as "no new violation" whenever the base was
    already red for ANY reason, even a different one — see
    :func:`run_contract_checks` for how ``ok``/``lines`` combine to avoid that.
    """

    ok: bool
    lines: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ContractBaselineResult:
    """The per-script :class:`ScriptBaseline` every discovered contract check
    already produces on *base_ref* (D-MW-9, CONCEPT:AU-OS.governance.
    test-regression-baseline applied one level up from tests).

    A script absent from *base_ref* entirely (a candidate that adds a brand-new
    contract) maps to ``ScriptBaseline(ok=True)`` — vacuously clean, nothing to
    compare against — so ANY failure it prints on the merged tree is correctly
    "new", which is what makes a freshly-added contract enforced against itself
    in the same gate run.

    ``readable=False`` is the fail-closed case: callers MUST refuse any script
    whose merged-tree run is non-clean rather than guess it is pre-existing red —
    the same non-negotiable contract :class:`BaselineResult` already holds for
    tests.
    """

    readable: bool
    base_sha: str = ""
    scripts: dict[str, ScriptBaseline] = field(default_factory=dict)
    detail: str = ""


def _contract_baseline_cache_path(
    scope: LaneScope, base_sha: str, *, interpreter: str, scripts_signature: str
) -> Path:
    """Cache location for one ``(base_sha, interpreter, scripts_signature)``'s
    contract-check output baseline — same shared arbitration dir and same "one
    entry per discovered item, not per whole request" shape as
    :func:`_baseline_cache_path`, just at script granularity instead of
    test-file granularity (there are only a handful of contract scripts, so no
    delta-run optimization is needed beyond "don't recompute a script already
    known for this base_sha").

    D-MQD-1: the key includes :func:`_venv_signature` alongside the bare
    interpreter path, so an in-place venv rebuild (same path, different
    installed packages, same ``base_sha``) busts this cache instead of
    silently serving a baseline computed against packages that no longer
    exist.

    BUG-176: the key ALSO includes :func:`_scripts_dir_signature` of the
    CANDIDATE'S ``scripts/`` tree — see that function's docstring. Without it,
    two candidates sharing a ``base_sha`` where one repairs a check script
    would have the second silently served a baseline computed under the
    first's (or main's) different instrument, reintroducing the exact defect
    this fix closes.
    """
    digest = hashlib.sha256(
        (
            base_sha
            + "\n"
            + interpreter
            + "\n"
            + _venv_signature(interpreter)
            + "\n"
            + scripts_signature
        ).encode("utf-8")
    ).hexdigest()
    return (
        scope.arbitration_dir
        / QUEUE_DIRNAME
        / CONTRACT_BASELINE_CACHE_DIRNAME
        / f"{digest}.json"
    )


def _load_contract_baseline_cache(path: Path) -> dict[str, ScriptBaseline]:
    """The ``{script_name: ScriptBaseline}`` map cached for one
    ``(base_sha, interpreter)``, or ``{}`` on a miss/corrupt entry — see
    :func:`_load_file_baseline_cache`'s identical reasoning."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        scripts = data["scripts"]
        return {
            str(k): ScriptBaseline(
                ok=bool(v["ok"]), lines=frozenset(str(x) for x in v["lines"])
            )
            for k, v in scripts.items()
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _merge_contract_baseline_cache(
    path: Path, base_sha: str, new_entries: dict[str, ScriptBaseline]
) -> None:
    """Merge ``new_entries`` into the cache — see
    :func:`_merge_file_baseline_cache`'s identical read-merge-atomic-replace."""
    if not new_entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _load_contract_baseline_cache(path)
    current.update(new_entries)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(
            {
                "base_sha": base_sha,
                "scripts": {
                    k: {"ok": v.ok, "lines": sorted(v.lines)}
                    for k, v in current.items()
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def _contract_check_argv(
    script: Path,
    rel: Path,
    *,
    interpreter: str,
    tree: Path,
    base_sha: str | None = None,
) -> list[str]:
    """The invocation the fast tier runs a discovered ``check_*.py`` with.

    ``--repository-root`` and ``--base`` are each forwarded only when the
    script's own source declares support for the flag (grepped literally, not
    probed with a throwaway subprocess) — the same detection convention this
    function already used for ``--repository-root`` extends verbatim to
    ``--base``.

    **Why ``--base`` matters as much as ``--repository-root`` (found reproducing
    a false-reject, D-MQ-FP-1).** A script that answers a question about "the
    range not yet pushed" — ``check_secret_history.py``'s own docstring — needs
    to know what range that is. Left to its own default, such a script picks
    its OWN idea of the base (``origin/main``, or whatever ``git`` resolves),
    which is not necessarily ``base_sha``: this repo's push is deliberately
    deferred, so ``origin/main`` sits over a hundred commits behind local
    ``main`` at any given time. Two consequences, both observed reproducing
    the incident this parameter fixes:

    1. The base-tree run and the merged-tree run each independently resolve
       "their own" default base, and since the merged tree's ``HEAD`` differs
       from the base tree's ``HEAD``, a script whose report embeds anything
       about the scanned range (a line count, the range boundaries) prints
       DIFFERENT text on the two runs even when the candidate contributes
       nothing new — a raw line-diff (:func:`_output_lines`) then reads that
       incidental drift as a "new" violation and rejects a candidate that
       introduced no defect at all.
    2. Worse, the range silently balloons to cover the ENTIRE unpushed
       backlog instead of just the candidate's own commits, so the script can
       exit non-zero for a hit that has nothing to do with the candidate
       being judged — the actual reason ``run_contract_checks``' ``if
       proc.returncode == 0: continue`` fast path was skipped for a script a
       manual ``--base main`` invocation showed clean.

    Passing the SAME ``base_sha`` — the exact commit :func:`compute_contract_baseline`
    already anchors the base-tree run to — to both the base-tree and the
    merged-tree invocation closes both holes: the base-tree run's range
    collapses to ``base_sha..base_sha`` (always empty, always clean, always
    skipped), and the merged-tree run's range becomes ``base_sha..merged_sha``
    — exactly the candidate's own contribution, nothing more.
    """
    argv = [interpreter, str(rel)]
    src = script.read_text(encoding="utf-8", errors="replace")
    if "--repository-root" in src:
        argv += ["--repository-root", str(tree)]
    if base_sha and "--base" in src:
        argv += ["--base", base_sha]
    return argv


_VOLATILE_JSON_SCALAR_RE = re.compile(r'^"[^"]+"\s*:\s*-?\d+(?:\.\d+)?,?$')


def _is_volatile_report_metadata(line: str) -> bool:
    """True for an output line that is purely a JSON object member with a bare
    numeric value — e.g. ``"addedLines": 33,`` — and therefore incidental
    report metadata rather than the identity of a violation.

    **Why this exists (D-MQ-FP-1, defense in depth alongside the ``--base``
    fix above).** ``run_contract_checks`` diffs a script's RAW OUTPUT LINES
    against its base-ref baseline because contract scripts are discovered
    dynamically (``CONTRACT_CHECK_GLOB``) and report their findings in
    whatever shape their author chose — some itemize one violation per line,
    some print a single JSON blob. A script that reports a count, a duration,
    or any other scalar statistic derived from the SIZE of what it scanned
    will legitimately print a different number on the base-tree run than on
    the merged-tree run even when the actual violations are byte-for-byte
    identical (or absent on both) — that number is a description of the scan,
    not a finding. Comparing it as if it were a violation makes every such
    script reject candidates that changed nothing it actually checks for.

    This is scoped narrowly on purpose: a line must be an ENTIRE, otherwise
    plain ``"key": <number>`` JSON member — a per-violation line that merely
    happens to end in digits (a file path, a line number embedded in
    ``path/to/file.py:42: message`` text, an itemized ``"file": "..."``
    string value) does not match, so real violation content is never
    dropped from the comparison. A future script's volatile COUNT is
    absorbed automatically without editing this function or that script;
    a future script's volatile non-numeric text (a timestamp string, an
    unordered list) is not — that residual class still wants the
    differential ``--base`` fix above (or, for a script this codebase owns,
    printing violations on a stable, minimal channel).
    """
    return bool(_VOLATILE_JSON_SCALAR_RE.match(line))


def _output_lines(proc: subprocess.CompletedProcess) -> frozenset[str]:
    return frozenset(
        line.strip()
        for line in (proc.stdout + "\n" + proc.stderr).splitlines()
        if line.strip() and not _is_volatile_report_metadata(line.strip())
    )


def compute_contract_baseline(
    repo: Path,
    base_ref: str,
    script_names: list[str],
    *,
    candidate_tree: Path,
    scope: LaneScope,
    interpreter: str,
    env: dict[str, str],
) -> ContractBaselineResult:
    """The :class:`ScriptBaseline` each named contract check already produces on
    *base_ref* — cached per ``(base_sha, script, scripts_signature)`` exactly
    like :func:`compute_test_baseline` is cached per ``(base_sha, file)``, and
    for the identical reason: :func:`run_fast_gate` calls this once per
    batch/sub-batch attempt, and bisection's sub-batches ask for the same (or a
    subset of the same) script set their parent batch already baselined at the
    same base_sha.

    **BUG-176 — same instrument, both sides.** Before this fix, the baseline
    run executed *base_ref*'s OWN copy of each script (``base_tree /
    contract_dir / name``) against the base tree, while the merged-tree run
    (:func:`run_contract_checks`) always executes the CANDIDATE's copy
    (discovered via ``tree.glob(CONTRACT_CHECK_GLOB)``) against the merged
    tree. A candidate that *repairs* a check script — teaches it to see files
    or violations it was previously blind to — was measured with a stricter
    instrument on the merged tree than the one that produced the baseline on
    the base tree, so the repair's own newly-visible, PRE-EXISTING findings
    read as violations the candidate introduced. Live incident: `fix/
    p0-workitem` fixed the ambient-``GIT_DIR``/``GIT_INDEX_FILE`` blindness
    (BUG-174) in ``scripts/_git_scan.py`` / ``scripts/check_current_only_
    contract.py``; the identical 28 findings existed on both trees under the
    FIXED instrument, but the base side's OLD, blind instrument reported 0,
    so all 28 read as "NEW" and the fix that made the gate able to see was
    rejected for making the gate able to see.

    The fix: this function now OVERLAYS the *candidate*'s entire ``scripts/``
    tree onto the materialized base worktree before running anything, so the
    CODE executed against ``base_ref``'s content is byte-for-byte the same
    code :func:`run_contract_checks` executes against the merged tree's
    content — the only thing that legitimately differs between the two runs
    is the tree being scanned, never the instrument doing the scanning. This
    also transparently fixes the forwarder-pattern scripts
    (``scripts/security/check_*_gate.py`` → a canonical target under bare
    ``scripts/`` via ``_fast_tier_forward.forward``, and any shared helper a
    check imports, e.g. ``scripts/_git_scan.py``): those resolve their own
    target/helper paths relative to whatever ``--repository-root`` they are
    given, so overlaying the whole ``scripts/`` tree (not just the discovered
    ``check_*.py`` files) is what makes the forwarder chain follow the
    candidate's code too, not just its entry point.

    **A script named here that genuinely did not exist under
    ``scripts/security/`` on *base_ref* at all** (a candidate that *adds* a
    brand-new contract) is decided BEFORE the overlay and maps to
    ``ScriptBaseline(ok=True)`` without ever being executed against the base
    tree — a deliberate choice, not an oversight: (1) it preserves the
    existing, documented semantics that "a candidate that adds an invariant
    has it enforced against itself in the same gate run" (:func:`contract_
    scripts`'s docstring) — a brand-new check has no legitimate "pre-existing
    debt" to exempt, so every line it prints on the merged tree is correctly
    "new" and blocking; (2) it cannot crash on a base tree that may be
    missing files or shapes the new script assumes exist (its own author
    only ever validated it against the tree it was added to). This is the
    "must not crash" case the base-missing possibility raises: because the
    check is skipped rather than run, there is nothing to crash.
    """
    base_sha = _require_git(["rev-parse", base_ref], repo)
    scripts_signature = _scripts_dir_signature(candidate_tree / "scripts")
    cache_path = _contract_baseline_cache_path(
        scope, base_sha, interpreter=interpreter, scripts_signature=scripts_signature
    )
    cached = _load_contract_baseline_cache(cache_path)

    unknown = [s for s in script_names if s not in cached]
    if not unknown:
        return ContractBaselineResult(
            readable=True,
            base_sha=base_sha,
            scripts={s: cached[s] for s in script_names},
            detail=(
                f"{len(script_names)} contract script(s) already baselined on "
                f"{base_ref} ({base_sha[:12]}) under this instrument — full "
                "cache hit"
            ),
        )

    contract_dir = Path(CONTRACT_CHECK_GLOB).parent
    new_entries: dict[str, ScriptBaseline] = {}
    with materialized(repo, base_sha, scope=scope) as base_tree:
        # Decide "did this contract exist on base_ref at all" BEFORE the
        # overlay below replaces base_tree/scripts wholesale with the
        # candidate's copy — see the "genuinely new contract" branch of this
        # function's docstring.
        pre_overlay_present = {
            name for name in unknown if (base_tree / contract_dir / name).is_file()
        }
        for name in unknown:
            if name not in pre_overlay_present:
                new_entries[name] = ScriptBaseline(ok=True)

        # BUG-176: overlay the CANDIDATE's scripts/ tree onto the base
        # worktree so every present script — and anything it forwards to or
        # imports from elsewhere under scripts/ — runs as the candidate's
        # code, scanning only the base tree's (untouched) content. This
        # worktree is thrown away at the end of the `with` block, so
        # mutating it here never touches the canonical checkout or base_ref
        # itself.
        candidate_scripts = candidate_tree / "scripts"
        if candidate_scripts.is_dir():
            base_scripts = base_tree / "scripts"
            if base_scripts.exists():
                shutil.rmtree(base_scripts)
            shutil.copytree(
                candidate_scripts,
                base_scripts,
                ignore=shutil.ignore_patterns("__pycache__"),
            )

        present: dict[str, Path] = {
            name: base_tree / contract_dir / name for name in pre_overlay_present
        }

        # D-MW-9: run the (usually empty, since this is cached) not-yet-known
        # scripts concurrently too — same reasoning as run_contract_checks.
        jobs = {
            name: (
                _contract_check_argv(
                    script,
                    script.relative_to(base_tree),
                    interpreter=interpreter,
                    tree=base_tree,
                    base_sha=base_sha,
                ),
                base_tree,
                CONTRACT_BASELINE_BUDGET_SECONDS,
            )
            for name, script in present.items()
        }
        results = _timed_run_parallel(jobs, env=env)
        for name, (proc, secs) in results.items():
            if isinstance(proc, _ExecFailure):
                return ContractBaselineResult(
                    readable=False,
                    base_sha=base_sha,
                    detail=(
                        f"baseline run of {name} on {base_ref} ({base_sha[:12]}) "
                        f"REFUSED: {proc.detail()}"
                    ),
                )
            if proc is None:
                return ContractBaselineResult(
                    readable=False,
                    base_sha=base_sha,
                    detail=(
                        f"baseline run of {name} on {base_ref} ({base_sha[:12]}) "
                        f"exceeded {CONTRACT_BASELINE_BUDGET_SECONDS}s — an "
                        "unproducible baseline is REFUSED, never silently "
                        f"treated as 'no pre-existing violations' (took "
                        f"{secs:.1f}s before timing out)"
                    ),
                )
            new_entries[name] = (
                ScriptBaseline(ok=True)
                if proc.returncode == 0
                else ScriptBaseline(ok=False, lines=_output_lines(proc))
            )

    _merge_contract_baseline_cache(cache_path, base_sha, new_entries)
    all_known = {**cached, **new_entries}
    hits = len(script_names) - len(unknown)
    return ContractBaselineResult(
        readable=True,
        base_sha=base_sha,
        scripts={s: all_known.get(s, ScriptBaseline(ok=True)) for s in script_names},
        detail=(
            f"{len(script_names)} contract script(s) evaluated on {base_ref} "
            f"({base_sha[:12]}) — {hits} from cache, {len(unknown)} freshly run"
        ),
    )


def run_contract_checks(
    tree: Path,
    *,
    interpreter: str,
    env: dict[str, str],
    baseline: set[str] | None = None,
    output_baseline: ContractBaselineResult | None = None,
) -> Check:
    """Run every discovered contract check against the merged tree.

    **Why this belongs in the gate, and why against the merged tree specifically.**
    ``scripts/security/check_tenant_identity_contract.py`` already mutation-tests
    the exact reverted line that re-binds engine placement resolution inside the
    identity minter (D-WD-1/D-SP-1) — and it was wired into **no** pre-commit hook,
    so nothing ran it at merge time. A semantic invariant asserted about the tree
    that will actually exist is the only check that survives the general case: a
    line-level diff heuristic cannot distinguish a lane's ordinary deletion from a
    reverted fix (measured: 17 of 23 live candidates delete lines ``main`` still
    has, median 8, max 667 — see ``docs/architecture/merge-queue.md``), whereas a
    contract states the invariant once and is silent until it is violated.

    **Differential (D-MW-9).** ``main`` legitimately carries contract debt (e.g.
    ``check_current_only_contract.py``'s ~490 pre-existing retired-surface
    references) — judging every discovered script by exit code alone, the
    original shape of this step, would refuse EVERY candidate the moment such a
    script's own debt was folded into discovery, recreating the exact
    absolute-green deadlock ``contract_baseline`` (dropped-script comparison,
    below) already escaped for deletions and ``compute_test_baseline`` escaped
    for tests. A non-clean script's OWN OUTPUT LINES are now diffed against
    ``output_baseline`` (:func:`compute_contract_baseline`): a line present on
    the merged tree but absent from the base-ref run is a NEW violation and
    blocks; a line present on both is pre-existing debt, reported explicitly,
    never silent, and never blocking. ``output_baseline=None`` degrades to the
    pre-D-MW-9 behavior (any non-clean exit fails) — callers that want
    differential gating must pass one, exactly as ``compute_test_baseline`` is an
    opt-in the caller wires up in :func:`run_fast_gate`.

    Runs with ``cwd`` set to the merged tree, and passes ``--repository-root`` only
    to scripts that declare it — detected by reading the script, not by probing it
    with a throwaway subprocess.
    """
    started = time.monotonic()
    scripts = contract_scripts(tree)
    failures: list[str] = []
    notes: list[str] = []
    dropped = sorted((baseline or set()) - {s.name for s in scripts})
    if dropped:
        # A candidate that deletes an invariant would otherwise land by having
        # nothing left to fail. "Fewer contracts than the base" is the degraded
        # read; "this repo has none" is a genuine empty and passes below.
        failures.append(
            "the merged tree DROPS contract check(s) the base still has: "
            + ", ".join(dropped)
            + " — deleting the check that guards an invariant is not a way to "
            "satisfy it"
        )
    if not scripts and not baseline:
        return Check(
            "contract-checks",
            ok=True,
            seconds=time.monotonic() - started,
            detail=(
                f"no contract check matches {CONTRACT_CHECK_GLOB} on the base or in "
                "the merged tree — a genuine empty, not a degraded read"
            ),
        )
    merged_base_sha = output_baseline.base_sha if output_baseline is not None else None
    jobs = {
        script.name: (
            _contract_check_argv(
                script,
                script.relative_to(tree),
                interpreter=interpreter,
                tree=tree,
                base_sha=merged_base_sha,
            ),
            tree,
            CONTRACT_CHECK_BUDGET_SECONDS,
        )
        for script in scripts
    }
    results = _timed_run_parallel(jobs, env=env)
    for script in scripts:
        rel = script.relative_to(tree)
        proc, _ = results[script.name]
        if isinstance(proc, _ExecFailure):
            failures.append(f"{rel}: REFUSED — {proc.detail()}")
            continue
        if proc is None:
            failures.append(
                f"{rel}: exceeded {CONTRACT_CHECK_BUDGET_SECONDS}s — a contract "
                "check that cannot answer is not a passing contract check"
            )
            continue
        if proc.returncode == 0:
            continue
        if output_baseline is None:
            failures.append(f"{rel}: {(proc.stderr or proc.stdout).strip()[:1500]}")
            continue
        if not output_baseline.readable:
            failures.append(
                f"{rel}: REFUSED — the base-ref baseline needed to tell a NEW "
                f"violation from pre-existing red could not be produced: "
                f"{output_baseline.detail}"
            )
            continue
        merged_lines = _output_lines(proc)
        base = output_baseline.scripts.get(script.name, ScriptBaseline(ok=True))
        if merged_lines:
            # Itemized script: line-level diff catches a genuinely new
            # violation even when the base was already red for something
            # else — the precise case a coarser script-level compare would
            # mask (see ScriptBaseline's docstring).
            new_lines = sorted(merged_lines - base.lines)
            pre_existing = merged_lines & base.lines
            if new_lines:
                detail = (
                    f"{rel}: NEW violation(s) not present on the base ref:\n"
                    + "\n".join(new_lines[:50])
                )
                if pre_existing:
                    detail += (
                        f"\n({len(pre_existing)} pre-existing violation(s) also "
                        "on the base ref, not blocking)"
                    )
                failures.append(detail)
            else:
                # new_lines empty and merged_lines non-empty => merged_lines is
                # entirely a subset of base.lines => pre_existing == merged_lines.
                shown = "\n".join(sorted(pre_existing)[:10])
                more = (
                    f" (+{len(pre_existing) - 10} more)"
                    if len(pre_existing) > 10
                    else ""
                )
                notes.append(
                    f"{rel}: {len(pre_existing)} pre-existing violation(s) on "
                    f"the base ref (allowed — not caused by this candidate):\n"
                    f"{shown}{more}"
                )
        elif not base.ok:
            # Non-itemized script (a static message or a bare nonzero exit
            # with no output at all) that was ALSO non-clean on the base:
            # the finest signal available is "was this script red before" —
            # allowed, reported as pre-existing debt, not blocking.
            notes.append(
                f"{rel}: non-clean on the base ref too (no itemized output to "
                "compare — allowed at script granularity, not blocking)"
            )
        else:
            # Non-itemized script that was clean (or absent/new) on the base:
            # any failure now is unambiguously new.
            failures.append(
                f"{rel}: NEW failure not present on the base ref "
                f"(exit {proc.returncode}, no itemized output): "
                f"{(proc.stderr or proc.stdout).strip()[:1500]}"
            )
    detail = "\n".join(failures)
    if notes:
        detail = (detail + "\n" if detail else "") + "\n".join(notes)
    return Check(
        "contract-checks",
        ok=not failures,
        seconds=time.monotonic() - started,
        detail=detail
        or f"{len(scripts)} contract(s) held: {', '.join(str(s.name) for s in scripts)}",
    )


# ---------------------------------------------------------------------------
# Tier fast — differential (regression) gating for targeted tests
# CONCEPT:AU-OS.governance.test-regression-baseline
#
# ``main`` itself can be red — ``contract_baseline`` already established that a
# gate must compare the merged tree against the base rather than judge it in
# isolation; this section applies the identical idea to ``targeted-tests``, the
# one step that previously had no baseline concept at all. A candidate that
# touches an already-red module was being rejected for main's own failures, even
# when it strictly improved them (measured: a branch that fixed 21 of 30 failing
# tests on the same two files was still rejected because 9 remained red).
#
# The rule stays narrow on purpose (see the module docstring's "gate must be
# cheaper than bypassing it" and this codebase's repeated fail-open failures): a
# failing test id is permitted ONLY when that EXACT id already fails, identically,
# on the base ref. Never by file, module, pattern, or count — an id-level compare
# is the only shape that cannot be gamed into masking a real regression.
# ---------------------------------------------------------------------------

#: pytest exit codes under which the ``-rfE`` short summary can be trusted to
#: name every failing/erroring test id: 0 = all green, 1 = some tests failed,
#: 5 = the selection collected zero tests (an honest empty, not a degraded read).
#: 2 (interrupted, e.g. a collection error), 3 (internal error), and 4 (usage
#: error) are NOT here — none of them enumerates failures reliably, so a run
#: that exits one of those is unreadable, not "zero failures".
_PYTEST_READABLE_EXIT_CODES = frozenset({0, 1, 5})


@dataclass(frozen=True)
class BaselineResult:
    """The failing-test-id set one selection already produces on the base ref.

    ``readable=False`` is the fail-closed case (:data:`_PYTEST_READABLE_EXIT_CODES`
    missed, or the run timed out/crashed) — callers MUST refuse the candidate
    rather than treat a missing baseline as "no pre-existing failures", which
    would silently become the exact allow-everything fallback this design forbids.
    """

    readable: bool
    base_sha: str = ""
    failing: frozenset[str] = frozenset()
    detail: str = ""


def _parse_failing_test_ids(stdout: str) -> set[str]:
    """Test ids pytest's ``-rfE`` short summary reports as failed or errored.

    Parsed from pytest's own ``FAILED <nodeid> - reason`` / ``ERROR <nodeid>``
    lines — pytest's stable, documented short-summary contract (every CI
    dashboard that greps test output already relies on this exact shape) — rather
    than a machine report format, so this needs no extra plugin dependency and no
    schema of our own to keep in sync with a pytest upgrade.
    """
    ids: set[str] = set()
    for line in stdout.splitlines():
        for prefix in ("FAILED ", "ERROR "):
            if line.startswith(prefix):
                nodeid = line[len(prefix) :].split(" - ", 1)[0].strip()
                if nodeid:
                    ids.add(nodeid)
    return ids


def _baseline_cache_path(scope: LaneScope, base_sha: str, *, interpreter: str) -> Path:
    """Content-addressed cache location for one ``(base_sha, interpreter)``.

    D-MW-10: no longer keyed by the test *selection* — one JSON file holds every
    file this ``(base_sha, interpreter)`` has EVER been asked to baseline, each
    keyed by its own relative path (see :func:`_load_file_baseline_cache` /
    :func:`_merge_file_baseline_cache`). A selection is now answered by looking up
    each of its files individually, so two selections that overlap (the common
    case: a bisected sub-batch's selection is always a *subset* of its parent
    batch's) share cache entries instead of each owning an all-or-nothing blob
    keyed by their exact, usually-never-repeated set.

    Lives under the same shared, unversioned arbitration dir as the queue itself
    (see :func:`queue_store`) — identical from every worktree, never rewritten by
    a checkout/reset/merge. Two runners computing the same file's entry write the
    same deterministic content, so a race between them is harmless (see
    :func:`_merge_file_baseline_cache`'s read-merge-atomic-replace); nothing needs
    the queue's APPEND-ONLY discipline here because a cache entry for a given
    ``(base_sha, file)`` is immutable once written — only new files get merged in.

    D-MQD-1: the key includes :func:`_venv_signature` alongside the bare
    interpreter path — see that function's docstring for why a bare path is
    not enough to catch an in-place venv rebuild.
    """
    digest = hashlib.sha256(
        (base_sha + "\n" + interpreter + "\n" + _venv_signature(interpreter)).encode(
            "utf-8"
        )
    ).hexdigest()
    return (
        scope.arbitration_dir
        / QUEUE_DIRNAME
        / BASELINE_CACHE_DIRNAME
        / f"{digest}.json"
    )


def _load_file_baseline_cache(path: Path) -> dict[str, list[str]]:
    """The ``{relative_test_file_path: [failing test id, ...]}`` map cached for
    one ``(base_sha, interpreter)``, or ``{}`` on a miss — including a corrupt
    entry.

    A corrupt file is treated as empty (recompute whatever is asked for), not as
    an unreadable baseline: the fail-closed contract is about each TEST RUN's own
    honesty, not about this cache's storage layer, and refusing every future
    candidate because one cache file got truncated would be a self-inflicted,
    permanent denial.
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        files = data["files"]
        return {str(k): [str(v) for v in vs] for k, vs in files.items()}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _merge_file_baseline_cache(
    path: Path, base_sha: str, new_entries: dict[str, list[str]]
) -> None:
    """Merge ``new_entries`` (only READABLE, freshly-run files) into the cache.

    Read-merge-write rather than a blind overwrite, so a concurrent second writer
    baselining a *different* file for the same ``(base_sha, interpreter)`` never
    loses the first writer's entries — both are deterministic per file, so the
    merge always converges to the same union regardless of write order. The write
    is atomic (write to a pid-suffixed temp file, then replace) so a concurrent
    reader can never observe a half-written cache file.
    """
    if not new_entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _load_file_baseline_cache(path)
    current.update(new_entries)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps({"base_sha": base_sha, "files": current}, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def compute_test_baseline(
    repo: Path,
    base_ref: str,
    tests: list[str],
    *,
    scope: LaneScope,
    interpreter: str,
    env: dict[str, str],
) -> BaselineResult:
    """The failing-test-id set the SAME targeted selection already produces on
    *base_ref* — cached **per file** (D-MW-10, CONCEPT:AU-OS.governance.
    test-regression-baseline), so ``main`` (static within a batch) is paid for
    once per FILE rather than once per distinct selection. A selection that is a
    pure subset of one already baselined at this ``base_sha`` — the common shape
    during :func:`integrate_batch`'s bisection, where every sub-batch's selection
    is by construction a subset of its parent's — costs nothing at all: no
    subprocess, no :func:`materialized` worktree checkout. Only files genuinely
    new to this ``base_sha`` are run, and only those are worth
    BASELINE_TEST_BUDGET_SECONDS of patience.

    A selected file that does not exist on *base_ref* at all contributes zero
    baseline failures (it is wholly new on this candidate) rather than making the
    whole baseline unreadable — running pytest against a base-relative path that
    does not exist there is a usage error, not evidence about pre-existing red, so
    those paths are filtered out before the base run rather than left to fail.
    """
    base_sha = _require_git(["rev-parse", base_ref], repo)
    cache_path = _baseline_cache_path(scope, base_sha, interpreter=interpreter)
    cached_files = _load_file_baseline_cache(cache_path)

    unknown = [t for t in tests if t not in cached_files]
    if not unknown:
        failing = {fid for t in tests for fid in cached_files[t]}
        return BaselineResult(
            readable=True,
            base_sha=base_sha,
            failing=frozenset(failing),
            detail=(
                f"{len(tests)} test file(s) already baselined on {base_ref} "
                f"({base_sha[:12]}) — full cache hit, no run needed"
            ),
        )

    with materialized(repo, base_sha, scope=scope) as base_tree:
        present = [t for t in unknown if (base_tree / t).is_file()]
        missing = [t for t in unknown if t not in present]
        new_entries: dict[str, list[str]] = {t: [] for t in missing}

        if present:
            basetemp = (
                partitioned_paths(scope.tree).pytest_basetemp / "merge-queue-baseline"
            )
            basetemp.mkdir(parents=True, exist_ok=True)
            proc, secs = _timed_run(
                [
                    interpreter,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:randomly",
                    "-rfE",
                    f"--basetemp={basetemp}",
                    *present,
                ],
                base_tree,
                timeout=BASELINE_TEST_BUDGET_SECONDS,
                env=env,
            )

            if isinstance(proc, _ExecFailure):
                return BaselineResult(
                    readable=False,
                    base_sha=base_sha,
                    detail=(
                        f"baseline run on {base_ref} ({base_sha[:12]}) for "
                        f"{len(present)} not-yet-cached file(s) REFUSED: "
                        f"{proc.detail()}"
                    ),
                )
            if proc is None:
                return BaselineResult(
                    readable=False,
                    base_sha=base_sha,
                    detail=(
                        f"baseline run on {base_ref} ({base_sha[:12]}) for "
                        f"{len(present)} not-yet-cached file(s) exceeded "
                        f"{BASELINE_TEST_BUDGET_SECONDS}s — an unproducible "
                        "baseline is REFUSED, never silently treated as 'no "
                        f"pre-existing failures' (took {secs:.1f}s before "
                        "timing out; the already-cached files this batch also "
                        "asked for are not re-run, but a whole answer needs the "
                        "new ones too)"
                    ),
                )
            if proc.returncode not in _PYTEST_READABLE_EXIT_CODES:
                return BaselineResult(
                    readable=False,
                    base_sha=base_sha,
                    detail=(
                        f"baseline run on {base_ref} ({base_sha[:12]}) exited "
                        f"{proc.returncode} (collection/usage/internal error, not "
                        "a test outcome) — an unreadable baseline is REFUSED, "
                        "never silently treated as 'no pre-existing failures'\n"
                        + proc.stdout[-2000:]
                        + "\n"
                        + proc.stderr[-1000:]
                    ),
                )
            fresh_failing = _parse_failing_test_ids(proc.stdout)
            for t in present:
                new_entries[t] = sorted(
                    fid for fid in fresh_failing if fid.startswith(t + "::")
                )

    _merge_file_baseline_cache(cache_path, base_sha, new_entries)
    all_files = {**cached_files, **new_entries}
    failing = {fid for t in tests for fid in all_files.get(t, [])}
    hits = len(tests) - len(unknown)
    return BaselineResult(
        readable=True,
        base_sha=base_sha,
        failing=frozenset(failing),
        detail=(
            f"{len(tests)} test file(s) evaluated on {base_ref} ({base_sha[:12]}) "
            f"— {hits} from cache, {len(present)} freshly run"
            + (f", {len(missing)} absent on base" if missing else "")
        ),
    )


def run_fast_gate(
    tree: Path,
    *,
    changed: list[str],
    duplicates: list[dict[str, Any]],
    scope: LaneScope,
    base_ref: str = "main",
) -> GateResult:
    """Everything the queue checks *inside* the lease, against the merged tree.

    Ordered cheapest-first so a candidate that is going to fail usually fails in
    milliseconds. Each check names the interpreter it ran under, because a verdict
    is a claim about an interpreter (see :func:`_interpreter`).
    """
    started = time.monotonic()
    checks: list[Check] = []
    interpreter = _interpreter(tree)
    env = dict(os.environ)
    env["PYTEST_ADDOPTS"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    # 1. duplicate symbols — already computed across the whole batch, ~ms.
    checks.append(
        Check(
            name="cross-branch-duplicate-symbols",
            ok=not duplicates,
            seconds=0.0,
            detail=(
                ""
                if not duplicates
                else "\n".join(
                    f"{d['symbol']} added by "
                    + ", ".join(f"{s['branch']} ({s['at']})" for s in d["added_by"])
                    for d in duplicates
                )
            ),
        )
    )

    # 2. repo-invariant contract checks, against the MERGED tree. Second because it
    #    is the cheapest thing that can catch a *semantic* regression, and because a
    #    silently-reverted security fix must not wait on a test selection to notice.
    #    D-MW-9: differential against base_ref's own output, exactly like
    #    targeted-tests below — main legitimately carries contract debt.
    discovered_names = [s.name for s in contract_scripts(tree)]
    checks.append(
        run_contract_checks(
            tree,
            interpreter=interpreter,
            env=env,
            baseline=contract_baseline(scope.main_tree, base_ref),
            output_baseline=(
                compute_contract_baseline(
                    scope.main_tree,
                    base_ref,
                    discovered_names,
                    candidate_tree=tree,
                    scope=scope,
                    interpreter=interpreter,
                    env=env,
                )
                if discovered_names
                else None
            ),
        )
    )

    # 3. import smoke over changed modules — this is the D-OB-17 catcher: a tree git
    #    merged without a single conflict marker that nonetheless does not import.
    modules = _module_names(tree, changed)
    if modules:
        script = (
            "import importlib, sys\n"
            f"for name in {modules!r}:\n"
            "    importlib.import_module(name)\n"
            "print(sys.executable)\n"
        )
        proc, secs = _timed_run(
            [interpreter, "-c", script],
            tree,
            timeout=IMPORT_SMOKE_BUDGET_SECONDS,
            env=env,
        )
        if isinstance(proc, _ExecFailure):
            checks.append(
                Check("import-smoke", ok=False, seconds=secs, detail=proc.detail())
            )
        elif proc is None:
            checks.append(
                Check(
                    "import-smoke",
                    ok=False,
                    seconds=secs,
                    detail=(
                        f"importing {len(modules)} changed modules exceeded "
                        f"{IMPORT_SMOKE_BUDGET_SECONDS}s — an import that slow is "
                        "itself a defect (import-time work belongs in a function)"
                    ),
                )
            )
        else:
            checks.append(
                Check(
                    "import-smoke",
                    ok=proc.returncode == 0,
                    seconds=secs,
                    detail=("" if proc.returncode == 0 else proc.stderr[-4000:]),
                )
            )
    else:
        checks.append(
            Check("import-smoke", ok=True, seconds=0.0, detail="no changed modules")
        )

    # 4. targeted tests over changed paths.
    tests = select_tests(tree, changed)
    if not tests:
        checks.append(
            Check(
                "targeted-tests",
                ok=True,
                seconds=0.0,
                detail="no tests map to the changed paths",
            )
        )
    elif len(tests) > MAX_TARGETED_TEST_FILES:
        checks.append(
            Check(
                "targeted-tests",
                ok=True,
                seconds=0.0,
                deferred=True,
                detail=(
                    f"{len(tests)} test files selected (> {MAX_TARGETED_TEST_FILES}); "
                    "this is a full run wearing a costume — deferred to the "
                    "post-merge suite so the queue keeps its latency budget"
                ),
            )
        )
    else:
        basetemp = partitioned_paths(scope.tree).pytest_basetemp / "merge-queue"
        basetemp.mkdir(parents=True, exist_ok=True)
        proc, secs = _timed_run(
            [
                interpreter,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:randomly",
                "-rfE",
                f"--basetemp={basetemp}",
                *tests,
            ],
            tree,
            timeout=TARGETED_TEST_BUDGET_SECONDS,
            env=env,
        )
        if isinstance(proc, _ExecFailure):
            # Distinct from the budget-overrun DEFER below: the interpreter
            # never ran at all, so this is not "no regression claim to make",
            # it is "no claim COULD be made" — refused, not deferred.
            checks.append(
                Check("targeted-tests", ok=False, seconds=secs, detail=proc.detail())
            )
        elif proc is None:
            # Preserve the existing budget-overrun contract verbatim (CONCEPT:
            # AU-OS.governance.tiered-merge-gate): exceeding the ceiling is a
            # DEFER to the post-merge suite, not a fail — and, distinctly from a
            # baseline that can't be produced (see compute_test_baseline), this is
            # the run we already had no regression claim to make, so there is
            # nothing here for a baseline to change.
            checks.append(
                Check(
                    "targeted-tests",
                    ok=True,
                    seconds=secs,
                    deferred=True,
                    detail=(
                        f"targeted tests exceeded {TARGETED_TEST_BUDGET_SECONDS}s — "
                        "deferred to the post-merge suite rather than holding the "
                        "lease; the queue's value is that it always returns"
                    ),
                )
            )
        elif proc.returncode not in _PYTEST_READABLE_EXIT_CODES:
            # The merged tree itself did not collect/run cleanly (D-OB-17's shape,
            # or worse). Per CONCEPT:AU-OS.governance.test-regression-baseline this
            # is refused outright without consulting the baseline: "a collection
            # error is not a pass" on EITHER tree, and an unreadable merged run
            # gives no failing-id set to diff against a baseline in the first
            # place — there is nothing to compare.
            checks.append(
                Check(
                    "targeted-tests",
                    ok=False,
                    seconds=secs,
                    detail=(
                        f"the merged tree's targeted-test run exited "
                        f"{proc.returncode} (collection/usage/internal error, not "
                        "a test outcome) — refused; a run that cannot enumerate "
                        "its failures cannot be diffed against the base\n"
                        + proc.stdout[-2000:]
                        + "\n"
                        + proc.stderr[-1000:]
                    ),
                )
            )
        else:
            merged_failing = _parse_failing_test_ids(proc.stdout)
            # The baseline is computed whenever the merged run itself is readable
            # — not only when it has failures — so that a candidate which FIXES a
            # pre-existing failure gets that improvement reported even when nothing
            # else in the selection is red (adversarial property (d)). The batch
            # (not each individual candidate) pays for this: `tests` here is the
            # union selection for the whole batch/sub-batch attempt
            # (:func:`integrate_batch`), and :func:`compute_test_baseline` caches
            # on ``(base_sha, selection, interpreter)`` — main is static within a
            # batch, so a repeated or later attempt against the same selection is
            # answered from disk rather than run again.
            baseline = compute_test_baseline(
                scope.main_tree,
                base_ref,
                tests,
                scope=scope,
                interpreter=interpreter,
                env=env,
            )

            def _ids(label: str, ids: list[str], limit: int = 25) -> str:
                shown = ", ".join(ids[:limit])
                more = f" (+{len(ids) - limit} more)" if len(ids) > limit else ""
                return f"{len(ids)} {label}: {shown}{more}"

            if not merged_failing:
                # Nothing failed on the merged tree, so there is NO POSSIBLE
                # regression no matter what the baseline says — mathematically,
                # `merged_failing - baseline.failing` is empty whenever
                # `merged_failing` is, regardless of whether `baseline.failing` is
                # even known. So an unreadable baseline here does not refuse: fail-
                # closed (property 2) governs cases where the baseline's answer
                # could change the verdict, and here it cannot.
                base_label = (
                    f"{base_ref} ({baseline.base_sha[:12]})"
                    if baseline.readable
                    else base_ref
                )
                detail = f"{len(tests)} test file(s) green on {interpreter}"
                if baseline.readable and baseline.failing:
                    detail += "\n" + _ids(
                        f"test(s) this candidate FIXES relative to {base_label}",
                        sorted(baseline.failing),
                    )
                elif not baseline.readable:
                    detail += (
                        f" (improvement delta unavailable — {base_ref} baseline "
                        f"could not be produced: {baseline.detail})"
                    )
                checks.append(
                    Check("targeted-tests", ok=True, seconds=secs, detail=detail)
                )
            elif not baseline.readable:
                # Fail-closed (CONCEPT:AU-OS.governance.test-regression-baseline,
                # property 2): an unproducible baseline REFUSES the candidate.
                # It must never fall back to "no pre-existing failures", which
                # would silently permit every failing id as if it were known-red.
                checks.append(
                    Check(
                        "targeted-tests",
                        ok=False,
                        seconds=secs,
                        detail=(
                            f"REFUSED: {len(merged_failing)} test(s) failed on "
                            f"the merged tree and the {base_ref} baseline "
                            "needed to tell a regression from pre-existing red "
                            f"could not be produced: {baseline.detail}"
                        ),
                    )
                )
            else:
                new_failures = sorted(merged_failing - baseline.failing)
                pre_existing = sorted(merged_failing & baseline.failing)
                fixed = sorted(baseline.failing - merged_failing)
                base_label = f"{base_ref} ({baseline.base_sha[:12]})"

                if new_failures:
                    detail = _ids(
                        "NEW failure(s) not present on " + base_label, new_failures
                    )
                    if pre_existing:
                        detail += "\n" + _ids(
                            f"pre-existing failure(s) also on {base_label} (not blocking)",
                            pre_existing,
                        )
                    if fixed:
                        detail += "\n" + _ids(
                            f"test(s) this candidate FIXES relative to {base_label}",
                            fixed,
                        )
                    checks.append(
                        Check("targeted-tests", ok=False, seconds=secs, detail=detail)
                    )
                else:
                    # Every merged-tree failure is identically present on the
                    # base: pre-existing red the branch did not cause, ALLOWED
                    # — but reported explicitly, with counts, so it never reads
                    # as a silent success (property 1: not a masking mechanism).
                    detail = _ids(
                        f"pre-existing failure(s) also failing on {base_label} "
                        "(allowed — not caused by this candidate)",
                        pre_existing,
                    )
                    if fixed:
                        detail += "\n" + _ids(
                            f"test(s) this candidate FIXES relative to {base_label}",
                            fixed,
                        )
                    checks.append(
                        Check("targeted-tests", ok=True, seconds=secs, detail=detail)
                    )

    return GateResult(
        ok=all(c.ok for c in checks),
        checks=checks,
        interpreter=interpreter,
        seconds=time.monotonic() - started,
    )


# ---------------------------------------------------------------------------
# Landing — the canonical ref only ever fast-forwards
# ---------------------------------------------------------------------------
def land(repo: Path, commit: str, *, base: str, scope: LaneScope) -> dict[str, Any]:
    """Advance *base* in the canonical checkout to *commit*, fast-forward only.

    ``commit``'s first parent is the current ``base`` tip by construction
    (:func:`integrate_batch` builds the chain that way), so this is always a
    fast-forward and git updates the ref and the working tree in one atomic
    operation. There is no window in which the canonical checkout holds a
    half-applied merge, no conflict is ever resolved there, and ``--ff-only`` is
    git's own refusal if any of that stops being true.

    Guarded by ``guarded_tree_mutation`` — the canonical tree is still someone
    else's tree, and a lane with uncommitted work there must not be fast-forwarded
    over. Deferring is the correct outcome; the candidates stay queued.
    """
    canonical = scope.main_tree
    # Qualify the ref. A bare `rev-parse main` can resolve a TAG or a
    # remote-tracking ref, so the batch could be built on one object and landed
    # against another. (D-RMD-1 / D-RMQ-2)
    base_ref = f"refs/heads/{base}"
    current = _require_git(["rev-parse", base_ref], repo)

    # ★ D-RMD-1: land onto the DECLARED base ref, never onto whatever HEAD
    # happens to be. `merge --ff-only` writes to HEAD, so it lands correctly
    # only while the canonical checkout happens to sit on `base` — true for
    # agent-utilities, and false the moment this queue drives another repo, a
    # release branch, or a checkout left mid-bisect. The failure was SILENT AND
    # POSITIVE: it reported `landed`, and the guarded prune then deleted the
    # branch as landed, leaving the work on a ref nobody reads.
    head_branch = _run_git(["symbolic-ref", "--quiet", "--short", "HEAD"], canonical)
    on_base = head_branch.ok and head_branch.out.strip() == base

    with guarded_tree_mutation(
        canonical, operation=f"land merge-queue batch onto {base}", owner=scope.lane
    ):
        if on_base:
            # Preferred path: git updates the ref AND the working tree in one
            # atomic operation. Keep it — a ref-only write here would leave the
            # canonical tree stale, and fleet pods hostPath-mount that tree.
            res = _run_git(["merge", "--ff-only", commit], canonical)
        else:
            # The base is not checked out here. Refuse if some OTHER worktree
            # holds it: `update-ref` would move the branch out from under that
            # tree's own HEAD. Git forbids this for `checkout`/`branch -f` but
            # NOT for `update-ref`, so the refusal has to be ours — and with
            # hundreds of live worktrees this is a real configuration.
            held = _run_git(["worktree", "list", "--porcelain"], repo)
            if held.ok and f"branch {base_ref}" in held.out:
                raise MergeQueueError(
                    f"{base} is checked out in another worktree; refusing to move "
                    f"{base_ref} underneath it. Land from that worktree, or detach it."
                )
            # Compare-and-swap: only advance if the ref still holds what the
            # batch was built on. `update-ref` will happily REWIND, so the
            # expected-old argument is what enforces fast-forward-only here.
            res = _run_git(["update-ref", base_ref, commit, current], repo)
        if not res.ok:
            raise MergeQueueError(
                f"fast-forward of {base} to {commit[:12]} refused: "
                f"{res.err or res.out} — {base} moved after the batch was built; "
                "the candidates stay queued and the next run rebuilds against it"
            )

        # ★ THE POST-CONDITION, and the durable half of this fix. Re-read the ref
        # and refuse unless it actually holds the computed commit. This catches a
        # wrong write target BY ITSELF — it would have caught D-RMD-1 even with
        # the buggy `merge --ff-only` still in place — and it catches the next
        # variant of the same mistake. Never report `landed` on an unverified write.
        landed_at = _run_git(["rev-parse", base_ref], repo)
        if not landed_at.ok or landed_at.out.strip() != commit:
            raise MergeQueueError(
                f"POST-CONDITION FAILED: {base_ref} holds "
                f"{(landed_at.out or '<unreadable>').strip()[:12]}, expected "
                f"{commit[:12]}. The merge was computed but NOT landed on the "
                "declared base; refusing to report success so the candidates stay "
                "queued and nothing is pruned as landed."
            )
    return {"base": base, "from": current, "to": commit}


# ---------------------------------------------------------------------------
# Prune on merge — repository-manager preferred, guarded inline fallback (D-ORC-21)
# ---------------------------------------------------------------------------
def _worktree_registrations(repo: Path) -> dict[str, str] | None:
    """``{worktree path: HEAD sha}`` for every worktree git currently has
    registered against *repo* — the canonical checkout and every linked
    worktree, this lane's and every sibling's alike.

    The before/after snapshot :func:`_prune_landed_inline` diffs its own
    mutation against (NE-058): the guarantee that a prune touched *only* the
    one worktree it was asked to remove must be verified from git's own
    registry, not assumed from "the code only ever names one path" — the
    exact assumption that held right up until it didn't.  ``None`` means git
    could not answer the registry query; callers must refuse the prune rather
    than interpret that degraded read as an empty registry.
    """
    listing = _run_git(["worktree", "list", "--porcelain"], repo)
    if not listing.ok:
        # An unreadable registry is a degraded safety read, not an empty
        # registry.  Returning ``{}`` here would let the caller remove a
        # worktree and branch without being able to prove that siblings were
        # untouched (the fail-open shape this guard exists to prevent).
        return None
    registrations: dict[str, str] = {}
    current_path: str | None = None
    for line in listing.out.splitlines():
        if line.startswith("worktree "):
            current_path = line[len("worktree ") :].strip()
        elif line.startswith("HEAD ") and current_path is not None:
            registrations[current_path] = line[len("HEAD ") :].strip()
    # A non-bare repository always has at least its canonical checkout in this
    # listing. Treat a successful-but-unparseable/empty response as degraded,
    # too; otherwise a malformed safety read would again look like "no sibling
    # worktrees" and permit an unverified prune.
    return registrations or None


def _prune_landed_inline(
    candidate: Candidate, *, repo: Path, base: str
) -> dict[str, Any]:
    """A guarded, single-worktree prune with **zero** repo-wide git calls.

    **NE-058, root-caused.** ``repository_manager.worktree.WorktreeManager
    .remove()`` — the accelerator this function used to defer to when
    importable — runs a bare ``git worktree prune`` after every single
    removal (reading its source directly confirms it: no ``--worktree``
    scoping, no path argument, the whole-repo form). ``git worktree prune``
    is git's OWN repo-wide sweep: it walks every entry under
    ``$GIT_COMMON_DIR/worktrees/`` and deletes the administrative directory
    of any it judges stale, using heuristics (a missing or unreadable
    ``gitdir`` link, a working directory git can't currently stat) that a
    shared checkout with dozens of concurrently-active linked worktrees can
    trip for a worktree that is very much alive — which is exactly the
    incident this fixes: one landed candidate's prune orphaned eight
    unrelated live tracks' worktree registrations in one call. That
    accelerator also deletes the branch with ``git branch -D`` — no
    merge-base recheck at all, unlike step 2 below. Neither behavior is
    reconcilable with "prune the one thing that landed," so this queue no
    longer calls it for this operation (see :func:`prune_landed`); the
    inline implementation below is now the ONLY implementation, not a
    minimal-environment fallback.

    1. Read the branch tip *now* — every later step names this exact object.
    2. ``git merge-base --is-ancestor <tip> <base>`` — re-asked at this moment,
       never trusted from an earlier scan. A non-zero exit is a real refusal,
       not a malfunction: the branch stays.
    3. **Guard:** if any OTHER worktree of this repo (not the one being
       pruned) currently holds uncommitted work, refuse the ENTIRE prune —
       not just skip the target. A prune is not the moment to discover which
       sibling lanes are mid-edit; a lane occupying a totally unrelated
       worktree must never have its window for that discovery narrowed by
       someone else's landing.
    4. If the candidate's worktree still exists, refuse to touch it while it
       holds uncommitted work (:func:`tree_has_uncommitted_work` — a lane may
       still be occupying it), then ``git worktree remove <that one path>`` —
       named explicitly, never a bare/global form.
    5. **Post-condition.** Re-list ``git worktree list --porcelain`` and
       diff it against the snapshot taken in step 3: the only registration
       allowed to have vanished is the one just removed. Anything else
       missing is refused loudly (:class:`MergeQueueError`) rather than
       reported as a clean prune — this is the same "verify, don't assume"
       shape as :func:`land`'s own post-condition.
    6. **Anchor before deleting.** ``refs/lane-backup/<branch>`` is pointed at
       the tip immediately before the delete, restored (or removed, if it did
       not pre-exist) should the delete itself then fail.
    7. ``git branch -d`` — **never** ``-D``. Git re-decides reachability
       itself, under its own ref lock, atomically with the delete, closing the
       check-then-delete race a python-side check alone cannot close.

    Fails closed at every step: any refusal returns ``pruned: False`` with a
    reason and leaves everything as found. An un-pruned branch is untidy; a
    wrongly-pruned one loses work — the second is never an acceptable trade
    for the first.
    """
    branch = candidate.branch
    tip = _run_git(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], repo)
    if not tip.ok or not tip.out:
        return {
            "pruned": False,
            "branch": branch,
            "reason": (
                f"branch {branch!r} does not exist in {repo} — nothing to prune"
            ),
        }
    tip_sha = tip.out

    ancestor = _run_git(["merge-base", "--is-ancestor", tip_sha, base], repo)
    if not ancestor.ok:
        return {
            "pruned": False,
            "branch": branch,
            "reason": (
                f"refused to delete branch {branch!r}: {tip_sha[:12]} has commits "
                f"that are not (or no longer) reachable from {base!r}, so deleting "
                "the ref would turn them into unreferenced objects"
            ),
        }

    worktree = Path(candidate.worktree).resolve() if candidate.worktree else None
    before = _worktree_registrations(repo)
    if before is None:
        return {
            "pruned": False,
            "branch": branch,
            "reason": (
                "refused to prune: could not read the repository's worktree "
                "registrations, so sibling worktrees cannot be verified as safe"
            ),
        }

    # Guard (bonus, NE-058): a prune touches only the worktree it landed, but
    # it must not even START while some UNRELATED lane worktree is mid-edit —
    # refusing here costs nothing (the candidate stays queued as landed and
    # simply un-pruned until the next run) and removes any incentive to ever
    # relax step 5's post-condition into "best effort." The canonical
    # checkout (`repo` itself) is excluded: it is the shared trunk `land()`
    # already guards on its own terms, not a "sibling lane" in this sense.
    repo_resolved = repo.resolve()
    dirty_siblings = [
        path
        for path in before
        if Path(path).resolve() not in {repo_resolved, worktree}
        if Path(path).is_dir() and tree_has_uncommitted_work(path)
    ]
    if dirty_siblings:
        return {
            "pruned": False,
            "branch": branch,
            "reason": (
                f"refused to prune {branch!r}: {len(dirty_siblings)} sibling "
                f"worktree(s) hold uncommitted work ({', '.join(dirty_siblings)}) "
                "— a prune does not run while any lane of this repo is mid-edit, "
                "landed candidate or not"
            ),
        }

    removed_worktree: str | None = None
    if worktree and worktree != repo.resolve() and worktree.is_dir():
        if tree_has_uncommitted_work(worktree):
            return {
                "pruned": False,
                "branch": branch,
                "reason": (
                    f"worktree {worktree} still holds uncommitted work — a lane may "
                    "still be occupying it, so it is skipped rather than removed "
                    "out from under it"
                ),
            }
        wt_remove = _run_git(["worktree", "remove", str(worktree)], repo)
        if not wt_remove.ok:
            return {
                "pruned": False,
                "branch": branch,
                "reason": (
                    f"git worktree remove {worktree} failed: "
                    f"{wt_remove.err or wt_remove.out}"
                ),
            }
        removed_worktree = str(worktree)

        # Post-condition (NE-058): the ONLY registration allowed to have
        # disappeared is the one just named above. A sibling missing here
        # means something touched the shared admin directory beyond what this
        # function itself asked for — refused loudly, never reported as a
        # clean prune, exactly as `land()` refuses an unverified write.
        after = _worktree_registrations(repo)
        if after is None:
            return {
                "pruned": False,
                "branch": branch,
                "removed": removed_worktree,
                "reason": (
                    "git worktree remove succeeded, but the post-condition "
                    "read of worktree registrations failed; the branch was "
                    "not deleted because sibling safety could not be verified"
                ),
            }
        expected = dict(before)
        expected.pop(str(worktree), None)
        if after != expected:
            missing = sorted(set(expected) - set(after))
            added = sorted(set(after) - set(expected))
            changed = sorted(
                path
                for path in set(expected) & set(after)
                if expected[path] != after[path]
            )
            raise MergeQueueError(
                f"prune of {branch!r} REFUSED post-hoc: worktree registrations "
                f"changed unexpectedly (missing={missing}, added={added}, "
                f"head-changed={changed}) after removing {worktree} — the run "
                "stops rather than reporting a clean prune over unrelated "
                "worktrees possibly having just lost their git registration"
            )

    anchor = f"refs/lane-backup/{branch.replace('/', '-')}"
    previous = _run_git(["rev-parse", "--verify", "--quiet", anchor], repo)
    if previous.code not in {0, 1}:
        return {
            "pruned": False,
            "branch": branch,
            "removed": removed_worktree,
            "reason": (
                f"refused to delete branch {branch!r}: could not inspect the "
                f"backup anchor {anchor}: {previous.err or previous.out}"
            ),
        }
    previous_sha = previous.out if previous.ok else ""
    anchored = _run_git(["update-ref", anchor, tip_sha], repo)
    if not anchored.ok:
        return {
            "pruned": False,
            "branch": branch,
            "removed": removed_worktree,
            "reason": (
                f"refused to delete branch {branch!r}: could not create the "
                f"backup anchor {anchor}: {anchored.err or anchored.out}"
            ),
        }

    deleted = _run_git(["branch", "-d", branch], repo)  # never -D — see docstring
    if not deleted.ok:
        # Restore the ref namespace exactly as found: put back a pre-existing
        # anchor, or remove only the one just written.
        if previous_sha:
            _run_git(["update-ref", anchor, previous_sha, tip_sha], repo)
        else:
            _run_git(["update-ref", "-d", anchor, tip_sha], repo)
        return {
            "pruned": False,
            "branch": branch,
            "reason": (
                f"git refused to delete branch {branch!r} as merged: "
                f"{deleted.err or deleted.out}"
            ),
        }
    return {
        "pruned": True,
        "branch": branch,
        "branch_anchor": anchor,
        "removed": removed_worktree,
    }


def prune_landed(
    candidate: Candidate,
    *,
    repo_name: str,
    base: str,
    repo: Path | str | None = None,
) -> dict[str, Any]:
    """Remove a landed candidate's worktree and branch.

    **NE-058: never delegates to repository-manager's ``WorktreeManager
    .remove()``, on purpose.** This function used to prefer that accelerator
    when importable (``CONCEPT:RM-PRUNE-GUARD``); it no longer does, for a
    reason discovered reproducing the incident, not a style preference:
    reading ``repository_manager/worktree.py`` directly shows ``remove()``
    always follows its own targeted ``git worktree remove`` with a bare,
    repo-wide ``git worktree prune`` and deletes the branch with ``git branch
    -D`` (no merge-base recheck). ``git worktree prune`` is git's OWN sweep
    of every entry under ``$GIT_COMMON_DIR/worktrees/`` — it is not scoped to
    the worktree that was just removed, and on a repo with dozens of
    concurrently-active linked worktrees its own staleness heuristics can
    (and, on 2026-08-17, did) misjudge a live sibling as prunable and delete
    ITS admin directory too. That is NE-058 exactly: one candidate's landing
    orphaned eight unrelated lanes in one call. :func:`_prune_landed_inline`
    is the only implementation now — it never invokes the whole-repo ``prune``
    form, names the one worktree it is allowed to touch explicitly at every
    step, and re-verifies via :func:`_worktree_registrations` that nothing
    else moved before it reports success. repository-manager itself is not
    changed by this — this queue simply stops calling the one operation of
    it that this incident traces to.

    *repo* is the canonical checkout the branch ref actually lives in — always
    pass it when known (:func:`run_queue` does). When omitted (an older
    caller, or a direct test), it is reconstructed by walking up from the
    candidate's own worktree looking for a ``.git`` — correct whenever that
    worktree still exists, since branch refs are shared repo-wide regardless
    of which of its worktrees a git command runs from. *repo_name* is kept in
    the signature for call-site compatibility; it named the repository for
    the now-unused accelerator and no longer does anything.
    """
    del repo_name  # kept for call-site compatibility; see docstring
    if repo is not None:
        repo_path = Path(repo).resolve()
    else:
        repo_path = Path(candidate.worktree or ".").resolve()
        while repo_path != repo_path.parent and not (repo_path / ".git").exists():
            repo_path = repo_path.parent
    result = _prune_landed_inline(candidate, repo=repo_path, base=base)
    result["accelerator"] = (
        "inline (repository-manager's remove() is never used — NE-058)"
    )
    return result


# ---------------------------------------------------------------------------
# Regenerate-on-land — the queue resolves a GENERATED-FILE-ONLY conflict
# itself instead of rejecting the candidate (CONCEPT:AU-OS.governance.merge-queue-regenerate-on-land)
# ---------------------------------------------------------------------------

#: The 4 files every land regenerates (see the module `build_concepts_yaml.py`
#: / `gen_docs.py` / `gen_agents_md.py` docstrings) — so with ~76 candidates
#: converging on one base, nearly every OTHER candidate that also touched one
#: of these (or added a `CONCEPT:` marker, changing the count) conflicts on
#: it too, even though there is no real disagreement: both sides just have a
#: stale copy of a purely-derived file. Regenerating from the ALREADY-MERGED
#: source of truth (the trial tree itself, after the two branches' real
#: content is combined) is the correct resolution — never "pick a side" via
#: ``checkout --theirs``, which silently drops whichever side lost.
GENERATED_FILES = frozenset(
    {
        "docs/concepts.yaml",
        "README.md",
        "AGENTS.md",
        "docs/project_structure.md",
    }
)

#: Regeneration order matters: ``gen_docs.py --write`` reads
#: ``docs/concepts.yaml`` (must be fresh first); ``gen_agents_md.py`` writes
#: both ``AGENTS.md`` and ``docs/project_structure.md`` last.
_REGENERATE_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("scripts/build_concepts_yaml.py",),
    ("scripts/gen_docs.py", "--write"),
    ("scripts/gen_agents_md.py",),
)


def _regenerate_generated_files(repo: Path, tree: str, *, scope: LaneScope) -> str:
    """Materialize *tree*, regenerate :data:`GENERATED_FILES` in place, and
    return the OID of the resulting tree.

    Runs the same 3 commands the module docstring's manual recipe named
    (``build_concepts_yaml.py && gen_docs.py --write && gen_agents_md.py``),
    against a REAL worktree (``materialized`` — the generators read/write
    actual files, not git objects) built from the trial tree, which already
    holds BOTH sides' real changes merged — so this is regeneration from
    truth, never a "pick a side" resolution. Fail-closed: a script that
    exits non-zero raises :class:`MergeQueueError` rather than silently
    keeping a stale or half-regenerated tree.
    """
    throwaway = _commit_trial(repo, tree, [], "merge-queue: regenerate-on-land scratch")
    interpreter = _interpreter(scope.tree)
    with materialized(repo, throwaway, scope=scope) as work_tree:
        for command in _REGENERATE_COMMANDS:
            script = work_tree / command[0]
            if not script.is_file():
                raise MergeQueueError(
                    f"regenerate-on-land: generator script missing in the "
                    f"materialized tree: {script}"
                )
            try:
                proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    [interpreter, str(script), *command[1:]],
                    cwd=str(work_tree),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise MergeQueueError(
                    f"regenerate-on-land: could not run {' '.join(command)}: {exc}"
                ) from exc
            if proc.returncode != 0:
                raise MergeQueueError(
                    "regenerate-on-land: "
                    f"{' '.join(command)} exited {proc.returncode}: "
                    f"{proc.stderr.strip() or proc.stdout.strip()}"
                )
        _require_git(["add", "--", *sorted(GENERATED_FILES)], work_tree)
        return _require_git(["write-tree"], work_tree)


def _resolve_generated_file_conflict(
    repo: Path, trial: TrialMerge, *, scope: LaneScope
) -> TrialMerge:
    """If *trial* conflicts ONLY within :data:`GENERATED_FILES`, resolve it by
    regenerating them; otherwise return *trial* unchanged.

    Deliberately narrow: if even one conflicted path falls OUTSIDE
    ``GENERATED_FILES``, this is a real, human-resolvable conflict and the
    candidate is rejected exactly as before — regeneration never touches a
    conflict that also involves hand-written content.
    """
    if trial.ok or not trial.conflicts:
        return trial
    if not set(trial.conflicts) <= GENERATED_FILES:
        return trial
    resolved_tree = _regenerate_generated_files(repo, trial.tree, scope=scope)
    return TrialMerge(ok=True, tree=resolved_tree, detail=trial.detail)


# ---------------------------------------------------------------------------
# The queue runner — optimistic batching with bisection on failure
# ---------------------------------------------------------------------------
def _build_chain(
    repo: Path, base: str, candidates: list[Candidate], *, scope: LaneScope
) -> tuple[str, list[Candidate], list[tuple[Candidate, TrialMerge]]]:
    """Merge each candidate onto a rolling trial commit; split off the conflicted.

    A conflicting candidate is identified precisely (it is the one whose
    ``merge-tree`` exited 1 against the rolling head) and dropped from the batch
    rather than poisoning it, so one lane's conflict never rejects seven innocent
    ones — UNLESS the conflict is confined to :data:`GENERATED_FILES`, in which
    case :func:`_resolve_generated_file_conflict` regenerates them from the
    merged truth and the candidate proceeds as accepted.
    """
    head = _require_git(["rev-parse", base], repo)
    accepted: list[Candidate] = []
    conflicted: list[tuple[Candidate, TrialMerge]] = []
    for candidate in candidates:
        trial = trial_merge(repo, head, candidate.branch)
        trial = _resolve_generated_file_conflict(repo, trial, scope=scope)
        if not trial.ok:
            conflicted.append((candidate, trial))
            continue
        tip = _require_git(["rev-parse", candidate.branch], repo)
        if tip == head or _run_git(["merge-base", "--is-ancestor", tip, head], repo).ok:
            # Already contained in the rolling head: nothing to land.
            accepted.append(candidate)
            continue
        head = _commit_trial(
            repo,
            trial.tree,
            [head, tip],
            f"merge({candidate.lane}): {candidate.branch} via the merge queue",
        )
        accepted.append(candidate)
    return head, accepted, conflicted


def integrate_batch(
    candidates: list[Candidate],
    *,
    base: str,
    scope: LaneScope,
    depth: int = 0,
) -> list[dict[str, Any]]:
    """Gate *candidates* together; on failure, bisect rather than serialize.

    The merge-train trade. Gating one candidate at a time makes the queue's
    throughput ``1 / gate_duration`` — at a 3-minute gate that is 20 merges an
    hour, which 100 concurrent lanes overrun immediately, and an overrun queue gets
    bypassed. Gating ``N`` together makes it ``N / gate_duration`` while a *failing*
    batch costs ``ceil(log2(N))`` extra runs to attribute. Since most candidates
    pass, the average cost is the batched one and the worst case is logarithmic.

    Returns one outcome record per candidate. Never raises for a candidate's own
    fault — a rejection is a *result*, carrying the evidence its lane needs.
    """
    repo = scope.main_tree
    if not candidates:
        return []

    head, accepted, conflicted = _build_chain(repo, base, candidates, scope=scope)
    outcomes: list[dict[str, Any]] = [
        {
            "branch": c.branch,
            "landed": False,
            "reason": (
                "conflicts with the current "
                f"{base} in: {', '.join(t.conflicts) or 'unreported paths'} — "
                f"sync {base} down into {c.branch}, resolve there, then re-enqueue"
            ),
            "conflicts": t.conflicts,
        }
        for c, t in conflicted
    ]
    if not accepted:
        return outcomes

    duplicates = duplicate_definitions(repo, base, [c.branch for c in accepted])
    changed = sorted({p for c in accepted for p in changed_paths(repo, base, c.branch)})
    with materialized(repo, head, scope=scope) as tree:
        gate = run_fast_gate(
            tree,
            changed=changed,
            duplicates=duplicates,
            scope=scope,
            base_ref=base,
        )

    if gate.ok:
        landing = land(repo, head, base=base, scope=scope)
        for candidate in accepted:
            outcomes.append(
                {
                    "branch": candidate.branch,
                    "landed": True,
                    "batch_size": len(accepted),
                    "gate": gate.as_dict(),
                    **landing,
                }
            )
        return outcomes

    if len(accepted) == 1:
        candidate = accepted[0]
        outcomes.append(
            {
                "branch": candidate.branch,
                "landed": False,
                "reason": "; ".join(
                    f"{c.name}: {c.detail.splitlines()[0] if c.detail else 'failed'}"
                    for c in gate.failures()
                ),
                "gate": gate.as_dict(),
            }
        )
        return outcomes

    # Bisect. The first half is re-gated against the unchanged base; the second is
    # then gated against whatever the first half actually produced, so a candidate
    # is never blamed for a failure it did not cause.
    middle = len(accepted) // 2
    outcomes += integrate_batch(
        accepted[:middle], base=base, scope=scope, depth=depth + 1
    )
    outcomes += integrate_batch(
        accepted[middle:], base=base, scope=scope, depth=depth + 1
    )
    return outcomes


def run_queue(
    *,
    base: str = "main",
    batch_size: int = DEFAULT_BATCH_SIZE,
    prune: bool = True,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Drain up to *batch_size* candidates, under the one serializing lease.

    Raises :class:`~agent_utilities.governance.lanes.LeaseUnavailable` when another
    runner holds ``reconciliation-merge`` — deferring is the correct outcome and the
    caller must make it explicit, exactly as every other LEASE-class resource here.
    """
    scope = lane_scope(path)
    repo = scope.main_tree
    repo_name = repo.name
    started = time.monotonic()
    with hold_lease(MERGE_LEASE, operation="drain the merge queue", path=scope.tree):
        batch = queued(scope.tree)[:batch_size]
        if not batch:
            return {"drained": 0, "outcomes": [], "seconds": 0.0}
        by_branch = {c.branch: c for c in batch}
        outcomes = integrate_batch(batch, base=base, scope=scope)
        for outcome in outcomes:
            candidate = by_branch.get(outcome["branch"])
            if candidate is None:
                continue
            if outcome["landed"]:
                _record_state(candidate, LANDED, "", scope.tree)
                if prune:
                    outcome["prune"] = prune_landed(
                        candidate, repo_name=repo_name, base=base, repo=repo
                    )
            else:
                _record_state(candidate, REJECTED, outcome["reason"], scope.tree)
    return {
        "drained": len(outcomes),
        "landed": sum(1 for o in outcomes if o["landed"]),
        "rejected": sum(1 for o in outcomes if not o["landed"]),
        "outcomes": outcomes,
        "seconds": round(time.monotonic() - started, 2),
        "budget_seconds": FAST_GATE_BUDGET_SECONDS,
    }


# ---------------------------------------------------------------------------
# Merge is not deploy — CONCEPT:AU-OS.governance.merge-deploy-decoupling
# ---------------------------------------------------------------------------
#: The ref the fleet's source mount is expected to follow. Merging moves ``main``;
#: only an explicit promotion moves this.
PROMOTION_REF = "refs/heads/deployed"


def promotion_state(path: Path | str | None = None) -> dict[str, Any]:
    """How far the deployed ref lags ``main``, and whether merge is safely decoupled.

    **The hazard this measures.** graph-os runs source-over-site-packages: the pod
    NFS-mounts the canonical checkout read-only at ``/au`` with ``PYTHONPATH=/au``,
    so the bytes a pod imports on its next start are whatever is in the canonical
    working tree at that instant. Nothing hot-reloads — but nothing pins, either, so
    *any* restart the operator did not choose (a node drain, an eviction, an OOM
    kill, a reschedule) deploys whatever happens to be on ``main``. Merge is not
    deploy; merge is an **armed** deploy that fires at a time nobody picked. At one
    merge a day that is survivable. At a hundred agents merging continuously it is
    not, and it is why a finished fix currently sits blocked on an operator decision.

    **The decoupling.** Point the mount at a checkout of :data:`PROMOTION_REF`
    instead of the canonical ``main`` tree, and the two facts separate cleanly:

    * **merge** — the queue fast-forwards ``main``. The fleet does not see it.
      Nothing is armed. This is the operation that must be cheap and continuous.
    * **promote** — an explicit, human-or-policy-gated fast-forward of ``deployed``
      to a named ``main`` SHA that the *slow* tier has since gone green on, followed
      by a rollout restart. This is the operation that must be deliberate.

    That is one NFS path change and one extra worktree; no new service, no new
    arbitration class, and it is the same fast-forward-only discipline
    :func:`land` already uses. It also gives rollback a meaning it does not
    currently have — ``deployed`` can be moved back to a known-good SHA without
    touching ``main`` at all.
    """
    scope = lane_scope(path)
    repo = scope.main_tree
    deployed = _run_git(["rev-parse", "--verify", "--quiet", PROMOTION_REF], repo)
    main_tip = _require_git(["rev-parse", "main"], repo)
    if not deployed.ok or not deployed.out:
        return {
            "decoupled": False,
            "promotion_ref": PROMOTION_REF,
            "main": main_tip,
            "deployed": None,
            "reason": (
                f"{PROMOTION_REF} does not exist, so the fleet's source mount can "
                "only be following the canonical `main` tree: every merge is an "
                "armed deploy that fires on the next unplanned pod restart. Create "
                "it and repoint the mount — see docs/architecture/merge-queue.md."
            ),
        }
    behind = _require_git(["rev-list", "--count", f"{deployed.out}..{main_tip}"], repo)
    return {
        "decoupled": True,
        "promotion_ref": PROMOTION_REF,
        "main": main_tip,
        "deployed": deployed.out,
        "unpromoted_commits": int(behind or 0),
    }


# ---------------------------------------------------------------------------
# A standalone entrypoint (D-ORC-20)
#
# ``agent-utilities merge-queue run`` (agent_utilities.cli) is the primary,
# fully-featured entrypoint and stays that way. This module previously had NO
# ``main()`` at all — a library function only, invoked exclusively by hand
# through the CLI. That is one reason ``enqueue`` reads like completion: there
# is no visible "thing that runs the queue" anywhere in this file, only a
# function a human remembers to call. This gives the module its own minimal,
# dependency-light entrypoint — usable directly as
# ``python3 -m agent_utilities.governance.merge_queue [run|status]`` — so a
# scheduler can drive it without importing the full CLI's argparse surface
# (dozens of unrelated subcommands, console-script install).
#
# PROPOSED, NOT DEPLOYED (D-ORC-20 item (c)): a supervised runner —
# a systemd timer (``OnUnitActiveSec=60`` + ``ExecStart=python3 -m
# agent_utilities.governance.merge_queue run``) or a Kubernetes CronJob
# (``schedule: "* * * * *"``) with the equivalent command — would close the
# remaining gap: even with the ``note``/``stale_queue_warning`` this lane
# added, draining is still an action an operator must remember to take.
# ``reconciliation-merge`` already serializes concurrent drains (a second
# runner just gets ``LeaseUnavailable``, exit 75), so nothing about the
# queue itself blocks this — only the scheduled unit is missing. Proposing
# it here rather than shipping it: adding a systemd unit or CronJob manifest
# is an infrastructure change outside this lane's remit (no target host/
# cluster namespace was specified), and the queue must keep working with or
# without it, which the tests in this module's test suite pin down.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Minimal CLI: ``run`` (default), ``status``, ``enqueue``, ``withdraw``."""
    import argparse
    import json

    p = argparse.ArgumentParser(
        prog="python3 -m agent_utilities.governance.merge_queue",
        description="Standalone merge-queue driver (see agent-utilities merge-queue "
        "for the full-featured CLI). Suitable as a systemd/cron ExecStart.",
    )
    p.add_argument(
        "action",
        nargs="?",
        default="run",
        choices=["run", "status", "enqueue", "withdraw"],
    )
    p.add_argument("--base", default="main")
    p.add_argument("--branch", default="")
    p.add_argument("--reason", default="")
    p.add_argument("--path", default=None)
    p.add_argument("--no-prune", action="store_true")
    from agent_utilities.governance.lanes import LeaseUnavailable

    args = p.parse_args(argv)
    try:
        if args.action == "status":
            out = queue_report(args.path)
        elif args.action == "enqueue":
            out = enqueue(args.branch, base=args.base, path=args.path)
        elif args.action == "withdraw":
            out = withdraw(args.branch, reason=args.reason, path=args.path)
        else:
            out = run_queue(base=args.base, prune=not args.no_prune, path=args.path)
    except LeaseUnavailable as exc:
        # Same contract as the full CLI: exit 75 so a shell chained with `&&`
        # stops instead of proceeding — another runner already holds the lease.
        print(json.dumps({"deferred": True, "holder": exc.holder}))
        return 75
    except LaneArbitrationError as exc:
        print(json.dumps({"refused": str(exc)}))
        return 1
    print(json.dumps(out, default=str))
    return 1 if out.get("rejected") else 0


if __name__ == "__main__":
    raise SystemExit(main())
