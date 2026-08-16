"""Measurement harness: makes a whole class of instrument-mismatch false alarms mechanically impossible.

CONCEPT:AU-OS.measurement.harness

Built from eight real, catalogued false alarms raised in one session — every
one traced back to the measuring INSTRUMENT differing from the thing being
claimed about (a pipeline stage's exit code stood in for the real one, a
two-dot git diff stood in for a merge, an unverified rsync copy stood in for
the source, an unbounded load average stood in for a controlled
environment, a local core-pinned sandbox stood in for the real CI runner, a
substring match stood in for a real process, a journal stood in for a log
file). Each incident has a dedicated module and a test that reproduces the
incident's exact bad input and proves the module catches it:

* :mod:`.provenance`      (A) — mandatory provenance header + verifier
* :mod:`.load_gate`       (B) — refuse to emit a verdict when overloaded
* :mod:`.copy_integrity`  (C) — manifest-hash-verified tree copies
* :mod:`.run`             (D) — exit-code-correct process execution + linter
* :mod:`.merged_tree`     (E) — `git merge-tree`, not a two-dot diff
* :mod:`.proc_safety`     (F) — pattern-based kill that cannot hit its caller
* :mod:`.background`      (G) — systemd-run with the redirect INSIDE the unit

Dependency-light by design: stdlib only (``subprocess``, ``os``, ``hashlib``,
``dataclasses``, ``re``), so this package is usable from a bare interpreter
and from inside pre-commit hooks with no extra install step. See
``agent_utilities/measurement/README.md`` (or the module docstrings above)
for the adoption path, and
``plans/graph-os-completion-program/BUG-LEDGER.md`` BUG-220..BUG-229 for the
defects found while building it.
"""

from __future__ import annotations

from .background import (
    BackgroundRun,
    SystemdRunUnavailableError,
    poll,
    run_background,
    wait_for_unit,
)
from .copy_integrity import (
    CopyIntegrityError,
    CopyIntegrityResult,
    copy_tree,
    manifest_for,
    verify_copy,
)
from .load_gate import (
    TOO_LOADED_TO_MEASURE,
    LoadGateResult,
    TooLoadedToMeasureError,
    check_load,
    default_threshold,
    gate_or_raise,
)
from .merged_tree import (
    MergedTreeResult,
    MergeTreeError,
    files_deleted_by_merge,
    merged_tree,
    naive_two_dot_diff_deletions,
)
from .proc_safety import (
    AmbiguousMatchError,
    KillResult,
    ProcCandidate,
    find_candidates,
    kill_by_pattern,
    live_proc_table,
)
from .provenance import (
    EnvironmentMismatchError,
    MissingProvenanceError,
    ProvenanceError,
    ProvenanceHeader,
    environment_mismatches,
    git_identity,
    require_provenance,
    require_same_environment,
)
from .run import (
    AntipatternHit,
    KilledBySignalError,
    RunResult,
    run,
    scan_for_pipeline_exit_antipattern,
)

__all__ = [
    # provenance (A)
    "ProvenanceHeader",
    "ProvenanceError",
    "MissingProvenanceError",
    "EnvironmentMismatchError",
    "require_provenance",
    "require_same_environment",
    "environment_mismatches",
    "git_identity",
    # load_gate (B)
    "TOO_LOADED_TO_MEASURE",
    "LoadGateResult",
    "TooLoadedToMeasureError",
    "check_load",
    "gate_or_raise",
    "default_threshold",
    # copy_integrity (C)
    "CopyIntegrityResult",
    "CopyIntegrityError",
    "copy_tree",
    "verify_copy",
    "manifest_for",
    # run (D)
    "RunResult",
    "KilledBySignalError",
    "run",
    "AntipatternHit",
    "scan_for_pipeline_exit_antipattern",
    # merged_tree (E)
    "MergedTreeResult",
    "MergeTreeError",
    "merged_tree",
    "files_deleted_by_merge",
    "naive_two_dot_diff_deletions",
    # proc_safety (F)
    "ProcCandidate",
    "KillResult",
    "AmbiguousMatchError",
    "find_candidates",
    "kill_by_pattern",
    "live_proc_table",
    # background (G)
    "BackgroundRun",
    "SystemdRunUnavailableError",
    "run_background",
    "poll",
    "wait_for_unit",
]
