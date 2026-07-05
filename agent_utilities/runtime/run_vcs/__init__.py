"""CONCEPT:AU-ORCH.runvcs.event-kernel — agent-native run version-control (run-VCS).

Shepherd unifies three fork primitives our platform kept *disjoint* — the message
checkpoint (``capabilities/checkpointing.py``), the snippet warm-fork sandbox
(``rlm/sandboxes/*``) and the observe-only run provenance (``runtime/provenance.py``,
``:RunTrace``) — into ONE content-addressed commit per agent↔environment interaction
that can be forked, reverted, replayed, and held as a reviewable proposal.

This package is that unification, built additively on top of the existing pieces:

* :mod:`.kernel` — Phase 1: a content-addressed, typed, projectable run-event log
  (shepherd's ``Fact``/``Cut``/``Slice`` shape; identity == digest).
* :mod:`.carrier` — Phase 2: a copy-on-write process+filesystem snapshot
  (``overlayfs``/``fuse-overlayfs`` accelerated; content-addressed blob-store
  fallback so it works unprivileged and in tests).
* :mod:`.run_commit` — Phase 2: ONE commit object unifying message checkpoint +
  fs snapshot + event-log cut.
* :mod:`.run_session` — Phase 3: fork/revert a *live* run from any commit.
* :mod:`.replay` — Phase 4: deterministic trace replay (a recorded exchange
  stands in for the model).
* :mod:`.retained_output` — Phase 5: hold a run's world delta as a proposal,
  materialize only on accept (reuses ``orchestration/action_policy`` +
  ``ontology/edits/revert``).
"""

from __future__ import annotations

from .carrier import FsCarrier, FsSnapshot, is_overlayfs_available
from .kernel import RunCut, RunEvent, RunEventLog
from .replay import ReplayModel, replay_run
from .retained_output import RetainedRunGate, RetainedRunProposal
from .run_commit import RunCommit, RunCommitStore
from .run_session import RunSession, RunSessionRegistry

__all__ = [
    "FsCarrier",
    "FsSnapshot",
    "ReplayModel",
    "RetainedRunGate",
    "RetainedRunProposal",
    "RunCommit",
    "RunCommitStore",
    "RunCut",
    "RunEvent",
    "RunEventLog",
    "RunSession",
    "RunSessionRegistry",
    "is_overlayfs_available",
    "replay_run",
]
