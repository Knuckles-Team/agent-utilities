"""CONCEPT:AU-ORCH.runvcs.retained-output-gate — hold a run's world delta as a reviewable proposal.

Shepherd keeps a child run's output *retained* (a Changeset) and materializes it into the parent
world only when a gate accepts — otherwise it is discarded and the world is untouched. This is
the run-scope generalization of our existing propose-only patterns (``SkillProposal``,
``ActionApproval``): a completed run's effects are a **proposal**, not a fait accompli.

A :class:`RetainedRunProposal` bundles the run's final :class:`~.run_commit.RunCommit` (the
proposed filesystem world) with any KG edits it made (the proposed graph delta). The
:class:`RetainedRunGate` routes the accept through the ONE autonomy decision point
(``orchestration/action_policy.py``, kind ``run.select``) and only materializes the fs delta —
via the same carrier restore used for revert — when the policy allows. Discard reverts any
provisional KG edits through ``ontology/edits/revert.py`` and leaves the filesystem untouched
(it was never materialized). Reuse, not reinvention: the gate is action-policy + revert wired to
a run commit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_utilities.orchestration.action_policy import (
    ActionDecision,
    ActionRequest,
    get_action_policy,
)

from .carrier import FsCarrier
from .run_commit import RunCommit

logger = logging.getLogger(__name__)


@dataclass
class RetainedRunProposal:
    """A completed run's world delta, held for review (not yet materialized).

    ``commit`` is the proposed final filesystem+message+process world; ``edit_ids`` are any KG
    edits the run recorded in an :class:`~agent_utilities.knowledge_graph.ontology.edits.ledger.EditLedger`
    that a discard must compensate. ``source`` names who produced the run (for the audit).
    """

    run_id: str
    commit: RunCommit
    edit_ids: list[str] = field(default_factory=list)
    source: str = "run"
    materialized: bool = False
    discarded: bool = False

    def action_request(self, *, target_root: str | Path) -> ActionRequest:
        return ActionRequest(
            kind="run.select",
            target=self.run_id,
            params={
                "commit_id": self.commit.commit_id,
                "fs_snapshot": self.commit.fs_snapshot.snapshot_id,
                "files": self.commit.fs_snapshot.file_count,
                "kg_edits": len(self.edit_ids),
                "target_root": str(target_root),
            },
            source=self.source,
            reason=f"materialize retained run {self.run_id} output",
        )


class RetainedRunGate:
    """Governed accept/discard of a :class:`RetainedRunProposal`.

    Wraps the shared :class:`~agent_utilities.orchestration.action_policy.ActionPolicy`; a
    ``run.select`` decision (approval_required by default) gates materialization, and discard is
    always safe. ``engine`` is threaded to the policy (durable audit + KG rule overrides) and to
    revert; both are optional so the gate works offline.
    """

    def __init__(self, engine: Any | None = None, policy: Any | None = None) -> None:
        self._engine = engine
        self._policy = policy or get_action_policy(engine)

    def review(
        self, proposal: RetainedRunProposal, *, target_root: str | Path
    ) -> ActionDecision:
        """Decide whether to materialize ``proposal`` — no side effects on the world."""
        return self._policy.decide(proposal.action_request(target_root=target_root))

    def select(
        self,
        proposal: RetainedRunProposal,
        carrier: FsCarrier,
        *,
        target_root: str | Path,
    ) -> dict[str, Any]:
        """Materialize the held fs delta into ``target_root`` — only if the policy allows.

        Returns ``{materialized, decision, ...}``. When the decision is queue/deny the world is
        NOT touched and the proposal stays pending (an approval was filed for the queue path).
        """
        decision = self.review(proposal, target_root=target_root)
        if not decision.allowed:
            return {
                "materialized": False,
                "decision": decision.decision,
                "reason": decision.reason,
                "approval_id": decision.approval_id,
            }
        result = carrier.restore(proposal.commit.fs_snapshot, target=target_root)
        proposal.materialized = True
        return {
            "materialized": True,
            "decision": decision.decision,
            "files_written": result["written"],
            "files_removed": result["removed"],
            "commit_id": proposal.commit.commit_id,
        }

    def discard(
        self,
        proposal: RetainedRunProposal,
        *,
        ledger: Any | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        """Discard the proposal, leaving the world untouched; compensate any KG edits.

        The filesystem delta was never materialized, so discard is a no-op there. If the run
        recorded KG edits and a ``ledger`` is supplied, they are reverted through the append-only
        compensating-edit path (``ontology/edits/revert.revert_edits``).
        """
        reverted: list[str] = []
        if ledger is not None and proposal.edit_ids:
            from agent_utilities.knowledge_graph.ontology.edits.revert import (
                revert_edits,
            )

            compensating = revert_edits(ledger, proposal.edit_ids, actor=actor)
            reverted = [e.id for e in compensating]
        proposal.discarded = True
        # Governed, but always safe — recorded for the audit trail.
        self._policy.decide(
            ActionRequest(
                kind="run.discard",
                target=proposal.run_id,
                params={"commit_id": proposal.commit.commit_id},
                source=proposal.source,
                reason=f"discard retained run {proposal.run_id} output",
            )
        )
        return {
            "discarded": True,
            "world_touched": False,
            "kg_edits_reverted": reverted,
        }
