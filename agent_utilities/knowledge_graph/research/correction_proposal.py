#!/usr/bin/python
from __future__ import annotations

"""Typed correction-proposal model — AU-HARNESS-R004.1 (typed model slice).

CONCEPT:AU-HARNESS.harness.graph-only-correction — AU records a proposed gap,
specification, change and supporting evidence through a typed model and
submits it for authorization; AU itself never creates a Git checkout, applies
a patch, or commits a change directly. This slice ships the typed
:class:`CorrectionProposal` plus validation and refusal tests only. Routing
the proposal through the engine's authorization contract, materializing it
via repository-manager, ingesting the immutable receipt and retiring
``change_publisher``'s direct-Git ``LocalBranchPublisher`` path are later
slices (AU-HARNESS-R004.2+) — see ``specs/harness-evolution/requirements.md``.
"""

from dataclasses import dataclass, field
from typing import Any

#: Lifecycle for a proposal moving toward an authorized, materialized change.
#: ``proposed`` (default) -> ``authorized`` -> ``materialized`` (terminal,
#: reached only once a repository-manager receipt is ingested) /
#: ``refused`` (terminal veto).
STATUS_PROPOSED = "proposed"
STATUS_AUTHORIZED = "authorized"
STATUS_MATERIALIZED = "materialized"
STATUS_REFUSED = "refused"

_VALID_STATUSES = frozenset(
    {STATUS_PROPOSED, STATUS_AUTHORIZED, STATUS_MATERIALIZED, STATUS_REFUSED}
)

#: Fields that would indicate a direct, local Git side effect. A proposal
#: carrying any of these is refused outright — AU never creates a checkout,
#: applies a patch, or commits a change itself (AU-HARNESS-R004).
_FORBIDDEN_DIRECT_GIT_FIELDS = frozenset(
    {
        "repo_path",
        "worktree_path",
        "commit_sha",
        "branch",
        "patch",
        "diff",
    }
)


class CorrectionProposalRefused(ValueError):
    """Raised when a proposed correction fails validation and is refused."""


@dataclass(frozen=True)
class MaterializationReceipt:
    """An immutable receipt proving an authorized correction was materialized.

    Produced only by repository-manager, never by AU. Its presence is the
    sole thing that makes the originating gap eligible to resolve.
    """

    receipt_id: str
    commit_ref: str
    materialized_at: str

    def __post_init__(self) -> None:
        if not self.receipt_id:
            raise CorrectionProposalRefused("materialization receipt requires a receipt_id")
        if not self.commit_ref:
            raise CorrectionProposalRefused("materialization receipt requires a commit_ref")
        if not self.materialized_at:
            raise CorrectionProposalRefused(
                "materialization receipt requires a materialized_at timestamp"
            )


@dataclass(frozen=True)
class CorrectionProposal:
    """A proposed gap, specification, change and supporting evidence.

    Submitted through the typed engine contract for authorization. This type
    deliberately has no Git-side fields (no repo path, worktree, branch,
    commit sha, patch or diff) — only opaque references to a gap and a
    change set are carried, so there is nothing here for AU to apply itself.
    """

    gap_ref: str
    specification: str
    change_ref: str
    evidence: tuple[str, ...] = field(default_factory=tuple)
    status: str = STATUS_PROPOSED
    receipt: MaterializationReceipt | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.gap_ref:
            raise CorrectionProposalRefused("a correction proposal requires a gap_ref")
        if not self.specification:
            raise CorrectionProposalRefused("a correction proposal requires a specification")
        if not self.change_ref:
            raise CorrectionProposalRefused("a correction proposal requires a change_ref")
        if not self.evidence:
            raise CorrectionProposalRefused(
                "a correction proposal requires at least one evidence reference"
            )
        if self.status not in _VALID_STATUSES:
            raise CorrectionProposalRefused(f"unknown correction proposal status: {self.status!r}")
        if self.status == STATUS_MATERIALIZED and self.receipt is None:
            raise CorrectionProposalRefused(
                "a materialized correction proposal requires a MaterializationReceipt"
            )
        forbidden = _FORBIDDEN_DIRECT_GIT_FIELDS & self.metadata.keys()
        if forbidden:
            raise CorrectionProposalRefused(
                "a correction proposal may not carry direct-Git fields: "
                f"{sorted(forbidden)} — AU never creates a checkout, applies a patch, "
                "or commits a change directly"
            )

    @property
    def eligible_to_resolve(self) -> bool:
        """Whether the originating gap may now resolve.

        True only once an immutable :class:`MaterializationReceipt` has been
        ingested for an ``materialized`` proposal — never on ``proposed`` or
        ``authorized`` alone.
        """
        return self.status == STATUS_MATERIALIZED and self.receipt is not None


def submit_for_authorization(proposal: CorrectionProposal) -> CorrectionProposal:
    """Return ``proposal`` unchanged if it validates for submission, else refuse.

    This is the typed submission seam: it performs no I/O and makes no Git or
    network call. A later slice (AU-HARNESS-R004.2) wires this to the engine's
    authorization contract and repository-manager materialization.
    """
    if proposal.status != STATUS_PROPOSED:
        raise CorrectionProposalRefused(
            f"only a {STATUS_PROPOSED!r} proposal may be submitted for authorization, "
            f"got {proposal.status!r}"
        )
    return proposal
