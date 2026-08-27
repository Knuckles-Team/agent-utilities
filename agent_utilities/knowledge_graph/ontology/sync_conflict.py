"""Backfeed preflight (CONCEPT:AU-KG.ingest.backfeed-preflight) -- per-field conflict resolution and a fail-closed gate every source-sync/writeback commit passes through (CA-22, DEC-CA-07/P11).

Closes the gap DEC-CA-07/P11 names: before this module, no ``_DELTA_HANDLERS`` entry
and no ``run_writeback`` call path could detect a source-vs-graph field conflict
before overwriting -- a handler that raced a concurrent external edit (or a prior
backfeed from a *different* source) silently clobbered it. :func:`resolve_field_conflict`
implements the four-policy matrix a connector declares per field
(``source_wins``/``graph_derived``/``manual_review``/``reject``); the two
review-producing policies never auto-resolve -- they always produce a
:class:`SyncConflict`, and :func:`evaluate_backfeed_preflight` decides, fail-closed,
whether that conflict may also raise a governed :class:`BackfeedProposal` (only when
the connector has declared a real ``backfeed.approval_class``) or must simply block the
write with a named :class:`PreflightRejection` reason.

**Recovery note (measured, CA-22-W01).** ``ConflictFieldPolicy``/``ConflictPolicySpec``/
``BackfeedCapabilitySpec`` were designed on branch
``agent-utilities:goc/goc-27-30-enterprise-closure`` as *manifest* types living in
``connector_manifest.py`` (recoverable, byte-for-byte, from
``refs/lane-park/goc/goc-27-30-enterprise-closure`` commit ``f6afa3c93a``, lines
387-495 of that ref's ``connector_manifest.py``). This lane does NOT cherry-pick them
there: ``file-ownership.yaml`` ``FO-CA-013`` (current, dated after this lane's own
brief) assigns ``connector_manifest.py`` **exclusively to CA-32** -- "CA-22 owns
``sync_conflict.py``" is FO-CA-013's own disambiguating note. Editing a file another
lane exclusively owns is a program-terminating violation of the file-partition
invariant (PROGRAM-CHARTER.md SS6), so these three types are defined HERE instead, with
the exact field shape the recovered ref used, so CA-32 can either (a) import them from
this module when it wires ``ConnectorManifest.conflict_policy``/``.backfeed``, or (b)
redeclare pydantic-identical types in its own file and this module's defensive
``getattr`` access (see :func:`_manifest_conflict_policy`/:func:`_manifest_backfeed`
in ``source_sync.py``) picks either shape up with zero further change here. Flagged
in the lane evidence as a program-level coordination note, not silently absorbed.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

__all__ = [
    "CONFLICT_POLICIES",
    "ConflictFieldPolicy",
    "ConflictPolicySpec",
    "BackfeedCapabilitySpec",
    "SyncConflict",
    "BackfeedProposal",
    "PreflightRejection",
    "resolve_field_conflict",
    "evaluate_backfeed_preflight",
]

#: The four-policy matrix (DEC-CA-07 contract). ``manual_review``/``reject`` never
#: auto-resolve -- both always produce a :class:`SyncConflict`; they differ only in
#: what the CALLER does with it (``reject`` additionally never raises a
#: :class:`BackfeedProposal`, see :func:`evaluate_backfeed_preflight`).
CONFLICT_POLICIES = ("source_wins", "graph_derived", "manual_review", "reject")


class ConflictFieldPolicy(BaseModel):
    """Per-field conflict resolution policy.

    A conflict is source-vs-graph disagreement on one field of an already
    mirrored record -- NOT an ordinary delta update (an ordinary update is the
    source disagreeing with the *previous version of itself*, which is exactly
    what a sync is for). ``policy`` never defaults to silently picking a side.
    """

    model_config = ConfigDict(extra="forbid")

    field: str
    policy: str  # one of CONFLICT_POLICIES

    @field_validator("policy")
    @classmethod
    def _validate_policy(cls, value: str) -> str:
        if value not in CONFLICT_POLICIES:
            raise ValueError(
                f"conflict policy must be one of {CONFLICT_POLICIES}, got {value!r}"
            )
        return value


class ConflictPolicySpec(BaseModel):
    """Per-resource conflict policy table plus the default for unlisted fields.

    Deliberately fail-closed as a DEFAULT: a connector that declares nothing
    (``ConflictPolicySpec()``) gets ``default_policy="manual_review"`` -- an
    undeclared field never silently picks a side either.
    """

    model_config = ConfigDict(extra="forbid")

    fields: list[ConflictFieldPolicy] = Field(default_factory=list)
    default_policy: str = "manual_review"

    @field_validator("default_policy")
    @classmethod
    def _validate_default_policy(cls, value: str) -> str:
        if value not in CONFLICT_POLICIES:
            raise ValueError(
                f"default conflict policy must be one of {CONFLICT_POLICIES}, got {value!r}"
            )
        return value

    def policy_for(self, field_name: str) -> str:
        """The declared policy for ``field_name``, or :attr:`default_policy`."""
        for entry in self.fields:
            if entry.field == field_name:
                return entry.policy
        return self.default_policy

    def declares(self, field_name: str) -> bool:
        """True when ``field_name`` has an explicit (non-default) policy entry.

        ``_apply_with_preflight`` (``source_sync.py``) only compares fields a
        connector has explicitly opted into via this table -- see that
        module's docstring on ``_apply_with_preflight`` for why an
        undeclared field is never diffed (never a behavior regression for
        the 31 existing handlers, none of which declare anything today).
        """
        return any(entry.field == field_name for entry in self.fields)


class BackfeedCapabilitySpec(BaseModel):
    """Declares that a connector MAY be backfed to, and what governs it.

    Declaration only -- no execution lives here. Execution is a governed
    GOC-19 WorkItem (dry-run diff -> policy/preflight -> approval -> source
    call -> result/provenance), out of this lane's scope; see this module's
    own docstring.
    """

    model_config = ConfigDict(extra="forbid")

    capabilities: list[str] = Field(
        default_factory=list
    )  # e.g. "servicenow.incident.update"
    approval_class: str | None = (
        None  # e.g. "change"; None means backfeed is not enabled
    )

    @property
    def enabled(self) -> bool:
        """Backfeed is enabled only when an approval class is named (fail-closed:
        a capability list with no approval class is still "not enabled")."""
        return bool(self.approval_class)


@dataclass(frozen=True)
class SyncConflict:
    """A detected, never-auto-resolved field disagreement (digest-only record).

    Never carries a resolution -- ``manual_review``/``reject`` policies produce
    this and nothing else; a human/gate decides what happens next.
    """

    connector: str
    node_id: str
    field: str
    source_value: Any
    graph_value: Any
    policy: str
    source_instance: str = ""
    reason: str = "field_conflict"


@dataclass(frozen=True)
class BackfeedProposal:
    """A governed proposal to push a conflicting field back to its source.

    Stamped (ownership/classification) identically to
    ``enrichment.writeback.approval.ProposalQueue``'s existing
    ``WritebackProposal`` convention -- this lane reuses that stamping seam
    rather than inventing a second one (see :func:`_stamp_backfeed_proposal`).
    Never auto-applied: raising one is the entire effect of a passed preflight
    for a ``manual_review``-conflicted, backfeed-enabled field.
    """

    connector: str
    node_id: str
    field: str
    source_value: Any
    graph_value: Any
    approval_class: str
    conflict: SyncConflict
    stamped: dict[str, Any] = dataclass_field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "connector": self.connector,
            "node_id": self.node_id,
            "field": self.field,
            "source_value": self.source_value,
            "graph_value": self.graph_value,
            "approval_class": self.approval_class,
            "reason": self.conflict.reason,
            **self.stamped,
        }


@dataclass(frozen=True)
class PreflightRejection:
    """A fail-closed block -- DEC-CA-07's failure-semantics table, reason-named.

    ``reason`` in {"undeclared_capability", "stale_expected_version",
    "backfeed_disabled"}. Never a bare ``False``/``None`` -- always names why.
    """

    reason: str
    detail: str = ""


def _stamp_backfeed_proposal(proposal: BackfeedProposal) -> BackfeedProposal:
    """Ownership/classification-stamp a proposal (mirrors ``approval.py::_stamp``)."""
    props: dict[str, Any] = {}
    try:
        from ..core.tenant_sharing import stamp_classification, stamp_ownership

        stamp_ownership(props)
        stamp_classification(props, "BackfeedProposal")
    except Exception:  # noqa: BLE001 - stamping is best-effort, never blocks the proposal
        logger.debug(
            "BackfeedProposal stamping failed for %s/%s/%s",
            proposal.connector,
            proposal.node_id,
            proposal.field,
            exc_info=True,
        )
    return dataclasses.replace(proposal, stamped=props)


def resolve_field_conflict(
    policy: str,
    source_value: Any,
    graph_value: Any,
    *,
    connector: str = "",
    node_id: str = "",
    field_name: str = "",
    source_instance: str = "",
) -> SyncConflict | Any:
    """Resolve ONE field's source-vs-graph disagreement per the 4-policy matrix.

    - ``source_wins``: returns ``source_value`` -- the incoming write proceeds.
    - ``graph_derived``: returns ``graph_value`` -- the graph's own (derived)
      value is authoritative; the incoming field write is discarded, no error.
    - ``manual_review`` / ``reject``: NEVER auto-resolve -- always returns a
      :class:`SyncConflict`, so the caller must not commit this field.

    Raises ``ValueError`` for an unrecognized policy (never guesses).
    """
    if policy == "source_wins":
        return source_value
    if policy == "graph_derived":
        return graph_value
    if policy in ("manual_review", "reject"):
        return SyncConflict(
            connector=connector,
            node_id=node_id,
            field=field_name,
            source_value=source_value,
            graph_value=graph_value,
            policy=policy,
            source_instance=source_instance,
        )
    raise ValueError(f"unknown conflict policy {policy!r}")


def evaluate_backfeed_preflight(
    *,
    connector: str,
    node_id: str = "",
    conflict: SyncConflict | None = None,
    backfeed: BackfeedCapabilitySpec | None = None,
    expected_source_version: str | None = None,
    current_source_version: str | None = None,
) -> BackfeedProposal | PreflightRejection | None:
    """Fail-closed backfeed preflight (DEC-CA-07 failure-semantics table).

    Called from every one of the 31 ``_DELTA_HANDLERS`` entries (via
    ``source_sync._apply_with_preflight``) and from ``run_writeback`` -- no
    exception path (DEC-CA-07). Three outcomes:

    - ``None`` -- nothing to propose (no conflict passed, or conflict policy
      already resolved automatically via ``source_wins``/``graph_derived``
      upstream in :func:`resolve_field_conflict`). The caller proceeds with
      its normal commit; this is the byte-identical-to-today path for every
      connector that has not declared a conflict policy (additive migration).
    - :class:`PreflightRejection` -- blocks the write, reason named:
        * ``stale_expected_version`` -- ``expected_source_version`` (an
          optimistic-concurrency token the caller believed was current) does
          not match ``current_source_version`` (what the graph is actually
          at). Checked FIRST, independent of ``conflict``: a stale version is
          a block even with no field-level conflict object at all.
        * ``backfeed_disabled`` -- a conflict exists but the connector
          declared capabilities with no ``approval_class`` (present but off).
        * ``undeclared_capability`` -- a conflict exists and the connector
          declared no backfeed capability at all (``backfeed`` is ``None`` or
          default-empty).
    - :class:`BackfeedProposal` -- a conflict exists AND backfeed is declared
      + enabled: a governed, stamped proposal is raised. Never auto-applied.

    Fail-closed throughout: this function never approves a write by default.
    """
    if expected_source_version is not None and current_source_version is not None:
        if str(expected_source_version) != str(current_source_version):
            return PreflightRejection(
                reason="stale_expected_version",
                detail=(
                    f"connector {connector!r} node {node_id!r}: expected source "
                    f"version {expected_source_version!r}, graph cursor is at "
                    f"{current_source_version!r}"
                ),
            )

    if conflict is None:
        return None

    spec = backfeed or BackfeedCapabilitySpec()
    if not spec.enabled:
        reason = "backfeed_disabled" if spec.capabilities else "undeclared_capability"
        return PreflightRejection(
            reason=reason,
            detail=(
                f"connector {connector!r} field {conflict.field!r}: no "
                "backfeed.approval_class declared for this connector"
            ),
        )

    proposal = BackfeedProposal(
        connector=connector,
        node_id=node_id or conflict.node_id,
        field=conflict.field,
        source_value=conflict.source_value,
        graph_value=conflict.graph_value,
        approval_class=spec.approval_class or "",
        conflict=conflict,
    )
    return _stamp_backfeed_proposal(proposal)
