"""Fail-closed source reconciliation over a typed repository seam."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

from .models import (
    CheckpointMutation,
    EntryReconciliation,
    SourceAuthority,
    SourceCatalogEntry,
    SourceCheckpoint,
    SourceEntryProjection,
    SourceGraphProjection,
    SourceManifest,
    SourceReconcileResult,
    SourceReconciliation,
    SourceReconciliationRequest,
    SourceRegistration,
    SourceScope,
    checkpoint_digest_for,
    checkpoint_id_for,
    path_digest_for,
    reconciliation_digest_for,
    reconciliation_id_for,
)
from .repository import RepositoryContractError, SourceRepository


class SourceReconciliationError(ValueError):
    """Stable fail-closed source reconciliation error code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


_TERMINAL_OUTCOMES = {"added", "updated", "unchanged", "tombstoned"}
_INCOMPLETE_OUTCOMES = {"failed", "partial", "timeout"}


def _scope_authority(scope: SourceScope, authority: SourceAuthority) -> None:
    if authority.tenant_id != scope.tenant_id:
        raise RepositoryContractError(
            "repository returned a cross-tenant source authority"
        )


def _scope_entry(scope: SourceScope, entry: SourceCatalogEntry) -> None:
    if entry.tenant_id != scope.tenant_id:
        raise RepositoryContractError("repository returned a cross-tenant source entry")


def _scope_manifest(scope: SourceScope, manifest: SourceManifest) -> None:
    if manifest.tenant_id != scope.tenant_id:
        raise RepositoryContractError(
            "repository returned a cross-tenant source manifest"
        )


def _validate_outcome_scope(
    scope: SourceScope,
    manifest: SourceManifest,
    outcome: EntryReconciliation,
) -> None:
    if outcome.authority_id != manifest.authority_id:
        raise SourceReconciliationError("outcome_authority_mismatch")
    if outcome.tenant_id != scope.tenant_id:
        raise SourceReconciliationError("outcome_tenant_mismatch")


def _manifest_entry(
    manifest: SourceManifest, entry_id: str
) -> SourceCatalogEntry | None:
    return next(
        (entry for entry in manifest.entries if entry.entry_id == entry_id),
        None,
    )


def _validate_incomplete_outcome(
    outcome: EntryReconciliation,
    current: SourceCatalogEntry | None,
) -> None:
    if outcome.terminal:
        raise SourceReconciliationError("incomplete_outcome_marked_terminal")
    if current is not None and current.relative_path != outcome.relative_path:
        raise SourceReconciliationError("reconciled_entry_path_mismatch")


def _validate_tombstone_outcome(
    repository: SourceRepository,
    *,
    scope: SourceScope,
    manifest: SourceManifest,
    outcome: EntryReconciliation,
    current: SourceCatalogEntry | None,
) -> None:
    if current is not None:
        raise SourceReconciliationError("present_entry_cannot_be_tombstoned")
    previous = repository.get_catalog_entry(scope, outcome.entry_id)
    if previous is None:
        raise SourceReconciliationError("tombstone_source_entry_unavailable")
    _scope_entry(scope, previous)
    if previous.authority_id != manifest.authority_id:
        raise RepositoryContractError("repository returned a cross-authority entry")
    if previous.entry_digest != outcome.expected_entry_digest:
        raise SourceReconciliationError("tombstone_digest_drift")
    if previous.relative_path != outcome.relative_path:
        raise SourceReconciliationError("tombstone_path_mismatch")


def _validate_present_outcome(
    outcome: EntryReconciliation,
    current: SourceCatalogEntry | None,
) -> None:
    if current is None:
        raise SourceReconciliationError("reconciled_entry_not_in_manifest")
    if current.relative_path != outcome.relative_path:
        raise SourceReconciliationError("reconciled_entry_path_mismatch")
    if outcome.observed_entry_digest != current.entry_digest:
        raise SourceReconciliationError("observed_entry_digest_mismatch")
    if outcome.outcome == "added":
        _validate_added_outcome(outcome)
        return
    if outcome.outcome == "updated":
        _validate_updated_outcome(outcome)
        return
    if outcome.outcome == "unchanged":
        _validate_unchanged_outcome(outcome)


def _validate_added_outcome(outcome: EntryReconciliation) -> None:
    if outcome.expected_entry_digest is not None:
        raise SourceReconciliationError("added_entry_has_prior_digest")


def _validate_updated_outcome(outcome: EntryReconciliation) -> None:
    if outcome.expected_entry_digest is None:
        raise SourceReconciliationError("updated_entry_missing_prior_digest")
    if outcome.expected_entry_digest == outcome.observed_entry_digest:
        raise SourceReconciliationError("updated_entry_digest_did_not_change")


def _validate_unchanged_outcome(outcome: EntryReconciliation) -> None:
    if outcome.expected_entry_digest != outcome.observed_entry_digest:
        raise SourceReconciliationError("unchanged_entry_digest_changed")


def _selected_entry_ids(
    request: SourceReconciliationRequest, manifest: SourceManifest
) -> tuple[str, ...]:
    selected_entry_ids = request.selected_entry_ids or manifest.entry_ids
    if set(outcome.entry_id for outcome in request.outcomes) != set(selected_entry_ids):
        raise SourceReconciliationError("reconciliation_selection_not_fully_evidenced")
    if tuple(sorted(selected_entry_ids)) != selected_entry_ids:
        raise SourceReconciliationError("reconciliation_selection_not_sorted")
    return selected_entry_ids


def _persist_reconciliation(
    repository: SourceRepository,
    scope: SourceScope,
    reconciliation: SourceReconciliation,
) -> SourceReconciliation:
    existing = repository.get_reconciliation(scope, reconciliation.reconciliation_id)
    if existing is not None:
        if (
            existing.reconciliation_digest != reconciliation.reconciliation_digest
            or existing.authority_id != reconciliation.authority_id
            or existing.tenant_id != reconciliation.tenant_id
            or existing.manifest_id != reconciliation.manifest_id
            or existing.manifest_digest != reconciliation.manifest_digest
        ):
            raise RepositoryContractError(
                "repository returned digest-drifted reconciliation"
            )
        return existing
    repository.put_reconciliation(reconciliation)
    return reconciliation


def _reconcile_result(
    reconciliation: SourceReconciliation,
    checkpoint: SourceCheckpoint | None = None,
) -> SourceReconcileResult:
    if checkpoint is None:
        return SourceReconcileResult(
            result_version="source-reconcile-result.v1",
            reconciliation=reconciliation,
            result_digest=_result_digest(reconciliation, None),
        )
    return SourceReconcileResult(
        result_version="source-reconcile-result.v1",
        reconciliation=reconciliation,
        checkpoint=checkpoint,
        result_digest=_result_digest(reconciliation, checkpoint),
    )


def _validate_checkpoint_scope(
    scope: SourceScope,
    manifest: SourceManifest,
    current: SourceCheckpoint | None,
) -> None:
    if current is not None and (
        current.authority_id != manifest.authority_id
        or current.tenant_id != scope.tenant_id
    ):
        raise RepositoryContractError(
            "repository returned a cross-scope source checkpoint"
        )


def _advance_checkpoint(
    repository: SourceRepository,
    *,
    request: SourceReconciliationRequest,
    manifest: SourceManifest,
    reconciliation: SourceReconciliation,
    current: SourceCheckpoint | None,
    checkpoint_for: Callable[
        [SourceManifest, SourceReconciliation, int], SourceCheckpoint
    ],
) -> SourceCheckpoint:
    current_revision = current.revision if current is not None else 0
    current_id = current.checkpoint_id if current is not None else None
    if current is not None and manifest.manifest_revision < current.manifest_revision:
        raise SourceReconciliationError("checkpoint_manifest_revision_conflict")
    if request.expected_checkpoint_revision != current_revision:
        raise SourceReconciliationError("checkpoint_revision_conflict")
    if request.expected_checkpoint_id != current_id:
        raise SourceReconciliationError("checkpoint_identity_conflict")
    checkpoint = checkpoint_for(manifest, reconciliation, current_revision + 1)
    mutation = CheckpointMutation(
        mutation_version="source-checkpoint-mutation.v1",
        authority_id=manifest.authority_id,
        tenant_id=manifest.tenant_id,
        expected_revision=current_revision,
        expected_checkpoint_id=current_id,
        manifest_id=manifest.manifest_id,
        manifest_revision=manifest.manifest_revision,
        manifest_digest=manifest.manifest_digest,
        reconciliation_id=reconciliation.reconciliation_id,
        change_ref=request.change_ref,
    )
    applied = repository.compare_and_swap_checkpoint(
        request.scope, mutation, checkpoint
    )
    if applied != checkpoint:
        raise RepositoryContractError(
            "repository returned a different source checkpoint"
        )
    return checkpoint


class SourceControlPlane:
    """Register source snapshots and advance checkpoints only on proof."""

    def __init__(self, repository: SourceRepository) -> None:
        self._repository = repository

    def register(self, registration: SourceRegistration) -> None:
        """Persist an authority and its immutable manifest/entry records."""

        self._repository.put_authority(registration.authority)
        for entry in registration.manifest.entries:
            self._repository.put_catalog_entry(entry)
        self._repository.put_manifest(registration.manifest)

    def _load_manifest(
        self, scope: SourceScope, authority_id: str, manifest_id: str
    ) -> SourceManifest:
        manifest = self._repository.get_manifest(scope, manifest_id)
        if manifest is None:
            raise SourceReconciliationError("source_manifest_unavailable")
        _scope_manifest(scope, manifest)
        if manifest.manifest_id != manifest_id:
            raise RepositoryContractError(
                "repository returned a mismatched source manifest"
            )
        authority = self._repository.get_authority(
            scope, authority_id, manifest.authority_digest
        )
        if authority is None:
            raise SourceReconciliationError("source_authority_unavailable")
        _scope_authority(scope, authority)
        if authority.authority_id != authority_id:
            raise RepositoryContractError(
                "repository returned a mismatched source authority"
            )
        if manifest.authority_id != authority.authority_id:
            raise RepositoryContractError(
                "repository returned a cross-authority manifest"
            )
        if manifest.authority_digest != authority.authority_digest:
            raise SourceReconciliationError("source_authority_digest_drift")
        return manifest

    @staticmethod
    def _validate_manifest_for_checkpoint(manifest: SourceManifest) -> None:
        if manifest.observation_status != "complete":
            raise SourceReconciliationError("source_manifest_not_complete")
        if not manifest.entries and not manifest.verified_empty:
            raise SourceReconciliationError("source_empty_manifest_unverified")

    def _validate_outcome(
        self,
        scope: SourceScope,
        manifest: SourceManifest,
        outcome: EntryReconciliation,
    ) -> None:
        _validate_outcome_scope(scope, manifest, outcome)
        current = _manifest_entry(manifest, outcome.entry_id)
        if outcome.outcome in _INCOMPLETE_OUTCOMES:
            _validate_incomplete_outcome(outcome, current)
            return
        if outcome.outcome not in _TERMINAL_OUTCOMES:
            raise SourceReconciliationError("unknown_reconciliation_outcome")
        if outcome.outcome == "tombstoned":
            _validate_tombstone_outcome(
                self._repository,
                scope=scope,
                manifest=manifest,
                outcome=outcome,
                current=current,
            )
            return
        _validate_present_outcome(outcome, current)

    def _build_reconciliation(
        self,
        request: SourceReconciliationRequest,
        manifest: SourceManifest,
        selected_entry_ids: tuple[str, ...],
    ) -> SourceReconciliation:
        outcomes = tuple(sorted(request.outcomes, key=lambda outcome: outcome.entry_id))
        digest = reconciliation_digest_for(
            manifest.authority_id,
            manifest.tenant_id,
            manifest.manifest_id,
            manifest.manifest_digest,
            selected_entry_ids,
            outcomes,
        )
        terminal_count = sum(outcome.terminal for outcome in outcomes)
        return SourceReconciliation(
            reconciliation_version="source-reconciliation.v1",
            reconciliation_id=reconciliation_id_for(
                manifest.authority_id, manifest.manifest_id, digest
            ),
            authority_id=manifest.authority_id,
            tenant_id=manifest.tenant_id,
            manifest_id=manifest.manifest_id,
            manifest_digest=manifest.manifest_digest,
            selected_entry_ids=selected_entry_ids,
            outcomes=outcomes,
            terminal_entry_count=terminal_count,
            complete=terminal_count == len(selected_entry_ids),
            reconciliation_digest=digest,
        )

    @staticmethod
    def _checkpoint_for(
        manifest: SourceManifest,
        reconciliation: SourceReconciliation,
        revision: int,
    ) -> SourceCheckpoint:
        digest = checkpoint_digest_for(
            manifest.authority_id,
            manifest.tenant_id,
            revision,
            manifest.manifest_id,
            manifest.manifest_revision,
            manifest.manifest_digest,
            reconciliation.reconciliation_id,
            reconciliation.terminal_entry_count,
        )
        return SourceCheckpoint(
            checkpoint_version="source-checkpoint.v1",
            checkpoint_id=checkpoint_id_for(manifest.authority_id, revision, digest),
            authority_id=manifest.authority_id,
            tenant_id=manifest.tenant_id,
            revision=revision,
            manifest_id=manifest.manifest_id,
            manifest_revision=manifest.manifest_revision,
            manifest_digest=manifest.manifest_digest,
            reconciliation_id=reconciliation.reconciliation_id,
            selected_entry_count=len(reconciliation.selected_entry_ids),
            terminal_entry_count=reconciliation.terminal_entry_count,
            checkpoint_digest=digest,
        )

    def reconcile(self, request: SourceReconciliationRequest) -> SourceReconcileResult:
        """Record outcomes and CAS a checkpoint only after complete evidence."""
        manifest = self._load_manifest(
            request.scope, request.authority_id, request.manifest_id
        )
        self._validate_manifest_for_checkpoint(manifest)
        selected_entry_ids = _selected_entry_ids(request, manifest)
        for outcome in request.outcomes:
            self._validate_outcome(request.scope, manifest, outcome)
        reconciliation = self._build_reconciliation(
            request, manifest, selected_entry_ids
        )
        reconciliation = _persist_reconciliation(
            self._repository, request.scope, reconciliation
        )
        if not reconciliation.complete:
            return _reconcile_result(reconciliation)

        current = self._repository.get_checkpoint(request.scope, manifest.authority_id)
        _validate_checkpoint_scope(request.scope, manifest, current)
        if current is not None and (
            current.reconciliation_id == reconciliation.reconciliation_id
        ):
            return _reconcile_result(reconciliation, current)
        checkpoint = _advance_checkpoint(
            self._repository,
            request=request,
            manifest=manifest,
            reconciliation=reconciliation,
            current=current,
            checkpoint_for=self._checkpoint_for,
        )
        return _reconcile_result(reconciliation, checkpoint)

    def project(
        self, scope: SourceScope, authority_id: str, manifest_id: str
    ) -> SourceGraphProjection:
        """Return a bounded summary without paths, bytes, or probe error bodies."""

        manifest = self._load_manifest(scope, authority_id, manifest_id)
        checkpoint = self._repository.get_checkpoint(scope, authority_id)
        if checkpoint is not None and (
            checkpoint.authority_id != manifest.authority_id
            or checkpoint.tenant_id != scope.tenant_id
        ):
            raise RepositoryContractError(
                "repository returned a cross-scope source checkpoint"
            )
        if checkpoint is not None and checkpoint.manifest_id != manifest.manifest_id:
            checkpoint = None
        reconciliation_id = checkpoint.reconciliation_id if checkpoint else None
        return SourceGraphProjection(
            projection_version="source-graph-projection.v1",
            tenant_id=manifest.tenant_id,
            authority_id=manifest.authority_id,
            manifest_id=manifest.manifest_id,
            observation_status=manifest.observation_status,
            verified_empty=manifest.verified_empty,
            entry_count=len(manifest.entries),
            terminal_entry_count=checkpoint.terminal_entry_count if checkpoint else 0,
            checkpoint_revision=checkpoint.revision if checkpoint else 0,
            manifest_digest=manifest.manifest_digest,
            reconciliation_id=reconciliation_id,
        )

    def project_reconciliation(
        self, scope: SourceScope, reconciliation_id: str
    ) -> tuple[SourceEntryProjection, ...]:
        """Return bounded per-entry summaries with repository paths redacted."""

        reconciliation = self._repository.get_reconciliation(scope, reconciliation_id)
        if reconciliation is None:
            raise SourceReconciliationError("source_reconciliation_unavailable")
        if reconciliation.tenant_id != scope.tenant_id:
            raise RepositoryContractError(
                "repository returned a cross-tenant reconciliation"
            )
        return tuple(
            SourceEntryProjection(
                projection_version="source-entry-projection.v1",
                tenant_id=reconciliation.tenant_id,
                authority_id=reconciliation.authority_id,
                entry_id=outcome.entry_id,
                path_digest=path_digest_for(outcome.relative_path),
                outcome=outcome.outcome,
                evidence_digest=outcome.evidence_digest,
                tombstone_verified=outcome.absence_verification == "verified",
            )
            for outcome in reconciliation.outcomes
        )


def _result_digest(
    reconciliation: SourceReconciliation, checkpoint: SourceCheckpoint | None
) -> str:
    payload = json.dumps(
        {
            "reconciliation": reconciliation.model_dump(mode="json"),
            "checkpoint": checkpoint.model_dump(mode="json")
            if checkpoint is not None
            else None,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
