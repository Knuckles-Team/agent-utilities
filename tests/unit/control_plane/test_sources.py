"""Focused NE-088 source authority/catalog/checkpoint fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from agent_utilities.control_plane.sources import (
    AbsenceVerification,
    CheckpointMutation,
    EntryKind,
    EntryOutcome,
    EntryReconciliation,
    ManifestStatus,
    RepositoryContractError,
    SourceAuthority,
    SourceCatalogEntry,
    SourceCheckpoint,
    SourceControlPlane,
    SourceKind,
    SourceManifest,
    SourceReconciliation,
    SourceReconciliationError,
    SourceReconciliationRequest,
    SourceRegistration,
    SourceScope,
    authority_digest_for,
    authority_id_for,
    catalog_entry_digest_for,
    entry_id_for,
    manifest_digest_for,
    manifest_id_for,
    scope_digest_for,
)


def _digest(char: str) -> str:
    return "sha256:" + char * 64


def _scope(tenant: str = "tenant:alpha") -> SourceScope:
    grant = _digest("1")
    return SourceScope(
        scope_version="source-scope.v1",
        tenant_id=tenant,
        principal_id="principal:operator",
        grant_digests=(grant,),
        scope_digest=scope_digest_for(tenant, "principal:operator", (grant,)),
    )


def _authority(tenant: str = "tenant:alpha") -> SourceAuthority:
    publisher = "knuckles"
    package_name = "gitlab-api"
    source_name = "issues"
    source_kind: SourceKind = "api"
    source_ref = "source:gitlab:issues"
    root_ref = "root:gitlab:project"
    authority_revision = 1
    schema_digest = _digest("5")
    return SourceAuthority(
        authority_version="source-authority.v1",
        authority_id=authority_id_for(tenant, publisher, package_name, source_name),
        tenant_id=tenant,
        publisher=publisher,
        package_name=package_name,
        source_name=source_name,
        source_kind=source_kind,
        source_ref=source_ref,
        root_ref=root_ref,
        authority_revision=authority_revision,
        schema_digest=schema_digest,
        authority_digest=authority_digest_for(
            tenant,
            publisher,
            package_name,
            source_name,
            source_kind,
            source_ref,
            root_ref,
            authority_revision,
            schema_digest,
        ),
    )


def _entry(
    authority: SourceAuthority,
    path: str = "issues/1.json",
    *,
    revision: int = 1,
    content_char: str = "a",
) -> SourceCatalogEntry:
    content_digest = _digest(content_char)
    artifact_ref = f"artifact:gitlab:{content_char}"
    schema_digest = _digest("b")
    entry_kind: EntryKind = "record"
    size_bytes = 42
    return SourceCatalogEntry(
        entry_version="source-catalog-entry.v1",
        entry_id=entry_id_for(authority.authority_id, path),
        authority_id=authority.authority_id,
        tenant_id=authority.tenant_id,
        relative_path=path,
        entry_revision=revision,
        entry_kind=entry_kind,
        content_digest=content_digest,
        artifact_ref=artifact_ref,
        schema_digest=schema_digest,
        size_bytes=size_bytes,
        entry_digest=catalog_entry_digest_for(
            authority.authority_id,
            authority.tenant_id,
            path,
            revision,
            entry_kind,
            content_digest,
            artifact_ref,
            schema_digest,
            size_bytes,
        ),
    )


def _manifest(
    authority: SourceAuthority,
    entries: tuple[SourceCatalogEntry, ...],
    *,
    revision: int = 1,
    status: ManifestStatus = "complete",
    verified_empty: bool = False,
) -> SourceManifest:
    empty_ref = "evidence:empty:1" if verified_empty else None
    empty_digest = _digest("e") if verified_empty else None
    ordered_entries = tuple(sorted(entries, key=lambda entry: entry.entry_id))
    digest = manifest_digest_for(
        authority.authority_id,
        authority.authority_digest,
        authority.tenant_id,
        revision,
        status,
        f"2026-08-19T00:00:0{revision}Z",
        False,
        ordered_entries,
        verified_empty,
        empty_ref,
        empty_digest,
    )
    return SourceManifest(
        manifest_version="source-manifest.v1",
        manifest_id=manifest_id_for(authority.authority_id, revision, digest),
        authority_id=authority.authority_id,
        authority_digest=authority.authority_digest,
        tenant_id=authority.tenant_id,
        manifest_revision=revision,
        observation_status=status,
        observed_at=f"2026-08-19T00:00:0{revision}Z",
        dirty=False,
        entries=ordered_entries,
        verified_empty=verified_empty,
        empty_evidence_ref=empty_ref,
        empty_evidence_digest=empty_digest,
        manifest_digest=digest,
    )


def _outcome(
    entry: SourceCatalogEntry,
    outcome: EntryOutcome,
    *,
    expected: str | None = None,
    observed: str | None = None,
    absence_verification: AbsenceVerification = "not_applicable",
    terminal: bool = True,
) -> EntryReconciliation:
    return EntryReconciliation(
        outcome_version="source-entry-reconciliation.v1",
        entry_id=entry.entry_id,
        authority_id=entry.authority_id,
        tenant_id=entry.tenant_id,
        relative_path=entry.relative_path,
        outcome=outcome,
        expected_entry_digest=expected,
        observed_entry_digest=observed,
        evidence_ref=f"evidence:entry:{entry.entry_revision}:{outcome}",
        evidence_digest=_digest("d"),
        absence_verification=absence_verification,
        terminal=terminal,
    )


@dataclass
class _MemoryRepository:
    authorities: dict[str, SourceAuthority] = field(default_factory=dict)
    entries: dict[tuple[str, int], SourceCatalogEntry] = field(default_factory=dict)
    manifests: dict[str, SourceManifest] = field(default_factory=dict)
    reconciliations: dict[str, SourceReconciliation] = field(default_factory=dict)
    checkpoint: SourceCheckpoint | None = None

    def put_authority(self, authority: SourceAuthority) -> None:
        prior = self.authorities.get(authority.authority_id)
        if prior is not None and prior != authority:
            raise RepositoryContractError("authority mutation")
        self.authorities[authority.authority_id] = authority

    def put_catalog_entry(self, entry: SourceCatalogEntry) -> None:
        key = (entry.entry_id, entry.entry_revision)
        prior = self.entries.get(key)
        if prior is not None and prior != entry:
            raise RepositoryContractError("entry mutation")
        self.entries[key] = entry

    def put_manifest(self, manifest: SourceManifest) -> None:
        prior = self.manifests.get(manifest.manifest_id)
        if prior is not None and prior != manifest:
            raise RepositoryContractError("manifest mutation")
        self.manifests[manifest.manifest_id] = manifest

    def put_reconciliation(self, reconciliation: SourceReconciliation) -> None:
        reconciliation_id = reconciliation.reconciliation_id
        prior = self.reconciliations.get(reconciliation_id)
        if prior is not None and prior != reconciliation:
            raise RepositoryContractError("reconciliation mutation")
        self.reconciliations[reconciliation_id] = reconciliation

    def get_authority(
        self,
        scope: SourceScope,
        authority_id: str,
        authority_digest: str | None = None,
    ) -> SourceAuthority | None:
        del scope
        authority = self.authorities.get(authority_id)
        if authority_digest is not None and authority is not None:
            return authority if authority.authority_digest == authority_digest else None
        return authority

    def get_manifest(
        self, scope: SourceScope, manifest_id: str
    ) -> SourceManifest | None:
        del scope
        return self.manifests.get(manifest_id)

    def get_catalog_entry(
        self,
        scope: SourceScope,
        entry_id: str,
        entry_revision: int | None = None,
    ) -> SourceCatalogEntry | None:
        del scope
        values = [
            entry
            for (candidate_id, candidate_revision), entry in self.entries.items()
            if candidate_id == entry_id
            and (entry_revision is None or candidate_revision == entry_revision)
        ]
        return max(values, key=lambda entry: entry.entry_revision) if values else None

    def get_reconciliation(
        self, scope: SourceScope, reconciliation_id: str
    ) -> SourceReconciliation | None:
        del scope
        return self.reconciliations.get(reconciliation_id)

    def get_checkpoint(
        self, scope: SourceScope, authority_id: str
    ) -> SourceCheckpoint | None:
        del scope, authority_id
        return self.checkpoint

    def compare_and_swap_checkpoint(
        self,
        scope: SourceScope,
        mutation: CheckpointMutation,
        checkpoint: SourceCheckpoint,
    ) -> SourceCheckpoint:
        del scope
        current_revision = self.checkpoint.revision if self.checkpoint else 0
        current_id = self.checkpoint.checkpoint_id if self.checkpoint else None
        if (
            mutation.expected_revision != current_revision
            or mutation.expected_checkpoint_id != current_id
        ):
            raise RepositoryContractError("checkpoint CAS conflict")
        self.checkpoint = checkpoint
        return checkpoint


def _fixture() -> tuple[
    _MemoryRepository,
    SourceControlPlane,
    SourceScope,
    SourceAuthority,
    SourceCatalogEntry,
    SourceManifest,
]:
    authority = _authority()
    entry = _entry(authority)
    manifest = _manifest(authority, (entry,))
    repository = _MemoryRepository()
    plane = SourceControlPlane(repository)
    plane.register(
        SourceRegistration(
            registration_version="source-registration.v1",
            authority=authority,
            manifest=manifest,
        )
    )
    return repository, plane, _scope(), authority, entry, manifest


def _request(
    scope: SourceScope,
    authority: SourceAuthority,
    manifest: SourceManifest,
    outcomes: tuple[EntryReconciliation, ...],
    *,
    selected: tuple[str, ...] | None = None,
    expected_revision: int = 0,
    expected_checkpoint_id: str | None = None,
) -> SourceReconciliationRequest:
    return SourceReconciliationRequest(
        request_version="source-reconciliation-request.v1",
        scope=scope,
        authority_id=authority.authority_id,
        manifest_id=manifest.manifest_id,
        selected_entry_ids=selected
        or tuple(sorted(outcome.entry_id for outcome in outcomes)),
        outcomes=outcomes,
        expected_checkpoint_revision=expected_revision,
        expected_checkpoint_id=expected_checkpoint_id,
        change_ref="change:source-sync:1",
    )


def test_path_identity_and_raw_or_dirty_source_content_fail_closed() -> None:
    authority = _authority()
    with pytest.raises(ValueError):
        _entry(authority, "../escape.json")
    with pytest.raises(ValueError):
        _entry(authority, "/absolute.json")
    with pytest.raises(ValidationError):
        SourceCatalogEntry(
            **{
                **_entry(authority).model_dump(mode="json"),
                "dirty": True,
            }
        )
    with pytest.raises(ValidationError):
        SourceCatalogEntry(
            **{
                **_entry(authority).model_dump(mode="json"),
                "artifact_ref": 'body:{"raw":true}',
            }
        )


def test_complete_reconcile_advances_once_and_replays_deterministically() -> None:
    repository, plane, scope, authority, entry, manifest = _fixture()
    outcome = _outcome(entry, "added", observed=entry.entry_digest)
    request = _request(scope, authority, manifest, (outcome,))
    first = plane.reconcile(request)
    assert first.checkpoint is not None
    assert first.checkpoint.revision == 1
    replay = plane.reconcile(request)
    assert replay == first
    assert repository.checkpoint == first.checkpoint


def test_failed_partial_timeout_evidence_holds_checkpoint() -> None:
    repository, plane, scope, authority, entry, manifest = _fixture()
    outcome = _outcome(entry, "timeout", terminal=False)
    result = plane.reconcile(_request(scope, authority, manifest, (outcome,)))
    assert result.checkpoint is None
    assert result.reconciliation.complete is False
    assert repository.checkpoint is None


def test_empty_incomplete_manifest_is_retained_but_cannot_advance() -> None:
    repository, plane, scope, authority, _, _ = _fixture()
    failed_manifest = _manifest(authority, (), revision=2, status="timeout")
    repository.put_manifest(failed_manifest)
    with pytest.raises(SourceReconciliationError, match="source_manifest_not_complete"):
        plane.reconcile(_request(scope, authority, failed_manifest, ()))
    assert repository.checkpoint is None


def test_verified_tombstone_requires_prior_digest_and_absence_evidence() -> None:
    repository, plane, scope, authority, entry, _ = _fixture()
    empty_manifest = _manifest(authority, (), revision=2, verified_empty=True)
    repository.put_manifest(empty_manifest)
    outcome = _outcome(
        entry,
        "tombstoned",
        expected=entry.entry_digest,
        absence_verification="verified",
    )
    result = plane.reconcile(
        _request(
            scope,
            authority,
            empty_manifest,
            (outcome,),
            selected=(entry.entry_id,),
        )
    )
    assert result.checkpoint is not None
    assert result.checkpoint.terminal_entry_count == 1


def test_stale_checkpoint_and_digest_drift_are_denied() -> None:
    repository, plane, scope, authority, entry, manifest = _fixture()
    result = plane.reconcile(
        _request(
            scope,
            authority,
            manifest,
            (_outcome(entry, "added", observed=entry.entry_digest),),
        )
    )
    assert result.checkpoint is not None
    with pytest.raises(SourceReconciliationError, match="checkpoint_revision_conflict"):
        changed = _outcome(
            entry,
            "updated",
            expected=_digest("9"),
            observed=entry.entry_digest,
        )
        plane.reconcile(
            _request(
                scope,
                authority,
                manifest,
                (changed,),
            )
        )
    empty_manifest = _manifest(authority, (), revision=2, verified_empty=True)
    repository.put_manifest(empty_manifest)
    with pytest.raises(SourceReconciliationError, match="tombstone_digest_drift"):
        plane.reconcile(
            _request(
                scope,
                authority,
                empty_manifest,
                (
                    _outcome(
                        entry,
                        "tombstoned",
                        expected=_digest("9"),
                        absence_verification="verified",
                    ),
                ),
                selected=(entry.entry_id,),
            )
        )
    with pytest.raises(ValidationError):
        _outcome(
            entry,
            "updated",
            expected=entry.entry_digest,
            observed=entry.entry_digest,
        )


def test_projection_is_bounded_and_scope_mismatch_is_rejected() -> None:
    _, plane, scope, authority, entry, manifest = _fixture()
    result = plane.reconcile(
        _request(
            scope,
            authority,
            manifest,
            (_outcome(entry, "added", observed=entry.entry_digest),),
        )
    )
    projection = plane.project(scope, authority.authority_id, manifest.manifest_id)
    assert projection.checkpoint_revision == 1
    assert "raw" not in projection.model_dump_json()
    assert result.checkpoint is not None
    entry_projection = plane.project_reconciliation(
        scope, result.reconciliation.reconciliation_id
    )[0]
    assert entry.relative_path not in entry_projection.model_dump_json()
    with pytest.raises(RepositoryContractError):
        plane.project(
            _scope("tenant:other"), authority.authority_id, manifest.manifest_id
        )
