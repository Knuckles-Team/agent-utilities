"""Strict source-authority, manifest, and checkpoint contracts.

The source control plane records *what* was observed and *why* it was safe to
advance a checkpoint.  It does not read a filesystem, retain source bytes, or
decide how an adapter fetches data.  Content is represented only by bounded
opaque artifact references and digests; the authoritative ingestion path stays
outside this domain module.

(CONCEPT:AU-KG.ingest.fleet-catalog-relational-tables,
AU-OS.governance.fail-closed-degraded-read,
AU-OS.governance.verified-write-state-advance)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from agent_utilities.protocols.epistemic_operations import ProtocolModel

Identifier: TypeAlias = Annotated[
    str,
    Field(
        min_length=1,
        max_length=192,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$",
    ),
]
Digest: TypeAlias = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
VersionText: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]*$"),
]
Timestamp: TypeAlias = Annotated[
    str,
    Field(
        min_length=20,
        max_length=64,
        pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|z|[+-][0-9]{2}:[0-9]{2})$",
    ),
]
PathText: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=2048, pattern=r"^[^\x00]+$"),
]

SourceKind = Literal[
    "git", "filesystem", "api", "database", "object_store", "workspace"
]
ManifestStatus = Literal["complete", "partial", "failed", "timeout"]
EntryKind = Literal["file", "directory", "record", "object"]
EntryOutcome = Literal[
    "added", "updated", "unchanged", "tombstoned", "failed", "partial", "timeout"
]
AbsenceVerification = Literal["not_applicable", "verified"]

_MAX_ENTRIES = 10_000
_MAX_GRANTS = 64
_MAX_ENTRY_EVIDENCE = 10_000
_FORBIDDEN_REF_PREFIXES = (
    "http:",
    "https:",
    "file:",
    "env:",
    "secret:",
    "vault:",
    "body:",
    "result:",
    "data:",
    "base64:",
)
_INLINE_MARKERS = (
    "password=",
    "secret=",
    "token=",
    "authorization:",
    "-----begin",
    '{"',
    "[{",
)


def _canonical(value: str) -> str:
    return value.strip().casefold()


def _digest_payload(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _opaque_ref(value: str, field_name: str) -> str:
    lowered = value.casefold()
    if lowered.startswith(_FORBIDDEN_REF_PREFIXES) or any(
        marker in lowered for marker in _INLINE_MARKERS
    ):
        raise ValueError(f"{field_name} must be an opaque controlled reference")
    return value


def _is_unsafe_relative_path_shape(value: str) -> bool:
    return (
        not value
        or "\\" in value
        or "\x00" in value
        or value.startswith("/")
        or PurePosixPath(value).is_absolute()
    )


def _validate_relative_path(value: str) -> str:
    """Accept one repository-relative POSIX path and reject escape aliases."""

    if _is_unsafe_relative_path_shape(value):
        raise ValueError("source identity path must be repository-relative POSIX text")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("source identity path contains an unsafe segment")
    if ":" in parts[0]:
        raise ValueError("source identity path must not contain a drive prefix")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("source identity path contains a control character")
    return value


def scope_digest_for(
    tenant_id: str, principal_id: str, grant_digests: Iterable[str]
) -> str:
    return _digest_payload(
        {
            "tenant_id": tenant_id,
            "principal_id": principal_id,
            "grant_digests": sorted(set(grant_digests)),
        }
    )


def authority_id_for(
    tenant_id: str, publisher: str, package_name: str, source_name: str
) -> str:
    payload = "\x1f".join(
        (
            _canonical(tenant_id),
            _canonical(publisher),
            _canonical(package_name),
            _canonical(source_name),
        )
    )
    return "source-authority:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def authority_digest_for(
    tenant_id: str,
    publisher: str,
    package_name: str,
    source_name: str,
    source_kind: SourceKind,
    source_ref: str,
    root_ref: str | None,
    authority_revision: int,
    schema_digest: str | None,
) -> str:
    return _digest_payload(
        {
            "tenant_id": tenant_id,
            "publisher": publisher,
            "package_name": package_name,
            "source_name": source_name,
            "source_kind": source_kind,
            "source_ref": source_ref,
            "root_ref": root_ref,
            "authority_revision": authority_revision,
            "schema_digest": schema_digest,
        }
    )


def entry_id_for(authority_id: str, relative_path: str) -> str:
    _validate_relative_path(relative_path)
    return (
        "source-entry:"
        + hashlib.sha256(
            "\x1f".join((authority_id, relative_path)).encode("utf-8")
        ).hexdigest()
    )


def catalog_entry_digest_for(
    authority_id: str,
    tenant_id: str,
    relative_path: str,
    entry_revision: int,
    entry_kind: EntryKind,
    content_digest: str,
    artifact_ref: str,
    schema_digest: str | None,
    size_bytes: int,
) -> str:
    return _digest_payload(
        {
            "authority_id": authority_id,
            "tenant_id": tenant_id,
            "relative_path": relative_path,
            "entry_revision": entry_revision,
            "entry_kind": entry_kind,
            "content_digest": content_digest,
            "artifact_ref": artifact_ref,
            "schema_digest": schema_digest,
            "size_bytes": size_bytes,
        }
    )


def manifest_digest_for(
    authority_id: str,
    authority_digest: str,
    tenant_id: str,
    manifest_revision: int,
    observation_status: ManifestStatus,
    observed_at: str,
    dirty: bool,
    entries: Iterable[SourceCatalogEntry],
    verified_empty: bool,
    empty_evidence_ref: str | None,
    empty_evidence_digest: str | None,
) -> str:
    return _digest_payload(
        {
            "authority_id": authority_id,
            "authority_digest": authority_digest,
            "tenant_id": tenant_id,
            "manifest_revision": manifest_revision,
            "observation_status": observation_status,
            "observed_at": observed_at,
            "dirty": dirty,
            "entries": [
                entry.model_dump(mode="json")
                for entry in sorted(entries, key=lambda item: item.entry_id)
            ],
            "verified_empty": verified_empty,
            "empty_evidence_ref": empty_evidence_ref,
            "empty_evidence_digest": empty_evidence_digest,
        }
    )


def manifest_id_for(
    authority_id: str, manifest_revision: int, manifest_digest: str
) -> str:
    return (
        "source-manifest:"
        + hashlib.sha256(
            "\x1f".join((authority_id, str(manifest_revision), manifest_digest)).encode(
                "utf-8"
            )
        ).hexdigest()
    )


def reconciliation_digest_for(
    authority_id: str,
    tenant_id: str,
    manifest_id: str,
    manifest_digest: str,
    selected_entry_ids: Iterable[str],
    outcomes: Iterable[EntryReconciliation],
) -> str:
    return _digest_payload(
        {
            "authority_id": authority_id,
            "tenant_id": tenant_id,
            "manifest_id": manifest_id,
            "manifest_digest": manifest_digest,
            "selected_entry_ids": sorted(selected_entry_ids),
            "outcomes": [
                outcome.model_dump(mode="json")
                for outcome in sorted(outcomes, key=lambda item: item.entry_id)
            ],
        }
    )


def reconciliation_id_for(
    authority_id: str, manifest_id: str, reconciliation_digest: str
) -> str:
    return (
        "source-reconciliation:"
        + hashlib.sha256(
            "\x1f".join((authority_id, manifest_id, reconciliation_digest)).encode(
                "utf-8"
            )
        ).hexdigest()
    )


def checkpoint_digest_for(
    authority_id: str,
    tenant_id: str,
    revision: int,
    manifest_id: str,
    manifest_revision: int,
    manifest_digest: str,
    reconciliation_id: str,
    terminal_entry_count: int,
) -> str:
    return _digest_payload(
        {
            "authority_id": authority_id,
            "tenant_id": tenant_id,
            "revision": revision,
            "manifest_id": manifest_id,
            "manifest_revision": manifest_revision,
            "manifest_digest": manifest_digest,
            "reconciliation_id": reconciliation_id,
            "terminal_entry_count": terminal_entry_count,
        }
    )


def checkpoint_id_for(authority_id: str, revision: int, checkpoint_digest: str) -> str:
    return (
        "source-checkpoint:"
        + hashlib.sha256(
            "\x1f".join((authority_id, str(revision), checkpoint_digest)).encode(
                "utf-8"
            )
        ).hexdigest()
    )


def path_digest_for(relative_path: str) -> str:
    return _digest_payload({"relative_path": _validate_relative_path(relative_path)})


class SourceScope(ProtocolModel):
    """Tenant/principal/grant scope required for every repository operation."""

    scope_version: Literal["source-scope.v1"]
    tenant_id: Identifier
    principal_id: Identifier
    grant_digests: tuple[Digest, ...] = Field(default=(), max_length=_MAX_GRANTS)
    scope_digest: Digest

    @model_validator(mode="after")
    def scope_is_self_consistent(self) -> SourceScope:
        if len(set(self.grant_digests)) != len(self.grant_digests):
            raise ValueError("source scope grants must be unique")
        if tuple(sorted(self.grant_digests)) != self.grant_digests:
            raise ValueError("source scope grants must be sorted")
        if self.scope_digest != scope_digest_for(
            self.tenant_id, self.principal_id, self.grant_digests
        ):
            raise ValueError("source scope digest does not match its subject")
        return self


class SourceAuthority(ProtocolModel):
    """Stable, tenant-scoped source authority record."""

    authority_version: Literal["source-authority.v1"]
    authority_id: Identifier
    tenant_id: Identifier
    publisher: Identifier
    package_name: Identifier
    source_name: Identifier
    source_kind: SourceKind
    source_ref: Identifier
    root_ref: Identifier | None = None
    authority_revision: int = Field(ge=1)
    schema_digest: Digest | None = None
    authority_digest: Digest

    @model_validator(mode="after")
    def authority_is_stable_and_opaque(self) -> SourceAuthority:
        if self.authority_id != authority_id_for(
            self.tenant_id, self.publisher, self.package_name, self.source_name
        ):
            raise ValueError("source authority id is not stable")
        _opaque_ref(self.source_ref, "source_ref")
        if self.root_ref is not None:
            _opaque_ref(self.root_ref, "root_ref")
        expected = authority_digest_for(
            self.tenant_id,
            self.publisher,
            self.package_name,
            self.source_name,
            self.source_kind,
            self.source_ref,
            self.root_ref,
            self.authority_revision,
            self.schema_digest,
        )
        if self.authority_digest != expected:
            raise ValueError("source authority digest does not match its record")
        return self


class SourceCatalogEntry(ProtocolModel):
    """Immutable digest-pinned catalog entry; never a raw source payload."""

    entry_version: Literal["source-catalog-entry.v1"]
    entry_id: Identifier
    authority_id: Identifier
    tenant_id: Identifier
    relative_path: PathText
    entry_revision: int = Field(ge=1)
    entry_kind: EntryKind
    content_digest: Digest
    artifact_ref: Identifier
    schema_digest: Digest | None = None
    size_bytes: int = Field(ge=0, le=10**12)
    is_symlink: bool = False
    dirty: bool = False
    entry_digest: Digest

    @model_validator(mode="after")
    def entry_is_path_safe_and_immutable(self) -> SourceCatalogEntry:
        _validate_relative_path(self.relative_path)
        if self.is_symlink:
            raise ValueError("symlink source entries are not admissible")
        if self.dirty:
            raise ValueError("dirty source entries are not admissible")
        _opaque_ref(self.artifact_ref, "artifact_ref")
        if self.entry_id != entry_id_for(self.authority_id, self.relative_path):
            raise ValueError("source catalog entry id is not path-derived")
        expected = catalog_entry_digest_for(
            self.authority_id,
            self.tenant_id,
            self.relative_path,
            self.entry_revision,
            self.entry_kind,
            self.content_digest,
            self.artifact_ref,
            self.schema_digest,
            self.size_bytes,
        )
        if self.entry_digest != expected:
            raise ValueError("source catalog entry digest does not match its record")
        return self


class SourceManifest(ProtocolModel):
    """Immutable bounded source snapshot with explicit empty-source proof."""

    manifest_version: Literal["source-manifest.v1"]
    manifest_id: Identifier
    authority_id: Identifier
    authority_digest: Digest
    tenant_id: Identifier
    manifest_revision: int = Field(ge=1)
    entries: tuple[SourceCatalogEntry, ...] = Field(max_length=_MAX_ENTRIES)
    observation_status: ManifestStatus
    observed_at: Timestamp
    dirty: bool = False
    verified_empty: bool = False
    empty_evidence_ref: Identifier | None = None
    empty_evidence_digest: Digest | None = None
    manifest_digest: Digest

    @property
    def entry_ids(self) -> tuple[str, ...]:
        return tuple(entry.entry_id for entry in self.entries)

    def _check_entry_scope_consistency(self) -> None:
        if any(entry.authority_id != self.authority_id for entry in self.entries):
            raise ValueError("source manifest contains a cross-authority entry")
        if any(entry.tenant_id != self.tenant_id for entry in self.entries):
            raise ValueError("source manifest contains a cross-tenant entry")

    def _check_entries_normalized(self) -> None:
        if self.dirty:
            raise ValueError("dirty source manifests are not admissible")
        if len(set(self.entry_ids)) != len(self.entry_ids):
            raise ValueError("source manifest entry identities must be unique")
        if (
            tuple(sorted(self.entries, key=lambda entry: entry.entry_id))
            != self.entries
        ):
            raise ValueError("source manifest entries must be sorted")

    def _check_non_empty_manifest_carries_no_empty_evidence(self) -> None:
        if self.verified_empty or self.empty_evidence_ref is not None:
            raise ValueError("non-empty source manifest cannot claim verified empty")
        if self.empty_evidence_digest is not None:
            raise ValueError("non-empty source manifest cannot carry empty evidence")

    def _check_empty_manifest_evidence_matches_status(self) -> None:
        if self.observation_status == "complete":
            if (
                not self.verified_empty
                or self.empty_evidence_ref is None
                or self.empty_evidence_digest is None
            ):
                raise ValueError(
                    "complete empty source manifest requires verified evidence"
                )
        elif (
            self.verified_empty
            or self.empty_evidence_ref is not None
            or self.empty_evidence_digest is not None
        ):
            raise ValueError(
                "incomplete empty source manifest cannot claim verified empty"
            )

    def _check_empty_evidence_consistency(self) -> None:
        if self.entries:
            self._check_non_empty_manifest_carries_no_empty_evidence()
            return
        self._check_empty_manifest_evidence_matches_status()

    def _check_manifest_digest_and_id(self) -> None:
        if self.empty_evidence_ref is not None:
            _opaque_ref(self.empty_evidence_ref, "empty_evidence_ref")
        expected_digest = manifest_digest_for(
            self.authority_id,
            self.authority_digest,
            self.tenant_id,
            self.manifest_revision,
            self.observation_status,
            self.observed_at,
            self.dirty,
            self.entries,
            self.verified_empty,
            self.empty_evidence_ref,
            self.empty_evidence_digest,
        )
        if self.manifest_digest != expected_digest:
            raise ValueError("source manifest digest does not match its contents")
        if self.manifest_id != manifest_id_for(
            self.authority_id, self.manifest_revision, self.manifest_digest
        ):
            raise ValueError("source manifest id is not content-derived")

    @model_validator(mode="after")
    def manifest_is_normalized_and_self_consistent(self) -> SourceManifest:
        self._check_entry_scope_consistency()
        self._check_entries_normalized()
        self._check_empty_evidence_consistency()
        self._check_manifest_digest_and_id()
        return self


class EntryReconciliation(ProtocolModel):
    """One evidence-backed terminal or incomplete reconciliation outcome."""

    outcome_version: Literal["source-entry-reconciliation.v1"]
    entry_id: Identifier
    authority_id: Identifier
    tenant_id: Identifier
    relative_path: PathText
    outcome: EntryOutcome
    expected_entry_digest: Digest | None = None
    observed_entry_digest: Digest | None = None
    evidence_ref: Identifier
    evidence_digest: Digest
    absence_verification: AbsenceVerification = "not_applicable"
    terminal: bool

    def _check_added_outcome_evidence(self) -> None:
        if self.expected_entry_digest is not None or self.observed_entry_digest is None:
            raise ValueError("added entry evidence must contain only observed digest")

    def _check_updated_outcome_evidence(self) -> None:
        if (
            self.expected_entry_digest is None
            or self.observed_entry_digest is None
            or self.expected_entry_digest == self.observed_entry_digest
        ):
            raise ValueError("updated entry evidence must prove digest drift")

    def _check_unchanged_outcome_evidence(self) -> None:
        if (
            self.expected_entry_digest is None
            or self.observed_entry_digest != self.expected_entry_digest
        ):
            raise ValueError("unchanged entry evidence must match its prior digest")

    def _check_tombstoned_outcome_evidence(self) -> None:
        if (
            self.expected_entry_digest is None
            or self.observed_entry_digest is not None
            or self.absence_verification != "verified"
        ):
            raise ValueError("tombstone requires verified absence evidence")

    def _check_incomplete_outcome_evidence(self) -> None:
        if (
            self.expected_entry_digest is not None
            or self.observed_entry_digest is not None
        ):
            raise ValueError("incomplete entry outcomes cannot carry a digest")
        if self.absence_verification != "not_applicable":
            raise ValueError("incomplete entry outcomes cannot verify absence")

    @model_validator(mode="after")
    def outcome_is_explicit_and_fail_closed(self) -> EntryReconciliation:
        _validate_relative_path(self.relative_path)
        _opaque_ref(self.evidence_ref, "evidence_ref")
        expected_terminal = self.outcome in {
            "added",
            "updated",
            "unchanged",
            "tombstoned",
        }
        if self.terminal != expected_terminal:
            raise ValueError("entry reconciliation terminal state is inconsistent")
        outcome_checks = {
            "added": self._check_added_outcome_evidence,
            "updated": self._check_updated_outcome_evidence,
            "unchanged": self._check_unchanged_outcome_evidence,
            "tombstoned": self._check_tombstoned_outcome_evidence,
        }
        outcome_checks.get(self.outcome, self._check_incomplete_outcome_evidence)()
        if (
            self.outcome != "tombstoned"
            and self.absence_verification != "not_applicable"
        ):
            raise ValueError("only tombstones may carry absence verification")
        return self


class SourceReconciliation(ProtocolModel):
    """Deterministic run evidence used for replay and checkpoint admission."""

    reconciliation_version: Literal["source-reconciliation.v1"]
    reconciliation_id: Identifier
    authority_id: Identifier
    tenant_id: Identifier
    manifest_id: Identifier
    manifest_digest: Digest
    selected_entry_ids: tuple[Identifier, ...] = Field(max_length=_MAX_ENTRIES)
    outcomes: tuple[EntryReconciliation, ...] = Field(max_length=_MAX_ENTRY_EVIDENCE)
    terminal_entry_count: int = Field(ge=0, le=_MAX_ENTRIES)
    complete: bool
    reconciliation_digest: Digest

    def _check_selected_entries_normalized(self) -> None:
        if len(set(self.selected_entry_ids)) != len(self.selected_entry_ids):
            raise ValueError("selected source entries must be unique")
        if tuple(sorted(self.selected_entry_ids)) != self.selected_entry_ids:
            raise ValueError("selected source entries must be sorted")

    def _check_outcomes_normalized_and_cover_selection(self) -> None:
        outcome_ids = tuple(outcome.entry_id for outcome in self.outcomes)
        if len(set(outcome_ids)) != len(outcome_ids):
            raise ValueError("source reconciliation outcomes must be unique")
        if (
            tuple(sorted(self.outcomes, key=lambda outcome: outcome.entry_id))
            != self.outcomes
        ):
            raise ValueError("source reconciliation outcomes must be sorted")
        if set(outcome_ids) != set(self.selected_entry_ids):
            raise ValueError("source reconciliation outcomes must cover the selection")

    def _check_outcomes_scope_and_terminal_count(self) -> None:
        if any(
            outcome.authority_id != self.authority_id
            or outcome.tenant_id != self.tenant_id
            for outcome in self.outcomes
        ):
            raise ValueError("source reconciliation contains a cross-scope outcome")
        terminal_count = sum(outcome.terminal for outcome in self.outcomes)
        if self.terminal_entry_count != terminal_count:
            raise ValueError("source reconciliation terminal count is inconsistent")
        if self.complete != (terminal_count == len(self.selected_entry_ids)):
            raise ValueError("source reconciliation completion is inconsistent")

    def _check_reconciliation_digest_and_id(self) -> None:
        expected_digest = reconciliation_digest_for(
            self.authority_id,
            self.tenant_id,
            self.manifest_id,
            self.manifest_digest,
            self.selected_entry_ids,
            self.outcomes,
        )
        if self.reconciliation_digest != expected_digest:
            raise ValueError("source reconciliation digest does not match its evidence")
        if self.reconciliation_id != reconciliation_id_for(
            self.authority_id, self.manifest_id, self.reconciliation_digest
        ):
            raise ValueError("source reconciliation id is not content-derived")

    @model_validator(mode="after")
    def reconciliation_is_complete_or_honestly_incomplete(
        self,
    ) -> SourceReconciliation:
        self._check_selected_entries_normalized()
        self._check_outcomes_normalized_and_cover_selection()
        self._check_outcomes_scope_and_terminal_count()
        self._check_reconciliation_digest_and_id()
        return self


class CheckpointMutation(ProtocolModel):
    """Revision-fenced checkpoint CAS request."""

    mutation_version: Literal["source-checkpoint-mutation.v1"]
    authority_id: Identifier
    tenant_id: Identifier
    expected_revision: int = Field(ge=0)
    expected_checkpoint_id: Identifier | None = None
    manifest_id: Identifier
    manifest_revision: int = Field(ge=1)
    manifest_digest: Digest
    reconciliation_id: Identifier
    change_ref: Identifier

    @model_validator(mode="after")
    def change_ref_is_opaque(self) -> CheckpointMutation:
        _opaque_ref(self.change_ref, "change_ref")
        return self


class SourceCheckpoint(ProtocolModel):
    """Only a complete evidence set can create this durable watermark."""

    checkpoint_version: Literal["source-checkpoint.v1"]
    checkpoint_id: Identifier
    authority_id: Identifier
    tenant_id: Identifier
    revision: int = Field(ge=1)
    manifest_id: Identifier
    manifest_revision: int = Field(ge=1)
    manifest_digest: Digest
    reconciliation_id: Identifier
    selected_entry_count: int = Field(ge=0, le=_MAX_ENTRIES)
    terminal_entry_count: int = Field(ge=0, le=_MAX_ENTRIES)
    checkpoint_digest: Digest

    @model_validator(mode="after")
    def checkpoint_is_complete_and_content_addressed(self) -> SourceCheckpoint:
        if self.selected_entry_count != self.terminal_entry_count:
            raise ValueError("source checkpoint cannot advance with incomplete entries")
        expected_digest = checkpoint_digest_for(
            self.authority_id,
            self.tenant_id,
            self.revision,
            self.manifest_id,
            self.manifest_revision,
            self.manifest_digest,
            self.reconciliation_id,
            self.terminal_entry_count,
        )
        if self.checkpoint_digest != expected_digest:
            raise ValueError("source checkpoint digest does not match its evidence")
        if self.checkpoint_id != checkpoint_id_for(
            self.authority_id, self.revision, self.checkpoint_digest
        ):
            raise ValueError("source checkpoint id is not content-derived")
        return self


class SourceReconciliationRequest(ProtocolModel):
    """Bounded replayable request; no source bytes or error bodies are accepted."""

    request_version: Literal["source-reconciliation-request.v1"]
    scope: SourceScope
    authority_id: Identifier
    manifest_id: Identifier
    selected_entry_ids: tuple[Identifier, ...] = Field(max_length=_MAX_ENTRIES)
    outcomes: tuple[EntryReconciliation, ...] = Field(max_length=_MAX_ENTRY_EVIDENCE)
    expected_checkpoint_revision: int = Field(ge=0)
    expected_checkpoint_id: Identifier | None = None
    change_ref: Identifier

    @model_validator(mode="after")
    def request_is_normalized(self) -> SourceReconciliationRequest:
        if len(set(self.selected_entry_ids)) != len(self.selected_entry_ids):
            raise ValueError("selected source entries must be unique")
        if tuple(sorted(self.selected_entry_ids)) != self.selected_entry_ids:
            raise ValueError("selected source entries must be sorted")
        if self.expected_checkpoint_id is not None:
            _opaque_ref(self.expected_checkpoint_id, "expected_checkpoint_id")
        _opaque_ref(self.change_ref, "change_ref")
        return self


class SourceReconcileResult(ProtocolModel):
    """Stable replay result; checkpoint is absent when evidence is incomplete."""

    result_version: Literal["source-reconcile-result.v1"]
    reconciliation: SourceReconciliation
    checkpoint: SourceCheckpoint | None = None
    result_digest: Digest

    @model_validator(mode="after")
    def result_is_self_consistent(self) -> SourceReconcileResult:
        if not self.reconciliation.complete and self.checkpoint is not None:
            raise ValueError("incomplete reconciliation cannot carry a checkpoint")
        if self.checkpoint is not None and (
            self.checkpoint.reconciliation_id != self.reconciliation.reconciliation_id
            or self.checkpoint.manifest_id != self.reconciliation.manifest_id
        ):
            raise ValueError(
                "source result checkpoint is not bound to its reconciliation"
            )
        expected = _digest_payload(
            {
                "reconciliation": self.reconciliation.model_dump(mode="json"),
                "checkpoint": self.checkpoint.model_dump(mode="json")
                if self.checkpoint is not None
                else None,
            }
        )
        if self.result_digest != expected:
            raise ValueError(
                "source reconcile result digest does not match its evidence"
            )
        return self


class SourceGraphProjection(ProtocolModel):
    """Bounded redacted manifest summary suitable for graph projection."""

    projection_version: Literal["source-graph-projection.v1"]
    tenant_id: Identifier
    authority_id: Identifier
    manifest_id: Identifier
    observation_status: ManifestStatus
    verified_empty: bool
    entry_count: int = Field(ge=0, le=_MAX_ENTRIES)
    terminal_entry_count: int = Field(ge=0, le=_MAX_ENTRIES)
    checkpoint_revision: int = Field(ge=0)
    manifest_digest: Digest
    reconciliation_id: Identifier | None = None


class SourceEntryProjection(ProtocolModel):
    """Per-entry projection with path redacted to a stable digest."""

    projection_version: Literal["source-entry-projection.v1"]
    tenant_id: Identifier
    authority_id: Identifier
    entry_id: Identifier
    path_digest: Digest
    outcome: EntryOutcome
    evidence_digest: Digest
    tombstone_verified: bool

    @model_validator(mode="after")
    def projection_is_redacted_and_consistent(self) -> SourceEntryProjection:
        if self.tombstone_verified != (self.outcome == "tombstoned"):
            raise ValueError("source entry tombstone projection is inconsistent")
        return self


class SourceRegistration(ProtocolModel):
    """Atomic authority/manifest registration bundle."""

    registration_version: Literal["source-registration.v1"]
    authority: SourceAuthority
    manifest: SourceManifest

    @model_validator(mode="after")
    def registration_is_consistent(self) -> SourceRegistration:
        if self.manifest.authority_id != self.authority.authority_id:
            raise ValueError("source manifest belongs to another authority")
        if self.manifest.authority_digest != self.authority.authority_digest:
            raise ValueError("source manifest authority digest is stale")
        if self.manifest.tenant_id != self.authority.tenant_id:
            raise ValueError("source manifest belongs to another tenant")
        return self
