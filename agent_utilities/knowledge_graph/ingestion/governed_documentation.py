from __future__ import annotations

"""Governed, metadata-only projection of repository Markdown.

CONCEPT:AU-KG.ingest.governed-documentation-projection.

Markdown remains the authoring authority.  This module only extracts the
bounded facts needed to make a repository document addressable and queryable:
the repository-relative path, the exact source revision, a content digest,
Concept IDs, provenance, ACL, and lifecycle/temporal state.  It deliberately
does not put the Markdown body into a ``ChangeEnvelope`` or its evidence.

The resulting envelope is consumed by the public native ingestion seam
(:func:`envelope_ingest.ingest_envelope`).  There is no private persistence
path and no graph-to-Markdown editor.  A verified rebuild can emit tombstones
for paths removed from an authoritative snapshot; an incomplete or failed
fetch cannot.
"""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_utilities.models.company_brain import DataClassification
from agent_utilities.protocols.source_connectors.base import ExternalAccess

from .change_envelope import ChangeEnvelope

__all__ = [
    "DOCUMENTATION_MAPPING_VERSION",
    "DOCUMENTATION_SCHEMA_VERSION",
    "DocumentationLifecycle",
    "DocumentationProjectionBatch",
    "DocumentationProjectionError",
    "DocumentationSource",
    "DocumentationEvidence",
    "GovernedDocumentationProjection",
    "GovernedDocumentationProjector",
    "extract_documentation",
    "project_markdown",
    "rebuild_documentation",
]


DOCUMENTATION_SCHEMA_VERSION = "1"
DOCUMENTATION_MAPPING_VERSION = "governed-markdown-v1"
_MAX_REPOSITORY_ID = 128
_MAX_PATH = 1024
_MAX_REVISION = 128
_MAX_CONCEPT_IDS = 64
_MAX_CONCEPT_ID = 160
_MAX_SUPERSEDER = 512

_REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$", re.ASCII)
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$", re.ASCII)
_OPAQUE_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$", re.ASCII)
_CONCEPT_MARKER_RE = re.compile(
    r"\bCONCEPT\s*:\s*([A-Za-z0-9][A-Za-z0-9._/-]{0,159})",
    re.ASCII,
)
_CONCEPT_FRONTMATTER_RE = re.compile(
    r"^concept(?:_ids|s)?\s*:\s*(.*?)\s*$", re.IGNORECASE
)
_FRONTMATTER_KEY_RE = re.compile(
    r"^(?P<key>[A-Za-z][A-Za-z0-9_-]{0,63})\s*:\s*(?P<value>.*?)\s*$"
)
_ISO_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)


class DocumentationProjectionError(ValueError):
    """A source cannot be admitted to the governed documentation projection."""


class DocumentationLifecycle(StrEnum):
    """Explicit state used by retrieval and temporal reconciliation."""

    CURRENT = "current"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    TOMBSTONED = "tombstoned"

    @property
    def is_current(self) -> bool:
        # Deprecated material is still the current revision of its source page;
        # callers can choose whether to include it without confusing it with a
        # removed/superseded page.
        return self in {
            DocumentationLifecycle.CURRENT,
            DocumentationLifecycle.DEPRECATED,
        }

    @property
    def is_archived(self) -> bool:
        return self in {
            DocumentationLifecycle.SUPERSEDED,
            DocumentationLifecycle.ARCHIVED,
            DocumentationLifecycle.TOMBSTONED,
        }


def _canonical_timestamp(value: str, *, field_name: str) -> str:
    raw = str(value or "").strip()
    if not raw or len(raw) > 64 or not _ISO_TIMESTAMP_RE.fullmatch(raw):
        raise DocumentationProjectionError(
            f"{field_name} must be an ISO-8601 timestamp with timezone"
        )
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DocumentationProjectionError(
            f"{field_name} is not a valid ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise DocumentationProjectionError(f"{field_name} must include a timezone")
    parsed = parsed.astimezone(UTC)
    timespec = "microseconds" if parsed.microsecond else "seconds"
    return parsed.isoformat(timespec=timespec).replace("+00:00", "Z")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_repository_id(value: str) -> str:
    raw = str(value or "").strip()
    if (
        not raw
        or len(raw) > _MAX_REPOSITORY_ID
        or not _OPAQUE_REPOSITORY_RE.fullmatch(raw)
    ):
        raise DocumentationProjectionError(
            "repository_id must be a bounded opaque repository identifier"
        )
    if ".." in raw or raw.startswith(("/", "\\")):
        raise DocumentationProjectionError("repository_id must not contain a path")
    return raw


def _validate_source_path(value: str) -> str:
    raw = str(value or "").strip()
    if not raw or len(raw) > _MAX_PATH or "\x00" in raw:
        raise DocumentationProjectionError("source_path is empty or too long")
    if "\\" in raw:
        raise DocumentationProjectionError("source_path must use POSIX separators")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DocumentationProjectionError(
            "source_path must be repository-relative and normalized"
        )
    normalized = path.as_posix()
    if normalized != raw:
        raise DocumentationProjectionError("source_path must be normalized POSIX text")
    return normalized


def _validate_revision(value: str) -> str:
    raw = str(value or "").strip()
    if len(raw) > _MAX_REVISION or not _REVISION_RE.fullmatch(raw):
        raise DocumentationProjectionError(
            "source_revision must be an exact lowercase Git SHA-1 or SHA-256"
        )
    return raw


def _digest_content(content: str) -> str:
    if not isinstance(content, str):
        raise DocumentationProjectionError("Markdown content must be text")
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


def _stable_id(prefix: str, *parts: str, length: int = 40) -> str:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(material).hexdigest()[:length]}"


def _parse_frontmatter(markdown: str) -> dict[str, str]:
    """Read only scalar frontmatter keys; never evaluate YAML."""
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    values: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        match = _FRONTMATTER_KEY_RE.match(line.strip())
        if not match:
            continue
        key = match.group("key").casefold()
        value = match.group("value").strip().strip("'\"")
        if len(value) <= _MAX_SUPERSEDER:
            values[key] = value
    return values


def _split_frontmatter_values(value: str) -> list[str]:
    rendered = str(value or "").strip().strip("[]")
    if not rendered:
        return []
    return [item.strip().strip("'\"") for item in rendered.split(",") if item.strip()]


def _extract_concept_ids(
    markdown: str, frontmatter: Mapping[str, str]
) -> tuple[str, ...]:
    candidates = list(_CONCEPT_MARKER_RE.findall(markdown))
    for key, value in frontmatter.items():
        if _CONCEPT_FRONTMATTER_RE.match(f"{key}: {value}"):
            candidates.extend(_split_frontmatter_values(value))
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        rendered = str(candidate).strip()
        if (
            rendered
            and len(rendered) <= _MAX_CONCEPT_ID
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", rendered, re.ASCII)
            and rendered not in seen
        ):
            seen.add(rendered)
            result.append(rendered)
        if len(result) >= _MAX_CONCEPT_IDS:
            break
    return tuple(result)


def _lifecycle_from_frontmatter(
    frontmatter: Mapping[str, str],
) -> tuple[DocumentationLifecycle, str | None, tuple[str, ...]]:
    raw_status = ""
    for key in ("lifecycle", "state", "status"):
        if frontmatter.get(key):
            raw_status = frontmatter[key].strip().casefold()
            break
    if frontmatter.get("archived", "").strip().casefold() in {"1", "true", "yes"}:
        raw_status = "archived"
    elif frontmatter.get("deprecated", "").strip().casefold() in {
        "1",
        "true",
        "yes",
    } and raw_status not in {"archived", "superseded", "tombstoned"}:
        raw_status = "deprecated"
    superseded_by = frontmatter.get("superseded_by") or frontmatter.get("superseded-by")
    superseded_by = superseded_by.strip() if superseded_by else None
    supersedes = tuple(
        _validate_source_path(item)
        for item in _split_frontmatter_values(frontmatter.get("supersedes", ""))
        if item
    )
    aliases = {
        "": DocumentationLifecycle.CURRENT,
        "current": DocumentationLifecycle.CURRENT,
        "active": DocumentationLifecycle.CURRENT,
        "live": DocumentationLifecycle.CURRENT,
        "published": DocumentationLifecycle.CURRENT,
        "deprecated": DocumentationLifecycle.DEPRECATED,
        "deprecation": DocumentationLifecycle.DEPRECATED,
        "superseded": DocumentationLifecycle.SUPERSEDED,
        "archived": DocumentationLifecycle.ARCHIVED,
        "history": DocumentationLifecycle.ARCHIVED,
        "tombstoned": DocumentationLifecycle.TOMBSTONED,
    }
    if raw_status not in aliases:
        raise DocumentationProjectionError(
            f"unsupported documentation lifecycle {raw_status!r}"
        )
    lifecycle = aliases[raw_status]
    if superseded_by:
        lifecycle = DocumentationLifecycle.SUPERSEDED
        if len(superseded_by) > _MAX_SUPERSEDER:
            raise DocumentationProjectionError("superseded_by is too long")
    return lifecycle, superseded_by, supersedes


def _classification_for(access: ExternalAccess) -> DataClassification:
    return (
        DataClassification.PUBLIC if access.is_public else DataClassification.INTERNAL
    )


class DocumentationSource(BaseModel):
    """One source-authority record held in memory during extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository_id: str
    source_path: str
    source_revision: str
    content: str
    valid_time: str | None = None
    recorded_at: str | None = None
    source_acl: ExternalAccess = Field(default_factory=ExternalAccess.quarantined)
    classification: DataClassification | None = None
    connector: str = "governed_documentation"
    source_instance: str = ""

    @field_validator("repository_id")
    @classmethod
    def _repository(cls, value: str) -> str:
        return _validate_repository_id(value)

    @field_validator("source_path")
    @classmethod
    def _path(cls, value: str) -> str:
        return _validate_source_path(value)

    @field_validator("source_revision")
    @classmethod
    def _revision(cls, value: str) -> str:
        return _validate_revision(value)

    @field_validator("valid_time", "recorded_at")
    @classmethod
    def _timestamp(cls, value: str | None, info: Any) -> str | None:
        return (
            None
            if value is None
            else _canonical_timestamp(value, field_name=str(info.field_name))
        )

    @field_validator("connector", "source_instance")
    @classmethod
    def _bounded_text(cls, value: str, info: Any) -> str:
        rendered = str(value or "").strip()
        if len(rendered) > 128 or "\x00" in rendered:
            raise DocumentationProjectionError(f"{info.field_name} is too long")
        return rendered


class DocumentationEvidence(BaseModel):
    """Privacy-safe evidence for one documentation revision.

    The model intentionally has no body/content field.  ``content_digest`` and
    ``source_ref`` let a verifier retrieve and compare the authoring source
    without copying potentially private text into graph evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    revision_id: str
    repository_id: str
    source_path: str
    source_revision: str
    content_digest: str
    concept_ids: tuple[str, ...] = ()
    lifecycle: DocumentationLifecycle
    valid_from: str
    recorded_at: str
    source_ref: str
    authority: str = "repository_markdown"
    schema_version: str = DOCUMENTATION_SCHEMA_VERSION
    tombstone_verified: bool = False
    superseded_by: str | None = None

    _MAX_SOURCE_REF: ClassVar[int] = 1_536

    @field_validator("content_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not _DIGEST_RE.fullmatch(str(value)):
            raise DocumentationProjectionError("content_digest must be sha256:<hex>")
        return str(value)

    @field_validator("source_ref")
    @classmethod
    def _source_ref(cls, value: str) -> str:
        rendered = str(value or "")
        if not rendered or len(rendered) > cls._MAX_SOURCE_REF:
            raise DocumentationProjectionError("source_ref is missing or too long")
        return rendered


class GovernedDocumentationProjection(BaseModel):
    """Metadata-only graph projection and its public envelope translation."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    document_id: str
    revision_id: str
    evidence_id: str
    repository_id: str
    source_path: str
    source_revision: str
    content_digest: str
    concept_ids: tuple[str, ...] = ()
    lifecycle: DocumentationLifecycle = DocumentationLifecycle.CURRENT
    current: bool = True
    deprecated: bool = False
    archived: bool = False
    superseded_by: str | None = None
    supersedes_paths: tuple[str, ...] = ()
    valid_from: str
    recorded_at: str
    source_ref: str
    source_acl: ExternalAccess = Field(default_factory=ExternalAccess.quarantined)
    classification: DataClassification = DataClassification.INTERNAL
    connector: str = "governed_documentation"
    source_instance: str = ""
    operation: str = "upsert"
    previous_revision: str | None = None
    previous_digest: str | None = None
    tombstone_verified: bool = False
    snapshot_digest: str | None = None
    tombstone_reason: str | None = None

    @model_validator(mode="after")
    def _invariants(self) -> GovernedDocumentationProjection:
        if self.operation not in {"upsert", "delete"}:
            raise DocumentationProjectionError("documentation operation is invalid")
        if (
            self.operation == "delete"
            and self.lifecycle != DocumentationLifecycle.TOMBSTONED
        ):
            raise DocumentationProjectionError("delete projection must be tombstoned")
        if self.lifecycle.is_archived and self.current:
            raise DocumentationProjectionError(
                "superseded, archived, and tombstoned documentation cannot be current"
            )
        if self.lifecycle == DocumentationLifecycle.DEPRECATED and not self.deprecated:
            raise DocumentationProjectionError(
                "deprecated projection must be marked deprecated"
            )
        if self.operation == "delete" and not self.tombstone_verified:
            raise DocumentationProjectionError("tombstones require verified provenance")
        return self

    def _primary_payload(self) -> dict[str, Any]:
        status = self.lifecycle.value
        # Existing retrieval code excludes status=ARCHIVED by default.  Keep
        # the richer lifecycle state while ensuring a superseded/archived page
        # cannot be ranked as current by a legacy status filter.
        retrieval_status = status if self.current else "archived"
        payload: dict[str, Any] = {
            "id": self.document_id,
            "node_type": "DocumentationPage",
            "repository_id": self.repository_id,
            "source_path": self.source_path,
            "source_kind": "markdown",
            "corpus": self.source_instance or self.repository_id,
            "relpath": self.source_path,
            "source_ref": self.source_ref,
            "source_revision": self.source_revision,
            "content_digest": self.content_digest,
            "concept_ids": list(self.concept_ids),
            "lifecycle_state": status,
            "status": retrieval_status,
            "current": self.current,
            "deprecated": self.deprecated,
            "archived": self.archived,
            "valid_from": self.valid_from,
            "recorded_at": self.recorded_at,
            "documentation_schema_version": DOCUMENTATION_SCHEMA_VERSION,
            "ontology_mapping_version": DOCUMENTATION_MAPPING_VERSION,
            "acl_verified": True,
            "acl_before_retrieval": True,
        }
        if self.superseded_by:
            payload["superseded_by"] = self.superseded_by
        if self.snapshot_digest:
            payload["snapshot_digest"] = self.snapshot_digest
        if self.tombstone_reason:
            payload["tombstone_reason"] = self.tombstone_reason
        return payload

    def _auxiliary_nodes(self) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = [
            {
                "id": self.revision_id,
                "node_type": "DocumentationRevision",
                "document_id": self.document_id,
                "repository_id": self.repository_id,
                "source_path": self.source_path,
                "source_revision": self.source_revision,
                "content_digest": self.content_digest,
                "concept_ids": list(self.concept_ids),
                "lifecycle_state": self.lifecycle.value,
                "current": self.current,
                "valid_from": self.valid_from,
                "recorded_at": self.recorded_at,
            }
        ]
        if self.previous_revision:
            previous_digest = self.previous_digest or ""
            if not _DIGEST_RE.fullmatch(previous_digest):
                raise DocumentationProjectionError(
                    "previous_digest is required when previous_revision is supplied"
                )
            nodes.append(
                {
                    "id": _stable_id(
                        "doc-revision",
                        self.document_id,
                        self.previous_revision,
                        previous_digest,
                    ),
                    "node_type": "DocumentationRevision",
                    "document_id": self.document_id,
                    "repository_id": self.repository_id,
                    "source_path": self.source_path,
                    "source_revision": self.previous_revision,
                    "content_digest": previous_digest,
                    "lifecycle_state": DocumentationLifecycle.SUPERSEDED.value,
                    "current": False,
                    "archived": True,
                    "valid_until": self.valid_from,
                    "recorded_at": self.recorded_at,
                }
            )
        return nodes

    def _evidence(self) -> DocumentationEvidence:
        return DocumentationEvidence(
            evidence_id=self.evidence_id,
            revision_id=self.revision_id,
            repository_id=self.repository_id,
            source_path=self.source_path,
            source_revision=self.source_revision,
            content_digest=self.content_digest,
            concept_ids=self.concept_ids,
            lifecycle=self.lifecycle,
            valid_from=self.valid_from,
            recorded_at=self.recorded_at,
            source_ref=self.source_ref,
            tombstone_verified=self.tombstone_verified,
            superseded_by=self.superseded_by,
        )

    def to_envelope(self) -> ChangeEnvelope:
        """Translate through the canonical public ``ChangeEnvelope`` contract."""
        evidence = self._evidence()
        provenance = {
            "kind": "governed_documentation",
            "repository_id": self.repository_id,
            "source_path": self.source_path,
            "relpath": self.source_path,
            "corpus": self.source_instance or self.repository_id,
            "source_ref": self.source_ref,
            "source_revision": self.source_revision,
            "content_digest": self.content_digest,
            "concept_ids": list(self.concept_ids),
            "lifecycle_state": self.lifecycle.value,
            "valid_from": self.valid_from,
            "recorded_at": self.recorded_at,
            "evidence_id": self.evidence_id,
            "revision_id": self.revision_id,
            "tombstone_verified": self.tombstone_verified,
            "acl_before_retrieval": True,
        }
        if self.connector == "git_markdown":
            provenance["git_commit"] = self.source_revision
        if self.previous_revision:
            provenance["previous_revision"] = self.previous_revision
            provenance["previous_digest"] = self.previous_digest
        if self.snapshot_digest:
            provenance["snapshot_digest"] = self.snapshot_digest
        if self.tombstone_reason:
            provenance["tombstone_reason"] = self.tombstone_reason
        if self.operation == "delete":
            # Delete envelopes have no row payload, so carry the same
            # metadata-only evidence through provenance for the native ingest
            # seam to project into its durable evidence table.
            provenance["evidence"] = [
                {
                    "evidence_id": evidence.evidence_id,
                    "object_id": self.document_id,
                    "modality": "documentation_tombstone",
                    "locus": evidence.model_dump(mode="json"),
                    "content_digest": self.content_digest,
                }
            ]
        payload = None
        if self.operation == "upsert":
            payload = self._primary_payload()
            payload["_nodes"] = self._auxiliary_nodes()
            payload["_evidence"] = [
                {
                    "evidence_id": evidence.evidence_id,
                    "object_id": self.document_id,
                    "modality": "documentation_provenance",
                    "locus": evidence.model_dump(mode="json"),
                    "content_digest": self.content_digest,
                }
            ]
        return ChangeEnvelope(
            connector=self.connector,
            operation=self.operation,  # type: ignore[arg-type]
            source_instance=self.source_instance or self.repository_id,
            source_object_id=self.document_id,
            source_version=self.source_revision,
            event_time=self.valid_from,
            valid_time=self.valid_from,
            observed_time=self.recorded_at,
            schema_version=DOCUMENTATION_SCHEMA_VERSION,
            ontology_mapping_version=DOCUMENTATION_MAPPING_VERSION,
            payload_type="documentation",
            typed_payload=payload,
            source_acl=self.source_acl,
            classification=self.classification,
            provenance=provenance,
            checkpoint=self.source_revision,
        )

    def commit(self, engine: Any) -> Any:
        """Commit through the public native ingestion seam."""
        from .envelope_ingest import ingest_envelope

        return ingest_envelope(engine, self.to_envelope())


def _projection_from_source(
    source: DocumentationSource,
    *,
    document_id: str | None = None,
    previous_revision: str | None = None,
    previous_digest: str | None = None,
) -> GovernedDocumentationProjection:
    frontmatter = _parse_frontmatter(source.content)
    lifecycle, superseded_by, supersedes_paths = _lifecycle_from_frontmatter(
        frontmatter
    )
    recorded_at = source.recorded_at or _now_iso()
    valid_from = source.valid_time or recorded_at
    digest = _digest_content(source.content)
    page_id = document_id or _stable_id(
        "doc-page", source.repository_id, source.source_path
    )
    revision_id = _stable_id("doc-revision", page_id, source.source_revision, digest)
    evidence_id = _stable_id(
        "doc-evidence", revision_id, source.repository_id, source.source_path
    )
    source_ref = (
        f"repo://{source.repository_id}/{source.source_path}@{source.source_revision}"
    )
    return GovernedDocumentationProjection(
        document_id=page_id,
        revision_id=revision_id,
        evidence_id=evidence_id,
        repository_id=source.repository_id,
        source_path=source.source_path,
        source_revision=source.source_revision,
        content_digest=digest,
        concept_ids=_extract_concept_ids(source.content, frontmatter),
        lifecycle=lifecycle,
        current=lifecycle.is_current,
        deprecated=lifecycle == DocumentationLifecycle.DEPRECATED,
        archived=lifecycle.is_archived,
        superseded_by=superseded_by,
        supersedes_paths=supersedes_paths,
        valid_from=valid_from,
        recorded_at=recorded_at,
        source_ref=source_ref,
        source_acl=source.source_acl,
        classification=source.classification or _classification_for(source.source_acl),
        connector=source.connector,
        source_instance=source.source_instance,
        previous_revision=previous_revision,
        previous_digest=previous_digest,
    )


def project_markdown(
    repository_id: str,
    source_path: str,
    source_revision: str,
    content: str,
    *,
    valid_time: str | None = None,
    recorded_at: str | None = None,
    source_acl: ExternalAccess | None = None,
    classification: DataClassification | None = None,
    connector: str = "governed_documentation",
    source_instance: str = "",
    document_id: str | None = None,
    previous_revision: str | None = None,
    previous_digest: str | None = None,
) -> GovernedDocumentationProjection:
    """Extract one bounded, privacy-safe Markdown projection."""
    source = DocumentationSource(
        repository_id=repository_id,
        source_path=source_path,
        source_revision=source_revision,
        content=content,
        valid_time=valid_time,
        recorded_at=recorded_at,
        source_acl=source_acl or ExternalAccess.quarantined(),
        classification=classification,
        connector=connector,
        source_instance=source_instance,
    )
    if previous_revision is not None:
        previous_revision = _validate_revision(previous_revision)
    if previous_digest is not None and not _DIGEST_RE.fullmatch(previous_digest):
        raise DocumentationProjectionError("previous_digest must be sha256:<hex>")
    return _projection_from_source(
        source,
        document_id=document_id,
        previous_revision=previous_revision,
        previous_digest=previous_digest,
    )


def extract_documentation(*args: Any, **kwargs: Any) -> GovernedDocumentationProjection:
    """Compatibility spelling for :func:`project_markdown`."""
    return project_markdown(*args, **kwargs)


@dataclass(frozen=True)
class DocumentationProjectionBatch:
    """Deterministic rebuild result; tombstones precede current upserts."""

    projections: tuple[GovernedDocumentationProjection, ...]
    tombstones: tuple[GovernedDocumentationProjection, ...]
    snapshot_digest: str
    snapshot_verified: bool

    @property
    def envelopes(self) -> tuple[ChangeEnvelope, ...]:
        return tuple(
            projection.to_envelope()
            for projection in (*self.tombstones, *self.projections)
        )


class GovernedDocumentationProjector:
    """Small deterministic projector for full, verified Markdown snapshots."""

    def __init__(
        self,
        *,
        connector: str = "governed_documentation",
        source_instance: str = "",
    ) -> None:
        self.connector = connector
        self.source_instance = source_instance

    def project(
        self, source: DocumentationSource | Mapping[str, Any]
    ) -> GovernedDocumentationProjection:
        record = (
            source
            if isinstance(source, DocumentationSource)
            else DocumentationSource.model_validate(source)
        )
        if (
            record.connector == "governed_documentation"
            and self.connector != record.connector
        ):
            record = record.model_copy(update={"connector": self.connector})
        if self.source_instance and not record.source_instance:
            record = record.model_copy(update={"source_instance": self.source_instance})
        return _projection_from_source(record)

    def rebuild(
        self,
        sources: Iterable[DocumentationSource | Mapping[str, Any]],
        *,
        previous_sources: Iterable[DocumentationSource | Mapping[str, Any]] = (),
        previous_source_paths: Iterable[str] = (),
        snapshot_revision: str | None = None,
        snapshot_verified: bool = False,
        recorded_at: str | None = None,
    ) -> DocumentationProjectionBatch:
        records = [
            source
            if isinstance(source, DocumentationSource)
            else DocumentationSource.model_validate(source)
            for source in sources
        ]
        records.sort(key=lambda item: (item.repository_id, item.source_path))
        current_keys = [(item.repository_id, item.source_path) for item in records]
        if len(current_keys) != len(set(current_keys)):
            raise DocumentationProjectionError(
                "rebuild contains duplicate source paths"
            )
        prior = [
            source
            if isinstance(source, DocumentationSource)
            else DocumentationSource.model_validate(source)
            for source in previous_sources
        ]
        prior_by_key = {(item.repository_id, item.source_path): item for item in prior}
        explicit_paths = {_validate_source_path(path) for path in previous_source_paths}
        prior_keys = set(prior_by_key)
        if explicit_paths:
            repository_ids = {item.repository_id for item in records} | {
                item.repository_id for item in prior
            }
            if len(repository_ids) != 1:
                raise DocumentationProjectionError(
                    "previous_source_paths require one repository in a rebuild"
                )
            repository_id = next(iter(repository_ids))
            prior_keys.update((repository_id, path) for path in explicit_paths)
        if snapshot_verified:
            revisions = {item.source_revision for item in records}
            if snapshot_revision is None and len(revisions) == 1:
                snapshot_revision = next(iter(revisions))
            if snapshot_revision is None:
                raise DocumentationProjectionError(
                    "snapshot_revision is required for a verified documentation rebuild"
                )
            snapshot_revision = _validate_revision(snapshot_revision)
        snapshot_time = recorded_at or _now_iso()
        snapshot_time = _canonical_timestamp(snapshot_time, field_name="recorded_at")
        projections: list[GovernedDocumentationProjection] = []
        for record in records:
            old = prior_by_key.get((record.repository_id, record.source_path))
            prior_revision = (
                old.source_revision
                if old and old.source_revision != record.source_revision
                else None
            )
            prior_digest = (
                _digest_content(old.content) if prior_revision and old else None
            )
            if record.recorded_at is None or record.valid_time is None:
                record = record.model_copy(
                    update={
                        "recorded_at": record.recorded_at or snapshot_time,
                        "valid_time": record.valid_time or snapshot_time,
                    }
                )
            projections.append(
                self.project_with_history(
                    record,
                    previous_revision=prior_revision,
                    previous_digest=prior_digest,
                )
            )
        current_key_set = set(current_keys)
        superseded_keys = {
            (projection.repository_id, path)
            for projection in projections
            for path in projection.supersedes_paths
        }
        if superseded_keys & current_key_set:
            raise DocumentationProjectionError(
                "a superseded documentation path is still present as current"
            )
        removed_keys = sorted((prior_keys | superseded_keys) - current_key_set)
        if removed_keys and not snapshot_verified:
            raise DocumentationProjectionError(
                "verified snapshot is required before emitting documentation tombstones"
            )
        snapshot_material = [
            {
                "repository_id": projection.repository_id,
                "source_path": projection.source_path,
                "source_revision": projection.source_revision,
                "content_digest": projection.content_digest,
                "lifecycle": projection.lifecycle.value,
            }
            for projection in projections
        ]
        snapshot_payload = json.dumps(
            {
                "revision": snapshot_revision or "",
                "records": snapshot_material,
                "removed": removed_keys,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        snapshot_digest = f"sha256:{hashlib.sha256(snapshot_payload).hexdigest()}"
        tombstones: list[GovernedDocumentationProjection] = []
        for repository_id, path in removed_keys:
            prior_record = prior_by_key.get((repository_id, path))
            previous_revision = prior_record.source_revision if prior_record else ""
            previous_digest = (
                _digest_content(prior_record.content) if prior_record else ""
            )
            tombstone_revision = snapshot_revision or ""
            tombstones.append(
                self.tombstone(
                    repository_id,
                    path,
                    tombstone_revision,
                    recorded_at=snapshot_time,
                    previous_revision=previous_revision or None,
                    previous_digest=previous_digest or None,
                    snapshot_digest=snapshot_digest,
                    reason="removed_from_verified_snapshot",
                )
            )
        return DocumentationProjectionBatch(
            projections=tuple(projections),
            tombstones=tuple(tombstones),
            snapshot_digest=snapshot_digest,
            snapshot_verified=snapshot_verified,
        )

    def project_with_history(
        self,
        source: DocumentationSource,
        *,
        previous_revision: str | None = None,
        previous_digest: str | None = None,
    ) -> GovernedDocumentationProjection:
        if (
            source.connector == "governed_documentation"
            and self.connector != source.connector
        ):
            source = source.model_copy(update={"connector": self.connector})
        if self.source_instance and not source.source_instance:
            source = source.model_copy(update={"source_instance": self.source_instance})
        return _projection_from_source(
            source,
            previous_revision=previous_revision,
            previous_digest=previous_digest,
        )

    def tombstone(
        self,
        repository_id: str,
        source_path: str,
        source_revision: str,
        *,
        recorded_at: str,
        document_id: str | None = None,
        previous_revision: str | None = None,
        previous_digest: str | None = None,
        snapshot_digest: str | None = None,
        reason: str = "removed_from_verified_snapshot",
    ) -> GovernedDocumentationProjection:
        revision = _validate_revision(source_revision)
        repository = _validate_repository_id(repository_id)
        path = _validate_source_path(source_path)
        timestamp = _canonical_timestamp(recorded_at, field_name="recorded_at")
        digest = previous_digest or "sha256:" + "0" * 64
        if not _DIGEST_RE.fullmatch(digest):
            raise DocumentationProjectionError("tombstone previous_digest is invalid")
        page_id = document_id or _stable_id("doc-page", repository, path)
        revision_id = _stable_id("doc-revision", page_id, revision, digest)
        evidence_id = _stable_id("doc-evidence", revision_id, repository, path)
        return GovernedDocumentationProjection(
            document_id=page_id,
            revision_id=revision_id,
            evidence_id=evidence_id,
            repository_id=repository,
            source_path=path,
            source_revision=revision,
            content_digest=digest,
            lifecycle=DocumentationLifecycle.TOMBSTONED,
            current=False,
            deprecated=False,
            archived=True,
            valid_from=timestamp,
            recorded_at=timestamp,
            source_ref=f"repo://{repository}/{path}@{revision}",
            source_acl=ExternalAccess.quarantined(),
            classification=DataClassification.INTERNAL,
            connector=self.connector,
            source_instance=self.source_instance,
            operation="delete",
            previous_revision=previous_revision,
            previous_digest=previous_digest,
            tombstone_verified=True,
            snapshot_digest=snapshot_digest,
            tombstone_reason=reason,
        )


def rebuild_documentation(
    sources: Iterable[DocumentationSource | Mapping[str, Any]],
    **kwargs: Any,
) -> DocumentationProjectionBatch:
    """Functional facade for a deterministic governed rebuild."""
    projector = GovernedDocumentationProjector(
        connector=str(kwargs.pop("connector", "governed_documentation")),
        source_instance=str(kwargs.pop("source_instance", "")),
    )
    return projector.rebuild(sources, **kwargs)
