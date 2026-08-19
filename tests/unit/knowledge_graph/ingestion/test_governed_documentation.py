"""Bounded tests for the governed Markdown projection (NE-143)."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ingestion.governed_documentation import (
    DocumentationLifecycle,
    DocumentationProjectionError,
    GovernedDocumentationProjector,
    project_markdown,
)
from agent_utilities.protocols.source_connectors.base import ExternalAccess


REVISION_1 = "a" * 40
REVISION_2 = "b" * 40
RECORDED_AT = "2026-08-19T12:00:00Z"


def test_projection_is_metadata_only_and_carries_provenance() -> None:
    projection = project_markdown(
        "repo-1",
        "docs/security.md",
        REVISION_1,
        "# Security\n\nCONCEPT:AU-KG.retrieval.acl-aware-vector-retrieval\n\nsecret body",
        valid_time=RECORDED_AT,
        recorded_at=RECORDED_AT,
        source_acl=ExternalAccess.public(),
    )

    assert projection.lifecycle == DocumentationLifecycle.CURRENT
    assert projection.current is True
    assert projection.concept_ids == (
        "AU-KG.retrieval.acl-aware-vector-retrieval",
    )
    envelope = projection.to_envelope()
    assert envelope.source_version == REVISION_1
    assert envelope.source_acl is not None and envelope.source_acl.is_public
    assert envelope.typed_payload is not None
    assert "content" not in envelope.typed_payload
    assert "text" not in envelope.typed_payload
    assert "secret body" not in repr(envelope.provenance)
    evidence = envelope.typed_payload["_evidence"][0]
    assert "content" not in evidence["locus"]
    assert evidence["content_digest"].startswith("sha256:")
    assert envelope.typed_payload["acl_before_retrieval"] is True


def test_superseded_document_is_never_current() -> None:
    projection = project_markdown(
        "repo-1",
        "docs/old.md",
        REVISION_1,
        "---\nstatus: superseded\nsuperseded_by: docs/new.md\n---\n# Old\n",
        valid_time=RECORDED_AT,
        recorded_at=RECORDED_AT,
    )

    assert projection.lifecycle == DocumentationLifecycle.SUPERSEDED
    assert projection.current is False
    assert projection.archived is True
    assert projection.to_envelope().typed_payload["status"] == "archived"  # type: ignore[index]


def test_rebuild_requires_verified_snapshot_for_tombstones() -> None:
    source = {
        "repository_id": "repo-1",
        "source_path": "docs/old.md",
        "source_revision": REVISION_1,
        "content": "# Old\n",
        "recorded_at": RECORDED_AT,
        "valid_time": RECORDED_AT,
    }
    projector = GovernedDocumentationProjector()
    with pytest.raises(DocumentationProjectionError, match="verified snapshot"):
        projector.rebuild(
            [],
            previous_sources=[source],
            snapshot_revision=REVISION_2,
            snapshot_verified=False,
            recorded_at=RECORDED_AT,
        )

    rebuilt = projector.rebuild(
        [],
        previous_sources=[source],
        snapshot_revision=REVISION_2,
        snapshot_verified=True,
        recorded_at=RECORDED_AT,
    )
    assert len(rebuilt.tombstones) == 1
    tombstone = rebuilt.tombstones[0]
    assert tombstone.lifecycle == DocumentationLifecycle.TOMBSTONED
    assert tombstone.current is False
    assert tombstone.tombstone_verified is True
    assert tombstone.to_envelope().operation == "delete"


def test_rebuild_is_deterministic_for_fixed_inputs() -> None:
    source = {
        "repository_id": "repo-1",
        "source_path": "docs/guide.md",
        "source_revision": REVISION_1,
        "content": "# Guide\n\nCONCEPT:AU-KG.ingest.change-envelope\n",
        "recorded_at": RECORDED_AT,
        "valid_time": RECORDED_AT,
    }
    projector = GovernedDocumentationProjector()
    first = projector.rebuild([source], recorded_at=RECORDED_AT)
    second = projector.rebuild([source], recorded_at=RECORDED_AT)
    assert first.snapshot_digest == second.snapshot_digest
    first_envelope = first.envelopes[0]
    second_envelope = second.envelopes[0]
    assert first_envelope.idempotency_key == second_envelope.idempotency_key
    assert first_envelope.source_object_id == second_envelope.source_object_id
    assert first_envelope.typed_payload == second_envelope.typed_payload
