"""Bounded, read-only quarantine inventory for pre-EH-509 Section rows.

Legacy ``Section`` rows have no trustworthy served tenant stamp or complete
tree marker. This inventory helps an authorized owner locate source documents
for governed re-ingestion; it never promotes an old row or infers authority
from its properties.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ...protocols.source_connectors.base import ExternalAccess
from ..core.session import resolve_session

_MAX_PAGE = 1_000
_MAX_REPLAY_DOCUMENTS = 16
_MAX_SOURCE_BYTES = 2 * 1024 * 1024
_QUERY = (
    "MATCH (s:Section) WHERE s.id > $after_id "
    "RETURN s.id AS id, s.document_id AS document_id "
    "ORDER BY id LIMIT $limit"
)


@dataclass(frozen=True)
class LegacySectionQuarantineReport:
    """One authorized page, never a proof that the legacy corpus is complete."""

    document_ids: tuple[str, ...]
    sampled_sections: int
    unattributed_sections: int
    last_seen_id: str
    possibly_more: bool
    state: str = "quarantined_requires_attested_source_replay"
    complete: bool = False


@dataclass(frozen=True)
class AttestedSectionSource:
    """Material returned by a trusted source resolver, never a legacy row."""

    document_id: str
    tenant: str
    source_ref: str
    text: str
    content_sha256: str
    external_access: ExternalAccess
    connector: str
    source_instance: str = ""
    checkpoint: str | None = None


class AttestedSectionSourceResolver(Protocol):
    """Production binding must re-fetch bytes and ACL from an authorized source."""

    def resolve(self, document_id: str, *, session: Any) -> AttestedSectionSource | None: ...


@dataclass(frozen=True)
class LegacySectionReplayReport:
    replayed_document_ids: tuple[str, ...]
    quarantined_document_ids: tuple[str, ...]


def _source_is_bound(source: Any, document_id: str, tenant: str) -> bool:
    """Reject incomplete or mismatched source evidence before native ingest."""
    return (
        isinstance(source, AttestedSectionSource)
        and source.document_id == document_id
        and source.tenant == tenant
        and isinstance(source.source_ref, str)
        and 0 < len(source.source_ref) <= 512
        and isinstance(source.connector, str)
        and 0 < len(source.connector) <= 128
        and isinstance(source.text, str)
        and len(source.text) <= _MAX_SOURCE_BYTES
        and isinstance(source.external_access, ExternalAccess)
        and len(source.text.encode("utf-8")) <= _MAX_SOURCE_BYTES
        and hashlib.sha256(source.text.encode("utf-8")).hexdigest() == source.content_sha256
    )


def replay_attested_legacy_sections(
    document_ids: Sequence[str],
    *,
    resolver: AttestedSectionSourceResolver,
    processor: Any,
) -> LegacySectionReplayReport:
    """Rebuild bounded documents from source, never from old Section properties.

    ``document_ids`` may come from the quarantine inventory; they confer no
    authority. The resolver is a process-owned source adapter and must attest
    source bytes, version/identity and ACL independently of legacy rows. Until
    a production resolver is bound, this function has no public tool route.
    Native DocumentProcessor enforces the verified session and commits each
    complete document/section tree atomically. A write error propagates so the
    caller cannot mistake a partially completed batch for full success.
    """
    session = resolve_session(required_scope="kg:write")
    session.require_scope("kg:read")
    if not session.graph:
        raise PermissionError("section replay requires a verified graph")
    from ..ontology.document_processing import DocumentProcessor

    if not isinstance(processor, DocumentProcessor) or processor.engine is None:
        raise TypeError("native DocumentProcessor authority is required")
    if len(document_ids) > _MAX_REPLAY_DOCUMENTS or any(
        not isinstance(identifier, str) or not 0 < len(identifier) <= 256
        for identifier in document_ids
    ):
        raise ValueError("section replay document set must contain at most 16 IDs")
    replayed: list[str] = []
    quarantined: list[str] = []
    for document_id in dict.fromkeys(document_ids):
        source = resolver.resolve(document_id, session=session)
        if not _source_is_bound(source, document_id, session.tenant):
            quarantined.append(document_id)
            continue
        result = processor.process(
            source.source_ref,
            document_id=document_id,
            text=source.text,
            external_access=source.external_access,
            persist=True,
            section_tree=True,
            connector=source.connector,
            source_instance=source.source_instance,
            checkpoint=source.checkpoint,
        )
        if not result.persisted:
            raise RuntimeError("native section replay did not persist")
        replayed.append(document_id)
    return LegacySectionReplayReport(tuple(replayed), tuple(quarantined))


def sample_legacy_section_quarantine(
    engine: Any, *, after_id: str = "", limit: int = 256
) -> LegacySectionQuarantineReport:
    """Sample legacy rows through the governed query path under verified read scope.

    The returned IDs are discovery hints only. Migration must re-fetch source
    bytes and source ACL under verified ownership, then use DocumentProcessor's
    native atomic write. Query-side tenant/owner filtering is owned by the
    engine's ``query_cypher`` implementation; no ``tenant`` is read from a
    Section payload or accepted as a function argument. A bounded sample may
    miss later visible rows after backend limit and ACL filtering, so
    ``complete`` is always false.
    """
    session = resolve_session(required_scope="kg:read")
    if not session.graph:
        raise PermissionError("legacy section audit requires a verified graph")
    if type(limit) is not int or not 1 <= limit <= _MAX_PAGE:
        raise ValueError("legacy section audit limit must be 1..1000")
    if not isinstance(after_id, str) or len(after_id) > 256:
        raise ValueError("invalid legacy section cursor")
    query = getattr(engine, "query_cypher", None)
    if not callable(query):
        raise TypeError("governed query_cypher is required")
    rows = query(_QUERY, {"after_id": after_id, "limit": limit + 1}, session=session)
    if not isinstance(rows, list):
        raise TypeError("governed section read returned an invalid page")
    if len(rows) > limit + 1:
        raise ValueError("governed section read exceeded its requested bound")

    selected = rows[:limit]
    ids: set[str] = set()
    unattributed = 0
    last_seen = after_id
    for row in selected:
        if not isinstance(row, dict):
            raise TypeError("governed section read returned an invalid row")
        section_id = row.get("id")
        if not isinstance(section_id, str) or not section_id:
            raise ValueError("governed section read lacks a section identity")
        last_seen = section_id
        document_id = row.get("document_id")
        if isinstance(document_id, str) and 0 < len(document_id) <= 256:
            ids.add(document_id)
        else:
            unattributed += 1
    return LegacySectionQuarantineReport(
        document_ids=tuple(sorted(ids)),
        sampled_sections=len(selected),
        unattributed_sections=unattributed,
        last_seen_id=last_seen,
        possibly_more=len(rows) > limit,
    )
