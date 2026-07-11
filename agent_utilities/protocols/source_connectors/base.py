from __future__ import annotations

"""Document-source connector framework — load / poll / slim ABCs.

CONCEPT:AU-ECO.connector.document-source-framework — Document-Source Connector Framework

This module models a *document source*: an external system (a website, a
filesystem, a SaaS app reached through its MCP server) that yields **documents**
to be ingested into the Knowledge Graph as first-class ``Document`` + ``Chunk``
ontology objects.

Provenance: this ports the connector surface of Onyx / Danswer
(``backend/onyx/connectors/interfaces.py`` — ``LoadConnector`` / ``PollConnector``
/ ``SlimConnector`` / ``SlimConnectorWithPermSync`` with generic checkpoint
typing) onto agent-utilities' *semantic* core. The "for free" win versus a flat
vector store: every :class:`SourceDocument` ingested through this framework flows
through the KG-2.48 ``DocumentProcessor`` and therefore inherits OWL semantics,
bitemporal slicing, reified ``HAS_CHUNK`` / ``CHUNK_OF`` links, and the
entailment-aware ACLs of KG-2.46 — capabilities a document index cannot offer.

Design choices:

  * **Three connector shapes**, exactly mirroring the Onyx contract:
      - :class:`LoadConnector` — ``load()`` yields a full snapshot.
      - :class:`PollConnector` — ``poll(checkpoint)`` yields an incremental,
        resumable batch (CONCEPT:AU-ECO.connector.incremental-poll-watermark).
      - :class:`SlimConnector` — ``slim()`` yields ids + access only, for a cheap
        permission sweep without re-fetching bodies.
      - :class:`PermSyncConnector` — adds ``fetch_access()`` for external ACL sync
        (CONCEPT:AU-ECO.connector.external-permission-sync).
  * **Abstract, fully implemented.** Abstract methods raise ``NotImplementedError``
    tagged ``# ABSTRACT-OK`` so the contract-completeness gate recognises them as
    genuine ABC contract holes, not incomplete code.
"""

import abc
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field

from .checkpoint import CheckpointedBatch, ConnectorCheckpoint

# Re-export so callers can ``from ...base import ConnectorCheckpoint``.
__all__ = [
    "ExternalAccess",
    "SourceDocument",
    "SlimDocument",
    "BaseSourceConnector",
    "LoadConnector",
    "PollConnector",
    "SlimConnector",
    "PermSyncConnector",
    "ConnectorCheckpoint",
    "CheckpointedBatch",
    "CONNECTOR_UNCONFIGURED_MARKING",
    "default_external_access",
]

# Mandatory-control marking applied to a document whose connector could not
# report a real ACL (CONCEPT:AU-P0-4 fail-closed connector permissions). No
# actor holds ``marking:connector-unconfigured-acl`` by default, so
# ``permission_sync.sync_access`` restricts the document to nobody until an
# operator explicitly reviews it and grants the marking — the fail-closed
# counterpart to the old ``ExternalAccess.public()`` default. A quarantined
# document with ``is_public=False`` and empty ``group_ids``/``user_emails``
# would otherwise register NO discretionary ACL at all (``sync_access`` only
# builds one when ``roles`` is non-empty) and silently fall through to the
# default-allow read gate — the marking is what actually closes that gap.
CONNECTOR_UNCONFIGURED_MARKING = "connector-unconfigured-acl"


class ExternalAccess(BaseModel):
    """The access-control facts a source reports for one document.

    CONCEPT:AU-ECO.connector.external-permission-sync — the shape :func:`permission_sync.sync_access` maps onto the
    KG-2.46 permissioning model. Mirrors Onyx's ``ExternalAccess``: a document is
    either world-public, or readable by an explicit set of users / groups, and may
    additionally carry mandatory compartment markings.

    Attributes:
        is_public: When true the document is readable by anyone (no ACL applied).
        user_emails: Individual principals granted read access.
        group_ids: Group principals granted read access.
        markings: Mandatory-control compartment names (KG-2.46 ``Marking``).
    """

    is_public: bool = False
    user_emails: list[str] = Field(default_factory=list)
    group_ids: list[str] = Field(default_factory=list)
    markings: list[str] = Field(default_factory=list)

    @classmethod
    def public(cls) -> ExternalAccess:
        """A world-readable access descriptor."""
        return cls(is_public=True)

    @classmethod
    def quarantined(cls) -> ExternalAccess:
        """The most-restrictive access descriptor (CONCEPT:AU-P0-4).

        Not public, no principals granted, and carries
        :data:`CONNECTOR_UNCONFIGURED_MARKING` so the document is actually
        denied by the KG-2.46 read gate (see the constant's docstring) rather
        than silently defaulting open. This is the fail-closed default for an
        unproven/unconfigured connector — the opposite of :meth:`public`.
        """
        return cls(is_public=False, markings=[CONNECTOR_UNCONFIGURED_MARKING])


def default_external_access() -> ExternalAccess:
    """The connector default when a source reports no ACL at all (CONCEPT:AU-P0-4).

    Fail-closed: "unknown" must never silently mean "public". Returns
    :meth:`ExternalAccess.quarantined` unless the deployment has explicitly
    opted into the legacy public-by-default behavior via
    ``CONNECTOR_DEFAULT_PUBLIC=true`` (a dev/local convenience toggle — default
    ``False`` so enterprise/unknown deployments fail closed).
    """
    from agent_utilities.core.config import setting

    if setting("CONNECTOR_DEFAULT_PUBLIC", default=False):
        return ExternalAccess.public()
    return ExternalAccess.quarantined()


class SourceDocument(BaseModel):
    """One document yielded by a connector, ready for the ingestion pipeline.

    CONCEPT:AU-ECO.connector.document-source-framework — the unit handed to the KG-2.48 ``DocumentProcessor`` via
    the ``CONNECTOR`` ingestion adaptor. ``text`` is the already-extracted body
    (connectors are responsible for extraction so the pipeline stays uniform).

    Attributes:
        id: Stable, connector-scoped identifier (deduplicates re-polls).
        source_uri: Canonical URI/URL/path the document came from (provenance).
        title: Human-readable title; falls back to ``id`` downstream when empty.
        text: Extracted plain-text/markdown body to chunk + embed.
        doc_type: Optional document type hint (``paper``/``ticket``/``message``…).
        metadata: Arbitrary provenance/context merged onto the ``Document`` node.
        external_access: Source-reported ACL (consumed by permission sync).
        updated_at: Source's last-modified marker (ISO string or opaque token),
            used to advance the incremental-poll watermark.
    """

    id: str
    source_uri: str = ""
    title: str = ""
    text: str = ""
    doc_type: str = "document"
    metadata: dict[str, Any] = Field(default_factory=dict)
    external_access: ExternalAccess | None = None
    updated_at: str | None = None


class SlimDocument(BaseModel):
    """A lightweight ``(id, access)`` record for permission/existence sweeps.

    CONCEPT:AU-ECO.connector.document-source-framework — Onyx's ``SlimDocument``: enumerated without re-fetching
    bodies so a permission sync or prune pass is cheap.
    """

    id: str
    source_uri: str = ""
    external_access: ExternalAccess | None = None


class BaseSourceConnector(abc.ABC):  # noqa: B024 — abstract surface lives on the load/poll/slim mixins
    """Base class for all document-source connectors.

    CONCEPT:AU-ECO.connector.document-source-framework — the document surface lives on the ``Load`` / ``Poll`` /
    ``Slim`` mixins, plus a cheap ``health_check`` probe.

    Subclasses set ``source_type`` (the registry key) and override ``configure``
    (or ``__init__``) to accept their config dict.
    """

    #: Registry key used by :func:`registry.register_source` / ``build_connector``.
    source_type: str = "base"
    #: Human-readable provider name (display/provenance metadata).
    provider: str = ""

    def __init__(self, **config: Any) -> None:
        self._config = dict(config)
        self.configure(**config)

    @property
    def name(self) -> str:
        """Unique connector identifier (defaults to the registry key)."""
        return self.source_type

    def configure(self, **config: Any) -> None:  # noqa: B027 — optional override hook, not abstract
        """Apply connector configuration. Override to validate/store options.

        The default merges ``config`` onto ``self._config`` (the same mapping
        ``__init__`` seeds); subclasses override to validate and bind options.
        """
        self._config.update(config)

    def health_check(self) -> bool:
        """Whether the connector is reachable/usable. Default: healthy.

        Override for sources whose reachability is cheap to probe.
        """
        return True


class LoadConnector(BaseSourceConnector):
    """A connector that yields a full snapshot of its documents.

    CONCEPT:AU-ECO.connector.document-source-framework — Onyx ``LoadConnector``: ``load()`` enumerates every
    document currently visible to the connector (used for a first full ingest or
    a non-incremental source).
    """

    @abc.abstractmethod
    def load(self) -> Iterator[SourceDocument]:
        """Yield every document the connector can currently see."""
        raise NotImplementedError  # ABSTRACT-OK


class PollConnector(BaseSourceConnector):
    """A connector that yields incremental, resumable batches.

    CONCEPT:AU-ECO.connector.incremental-poll-watermark — Onyx ``PollConnector`` with generic checkpoint typing.
    ``poll(checkpoint)`` returns a :class:`CheckpointedBatch` whose ``checkpoint``
    is fed back on the next call; ``checkpoint.has_more`` drives the drain loop and
    ``checkpoint.watermark`` makes re-polls cheap (delta, not full scan).
    """

    @abc.abstractmethod
    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        """Return the next incremental batch given the prior checkpoint."""
        raise NotImplementedError  # ABSTRACT-OK

    def poll_all(
        self, checkpoint: ConnectorCheckpoint | None = None, *, max_batches: int = 1000
    ) -> Iterator[SourceDocument]:
        """Drain ``poll`` until ``has_more`` is false, yielding every document.

        A concrete, reusable drain loop over :meth:`poll` (bounded by
        ``max_batches`` as a runaway backstop). The final checkpoint is available
        via :attr:`last_checkpoint` after iteration.
        """
        cp = checkpoint
        batches = 0
        while batches < max_batches:
            batch = self.poll(cp)
            yield from batch.documents
            cp = batch.checkpoint
            self.last_checkpoint = cp
            batches += 1
            if not cp.has_more:
                break

    #: Set by :meth:`poll_all` so the ingestion adaptor can persist it.
    last_checkpoint: ConnectorCheckpoint | None = None


class SlimConnector(BaseSourceConnector):
    """A connector that can enumerate ids + access without fetching bodies.

    CONCEPT:AU-ECO.connector.document-source-framework — Onyx ``SlimConnector``: powers cheap permission sweeps and
    prune detection.
    """

    @abc.abstractmethod
    def slim(self) -> Iterator[SlimDocument]:
        """Yield ``(id, access)`` records for every visible document."""
        raise NotImplementedError  # ABSTRACT-OK


class PermSyncConnector(SlimConnector):
    """A slim connector that also reports per-document external access.

    CONCEPT:AU-ECO.connector.external-permission-sync — Onyx ``SlimConnectorWithPermSync``. ``fetch_access`` yields
    ``(document_id, ExternalAccess)`` pairs consumed by
    :func:`permission_sync.sync_access` to mirror source ACLs into KG-2.46.
    """

    @abc.abstractmethod
    def fetch_access(self) -> Iterator[tuple[str, ExternalAccess]]:
        """Yield ``(document_id, ExternalAccess)`` for every visible document."""
        raise NotImplementedError  # ABSTRACT-OK
