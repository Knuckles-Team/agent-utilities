from __future__ import annotations

"""Document-source connector framework — load / poll / slim ABCs.

CONCEPT:ECO-4.25 — Document-Source Connector Framework

Where :mod:`agent_utilities.protocols.data_connector` (CONCEPT:ECO-4.0) models a
*query → rows* data source (market data, tabular fetch), this module models a
*document source*: an external system (a website, a filesystem, a SaaS app
reached through its MCP server) that yields **documents** to be ingested into the
Knowledge Graph as first-class ``Document`` + ``Chunk`` ontology objects.

Provenance: this ports the connector surface of Onyx / Danswer
(``backend/onyx/connectors/interfaces.py`` — ``LoadConnector`` / ``PollConnector``
/ ``SlimConnector`` / ``SlimConnectorWithPermSync`` with generic checkpoint
typing) onto agent-utilities' *semantic* core. The "for free" win versus a flat
vector store: every :class:`SourceDocument` ingested through this framework flows
through the KG-2.48 ``DocumentProcessor`` and therefore inherits OWL semantics,
bitemporal slicing, reified ``HAS_CHUNK`` / ``CHUNK_OF`` links, and the
entailment-aware ACLs of KG-2.46 — capabilities a document index cannot offer.

Design choices:

  * **Compose, do not compete.** :class:`BaseSourceConnector` carries the same
    ``name`` / ``provider`` / ``priority`` / ``supported_instruments`` /
    ``health_check`` / ``fetch`` surface as :class:`DataConnectorProtocol`, so a
    document connector is *also* registerable in the existing
    :class:`DataConnectorRegistry` (its ``fetch`` returns a row-shaped summary).
    Its real document surface is the ``load`` / ``poll`` / ``slim`` methods.
  * **Three connector shapes**, exactly mirroring the Onyx contract:
      - :class:`LoadConnector` — ``load()`` yields a full snapshot.
      - :class:`PollConnector` — ``poll(checkpoint)`` yields an incremental,
        resumable batch (CONCEPT:ECO-4.26).
      - :class:`SlimConnector` — ``slim()`` yields ids + access only, for a cheap
        permission sweep without re-fetching bodies.
      - :class:`PermSyncConnector` — adds ``fetch_access()`` for external ACL sync
        (CONCEPT:ECO-4.28).
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
]


class ExternalAccess(BaseModel):
    """The access-control facts a source reports for one document.

    CONCEPT:ECO-4.28 — the shape :func:`permission_sync.sync_access` maps onto the
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


class SourceDocument(BaseModel):
    """One document yielded by a connector, ready for the ingestion pipeline.

    CONCEPT:ECO-4.25 — the unit handed to the KG-2.48 ``DocumentProcessor`` via
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

    CONCEPT:ECO-4.25 — Onyx's ``SlimDocument``: enumerated without re-fetching
    bodies so a permission sync or prune pass is cheap.
    """

    id: str
    source_uri: str = ""
    external_access: ExternalAccess | None = None


class BaseSourceConnector(abc.ABC):  # noqa: B024 — abstract surface lives on the load/poll/slim mixins
    """Base class for all document-source connectors.

    CONCEPT:ECO-4.25 — carries the :class:`DataConnectorProtocol` (ECO-4.0)
    metadata surface (``name`` / ``provider`` / ``priority`` /
    ``supported_instruments`` / ``health_check`` / ``fetch``) so a document
    connector can co-register in the existing :class:`DataConnectorRegistry`
    while exposing its real document surface through the ``Load`` / ``Poll`` /
    ``Slim`` mixins.

    Subclasses set ``source_type`` (the registry key) and override ``configure``
    (or ``__init__``) to accept their config dict.
    """

    #: Registry key used by :func:`registry.register_source` / ``build_connector``.
    source_type: str = "base"
    #: Human-readable provider name (DataConnectorProtocol field).
    provider: str = ""
    #: Fallback priority (lower tried first); document connectors default mid-pack.
    priority: int = 50
    #: Instrument scope (unused for document sources; kept for protocol parity).
    supported_instruments: list[str] = []

    def __init__(self, **config: Any) -> None:
        self._config = dict(config)
        self.configure(**config)

    # -- DataConnectorProtocol parity --------------------------------------

    @property
    def name(self) -> str:
        """Unique connector identifier (``DataConnectorProtocol.name``)."""
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

    def fetch(self, query: str = "", **kwargs: Any) -> Any:
        """:class:`DataConnectorProtocol` parity: summarise ``load()`` as rows.

        Returns a :class:`DataFetchResult` whose ``rows`` are the slim
        ``{id, source_uri, title}`` projections of the loaded documents, so a
        document connector also satisfies the row-oriented registry. The document
        surface proper is ``load`` / ``poll`` / ``slim``.
        """
        from ..data_connector import DataFetchResult

        rows: list[dict[str, Any]] = []
        if isinstance(self, LoadConnector):
            for doc in self.load():
                rows.append(
                    {"id": doc.id, "source_uri": doc.source_uri, "title": doc.title}
                )
        return DataFetchResult(
            rows=rows, row_count=len(rows), connector_name=self.name, query=query
        )


class LoadConnector(BaseSourceConnector):
    """A connector that yields a full snapshot of its documents.

    CONCEPT:ECO-4.25 — Onyx ``LoadConnector``: ``load()`` enumerates every
    document currently visible to the connector (used for a first full ingest or
    a non-incremental source).
    """

    @abc.abstractmethod
    def load(self) -> Iterator[SourceDocument]:
        """Yield every document the connector can currently see."""
        raise NotImplementedError  # ABSTRACT-OK


class PollConnector(BaseSourceConnector):
    """A connector that yields incremental, resumable batches.

    CONCEPT:ECO-4.26 — Onyx ``PollConnector`` with generic checkpoint typing.
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

    CONCEPT:ECO-4.25 — Onyx ``SlimConnector``: powers cheap permission sweeps and
    prune detection.
    """

    @abc.abstractmethod
    def slim(self) -> Iterator[SlimDocument]:
        """Yield ``(id, access)`` records for every visible document."""
        raise NotImplementedError  # ABSTRACT-OK


class PermSyncConnector(SlimConnector):
    """A slim connector that also reports per-document external access.

    CONCEPT:ECO-4.28 — Onyx ``SlimConnectorWithPermSync``. ``fetch_access`` yields
    ``(document_id, ExternalAccess)`` pairs consumed by
    :func:`permission_sync.sync_access` to mirror source ACLs into KG-2.46.
    """

    @abc.abstractmethod
    def fetch_access(self) -> Iterator[tuple[str, ExternalAccess]]:
        """Yield ``(document_id, ExternalAccess)`` for every visible document."""
        raise NotImplementedError  # ABSTRACT-OK
