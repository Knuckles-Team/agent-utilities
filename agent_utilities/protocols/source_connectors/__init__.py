"""Document-source connector framework (CONCEPT:ECO-4.25–4.29).

Onyx-parity document ingestion bolted onto agent-utilities' semantic core: a
``load`` / ``poll`` / ``slim`` connector abstraction (:mod:`base`), resumable
checkpoints (:mod:`checkpoint`, ECO-4.26), a self-registering factory
(:mod:`registry`, ECO-4.27), external permission sync into KG-2.46
(:mod:`permission_sync`, ECO-4.28), and reference connectors under
:mod:`connectors` (web/filesystem/rest + the agent-package fleet adapter, ECO-4.29).

Documents enter the Knowledge Graph through the ``CONNECTOR`` ingestion adaptor,
which runs each :class:`SourceDocument` through the KG-2.48 ``DocumentProcessor``
so it becomes first-class ``Document`` + ``Chunk`` ontology objects with OWL
semantics, bitemporal slicing, and entailment-aware ACLs.
"""

from .base import (
    BaseSourceConnector,
    ExternalAccess,
    LoadConnector,
    PermSyncConnector,
    PollConnector,
    SlimConnector,
    SlimDocument,
    SourceDocument,
)
from .checkpoint import CheckpointedBatch, ConnectorCheckpoint
from .permission_sync import sync_access
from .registry import (
    build_connector,
    discover,
    get_connector_class,
    list_sources,
    register_source,
)

__all__ = [
    "BaseSourceConnector",
    "LoadConnector",
    "PollConnector",
    "SlimConnector",
    "PermSyncConnector",
    "SourceDocument",
    "SlimDocument",
    "ExternalAccess",
    "ConnectorCheckpoint",
    "CheckpointedBatch",
    "register_source",
    "build_connector",
    "get_connector_class",
    "list_sources",
    "discover",
    "sync_access",
]
