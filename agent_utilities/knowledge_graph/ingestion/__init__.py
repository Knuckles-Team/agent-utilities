"""Ingestion Module.

CONCEPT:AU-KG.ingest.ingestion-engine — Ingestion Engine

Single entrypoint for all data ingestion into the Knowledge Graph.
Content-typed adaptors handle codebase, document, social, SPARQL,
skill, MCP server, policy, event stream, and prompt ingestion.
"""

from .change_envelope import OPERATIONS, ChangeEnvelope, Operation
from .engine import ContentType, IngestionEngine, IngestionManifest, IngestionResult
from .evidence_spine import (
    Artifact,
    Fragment,
    FragmentKind,
    artifact_id_for,
    content_digest,
    fragment_id_for,
    resolve_fragment,
)

__all__ = [
    "ContentType",
    "IngestionEngine",
    "IngestionManifest",
    "IngestionResult",
    "ChangeEnvelope",
    "Operation",
    "OPERATIONS",
    "Artifact",
    "Fragment",
    "FragmentKind",
    "artifact_id_for",
    "content_digest",
    "fragment_id_for",
    "resolve_fragment",
]
