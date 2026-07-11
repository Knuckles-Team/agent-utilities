"""Ingestion Module.

CONCEPT:AU-KG.ingest.ingestion-engine — Ingestion Engine

Single entrypoint for all data ingestion into the Knowledge Graph.
Content-typed adaptors handle codebase, document, social, SPARQL,
skill, MCP server, policy, event stream, and prompt ingestion.
"""

from .change_envelope import OPERATIONS, ChangeEnvelope, Operation
from .engine import ContentType, IngestionEngine, IngestionManifest, IngestionResult

__all__ = [
    "ContentType",
    "IngestionEngine",
    "IngestionManifest",
    "IngestionResult",
    "ChangeEnvelope",
    "Operation",
    "OPERATIONS",
]
