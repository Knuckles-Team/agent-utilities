#!/usr/bin/python
"""Graph Backend Base Interface.

CONCEPT:AU-KG.query.vendor-agnostic-traversal — Vendor-Agnostic Graph Backend Abstraction

Provides the ``GraphBackend`` ABC that all graph storage backends must implement.
Backends may optionally support SPARQL via ``supports_sparql`` / ``execute_sparql()``.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any


def sanitize_label(label: str) -> str:
    """Reduce a label/relationship to a transpiler-safe identifier (``\\w+``)."""
    s = re.sub(r"\W+", "_", str(label or "Node")).strip("_")
    return s or "Node"


# Cypher clauses that mutate the graph. A query containing any of these (even
# alongside a leading MATCH) is treated as a write (e.g. a fan-out backend mirrors
# it to its mirror stores).
_WRITE_RE = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH|FOREACH|LOAD\s+CSV)\b",
    re.IGNORECASE,
)


def is_write(query: str) -> bool:
    """True if the Cypher query mutates the graph."""
    return bool(_WRITE_RE.search(query or ""))


# Backends for which a derived store (manifest, card cache, registry graph, …)
# must fall back to its zero-infra local store instead of routing arbitrary
# labelled nodes through ``backend.execute()``:
#   - EpistemicGraphBackend: pure in-memory, not durable across a restart.
#   - PostgreSQLBackend: schema-constrained tables have no arbitrary-label table,
#     so a graph-native MERGE errors ("relation does not exist").
# The epistemic-graph engine authority (reached through a fanout/durable backend)
# holds arbitrary nodes, so the engine path keeps graph-native stores.
_NON_DURABLE_BACKENDS = {
    "EpistemicGraphBackend",
    "PostgreSQLBackend",
}


def is_durable_backend(backend: Any) -> bool:
    """True if writes through ``backend.execute()`` survive a restart.

    The dual-mode predicate the **ingest DeltaManifest** (CONCEPT:AU-KG.ingest.enterprise-source-extractor) reuses to
    decide engine-graph mode vs. its zero-infra local fallback. A pure in-memory
    backend can't persist graph-side, and a schema-constrained durable store
    (pggraph) can't hold an arbitrary label — both → the local fallback. (The
    consolidated registry/card-cache/timeseries/writeback/code-health stores no
    longer use this — they are engine-only; see ``is_engine_authority_backend``.)
    """
    if backend is None or not hasattr(backend, "execute"):
        return False
    return type(backend).__name__ not in _NON_DURABLE_BACKENDS


# Backends that CANNOT hold arbitrary-label nodes through ``execute()`` — a
# graph-native ``MERGE (n:SomeLabel …)`` errors against them. The consolidated
# engine-only stores (CONCEPT:AU-KG.backend.cache-lives-as-248) route around these to the engine
# authority. ``EpistemicGraphBackend`` (the engine client) IS the authority and
# runs arbitrary-label Cypher, so — unlike the manifest's ``_NON_DURABLE_BACKENDS``
# — it is NOT excluded here: the consolidated stores live ON the engine.
_NON_LABEL_BACKENDS = {
    "PostgreSQLBackend",
}


def is_engine_authority_backend(backend: Any) -> bool:
    """True if ``backend`` can run arbitrary-label Cypher against the engine.

    The predicate the consolidated engine-only stores (registry / card-cache /
    timeseries / writeback / code-health, CONCEPT:AU-KG.backend.cache-lives-as-248) use to accept a
    supplied backend as the engine authority. It admits the in-process engine
    client (``EpistemicGraphBackend``) and any fan-out/durable backend with an
    ``execute()``, and rejects only the schema-constrained relational stores that
    have no arbitrary-label table (pggraph). There is no SQLite fallback — when no
    engine backend is supplied the store resolves one via
    :func:`require_engine_authority_backend`.
    """
    if backend is None or not hasattr(backend, "execute"):
        return False
    return type(backend).__name__ not in _NON_LABEL_BACKENDS


def require_engine_authority_backend(consumer: str) -> Any:
    """Return the active engine-authority backend, raising a clear error if absent.

    The single entry point the consolidated engine-only stores (CONCEPT:AU-KG.backend.cache-lives-as-
    248) use when no backend is supplied. It returns the active backend when it is
    engine-capable; otherwise it builds a fresh in-process engine client backend
    (``EpistemicGraphBackend``), which connects through the OS-5.63 resolver — the
    resolver auto-starts the pi-tier engine in prod, and the KG-2.238 test fixture
    provides a real ephemeral one. If the engine is genuinely unreachable this
    raises ``RuntimeError`` (NEVER a SQLite fallback).

    ``consumer`` names the calling store for the error message.
    """
    from . import get_active_backend

    active = get_active_backend()
    if is_engine_authority_backend(active):
        return active
    try:
        from .epistemic_graph_backend import EpistemicGraphBackend

        return EpistemicGraphBackend()
    except Exception as exc:  # noqa: BLE001 — re-raise as a clear, typed error
        raise RuntimeError(
            f"{consumer} requires the epistemic-graph engine, but no engine is "
            "reachable. The OS-5.63 resolver auto-starts the pi-tier engine in "
            "prod and the KG-2.238 test fixture provides a real ephemeral one — "
            f"there is no SQLite fallback. Underlying error: {exc}"
        ) from exc


def coerce_cypher_property(value: Any) -> Any:
    """Coerce a property value to something a Cypher backend (Neo4j/FalkorDB) accepts.

    Cypher property values must be primitives or *arrays of primitives*. A Map (dict),
    or a list containing non-primitives, raises ``Neo.ClientError.Statement.TypeError``
    ("Property values can only be of primitive types or arrays thereof") — and on a
    fan-out *mirror* that error stalls replication permanently (the outbox entry retries
    forever, dragging the write path). Serialize such values to a JSON string so the
    write persists losslessly (readers can ``json.loads`` it back); primitives and
    primitive arrays (e.g. embedding vectors) pass through untouched. (CONCEPT:AU-KG.backend.mirror-health-repair)
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        if all(v is None or isinstance(v, str | int | float | bool) for v in value):
            return list(value)
        return json.dumps(value, default=str)
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode("utf-8", "replace")
    # dict (Map), set, or any other non-primitive → lossless JSON string.
    return json.dumps(value, default=str)


class GraphBackend(ABC):
    """Abstract interface for Graph Database operations.

    All concrete backends (Memory, LadybugDB, Neo4j, FalkorDB, PostgreSQL,
    Fuseki, Stardog) must implement the core methods below.

    Backends that support SPARQL should override ``supports_sparql`` to return
    ``True`` and implement ``execute_sparql()`` with real SPARQL execution.
    """

    # ------------------------------------------------------------------
    # Core CRUD & Query
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute a graph query (e.g., Cypher) and return results.

        Args:
            include_epistemic: Opt-in (CONCEPT:AU-KB-CURRENCY, Seam 1 — the
                ``KnowledgeBatch`` currency, extended to this unguarded/unaudited
                direct-backend path — the counterpart of
                ``KnowledgeGraph.query(..., include_epistemic=True)`` for callers
                that use ``store.execute`` directly). Default ``False`` —
                byte-for-byte the same ``list[dict]`` rows as before this
                parameter existed. When ``True``, returns
                ``list[EpistemicRow]`` (see
                ``agent_utilities.knowledge_graph.core.epistemic_row``) instead:
                the same rows widened with the engine's per-row epistemic
                envelope. Only a backend with its own id-seeded provenance
                primitive (``EpistemicGraphBackend``, whose ``GraphComputeEngine``
                exposes ``explain_provenance_by_ids``) can honor this; any other
                backend degrades to an empty list rather than raising or
                silently ignoring the flag (never returns plain ``dict`` rows
                under a ``True`` request).
        """
        pass

    @abstractmethod
    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute a graph query over a batch of parameters for high-throughput ingestion."""
        pass

    @abstractmethod
    def create_schema(self) -> None:
        """Initialize required database schema (DDL for Ladybug, etc)."""
        pass

    def compare_and_set_node_fields(
        self,
        node_id: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
    ) -> bool:
        """Atomic compare-and-set on a node's fields (CONCEPT:AU-KG.compute.user-override-prompt-library).

        Optional capability: backends that support an atomic conditional update
        (engine L1, tiered) override this; the default declines so a caller can
        feature-detect rather than silently no-op.
        """
        raise NotImplementedError(  # ABSTRACT-OK — optional CAS capability
            f"{type(self).__name__} does not support compare_and_set_node_fields"
        )

    # ------------------------------------------------------------------
    # Vector / Embedding Support
    # ------------------------------------------------------------------

    @abstractmethod
    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Add an embedding vector to a specific node."""
        pass

    @abstractmethod
    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Perform a semantic vector search returning top matching nodes."""
        pass

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    @abstractmethod
    def prune(self, criteria: dict[str, Any]) -> None:
        """Run pruning logic based on criteria."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass

    # ------------------------------------------------------------------
    # Cypher portability tier  (CONCEPT:AU-KG.backend.multi-connection-registry)
    # ------------------------------------------------------------------

    @property
    def cypher_support(self) -> str:
        """How much of openCypher this backend can run unchanged.

        Drives capability-aware multi-connection fan-out (CONCEPT:AU-KG.backend.multi-connection-registry): when
        the SAME query is run against several connections, a backend that can
        only serve a bounded subset can be surfaced honestly rather than failing
        silently. Values:

        * ``"full"`` — native openCypher (neo4j, falkordb, Apache AGE). Default.
        * ``"subset"`` — only the bounded operational subset the engine emits
          runs (the regex Postgres transpiler, the in-memory epistemic graph).

        Override to ``"subset"`` in backends that do not run native Cypher.
        """
        return "full"

    # Optional SPARQL Capability  (CONCEPT:AU-KG.query.vendor-agnostic-traversal)
    # ------------------------------------------------------------------

    @property
    def supports_sparql(self) -> bool:
        """Whether this backend supports SPARQL queries.

        Override to ``True`` in backends backed by an RDF store
        (Fuseki, Stardog).
        """
        return False

    def execute_sparql(
        self,
        query: str,
        *,
        default_graph: str | None = None,
        timeout_ms: int = 30_000,
    ) -> list[dict[str, Any]]:
        """Execute a SPARQL SELECT/ASK/CONSTRUCT query.

        Backends that advertise ``supports_sparql = True`` must override
        this with a real implementation.

        Args:
            query: W3C SPARQL 1.1 query string.
            default_graph: Optional default graph IRI.
            timeout_ms: Query timeout in milliseconds.

        Returns:
            List of solution dicts (SELECT) or ``[{"result": bool}]`` (ASK).

        Raises:
            NotImplementedError: If the backend does not support SPARQL.
        """
        raise RuntimeError(
            f"{type(self).__name__} does not support SPARQL queries. "
            f"Use a SPARQL-capable backend (jena_fuseki, stardog)."
        )
