#!/usr/bin/python
from __future__ import annotations

"""Neo4j Backend Implementation."""

# CONCEPT:AU-KG.query.object-graph-mapper


import logging
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlparse

from agent_utilities.core.transport_security import (
    ResolvedTLSProfile,
    TransportSecurityError,
    resolve_configured_tls_profile,
)

from ..base import GraphBackend, coerce_cypher_property

logger = logging.getLogger(__name__)

try:
    from neo4j import (
        ClientCertificate as _ClientCertificate,
    )
    from neo4j import (
        ClientCertificateProviders as _ClientCertificateProviders,
    )
    from neo4j import GraphDatabase as _GraphDatabase
    from neo4j import (
        TrustCustomCAs as _TrustCustomCAs,
    )
    from neo4j import (
        TrustSystemCAs as _TrustSystemCAs,
    )

    GraphDatabase: Any = _GraphDatabase
    ClientCertificate: Any = _ClientCertificate
    ClientCertificateProviders: Any = _ClientCertificateProviders
    TrustCustomCAs: Any = _TrustCustomCAs
    TrustSystemCAs: Any = _TrustSystemCAs
except ImportError:
    GraphDatabase = None
    ClientCertificate = None
    ClientCertificateProviders = None
    TrustCustomCAs = None
    TrustSystemCAs = None


class Neo4jBackend(GraphBackend):
    """Neo4j backend for the unified graph."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str | None = None,
        tls_profile: str | None = None,
        tls_profile_ref: str | None = None,
        tls_profile_config: Mapping[str, Any] | None = None,
        profile_resolver: Callable[[str], str | None] | None = None,
        connection_timeout: float | None = None,
        max_connection_pool_size: int | None = None,
    ):
        if GraphDatabase is None:
            raise ImportError(
                "Neo4j driver is not installed. Please install with `pip install agent-utilities[neo4j]`"
            )
        parsed = urlparse(uri)
        if (
            parsed.scheme
            not in {
                "bolt",
                "neo4j",
                "bolt+s",
                "neo4j+s",
            }
            or not parsed.hostname
        ):
            raise ValueError("Neo4j URI has an unsupported transport shape")
        try:
            trust = resolve_configured_tls_profile(
                "NEO4J",
                profile_name=tls_profile,
                profile_ref=tls_profile_ref,
                profile=tls_profile_config,
                resolver=profile_resolver,
            )
        except TransportSecurityError:
            raise ValueError("Neo4j transport security profile is invalid") from None
        self._tls_profile: ResolvedTLSProfile = trust
        self.database = str(database or "").strip() or None

        if trust.proxy_url and trust.configured:
            trust.cleanup()
            raise ValueError(
                "Neo4j Bolt transport does not support HTTP proxy profiles"
            )
        scheme_managed = parsed.scheme.endswith("+s")
        if parsed.scheme.endswith("+s") and (
            trust.ca_bundle_path is not None or trust.ca_directory is not None
        ):
            trust.cleanup()
            raise ValueError(
                "Neo4j custom CA profiles require a bolt:// or neo4j:// URI"
            )

        driver_options: dict[str, Any] = {"auth": (user, password)}
        if connection_timeout is not None:
            driver_options["connection_timeout"] = max(
                0.1, min(float(connection_timeout), 300.0)
            )
        if max_connection_pool_size is not None:
            driver_options["max_connection_pool_size"] = max(
                1, min(int(max_connection_pool_size), 1_000)
            )
        if not scheme_managed and trust.configured:
            driver_options["encrypted"] = True
            if trust.ca_bundle_path is not None or trust.ca_directory is not None:
                if TrustCustomCAs is None:
                    trust.cleanup()
                    raise ImportError("Neo4j driver lacks custom CA support")
                certificates = self._neo4j_ca_files(trust)
                driver_options["trusted_certificates"] = TrustCustomCAs(
                    *(str(path) for path in certificates)
                )
            elif TrustSystemCAs is not None:
                driver_options["trusted_certificates"] = TrustSystemCAs()

        if trust.client_cert_path is not None:
            if ClientCertificate is None or ClientCertificateProviders is None:
                trust.cleanup()
                raise ImportError("Neo4j driver lacks mTLS client certificate support")
            certificate = ClientCertificate(
                certfile=str(trust.client_cert_path),
                keyfile=str(trust.client_key_path),
                password=trust.client_key_password,
            )
            driver_options["client_certificate"] = ClientCertificateProviders.static(
                certificate
            )
        try:
            self.driver = GraphDatabase.driver(uri, **driver_options)
        except Exception:
            trust.cleanup()
            raise
        logger.info("Initialized Neo4j backend with runtime connection profile")

    @staticmethod
    def _neo4j_ca_files(trust: ResolvedTLSProfile) -> tuple[Any, ...]:
        if trust.ca_bundle_path is not None:
            return (trust.ca_bundle_path,)
        if trust.ca_directory is None:
            return ()
        try:
            candidates = tuple(
                path
                for path in sorted(trust.ca_directory.iterdir())
                if path.is_file() and path.suffix.casefold() in {".pem", ".crt", ".cer"}
            )
        except OSError:
            candidates = ()
        if not candidates or len(candidates) > 64:
            raise ValueError("Neo4j CA directory is empty or exceeds the safe bound")
        return candidates

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """Unwrap neo4j driver Node/Relationship objects into plain property dicts.

        ``dict(record)`` leaves graph entities as opaque ``Node``/``Relationship``
        objects, so a ``RETURN n`` row carries no readable properties for callers
        that expect dicts. Unwrap to ``dict(entity)`` (its properties) so reads are
        on par with the other backends.
        """
        # neo4j Node/Relationship are Mapping-like; ``.labels``/``.type`` distinguish them.
        if hasattr(value, "items") and (
            hasattr(value, "labels") or hasattr(value, "type")
        ):
            return dict(value)
        if isinstance(value, list):
            return [Neo4jBackend._normalize_value(v) for v in value]
        return value

    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        if include_epistemic:
            # CONCEPT:AU-KB-CURRENCY (Seam 1) — no id-seeded epistemic-envelope
            # primitive on this backend; degrade to ``[]`` per the ABC contract.
            logger.debug(
                "Neo4jBackend.execute(include_epistemic=True): no epistemic "
                "envelope primitive; returning []"
            )
            return []
        params = {k: coerce_cypher_property(v) for k, v in (params or {}).items()}
        session_options = {"database": self.database} if self.database else {}
        with self.driver.session(**session_options) as session:
            result = session.run(query, params)
            return [
                {k: self._normalize_value(v) for k, v in record.items()}
                for record in result
            ]

    def execute_read(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute Cypher in a server-declared read transaction.

        The Neo4j driver sends READ access mode through ``execute_read``; a
        caller-supplied mutation is therefore rejected by the server rather
        than guessed from query text.
        """
        if include_epistemic:
            return []
        safe_params = {
            key: coerce_cypher_property(value) for key, value in (params or {}).items()
        }
        session_options = {"database": self.database} if self.database else {}

        def run(transaction: Any) -> list[dict[str, Any]]:
            result = transaction.run(query, safe_params)
            return [
                {key: self._normalize_value(value) for key, value in record.items()}
                for record in result
            ]

        with self.driver.session(**session_options) as session:
            return session.execute_read(run)

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute a batch query using Cypher UNWIND."""
        if not self.driver:
            raise RuntimeError("Neo4j backend not connected.")

        # Ensure query contains UNWIND if the user hasn't explicitly structured it
        if "UNWIND" not in query.upper():
            logger.warning(
                "Batch query does not contain UNWIND. Execution might be slow or fail."
            )

        # Sanitize each row's property values (Map/nested → JSON string) so a single
        # Map-valued prop can't fail the whole UNWIND batch (and stall a mirror).
        batch = [
            {k: coerce_cypher_property(v) for k, v in row.items()}
            if isinstance(row, dict)
            else row
            for row in batch
        ]
        session_options = {"database": self.database} if self.database else {}
        with self.driver.session(**session_options) as session:
            try:
                result = session.run(query, {"batch": batch})
                return [dict(record) for record in result]
            except Exception as e:
                logger.error("Neo4j batch execution error (%s)", type(e).__name__)
                return []

    def create_schema(self) -> None:
        # Index a SHARED ``:Embeddable`` label that every embedded node is tagged
        # with (see add_embedding). The old index targeted a ``:Chunk`` label that
        # no node ever carries, so vector search silently returned nothing.
        logger.info("Creating Neo4j vector index for embeddings (:Embeddable).")
        # Embedding dimensionality is driven by the unified XDG config
        # (config.kg_embedding_dim) — never hardcoded — so every backend agrees with
        # the configured embedding model; 768 is only a last-resort fallback.
        from agent_utilities.core.config import config

        dim = int(config.kg_embedding_dim or "768")
        query = f"""
        CREATE VECTOR INDEX idx_embedding IF NOT EXISTS
        FOR (n:Embeddable) ON (n.embedding)
        OPTIONS {{indexConfig: {{
          `vector.dimensions`: {dim},
          `vector.similarity_function`: 'cosine'
        }}}}
        """
        try:
            self.execute(query)
        except Exception as e:
            logger.warning(f"Could not create vector index in Neo4j: {e}")

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        # Tag the node ``:Embeddable`` so it enters the vector index regardless of
        # its primary label (Code/Concept/Document/…).
        query = "MATCH (n {id: $id}) SET n:Embeddable, n.embedding = $embedding"
        self.execute(query, {"id": node_id, "embedding": embedding})

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Perform a semantic vector search returning top matching nodes using Neo4j 5.11+."""
        query = """
        CALL db.index.vector.queryNodes('idx_embedding', $n_results, $query_embedding)
        YIELD node, score
        RETURN node
        """
        return self.execute(
            query, {"query_embedding": query_embedding, "n_results": n_results}
        )

    def prune(self, criteria: dict[str, Any]) -> None:
        query = "MATCH (n) WHERE n.last_accessed < $timestamp DETACH DELETE n"
        if "last_accessed" in criteria:
            self.execute(query, {"timestamp": criteria["last_accessed"]})

    def close(self) -> None:
        try:
            self.driver.close()
        finally:
            self._tls_profile.cleanup()
