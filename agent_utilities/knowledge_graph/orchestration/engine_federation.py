#!/usr/bin/python
from __future__ import annotations

"""Reference-only federation for external ontologies and graph sources."""

import hashlib
import logging
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_ALIAS_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_RUNTIME_REF_RE = re.compile(r"^(?:env|vault|secret|sqlite)://[^\s]+$")


def _alias(value: object, label: str) -> str:
    rendered = str(value or "").strip().lower()
    if not _ALIAS_RE.fullmatch(rendered):
        raise ValueError(f"{label} must be a neutral lowercase alias")
    return rendered


def _runtime_ref(value: object, label: str) -> str:
    rendered = str(value or "").strip()
    if not _RUNTIME_REF_RE.fullmatch(rendered):
        raise ValueError(f"{label} must be a runtime secret reference")
    return rendered


def _pseudonymous_id(kind: str, *parts: object) -> str:
    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()[:24]
    return f"{kind}_{digest}"


class FederationMixin:
    """Mixin for external ontology federation and external KG metadata ingestion.

    CONCEPT:AU-KG.ingest.external-graph-federation — External Graph Federation
    """

    def register_external_ontology(
        self,
        *,
        reference_id: str,
        ontology_ref: str,
        connection: str,
    ) -> str:
        """Register a runtime-resolved ontology on a named graph connection.

        ``ontology_ref`` resolves to the concrete ontology URI only at use time;
        ``connection`` names an AgentConfig/connection-registry entry whose
        endpoint and credentials are likewise reference-backed. No URI, endpoint,
        credential, or local path is written into the operational graph.
        """
        alias = _alias(reference_id, "reference_id")
        connection_alias = _alias(connection, "connection")
        ref = _runtime_ref(ontology_ref, "ontology_ref")
        node_id = _pseudonymous_id("ExternalGraphReference", alias)
        if not hasattr(self, "_external_ontologies"):
            self._external_ontologies = {}
        self._external_ontologies[node_id] = {
            "reference_id": alias,
            "ontology_ref": ref,
            "connection": connection_alias,
        }
        self.add_node(  # type: ignore[attr-defined]
            node_id=node_id,
            node_type="ExternalGraphReference",
            properties={
                "referenceAlias": alias,
                "ontologyRef": ref,
                "connection": connection_alias,
                "platform": "federated",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        logger.info("Registered reference-backed external ontology")
        return node_id

    def get_registered_ontologies(self) -> dict[str, dict[str, str]]:
        """Return reference-only ontology declarations keyed by pseudonymous ID."""
        if not hasattr(self, "_external_ontologies"):
            return {}
        return {
            key: dict(value) for key, value in self._external_ontologies.items()
        }

    def ingest_external_entity_stub(
        self,
        internal_node_id: str,
        external_id: str,
        external_uri_ref: str,
        source_alias: str,
        name: str | None = None,
    ) -> str:
        """Ingest a high-level metadata stub from an external KG (e.g. EAR).

        Creates an `ExternalEntity` node and links it to the specified `internal_node_id`
        via `mapped_to_external` to create a bridge between the internal structural graph
        and the external metadata graph.

        Returns the ID of the created external stub node.
        """
        source = _alias(source_alias, "source_alias")
        uri_ref = _runtime_ref(external_uri_ref, "external_uri_ref")
        external_digest = hashlib.sha256(str(external_id).encode("utf-8")).hexdigest()
        stub_id = _pseudonymous_id("ExternalEntity", source, external_id)

        self.add_node(  # type: ignore[attr-defined]
            node_id=stub_id,
            node_type="ExternalEntity",
            properties={
                "externalIdDigest": external_digest,
                "externalUriRef": uri_ref,
                "sourceAlias": source,
                "name": name or "External Entity",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        # Link internal node to this external reference
        self.link_nodes(  # type: ignore[attr-defined]
            source_id=internal_node_id, target_id=stub_id, rel_type="mapped_to_external"
        )

        logger.debug(
            "Bridged internal node %s to external source %s",
            internal_node_id,
            source,
        )
        return stub_id

    def execute_federated_query(
        self, reference_id: str, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query against an external graph reference.

        Args:
            reference_id: The ID of the ExternalGraphReferenceNode in the local graph.
            query: The SPARQL or Cypher query string.
            parameters: Optional query parameters (mostly for Cypher/LPG).

        Returns:
            A list of dictionary records.
        """
        # 0. REST virtual sources are served by invoking their extractor (the
        # `query` string carries no SPARQL endpoint; an optional `node_type`
        # parameter filters to one canonical type).
        if reference_id in getattr(self, "_rest_sources", {}):
            node_type = (parameters or {}).get("node_type")
            return self.query_rest_source(reference_id, node_type=node_type)

        # 1. Retrieve only the neutral connection alias from the local graph.
        if not hasattr(self, "backend") or not self.backend:  # type: ignore[attr-defined]
            node_data = self.graph.nodes.get(reference_id)  # type: ignore[attr-defined]
            if not node_data:
                raise ValueError(
                    f"External graph reference {reference_id} not found in local graph."
                )
            connection = node_data.get("connection")
        else:
            res = self.backend.execute_read(  # type: ignore[attr-defined]
                "MATCH (n) WHERE n.id = $id RETURN n.connection as connection",
                {"id": reference_id},
            )
            if not res:
                raise ValueError(
                    f"External graph reference {reference_id} not found in persistent graph."
                )
            connection = res[0].get("connection")

        connection_alias = _alias(connection, "connection")
        return self._execute_federated_connection(
            connection_alias, query, parameters=parameters
        )

    # ── REST virtualization (query-time, extractor-backed) ───────────────────
    #
    # Camunda/ServiceNow/ERPNext speak REST/JSON, not SPARQL, so true Ontop-style
    # virtual-SPARQL is out of scope. Instead we virtualize by invoking the
    # *existing* self-registering extractor on demand (TTL-cached) and returning
    # its materialized records — no duplicate mapping code, no extra dependency.
    # Limitation: reasoning applies only over the fetched slice; for full
    # cross-source reasoning, materialize via the ingestion pipeline instead.

    def register_rest_source(
        self,
        reference_id: str,
        extractor_category: str,
        client: Any,
        *,
        ttl_seconds: float = 60.0,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Register a REST-backed virtual source keyed to an existing extractor.

        ``extractor_category`` is an enrichment registry key (e.g. ``"camunda"``,
        ``"servicenow"``, ``"erpnext"``, ``"leanix"``); ``client`` is the
        duck-typed API client that extractor consumes. Fetches are cached for
        ``ttl_seconds`` to bound query-time latency.
        """
        if not hasattr(self, "_rest_sources"):
            self._rest_sources: dict[str, dict[str, Any]] = {}
        self._rest_sources[reference_id] = {
            "category": extractor_category,
            "client": client,
            "config": dict(config or {}),
            "ttl_seconds": float(ttl_seconds),
            "cache": None,  # tuple(monotonic_ts, batch)
        }
        # Make the virtual source discoverable like SPARQL references.
        self.add_node(  # type: ignore[attr-defined]
            node_id=reference_id,
            node_type="ExternalGraphReference",
            properties={
                "platform": "rest",
                "extractorCategory": extractor_category,
                "ttlSeconds": ttl_seconds,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        logger.info(
            "Registered REST virtual source %s (extractor=%s, ttl=%ss)",
            reference_id,
            extractor_category,
            ttl_seconds,
        )

    def _fetch_rest_batch(self, reference_id: str) -> Any:
        """Fetch (or return cached) ExtractionBatch for a REST virtual source."""
        from ..enrichment.registry import discover_extractors, get_source

        src = getattr(self, "_rest_sources", {}).get(reference_id)
        if src is None:
            raise ValueError(f"No REST virtual source registered as {reference_id}.")

        cache = src.get("cache")
        if cache is not None:
            ts, batch = cache
            if (time.monotonic() - ts) < src["ttl_seconds"]:
                return batch

        extractor = get_source(src["category"])
        if extractor is None:
            discover_extractors()  # lazy-load extractor modules, then retry
            extractor = get_source(src["category"])
        if extractor is None:
            raise ValueError(f"Unknown extractor category {src['category']!r}.")

        config = {"client": src["client"], **src["config"]}
        batch = extractor.extract(config)
        src["cache"] = (time.monotonic(), batch)
        return batch

    def query_rest_source(
        self, reference_id: str, node_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Return materialized records from a REST virtual source (TTL-cached).

        Each record is ``{"id", "type", **props}``. Pass ``node_type`` to filter
        to one canonical type (e.g. ``"Incident"``, ``"BusinessProcess"``).
        """
        batch = self._fetch_rest_batch(reference_id)
        records: list[dict[str, Any]] = []
        for node in batch.nodes:
            if node_type is not None and node.type != node_type:
                continue
            records.append({"id": node.id, "type": node.type, **dict(node.props)})
        return records

    def query_rest_union(
        self,
        reference_id: str,
        local_records: list[dict[str, Any]],
        node_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Union freshly-fetched REST records with locally materialized ones.

        De-duplicated by record ``id`` (local records take precedence).
        """
        merged: dict[str, dict[str, Any]] = {}
        for rec in self.query_rest_source(reference_id, node_type=node_type):
            rid = rec.get("id")
            if rid:
                merged[rid] = rec
        for rec in local_records:  # local wins on id collision
            rid = rec.get("id")
            if rid:
                merged[rid] = rec
        return list(merged.values())

    def _execute_federated_connection(
        self,
        connection: str,
        query: str,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve one named connection and use its enforced read primitive."""
        from agent_utilities.mcp.kg_server import get_connection_registry

        try:
            engine = get_connection_registry().get_engine(connection)
            backend = getattr(engine, "backend", None) or engine
            execute_read = getattr(backend, "execute_read", None)
            if not callable(execute_read):
                raise RuntimeError("federated source lacks a read-only contract")
            return list(execute_read(query, parameters or {}) or [])
        except Exception as exc:
            logger.error("Federated graph query failed (%s)", type(exc).__name__)
            raise RuntimeError("Federated graph execution failed") from exc
