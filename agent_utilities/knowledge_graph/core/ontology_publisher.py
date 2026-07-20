#!/usr/bin/python
"""Ontology Publisher — Export & Push to External Triplestores.

CONCEPT:AU-KG.ontology.enterprise-ontology-distribution — Ontology Distribution

Exports the materialized RDF ontology and pushes it to external
triplestores (Stardog, Apache Jena Fuseki) for enterprise-wide
consumption via SPARQL federation.

Supports:
- Local TTL/RDF-XML export
- Stardog push via pystardog
- Apache Jena Fuseki push via REST API
- Versioned publishing with timestamps

CONCEPT:AU-KG.ontology.authoritative-tbox — Fuseki publish daemon tick: :func:`publish_ontology_to_fuseki`
collects every bundled ``ontology*.ttl`` module into one rdflib graph and pushes
it through :meth:`OntologyPublisher.push_to_jena_fuseki`, so the engine's
maintenance scheduler (``fuseki_publish`` tick, gated by ``KG_FUSEKI_PUBLISH``)
keeps an optional enterprise Fuseki deployment in sync with the evolving
authoritative ontology.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def _http_endpoint(value: Any) -> str:
    endpoint = str(value or "").strip().rstrip("/")
    parsed = urlparse(endpoint)
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("external graph endpoint is missing or invalid")
    return endpoint


def _identifier(value: Any, *, label: str) -> str:
    rendered = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(rendered):
        raise ValueError(f"{label} is missing or invalid")
    return rendered


def _named_graph(value: str | None) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    parsed = urlparse(rendered)
    if (
        len(rendered) > 2048
        or not parsed.scheme
        or any(character in rendered for character in '<>{}"\r\n\t ')
    ):
        raise ValueError("named graph URI is invalid")
    return rendered


def _secret(reference: str | None, fallback: str | None) -> str | None:
    if reference:
        try:
            if reference.startswith("env://"):
                return setting(reference[len("env://") :])
            from agent_utilities.security.secrets_client import create_secrets_client

            return create_secrets_client().resolve_ref(reference)
        except Exception:
            return None
    return fallback


class OntologyPublisher:
    """Export and distribute ontologies to enterprise triplestores.

    CONCEPT:AU-KG.ontology.enterprise-ontology-distribution — Enterprise Ontology Distribution

    This class enables agent-utilities to serve as both the authoritative
    ontology source and a consumer — pushing evolved ontologies back to
    centralized infrastructure (Stardog, Fuseki) for enterprise-wide
    consumption.

    Example::

        publisher = OntologyPublisher()
        # Export locally
        publisher.export_ontology(rdf_graph, ".tmp/ontology.ttl")

        # Push to Stardog
        publisher.push_to_stardog(rdf_graph, endpoint="http://stardog:5820")

        # Push to Fuseki
        publisher.push_to_jena_fuseki(rdf_graph, endpoint="http://jena_fuseki:3030")
    """

    def export_ontology(
        self,
        rdf_graph: Any,
        output_path: str | Path,
        fmt: str = "turtle",
        version_tag: str | None = None,
    ) -> dict[str, Any]:
        """Serialize RDF graph to a local file.

        Args:
            rdf_graph: An rdflib.Graph to serialize.
            output_path: Filesystem path for output.
            fmt: Serialization format (turtle, xml, n3, ntriples, json-ld).
            version_tag: Optional version tag appended to filename.

        Returns:
            Dict with export metadata (path, triple_count, timestamp).
        """
        path = Path(output_path)

        if version_tag:
            stem = path.stem
            suffix = path.suffix
            path = path.parent / f"{stem}_{version_tag}{suffix}"

        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = rdf_graph.serialize(format=fmt)
            if isinstance(data, bytes):
                path.write_bytes(data)
            else:
                path.write_text(data, encoding="utf-8")

            triple_count = len(rdf_graph)
            logger.info("Exported %d ontology triples (format: %s)", triple_count, fmt)

            from agent_utilities.security.persistence_privacy import (
                persistence_reference,
            )

            return {
                "status": "success",
                "artifact_ref": persistence_reference("ontology_export", path),
                "triple_count": triple_count,
                "format": fmt,
                "timestamp": datetime.now(UTC).isoformat(),
                "version_tag": version_tag,
            }
        except Exception as exc:
            logger.error("Ontology export failed (%s)", type(exc).__name__)
            return {"status": "error", "error": type(exc).__name__}

    def push_to_stardog(
        self,
        rdf_graph: Any,
        endpoint: str | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
        password_ref: str | None = None,
        named_graph: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Push ontology to a Stardog triplestore.

        Args:
            rdf_graph: An rdflib.Graph to push.
            endpoint: Stardog server URL (default: env STARDOG_ENDPOINT).
            database: Database name (default: env STARDOG_DATABASE).
            username: Auth username (default: env STARDOG_USER).
            password: Auth password (default: env STARDOG_PASSWORD).
            named_graph: Optional named graph URI for the upload.
            overwrite: When True, REPLACE the target graph — clear it first, then add —
                so re-publishing an updated ontology UPDATES the catalog instead of
                accumulating duplicate/stale triples (CONCEPT:AU-KG.ontology.stardog-catalog-overwrite).
                Scoped to ``named_graph`` when given; otherwise clears the DEFAULT graph.

        Returns:
            Dict with push status and metadata.
        """

        try:
            import stardog
        except ImportError:
            return {
                "status": "error",
                "error": "pystardog not installed. Install with: pip install pystardog",
            }

        try:
            endpoint = _http_endpoint(endpoint or setting("STARDOG_ENDPOINT"))
            database = _identifier(
                database or setting("STARDOG_DATABASE"), label="Stardog database"
            )
            username = str(username or setting("STARDOG_USER") or "").strip()
            password = _secret(
                password_ref or setting("STARDOG_PASSWORD_REF"),
                password or setting("STARDOG_PASSWORD"),
            )
            named_graph = _named_graph(named_graph)
            if not username or not password:
                raise ValueError("Stardog credentials are not configured")
        except Exception as exc:
            return {"status": "error", "error": type(exc).__name__}

        conn_details = {
            "endpoint": endpoint,
            "username": username,
            "password": password,
        }

        try:
            # Serialize to turtle for upload
            ttl_data = rdf_graph.serialize(format="turtle")
            if isinstance(ttl_data, str):
                ttl_data = ttl_data.encode("utf-8")

            conn = stardog.Connection(database, **conn_details)
            try:
                conn.begin()
                # Overwrite = clear-then-add so an updated ontology REPLACES the prior
                # catalog slice rather than accumulating (CONCEPT:AU-KG.ontology.stardog-catalog-overwrite).
                if overwrite:
                    if named_graph:
                        try:
                            conn.clear(graph_uri=named_graph)
                        except TypeError:  # older pystardog: clear() takes no kwarg
                            conn.update(f"CLEAR GRAPH <{named_graph}>")
                    else:
                        conn.update("CLEAR DEFAULT")
                content = stardog.content.Raw(ttl_data, content_type="text/turtle")
                if named_graph:
                    conn.add(content, graph_uri=named_graph)
                else:
                    conn.add(content)
                conn.commit()

                triple_count = len(rdf_graph)
                logger.info("Pushed %d ontology triples to Stardog", triple_count)

                return {
                    "status": "success",
                    "endpoint_configured": True,
                    "triple_count": triple_count,
                    "named_graph": named_graph,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

        except Exception as exc:
            logger.error("Stardog push failed (%s)", type(exc).__name__)
            return {"status": "error", "error": type(exc).__name__}

    def push_to_jena_fuseki(
        self,
        rdf_graph: Any,
        endpoint: str | None = None,
        dataset: str = "agent_kg",
        named_graph: str | None = None,
        username: str | None = None,
        password_ref: str | None = None,
        tls_profile: str | None = None,
        tls_profile_ref: str | None = None,
    ) -> dict[str, Any]:
        """Push ontology to Apache Jena Fuseki via REST API.

        Args:
            rdf_graph: An rdflib.Graph to push.
            endpoint: Fuseki server URL; ``None`` defers to the canonical
                ``kg_fuseki_endpoint`` config field (``KG_FUSEKI_ENDPOINT``).
            dataset: Dataset name.
            named_graph: Optional named graph URI.

        Returns:
            Dict with push status and metadata.
        """

        if endpoint is None:
            from agent_utilities.core.config import config as _cfg

            endpoint = _cfg.kg_fuseki_endpoint

        try:
            endpoint = _http_endpoint(endpoint)
            dataset = _identifier(dataset, label="Fuseki dataset")
            named_graph = _named_graph(named_graph)
            username = str(username or setting("GRAPH_FUSEKI_USER") or "").strip()
            password = _secret(
                password_ref or setting("GRAPH_FUSEKI_PASSWORD_REF"),
                None,
            )
            if bool(username) != bool(password):
                raise ValueError("Fuseki credentials are incomplete")
        except Exception as exc:
            return {"status": "error", "error": type(exc).__name__}

        # Fuseki Graph Store Protocol endpoint
        url = f"{endpoint}/{dataset}/data"
        params = {}
        if named_graph:
            params["graph"] = named_graph
        else:
            params["default"] = ""

        try:
            from agent_utilities.core.http_client import create_requests_session
            from agent_utilities.core.transport_security import (
                resolve_configured_tls_profile,
            )

            trust = resolve_configured_tls_profile(
                "FUSEKI",
                profile_name=tls_profile,
                profile_ref=tls_profile_ref,
            )
            ttl_data = rdf_graph.serialize(format="turtle")
            if isinstance(ttl_data, str):
                ttl_data = ttl_data.encode("utf-8")

            with create_requests_session(transport_security=trust) as session:
                response = session.put(
                    url,
                    data=ttl_data,
                    params=params,
                    headers={"Content-Type": "text/turtle"},
                    auth=(username, password) if username and password else None,
                    timeout=(10, 30),
                    allow_redirects=False,
                )
                response.raise_for_status()

            triple_count = len(rdf_graph)
            logger.info("Pushed %d ontology triples to Fuseki", triple_count)

            return {
                "status": "success",
                "endpoint_configured": True,
                "triple_count": triple_count,
                "named_graph": named_graph,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as exc:
            logger.error("Fuseki push failed (%s)", type(exc).__name__)
            return {"status": "error", "error": type(exc).__name__}
        finally:
            if "trust" in locals():
                trust.cleanup()


def import_ontology_from_stardog(
    *,
    endpoint: str | None = None,
    database: str | None = None,
    username: str | None = None,
    password: str | None = None,
    password_ref: str | None = None,
    named_graph: str | None = None,
    engine: Any = None,
    activate: bool = True,
) -> dict[str, Any]:
    """Consume an ontology FROM Stardog INTO epistemic-graph (CONCEPT:AU-KG.ontology.stardog-catalog-import).

    The reverse of :meth:`OntologyPublisher.push_to_stardog`: export the TBox that already
    lives in a Stardog database / named graph as Turtle and, when an ``engine`` is given,
    parse → validate → register → activate it through the ontology lifecycle so the engine
    reasons over the catalog we already have there. Without an ``engine`` it just returns the
    pulled Turtle (offline inspection / round-trip).

    Returns a status dict; ``load`` carries the lifecycle report when loaded into an engine.
    """
    try:
        import stardog
    except ImportError:
        return {
            "status": "error",
            "error": "pystardog not installed. Install with: pip install pystardog",
        }

    try:
        endpoint = _http_endpoint(endpoint or setting("STARDOG_ENDPOINT"))
        database = _identifier(
            database or setting("STARDOG_DATABASE"), label="Stardog database"
        )
        username = str(username or setting("STARDOG_USER") or "").strip()
        password = _secret(
            password_ref or setting("STARDOG_PASSWORD_REF"),
            password or setting("STARDOG_PASSWORD"),
        )
        named_graph = _named_graph(named_graph)
        if not username or not password:
            raise ValueError("Stardog credentials are not configured")
    except Exception as exc:
        return {"status": "error", "error": type(exc).__name__}
    conn_details = {"endpoint": endpoint, "username": username, "password": password}

    try:
        conn = stardog.Connection(database, **conn_details)
        try:
            ttl = conn.export(content_type="text/turtle", graph_uri=named_graph)
        finally:
            conn.close()
        if isinstance(ttl, bytes):
            ttl = ttl.decode("utf-8")
    except Exception as exc:
        logger.error("Stardog ontology import failed (%s)", type(exc).__name__)
        return {"status": "error", "error": type(exc).__name__}

    result: dict[str, Any] = {
        "status": "success",
        "endpoint_configured": True,
        "named_graph": named_graph,
        "bytes": len(ttl or ""),
    }
    if engine is not None and ttl:
        try:
            from ..ontology.lifecycle import OntologyLifecycle

            result["load"] = OntologyLifecycle(engine).load(
                ttl, source_type="turtle", activate=activate, force=True
            )
        except Exception as exc:  # noqa: BLE001 — engine load is best-effort
            result["load"] = {"status": "error", "error": type(exc).__name__}
    else:
        result["turtle"] = ttl
    return result


def import_ontology_from_stardog(
    *,
    endpoint: str | None = None,
    database: str | None = None,
    username: str | None = None,
    password: str | None = None,
    named_graph: str | None = None,
    engine: Any = None,
    activate: bool = True,
) -> dict[str, Any]:
    """Consume an ontology FROM Stardog INTO epistemic-graph (CONCEPT:AU-KG.ontology.stardog-catalog-import).

    The reverse of :meth:`OntologyPublisher.push_to_stardog`: export the TBox that already
    lives in a Stardog database / named graph as Turtle and, when an ``engine`` is given,
    parse → validate → register → activate it through the ontology lifecycle so the engine
    reasons over the catalog we already have there. Without an ``engine`` it just returns the
    pulled Turtle (offline inspection / round-trip).

    Returns a status dict; ``load`` carries the lifecycle report when loaded into an engine.
    """
    try:
        import stardog
    except ImportError:
        return {
            "status": "error",
            "error": "pystardog not installed. Install with: pip install pystardog",
        }

    endpoint = endpoint or setting("STARDOG_ENDPOINT", "http://localhost:5820")
    database = database or setting("STARDOG_DATABASE", "agent_kg")
    username = username or setting("STARDOG_USER", "admin")
    password = password or setting("STARDOG_PASSWORD", "admin")
    conn_details = {"endpoint": endpoint, "username": username, "password": password}

    try:
        conn = stardog.Connection(database, **conn_details)
        try:
            ttl = conn.export(content_type="text/turtle", graph_uri=named_graph)
        finally:
            conn.close()
        if isinstance(ttl, bytes):
            ttl = ttl.decode("utf-8")
    except Exception as e:
        logger.error("Stardog ontology import failed: %s", e)
        return {"status": "error", "error": str(e)}

    result: dict[str, Any] = {
        "status": "success",
        "endpoint": endpoint,
        "database": database,
        "named_graph": named_graph,
        "bytes": len(ttl or ""),
    }
    if engine is not None and ttl:
        try:
            from ..ontology.lifecycle import OntologyLifecycle

            result["load"] = OntologyLifecycle(engine).load(
                ttl, source_type="turtle", activate=activate, force=True
            )
        except Exception as e:  # noqa: BLE001 — engine load is best-effort
            result["load"] = {"status": "error", "error": str(e)}
    else:
        result["turtle"] = ttl
    return result


def collect_bundled_ontology_graph() -> Any:
    """Parse every bundled ``ontology*.ttl`` module into one rdflib graph.

    CONCEPT:AU-KG.ontology.authoritative-tbox — the authoritative TBox the platform ships (core
    ``ontology.ttl`` plus all domain modules under ``knowledge_graph/``) merged
    into a single graph for distribution. Unparseable modules are skipped with
    a warning so one bad file never blocks the publish of the rest.

    Returns:
        An ``rdflib.Graph`` with the merged ontology (empty if rdflib is
        unavailable or no module parses).
    """
    import rdflib

    graph = rdflib.Graph()
    modules_dir = Path(__file__).parent.parent  # .../knowledge_graph
    for ttl_path in sorted(modules_dir.glob("ontology*.ttl")):
        try:
            graph.parse(str(ttl_path), format="turtle")
        except Exception as exc:  # noqa: BLE001 — one bad module never blocks the rest
            logger.warning(
                "Skipping an unparseable ontology module (%s)",
                type(exc).__name__,
            )
    # CONCEPT:AU-KG.ontology.federation-provider-leg — federation: contributed ontology modules from installed
    # fleet packages (declared via the ``agent_utilities.ontology_providers``
    # entry-point) are parsed into the SAME published TBox as the bundled modules,
    # so a moved module (e.g. servicenow now living in the servicenow-api wheel) is
    # indistinguishable from a bundled one. Failure-isolated per file.
    try:
        from .ontology_federation import resolve_provider_ontologies

        for _provider, ttl_path in resolve_provider_ontologies():
            try:
                graph.parse(str(ttl_path), format="turtle")
            except Exception as exc:  # noqa: BLE001 — one bad module never blocks rest
                logger.warning(
                    "Skipping an unparseable contributed ontology (%s)",
                    type(exc).__name__,
                )
    except Exception as exc:  # noqa: BLE001 — federation is additive, never fatal
        logger.debug(
            "Ontology federation discovery unavailable (%s)", type(exc).__name__
        )
    return graph


def publish_ontology_to_fuseki(
    endpoint: str | None = None,
    dataset: str = "agent_kg",
    named_graph: str | None = None,
    publisher: OntologyPublisher | None = None,
) -> dict[str, Any]:
    """Push the bundled ontology modules to Apache Jena Fuseki.

    CONCEPT:AU-KG.ontology.authoritative-tbox — the callable the ``fuseki_publish`` daemon tick runs
    (and tests exercise with an injected ``publisher``). Resolution of a
    ``None`` ``endpoint`` is delegated to
    :meth:`OntologyPublisher.push_to_jena_fuseki` (the canonical
    ``kg_fuseki_endpoint`` config field / ``KG_FUSEKI_ENDPOINT``) so endpoint
    config lives in exactly one place.

    Args:
        endpoint: Fuseki server URL; ``None`` defers to the publisher.
        dataset: Fuseki dataset name.
        named_graph: Optional named-graph URI for the upload.
        publisher: Injected publisher (tests/mocks); default builds one.

    Returns:
        The publisher's push report dict (``status``/``triple_count``/...).
    """
    try:
        graph = collect_bundled_ontology_graph()
    except ImportError:
        return {"status": "error", "error": "rdflib not installed."}
    if len(graph) == 0:
        return {"status": "skipped", "reason": "no ontology triples collected"}
    publisher = publisher or OntologyPublisher()
    return publisher.push_to_jena_fuseki(
        graph, endpoint=endpoint, dataset=dataset, named_graph=named_graph
    )
