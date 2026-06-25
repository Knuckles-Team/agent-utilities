#!/usr/bin/python
"""SPARQL HTTP Endpoint.

CONCEPT:KG-2.6 — W3C SPARQL Protocol HTTP Endpoint

Provides a standards-compliant SPARQL endpoint that other agent-utilities
deployments can consume via HTTP. Backed by rdflib materialization from
the OWLBridge.

This module creates a Starlette ASGI app that can be mounted onto
FastMCP's underlying server or run standalone.

References:
    - W3C SPARQL Protocol: https://www.w3.org/TR/sparql11-protocol/
    - W3C SPARQL Results JSON: https://www.w3.org/TR/sparql11-results-json/
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SPARQLEndpoint:
    """W3C SPARQL Protocol endpoint backed by rdflib.

    CONCEPT:KG-2.6 — SPARQL HTTP Endpoint

    Provides GET and POST handlers for SPARQL queries. The endpoint
    materializes the LPG into an rdflib Graph (via OWLBridge) and
    executes standard SPARQL SELECT, ASK, CONSTRUCT queries.

    Usage::

        endpoint = SPARQLEndpoint(owl_bridge)
        # Mount as Starlette route
        app.mount("/sparql", endpoint.as_asgi())

        # Or query directly
        results = endpoint.execute("SELECT ?s WHERE { ?s a au:Agent }")
    """

    def __init__(self, owl_bridge: Any) -> None:
        """Initialize with an OWLBridge instance.

        Args:
            owl_bridge: OWLBridge exposing query_sparql() (engine-native SPARQL
                with a rdflib/regex last-resort fallback).
        """
        self._bridge = owl_bridge

    def execute(
        self,
        query: str,
        result_format: str = "json",
    ) -> dict[str, Any]:
        """Execute a SPARQL query and return structured results."""
        if result_format not in ["json", "turtle", "xml"]:
            return {"error": f"Unsupported format: {result_format}"}

        try:
            # CONCEPT:KG-2.204 — go through query_sparql so the query runs on the
            # engine's native SPARQL surface (live graph) by default, with the rdflib
            # materialization only as a no-engine last resort.
            raw_results = self._bridge.query_sparql(query)
        except ImportError:
            return {"error": "rdflib not installed. Install with: pip install rdflib"}
        except Exception as e:
            return {"error": f"SPARQL execution failed: {e}"}

        # Convert to W3C SPARQL Results JSON format
        if raw_results and "result" in raw_results[0]:
            # ASK query result
            return {
                "head": {},
                "boolean": raw_results[0]["result"],
            }

        if raw_results and "subject" in raw_results[0]:
            # CONSTRUCT result — return as-is
            return {
                "results": {"bindings": raw_results},
                "type": "construct",
            }

        # SELECT results
        variables = list(raw_results[0].keys()) if raw_results else []
        bindings = []
        for row in raw_results:
            binding = {}
            for var in variables:
                val = row.get(var)
                if val is not None:
                    # Determine type
                    if val.startswith("http://") or val.startswith("https://"):
                        binding[var] = {"type": "uri", "value": val}
                    else:
                        binding[var] = {"type": "literal", "value": val}
            bindings.append(binding)

        return {
            "head": {"vars": variables},
            "results": {"bindings": bindings},
        }

    def handle_request(
        self,
        query: str | None = None,
        accept: str = "application/sparql-results+json",
    ) -> tuple[str, str, int]:
        """Handle an HTTP SPARQL request.

        Args:
            query: SPARQL query string from ?query= parameter or POST body.
            accept: Accept header for content negotiation.

        Returns:
            Tuple of (response_body, content_type, status_code).
        """
        if not query:
            return (
                json.dumps({"error": "Missing 'query' parameter"}),
                "application/json",
                400,
            )

        result = self.execute(query)

        if "error" in result:
            return (
                json.dumps(result),
                "application/json",
                500,
            )

        return (
            json.dumps(result, default=str),
            "application/sparql-results+json",
            200,
        )

    def as_asgi(self) -> Any:
        """Create a Starlette ASGI app for this endpoint.

        Returns:
            A Starlette application that handles GET/POST SPARQL requests.

        Raises:
            ImportError: If starlette is not installed.
        """
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import Response
        from starlette.routing import Route

        async def sparql_handler(request: Request) -> Response:
            if request.method == "GET":
                query = request.query_params.get("query")
            else:
                body = await request.body()
                content_type = request.headers.get("content-type", "")
                if "application/sparql-query" in content_type:
                    query = body.decode("utf-8")
                else:
                    # Form-encoded
                    from urllib.parse import parse_qs

                    params = parse_qs(body.decode("utf-8"))
                    query = params.get("query", [None])[0]

            accept = request.headers.get("accept", "application/sparql-results+json")
            body_str, content_type_str, status = self.handle_request(query, accept)

            return Response(
                content=body_str,
                media_type=content_type_str,
                status_code=status,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                },
            )

        return Starlette(
            routes=[
                Route("/", sparql_handler, methods=["GET", "POST"]),
            ]
        )
