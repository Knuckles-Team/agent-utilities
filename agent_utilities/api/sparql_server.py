import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sparql", tags=["sparql"])


class SparqlTranspiler:
    """
    Translates incoming standard SPARQL queries into native
    epistemic-graph traversals. This ensures we can serve Semantic Web
    queries without risking split-brain state issues, as all requests
    are securely proxied to the local Rust operational Truth.
    """

    def __init__(self):
        self.engine = GraphComputeEngine()

    def transpile_and_execute(self, sparql_query: str) -> dict[str, Any]:
        """
        Takes a SPARQL string, parses the abstract syntax, and maps
        it to native `self.engine` graph lookups.

        Currently acts as an architectural translation stub demonstrating the connection
        pipeline for basic Subject-Predicate-Object selections.
        """
        results = []
        # In a full deployment, this logic is replaced by rdflib.plugins.sparql.parser
        # to generate ASTs matching our epistemic_graph topology constraints.
        if "SELECT" in sparql_query.upper():
            # For demonstration, we simply dump edges representing a wildcard graph query
            # and format them perfectly into the standard SPARQL-JSON W3C output schema.
            for u, v in self.engine._get_all_edges():
                props = self.engine._get_edge_properties(u, v)
                results.append(
                    {
                        "subject": {"type": "uri", "value": u},
                        "predicate": {
                            "type": "literal",
                            "value": props.get("type", "UNKNOWN"),
                        },
                        "object": {"type": "uri", "value": v},
                    }
                )

        return {
            "head": {"vars": ["subject", "predicate", "object"]},
            "results": {"bindings": results},
        }


transpiler = SparqlTranspiler()


@router.get("/")
@router.post("/")
async def sparql_endpoint(request: Request):
    """
    Standard Semantic Web SPARQL 1.1 Endpoint.
    Accepts standard GET (via query param) and POST (via form/json body) requests.
    """
    query = ""
    if request.method == "GET":
        query = request.query_params.get("query", "")
    elif request.method == "POST":
        content_type = request.headers.get("Content-Type", "")
        if "application/x-www-form-urlencoded" in content_type:
            form_body = await request.form()
            _q = form_body.get("query", "")
            query = _q if isinstance(_q, str) else ""
        elif "application/sparql-query" in content_type:
            body = await request.body()
            query = body.decode("utf-8")
        else:
            try:
                json_body = await request.json()
                query = json_body.get("query", "")
            except Exception:
                pass

    if not query:
        raise HTTPException(
            status_code=400, detail="Missing 'query' parameter for SPARQL endpoint."
        )

    try:
        # Route query through our native Epistemic Graph transpiler
        response_data = transpiler.transpile_and_execute(str(query))
        return JSONResponse(
            content=response_data,
            headers={"Content-Type": "application/sparql-results+json"},
        )
    except Exception as e:
        logger.error(f"SPARQL execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to transpile and execute SPARQL query."
        ) from e
