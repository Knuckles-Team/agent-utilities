"""Pydantic request/response models for the ``agent_utilities.gateway`` REST surface.

CONCEPT:AU-KG.query.typed-rest-schemas — the ``/api/graph/*`` REST routes are mounted
in ``agent_utilities/mcp/kg_server.py::_mount_rest_routes`` with raw Starlette
``app.add_route(...)`` handlers, which carry no OpenAPI schema. The models in this
package are the typed request/response contracts a later wiring pass attaches to those
routes (via FastAPI's ``add_api_route(..., response_model=...)``) so the live OpenAPI
document actually describes the ``/api/graph/*`` surface instead of showing zero paths
for it.

Sibling modules, one per route family (e.g. ``graph_analyze.py`` for
``/api/graph/analyze/*`` and ``/api/graph/search/*``, ``graph_query.py`` for
``/api/graph/query/*``, ...), are added by separate, parallel lanes. This module
deliberately has NO imports or re-exports: two lanes land sibling modules here at the
same time, and any export list here would be a guaranteed merge collision. Import each
family's models directly from its module, e.g.::

    from agent_utilities.gateway.schemas.graph_analyze import GraphAnalyzeRequest
"""

from __future__ import annotations
