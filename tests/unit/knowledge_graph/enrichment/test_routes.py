"""HTTP route extraction + code↔service linking (CONCEPT:AU-KG.compute.http-route-graph)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.models import CodeEntity, GraphNode
from agent_utilities.knowledge_graph.enrichment.routes import (
    extract_routes,
    link_routes_to_service,
    resolve_service_id,
)


def _code(name, decorators):
    return CodeEntity(
        id=f"code:app.py::{name}",
        name=name,
        qualname=name,
        kind="function",
        language="python",
        file_path="app.py",
        line=1,
        ast_hash="h",
        decorators=decorators,
    )


def test_extract_routes_flask_and_fastapi():
    code = [
        _code("list_users", ['app.route("/users", methods=["GET", "POST"])']),
        _code("get_user", ['router.get("/users/{id}")']),
        _code("plain", ["staticmethod"]),  # not a route
    ]
    routes, edges = extract_routes(code)

    rids = {r.id for r in routes}
    # Flask methods= kwarg fans out to one Route per method.
    assert "route:GET:/users" in rids
    assert "route:POST:/users" in rids
    # FastAPI verb decorator.
    assert "route:GET:/users/{id}" in rids
    # Non-route decorator yields nothing.
    assert not any("plain" in e.source and e.rel_type == "SERVES" for e in edges)

    serves = {(e.source, e.target) for e in edges if e.rel_type == "SERVES"}
    assert ("code:app.py::list_users", "route:GET:/users") in serves
    assert ("code:app.py::get_user", "route:GET:/users/{id}") in serves
    # Route node carries method + path props.
    r = next(r for r in routes if r.id == "route:GET:/users/{id}")
    assert (
        r.type == "Route"
        and r.props["method"] == "GET"
        and r.props["path"] == "/users/{id}"
    )


def test_resolve_service_id_matches_by_name():
    services = {"service:cluster:payments-api", "service:cluster:users-api"}
    assert (
        resolve_service_id("gitlab:acme:42:users-api", services)
        == "service:cluster:users-api"
    )
    # No confident match → empty (never invent a topology link).
    assert resolve_service_id("gitlab:acme:42:unknown", services) == ""
    assert resolve_service_id("", services) == ""


def test_link_routes_to_service_emits_served_by():
    routes = [GraphNode(id="route:GET:/x", type="Route", props={})]
    edges = link_routes_to_service(routes, "service:cluster:users-api")
    assert len(edges) == 1
    assert edges[0].source == "route:GET:/x"
    assert edges[0].target == "service:cluster:users-api"
    assert edges[0].rel_type == "SERVED_BY"
    # No service id → no edge.
    assert link_routes_to_service(routes, "") == []
