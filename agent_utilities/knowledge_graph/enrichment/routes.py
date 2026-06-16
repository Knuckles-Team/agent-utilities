"""HTTP route extraction + code↔service linking (CONCEPT:KG-2.102).

A web service's *routes* are the seam between its code and the rest of the
ecosystem: a handler function is reached through an HTTP route, and that route is
served by a deployed Service that runs on a Node. We extract the code-level half
here — ``Route`` nodes + ``serves`` edges from the handler ``Code`` symbol — from
the route decorators the Rust parser captured. The OWL surpass then links each
``Route`` to a deployed ecosystem ``Service`` (``servedBy``), so reasoning can
chain ``Code –serves→ Route –servedBy→ Service –deployedOn→ Node`` — something a
siloed per-repo code tool cannot do, because it never sees the live topology.
"""

from __future__ import annotations

import re

from .models import CodeEntity, EnrichmentEdge, GraphNode

# `<obj>.<verb>("/path"...)` route decorators (Flask `@app.route`, FastAPI/Flask
# 2.x `@app.get`/`@router.post`, etc.). The verb selects the HTTP method; `route`
# defers to a `methods=[...]` kwarg (default GET).
_ROUTE_RE = re.compile(
    r"""^\w+\.(?P<verb>route|get|post|put|delete|patch|head|options)\s*\(\s*"""
    r"""['"](?P<path>[^'"]+)['"]""",
    re.IGNORECASE,
)
_METHODS_RE = re.compile(r"""methods\s*=\s*\[([^\]]*)\]""", re.IGNORECASE)


def _parse_route_decorator(decorator: str) -> list[tuple[str, str]]:
    """The (method, path) pairs a route decorator declares, or ``[]``.

    A ``@app.route("/x", methods=["GET","POST"])`` yields one pair per method; a
    verb decorator (``@app.get("/x")``) yields a single pair.
    """
    m = _ROUTE_RE.match(decorator.strip())
    if not m:
        return []
    path = m.group("path")
    verb = m.group("verb").lower()
    if verb == "route":
        mm = _METHODS_RE.search(decorator)
        methods = (
            [
                s.strip().strip("'\"").upper()
                for s in mm.group(1).split(",")
                if s.strip()
            ]
            if mm
            else ["GET"]
        )
        return [(method, path) for method in methods if method]
    return [(verb.upper(), path)]


def extract_routes(
    code: list[CodeEntity],
) -> tuple[list[GraphNode], list[EnrichmentEdge]]:
    """Extract ``Route`` nodes + ``SERVES`` edges (handler → route) from the
    route decorators on code symbols (CONCEPT:KG-2.102)."""
    routes: dict[str, GraphNode] = {}
    edges: list[EnrichmentEdge] = []
    seen: set[tuple[str, str]] = set()
    for c in code:
        for dec in c.decorators:
            for method, path in _parse_route_decorator(dec):
                rid = f"route:{method}:{path}"
                routes.setdefault(
                    rid,
                    GraphNode(
                        id=rid, type="Route", props={"method": method, "path": path}
                    ),
                )
                key = (c.id, rid)
                if key not in seen:
                    seen.add(key)
                    edges.append(
                        EnrichmentEdge(source=c.id, target=rid, rel_type="SERVES")
                    )
    return list(routes.values()), edges


def link_routes_to_service(
    routes: list[GraphNode], service_id: str
) -> list[EnrichmentEdge]:
    """Link every ``Route`` to the deployed ecosystem ``Service`` that serves it
    (``servedBy``) — the code↔topology bridge the OWL reasoner chains through
    (CONCEPT:KG-2.102). ``service_id`` is the resolved ecosystem Service node id."""
    if not service_id:
        return []
    return [
        EnrichmentEdge(source=r.id, target=service_id, rel_type="SERVED_BY")
        for r in routes
    ]


def resolve_service_id(source_system: str, service_ids: set[str]) -> str:
    """Best-effort match of an ingested code source to a deployed Service node by
    name. ``source_system`` is e.g. ``gitlab:acme:<proj>`` or a repo slug; the
    Service node id is matched on its trailing name segment. Empty string when no
    confident match — we never invent a topology link."""
    if not source_system or not service_ids:
        return ""
    name = source_system.rsplit(":", 1)[-1].rsplit("/", 1)[-1].lower()
    if not name:
        return ""
    for sid in service_ids:
        sid_name = sid.rsplit(":", 1)[-1].rsplit("/", 1)[-1].lower()
        if sid_name == name:
            return sid
    return ""
