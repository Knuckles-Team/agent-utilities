"""Auto-extracted graph-os MCP tools: query_tools (register_query_tools).

Split out of kg_server._build_server to deepen the MCP surface into focused
modules without changing tool behavior or names.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import Field

from agent_utilities.core.event_loop import run_blocking_ordered
from agent_utilities.knowledge_graph.orchestration.engine_query import (
    is_aggregation_cypher,
)
from agent_utilities.mcp import kg_server
from agent_utilities.models.evidence_bundle import EvidenceBundle
from agent_utilities.security.error_surface import public_error_json, public_error_text

logger = logging.getLogger(__name__)

# CONCEPT:AU-KG.query.query-aggregation — `is_aggregation_cypher` (imported above)
# is re-exported here so `from agent_utilities.mcp.tools.query_tools import
# is_aggregation_cypher` keeps working for existing callers/tests. Its canonical
# definition + detection regex now live in `knowledge_graph.orchestration.
# engine_query` (single source of truth): the governed read path
# (`QueryMixin.query_cypher`) needs the SAME aggregate-vs-row detection this
# federation router uses, and `engine_query` sits below `mcp.tools` in the
# dependency direction, so it is the correct owner, not this module.


def _parse_ranges(spec: str) -> list[tuple[int, int]]:
    """Parse a char-range spec like '96..208,300..420' into (lo, hi) pairs.

    CONCEPT:AU-KG.retrieval.tree-navigation — the fetch half of map-then-fetch.
    """
    out: list[tuple[int, int]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ".." not in part:
            raise ValueError(f"invalid range {part!r}: use 'start..end'")
        lo_s, hi_s = part.split("..", 1)
        try:
            lo, hi = int(lo_s), int(hi_s)
        except ValueError:
            raise ValueError("invalid range") from None
        if hi < lo:
            raise ValueError(f"invalid range {part!r}: end < start")
        out.append((lo, hi))
    if not out:
        raise ValueError("no ranges given (use 'start..end,start..end')")
    return out


def _persist_sections(
    engine: Any,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> bool:
    """Write section-tree nodes/edges through the engine's own API (CONCEPT:AU-KG.retrieval.section-tree).

    Uses ``engine.add_node(id, type, props)`` / ``engine.add_edge(src, tgt, rel)``
    — the same convenience write path ``write_ingest_tools`` uses — so the tree
    persists regardless of the backend's keyword-writer support. Best-effort.
    """
    ok = False
    for n in nodes:
        if _persist_section_node(engine, n):
            ok = True
    for e in edges:
        if _persist_section_edge(engine, e):
            ok = True
    return ok


def _persist_section_node(engine: Any, n: dict[str, Any]) -> bool:
    props = {k: v for k, v in n.items() if k not in ("id", "type")}
    try:
        engine.add_node(n["id"], n["type"], props)
        return True
    except Exception:  # noqa: BLE001 — best-effort per node
        return False


def _persist_section_edge(engine: Any, e: dict[str, Any]) -> bool:
    props = {k: v for k, v in e.items() if k not in ("source", "target", "type")}
    try:
        engine.add_edge(e["source"], e["target"], e["type"], **props)
        return True
    except Exception:  # noqa: BLE001 — best-effort per edge
        return False


def _json_default(obj: Any) -> Any:
    """``json.dumps(default=...)`` helper that dataclass-serializes an
    :class:`~agent_utilities.knowledge_graph.core.epistemic_row.EpistemicRow`
    (or any other dataclass instance) instead of stringifying it (CONCEPT:AU-KB-CURRENCY).

    Used wherever a tool's result may carry ``include_epistemic=True`` rows —
    plain dicts pass through ``json.dumps`` untouched; only genuinely
    non-serializable values (dataclass instances) reach this hook.
    """
    import dataclasses

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return str(obj)


#: Code-symbol nav actions over the resolved graph (CONCEPT:AU-KG.backend.declared-columns-so-schema).
CODE_NAV_ACTIONS = frozenset(
    {"find_definition", "find_references", "trace_call_graph", "impact_of_change"}
)

#: Symbol→symbol path action (CONCEPT:EG-KG.compute.handled-outside-single-anchor) — handled outside the single-anchor
#: Cypher template because it resolves TWO endpoints and runs a native path search.
PATH_ACTIONS = frozenset({"connects"})

# Columns returned for a :Code node (kept identical across actions for a stable shape).
_CODE_COLS = (
    "{var}.id AS id, {var}.name AS name, {var}.file_path AS file_path, "
    "{var}.line AS line, {var}.language AS language, {var}.kind_detail AS kind, "
    "{var}.instance AS instance, {var}.source_system AS source_system"
)


def build_code_nav_query(
    *,
    action: str,
    symbol: str = "",
    node_id: str = "",
    source_system: str = "",
    depth: int = 3,
    limit: int = 200,
) -> tuple[str, dict[str, Any]]:
    """Build the (read-only) Cypher + params for a ``graph_code_nav`` action.

    Pure and side-effect-free so the templates are unit-tested without an engine.
    Operates on the resolved code graph: ``:Code`` symbols joined by ``calls``
    (symbol→symbol) edges. ``depth`` is validated and inlined (Cypher forbids a
    parameterized variable-length bound); everything else is parameterized.
    """
    if action not in CODE_NAV_ACTIONS:
        raise ValueError(
            f"unknown action '{action}'; expected one of {sorted(CODE_NAV_ACTIONS)}"
        )
    if not symbol and not node_id:
        raise ValueError("provide 'symbol' or 'node_id'")
    try:
        depth = max(1, min(10, int(depth)))
        limit = max(1, min(5000, int(limit)))
    except (TypeError, ValueError) as exc:
        raise ValueError("depth/limit must be integers") from exc

    params: dict[str, Any] = {}
    # Match clause for the anchor :Code node (id wins over name), plus an optional
    # source_system scope, built as a WHERE so it composes across actions.
    conds: list[str] = []
    if node_id:
        conds.append("{var}.id = $node_id")
        params["node_id"] = node_id
    else:
        conds.append("{var}.name = $symbol")
        params["symbol"] = symbol
    if source_system:
        conds.append("{var}.source_system = $src")
        params["src"] = source_system

    def where_for(var: str) -> str:
        return " AND ".join(c.format(var=var) for c in conds)

    cols = _CODE_COLS

    if action == "find_definition":
        cypher = (
            f"MATCH (c:Code) WHERE {where_for('c')} "
            f"RETURN {cols.format(var='c')} LIMIT {limit}"
        )
    elif action == "find_references":
        # Callers: incoming `calls` edges to the anchor symbol.
        cypher = (
            f"MATCH (caller:Code)-[:calls]->(def:Code) WHERE {where_for('def')} "
            f"RETURN DISTINCT {cols.format(var='caller')} LIMIT {limit}"
        )
    elif action == "trace_call_graph":
        # Transitive callees reachable from the anchor (downstream).
        cypher = (
            f"MATCH (s:Code)-[:calls*1..{depth}]->(callee:Code) WHERE {where_for('s')} "
            f"RETURN DISTINCT {cols.format(var='callee')} LIMIT {limit}"
        )
    else:  # impact_of_change — transitive callers (upstream blast radius).
        cypher = (
            f"MATCH (caller:Code)-[:calls*1..{depth}]->(t:Code) WHERE {where_for('t')} "
            f"RETURN DISTINCT {cols.format(var='caller')} LIMIT {limit}"
        )
    return cypher, params


def _evidence_bundle_for_rows(engine: Any, rows: list[Any]) -> Any:
    """Build an :class:`~agent_utilities.models.evidence_bundle.EvidenceBundle`
    over a plain Cypher result ``rows`` for ``graph_query``'s typed response.

    Currency-upgrades ``rows`` the SAME way
    :meth:`~agent_utilities.knowledge_graph.facade.KnowledgeGraph._attach_epistemic`
    does for its per-row ``EpistemicRow`` path: extract the distinct ids
    (:func:`~agent_utilities.knowledge_graph.core.epistemic_row.row_ids_from_plain_rows`),
    resolve their engine-side epistemic envelope in ONE round-trip via
    ``engine.graph.explain_provenance_by_ids`` (``Method::ExplainProvenanceByIds``),
    then fold the resulting ``KnowledgeSet`` rows into ONE aggregate
    :class:`EvidenceBundle` via :meth:`EvidenceBundle.from_engine_wire` — confidence
    (top-scored row), freshness (bitemporal min/max), contradictions (scanned across
    row claims), and policy exclusions, never fabricated. Degrades to an empty
    bundle (never raises) when the engine has no ``explain_provenance_by_ids``
    primitive or no row carries a resolvable id.
    """
    from agent_utilities.knowledge_graph.core.epistemic_row import (
        row_ids_from_plain_rows,
    )
    from agent_utilities.models.evidence_bundle import EvidenceBundle

    fetch = getattr(getattr(engine, "graph", None), "explain_provenance_by_ids", None)
    if fetch is None:
        return EvidenceBundle.from_engine_wire({"rows": []})
    plain_rows = (
        [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
    )
    id_props = row_ids_from_plain_rows(plain_rows)
    if not id_props:
        return EvidenceBundle.from_engine_wire({"rows": []})
    wire_rows = fetch([ip["id"] for ip in id_props]) or []
    return EvidenceBundle.from_engine_wire({"rows": wire_rows})


def _resolve_symbol_id(engine, *, symbol: str, node_id: str) -> dict[str, Any] | None:
    """Resolve a symbol name (or exact id) to its best :Code node row."""
    if node_id:
        cypher, params = build_code_nav_query(
            action="find_definition", node_id=node_id, limit=1
        )
    else:
        cypher, params = build_code_nav_query(
            action="find_definition", symbol=symbol, limit=1
        )
    rows = engine.query_cypher(cypher, params)
    return rows[0] if rows else None


def code_connects(
    engine,
    *,
    symbol: str = "",
    node_id: str = "",
    target_symbol: str = "",
    target_node_id: str = "",
) -> dict[str, Any]:
    """CONCEPT:EG-KG.compute.handled-outside-single-anchor — "what connects A to B": the shortest path between two
    :Code symbols, rendered hop-by-hop with the relation + confidence of each edge.

    Resolves both endpoints, runs the engine's native path search (BFS over the
    resolved graph; tries A→B then B→A so an undirected connection is found), and
    annotates each consecutive pair with the connecting edge. This is the durable,
    KG-native equivalent of Graphify's ``path`` command.
    """
    src = _resolve_symbol_id(engine, symbol=symbol, node_id=node_id)
    dst = _resolve_symbol_id(engine, symbol=target_symbol, node_id=target_node_id)
    if not src:
        return {"error": f"could not resolve source symbol '{symbol or node_id}'"}
    if not dst:
        return {
            "error": f"could not resolve target symbol '{target_symbol or target_node_id}'"
        }
    src_id, dst_id = src.get("id"), dst.get("id")
    if src_id == dst_id:
        return {"error": "source and target resolve to the same symbol", "id": src_id}

    path = engine.get_shortest_path(src_id, dst_id) or engine.get_shortest_path(
        dst_id, src_id
    )
    if not path:
        return {
            "source": src_id,
            "target": dst_id,
            "connected": False,
            "path": [],
        }

    hops = _annotate_path_hops(engine, path)

    return {
        "source": src_id,
        "target": dst_id,
        "connected": True,
        "length": len(path) - 1,
        "path": path,
        "hops": hops,
    }


def _annotate_path_hops(engine: Any, path: list[Any]) -> list[dict[str, Any]]:
    """Annotate each hop with the connecting edge (undirected match for the
    relation); best-effort per hop."""
    hops: list[dict[str, Any]] = []
    for a, b in zip(path, path[1:], strict=False):
        rel, conf = None, None
        try:
            erows = engine.query_cypher(
                "MATCH (x {id: $a})-[r]-(y {id: $b}) "
                "RETURN type(r) AS rel, r.confidence AS confidence LIMIT 1",
                {"a": a, "b": b},
            )
            if erows:
                rel = erows[0].get("rel")
                conf = erows[0].get("confidence")
        except Exception:  # noqa: BLE001 — annotation is best-effort
            pass
        hops.append({"from": a, "to": b, "rel": rel, "confidence": conf})
    return hops


def _run_graph_query_sql(cypher: str, connection: str, graph: str) -> str:
    """``scope=='sql'`` branch of ``_run_graph_query`` (CONCEPT:AU-KG.query.read-only-sql-over).

    Statement-shape gate (WD10-A-BACKEND security review) lives in this thin
    wrapper, in FRONT of the original implementation
    (:func:`_run_graph_query_sql_engine`, unchanged) — kept as a separate
    function rather than added inline so this pre-existing name's own
    complexity does not regress (`verify_both.py`'s per-function, no-
    pre-existing-function-may-get-worse rule): the gate's rejection routes
    through :func:`public_error_json` on a synthetic ``ValueError`` to match
    the EXACT SAME dict-shaped, message-redacted envelope every other
    rejection this branch returns already uses (required for
    ``EvidenceBundle.from_payload`` — this function's sole caller, via
    ``graph_query`` — to stay valid, and for the message-redaction contract
    ``test_sql_scope_surfaces_engine_error`` already asserts on).
    """
    rejection = _reject_unsafe_table_sql(str(cypher or ""))
    if rejection is not None:
        return public_error_json(ValueError(rejection))
    return _run_graph_query_sql_engine(cypher, connection, graph)


def _run_graph_query_sql_engine(cypher: str, connection: str, graph: str) -> str:
    try:
        entries, errors, fanout = kg_server._resolve_target_engines(connection)
        entries = kg_server.resolve_explicit_graph(entries, graph, fanout=fanout)
    except kg_server.GraphNotFoundError as e:
        return public_error_json(e, code="graph_not_found")
    except kg_server.GraphSelectionConflictError as e:
        return public_error_json(e, code="graph_selection_conflict")
    except Exception as e:
        return public_error_json(e)
    if not fanout:
        name, engine = entries[0]
        try:
            with kg_server.bound_to_graph(graph):
                rows = engine.sql(cypher)
            return json.dumps(
                {"rows": rows, "connection": name, "graph": graph},
                default=str,
            )
        except PermissionError as e:
            return public_error_json(
                e, code="permission_denied" if graph else "operation_failed"
            )
        except Exception as e:
            return public_error_json(e)
    results, fan_errors = kg_server.fanout_execute(
        entries, lambda name, engine: engine.sql(cypher)
    )
    return json.dumps(
        {
            "targets": results,
            "errors": {**errors, **fan_errors},
            "connection": connection,
            "graph": graph,
        },
        default=str,
    )


def _run_graph_query_sparql(cypher: str, connection: str, graph: str) -> str:
    """``scope=='sparql'`` branch of ``_run_graph_query`` (CONCEPT:AU-KG.ingest.mirror-inbound)."""
    try:
        entries, errors, fanout = kg_server._resolve_target_engines(connection)
        entries = kg_server.resolve_explicit_graph(entries, graph, fanout=fanout)
    except kg_server.GraphNotFoundError as e:
        return public_error_json(e, code="graph_not_found")
    except kg_server.GraphSelectionConflictError as e:
        return public_error_json(e, code="graph_selection_conflict")
    except Exception as e:
        return public_error_json(e)
    if not fanout:
        name, engine = entries[0]
        try:
            with kg_server.bound_to_graph(graph):
                rows = engine.sparql(cypher)
            return json.dumps(
                {"rows": rows, "connection": name, "graph": graph},
                default=str,
            )
        except PermissionError as e:
            return public_error_json(
                e, code="permission_denied" if graph else "operation_failed"
            )
        except Exception as e:
            return public_error_json(e)
    results, fan_errors = kg_server.fanout_execute(
        entries, lambda name, engine: engine.sparql(cypher)
    )
    return json.dumps(
        {
            "targets": results,
            "errors": {**errors, **fan_errors},
            "connection": connection,
            "graph": graph,
        },
        default=str,
    )


def _run_graph_query_federated(
    cypher: str, reference_id: str, parsed_params: dict[str, Any]
) -> str:
    """``scope=='federated'`` branch of ``_run_graph_query``."""
    engine = kg_server._get_engine()
    if not reference_id:
        return json.dumps({"error": "reference_id required for federated queries"})
    try:
        results = engine.execute_federated_query(reference_id, cypher, parsed_params)
        return json.dumps(results, default=str)
    except Exception as e:
        return public_error_json(e)


def _run_graph_query_normalize_args(
    graph: str, params: str, include_epistemic: bool
) -> tuple[str, dict[str, Any], bool]:
    """Normalize a direct call's raw ``Field``-default bindings.

    A direct call bypassing `_execute_tool` (which resolves `Field` defaults)
    binds an omitted `graph`/`include_epistemic` to its raw, truthy
    `pydantic.fields.FieldInfo` rather than the declared default — normalize
    once here so every use below sees a clean value, mirroring the SAME
    defensiveness `ConnectionRegistry.resolve_names` already applies to
    `connection`.
    """
    normalized_graph = graph if isinstance(graph, str) else ""
    parsed_params = json.loads(params) if params else {}
    include_epistemic_flag = (
        include_epistemic if isinstance(include_epistemic, bool) else False
    )
    return normalized_graph, parsed_params, include_epistemic_flag


def _run_graph_query_resolve_local(
    connection: str, graph: str
) -> tuple[list[tuple[str, Any]], dict[str, Any], bool, str | None]:
    """Resolve the local (Cypher) read target set, or a ready-to-return error body.

    Mirrors the identical three-way except clause every other ``_run_graph_query``
    scope branch uses, just returning the error JSON instead of returning it
    directly so the caller can still short-circuit.
    """
    try:
        entries, errors, fanout = kg_server._resolve_read_engines(connection)
        entries = kg_server.resolve_explicit_graph(entries, graph, fanout=fanout)
        return entries, errors, fanout, None
    except kg_server.GraphNotFoundError as e:
        return [], {}, False, public_error_json(e, code="graph_not_found")
    except kg_server.GraphSelectionConflictError as e:
        return [], {}, False, public_error_json(e, code="graph_selection_conflict")
    except Exception as e:
        return [], {}, False, public_error_json(e)


def _run_graph_query_is_union_read(fanout: bool, connection: str | None) -> bool:
    """Whether this fan-out is the implicit content-graph UNION (no explicit connection)."""
    if not fanout:
        return False
    if connection is None:
        return True
    return isinstance(connection, str) and connection.strip().lower() in (
        "",
        "default",
    )


def _run_graph_query_union_aggregate(
    cypher: str,
    parsed_params: dict[str, Any],
    as_of: str,
    include_epistemic_flag: bool,
    graph: str,
) -> str:
    """CONCEPT:AU-KG.query.query-aggregation — an aggregation under the implicit content-graph
    union runs against the canonical default graph only (see the caller for why).
    """
    engine = kg_server._get_engine()
    try:
        if include_epistemic_flag:
            results = engine.query_cypher(
                cypher, parsed_params, as_of=as_of or None, include_epistemic=True
            )
        else:
            results = engine.query_cypher(cypher, parsed_params, as_of=as_of or None)
        return json.dumps(
            {"rows": results, "connection": "default", "graph": graph},
            default=str,
        )
    except Exception as e:
        return public_error_json(e)


def _run_graph_query_single(
    cypher: str,
    parsed_params: dict[str, Any],
    as_of: str,
    include_epistemic: bool,
    graph: str,
    name: str,
    engine: Any,
) -> str:
    """Single connection (default or one named) local Cypher read."""
    # Same raw-call defensive normalization as `envelope`: a direct call
    # bypassing FastMCP schema resolution binds an omitted bool Field to its
    # FieldInfo, not the `False` default.
    include_epistemic_flag = (
        include_epistemic if isinstance(include_epistemic, bool) else False
    )
    try:
        with kg_server.bound_to_graph(graph):
            if include_epistemic_flag:
                # Only pass the new kwarg when actually requested — keeps the
                # default call shape byte-identical for any `query_cypher`
                # implementation (real or test double) that predates this
                # parameter and doesn't accept it.
                results = engine.query_cypher(
                    cypher,
                    parsed_params,
                    as_of=as_of or None,
                    include_epistemic=True,
                )
                # Per-row epistemic envelope takes precedence over `envelope`
                # (there is no aggregate-bundle-of-epistemic-rows shape).
                return json.dumps(
                    {
                        "rows": results,
                        "connection": name,
                        "graph": graph,
                    },
                    default=_json_default,
                )
            results = engine.query_cypher(
                cypher,
                parsed_params,
                as_of=as_of or None,
                include_epistemic=include_epistemic_flag,
            )
        if include_epistemic_flag:
            return json.dumps(
                {"rows": results, "connection": name, "graph": graph},
                default=_json_default,
            )
        return json.dumps(
            {
                "rows": results,
                "connection": name,
                "graph": graph,
                "evidence_bundle": _evidence_bundle_for_rows(
                    engine, results
                ).model_dump(),
            },
            default=str,
        )
    except kg_server.GraphSelectionConflictError as e:
        return public_error_json(e, code="graph_selection_conflict")
    except PermissionError as e:
        # Only an EXPLICIT `graph` request classifies its own denial as
        # `permission_denied` — the pre-existing behavior for a plain
        # backend-level PermissionError (e.g. a write rejected on the
        # read-only query path) with no `graph` selection in play is
        # unchanged (`operation_failed`, BUG-048's ambiguous default).
        return public_error_json(
            e, code="permission_denied" if graph else "operation_failed"
        )
    except Exception as e:
        return public_error_json(e)


def _run_graph_query_fanout_primary(
    primary: list[tuple[str, Any]],
    fanout_query: Any,
    content_graph_query: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    results: dict[str, Any] = {}
    fan_errors: dict[str, Any] = {}
    for name, engine in primary:
        # Same partial-success contract `fanout_execute` gives every
        # target below (and the ACL/tenant denial path relies on): a
        # backend that legitimately REJECTS this query (e.g. a
        # permission/ACL denial from `engine.query_cypher`'s own
        # enforcement) must land in `fan_errors`, not crash the whole
        # tool call — the primary is queried directly (not through
        # `fanout_execute`) only to skip its concurrency/timeout
        # machinery, not its error handling.
        try:
            # B-18: the robustness fallback above can promote a real
            # content graph into `primary` when no "default" entry was
            # resolved — bind it too, never just the literal-"default"
            # entry (which needs no narrowing since it already targets
            # the ambient session's own graph).
            fn = fanout_query if name == "default" else content_graph_query
            results[name] = fn(name, engine)
        except Exception as exc:  # noqa: BLE001 — mirrors fanout_execute's per-target catch below (same non-leaking, BUG-048-classified label): a denied/failed primary must surface as a labeled error, never propagate raw or silently vanish
            logger.warning(
                "Graph fan-out primary target failed (exception_type=%s)",
                type(exc).__name__,
            )
            fan_errors[name] = kg_server.fanout_error_label(exc)
    return results, fan_errors


def _run_graph_query_fanout_targets(
    entries: list[tuple[str, Any]],
    union_read: bool,
    fanout_query: Any,
    content_graph_query: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not union_read:
        # Explicit cross-repo fan-out (`target='all'`/list) — per-target
        # timeout at the full budget so one slow backend can't stall the set.
        return kg_server.fanout_execute(entries, fanout_query)

    # CONCEPT:AU-KG.ingest.unified-query-routing — mirror `graph_search`'s
    # primary/supplementary split (see the `graph_search` implementation
    # above). Before this, an implicit-default `graph_query` shared ONE
    # `DEFAULT_FANOUT_TIMEOUT_S` (30s) budget across every resolved
    # content-graph entry, including the `default` connection. Under a
    # wide implicit fan-out (dozens of idle/unreachable `code:*`/`src:*`
    # graphs) the primary was queued behind the hung ones on the same
    # 8-worker pool and timed out too — a plain Cypher read against the
    # live default graph could take minutes even though the engine
    # itself (`engine_query action=cypher`, which bypasses this router)
    # answers instantly. The primary/`default` connection now always
    # grounds at the full budget, run separately and first; only the
    # SUPPLEMENTARY content connections take the short
    # `DEFAULT_CONTENT_FANOUT_TIMEOUT_S` skip-budget. An explicit
    # `target='all'`/list is a deliberate cross-repo request and is
    # NEVER `_union_read` (see its definition above), so it is
    # unaffected and keeps the full per-target budget below.
    primary = [(n, e) for n, e in entries if n == "default"]
    supplementary = [(n, e) for n, e in entries if n != "default"]
    # Robustness: if the resolver produced no `default` entry, treat the
    # first as primary so SOMETHING always grounds at the full budget.
    if not primary and entries:
        primary, supplementary = [entries[0]], entries[1:]
    results, fan_errors = _run_graph_query_fanout_primary(
        primary, fanout_query, content_graph_query
    )
    if supplementary:
        sup_results, sup_errors = kg_server.fanout_execute(
            supplementary,
            content_graph_query,
            timeout=kg_server.DEFAULT_CONTENT_FANOUT_TIMEOUT_S,
        )
        results.update(sup_results)
        fan_errors.update(sup_errors)
    return results, fan_errors


def _run_graph_query_fanout_response(
    results: dict[str, Any],
    fan_errors: dict[str, Any],
    errors: dict[str, Any],
    connection: str,
    graph: str,
    union_read: bool,
) -> str:
    if union_read:
        merged = _merge_fanout_rows(results)
        return json.dumps(
            {"rows": merged, "connection": connection, "graph": graph},
            default=_json_default,
        )
    return json.dumps(
        {
            "targets": results,
            "errors": {**errors, **fan_errors},
            "connection": connection,
            "graph": graph,
        },
        default=_json_default,
    )


def _merge_fanout_rows(results: dict[str, Any]) -> list[Any]:
    """Merge the per-graph row lists into one id-deduped canonical row set."""
    merged: list[Any] = []
    seen_ids: set[str] = set()
    for _name in results:
        for row in results[_name] or []:
            rid = row.get("id") if isinstance(row, dict) else None
            if rid is not None:
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
            merged.append(row)
    return merged


def _run_graph_query_fanout(
    cypher: str,
    parsed_params: dict[str, Any],
    as_of: str,
    include_epistemic_flag: bool,
    connection: str,
    graph: str,
    entries: list[tuple[str, Any]],
    errors: dict[str, Any],
    union_read: bool,
) -> str:
    """Fan-out — per-target timeout so one slow backend can't stall the set."""

    def _fanout_query(name: str, engine: Any) -> Any:
        del name
        # Same defensive kwarg omission as the single-connection branch
        # above: only pass `include_epistemic` when actually requested,
        # so a `query_cypher` implementation (real or test double) that
        # predates this parameter and doesn't accept it keeps working.
        if include_epistemic_flag:
            return engine.query_cypher(
                cypher, parsed_params, as_of=as_of or None, include_epistemic=True
            )
        return engine.query_cypher(cypher, parsed_params, as_of=as_of or None)

    def _content_graph_query(name: str, engine: Any) -> Any:
        # B-18 fix: within the implicit content-graph UNION fan-out (never
        # the explicit multi-connection fan-out below, which never calls
        # this), `name` IS the physical graph `engine` is scoped to (see
        # `kg_server._resolve_read_engines`'s
        # `ingest_routing.safe_engine_for_graph(gname)` — a `for_graph()`
        # view that fixes the outbound wire request's `graph` field but
        # never rebinds `session.graph` to match). The wire layer's own
        # mismatch lock (`_SessionRoutedAsyncClient._send`) then rejects
        # every non-"default" leg with `PermissionError`, degrading the
        # union to per-target errors. Narrow the verified session to the
        # SAME graph via the sanctioned `bound_to_graph` primitive
        # (CONCEPT:AU-KG.backend.explicit-graph-selection) so the request
        # is self-consistent instead of inventing a second mechanism.
        with kg_server.bound_to_graph(name):
            return _fanout_query(name, engine)

    results, fan_errors = _run_graph_query_fanout_targets(
        entries, union_read, _fanout_query, _content_graph_query
    )
    return _run_graph_query_fanout_response(
        results, fan_errors, errors, connection, graph, union_read
    )


_GRAPH_SEARCH_RESULTS_MODES = frozenset(
    {
        "hyde",
        "deep",
        "hybrid",
        "concept",
        "analogy",
        "adore",
        "chrono_ids",
        "dci",
        "memory",
        "latent",
        "sira",
        "rerank",
    }
)


def _graph_search_rerank(engine: Any, session: Any, *, query: str, top_k: int) -> Any:
    # Semantic-retrieval hybrid re-scoring of a candidate set.
    from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (  # noqa: E501
        HybridSearchScorer,
    )

    base = engine.search_hybrid(query=query, top_k=top_k, session=session) or []
    docs = [
        {
            "id": (r.get("node", r) or {}).get("id", ""),
            "text": (r.get("node", r) or {}).get("description", ""),
            "embedding": (r.get("node", r) or {}).get("embedding"),
        }
        for r in base
    ]
    qemb: list[float] = []
    embed_model = getattr(
        getattr(engine, "hybrid_retriever", None), "embed_model", None
    )
    if embed_model is not None:
        try:
            qemb = embed_model.get_text_embedding(query)
        except Exception:  # noqa: BLE001
            qemb = []
    return HybridSearchScorer().score_documents(query, qemb, docs)


def _graph_search_produce_results_extended(
    engine: Any, session: Any, *, mode: str, query: str, top_k: int
) -> Any:
    if mode == "adore":
        return engine.search_adore(query=query, top_k=top_k)
    if mode == "chrono_ids":
        return engine.temporal_semantic_ids(query=query, top_k=top_k)
    if mode == "dci":
        # search_dci is fail-closed (CONCEPT:AU-KG.retrieval.acl-aware-vector-retrieval):
        # it always resolves a session and raises rather than returning an
        # unfiltered traversal, so this served path must pass the already-ambient
        # verified session (same pattern the hybrid/hyde/deep/concept/analogy
        # branches use) rather than relying on an implicit fallback.
        return engine.search_dci(query=query, top_k=top_k, session=session)
    if mode == "memory":
        return engine.search_memories(query=query, top_k=top_k)
    if mode == "latent":
        # KG-2.3 — route through the latent topology hierarchy.
        from agent_utilities.knowledge_graph.retrieval.latent_topology_rag import (  # noqa: E501
            LatentTopologicalRAG,
        )

        return LatentTopologicalRAG(engine).retrieve(query, top_k=top_k)
    if mode == "sira":
        # Single-shot SIRA: hybrid-retrieve, then sparsity-align the set.
        from agent_utilities.knowledge_graph.retrieval.single_shot_sira import (
            SingleShotSIRA,
        )

        base = engine.search_hybrid(query=query, top_k=top_k, session=session) or []
        return SingleShotSIRA(engine).align_context(base)
    # Only "rerank" remains among `_GRAPH_SEARCH_RESULTS_MODES` at this point —
    # the caller has already dispatched hyde/deep/hybrid/concept/analogy above,
    # and every other results-producing mode just above.
    return _graph_search_rerank(engine, session, query=query, top_k=top_k)


def _graph_search_produce_results(
    engine: Any,
    session: Any,
    *,
    mode: str,
    query: str,
    top_k: int,
    self_correct: bool,
    as_of: str,
) -> Any:
    """Modes in `_GRAPH_SEARCH_RESULTS_MODES` that feed the shared formatter
    in `_search_with_engine` below. Precondition: `mode` is a member."""
    if mode in ("hyde", "deep"):
        return engine.search_hybrid(
            query=query,
            top_k=top_k,
            mode=mode,
            self_correct=self_correct,
            session=session,
        )
    if mode == "hybrid":
        return engine.search_hybrid(
            query=query,
            top_k=top_k,
            self_correct=self_correct,
            as_of=as_of or None,
            session=session,
        )
    if mode in ("concept", "analogy"):
        return engine.search_hybrid(query=query, top_k=top_k, session=session)
    return _graph_search_produce_results_extended(
        engine, session, mode=mode, query=query, top_k=top_k
    )


def _graph_search_discover(engine: Any, query: str) -> str:
    try:
        from agent_utilities.capabilities.manager import CapabilityManager

        manager = CapabilityManager(engine)
        results = manager.discover_capabilities(query)
        if not results:
            return f"No capabilities found for '{query}'"
        return "\n".join([f"- {r.name}: {r.description}" for r in results])
    except ImportError:
        return "Error: capabilities module not available"


def _graph_search_hard_negatives(engine: Any, query: str) -> str:
    # KG-2.3 — mine hard negatives via the engine's hybrid retriever.
    from agent_utilities.knowledge_graph.retrieval.hard_negative_miner import (  # noqa: E501
        HardNegativeMiner,
    )

    retriever = getattr(engine, "hybrid_retriever", None)
    if retriever is None:
        return "Error: hybrid retriever unavailable for hard-negative mining."
    negs = HardNegativeMiner(retriever).mine(query)
    if not negs:
        return f"No hard negatives mined for: '{query}'"
    return "\n".join(f"- {n.doc_id}: {getattr(n, 'reason', '')}" for n in negs)


def _graph_search_compiled(
    engine: Any, query: str, *, top_k: int, as_of: str, token_budget: int
) -> str:
    # CONCEPT:AU-KG.retrieval.context-compiler — policy-aware ContextCompiler bundle: reuses the
    # SAME engine ANN/hybrid retriever the other modes call, but additionally
    # MMR-diversifies, scores evidence-quality/freshness from the epistemic
    # columns (EPI-P3-1), fits a token budget, and runs every candidate through
    # the live permissioning gate before returning citations + a proof graph —
    # replacing the plain relevance-sorted concat the shared formatter performs
    # for every other mode.
    from agent_utilities.knowledge_graph.core.session import GraphSession
    from agent_utilities.knowledge_graph.retrieval.context_compiler import (  # noqa: E501
        ContextCompiler,
    )

    session = GraphSession.from_ambient()
    compiler = ContextCompiler(engine)
    kwargs: dict[str, Any] = {"top_k": top_k, "as_of": as_of or None}
    if token_budget:
        kwargs["token_budget"] = token_budget
    bundle = compiler.compile(query, session=session, **kwargs)
    return bundle.as_text()


def _graph_search_format_results(results: Any) -> str:
    # The direct retrieval modes use a flat, score-sorted text block with no
    # diversity/evidence/freshness/policy/budget shaping — prefer
    # mode='compiled' for a citation- and proof-graph-bearing bundle.
    formatted_results = []
    for res in results:
        score = res.get("score", 0)
        score = float(score) if score is not None else 0.0
        node = res.get("node", res)
        label = node.get("type", node.get("label", "Unknown"))
        name = node.get("name", "Unnamed")
        desc = node.get("description", "")
        nid = node.get("id", "N/A")
        formatted_results.append(
            f"[{label}] {name} (ID: {nid}) - Score: {score:.2f}\n{desc}"
        )
    return "\n---\n".join(formatted_results)


def _graph_search_single_target(
    entries: list[tuple[str, Any]],
    graph: str,
    run_search: Callable[[Any], str],
) -> str:
    """Single-connection ``graph_search`` path: bind the graph and run once."""
    name, engine = entries[0]
    try:
        with kg_server.bound_to_graph(graph):
            text = run_search(engine)
    except PermissionError as e:
        return public_error_text(
            e, code="permission_denied" if graph else "operation_failed"
        )
    return f"{text}\n\n[connection={name} graph={graph or '(default)'}]"


def _graph_search_implicit_fanout(
    entries: list[tuple[str, Any]],
    run_search: Callable[[Any], str],
    content_graph_search: Callable[[str, Any], str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """CONCEPT:AU-KG.ingest.unified-query-routing — an implicit-default connection
    (no ``connection`` passed, or explicitly "default") fans across every active
    content graph, which can be dozens of ``code:<repo>``/``src:<repo>``
    connections, often idle/unreachable.

    The PRIMARY/``default`` backend must ALWAYS ground: it holds the primary
    ``__commons__`` + control-plane content and is the source of the real
    ranked hits. So run it separately at the normal budget (never the
    skip-timeout) and apply the SHORT skip-timeout ONLY to the supplementary
    content backends. Under a wide implicit fan-out (~70 ``code:*`` graphs) a
    single shared short wall-clock across ALL targets starved the primary —
    it was queued behind the hung code backends and timed out too, so the
    search returned zero results even though the engine was healthy.
    Splitting the primary out fixes that: a search still returns the
    primary's real hits in a few seconds when every supplementary backend is
    dead.
    """
    primary = [(n, e) for n, e in entries if n == "default"]
    supplementary = [(n, e) for n, e in entries if n != "default"]
    # Robustness: if the resolver produced no ``default`` entry, treat the
    # first as primary so SOMETHING always grounds at the full budget.
    if not primary and entries:
        primary, supplementary = [entries[0]], entries[1:]
    results: dict[str, Any] = {}
    fan_errors: dict[str, str] = {}
    for name, engine in primary:
        # B-18: bind for a real content graph promoted into `primary` by
        # the robustness fallback above too — only the literal-"default"
        # entry already targets the ambient session's own graph and needs
        # no narrowing.
        if name == "default":
            results[name] = run_search(engine)
        else:
            results[name] = content_graph_search(name, engine)
    if supplementary:
        sup_results, sup_errors = kg_server.fanout_execute(
            supplementary,
            content_graph_search,
            timeout=kg_server.DEFAULT_CONTENT_FANOUT_TIMEOUT_S,
        )
        results.update(sup_results)
        fan_errors.update(sup_errors)
    return results, fan_errors


def _graph_search_fanout(
    entries: list[tuple[str, Any]],
    errors: dict[str, str],
    connection: str,
    run_search: Callable[[Any], str],
    content_graph_search: Callable[[str, Any], str],
) -> str:
    """Multi-connection ``graph_search`` path: implicit content-graph union
    or an explicit cross-repo fan-out, formatted into one labeled block."""
    is_implicit_target = connection is None or (
        isinstance(connection, str) and connection.strip().lower() in ("", "default")
    )
    if is_implicit_target:
        results, fan_errors = _graph_search_implicit_fanout(
            entries, run_search, content_graph_search
        )
    else:
        # Explicit cross-repo fan-out — per-target timeout at the full budget so
        # one slow backend can't stall the set.
        results, fan_errors = kg_server.fanout_execute(
            entries, lambda name, engine: run_search(engine)
        )
    out_lines = [f"=== {name} ===\n{results[name]}" for name in results]
    out_lines += [
        f"=== {name} (error) ===\n{err}"
        for name, err in {**errors, **fan_errors}.items()
    ]
    out_lines.append(f"[connection={connection or 'default'} graph=(none — fan-out)]")
    return "\n\n".join(out_lines)


# ══════════════════════════════════════════════════════════════════
# WD10-A-BACKEND security review (plans/semantic-indexing/DESIGN-embedding-
# bindings.md §4): defense-in-depth statement-shape gate in FRONT of every
# raw-SQL call site this file owns (`graph_table` action='query' and
# `graph_query` scope='sql'). `QueryMixin.sql()`
# (agent_utilities/knowledge_graph/orchestration/engine_query.py — owned by a
# sibling lane this wave, NOT this file) already rejects anything whose first
# 8 characters are not SELECT/WITH/EXPLAIN, but that is a HEAD-TOKEN check
# only: it does not parse the statement. Probed empirically (see the
# WD10-A-BACKEND lane report) — all of the following reach that guard's
# ADMIT branch unexamined:
#   * a writable CTE: "WITH x AS (INSERT ... RETURNING id) SELECT * FROM x"
#   * a ';'-stacked second statement: "SELECT 1; DROP TABLE nodes;"
#   * "EXPLAIN ANALYZE <mutation>" (ANALYZE mode EXECUTES the wrapped
#     statement in every SQL engine that implements it)
# This gate is additive and strictly narrower than the engine-side one — it
# never admits anything the engine-side guard would reject, only rejects
# more. It does NOT and cannot defend against a side-effecting/volatile
# function called from an otherwise-legal SELECT list (e.g.
# "SELECT pg_read_file(...)"-class builtins) — that requires a function
# ALLOWLIST enforced by the engine's own SQL executor (eg repo, out of this
# lane's ownership this wave); filed as a finding in the lane report rather
# than reached into.
_TABLE_QUERY_HEAD_RE = re.compile(r"^\s*(SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)
_TABLE_QUERY_EXPLAIN_ANALYZE_RE = re.compile(r"^\s*EXPLAIN\s+ANALYZE\b", re.IGNORECASE)
_TABLE_QUERY_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TABLE_QUERY_WRITE_TOKENS = frozenset(
    {
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create",
        "grant",
        "revoke",
        "copy",
        "merge",
        "call",
        "exec",
        "execute",
        "vacuum",
        "reindex",
        "attach",
        "detach",
        "pragma",
        "analyze",
    }
)


def _table_query_statements(sql: str) -> list[str]:
    """Split on top-level ``;`` — a stacked second statement is the classic
    defeat of a head-token-only guard (a trailing ``;`` alone is harmless)."""
    return [part.strip() for part in sql.split(";") if part.strip()]


def _reject_unsafe_table_sql(sql: str) -> str | None:
    """Return an error string when ``sql`` fails this gate's stricter shape
    check, else ``None``. See the module comment above for what this closes
    and what it deliberately does NOT (function-call side effects)."""
    statements = _table_query_statements(sql)
    if len(statements) != 1:
        return "query must be a single statement (no ';'-stacked statements)"
    statement = statements[0]
    if not _TABLE_QUERY_HEAD_RE.match(statement):
        return "query must start with SELECT, WITH, or EXPLAIN"
    if _TABLE_QUERY_EXPLAIN_ANALYZE_RE.match(statement):
        return "EXPLAIN ANALYZE executes the wrapped statement; use plain EXPLAIN"
    tokens = {t.lower() for t in _TABLE_QUERY_TOKEN_RE.findall(statement)}
    banned = sorted(tokens & _TABLE_QUERY_WRITE_TOKENS)
    if banned:
        return f"query contains a disallowed keyword: {banned[0]}"
    return None


def _graph_table_ingest(
    engine: Any,
    table_ingest: Any,
    source: str,
    table: str,
    config_json: str,
    limit: int,
    replace: bool,
) -> str:
    if not source:
        return json.dumps({"error": "ingest needs a source connector"})
    return json.dumps(
        table_ingest.ingest_connector_to_table(
            engine,
            str(source),
            table=table or None,
            config=json.loads(config_json) if config_json else None,
            limit=int(limit),
            replace=bool(replace),
        ),
        default=str,
    )


def _graph_table_rows(
    engine: Any, table_ingest: Any, table: str, rows_json: str, replace: bool
) -> str:
    if not table:
        return json.dumps({"error": "rows needs a table"})
    return json.dumps(
        table_ingest.ingest_rows_to_table(
            engine,
            str(table),
            json.loads(rows_json) if rows_json else [],
            replace=bool(replace),
        ),
        default=str,
    )


def _graph_table_create(
    engine: Any, table_ingest: Any, table: str, columns_json: str
) -> str:
    if not table:
        return json.dumps({"error": "create needs a table"})
    cols = json.loads(columns_json) if columns_json else []
    if not cols:
        return json.dumps({"error": "create needs columns_json"})
    return json.dumps(table_ingest.ensure_table(engine, str(table), cols), default=str)


def _graph_table_drop(engine: Any, table_ingest: Any, table: str) -> str:
    if not table:
        return json.dumps({"error": "drop needs a table"})
    return json.dumps(table_ingest.drop_table(engine, str(table)), default=str)


def _graph_table_query(engine: Any, sql: str) -> str:
    if not sql:
        return json.dumps({"error": "query needs a sql SELECT"})
    return _graph_table_query_checked(engine, sql)


def _graph_table_query_checked(engine: Any, sql: str) -> str:
    """The statement-shape-gated execute step (WD10-A-BACKEND security
    review) — split from :func:`_graph_table_query` so that pre-existing
    name's own complexity does not regress (`verify_both.py`)."""
    rejection = _reject_unsafe_table_sql(str(sql))
    if rejection is not None:
        return json.dumps({"error": rejection})
    return json.dumps(engine.sql(str(sql)), default=str)


# ══════════════════════════════════════════════════════════════════
# WD10-A-BACKEND — node-link JSON projection (wD10 Atlas backend gap B).
#
# No first-class {nodes, links} serialization exists today for a graph query
# RESULT (as opposed to `GraphComputeEngine.to_json()`'s whole-graph dump,
# agent_utilities/knowledge_graph/core/graph_compute.py, which uses
# {"source","target","properties"} for the FULL graph, not a query
# projection). Every dialect (Cypher/SQL/SPARQL) can return dict-valued
# columns shaped like a node ({id, ...}) or a relationship
# ({source/start/..., target/end/..., type, ...}) depending on backend and
# RETURN clause; this projector is deliberately generic across that
# variance and NEVER fabricates a node/edge from a value it can't identify
# — a scalar/flat column contributes nothing (see class docstring).
# ══════════════════════════════════════════════════════════════════

_NODE_LINK_SOURCE_KEYS: tuple[str, ...] = ("source", "start", "_start", "from")
_NODE_LINK_TARGET_KEYS: tuple[str, ...] = ("target", "end", "_end", "to")
_NODE_LINK_ID_KEYS: tuple[str, ...] = ("id", "node_id", "_id")
_NODE_LINK_RESERVED_EDGE_KEYS: frozenset[str] = frozenset(
    _NODE_LINK_SOURCE_KEYS
    + _NODE_LINK_TARGET_KEYS
    + ("type", "label", "properties", "id")
)
_NODE_LINK_RESERVED_NODE_KEYS: frozenset[str] = frozenset(
    _NODE_LINK_ID_KEYS + ("label", "type", "properties")
)


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _node_link_endpoint_id(value: Any) -> Any:
    """A relationship endpoint may itself be a nested node dict or a bare id."""
    if isinstance(value, dict):
        return _first_present(value, _NODE_LINK_ID_KEYS)
    return value


def _node_link_edge_endpoints(value: dict[str, Any]) -> tuple[Any, Any] | None:
    source = _first_present(value, _NODE_LINK_SOURCE_KEYS)
    target = _first_present(value, _NODE_LINK_TARGET_KEYS)
    if source is None or target is None:
        return None
    source_id = _node_link_endpoint_id(source)
    target_id = _node_link_endpoint_id(target)
    if source_id is None or target_id is None:
        return None
    return source_id, target_id


def _node_link_edge_properties(value: dict[str, Any]) -> dict[str, Any]:
    properties = value.get("properties")
    if isinstance(properties, dict):
        return properties
    return {k: v for k, v in value.items() if k not in _NODE_LINK_RESERVED_EDGE_KEYS}


def _as_node_link_edge(value: dict[str, Any]) -> dict[str, Any] | None:
    endpoints = _node_link_edge_endpoints(value)
    if endpoints is None:
        return None
    source_id, target_id = endpoints
    return {
        "source": source_id,
        "target": target_id,
        "label": value.get("type") or value.get("label") or "",
        "properties": _node_link_edge_properties(value),
    }


def _as_node_link_node(value: dict[str, Any]) -> dict[str, Any] | None:
    node_id = _first_present(value, _NODE_LINK_ID_KEYS)
    if node_id is None:
        return None
    properties = value.get("properties")
    if not isinstance(properties, dict):
        properties = {
            k: v for k, v in value.items() if k not in _NODE_LINK_RESERVED_NODE_KEYS
        }
    return {
        "id": node_id,
        "label": value.get("label") or value.get("type") or "",
        "properties": properties,
    }


def _absorb_node_link_value(
    value: Any, nodes: dict[Any, dict[str, Any]], links: list[dict[str, Any]]
) -> None:
    if not isinstance(value, dict):
        return
    edge = _as_node_link_edge(value)
    if edge is not None:
        links.append(edge)
        return
    node = _as_node_link_node(value)
    if node is not None:
        nodes.setdefault(node["id"], node)


def _parse_projection_request_object(request_json: str) -> dict[str, Any] | str:
    """Parse+validate ``request_json``'s outer JSON shape; the parsed dict, or
    an error string."""
    try:
        request = json.loads(request_json) if request_json else {}
    except (TypeError, ValueError):
        return "request_json must be a JSON object"
    if not isinstance(request, dict):
        return "request_json must be a JSON object"
    return request


_PROJECTION_REQUEST_STRING_FIELDS: tuple[str, ...] = (
    "scope",
    "reference_id",
    "as_of",
    "connection",
    "graph",
)


def _projection_request_params(request: dict[str, Any]) -> str:
    params = request.get("params")
    return params if isinstance(params, str) else json.dumps(params or {})


def _projection_request_kwargs(request: dict[str, Any], cypher: str) -> dict[str, str]:
    kwargs = {
        name: str(request.get(name) or "") for name in _PROJECTION_REQUEST_STRING_FIELDS
    }
    kwargs["scope"] = kwargs["scope"] or "local"
    kwargs["cypher"] = cypher
    kwargs["params"] = _projection_request_params(request)
    return kwargs


def _parse_projection_request(request_json: str) -> tuple[dict[str, str], str | None]:
    """Parse ``graph_projection``'s single JSON request object into
    ``_run_graph_query`` kwargs, or return ``(_, error_message)``.

    Collapses what would otherwise be 7 individual ``Field`` parameters
    (mirroring ``graph_query``'s own signature) into ONE typed request
    object — this file's `graph_table` already uses the identical
    "bundle a multi-field payload into one JSON-string param" idiom
    (``config_json``/``columns_json``/``rows_json``), so this follows the
    SAME established convention rather than introducing a new one.
    """
    request = _parse_projection_request_object(request_json)
    if isinstance(request, str):
        return {}, request
    cypher = str(request.get("cypher") or "")
    if not cypher:
        return {}, "request_json.cypher is required"
    return _projection_request_kwargs(request, cypher), None


def _projection_rows(payload: Any) -> list[Any]:
    """Extract the raw row list from a ``_run_graph_query`` JSON payload.

    Deliberately bypasses ``EvidenceBundle.from_payload`` here: that class's
    dict branch (``_from_embedded_bundle``) prioritizes an EMBEDDED
    ``evidence_bundle`` when present — and ``_run_graph_query_single`` always
    embeds one (``_evidence_bundle_for_rows``), whose ``claims`` are the
    epistemic CURRENCY-UPGRADE summary (id/text/kind only, and EMPTY whenever
    the connected engine has no ``explain_provenance_by_ids`` primitive —
    verified empirically, see the WD10-A-BACKEND lane report), not the raw
    per-row node/edge properties this projection needs. This mirrors
    ``EvidenceBundle._dict_payload_claims``'s own rows/results extraction
    instead, plus the fan-out ``targets`` shape ``_run_graph_query_fanout``/
    ``_run_graph_query_sql`` use.
    """
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    rows = payload.get("rows")
    if isinstance(rows, list):
        return rows
    results = payload.get("results")
    if isinstance(results, list):
        return results
    targets = payload.get("targets")
    if not isinstance(targets, dict):
        return []
    merged: list[Any] = []
    for value in targets.values():
        if isinstance(value, list):
            merged.extend(value)
    return merged


def _project_node_link(rows: list[Any]) -> dict[str, Any]:
    """Project arbitrary graph-query result rows into a stable
    ``{nodes:[{id,label,properties}], links:[{source,target,label,properties}]}``
    shape — the D3/force-graph convention every 2D/3D renderer already speaks,
    so a frontend adapter never has to re-derive node/edge shape from raw
    per-dialect rows itself.
    """
    nodes: dict[Any, dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for value in row.values():
            _absorb_node_link_value(value, nodes, links)
    return {"nodes": list(nodes.values()), "links": links}


# ══════════════════════════════════════════════════════════════════
# WD10-A-BACKEND — unified capability/catalog introspection (wD10 Atlas
# backend gap A). Schema discovery is fragmented per protocol today (SQL via
# information_schema, GraphQL via live __schema, SPARQL via
# `SELECT DISTINCT ?type`, ontology via Method::OwlReason, KV only via
# `scan`, vector indexes with no listing method at all) — a frontend had to
# make 4-6 protocol-specific calls and guess. This assembles ONE
# capability-shaped response: every modality entry says whether it is
# available, and an ABSENT backend (a modality compiled out of this engine
# build) yields {"available": false, "reason": ...} rather than an error —
# matching the agent-webui `src/lib/gateway.ts` {ok, data, unavailable,
# error} envelope convention one layer up.
#
# Capability detection reuses `engine_tools.ENGINE_DOMAINS`
# (agent_utilities/mcp/tools/engine_tools.py, unowned this wave) — the SAME
# live introspection of the installed `epistemic_graph` client's sub-client
# classes every `engine_<domain>` tool is generated from (see that module's
# `_discover_domains()` docstring) — rather than re-deriving "what's
# compiled in" by hand. A domain absent from `ENGINE_DOMAINS` (its client
# class not present in the installed engine build) is reported unavailable;
# NEVER fabricated as live.
# ══════════════════════════════════════════════════════════════════


def _capability_entry(
    available: bool,
    *,
    reason: str = "",
    methods: list[str] | None = None,
    items: Any = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"available": available}
    if not available:
        entry["reason"] = reason or "not available on the connected engine build"
        return entry
    if methods is not None:
        entry["methods"] = methods
    if items is not None:
        entry["items"] = items
    return entry


def _engine_domain_methods(domain: str) -> list[str] | None:
    """The introspected method list for one ``engine_tools.ENGINE_DOMAINS``
    entry, or ``None`` when that domain's client class is absent from the
    installed engine build (never guessed — see module comment above)."""
    try:
        from agent_utilities.mcp.tools.engine_tools import ENGINE_DOMAINS
    except Exception:  # noqa: BLE001 — engine_tools itself unavailable
        return None
    return ENGINE_DOMAINS.get(domain)


def _catalog_unavailable_domain(domain: str, hint: str) -> dict[str, Any]:
    """A modality reported by capability alone (compiled in or not) — no
    listing call attempted, so ``items`` is intentionally absent rather than
    guessed at from a method NAME (e.g. picking whichever ``TimeSeriesClient``
    method sounds like "list" would risk calling a mutating/expensive method
    that merely has a plausible name)."""
    methods = _engine_domain_methods(domain)
    if not methods:
        return _capability_entry(False, reason=hint)
    return _capability_entry(True, methods=methods)


async def _catalog_graphs() -> dict[str, Any]:
    """Physical graphs — via the already-registered ``engine_tenants`` tool
    (CONCEPT reference: `engine_tenants(action='list')`, referenced by name
    in several docstrings across this codebase but never centralized behind
    one discoverable call before this). ``tenants`` is an AU-P0-6
    ADMIN_DOMAINS entry (agent_utilities/mcp/tools/engine_tools.py) — a
    non-admin caller is correctly DENIED here, reported as ``denied`` rather
    than conflated with "not compiled in"."""
    methods = _engine_domain_methods("tenants")
    if not methods:
        return _capability_entry(
            False, reason="no MultiTenantClient on the installed engine client"
        )
    listing_fn = kg_server.REGISTERED_TOOLS.get("engine_tenants")
    if listing_fn is None:
        return _capability_entry(True, methods=methods)
    try:
        # Explicit kwargs for every parameter — a direct call bypassing
        # FastMCP's Field-default resolution binds an omitted arg to its raw,
        # truthy `pydantic.fields.FieldInfo` rather than its intended default
        # (the SAME documented gotcha `engine_tools._dispatch` and
        # `query_tools._run_graph_query` both already guard against).
        raw = await listing_fn(action="list", params_json="{}", graph="")
    except PermissionError:
        entry = _capability_entry(True, methods=methods)
        entry["denied"] = True
        entry["reason"] = (
            "tenants is an ADMIN-scoped domain (AU-P0-6); caller lacks the "
            "required scope"
        )
        return entry
    except Exception as exc:  # noqa: BLE001 — best-effort listing
        entry = _capability_entry(True, methods=methods)
        entry["list_error"] = type(exc).__name__
        return entry
    payload = json.loads(raw) if isinstance(raw, str) else raw
    return _capability_entry(True, methods=methods, items=payload)


async def _catalog_sql() -> dict[str, Any]:
    """Delegate to the already-landed, no-caller-SQL ``/graph/sql-schema``
    projection (``agent_utilities/mcp/tools/graph_tools.sql_schema`` —
    confirmed live and already consumed by the sibling WD10-A-SQL lane's own
    Table Explorer) rather than re-deriving table/column introspection here.
    Richer (catalog/schema nesting, primary keys) and safer (server-authored
    ``information_schema`` constants only — see
    ``graph_tools.CATALOG_STATEMENTS`` — never a per-table caller-shaped
    query) than a bespoke per-table probe would be."""
    from agent_utilities.mcp.tools import graph_tools

    try:
        projection = await graph_tools.sql_schema()
    except Exception as exc:  # noqa: BLE001 — covers SqlSchemaUnavailable + any other failure
        return _capability_entry(
            False,
            reason=f"sql-schema introspection unavailable: {type(exc).__name__}",
        )
    return _capability_entry(True, items=projection)


async def _safe_catalog_entry(awaitable: Any) -> dict[str, Any]:
    """One modality's introspection failure never breaks the whole catalog."""
    try:
        return await awaitable
    except Exception as exc:  # noqa: BLE001 — see docstring
        return _capability_entry(
            False, reason=f"introspection failed: {type(exc).__name__}"
        )


async def _catalog_sources() -> dict[str, Any]:
    """Project the process-owned source/connection registries for Atlas.

    The provider catalogue is synchronous and side-effect free, but wrapping it
    in this async adapter keeps the existing ``graph_catalog`` fan-in shape and
    ensures one malformed optional profile degrades only the ``sources`` leg.
    """

    from agent_utilities.knowledge_graph.core.source_catalog import (
        build_source_catalog,
    )

    return build_source_catalog()


async def _build_graph_catalog() -> dict[str, Any]:
    catalog: dict[str, Any] = {
        "graphs": await _safe_catalog_entry(_catalog_graphs()),
        "sql": await _safe_catalog_entry(_catalog_sql()),
        # Atlas's governed external-source view is one more modality in this
        # catalog.  It projects graph_configure's named connection registry,
        # the source connector registry, and reference-only AgentConfig
        # declarations; it never opens a provider or exposes resolved secrets.
        "sources": await _safe_catalog_entry(_catalog_sources()),
        # KV namespaces and vector indexes have NO listing surface anywhere in
        # this codebase today (verified: no domain class in ENGINE_DOMAINS,
        # no `list_*` helper in agent_utilities) — reported honestly rather
        # than fabricated, per the WD10-A-BACKEND brief's explicit
        # instruction not to claim a compiled-out modality is live.
        "kv": _capability_entry(
            False,
            reason=(
                "no KVClient / namespace-listing surface on the engine client; "
                "graph_context provides KV-like storage via graph nodes, not a "
                "named-namespace system"
            ),
        ),
        "vector": _capability_entry(
            False,
            reason=(
                "no VectorClient / list-collections method on the engine "
                "client; embeddings are stored as node properties, not named "
                "collections (see WD10-A-BACKEND lane report)"
            ),
        ),
        "ontology": _catalog_unavailable_domain(
            "reasoning",
            "no ReasoningClient (Method::OwlReason) on the installed engine client",
        ),
        "timeseries": _catalog_unavailable_domain(
            "timeseries", "no TimeSeriesClient on the installed engine client"
        ),
        "blob": _catalog_unavailable_domain(
            "blob", "no BlobClient on the installed engine client"
        ),
        "broker": _catalog_unavailable_domain(
            "broker", "no BrokerClient on the installed engine client"
        ),
        "saved_queries": _capability_entry(
            False,
            reason=(
                "no durable saved/continuous-query store on the AU side today "
                "(only ad hoc engine-level continuous-query primitives "
                "referenced in code comments, not wired to a listing surface)"
            ),
        ),
    }
    return catalog


async def _graph_catalog_response(
    action: str,
    *,
    source: str,
    mode: str,
    ids_json: str,
    connection: str,
    graph: str,
) -> str:
    normalized_action = action if isinstance(action, str) else "list"
    normalized_action = normalized_action.strip().lower()
    if normalized_action == "preview_sync":
        try:
            from agent_utilities.knowledge_graph.core.source_catalog import (
                normalize_source_sync_preview,
            )

            return json.dumps(
                normalize_source_sync_preview(
                    source=source,
                    mode=mode,
                    ids_json=ids_json,
                    connection=connection,
                    graph=graph,
                ),
                default=str,
            )
        except Exception as exc:  # noqa: BLE001 — deterministic client error
            return public_error_json(exc, code="invalid_request")
    if normalized_action != "list":
        return public_error_json(
            ValueError("action must be 'list' or 'preview_sync'"),
            code="invalid_request",
        )
    try:
        kg_server._get_engine()
    except Exception as exc:  # noqa: BLE001
        return public_error_json(exc, code="dependency_unavailable")
    catalog = await _build_graph_catalog()
    return json.dumps(catalog, default=str)


async def _graph_context_put(
    engine: Any,
    content: str,
    context_id: str,
    session_id: str,
    key: str,
    ttl_s: int,
) -> str:
    import contextlib
    import time
    import uuid as _uuid

    if not content:
        return json.dumps({"error": "content required for put"})
    sid = session_id or _uuid.uuid4().hex
    cid = context_id or f"ctx:{sid}:{key or _uuid.uuid4().hex}"
    snode = f"session:{sid}"

    def _persist_context_blob() -> None:
        engine.add_node(
            cid,
            "ContextBlob",
            properties={
                "id": cid,
                "content": content,
                "session_id": sid,
                "key": key,
                "ttl_s": int(ttl_s),
                "created_at": time.time(),
                "producer": kg_server._SESSION_ID,
            },
        )
        # CONCEPT:AU-ORCH.session.session-anchored-collections-native — session-anchored collection: upsert the id-addressable
        # Session node and link it, so "list by session" is a reliable id-anchored
        # traversal (the engine has no property index; property scans are unreliable).
        with contextlib.suppress(Exception):
            engine.add_node(
                snode, "Session", properties={"id": snode, "session_id": sid}
            )
            engine.add_edge(snode, cid, "HAS_CONTEXT")

    await run_blocking_ordered(_persist_context_blob)
    return json.dumps({"context_id": cid, "session_id": sid})


async def _graph_context_get(engine: Any, context_id: str) -> str:
    import time

    if not context_id:
        return json.dumps({"error": "context_id required for get"})
    try:
        rows = await run_blocking_ordered(
            engine.query_cypher,
            "MATCH (c:ContextBlob) WHERE c.id = $id "
            "RETURN c.id AS id, c.content AS content, "
            "c.session_id AS session_id, "
            "c.created_at AS created_at, c.ttl_s AS ttl_s",
            {"id": context_id},
        )
        if not rows:
            return json.dumps({})
        row = rows[0]
        # TTL: treat an expired blob as gone (created_at + ttl_s < now).
        _ttl = row.get("ttl_s") or 0
        _created = row.get("created_at") or 0
        if _ttl and _created and (float(_created) + float(_ttl) < time.time()):
            return json.dumps({"error": "context expired", "expired": True})
        return json.dumps(row, default=str)
    except Exception as exc:  # noqa: BLE001
        return public_error_json(exc)


async def _graph_context_prune(engine: Any) -> str:
    import contextlib
    import time

    # Delete expired ContextBlobs (CONCEPT:AU-ORCH.session.invoker-agent-handoff lifecycle).
    try:

        def _prune_expired() -> tuple[int, int]:
            rows = engine.query_cypher(
                "MATCH (c:ContextBlob) WHERE c.ttl_s > 0 AND "
                "(c.created_at + c.ttl_s) < $now RETURN c.id AS id",
                {"now": time.time()},
            )
            count = 0
            _del = getattr(engine, "delete_node", None) or getattr(
                getattr(engine, "backend", None), "delete_node", None
            )
            for r in rows or []:
                if callable(_del):
                    with contextlib.suppress(Exception):
                        _del(r["id"])
                        count += 1
            return count, len(rows or [])

        pruned, expired = await run_blocking_ordered(_prune_expired)
        return json.dumps({"pruned": pruned, "expired": expired})
    except Exception as exc:  # noqa: BLE001
        return public_error_json(exc)


async def _graph_context_list(engine: Any, session_id: str) -> str:
    try:
        # CONCEPT:AU-ORCH.session.session-anchored-collections-native — id-anchored traversal from the Session node (the engine's
        # reliable, fast O(degree) path; the index-less backend can't serve property
        # scans). The traversal reader returns whole nodes (`RETURN c`), so project +
        # sort + limit client-side.
        rows = await run_blocking_ordered(
            engine.query_cypher,
            "MATCH (s {id: $snode})-[:HAS_CONTEXT]->(c:ContextBlob) RETURN c",
            {"snode": f"session:{session_id}"},
        )
        items = []
        for r in rows or []:
            c = r.get("c") if isinstance(r, dict) else None
            if isinstance(c, dict) and str(c.get("id", "")).startswith("ctx:"):
                items.append(
                    {
                        "context_id": c.get("id"),
                        "key": c.get("key"),
                        "created_at": c.get("created_at"),
                    }
                )
        items.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
        return json.dumps(items[:50], default=str)
    except Exception as exc:  # noqa: BLE001
        return public_error_json(exc)


def register_query_tools(mcp):
    """Register the query_tools group on the given FastMCP server."""

    def _run_graph_query(
        cypher: str = Field(
            description="A Cypher query string (read-only — no CREATE/MERGE/DELETE)."
        ),
        params: str = Field(default="{}", description="JSON-encoded query parameters."),
        scope: str = Field(
            default="local",
            description=(
                "'local' for the internal KG (Cypher), 'sql' to run read-only SQL over the "
                "KG + user tables via the engine's DataFusion surface (e.g. SELECT ... FROM "
                "nodes — CONCEPT:AU-KG.query.read-only-sql-over, same path as the pg-wire listener), 'sparql' to "
                "run a SPARQL 1.1 SELECT/ASK over the engine's RDF projection of the graph "
                "(CONCEPT:AU-KG.ingest.mirror-inbound), or 'federated' to query an external graph endpoint. For "
                "'sql'/'sparql' the `cypher` arg carries the SQL/SPARQL string."
            ),
        ),
        reference_id: str = Field(
            default="",
            description="Required when scope='federated'. The ExternalGraphReference node ID.",
        ),
        as_of: str = Field(
            default="",
            description=(
                "CONCEPT:AU-KG.query.as-of-instant-filter — optional ISO-8601 instant. When set, rows are filtered to "
                "those whose bi-temporal validity (valid_from <= as_of < valid_to) holds, "
                "answering 'what was true as of date T'."
            ),
        ),
        connection: str = Field(
            default="",
            description=(
                "CONCEPT:AU-KG.backend.multi-connection-registry — named BACKEND connection to query "
                "(default = primary). Use a registered connection name (e.g. "
                "'prod-neo4j'), or 'all' (or a comma-separated list) to fan out the "
                "same query to several backends and get per-connection labeled "
                "results. This selects WHICH BACKEND, never which physical graph "
                "within it — see `graph` for that."
            ),
        ),
        graph: str = Field(
            default="",
            description=(
                "CONCEPT:AU-KG.backend.explicit-graph-selection — explicit PHYSICAL engine graph to query "
                "(one of the names `engine_tenants(action='list')`/the engine's own "
                "ListGraphs returns), independent of `connection`. Empty = the "
                "caller's own bound graph (unchanged default behavior). Requires "
                "exactly one resolved `connection` (the default one) — never "
                "combinable with `connection='all'`/a list. Authorization-checked "
                "by the engine's own RBAC/RLS on every call; an unknown graph, or "
                "a `connection` with no physical-graph concept, is a typed error — "
                "never a silent fallback to a default graph and never a union "
                "across graphs. Every response echoes the `connection`/`graph` "
                "actually used."
            ),
        ),
        include_epistemic: bool = Field(
            default=False,
            description=(
                "CONCEPT:AU-KB-CURRENCY — opt-in for local Cypher "
                "(ignored on scope='sql'/'sparql'/'federated'). When true, each "
                "result row is currency-upgraded via the "
                "engine's `explain_provenance_by_ids` into a per-row epistemic "
                "envelope — confidence, bitemporal valid/tx time, evidence "
                "provenance, policy labels — alongside the row's own properties "
                "(never fabricated, resolved server-side). The widened rows are "
                "projected into the same EvidenceBundle response. Degrades to an empty "
                "result when the "
                "connected backend has no epistemic primitive."
            ),
        ),
    ) -> str:
        """Execute a read-only Cypher query against the Knowledge Graph. Use this to fetch graph data, explore relationships, and read node properties."""
        # A direct call bypassing `_execute_tool` (which resolves `Field`
        # defaults) binds an omitted `graph` to its raw, truthy
        # `pydantic.fields.FieldInfo` rather than `""` — normalize once here so
        # every `if graph`/`resolve_explicit_graph`/`bound_to_graph` use below
        # sees a clean string, mirroring the SAME defensiveness
        # `ConnectionRegistry.resolve_names` already applies to `connection`.
        graph, parsed_params, include_epistemic_flag = _run_graph_query_normalize_args(
            graph, params, include_epistemic
        )

        if scope == "sql":
            # CONCEPT:AU-KG.query.read-only-sql-over — read-only SQL over the KG via the engine's
            # DataFusion surface (the same path the pg-wire listener uses). The
            # `cypher` arg carries the SQL string. RLS-governed + read-path-first
            # (engine.sql refuses non-SELECT). Honors `connection` fan-out like
            # Cypher; `graph` (CONCEPT:AU-KG.backend.explicit-graph-selection) selects a physical engine
            # graph, independent of `connection` — see `resolve_explicit_graph`.
            return _run_graph_query_sql(cypher, connection, graph)

        if scope == "sparql":
            # CONCEPT:AU-KG.ingest.mirror-inbound — SPARQL 1.1 (SELECT/ASK/CONSTRUCT/DESCRIBE) over the
            # engine's RDF projection of the live graph. The `cypher` arg carries the
            # SPARQL string. RLS-governed (engine.sparql visibility-filters rows) and
            # honors `connection` fan-out like Cypher/SQL; `graph` selects a physical
            # engine graph, independent of `connection`.
            return _run_graph_query_sparql(cypher, connection, graph)

        if scope == "federated":
            return _run_graph_query_federated(cypher, reference_id, parsed_params)

        # Local reads use each backend's server-enforced read-only transaction.
        # The native engine requires an explicit read mode and validates it with
        # the complete parser; external backends without an equivalent contract
        # fail closed. No lexical query filter is an authorization boundary.
        # CONCEPT:AU-KG.backend.multi-connection-registry — resolve the connection(s). CONCEPT:AU-KG.ingest.unified-query-routing —
        # with ingestion graph routing on, an implicit-default read fans across the
        # active content-graph set so split content is still queryable as one KG.
        # CONCEPT:AU-KG.backend.explicit-graph-selection — `graph` (a physical engine graph) is a SEPARATE axis
        # from `connection` (a backend alias); it requires exactly one resolved
        # connection, so it fails closed against any fan-out (explicit or the
        # implicit content-graph union below) rather than silently ignoring the
        # request or unioning across graphs.
        entries, errors, fanout, error_response = _run_graph_query_resolve_local(
            connection, graph
        )
        if error_response is not None:
            return error_response

        # Whether this fan-out is the implicit content-graph UNION (no explicit
        # connection). Those rows are merged into the canonical ``rows`` field; an
        # explicit ``connection='all'``/list keeps the per-target map.
        _union_read = _run_graph_query_is_union_read(fanout, connection)

        # CONCEPT:AU-KG.query.query-aggregation — an aggregation (count/sum/group-by) under the implicit
        # content-graph union CANNOT be fanned: aggregate rows carry no node id to
        # dedup on, so id-dedup leaves one copy of every group row PER
        # graph (for example, one aggregate row repeated per graph). Summing generically is
        # unsafe (wrong for avg/min/max/distinct). Run the aggregation against the
        # canonical default graph only — control-plane/aggregate reads resolve there.
        if _union_read and is_aggregation_cypher(cypher):
            return _run_graph_query_union_aggregate(
                cypher, parsed_params, as_of, include_epistemic_flag, graph
            )

        if not fanout:
            # Single connection (default or one named).
            _name, engine = entries[0]
            return _run_graph_query_single(
                cypher, parsed_params, as_of, include_epistemic, graph, _name, engine
            )

        return _run_graph_query_fanout(
            cypher,
            parsed_params,
            as_of,
            include_epistemic_flag,
            connection,
            graph,
            entries,
            errors,
            _union_read,
        )

    @mcp.tool(
        name="graph_query",
        description=(
            "Execute a read-only Cypher, SQL, SPARQL, or federated graph query and "
            "return the sole typed EvidenceBundle response."
        ),
        tags=["graph-os", "query"],
    )
    def graph_query(
        cypher: str = Field(
            description="A read-only query string; the selected scope determines its dialect."
        ),
        params: str = Field(default="{}", description="JSON-encoded query parameters."),
        scope: str = Field(
            default="local",
            description="local | sql | sparql | federated",
        ),
        reference_id: str = Field(
            default="",
            description="ExternalGraphReference id required for federated queries.",
        ),
        as_of: str = Field(
            default="", description="Optional ISO-8601 bitemporal query instant."
        ),
        connection: str = Field(
            default="",
            description=(
                "Named BACKEND connection, 'all', or a connection list. Selects "
                "WHICH BACKEND, never which physical graph — see `graph`."
            ),
        ),
        graph: str = Field(
            default="",
            description=(
                "CONCEPT:AU-KG.backend.explicit-graph-selection — explicit physical engine graph (see "
                "`engine_tenants(action='list')`), independent of `connection`. "
                "Empty = the caller's own bound graph. Fail-closed: an unknown "
                "graph or a `connection`/fan-out that has no physical-graph "
                "concept is a typed error, never a silent default/union. Echoed "
                "back in the response alongside `connection`."
            ),
        ),
        include_epistemic: bool = Field(
            default=False,
            description="Resolve per-row epistemic data before building the bundle.",
        ),
    ) -> EvidenceBundle:
        raw = _run_graph_query(
            cypher=cypher,
            params=params,
            scope=scope,
            reference_id=reference_id,
            as_of=as_of,
            connection=connection,
            graph=graph,
            include_epistemic=include_epistemic,
        )
        return EvidenceBundle.from_payload(raw, operation="graph_query")

    kg_server.REGISTERED_TOOLS["graph_query"] = graph_query

    # ══════════════════════════════════════════════════════════════════
    # 1a-bis. graph_ask — CONCEPT:AU-KG.ingest.mirror-inbound natural-language → query
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_ask",
        description=(
            "CONCEPT:AU-KG.ingest.mirror-inbound — ask the Knowledge Graph in plain English. An LLM "
            "translates your question (grounded in the live node-label + SQL-table "
            "schema) into a single read-only query in the best dialect — Cypher over "
            "the property graph, SQL over the KG + user tables, or SPARQL over the RDF "
            "projection — then executes it through the matching engine surface. Returns "
            "the GENERATED query (auditable), the result rows, and citations (the node/"
            "source ids touched), so the answer is grounded and verifiable, not a black "
            "box. Set execute=false to preview the query without running it; pin "
            "dialect='cypher'|'sql'|'sparql' to force one (default 'auto' lets the model "
            "choose)."
        ),
        tags=["graph-os", "query", "nl"],
    )
    def graph_ask(
        question: str = Field(description="The natural-language question to answer."),
        dialect: str = Field(
            default="auto",
            description="'auto' (model chooses) or 'cypher'|'sql'|'sparql' to force one.",
        ),
        execute: bool = Field(
            default=True,
            description="When false, return only the generated query (preview/dry-run).",
        ),
        limit: int = Field(default=50, description="Max result rows to return."),
        include_epistemic: bool = Field(
            default=False,
            description=(
                "CONCEPT:AU-KB-CURRENCY — opt-in. Only takes effect when the "
                "generated (or forced) query resolves to the 'cypher' dialect "
                "(sql/sparql have no epistemic-envelope surface, so this is a "
                "silent no-op for those). When true and honored, `results` holds "
                "per-row epistemic envelopes (confidence, bitemporal valid/tx "
                "time, evidence provenance, policy labels) instead of plain rows, "
                "and `citations` degrades to an empty list."
            ),
        ),
    ) -> EvidenceBundle:
        from agent_utilities.knowledge_graph.core.nl_query import nl_to_query

        try:
            engine = kg_server._get_engine()
        except Exception as e:  # noqa: BLE001
            return EvidenceBundle.from_payload(
                public_error_json(e, code="dependency_unavailable"),
                operation="graph_ask",
            )
        try:
            result = nl_to_query(
                engine,
                str(question),
                dialect=str(dialect),
                execute=bool(execute),
                limit=int(limit),
                include_epistemic=bool(include_epistemic),
            )
            return EvidenceBundle.from_nl_query(result)
        except Exception as e:  # noqa: BLE001
            return EvidenceBundle.from_payload(
                public_error_json(e), operation="graph_ask"
            )

    kg_server.REGISTERED_TOOLS["graph_ask"] = graph_ask

    # ══════════════════════════════════════════════════════════════════
    # 1a-quater. nl_query — CONCEPT:AU-KG.query.ask-gateway-rest-twin AU-fleet-LLM as the engine's NL planner
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="nl_query",
        description=(
            "CONCEPT:AU-KG.query.ask-gateway-rest-twin — ask the Knowledge Graph in plain English, planned by "
            "agent-utilities' OWN configured fleet LLM (the local vLLM / provider the rest "
            "of AU uses) acting as the engine's NL planner. The fleet model translates your "
            "request (grounded in the live node-label + SQL-table schema) into a single "
            "read-only query — preferring UQL, the engine's native cross-modal language "
            "(also cypher/sql/sparql) — which is then submitted to the engine's OWN "
            "deterministic executor via the existing AU->engine query path. This is the "
            "'LLM opt-out from AU' half of the NL->query dual-mode: it uses the fleet LLM "
            "instead of the engine's standalone ureq client. Returns the GENERATED query "
            "(auditable), result rows, and citations. Set execute=false to preview without "
            "running; pin dialect='uql'|'cypher'|'sql'|'sparql' to force one. Falls back to "
            "a clean error when no LLM is configured."
        ),
        tags=["graph-os", "query", "nl"],
    )
    def nl_query(
        text: str = Field(description="The natural-language request to answer."),
        dialect: str = Field(
            default="auto",
            description="'auto' (model chooses, prefers uql) or 'uql'|'cypher'|'sql'|'sparql'.",
        ),
        schema_hint: str = Field(
            default="",
            description="Optional extra schema/context hint to ground the planner.",
        ),
        execute: bool = Field(
            default=True,
            description="When false, return only the generated query (preview/dry-run).",
        ),
        limit: int = Field(default=50, description="Max result rows to return."),
    ) -> EvidenceBundle:
        from agent_utilities.knowledge_graph.core import nl_planner

        try:
            engine = kg_server._get_engine()
        except Exception as e:  # noqa: BLE001
            return EvidenceBundle.from_payload(
                public_error_json(e, code="dependency_unavailable"),
                operation="nl_query",
            )
        try:
            result = nl_planner.nl_query(
                engine,
                str(text),
                dialect=str(dialect),
                schema_hint=str(schema_hint),
                execute=bool(execute),
                limit=int(limit),
            )
            return EvidenceBundle.from_nl_query(result)
        except Exception as e:  # noqa: BLE001
            return EvidenceBundle.from_payload(
                public_error_json(e), operation="nl_query"
            )

    kg_server.REGISTERED_TOOLS["nl_query"] = nl_query
    # CONCEPT:AU-KG.query.ask-gateway-rest-twin — gateway REST twin (W1 exposure; MCP⇄REST parity)
    kg_server.ACTION_TOOL_ROUTES["nl_query"] = "/graph/nl-query"

    # ══════════════════════════════════════════════════════════════════
    # 1a-quinquies. ask_data — CONCEPT:AU-KG.query.data-gateway-rest-twin DB-GPT-style data-analysis agent loop
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="ask_data",
        description=(
            "CONCEPT:AU-KG.query.data-gateway-rest-twin — answer a DATA question over the Knowledge Graph with a "
            "DB-GPT-style, multi-step data-analysis agent (distinct from the single-shot "
            "nl_query). The agent runs a bounded ReAct loop: schema-link the question to the "
            "most relevant tables/labels, generate a read-only query via the AU fleet-LLM "
            "planner (KG-2.305), execute it through the engine's own deterministic executor, "
            "and — on a query error — feed the failing query + error back for up to "
            "'max_corrections' self-correction retries; on success it synthesizes a "
            "natural-language ANSWER from the rows. Returns the synthesized answer, the "
            "query used (auditable), the result rows, citations, the linked schema, and the "
            "full attempt trace. Falls back to a clean error when no LLM is configured."
        ),
        tags=["graph-os", "query", "nl", "data-analysis"],
    )
    def ask_data(
        question: str = Field(
            description="The natural-language DATA question to answer."
        ),
        dialect: str = Field(
            default="auto",
            description="'auto' (planner chooses, prefers uql) or 'uql'|'cypher'|'sql'|'sparql'.",
        ),
        max_corrections: int = Field(
            default=2,
            description="Bounded self-correction retries after a failed query (0 disables).",
        ),
        limit: int = Field(default=50, description="Max result rows to return."),
    ) -> str:
        from agent_utilities.knowledge_graph.orchestration import data_analyst

        try:
            engine = kg_server._get_engine()
        except Exception as e:  # noqa: BLE001
            return public_error_json(e, code="dependency_unavailable")
        try:
            return json.dumps(
                data_analyst.ask_data(
                    engine,
                    str(question),
                    dialect=str(dialect),
                    max_corrections=int(max_corrections),
                    limit=int(limit),
                ),
                default=str,
            )
        except Exception as e:  # noqa: BLE001
            return public_error_json(e)

    kg_server.REGISTERED_TOOLS["ask_data"] = ask_data
    # CONCEPT:AU-KG.query.data-gateway-rest-twin — gateway REST twin (W1 exposure; MCP⇄REST parity)
    kg_server.ACTION_TOOL_ROUTES["ask_data"] = "/graph/ask-data"

    # ══════════════════════════════════════════════════════════════════
    # 1a-ter. graph_table — CONCEPT:AU-KG.ingest.mirror-inbound connector/ETL → native engine SQL tables
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_table",
        description=(
            "CONCEPT:AU-KG.ingest.mirror-inbound — mirror data into native engine SQL tables (DataFusion + "
            "pg-wire) and manage them. Actions: 'ingest' (mirror a registered source "
            "connector's documents into a table via CREATE TABLE + bulk INSERT — "
            "source=<connector> e.g. rest/database/web/rss/filesystem/reader, "
            "config_json=connector config, table=<name>, replace=true to recreate), "
            "'rows' (bulk-INSERT arbitrary rows from rows_json into table), 'create' "
            "(CREATE TABLE table with columns_json), 'list' (list user tables), 'drop' "
            "(DROP TABLE table), 'query' (run a read-only SELECT via the engine SQL "
            "surface — sql=<SELECT ...>). This is how 'ingest tables from any connector "
            "/ mirror data into our DB' works."
        ),
        tags=["graph-os", "ingestion", "table"],
    )
    def graph_table(
        action: str = Field(
            default="list",
            description="'ingest' | 'rows' | 'create' | 'list' | 'drop' | 'query'.",
        ),
        source: str = Field(
            default="", description="Registered connector key (action='ingest')."
        ),
        table: str = Field(default="", description="Target SQL table name."),
        config_json: str = Field(
            default="{}", description="JSON connector config (action='ingest')."
        ),
        columns_json: str = Field(
            default="[]", description="JSON list of column names (action='create')."
        ),
        rows_json: str = Field(
            default="[]", description="JSON list of row dicts (action='rows')."
        ),
        sql: str = Field(
            default="", description="A read-only SELECT statement (action='query')."
        ),
        limit: int = Field(
            default=1000, description="Max rows to mirror (action='ingest')."
        ),
        replace: bool = Field(
            default=False, description="Drop+recreate the table first (ingest/rows)."
        ),
    ) -> str:
        from agent_utilities.knowledge_graph.core import table_ingest

        try:
            engine = kg_server._get_engine()
        except Exception as e:  # noqa: BLE001
            return public_error_json(e, code="dependency_unavailable")

        try:
            if action == "ingest":
                return _graph_table_ingest(
                    engine, table_ingest, source, table, config_json, limit, replace
                )
            if action == "rows":
                return _graph_table_rows(
                    engine, table_ingest, table, rows_json, replace
                )
            if action == "create":
                return _graph_table_create(engine, table_ingest, table, columns_json)
            if action == "list":
                return json.dumps({"tables": table_ingest.list_tables(engine)})
            if action == "drop":
                return _graph_table_drop(engine, table_ingest, table)
            if action == "query":
                return _graph_table_query(engine, sql)
            return json.dumps({"error": f"unknown action {action!r}"})
        except Exception as e:  # noqa: BLE001
            return public_error_json(e)

    kg_server.REGISTERED_TOOLS["graph_table"] = graph_table

    # ══════════════════════════════════════════════════════════════════
    # 1a-quater. graph_projection — CONCEPT:AU-KG.query.node-link-projection
    # (wD10 Atlas backend gap B): a stable {nodes, links} JSON shape for a
    # graph query result, so the agent-webui data explorer's 2D/3D renderers
    # never have to re-derive node/edge shape from raw per-dialect rows.
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_projection",
        description=(
            "CONCEPT:AU-KG.query.node-link-projection — run the SAME query `graph_query` would "
            "and project the result into a stable node-link JSON shape: "
            "{nodes:[{id,label,properties}], links:[{source,target,label,properties}]} — "
            "the D3/force-graph convention every 2D/3D graph renderer already speaks. "
            "A dict-valued result column shaped like a node ({id, ...}) becomes a node; "
            "one shaped like a relationship ({source/start/..., target/end/..., type, ...}) "
            "becomes a link; a plain scalar column is never fabricated into either. "
            "Nodes are de-duplicated by id across all rows. Takes ONE JSON request "
            "object (project rule, wD10 preamble addendum — no new wide MCP-tool "
            "signatures) rather than one Field per argument: request_json is "
            "{cypher (required), params, scope, reference_id, as_of, connection, "
            "graph} — identical fields/semantics to graph_query's own arguments."
        ),
        tags=["graph-os", "query", "visualization"],
    )
    def graph_projection(
        request_json: str = Field(
            description=(
                "JSON object: {cypher (required), params, scope, reference_id, "
                "as_of, connection, graph} — same fields/semantics as graph_query's "
                "individual arguments."
            )
        ),
    ) -> str:
        run_kwargs, request_error = _parse_projection_request(request_json)
        if request_error is not None:
            return json.dumps(
                {"error": {"code": "invalid_request", "message": request_error}}
            )
        raw = _run_graph_query(**run_kwargs, include_epistemic=False)
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            return json.dumps({"error": {"code": "operation_failed"}})
        if isinstance(payload, dict) and payload.get("error"):
            return json.dumps({"error": payload["error"]})
        projection = _project_node_link(_projection_rows(payload))
        projection["connection"] = run_kwargs["connection"] or "default"
        projection["graph"] = run_kwargs["graph"]
        return json.dumps(projection, default=str)

    kg_server.REGISTERED_TOOLS["graph_projection"] = graph_projection
    kg_server.ACTION_TOOL_ROUTES["graph_projection"] = "/graph/projection"

    # ══════════════════════════════════════════════════════════════════
    # 1a-quinquies. graph_catalog — CONCEPT:AU-KG.query.unified-capability-catalog
    # (wD10 Atlas backend gap A): one capability-shaped call across every
    # modality instead of 4-6 protocol-specific ones. See the
    # `_build_graph_catalog` module docstring for the full design rationale.
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_catalog",
        description=(
            "CONCEPT:AU-KG.query.unified-capability-catalog — one capability-shaped call across "
            "every KG modality (graphs, sql, kv, vector, ontology, timeseries, blob, "
            "broker, saved_queries) instead of making a separate protocol-specific "
            "introspection call per modality. Each entry reports "
            "{available: bool, reason?, methods?, items?, denied?}: an absent/"
            "not-compiled-in backend yields available=false with a reason, NEVER "
            "an error — a frontend source-tree view can render every modality's "
            "state from this one response. 'graphs' lists physical engine graphs "
            "(ADMIN-scoped; a non-admin caller sees denied=true, not an error). "
            "'sql' lists user SQL tables with a best-effort per-table column probe "
            "(capped). Other modalities report compiled-in capability + their "
            "engine method list where the installed engine build has no safe "
            "generic listing method — never a guessed/fabricated item list. "
            "The 'sources' entry is Atlas's governed provider catalogue: it "
            "classifies PostgreSQL/database, registered Neo4j/AGE/Ladybug/"
            "Epistemic Graph backends, generic OpenCypher and "
            "PuppyGraph, GraphQL, virtual graphs, Spark (compute-only), "
            "Iceberg/Trino, S3/object stores, and unsupported Teradata. It "
            "returns only neutral connection/profile references and truthful "
            "availability/reason/queryMode/dialects/sync/capabilities. "
            "action='preview_sync' validates one source_sync request and "
            "returns a typed non-executable normalization; it never runs a "
            "connector."
        ),
        tags=["graph-os", "query", "introspection"],
    )
    async def graph_catalog(
        action: Literal["list", "preview_sync"] = Field(
            default="list",
            description=(
                "'list' (default) returns modality and provider catalogues; "
                "'preview_sync' validates one source_sync request without executing it."
            ),
        ),
        source: str = Field(
            default="",
            description="For preview_sync: exactly one neutral source identifier.",
        ),
        mode: str = Field(
            default="delta",
            description="For preview_sync: 'delta', 'full', or 'reconcile'.",
        ),
        ids_json: str = Field(
            default="[]",
            description="For preview_sync: JSON list of bounded source record ids.",
        ),
        connection: str = Field(
            default="",
            description="For preview_sync: one neutral named backend connection.",
        ),
        graph: str = Field(
            default="",
            description="For preview_sync: one neutral physical graph name.",
        ),
    ) -> str:
        try:
            # ``_execute_tool`` already scopes served calls, but this explicit
            # check protects direct MCP-function invocation and documents the
            # metadata boundary: profile/connection status is a governed read.
            from agent_utilities.knowledge_graph.core.session import resolve_session

            resolve_session(required_scope="kg:read")
        except Exception as e:  # noqa: BLE001
            return public_error_json(e)
        return await _graph_catalog_response(
            action,
            source=source,
            mode=mode,
            ids_json=ids_json,
            connection=connection,
            graph=graph,
        )

    kg_server.REGISTERED_TOOLS["graph_catalog"] = graph_catalog
    kg_server.ACTION_TOOL_ROUTES["graph_catalog"] = "/graph/catalog"

    # ══════════════════════════════════════════════════════════════════
    # 1b. graph_context — CONCEPT:AU-ORCH.session.invoker-agent-handoff cross-process curated-context store
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_context",
        description=(
            "CONCEPT:AU-ORCH.session.invoker-agent-handoff — store/fetch curated context for invoker→spawned-agent "
            "handoff, persisted in the epistemic-graph so a SEPARATELY-spawned agent can read "
            "it by id. Actions: 'put' (store content, returns context_id), 'get' (fetch by "
            "context_id), 'list' (by session_id). Pass the returned context_id to "
            "graph_orchestrate(context_ref=...)."
        ),
        tags=["graph-os", "orchestrate", "context"],
    )
    async def graph_context(
        action: str = Field(default="put", description="put | get | list"),
        content: str = Field(
            default="", description="Context text to store (action=put)."
        ),
        context_id: str = Field(default="", description="ContextBlob id (action=get)."),
        session_id: str = Field(default="", description="Session scope key."),
        key: str = Field(
            default="", description="Optional sub-key within the session."
        ),
        ttl_s: int = Field(
            default=0, description="Optional time-to-live in seconds (0 = persistent)."
        ),
    ) -> str:
        engine = kg_server._get_engine()
        if not engine:
            return json.dumps({"error": "IntelligenceGraphEngine not active."})
        if action == "put":
            return await _graph_context_put(
                engine, content, context_id, session_id, key, ttl_s
            )
        if action == "get":
            return await _graph_context_get(engine, context_id)
        if action == "prune":
            return await _graph_context_prune(engine)
        if action == "list":
            return await _graph_context_list(engine, session_id)
        return json.dumps({"error": f"unknown action: {action}"})

    kg_server.REGISTERED_TOOLS["graph_context"] = graph_context

    # ══════════════════════════════════════════════════════════════════
    # 1c. graph_message — CONCEPT:AU-ORCH.session.session-anchored-collections-native invoker↔spawned-agent message channel
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_message",
        description=(
            "CONCEPT:AU-ORCH.session.session-anchored-collections-native — bidirectional, cross-process, ordered message channel between "
            "an invoking agent and a spawned agent, over the epistemic-graph native channels. "
            "Actions: 'open' (session_id+run_id → channel_id), 'send' (channel_id+sender+payload "
            "[+durable]), 'receive' (channel_id [+since cursor] → new messages + cursor), "
            "'history' (durable replay, survives restart), 'close'. Use the channel_id returned "
            "by graph_orchestrate(open_channel=True) to talk to the spawned agent."
        ),
        tags=["graph-os", "orchestrate", "messaging"],
    )
    async def graph_message(
        action: str = Field(
            default="receive", description="open | send | receive | history | close"
        ),
        channel_id: str = Field(
            default="", description="Channel id (send/receive/history/close)."
        ),
        session_id: str = Field(default="", description="Session id (open)."),
        run_id: str = Field(default="", description="Spawned run id (open)."),
        sender: str = Field(default="invoker", description="Sender label (send)."),
        payload: str = Field(default="", description="Message text (send)."),
        since: int = Field(
            default=0, description="Cursor: messages already consumed (receive)."
        ),
        durable: bool = Field(
            default=False,
            description="When True (send), also persist the message as a graph AgentMessage "
            "node so it survives engine restart and is replayable via action='history'.",
        ),
    ) -> str:
        from agent_utilities.messaging import agent_channel

        engine = kg_server._get_engine()
        if not engine:
            return json.dumps({"error": "IntelligenceGraphEngine not active."})
        if action == "open":
            cid = agent_channel.open_channel(engine, session_id, run_id)
            return json.dumps({"channel_id": cid})
        if action == "send":
            return json.dumps(
                {
                    "sent": agent_channel.send(
                        engine, channel_id, sender, payload, durable=bool(durable)
                    )
                }
            )
        if action == "receive":
            try:
                msgs, cursor = agent_channel.receive(engine, channel_id, since=since)
            except agent_channel.ChannelNotFoundError:
                # BUG-8: surface the not-found condition instead of a silent
                # empty result indistinguishable from "no new messages".
                return json.dumps(
                    {"error": "channel not found", "channel_id": channel_id}
                )
            return json.dumps({"messages": msgs, "cursor": cursor}, default=str)
        if action == "history":
            return json.dumps(
                {"messages": agent_channel.history(engine, channel_id)}, default=str
            )
        if action == "close":
            return json.dumps({"closed": agent_channel.close(engine, channel_id)})
        return json.dumps({"error": f"unknown action: {action}"})

    kg_server.REGISTERED_TOOLS["graph_message"] = graph_message

    # ══════════════════════════════════════════════════════════════════
    # 2. kg_search — Unified search (hybrid, concept, analogy, memory)
    # ══════════════════════════════════════════════════════════════════

    @mcp.tool(
        name="graph_search",
        description="Search the Knowledge Graph using multiple strategies (hybrid, concept, analogy, memory, discover, dci).",
        tags=["graph-os", "search"],
    )
    async def graph_search(
        query: str = Field(description="Natural language search query or concept ID."),
        mode: str = Field(
            default="hybrid",
            description="Search strategy:\n- 'hybrid': Semantic + keyword weighted search (default).\n- 'hyde': Memory-first HyDE multi-query plan + dual threshold (CONCEPT:AU-KG.retrieval.self-correcting-second-pass).\n- 'deep': Wide-recall single query at the 0.28 deep threshold.\n- 'concept': Look up a CONCEPT:ID (e.g. 'AU-KG.query.vendor-agnostic-traversal', 'AU-ORCH.execution.inject-signal-board-observations').\n- 'analogy': Find structurally similar concepts.\n- 'memory': Search tiered memory (episodic/semantic/procedural).\n- 'discover': Cross-reference query against all ingested content.\n- 'dci': Direct Corpus Interaction.\n- 'latent': Latent-topology hierarchical routing (CONCEPT:AU-KG.memory.auto-similarity-memory-graph).\n- 'sira': Single-shot SIRA sparsity-aligned context.\n- 'hard_negatives': Mine hard negatives for the query (CONCEPT:AU-KG.memory.auto-similarity-memory-graph).\n- 'rerank': Hybrid semantic+keyword re-scoring of candidates.\n- 'adore': Iterative query expansion with retrieval-grounded graded relevance feedback + training-free stopping (CONCEPT:AU-KG.query.adore-concept-expansion/2.87).\n- 'chrono_ids': Attach an explicit temporal semantic ID (+recency bucket) to each result for generative retrieval (CONCEPT:AU-KG.query.chronoid-fits-residual-quantization).\n- 'compiled': Policy-aware ``ContextCompiler`` bundle (CONCEPT:AU-KG.retrieval.context-compiler) — MMR-diversified, evidence/freshness-weighted, token-budgeted, policy-filtered context with citations + a proof graph, instead of the plain relevance-sorted text the other modes return.",
        ),
        top_k: int = Field(default=10, description="Maximum results to return."),
        self_correct: bool = Field(
            default=False,
            description="CONCEPT:AU-KG.retrieval.self-correcting-second-pass — run a self-correcting second retrieval pass at the deep threshold when the quality gate fails.",
        ),
        as_of: str = Field(
            default="",
            description="Optional ISO-8601 instant. Pack-driven recency decay is measured relative to this time, enabling knowledge-state-as-of-date-D retrieval such as an academic literature state. Defaults to now (CONCEPT:EG-KG.compute.rust-native-training-loss).",
        ),
        connection: str = Field(
            default="",
            description=(
                "CONCEPT:AU-KG.backend.multi-connection-registry — named BACKEND connection to search "
                "(default = primary). Use a registered connection name, or 'all' "
                "(or a comma-separated list) to fan out and get per-connection "
                "labeled results. Selects WHICH BACKEND, never which physical "
                "graph — see `graph`."
            ),
        ),
        graph: str = Field(
            default="",
            description=(
                "CONCEPT:AU-KG.backend.explicit-graph-selection — explicit physical engine graph, independent of "
                "`connection`. Empty = the caller's own bound graph. Requires "
                "exactly one resolved `connection`; fails closed (typed error, "
                "never a silent default/union) on an unknown graph or a fan-out/"
                "unsupported-connection combination. Echoed in the response."
            ),
        ),
        token_budget: int = Field(
            default=0,
            description=(
                "mode='compiled' only: token budget the assembled bundle must fit "
                "inside (CONCEPT:AU-KG.retrieval.context-compiler). 0 uses the compiler's default budget."
            ),
        ),
    ) -> str:
        """Search the Knowledge Graph using multiple strategies. Useful for finding context, concepts, memories, and capabilities across the ecosystem."""
        # See `_run_graph_query`'s identical normalization: a direct call
        # bypassing `_execute_tool` binds an omitted `graph` to a raw, truthy
        # `FieldInfo` rather than `""`.
        graph = graph if isinstance(graph, str) else ""

        def _execute() -> str:
            def _run_search(engine: Any) -> str:
                if not engine:
                    return "Error: IntelligenceGraphEngine not active."
                try:
                    return _search_with_engine(engine)
                except Exception as e:
                    return public_error_text(e)

            def _content_graph_search(name: str, engine: Any) -> str:
                # B-18 fix: mirrors `_run_graph_query`'s `_content_graph_query`
                # above — within the implicit content-graph UNION fan-out,
                # `name` is the physical graph `engine` is scoped to
                # (`ingest_routing.safe_engine_for_graph`'s `for_graph()` view
                # never rebinds `session.graph`), so the wire layer's mismatch
                # lock rejected every non-"default" leg. Narrow the verified
                # session to the same graph via the sanctioned
                # `bound_to_graph` primitive before running the search.
                # `_run_search` never raises (it self-catches into an error
                # string) — match that contract so `bound_to_graph` itself
                # failing (e.g. no ambient session) degrades the same way
                # instead of escaping this helper uncaught.
                try:
                    with kg_server.bound_to_graph(name):
                        return _run_search(engine)
                except Exception as e:
                    return public_error_text(e)

            def _search_with_engine(engine: Any) -> str:
                # CONCEPT:AU-KG.retrieval.acl-aware-vector-retrieval — every served
                # `graph_search` call already runs inside the middleware-minted
                # ambient GraphSession (`verified_tool_session_scope`). Passing it
                # explicitly opts THIS served path into `search_hybrid`'s per-node
                # ACL + owner/scope + audit enforcement (a no-op for internal
                # callers that pass no session — see `search_hybrid`'s docstring),
                # closing the gap where vector/hybrid results reached an external
                # caller completely unfiltered, unlike the guarded `graph_query`
                # Cypher path.
                from agent_utilities.knowledge_graph.core.session import (
                    current_session,
                )

                _session = current_session()
                if mode in _GRAPH_SEARCH_RESULTS_MODES:
                    results = _graph_search_produce_results(
                        engine,
                        _session,
                        mode=mode,
                        query=query,
                        top_k=top_k,
                        self_correct=self_correct,
                        as_of=as_of,
                    )
                elif mode == "discover":
                    return _graph_search_discover(engine, query)
                elif mode == "hard_negatives":
                    return _graph_search_hard_negatives(engine, query)
                elif mode == "compiled":
                    return _graph_search_compiled(
                        engine,
                        query,
                        top_k=top_k,
                        as_of=as_of,
                        token_budget=token_budget,
                    )
                else:
                    return f"Error: Unknown search mode '{mode}'"

                if not results:
                    return f"No results found for query: '{query}'"
                return _graph_search_format_results(results)

            # CONCEPT:AU-KG.backend.multi-connection-registry — resolve the connection(s). CONCEPT:AU-KG.ingest.unified-query-routing — an
            # implicit-default search fans across the active content-graph set so content
            # routed to ``code:*``/``src:*`` graphs stays findable as one KG.
            # CONCEPT:AU-KG.backend.explicit-graph-selection — `graph` requires exactly one resolved
            # connection, so it fails closed against any fan-out (explicit or the
            # implicit content-graph union) — never a silent ignore or union.
            try:
                entries, errors, fanout = kg_server._resolve_read_engines(connection)
                entries = kg_server.resolve_explicit_graph(
                    entries, graph, fanout=fanout
                )
            except kg_server.GraphNotFoundError as e:
                return public_error_text(e, code="graph_not_found")
            except kg_server.GraphSelectionConflictError as e:
                return public_error_text(e, code="graph_selection_conflict")
            except Exception as e:
                return public_error_text(e)

            if not fanout:
                return _graph_search_single_target(entries, graph, _run_search)

            return _graph_search_fanout(
                entries, errors, connection, _run_search, _content_graph_search
            )

        return await run_blocking_ordered(_execute)

    kg_server.REGISTERED_TOOLS["graph_search"] = graph_search

    @mcp.tool(
        name="graph_search_synthesis",
        description=(
            "Synthesize a shortcut-resistant deep-search task from the evidence graph, "
            "or diagnose realized search difficulty of solver trajectories "
            "(CONCEPT:AU-KG.retrieval.evidence-graph-workspace/2.71/2.72, AHE-3.30; distills arXiv:2606.12087)."
        ),
        tags=["graph-os", "search", "synthesis", "training-data"],
    )
    def graph_search_synthesis(
        action: str = Field(
            default="synthesize",
            description=(
                "'synthesize': build an evidence subgraph around an answer entity and "
                "formulate + adversarially refine a question that forces multi-hop "
                "search (no exposed constants / single-clue / co-coverage shortcuts). "
                "'diagnose': score solver trajectories with the FORT signatures "
                "(solving cost, answer hit time, prior-shortcut rate) + a search-heavy "
                "verdict."
            ),
        ),
        answer_id: str = Field(
            default="",
            description="action=synthesize — node id of the gold answer entity to build the task around.",
        ),
        hops: int = Field(
            default=2, description="action=synthesize — evidence-graph BFS depth."
        ),
        fanout: int = Field(
            default=8,
            description="action=synthesize — max neighbors expanded per node.",
        ),
        min_trust: float = Field(
            default=0.0,
            description="action=synthesize — drop facts whose source_trust is below this.",
        ),
        max_per_source: int = Field(
            default=1,
            description="action=synthesize — max clues allowed to share one evidence source before co-coverage trips.",
        ),
        root_popularity: float = Field(
            default=0.0,
            description="action=synthesize — 0..1 familiarity of the answer entity (high → prior-binding risk).",
        ),
        trajectories: str = Field(
            default="",
            description='action=diagnose — JSON list of trajectories: [{"steps":[{"kind","observation","model_text"}],"answer_aliases":[...]}].',
        ),
    ) -> str:
        """Shortcut-resistant search-task synthesis and realized-difficulty diagnosis."""
        import json as _json

        from agent_utilities.graph.training_signals import realized_difficulty

        if action == "diagnose":
            try:
                trajs = _json.loads(trajectories) if trajectories else []
            except Exception as e:  # noqa: BLE001
                return public_error_text(e)
            return _json.dumps(realized_difficulty(trajs))

        if action != "synthesize":
            return f"Error: unknown action '{action}' (expected 'synthesize' or 'diagnose')."
        if not answer_id:
            return "Error: action=synthesize requires answer_id."

        from agent_utilities.knowledge_graph.search_synthesis import synthesize

        class _Reader:
            def __init__(self, eng: Any) -> None:
                self._eng = eng

            def query(self, cypher: str, params: Any = None) -> list[dict[str, Any]]:
                query_cypher = getattr(self._eng, "query_cypher", None)
                if not callable(query_cypher):
                    raise RuntimeError("authoritative graph query service unavailable")
                return query_cypher(cypher, params or {}) or []

        engine = kg_server._get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            task = synthesize(
                _Reader(engine),
                answer_id,
                hops=hops,
                fanout=fanout,
                min_trust=min_trust,
                root_popularity=root_popularity,
                max_per_source=max_per_source,
            )
        except Exception as e:  # noqa: BLE001
            return public_error_text(e)
        return _json.dumps(task.to_dict())

    kg_server.REGISTERED_TOOLS["graph_search_synthesis"] = graph_search_synthesis

    # ══════════════════════════════════════════════════════════════════
    # graph_code_nav — CONCEPT:AU-KG.backend.declared-columns-so-schema code-symbol navigation over the
    # RESOLVED graph (`:Code` + `calls`/`depends_on`/`IMPLEMENTS`). Templated
    # so agents (and Duo-style flows) get gkg's find-def / find-refs / trace /
    # impact as first-class tools instead of hand-written Cypher.
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_code_nav",
        description=(
            "Navigate the resolved code graph (CONCEPT:AU-KG.backend.declared-columns-so-schema). action: "
            "'find_definition' (locate a symbol's :Code node), 'find_references' "
            "(callers of a symbol), 'trace_call_graph' (transitive callees), "
            "'impact_of_change' (transitive callers = blast radius), 'connects' "
            "(shortest path between TWO symbols — set symbol/node_id AND "
            "target_symbol/target_node_id — rendered hop-by-hop with each edge's "
            "relation + confidence, CONCEPT:EG-KG.compute.handled-outside-single-anchor). Start from a symbol name or "
            "an exact node_id; optionally scope to a source_system "
            "(e.g. 'gitlab:gitlab.com')."
        ),
        tags=["graph-os", "query", "code"],
    )
    def graph_code_nav(
        action: str = Field(
            description="find_definition | find_references | trace_call_graph | impact_of_change | connects"
        ),
        symbol: str = Field(
            default="", description="Symbol name to start from (function/class/method)."
        ),
        node_id: str = Field(
            default="", description="Exact :Code node id (overrides 'symbol' when set)."
        ),
        target_symbol: str = Field(
            default="",
            description="For action='connects': the destination symbol name.",
        ),
        target_node_id: str = Field(
            default="",
            description="For action='connects': the destination :Code node id.",
        ),
        source_system: str = Field(
            default="",
            description="Optional source_system filter, e.g. 'gitlab:gitlab.com'.",
        ),
        depth: int = Field(
            default=3,
            description="Max hops for trace_call_graph / impact_of_change (1-10).",
        ),
        limit: int = Field(default=200, description="Max rows to return."),
    ) -> str:
        """Templated code-symbol navigation over the resolved KG code graph."""
        engine = kg_server._get_engine()
        if not engine:
            return json.dumps({"error": "IntelligenceGraphEngine not active"})

        # 'connects' resolves two endpoints + runs a native path search — it lives
        # outside the single-anchor Cypher template.
        if action in PATH_ACTIONS:
            try:
                return json.dumps(
                    {
                        "action": action,
                        "results": code_connects(
                            engine,
                            symbol=symbol,
                            node_id=node_id,
                            target_symbol=target_symbol,
                            target_node_id=target_node_id,
                        ),
                    },
                    default=str,
                )
            except Exception as e:  # noqa: BLE001
                return public_error_json(e)

        try:
            cypher, qparams = build_code_nav_query(
                action=action,
                symbol=symbol,
                node_id=node_id,
                source_system=source_system,
                depth=depth,
                limit=limit,
            )
        except ValueError as e:
            return public_error_json(e)

        try:
            rows = engine.query_cypher(cypher, qparams)
            return json.dumps({"action": action, "results": rows}, default=str)
        except Exception as e:  # noqa: BLE001
            return public_error_json(e)

    kg_server.REGISTERED_TOOLS["graph_code_nav"] = graph_code_nav

    def _graph_document_tree_build(
        document_id: str,
        text: str,
        persist: bool,
        thin: bool,
        summarize: bool,
        engine: Any,
    ) -> str:
        from agent_utilities.knowledge_graph.ontology.document_processing import (
            SectionTreeConfig,
            build_section_tree,
            section_nodes_and_edges,
        )
        from agent_utilities.knowledge_graph.retrieval.hierarchical_document_retriever import (
            structure_view,
        )

        if not text and not document_id:
            return json.dumps({"error": "build requires 'text' or 'document_id'"})
        cfg = SectionTreeConfig(thin=thin, summarize=summarize)
        roots = build_section_tree(text, config=cfg)
        nodes, edges = section_nodes_and_edges(document_id or "doc:inline", roots)
        persisted = False
        if persist and document_id and engine is not None:
            persisted = _persist_sections(engine, nodes, edges)
        return json.dumps(
            {
                "action": "build",
                "document_id": document_id,
                "section_count": len(nodes),
                "persisted": persisted,
                "structure": structure_view(roots),
            },
            default=str,
        )

    def _graph_document_tree_structure(document_id: str, engine: Any) -> str:
        from agent_utilities.knowledge_graph.retrieval.hierarchical_document_retriever import (
            HierarchicalDocumentRetriever,
            structure_view,
        )

        if not document_id:
            return json.dumps({"error": "structure requires 'document_id'"})
        if engine is None:
            return json.dumps({"error": "IntelligenceGraphEngine not active"})
        roots = HierarchicalDocumentRetriever(engine).load_tree(document_id)
        if not roots:
            return json.dumps(
                {"error": f"no section tree for document {document_id!r}"}
            )
        return json.dumps(
            {
                "action": "structure",
                "document_id": document_id,
                "structure": structure_view(roots),
            },
            default=str,
        )

    def _graph_document_tree_content(document_id: str, ranges: str, engine: Any) -> str:
        from agent_utilities.knowledge_graph.retrieval.hierarchical_document_retriever import (
            HierarchicalDocumentRetriever,
            content_for_ranges,
        )

        if not document_id:
            return json.dumps({"error": "content requires 'document_id'"})
        if engine is None:
            return json.dumps({"error": "IntelligenceGraphEngine not active"})
        try:
            parsed = _parse_ranges(ranges)
        except ValueError as e:
            return public_error_json(e)
        roots = HierarchicalDocumentRetriever(engine).load_tree(document_id)
        if not roots:
            return json.dumps(
                {"error": f"no section tree for document {document_id!r}"}
            )
        return json.dumps(
            {
                "action": "content",
                "document_id": document_id,
                "sections": content_for_ranges(roots, parsed),
            },
            default=str,
        )

    def _graph_document_tree_retrieve(
        query: str,
        text: str,
        document_id: str,
        top_k: int,
        use_llm: bool,
        engine: Any,
    ) -> str:
        from agent_utilities.knowledge_graph.ontology.document_processing import (
            SectionTreeConfig,
            build_section_tree,
        )
        from agent_utilities.knowledge_graph.retrieval.hierarchical_document_retriever import (
            HierarchicalDocumentRetriever,
        )

        if not query:
            return json.dumps({"error": "retrieve requires 'query'"})
        retriever = HierarchicalDocumentRetriever(engine)
        tree = None
        if text:
            tree = build_section_tree(
                text, config=SectionTreeConfig(thin=True, summarize=True)
            )
        elif not document_id:
            return json.dumps({"error": "retrieve requires 'text' or 'document_id'"})
        matches = retriever.retrieve(
            query,
            document_id=document_id,
            tree=tree,
            top_k=top_k,
            use_llm=use_llm,
        )
        return json.dumps(
            {
                "action": "retrieve",
                "query": query,
                "results": [m.as_dict() for m in matches],
            },
            default=str,
        )

    def _graph_document_tree_spine_resolve(
        engine: Any, action: str, text: str, artifact_id: str, document_id: str
    ) -> tuple[Sequence[Any], str, str | None]:
        from agent_utilities.knowledge_graph.ingestion.evidence_spine import (
            artifact_id_for,
            fragment_markdown,
            load_fragments,
        )

        if text:
            # Inline text is fragmented deterministically — the same call the
            # ingest path makes, so an inline preview and a stored spine
            # agree on every address.
            resolved_artifact = artifact_id or artifact_id_for(
                "inline", "", document_id or "inline-document"
            )
            fragments = fragment_markdown(text, artifact_id=resolved_artifact)
            return fragments, resolved_artifact, None
        if artifact_id or document_id:
            fragments = load_fragments(
                engine, artifact_id=artifact_id, document_id=document_id
            )
            resolved_artifact = artifact_id or (
                fragments[0].artifact_id if fragments else ""
            )
            return fragments, resolved_artifact, None
        return (
            [],
            "",
            json.dumps(
                {"error": f"{action} requires 'text', 'artifact_id' or 'document_id'"}
            ),
        )

    def _graph_document_tree_spine_cite(
        fragments: Sequence[Any],
        resolved_artifact: str,
        fragment_id: str,
        content_hash: str,
    ) -> str:
        from agent_utilities.knowledge_graph.ingestion.evidence_spine import (
            citation_status,
        )

        if not (fragment_id or content_hash):
            return json.dumps(
                {"error": "cite requires 'fragment_id' and/or 'content_hash'"}
            )
        return json.dumps(
            {
                "action": "cite",
                "artifact_id": resolved_artifact,
                **citation_status(
                    fragments, fragment_id=fragment_id, content_hash=content_hash
                ),
            },
            default=str,
        )

    def _graph_document_tree_spine_fragments(
        fragments: Sequence[Any], resolved_artifact: str, document_id: str, kinds: str
    ) -> str:
        wanted = {k.strip() for k in kinds.split(",") if k.strip()}
        selected = [f for f in fragments if not wanted or f.kind in wanted]
        return json.dumps(
            {
                "action": "fragments",
                "artifact_id": resolved_artifact,
                "document_id": document_id,
                "fragment_count": len(selected),
                "fragments": [
                    {
                        "fragment_id": f.fragment_id,
                        "address": f.address,
                        "kind": f.kind,
                        "content_hash": f.content_hash,
                        "version_id": f.version_id,
                        "sequence": f.sequence,
                        "ordinal": f.ordinal,
                        "depth": f.depth,
                        "parent_fragment_id": f.parent_fragment_id,
                        "char_start": f.char_start,
                        "char_end": f.char_end,
                        "text": f.text,
                    }
                    for f in selected
                ],
            },
            default=str,
        )

    def _graph_document_tree_spine(
        action: str,
        engine: Any,
        text: str,
        artifact_id: str,
        document_id: str,
        fragment_id: str,
        content_hash: str,
        kinds: str,
    ) -> str:
        # ── evidence spine (CONCEPT:AU-KG.ingest.stable-fragment-address) ──
        # The citation surface: 'fragments' lists what can be cited, 'cite'
        # answers whether an existing citation still holds. Both reach the
        # same core the ingest path writes, so the REST twin
        # (/graph/document-tree) gets them with no second implementation.
        fragments, resolved_artifact, error_response = (
            _graph_document_tree_spine_resolve(
                engine, action, text, artifact_id, document_id
            )
        )
        if error_response is not None:
            return error_response
        if action == "cite":
            return _graph_document_tree_spine_cite(
                fragments, resolved_artifact, fragment_id, content_hash
            )
        return _graph_document_tree_spine_fragments(
            fragments, resolved_artifact, document_id, kinds
        )

    # ══════════════════════════════════════════════════════════════════
    # graph_document_tree — CONCEPT:AU-KG.retrieval.section-tree +
    # CONCEPT:AU-KG.retrieval.tree-navigation. PageIndex-style map-then-fetch over
    # a document's section tree: build the tree, view the text-free structure,
    # fetch cited char/page ranges, or navigate the tree by relevance. Complements
    # (does not replace) graph_search's vector/community retrieval — route long
    # single-document queries here where "similar != relevant".
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_document_tree",
        description=(
            "Reasoning-tree (vectorless) document retrieval over a per-document "
            "section tree (CONCEPT:AU-KG.retrieval.section-tree/tree-navigation; "
            "distills PageIndex). action: 'build' (build + optionally persist the "
            "section tree from text or a stored document), 'structure' (return the "
            "text-free table-of-contents map = get_document_structure), 'content' "
            "(fetch section bodies for cited char ranges like '96..208,300..420' = "
            "get_page_content), 'retrieve' (walk the tree by relevance and return "
            "sections with cited start..end ranges), 'fragments' (the addressable "
            "evidence spine — every citable Fragment of an artifact/document with "
            "its stable address + content hash; CONCEPT:AU-KG.ingest.stable-fragment-address), "
            "'cite' (resolve a stored citation against the current artifact and "
            "report current | stale | moved | lost). Complements graph_search for "
            "long single documents where similar != relevant."
        ),
        tags=["graph-os", "retrieval", "document", "tree"],
    )
    def graph_document_tree(
        action: str = Field(
            description="build | structure | content | retrieve | fragments | cite",
        ),
        document_id: str = Field(
            default="",
            description="Target Document id (structure/content, and build/retrieve when loading from the graph).",
        ),
        text: str = Field(
            default="",
            description="action=build/retrieve — inline document text (markdown) to build the tree from, instead of loading a stored document.",
        ),
        query: str = Field(
            default="",
            description="action=retrieve — the natural-language query to navigate the tree with.",
        ),
        ranges: str = Field(
            default="",
            description="action=content — char ranges to fetch, e.g. '96..208,300..420'.",
        ),
        top_k: int = Field(
            default=5, description="action=retrieve — max sections to return."
        ),
        persist: bool = Field(
            default=True,
            description="action=build — write the Section nodes/edges to the graph (requires document_id).",
        ),
        thin: bool = Field(
            default=True,
            description="action=build — collapse tiny sections into their parent (token-budget thinning).",
        ),
        summarize: bool = Field(
            default=True,
            description="action=build — compute per-node summaries for a text-free structure map.",
        ),
        use_llm: bool = Field(
            default=False,
            description="action=retrieve — try LLM tree navigation before the lexical walk.",
        ),
        artifact_id: str = Field(
            default="",
            description="action=fragments/cite — the source Artifact to read the evidence spine of (alternative to document_id).",
        ),
        fragment_id: str = Field(
            default="",
            description="action=cite — the stable fragment address the citation stored.",
        ),
        content_hash: str = Field(
            default="",
            description="action=cite — the 'sha256:<hex>' content hash the citation stored, used to detect staleness and to relocate a moved fragment.",
        ),
        kinds: str = Field(
            default="",
            description="action=fragments — comma-separated fragment kinds to keep (heading,paragraph,table,table_row,list_item,code_block,quote).",
        ),
    ) -> str:
        """Map-then-fetch + tree-navigation retrieval over a document section tree."""
        engine = kg_server._get_engine()

        if action == "build":
            return _graph_document_tree_build(
                document_id, text, persist, thin, summarize, engine
            )
        if action == "structure":
            return _graph_document_tree_structure(document_id, engine)
        if action == "content":
            return _graph_document_tree_content(document_id, ranges, engine)
        if action == "retrieve":
            return _graph_document_tree_retrieve(
                query, text, document_id, top_k, use_llm, engine
            )
        if action in {"fragments", "cite"}:
            return _graph_document_tree_spine(
                action,
                engine,
                text,
                artifact_id,
                document_id,
                fragment_id,
                content_hash,
                kinds,
            )

        return json.dumps(
            {
                "error": f"unknown action '{action}' "
                "(expected build|structure|content|retrieve|fragments|cite)"
            }
        )

    kg_server.REGISTERED_TOOLS["graph_document_tree"] = graph_document_tree
