from __future__ import annotations

"""Query and search mixin for IntelligenceGraphEngine.

Extracted from engine.py for maintainability. Contains all read-only
query, search, and retrieval methods.
"""


import typing

if typing.TYPE_CHECKING:
    from .._engine_protocol import _EngineProtocol
    from ..core.session import GraphSession

    # D-CDX-71: a plain `_Base = _EngineProtocol` variable assignment is not
    # valid as a mypy base-class expression once the defining module is
    # outside the checked import graph (e.g. an isolated
    # `--follow-imports=skip` run on this file alone) -- mypy cannot resolve
    # what the variable's *value* names as a class in that mode, even though
    # a real class statement referencing the same name resolves fine either
    # way. Declaring an actual TYPE_CHECKING-only class named `_Base` (rather
    # than assigning a variable) gives mypy a structural definition it can
    # always see, so both the isolated and the fully-configured (import
    # following) runs agree. This branch never executes at runtime.
    class _Base(_EngineProtocol):
        pass
else:
    _Base = object


import logging
import re
from typing import Any

from ...models.knowledge_graph import RegistryNodeType
from ..retrieval.iterative_expansion import IterativeQueryExpander
from ..retrieval.score_gate import score_gate
from ..retrieval.temporal_semantic_id import TemporalSemanticIdEncoder

logger = logging.getLogger(__name__)

# CONCEPT:AU-KG.backend.schedule-on-control-graph — current control-plane labels.
# WorkItem is the sole writable work-state authority. Schedule is desired state,
# and ProfileSpan is immutable operational evidence. Loop definitions remain
# content-plane concepts; their execution lifecycle lives exclusively on the
# corresponding WorkItem.
_CONTROL_PLANE_LABELS = frozenset({"workitem", "schedule", "profilespan"})
# Node-pattern label extractor: matches the label after a ':' inside a
# ``(var:Label`` / ``(:Label`` pattern (Cypher node patterns only).
_NODE_LABEL_RE = re.compile(r"\(\s*\w*\s*:\s*([A-Za-z_][A-Za-z0-9_]*)")


def _is_control_plane_query(query: str) -> bool:
    """True when ``query`` references ONLY control-plane node labels (KG-2.148).

    Returns ``False`` for any query that references a content label, or that
    references no node label at all (e.g. a bare ``MATCH (n) WHERE n.id = …``),
    so a content read is never misrouted to the control graph. Returns ``True``
    only when at least one node label is present and every referenced node label
    is in :data:`_CONTROL_PLANE_LABELS`.
    """
    labels = _NODE_LABEL_RE.findall(query or "")
    if not labels:
        return False
    return all(lbl.lower() in _CONTROL_PLANE_LABELS for lbl in labels)


# CONCEPT:AU-KG.query.query-aggregation — Cypher aggregate functions. A query that
# calls one of these in a projection collapses many rows into per-group rows that
# carry no per-row node id to govern (an aggregate/scalar result is not a node
# projection). Canonical home for this detection: the MCP `graph_query` federation
# router (`mcp/tools/query_tools.is_aggregation_cypher`, re-exported from here) uses
# it to decide fan-out eligibility; `QueryMixin.query_cypher` below uses it to route
# an aggregate read around the per-row ACL post-filter, which requires an
# identifiable node per row and would otherwise deny EVERY aggregate/scalar read
# outright (there is no id to check) regardless of caller privilege.
_AGG_FUNCS = (
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "collect",
    "stdev",
    "stdevp",
    "percentilecont",
    "percentiledisc",
    "variance",
)
_AGG_CALL_RE = re.compile(r"\b(?:" + "|".join(_AGG_FUNCS) + r")\s*\(", re.IGNORECASE)


def is_aggregation_cypher(cypher: str) -> bool:
    """True when ``cypher`` projects an aggregate (CONCEPT:AU-KG.query.query-aggregation).

    Strips quoted string literals first so a literal like ``'count(*)'`` or a
    property named ``max_depth`` is not misread as an aggregate call.
    """
    no_literals = re.sub(r"'[^']*'|\"[^\"]*\"", "", cypher or "")
    return bool(_AGG_CALL_RE.search(no_literals))


def _describe_secured_read_failure(exc: BaseException) -> str:
    """Render a secured-read failure WITH its real chained cause(s), untruncated.

    ``secured_reads.visible()``, ``filter_rows()`` (and the ``row_node_ids()`` it
    calls), and ``audit_read()`` each wrap their OWN internal failure into a
    distinctly-worded :class:`PermissionError` (e.g. "Graph result contains a row
    without a governed node id", "Row visibility evaluation failed", "Read audit
    recording failed") before it reaches :meth:`QueryMixin.query_cypher`'s outer
    handler — which re-wraps AGAIN into the single generic "Graph row-policy or
    audit enforcement failed" every caller sees. That is the correct behavior for
    the PUBLIC/caller-facing surface (exception internals must not leak), but
    collapsing it identically in the SERVER LOG too made every distinct failure —
    a denied scope, a missing node id, a bad durable ACL record, an audit
    infrastructure failure — indistinguishable noise server-side as well
    (the documented swallowed-error anti-pattern). Mirrors
    :func:`agent_utilities.knowledge_graph.retrieval.hybrid_retriever._describe_embedding_failure`,
    extended to walk the FULL cause chain (this boundary wraps up to three deep:
    e.g. ``permit()``'s own wrap over ``_hydrate_missing_acls``'s over a raw
    ``ValueError``), not just one level.
    """
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = exc.__cause__
    seen: set[int] = {id(exc)}
    while cause is not None and id(cause) not in seen and len(parts) < 6:
        parts.append(f"{type(cause).__name__}: {cause}")
        seen.add(id(cause))
        cause = cause.__cause__
    return " <- caused by ".join(parts)


class QueryMixin(_Base):
    """Query and search capabilities for the KG engine."""

    def query_cypher(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        clearance_level: int = 999,
        as_of: str | None = None,
        *,
        session: GraphSession | None = None,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query against the persistent Graph store.

        Enterprise ABAC/RBAC Middleware Interceptor:
        Queries are automatically scoped by the user's/agent's security clearance.
        Nodes with a requiresClassification > clearance_level are omitted.

        CONCEPT:AU-KG.temporal.bi-temporal-memory-layers — when ``as_of`` (an ISO-8601 instant) is supplied, result rows are
        post-filtered to those whose bi-temporal validity interval
        (``valid_from <= as_of < valid_to``) contains that instant. Rows without temporal
        metadata pass through unchanged. This answers "what was true as of date T" — something
        Quarq's flat storage-ordered files cannot do.

        Args:
            session: Explicit :class:`~agent_utilities.knowledge_graph.core.session.GraphSession`
                this read runs under (CONCEPT:AU-P0-1 — the one currency: identity + policy +
                trace in one object). When omitted, one is derived from today's
                ambient actor via ``GraphSession.from_ambient()`` so existing
                callers are unaffected.
            include_epistemic: Opt-in (CONCEPT:AU-KB-CURRENCY, Seam 1 — the
                ``KnowledgeBatch`` currency, extended to the MCP-facing Cypher
                surface). Default ``False`` — byte-for-byte the same
                ``list[dict]`` rows as before this parameter existed. When
                ``True``, the SAME rows (after ``as_of``/visibility filtering)
                are currency-upgraded via :func:`~agent_utilities.knowledge_graph
                .core.epistemic_row.attach_epistemic_rows` into
                :class:`~agent_utilities.knowledge_graph.core.epistemic_row.EpistemicRow`
                results — confidence, bitemporal valid/tx window, evidence
                provenance, policy labels — resolved server-side via the read
                backend's own ``explain_provenance_by_ids``, never fabricated.
                Degrades to ``[]`` (never raises) when the backend has no such
                primitive, mirroring :meth:`KnowledgeGraph.query`'s documented
                contract.
        """
        if params is None:
            params = {}

        from agent_utilities.knowledge_graph.core.session import resolve_session

        session = resolve_session(session, required_scope="kg:read")

        # Inject clearance level for the backend parser to utilize
        params["_clearance_level"] = clearance_level
        logger.debug(
            "RBAC Interceptor: Executing query with clearance level %d", clearance_level
        )

        # An aggregate/scalar projection (count/sum/avg/collect/...) collapses many
        # rows into one — the result carries no per-row node id, so the per-row ACL
        # post-filter below (which needs one) cannot apply to it. Detected once,
        # up front, from the CALLER'S query text (aggregate-ness is a property of
        # what was asked, unaffected by scope/visibility injection below).
        aggregate_query = is_aggregation_cypher(query)

        # Tenant scoping + owner/scope visibility on the MCP/orchestration read
        # chokepoint (CONCEPT:AU-KG.query.skip-assimilated-papers + KG-2.60). Mandatory
        # This isolates ``graph_query`` on a shared backend graph (the named-graph
        # physical partition only applies in sharded/direct-engine paths).
        try:
            from agent_utilities.knowledge_graph.core.secured_reads import scope

            # D-W2T-2: `scope()` now returns (query, extra_params) — the tenant
            # id is injected as a bound `$_tenant_scope_id` parameter, not a
            # string-literal splice. Merge it into this call's own params dict
            # (same convention as the `_clearance_level` system param above).
            scoped_query, tenant_params = scope(query, session.actor)
            params.update(tenant_params)
            if aggregate_query:
                # CONCEPT:AU-KG.compute.data-is-private-its — an aggregate has no row
                # to post-filter by owner/scope (below), so that boundary is pushed
                # INTO the query text instead: the aggregate is computed only over
                # rows the actor's owner/scope would have kept anyway. This is a
                # no-op (returns the query unchanged) for a privileged actor, exactly
                # mirroring `secured_reads.visible`'s own privileged bypass.
                #
                # D-SH-4 (reports/deferred/lane-skill-harvest.md): `apply_visibility`
                # defaults to writing its predicate against a hardcoded `n` variable.
                # Detect THIS query's own primary bound variable (the same detector
                # `scope()`/`scope_cypher_query` now uses) and pass it explicitly —
                # a query aliased as e.g. `MATCH (w:WorkItem) RETURN count(w) AS c`
                # otherwise gets a visibility predicate referencing a variable that
                # doesn't exist, which Cypher treats as never matching (a silent
                # zero-row/zero-count aggregate instead of the real answer). A fully
                # anonymous first pattern has no variable to scope by; skip the
                # injection rather than reference a fabricated name (mirrors
                # `scope_cypher_query`'s own fail-open-to-unscoped decision for the
                # same case).
                from agent_utilities.knowledge_graph.core.cypher_scope_vars import (
                    primary_bound_variable,
                )
                from agent_utilities.knowledge_graph.core.tenant_sharing import (
                    apply_visibility,
                )

                agg_var = primary_bound_variable(query)
                if agg_var is not None:
                    # D-W2T-2: same (query, extra_params) contract as scope()
                    # above — the visibility owner id is a bound
                    # `$_visibility_owner_id` parameter now.
                    scoped_query, vis_params = apply_visibility(
                        scoped_query, session.actor, var=agg_var
                    )
                    params.update(vis_params)

                    # GOC-61 phase 1 (2026-08-09 owner ruling, read/write
                    # split): the commons catalog READ restriction, injected
                    # the same way and for the same reason — this is the
                    # ONLY place the restriction can reach an aggregate
                    # projection (count(n), etc.), since it has no per-row
                    # properties for filter_commons_catalog to post-filter.
                    # A no-op unless this engine is bound to the commons
                    # graph and the actor is unprivileged (see
                    # apply_commons_catalog_restriction's own docstring).
                    from agent_utilities.knowledge_graph.core.tenant_sharing import (
                        apply_commons_catalog_restriction,
                    )

                    graph_name = getattr(
                        getattr(self, "graph_compute", None), "graph_name", None
                    )
                    scoped_query, catalog_params = apply_commons_catalog_restriction(
                        scoped_query, session.actor, graph_name, var=agg_var
                    )
                    params.update(catalog_params)
        except PermissionError as exc:
            # A genuine, already-typed fail-closed scoping/authorization
            # decision reached here — e.g. `cypher_scoping.UnscopableQueryError`
            # (the query binds no node variable a tenant/visibility predicate
            # can be written against) or `secured_reads.scope()`'s own
            # verified-actor/infrastructure denial. Propagate the SAME
            # exception: its specific type and message are the only thing
            # that lets an operator tell "this query's shape can't be safely
            # scoped" apart from "this actor lacks permission" apart from any
            # other denial. Re-wrapping it into one generic "Graph query
            # scoping failed" — the bug this replaces — made a pure
            # query-shape problem (``MATCH ()-[r]->() RETURN count(r)``, which
            # binds no node variable at all) indistinguishable from an actual
            # authorization denial.
            from agent_utilities.core.log_privacy import sanitize_log_text

            logger.error(
                "Graph query scoping denied: %s | query=%s",
                sanitize_log_text(_describe_secured_read_failure(exc)),
                sanitize_log_text(str(query))[:240],
            )
            raise
        except Exception as exc:
            # NOT an authorization decision: an `AttributeError`/`TypeError`/
            # etc. raised INSIDE the scoping pipeline itself (e.g. a
            # caller-supplied actor object missing an expected method) is a
            # CODE DEFECT, not a permission denial. Failing closed is still
            # correct here — `scoped_query`/`params` are never used past this
            # point on this path, so a defect here can never fall through to
            # executing an unscoped read — but labeling it `PermissionError`
            # actively misdirects debugging toward the authorization layer.
            # Reproduced: an actor object missing `ensure_credential_current`
            # alone raised the exact same "Graph query scoping failed" a real
            # denial would.
            from agent_utilities.core.log_privacy import sanitize_log_text
            from agent_utilities.knowledge_graph.core.cypher_scoping import (
                QueryScopingError,
            )

            logger.error(
                "Graph query scoping raised a non-authorization failure "
                "before a scope could be established (code defect, not a "
                "permission denial): %s | query=%s",
                sanitize_log_text(_describe_secured_read_failure(exc)),
                sanitize_log_text(str(query))[:240],
            )
            raise QueryScopingError(
                "Graph query scoping failed due to an internal defect, not "
                "an authorization denial"
            ) from exc

        # CONCEPT:AU-KG.backend.schedule-on-control-graph — control/content read
        # isolation. A query containing only current control-plane labels must
        # use the configured control authority. Missing control authority is a
        # hard configuration error: reading the content graph would return a
        # misleading empty result and silently split operational truth.
        is_control_query = _is_control_plane_query(scoped_query)
        if is_control_query:
            read_backend = getattr(self, "control_backend", None)
            if read_backend is None:
                raise RuntimeError(
                    "Control-plane query requires the configured WorkItem authority"
                )
        else:
            read_backend = self.backend

        if not read_backend:
            raise RuntimeError("The authoritative graph read service is unavailable")
        rows = read_backend.execute_read(scoped_query, params)

        try:
            from agent_utilities.knowledge_graph.core.secured_reads import (
                audit_read,
                filter_rows,
                row_node_ids,
                visible,
            )

            if aggregate_query:
                # No per-row node id exists to ACL-check or audit by id (the rows
                # ARE the aggregate result, e.g. a single {"c": 142} row) — the
                # owner/scope boundary already ran query-side above, and the
                # fine-grained per-node classification ACL (KG-2.46) has no row to
                # apply to. This is a scoped, deliberate trade-off (aggregate reads
                # are governed by tenant + owner/scope, not per-node classification
                # ACL — consistent with this query shape already being denied a
                # cross-graph fan-out elsewhere, CONCEPT:AU-KG.query.query-aggregation),
                # not a silent bypass of tenant/owner-scope isolation. The read is
                # still audited, with an empty node-id list (nothing governable to
                # name).
                audit_read(
                    [],
                    summary="native-cypher-read (aggregate)",
                    actor=session.actor,
                )
            else:
                rows = visible(filter_rows(rows, session.actor), session.actor)

                # GOC-61 phase 1 (2026-08-09 owner ruling, read/write split):
                # the commons catalog READ restriction, Python-side. Runs
                # AFTER visible()/filter_rows() (owner/scope) — this is an
                # ADDITIONAL restriction, not a replacement: owner/scope
                # already hides another tenant's OWNED-private rows; this
                # closes the remaining gap for UNOWNED/commons-scoped
                # operational rows (WorkItem, RuntimeSignal, Concept, ...)
                # that owner/scope alone treats as visible to everyone. A
                # no-op unless this engine is bound to the commons graph and
                # the actor is unprivileged.
                from agent_utilities.knowledge_graph.core.tenant_sharing import (
                    filter_commons_catalog,
                )

                graph_name = getattr(
                    getattr(self, "graph_compute", None), "graph_name", None
                )
                rows = filter_commons_catalog(rows, session.actor, graph_name)

                # The engine also emits its protocol audit. This service-level
                # record proves the guarded GraphSession/query boundary ran without
                # persisting raw query text or parameters.
                audit_read(
                    row_node_ids(rows),
                    summary="native-cypher-read",
                    actor=session.actor,
                )
        except Exception as exc:
            # Surface the TRUE failing step server-side before collapsing it to the
            # generic caller-facing message — this exact boundary was a documented
            # repeat instance of the swallowed-cause anti-pattern (a bare
            # ``raise PermissionError(...) from exc`` with nothing logged first).
            # The public PermissionError message is unchanged; only the log gains
            # detail, and it is still sanitized (endpoints/paths/emails redacted)
            # before it reaches any handler.
            from agent_utilities.core.log_privacy import sanitize_log_text

            # Include the (sanitized, truncated) QUERY too: the failing site is
            # otherwise unidentifiable from the log alone — a bare "row without a
            # governed node id" forced a full call-graph trace to find which
            # caller's RETURN dropped the id. The query text names it immediately.
            logger.error(
                "Graph row-policy or audit enforcement failed: %s | query=%s",
                sanitize_log_text(_describe_secured_read_failure(exc)),
                sanitize_log_text(str(query))[:240],
            )
            raise PermissionError(
                "Graph row-policy or audit enforcement failed"
            ) from exc

        if as_of:
            from agent_utilities.knowledge_graph.core.bitemporal import filter_as_of

            rows = filter_as_of(rows, as_of)
        if not include_epistemic:
            return rows

        from agent_utilities.knowledge_graph.core.epistemic_row import (
            attach_epistemic_rows,
        )

        graph = getattr(read_backend, "graph", None)
        fetch = getattr(graph, "explain_provenance_by_ids", None)
        return attach_epistemic_rows(rows, fetch)  # type: ignore[return-value]

    def sql(self, query: str) -> list[dict[str, Any]]:
        """Run read-only SQL over the KG via the engine's DataFusion surface (KG-2.243).

        ``SELECT ... FROM nodes WHERE ... LIMIT ...`` is parsed/planned/executed by
        the same engine path ``Method::Sql`` and the pg-wire listener use
        (``eg_query::exec_sql_typed`` over an off-lock ``GraphView``). This makes
        SQL-on-the-KG reachable natively over the graph-os MCP ``graph_query`` tool
        (``scope='sql'``) and its REST twin ``/graph/query`` — not just over pg-wire
        (CONCEPT:AU-KG.query.raw-python) or a raw Python ``client.query.sql()``.

        Read-path-first: only ``SELECT``/``WITH``/``EXPLAIN`` statements are accepted;
        mutations must go through ``kg_write`` so they get the engine's governed write
        path. RLS is enforced engine-side (the off-lock ``GraphView`` honours the
        actor's row-level filter), exactly as the pg-wire surface does.

        The underlying engine client is reached through the active backend's
        ``GraphComputeEngine`` (``backend.graph._client.query.sql``). A backend with no
        engine client (e.g. a pure-Postgres mirror) raises a clear error rather than
        silently returning nothing.
        """
        stripped = (query or "").lstrip()
        head = stripped[:8].upper()
        if not (
            head.startswith("SELECT")
            or head.startswith("WITH")
            or head.startswith("EXPLAIN")
        ):
            raise ValueError(
                "sql() is read-only: only SELECT/WITH/EXPLAIN are allowed "
                "(use kg_write for mutations)."
            )

        graph = getattr(self.backend, "graph", None)
        client = getattr(graph, "_client", None)
        query_ns = getattr(client, "query", None)
        sql_fn = getattr(query_ns, "sql", None)
        if not callable(sql_fn):
            raise RuntimeError(
                "The active backend has no epistemic-graph SQL surface; "
                "SQL-on-the-KG requires the engine backend (build with the "
                "'query' feature)."
            )
        rows = sql_fn(query)
        try:
            from agent_utilities.knowledge_graph.core.secured_reads import (
                filter_rows,
                visible,
            )

            rows = visible(filter_rows(rows))
        except Exception as exc:  # pragma: no cover - never break a read  # noqa: BLE001 — row-visibility filtering is a defense-in-depth layer over `rows`, which is already the query result being returned either way (filtered or not) — 'never break a read' per the comment above
            logger.debug("sql() row filtering skipped: %s", exc)
        return rows

    def sparql(
        self,
        query: str,
        base_iri: str = "",
        type_convention: str = "",
    ) -> list[dict[str, Any]]:
        """Run a SPARQL 1.1 query over the KG via the engine's native RDF surface (KG-2.266).

        Makes engine-native SPARQL (``SELECT``/``ASK``/``CONSTRUCT``/``DESCRIBE``)
        reachable over the graph-os MCP ``graph_query`` tool (``scope='sparql'``) and
        its REST twin ``/graph/query`` — the SPARQL sibling of :meth:`sql`. Routes to
        the active backend's ``GraphComputeEngine.sparql`` (``backend.graph.sparql`` →
        the engine's ``client.rdf.sparql`` wire op), which queries the engine's RDF
        projection of the live property graph (NOT a rdflib materialization). Returns
        one dict per result row keyed by the projected variable (an ``ASK`` returns a
        single ``{"boolean": ...}`` row, per the engine's projection).

        A backend with no engine RDF surface (e.g. a server built without the
        ``sparql`` feature, or a pure-Postgres mirror) raises a clear error rather than
        silently returning nothing — symmetric with :meth:`sql`.
        """
        graph = getattr(self.backend, "graph", None)
        sparql_fn = getattr(graph, "sparql", None)
        if not callable(sparql_fn):
            raise RuntimeError(
                "The active backend has no epistemic-graph SPARQL surface; "
                "SPARQL-on-the-KG requires the engine backend (build with the "
                "'sparql' feature)."
            )
        rows = sparql_fn(query, base_iri, type_convention)
        try:
            from agent_utilities.knowledge_graph.core.secured_reads import (
                filter_rows,
                visible,
            )

            rows = visible(filter_rows(rows))
        except Exception as exc:  # pragma: no cover - never break a read  # noqa: BLE001 — row-visibility filtering for sparql() — same 'never break a read' defense-in-depth layer as sql() above
            logger.debug("sparql() row filtering skipped: %s", exc)
        return rows

    def uql(
        self, query: str, *, include_epistemic: bool = False
    ) -> list[dict[str, Any]]:
        """Run a UQL text query over the KG via the engine's unified surface (KG-2.214).

        CONCEPT:AU-KG.query.au-engine-execution-path — the AU→engine execution path for the engine's native
        cross-modal query language. ``MATCH (:Label) [WHERE ...] |> TRAVERSE ... |>
        RANK BY ~[...] |> LIMIT k`` is PARSED by the engine into the SAME cross-modal
        ``Plan`` the ``UnifiedQuery`` path carries, then run through the IDENTICAL fused
        executor (``eg_plan::uql::parse`` → filter/traverse/rank over one off-lock
        snapshot). This is the surface the NL→query planner (CONCEPT:AU-KG.query.au-engine-execution-path) submits a
        fleet-LLM-generated UQL string to, so NL→query runs through the engine's
        deterministic pipeline exactly like the engine's own ``NlPlanner`` seam (EG-078).

        Routes to the active backend's engine client (``backend.graph._client.query.uql``
        → the engine's ``UnifiedQueryText`` wire op). Returns one ``{"id", "score"}`` row
        per result, in the plan's final (post-``RANK``) order. A backend with no engine
        UQL surface (server built without the ``query`` feature, or a pure-Postgres
        mirror) raises a clear error rather than silently returning nothing — symmetric
        with :meth:`sql` / :meth:`sparql`.

        Args:
            include_epistemic: Opt-in (CONCEPT:AU-KB-CURRENCY, Seam 1 — the
                ``KnowledgeBatch`` currency, extended to this cross-modal surface).
                Default ``False`` — byte-for-byte the same ``[{"id", "score"}]``
                rows. When ``True``, currency-upgrades the SAME ids (SAME order)
                via the engine's ``explain_provenance_by_ids`` into
                :class:`~agent_utilities.knowledge_graph.core.epistemic_row.EpistemicRow`
                results. Degrades to ``[]`` (never raises) when the backend's
                ``GraphComputeEngine`` has no ``explain_provenance_by_ids`` —
                mirroring :meth:`KnowledgeGraph.query`'s documented contract.
        """
        graph = getattr(self.backend, "graph", None)
        client = getattr(graph, "_client", None)
        query_ns = getattr(client, "query", None)
        uql_fn = getattr(query_ns, "uql", None)
        if not callable(uql_fn):
            raise RuntimeError(
                "The active backend has no epistemic-graph UQL surface; "
                "UQL-on-the-KG requires the engine backend (build with the "
                "'query' feature)."
            )
        rows = list(uql_fn(query) or [])
        fetch = getattr(graph, "explain_provenance_by_ids", None)
        if include_epistemic:
            from agent_utilities.knowledge_graph.core.epistemic_row import (
                attach_epistemic_rows,
            )

            return attach_epistemic_rows(rows, fetch)  # type: ignore[return-value]
        # Light epistemic layer (CONCEPT:AU-KB-CURRENCY, Native by default) —
        # see `KnowledgeGraph.query`'s identical wiring for the full rationale.
        from agent_utilities.core.config import config as _app_config
        from agent_utilities.knowledge_graph.core.epistemic_row import (
            attach_epistemic_columns,
            should_attach_epistemic_columns,
        )

        if should_attach_epistemic_columns(
            rows, default=_app_config.epistemic_light_default
        ):
            rows = attach_epistemic_columns(rows, fetch)
        return rows

    def resolve_temporal_contradiction(
        self,
        fact_a_id: str,
        fact_b_id: str,
        fact_a_props: dict[str, Any],
        fact_b_props: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve two contradicting facts by event-time precedence (CONCEPT:AU-KG.temporal.bi-temporal-memory-layers).

        The later ``event_time`` wins; a ``SUPERSEDES`` edge is written winner→loser and the
        loser's ``valid_to`` is closed at the winner's ``event_time`` (the fact is never
        deleted, preserving as-of history). Implements Quarq's prompt-only "newer supersedes
        older" rule (agent-oss/agent.py:2462-2468) as a structural graph mutation.

        Returns a summary dict ``{"winner": id, "loser": id, "valid_to": boundary}``.
        """
        from agent_utilities.knowledge_graph.core.bitemporal import (
            resolve_precedence,
            supersede,
        )

        a = {**fact_a_props, "id": fact_a_id}
        b = {**fact_b_props, "id": fact_b_id}
        winner, loser = resolve_precedence(a, b)
        supersede(winner, loser)
        # Persist: SUPERSEDES edge + close the loser's validity interval.
        try:
            self.link_nodes(
                winner["id"],
                loser["id"],
                "SUPERSEDES",
                properties={"event_time": winner.get("event_time")},
            )
            if self.backend:
                self.backend.execute(
                    "MATCH (n) WHERE n.id = $id SET n.valid_to = $vt",
                    {"id": loser["id"], "vt": loser.get("valid_to")},
                )
        except Exception as e:  # pragma: no cover - persistence is best-effort
            logger.warning("Temporal contradiction persistence failed: %s", e)
        return {
            "winner": winner["id"],
            "loser": loser["id"],
            "valid_to": loser.get("valid_to"),
        }

    def _search_keyword(
        self, query: str, top_k: int = 10, clearance_level: int = 999
    ) -> list[dict[str, Any]]:
        """Perform a multi-faceted search across code, agents, and memory using keywords."""
        results = []
        query_lower = query.lower()

        # Prepare keywords
        keywords = [k.strip() for k in query_lower.split() if len(k.strip()) > 1]
        if not keywords:
            keywords = [query_lower]

        # 1. Engine-scalable keyword leg.
        # The engine's `discover` ranks keyword overlap (name/description/type)
        # server-side in ONE round-trip and returns the top-k, instead of the O(N)
        # full-node scan the `MATCH (n) WHERE … CONTAINS` path below degrades to on the
        # engine (which does not evaluate the WHERE server-side and returns every node).
        # Per the dependency edict, keyword/vector ranking belongs on the engine, never
        # an O(N) Python loop. Hydrate the hits for RBAC + soft-delete enforcement
        # (discover returns only id/name/description/type/score) in ONE round-trip —
        # a per-hit read here was O(N) serialized point-reads that dominated retrieval
        # latency under engine contention (blew the delegation's time budget).
        disc = getattr(self.graph, "discover", None)
        if callable(disc):
            try:
                hits = [
                    item
                    for item in (disc(keywords, [], max(top_k * 4, top_k)) or [])
                    if isinstance(item, dict) and str(item.get("id", ""))
                ]
                hydrated = self.graph._get_node_properties_batch(
                    [str(item["id"]) for item in hits]
                )
                for item in hits:
                    node_id = str(item["id"])
                    data = hydrated.get(node_id) or dict(item)
                    data["id"] = node_id
                    # ``discover`` server-side ranks each hit under "score";
                    # hydration replaces ``data`` with the raw node-property
                    # dict, which drops it. Carry it forward as "_score" (the
                    # convention every other retrieval arm in
                    # ``HybridRetriever.retrieve_hybrid`` uses) so a caller that
                    # runs quality-gated retrieval over this keyword-only path
                    # (no embedder configured/reachable) doesn't read every hit
                    # as an un-scored 0.0 and always fail LOW_RELEVANCE_TOPK.
                    if "_score" not in data and "score" in item:
                        data["_score"] = item["score"]
                        # D-GS27-6/D-EMB-6: this is the engine-native `discover`
                        # keyword-overlap score, NOT a cosine similarity — the
                        # same category error _lexical_fallback's flat 0.2
                        # sentinel had. Tagged so RetrievalQualityGate grades it
                        # against its own separately-calibrated
                        # _keyword_discover_threshold (retrieval_quality.py's
                        # _result_threshold) instead of the vector-calibrated
                        # default, which previously always failed it (e.g.
                        # composite=0.02) regardless of actual keyword relevance.
                        data.setdefault("_fallback", "keyword_discover")
                    req_class = data.get("requiresClassification", 0)
                    if isinstance(req_class, int) and req_class > clearance_level:
                        continue
                    if str(data.get("status", "")).upper() == "ARCHIVED":
                        continue
                    results.append(data)
            except Exception as e:  # noqa: BLE001 — one candidate source (engine-native discover) inside a multi-source keyword search; results simply doesn't get this source's contribution, and the function already falls through to the backend/GCE-scan fallbacks below
                logger.debug(f"engine discover keyword search unavailable: {e}")

        # 2. Fallback — a backend that DOES evaluate a Cypher WHERE server-side
        # (pg-age / neo4j mirror). Only when the engine discover path found nothing.
        if not results and self.backend:
            # Simple keyword search across all nodes in backend
            q = []
            params = {}
            for i, k in enumerate(keywords):
                q.append(
                    f"(toLower(n.name) CONTAINS $k{i} OR toLower(n.description) CONTAINS $k{i} OR toLower(n.id) CONTAINS $k{i})"
                )
                params[f"k{i}"] = k

            where_clause = " OR ".join(q)
            if where_clause:
                query_str = f"MATCH (n) WHERE ({where_clause}) AND coalesce(n.status, '') <> 'ARCHIVED' RETURN n"
                try:
                    res = self.backend.execute(query_str, params)
                    for row in res:
                        if not isinstance(row, dict):
                            continue
                        # Handle both the Cypher `RETURN n` wrapping ({"n": {...}})
                        # and flat-dict backends (e.g. EpistemicGraphBackend).
                        node = row.get("n", row)
                        if not isinstance(node, dict) or not node:
                            continue
                        # Filter client-side: some backends (in-memory
                        # EpistemicGraph) do not evaluate the WHERE clause and
                        # return every node, so re-apply the keyword match here.
                        name = str(node.get("name", "")).lower()
                        desc = str(node.get("description", "")).lower()
                        nid = str(node.get("id", "")).lower()
                        if not any(
                            k in nid or k in name or k in desc for k in keywords
                        ):
                            continue
                        # Soft-delete enforcement (backend-agnostic): in-memory /
                        # service backends may not evaluate the WHERE clause, so
                        # re-apply the ARCHIVED exclusion here too (matches the
                        # GCE fallback path and the Cypher intent).
                        if str(node.get("status", "")).upper() == "ARCHIVED":
                            continue
                        req_class = node.get("requiresClassification", 0)
                        if isinstance(req_class, int) and req_class > clearance_level:
                            continue
                        results.append(node)
                except Exception as e:  # noqa: BLE001 — one candidate source (Cypher-evaluating backend) inside the same multi-source search; the function already falls through to the last-resort GCE scan below when results is still empty
                    logger.debug(f"Backend keyword search failed: {e}")

        # 3. Last-resort O(N) GCE scan for name/ID matches — only when neither the
        # engine discover nor a Cypher-evaluating backend returned anything. Every
        # node's type must be inspected to match, so this can't skip hydration —
        # batch-hydrate the WHOLE id list in ONE round-trip (CONCEPT:AU-KG.retrieval.batch-hydrate)
        # rather than one `_get_node_properties` point-read per node scanned.
        if not results:
            all_node_ids = self.graph.node_ids()
            hydrated_all = self.graph._get_node_properties_batch(all_node_ids)
            for node_id in all_node_ids:
                data = hydrated_all.get(str(node_id), {})
                # RBAC Enforcement Fallback
                req_class = data.get("requiresClassification", 0)
                if isinstance(req_class, int) and req_class > clearance_level:
                    continue

                # Soft-delete enforcement
                if str(data.get("status", "")).upper() == "ARCHIVED":
                    continue

                name = str(data.get("name", "")).lower()
                desc = str(data.get("description", "")).lower()
                nid = str(node_id).lower()

                # Match if any keyword is in name, desc, or ID
                if any(k in nid or k in name or k in desc for k in keywords):
                    result = dict(data)
                    result["id"] = node_id
                    results.append(result)

        logger.debug(f"Search hybrid for '{query}' found {len(results)} nodes.")

        # D-ORC-8/Exit-C: every result this method returns is keyword-only —
        # the engine-native ``discover`` leg above passes an EMPTY
        # ``query_embedding`` (never blends a semantic signal), and the
        # backend/GCE legs below are pure ``CONTAINS``/substring scans. The
        # discover leg's carried-forward "_score" is
        # ``matched_keywords / total_query_keywords`` (a raw overlap
        # FRACTION diluted by every non-matching function word in the
        # query — see epistemic-graph's ``keyword_overlap``), not a cosine
        # similarity; it was never calibrated against — and routinely sits
        # well below — ``RetrievalQualityGate``'s vector-tuned 0.6 default
        # threshold, so callers that quality-gate this method's output
        # (``HybridRetriever.retrieve_hybrid``'s "no semantic matches ->
        # keyword" arm, hit on every query once the graph's embedding
        # coverage is sparse) always failed LOW_RELEVANCE_TOPK regardless of
        # real relevance — the identical defect class already fixed for
        # ``HybridRetriever._lexical_fallback``'s flat 0.2 sentinel. Tag
        # every result the same way so the gate grades it against the lower,
        # keyword-appropriate ``_lexical_threshold`` instead.
        for _r in results:
            _r.setdefault("_fallback", "lexical")

        # Sort by graded relevance, not a coarse boolean. A bare "any keyword in
        # name" predicate cannot distinguish "Agent A" from "Agent B" (both
        # contain "agent"), so the result order would fall back to the backend's
        # arbitrary iteration order. Rank by: (1) whole-query substring match in
        # name, then in id; (2) count of distinct keywords matched in name; then
        # in id/description — so the closest-matching node wins deterministically.
        def _relevance(x: dict[str, Any]) -> tuple:
            name = str(x.get("name", "")).lower()
            desc = str(x.get("description", "")).lower()
            nid = str(x.get("id", "")).lower()
            name_kw = sum(1 for k in keywords if k in name)
            id_desc_kw = sum(1 for k in keywords if k in nid or k in desc)
            return (
                query_lower in name,
                query_lower in nid,
                name_kw,
                id_desc_kw,
            )

        results.sort(key=_relevance, reverse=True)

        return results[:top_k]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        include_archived: bool = False,
        skip_quality_gate: bool = False,
        relevance_threshold: float | None = None,
        target_paths: list[str] | None = None,
        mode: str = "standard",
        self_correct: bool = False,
        corpus_id: str | None = None,
        as_of: str | None = None,
        session: GraphSession | None = None,
    ) -> list[dict[str, Any]]:
        """Perform a multi-faceted search using Hybrid GraphRAG.

        CONCEPT:AU-KG.retrieval.memory-first-retrieval — when ``mode`` is ``"hyde"``/``"deep"`` (or ``self_correct`` is set),
        delegates to the memory-first ``plan_and_retrieve`` policy (HyDE multi-query plan, dual
        thresholds, and an evidence-gated second pass). ``mode="standard"`` preserves the prior
        single-query behavior.

        ``as_of`` (ISO-8601) sets the reference time for pack-driven recency decay,
        enabling knowledge-state-as-of-date-D retrieval (CONCEPT:EG-KG.compute.rust-native-training-loss).

        ``session`` (CONCEPT:AU-KG.retrieval.acl-aware-vector-retrieval): optional
        explicit :class:`~..core.session.GraphSession` this read runs under.
        ``None`` (the default) is a **no-op** — exactly the prior, unfiltered
        behavior, for the many existing internal/test callers with no backing
        ACL infrastructure. Passing a session applies the SAME per-node ACL +
        owner/scope + audit boundary :meth:`query_cypher` already applies to
        raw Cypher reads: the vector/hybrid path previously returned every
        ranked node completely unfiltered even when the caller held a verified
        session, unlike the guarded Cypher path. The served ``graph_search``
        MCP tool passes its ambient session explicitly, so the actual
        externally-reachable surface is governed by default with no operator
        configuration; enforcement never silently no-ops once a session IS
        supplied — an infrastructure failure raises ``PermissionError`` rather
        than falling back to unfiltered results.
        """
        # Lazy-ensure the retriever (CONCEPT:AU-KG.retrieval.memory-first-retrieval): the background-task host and
        # some engine-construction paths can reach search before ``__init__`` wired
        # ``hybrid_retriever``, which made deep_analysis / discover_innovations
        # AttributeError. Build it on first use so the path is always live.
        if getattr(self, "hybrid_retriever", None) is None:
            from ..retrieval.hybrid_retriever import HybridRetriever

            # ``self`` is the QueryMixin composed into the live IntelligenceGraphEngine
            # at runtime; mypy sees only the mixin type.
            self.hybrid_retriever = HybridRetriever(
                self,  # type: ignore[arg-type]
                schema_pack=getattr(self, "active_schema_pack", None),
            )
        results: list[dict[str, Any]]
        if mode in ("hyde", "deep") or self_correct:
            planned = self.hybrid_retriever.plan_and_retrieve(
                query,
                context_window=top_k * 2,
                mode=mode if mode in ("hyde", "standard", "deep") else "hyde",
                self_correct=self_correct,
                corpus_id=corpus_id,
                # GOC-83-W04: thread through so each sub-query's own
                # retrieve_hybrid ACL-filters its raw pool before internal
                # trim, same as the non-hyde/deep branch below. No-op when
                # `session` is None.
                session=session,
            )
            # ``plan_and_retrieve`` may return either the node list or a wrapper
            # dict ({results|nodes: [...]}); normalize to the node list.
            if isinstance(planned, list):
                results = planned
            else:
                results = planned.get("results") or planned.get("nodes") or []
        else:
            results = self.hybrid_retriever.retrieve_hybrid(
                query,
                context_window=top_k * 2,
                skip_quality_gate=skip_quality_gate,
                relevance_threshold=relevance_threshold,
                target_paths=target_paths,
                corpus_id=corpus_id,
                as_of=as_of,
                # GOC-83-W04: thread the session down so `retrieve_hybrid`
                # ACL-filters its OWN raw candidate pools before its internal
                # sort/rerank/trim — closing the crowd-out channel that lives
                # inside the retriever, not just at this method's own return
                # (see `retrieve_hybrid`'s `session` docstring). No-op when
                # `session` is None, matching every other caller exactly.
                session=session,
            )
        # U-107/U-132 (CONCEPT:AU-KG.retrieval.acl-aware-vector-retrieval): ACL/owner/
        # scope enforcement runs on the RAW candidate set, before archive trimming,
        # the score gate, or any other ranking/fusion step. A denied high-score
        # candidate must never be able to consume top-k capacity that an
        # authorized lower-score candidate would otherwise have won — enforcing
        # ACL only at the very end (the prior order) let unauthorized rows crowd
        # authorized ones out of the response even though they were never
        # returned themselves. `_enforce_acl_on_results` is a no-op when
        # `session` is None, matching prior behavior exactly for callers that
        # pass none.
        results = self._enforce_acl_on_results(
            results, session=session, summary="hybrid-search"
        )
        if not include_archived:
            results = [
                r for r in results if str(r.get("status", "")).upper() != "ARCHIVED"
            ]
        # CONCEPT:AU-KG.retrieval.unset-dependency-free (ScoreGate) — adaptively trim the weak tail by statistically
        # fusing the bi-encoder (`_score`) and cross-encoder (`_rerank_score`) signals
        # rather than a blind top_k cut. Recall-safe: keeps >= min_results, never more
        # than top_k, and only trims results more than ~1σ below the fused mean.
        results = score_gate(
            results, min_results=min(top_k, 5), max_results=top_k, keep_z=-1.0
        )
        # CONCEPT:AU-KG.query.chronoid-fits-residual-quantization (ChronoID) — infuse an explicit temporal-recency bucket onto
        # every result so generative/recommendation consumers can condition on time.
        self._annotate_time_buckets(results)
        # Public contract: every search result exposes its relevance under the
        # `score` key. The hybrid retriever ranks into the internal `_score` (it
        # re-weights it during fusion); without this projection the MCP formatter
        # and other consumers read `score`, miss the value, and show 0.00 even
        # though the ranking is correct. (Fix: producer wrote `_score`, consumers
        # read `score`, no layer mapped them.)
        for r in results:
            if isinstance(r, dict) and r.get("score") is None and "_score" in r:
                r["score"] = r["_score"]
        return results

    def _enforce_acl_on_results(
        self,
        results: list[dict[str, Any]],
        *,
        session: GraphSession | None,
        summary: str,
    ) -> list[dict[str, Any]]:
        """Apply the SAME per-node ACL + owner/scope + audit boundary as
        :meth:`query_cypher` to a vector/hybrid retrieval result set
        (CONCEPT:AU-KG.retrieval.acl-aware-vector-retrieval).

        Deliberately opt-in on the EXPLICIT ``session`` argument only — it does
        NOT fall back to the ambient :func:`~..core.session.current_session`.
        Unlike the raw Cypher path (every ``query_cypher`` caller already goes
        through a governed KG with real per-node ACLs), ``search_hybrid`` has
        ~20 existing internal/test callers that build ad-hoc result dicts with
        no backing ACL infrastructure at all — auto-enforcing on any ambient
        session (which the unit-test harness sets for unrelated reasons on
        EVERY test) would default-deny every one of them, since
        ``secured_reads.permit`` denies any node lacking a registered ACL. The
        MCP-served ``graph_search`` tool — the actual externally-reachable
        surface this closes — opts in explicitly by passing the ambient
        session it already has (CONCEPT:AU-KG.retrieval.acl-aware-vector-retrieval),
        so real served reads are governed by default with **no operator
        configuration**, while every other in-process caller keeps its exact
        prior behavior until it is deliberately updated to pass one. A
        ``session`` that IS supplied always gets full enforcement — an
        infrastructure failure raises ``PermissionError`` rather than
        silently degrading to the unfiltered ``results`` (the fail-open shape
        this closes).
        """
        if session is None:
            return results
        from agent_utilities.knowledge_graph.core.session import resolve_session

        resolved = resolve_session(session, required_scope="kg:read")
        from agent_utilities.observability.gateway_metrics import (
            RETRIEVAL_ACL_ENFORCEMENT,
        )

        try:
            from agent_utilities.knowledge_graph.core.secured_reads import (
                audit_read,
                filter_rows,
                row_node_ids,
                visible,
            )

            governed = visible(filter_rows(results, resolved.actor), resolved.actor)
            audit_read(row_node_ids(governed), summary=summary, actor=resolved.actor)
            RETRIEVAL_ACL_ENFORCEMENT.labels(outcome="admitted").inc(len(governed))
            RETRIEVAL_ACL_ENFORCEMENT.labels(outcome="denied").inc(
                len(results) - len(governed)
            )
            return governed
        except Exception as exc:
            RETRIEVAL_ACL_ENFORCEMENT.labels(outcome="enforcement_failed").inc()
            from agent_utilities.core.log_privacy import sanitize_log_text

            logger.error(
                "Graph %s ACL/audit enforcement failed: %s",
                summary,
                sanitize_log_text(_describe_secured_read_failure(exc)),
            )
            raise PermissionError(
                f"Graph {summary} row-policy or audit enforcement failed"
            ) from exc

    # ------------------------------------------------------------------
    # Temporal semantic IDs (CONCEPT:AU-KG.query.chronoid-fits-residual-quantization — ChronoID)
    # ------------------------------------------------------------------
    @staticmethod
    def _node_event_epoch(node: dict[str, Any]) -> float | None:
        """Best-effort epoch (seconds) for a result's event time, or None.

        Reads the bi-temporal fields (CONCEPT:AU-KG.temporal.bi-temporal-memory-layers) first, then common
        creation/timestamp aliases, parsing either an epoch number or an
        ISO-8601 string (``Z`` accepted).
        """
        import datetime as _dt

        inner = node.get("node", node) if isinstance(node, dict) else {}
        src = inner if isinstance(inner, dict) else {}
        for key in (
            "event_time",
            "valid_from",
            "storage_time",
            "created_at",
            "timestamp",
        ):
            val = node.get(key) if isinstance(node, dict) else None
            val = val if val is not None else src.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, int | float):
                return float(val)
            try:
                text = str(val).replace("Z", "+00:00")
                return _dt.datetime.fromisoformat(text).timestamp()
            except (ValueError, TypeError):
                continue
        return None

    def _annotate_time_buckets(self, results: list[dict[str, Any]]) -> None:
        """Annotate each result with a ``_time_bucket`` recency token (KG-2.86).

        Uses only the (fit-free) time bucketer, so it is cheap enough to run on
        every search; recent results land in bucket 0, unknown-date in the last.
        """
        import time as _time

        bucketer = TemporalSemanticIdEncoder()
        now = _time.time()
        for r in results:
            if isinstance(r, dict):
                r["_time_bucket"] = bucketer.time_bucket(
                    self._node_event_epoch(r), now_epoch=now
                )

    def temporal_semantic_ids(
        self, query: str, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Retrieve, then attach a temporal **semantic ID** to each result (KG-2.86).

        CONCEPT:AU-KG.query.chronoid-fits-residual-quantization — ChronoID. Fits a residual-quantization codebook on the
        retrieved embeddings and infuses an explicit temporal token, exposing a
        compact ``_temporal_sid`` per node for generative retrieval/recommendation.
        """
        import time as _time

        results = self.search_hybrid(query, top_k=top_k) or []
        embeddings: list[Any] = []
        for r in results:
            inner = r.get("node", r) if isinstance(r, dict) else {}
            emb = (inner or {}).get("embedding") if isinstance(inner, dict) else None
            if emb:
                embeddings.append(emb)
        encoder = TemporalSemanticIdEncoder()
        if embeddings:
            encoder.fit(embeddings)
        now = _time.time()
        for r in results:
            if not isinstance(r, dict):
                continue
            inner = r.get("node", r)
            emb = (inner or {}).get("embedding") if isinstance(inner, dict) else None
            event = self._node_event_epoch(r)
            r["_time_bucket"] = encoder.time_bucket(event, now_epoch=now)
            if emb and encoder.is_fitted:
                r["_temporal_sid"] = list(encoder.encode(emb, event, now_epoch=now))
        return results

    # ------------------------------------------------------------------
    # Iterative query expansion with graded relevance feedback
    # (CONCEPT:AU-KG.query.adore-concept-expansion — ADORE, with CONCEPT:AU-KG.retrieval.adaptive-stopping-iterative-retrieval TASR stopping)
    # ------------------------------------------------------------------
    def search_adore(
        self, query: str, top_k: int = 10, max_rounds: int = 4
    ) -> list[dict[str, Any]]:
        """Iterative query expansion with retrieval-grounded relevance feedback.

        CONCEPT:AU-KG.query.adore-concept-expansion (ADORE) — reformulate → retrieve → judge → partition by
        graded relevance → repeat, until CONCEPT:AU-KG.retrieval.adaptive-stopping-iterative-retrieval (TASR) training-free
        stopping (answer repeat / coverage saturation) or ``max_rounds``. The
        judge is the deterministic lexical relevance proxy (no network), and the
        relevance feedback is grounded in the highest-graded retrieved passages.
        """
        from ..retrieval.reasoning_reranker import LexicalRelevanceScorer

        scorer = LexicalRelevanceScorer()
        node_cache: dict[str, dict[str, Any]] = {}

        def retrieve_fn(expanded: str, k: int) -> list[dict[str, Any]]:
            nodes = self.search_hybrid(query=expanded, top_k=k) or []
            for n in nodes:
                if isinstance(n, dict):
                    nid = str(n.get("id") or (n.get("node", {}) or {}).get("id") or "")
                    if nid:
                        node_cache.setdefault(nid, n)
            return nodes

        def judge_fn(q: str, text: str) -> int:
            return max(0, min(3, round(scorer.score(q, text) * 3)))

        def reformulate_fn(
            original: str,
            prev_pseudo: list[str],
            graded: dict[int, list[str]],
        ) -> list[str]:
            feedback = list(graded.get(3, [])) + list(graded.get(2, []))
            return feedback[:3] or prev_pseudo or [original]

        expander = IterativeQueryExpander(
            retrieve_fn,
            judge_fn,
            reformulate_fn,
            max_rounds=max_rounds,
            top_k=max(top_k * 2, 20),
            judge_depth=10,
        )
        history = expander.run(query, query)
        ranked: list[dict[str, Any]] = []
        for doc_id, _score in history.final_ranking[:top_k]:
            node = node_cache.get(doc_id)
            if node is not None:
                ranked.append(node)
        return ranked[:top_k]

    def search_dci(
        self,
        query: str,
        max_hops: int = 2,
        top_k: int = 10,
        evidence_chain: bool = True,
    ) -> list[dict[str, Any]]:
        """Direct Corpus Interaction — multi-hop graph traversal retrieval.

        CONCEPT:AU-KG.memory.auto-similarity-memory-graph — Research: 2605.05242v1

        Unlike vector search (finds similar chunks by embedding distance),
        DCI traverses the graph structure to find connected evidence chains.
        This enables multi-hop reasoning: "What papers cite the same methods?"
        or "Which concepts are connected through shared implementations?"

        Operates in three explicit stages:
            Seed: initial vector retrieval (same as search_hybrid).
            Neighbors: graph expansion around seed results.
            Multi-hop: follow edges up to ``max_hops`` deep,
                building an evidence chain that tracks the traversal path.

        Args:
            query: Search query for the initial vector seed.
            max_hops: Maximum graph traversal depth (default 2).
            top_k: Maximum results to return.
            evidence_chain: If True, include the traversal path in results.

        Returns:
            List of result dicts, each with:
                - Standard node properties (id, name, type, score, etc.)
                - ``hop_depth``: How many hops from the seed this result is.
                - ``evidence_path``: List of (node_id, edge_type) tuples
                  showing how this result was reached (if evidence_chain=True).
        """
        # Vector seed — fast retrieval.
        seeds = self.search_hybrid(query, top_k=min(top_k, 5))
        if not seeds:
            return []

        # Track all discovered nodes to avoid duplicates
        seen: set[str] = set()
        results: list[dict[str, Any]] = []

        # Add seeds as hop-0 results
        for seed in seeds:
            sid = str(seed.get("id", ""))
            if not sid or sid in seen:
                continue
            seen.add(sid)
            seed["hop_depth"] = 0
            seed["evidence_path"] = [(sid, "seed")]
            results.append(seed)

        # Graph traversal — expand outward from seeds.
        frontier = [
            (str(s.get("id", "")), s.get("evidence_path", []))
            for s in seeds
            if s.get("id")
        ]

        for hop in range(1, max_hops + 1):
            next_frontier: list[tuple[str, list]] = []

            # Collect this hop's neighbor expansion in ONE pass (preserving the
            # existing `seen`-dedup order across every frontier node), THEN
            # hydrate every discovered neighbor in a SINGLE batched engine
            # round-trip (CONCEPT:AU-KG.retrieval.batch-hydrate) rather than one
            # `_get_node_properties` point-read per neighbor — the old per-neighbor
            # call serialized O(N) engine reads per hop and dominated this
            # multi-hop retrieval leg's latency under engine contention.
            pending: list[tuple[str, list]] = []
            for node_id, path in frontier:
                if not self.graph.has_node(node_id):
                    continue

                # Expand neighbors (both directions)
                neighbors = list(self.graph.get_successors(node_id)) + list(
                    self.graph.get_predecessors(node_id)
                )

                for neighbor_id in neighbors:
                    if neighbor_id in seen:
                        continue
                    seen.add(neighbor_id)
                    pending.append((neighbor_id, path))

            hydrated = self.graph._get_node_properties_batch(
                [neighbor_id for neighbor_id, _path in pending]
            )
            for neighbor_id, path in pending:
                # Get node data
                node_data = dict(hydrated.get(neighbor_id, {}))
                node_data["id"] = neighbor_id
                node_data["hop_depth"] = hop

                if evidence_chain:
                    node_data["evidence_path"] = path + [(neighbor_id, "RELATED")]

                # Score decay: further hops get lower scores
                base_score = float(node_data.get("importance_score", 0.5))
                node_data["_score"] = round(base_score * (0.8**hop), 4)

                results.append(node_data)
                next_frontier.append((neighbor_id, node_data.get("evidence_path", [])))

            frontier = next_frontier
            if not frontier:
                break

        # Sort by score (seeds first, then by decayed importance)
        results.sort(key=lambda x: (-x.get("_score", 0), x.get("hop_depth", 99)))

        return results[:top_k]

    def search_memories(
        self, query: str, top_k: int = 5, skip_quality_gate: bool = False
    ) -> list[dict[str, Any]]:
        """Search specifically for memory nodes.

        ``skip_quality_gate`` forwards to :meth:`search_hybrid`: the retrieval
        quality gate scores relevance from embeddings, so without an embedding
        model configured every result scores as low-relevance and gets
        dropped even for an exact content match. Defaults to ``False`` to
        preserve existing behavior for callers that rely on the gate.
        """
        results = self.search_hybrid(
            query, top_k=50, skip_quality_gate=skip_quality_gate
        )
        # 'type' is the retired node property (see engine.py::_serialize_node,
        # which folds it into 'node_type' before every write) -- reading
        # 'type' here silently matched nothing, so search_memories() always
        # returned empty regardless of what was actually found.
        return [r for r in results if r.get("node_type") == RegistryNodeType.MEMORY][
            :top_k
        ]

    def query_impact(self, symbol_or_file: str) -> list[dict[str, Any]]:
        """Calculate the topological impact set for a code entity."""
        target_id = symbol_or_file
        if not self.graph.has_node(target_id):
            # Try fuzzy match by name — every node must be inspected to compare
            # names, so hydrate the WHOLE graph in ONE round-trip
            # (CONCEPT:AU-KG.retrieval.batch-hydrate) via the id+properties bulk
            # scan rather than one `_get_node_properties` point-read per node.
            for node, data in self.graph._get_all_nodes_with_properties():
                if data.get("name") == symbol_or_file:
                    target_id = node
                    break

        if not self.graph.has_node(target_id):
            return []

        # Ancestors via BFS on predecessors
        ancestors: set[str] = set()
        frontier = set(self.graph.get_predecessors(target_id))
        while frontier:
            next_frontier: set[str] = set()
            for n in frontier:
                if n not in ancestors:
                    ancestors.add(n)
                    next_frontier.update(self.graph.get_predecessors(n))
            frontier = next_frontier - ancestors
        # Batch-hydrate the whole ancestor set in ONE round-trip
        # (CONCEPT:AU-KG.retrieval.batch-hydrate) instead of one point-read per
        # ancestor.
        ancestor_ids = list(ancestors)
        hydrated = self.graph._get_node_properties_batch(ancestor_ids)
        return [{"id": n, **hydrated.get(n, {})} for n in ancestor_ids]

    def find_path(self, source: str, target: str) -> list[str]:
        """Find the shortest logical path between two nodes.

        The native engine performs the traversal in one bounded round trip.
        Transport and capability errors propagate instead of selecting a hidden
        Python implementation with different limits or semantics.
        """
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return []
        path = self.graph.get_shortest_path(source, target)
        return list(path) if path is not None else []

    def get_agent_tools(self, agent_name: str) -> list[str]:
        """Get all tools provided by a specific agent."""
        tools = []
        if self.graph.has_node(agent_name):
            for v in self.graph.get_successors(agent_name):
                # Edge type info not directly available in GCE
                tools.append(v.replace("tool:", ""))
        return tools

    def find_agent_for_tool(self, tool_name: str) -> list[str]:
        """Find all agents that provide a specific tool."""
        agents = []
        tool_id = f"tool:{tool_name}"
        if not self.graph.has_node(tool_id):
            return []
        for u in self.graph.get_predecessors(tool_id):
            agents.append(u)
        return agents

    def run_inference(self) -> int:
        """Run standard inference rules over the graph to derive new facts."""
        return self.inference_engine.run_inference()

    def find_relevant_callable_resources(
        self, task_description: str, required_caps: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Find the most relevant Tools, Agents, or Skills for a given task.

        CONCEPT:AU-ECO.toolkit.self-describing-registry enhanced: searches ALL resource types (tools,
        agents, skills, memory) in a single hybrid query for unified
        capability discovery (AgentOS-style category collapse).
        """
        if required_caps is None:
            required_caps = []
        # Hybrid search: semantic similarity + capability filtering
        # Search across all node types, not just callable_resource
        candidates = self.search_hybrid(task_description, top_k=30)
        filtered = []

        # Resource types that constitute "callable" capabilities
        _callable_types = {
            "callable_resource",
            "tool",
            "skill",
            "agent",
            "mcp_tool",
            "a2a_agent",
            "internal_skill",
            "agent_skill",
        }

        for c in candidates:
            c_type = str(c.get("type", "")).lower()
            c_resource_type = str(c.get("resource_type", "")).lower()

            # Match callable resources AND their component types
            if c_type not in _callable_types and c_resource_type not in _callable_types:
                continue

            # Check caps in linked metadata if backend available
            if self.backend and required_caps:
                res = self.query_cypher(
                    "MATCH (r {id: $id})-[:HAS_METADATA]->(m) RETURN m.capabilities as caps",
                    {"id": c["id"]},
                )
                if res and res[0].get("caps"):
                    caps_raw = res[0]["caps"]
                    # Robust parsing for list or string representation
                    if isinstance(caps_raw, str):
                        import ast

                        try:
                            caps = ast.literal_eval(caps_raw)
                        except Exception:
                            caps = [caps_raw]
                    else:
                        caps = caps_raw

                    if all(cap in caps for cap in required_caps):
                        filtered.append(c)
                else:
                    # Fallback if no metadata
                    if not required_caps:
                        filtered.append(c)
            else:
                filtered.append(c)
        return filtered

    def discover_all_capabilities(
        self,
        query: str | None = None,
        resource_types: list[str] | None = None,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """Discover all capabilities across all resource types in a unified view.

        CONCEPT:AU-ECO.toolkit.self-describing-registry — Self-Describing Function Registry

        Returns a unified function-level view of all discoverable
        capabilities, inspired by AgentOS's category collapse pattern.
        Each entry includes self-describing metadata (input/output
        schemas, trigger bindings) when available.

        Args:
            query: Optional semantic query to filter by relevance.
            resource_types: Optional filter by resource types.
            top_k: Maximum results to return.

        Returns:
            List of unified capability dicts with ``fn_namespace`` field
            matching AgentOS's ``fn namespace::action(args)`` pattern.
        """
        capabilities: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        # If query provided, use hybrid search for relevance ordering
        if query:
            results = self.search_hybrid(query, top_k=top_k * 2)
        else:
            # Full-graph scan — bulk id+properties in ONE round-trip
            # (CONCEPT:AU-KG.retrieval.batch-hydrate) instead of one
            # `_get_node_properties` point-read per node.
            results = [
                {"id": nid, **props}
                for nid, props in self.graph._get_all_nodes_with_properties()
            ]

        for r in results:
            rid = r.get("id", "")
            if rid in seen_ids:
                continue

            if str(r.get("status", "")).upper() == "ARCHIVED":
                continue

            r_type = str(r.get("type", "")).lower()
            r_resource = str(r.get("resource_type", "")).lower()

            # Filter to capability-bearing node types
            _cap_types = {
                "callable_resource",
                "tool",
                "skill",
                "agent",
                "tool_metadata",
                "spawned_agent",
            }
            if r_type not in _cap_types and r_resource not in _cap_types:
                continue

            # Apply resource_type filter
            if resource_types:
                if r_resource not in [rt.lower() for rt in resource_types]:
                    continue

            seen_ids.add(rid)

            # Build unified fn namespace (AgentOS pattern)
            namespace = r_resource or r_type
            action = r.get("name", rid).replace(" ", "_").lower()
            fn_namespace = f"fn {namespace}::{action}"

            capabilities.append(
                {
                    "id": rid,
                    "fn_namespace": fn_namespace,
                    "name": r.get("name", rid),
                    "description": r.get("description", ""),
                    "resource_type": r_resource or r_type,
                    "input_schema": r.get("input_schema", {}),
                    "output_schema": r.get("output_schema", {}),
                    "trigger_bindings": r.get("trigger_bindings", []),
                    "endpoint": r.get("endpoint"),
                    "_score": r.get("_score", 0.0),
                }
            )

            if len(capabilities) >= top_k:
                break

        return capabilities

    def list_callable_resources(self) -> list[dict[str, Any]]:
        """List all callable resources (MCP tools, A2A agents, skills)."""
        resources = []
        # Full-graph scan — bulk id+properties in ONE round-trip
        # (CONCEPT:AU-KG.retrieval.batch-hydrate) instead of one
        # `_get_node_properties` point-read per node.
        for n, data in self.graph._get_all_nodes_with_properties():
            # 'type' is the retired node property; the canonical name is
            # 'node_type' (see engine.py::_serialize_node).
            if data.get("node_type") == RegistryNodeType.CALLABLE_RESOURCE:
                resources.append({"id": n, **data})
        return resources

    def retrieve_orthogonal_context(
        self,
        query: str,
        views: list[str] | None = None,
    ) -> dict[str, Any]:
        """Perform policy-guided retrieval across all orthogonal MAGMA views.

        The temporal view is the canonical execution-provenance stream: it
        returns ``RunTrace`` rows in descending numeric ``event_sequence``
        order through :meth:`query_cypher`, retaining the normal tenant and
        read-policy boundary.  Conversational ``Episode`` nodes are not
        execution provenance and are intentionally excluded here.
        """
        if views is None:
            # V1 backward-compat invariant (§6): callers that pass no ``views``
            # must keep getting exactly these four. ``place``/``epistemic`` are
            # real views (see :meth:`retrieve_place_view` /
            # :meth:`retrieve_epistemic_view`) but are opt-in only — pass them
            # explicitly via ``views=[...]``.
            views = [
                "semantic",
                "temporal",
                "causal",
                "entity",
            ]
        context: dict[str, Any] = {"query": query, "views": {}}
        if "semantic" in views:
            context["views"]["semantic"] = self.search_hybrid(query, top_k=5)
        if "temporal" in views:
            context["views"]["temporal"] = self.query_cypher(
                "MATCH (r:RunTrace) RETURN r ORDER BY r.event_sequence DESC LIMIT 5"
            )
        if "causal" in views:
            context["views"]["causal"] = self.query_cypher(
                "MATCH (r:ReasoningTrace)-[:CAUSED_BY]->(p) RETURN r, p LIMIT 5"
            )
        if "entity" in views:
            context["views"]["entity"] = self.query_cypher(
                "MATCH (e:Entity) WHERE e.id CONTAINS $q OR e.entity_type CONTAINS $q RETURN e",
                {"q": query},
            )
        if "place" in views:
            context["views"]["place"] = self.retrieve_place_view(query, top_k=5)
        if "epistemic" in views:
            context["views"]["epistemic"] = self.retrieve_epistemic_view(query, top_k=5)
        return context

    def retrieve_place_view(
        self,
        query: str,
        place_ids: list[str] | None = None,
        phase_ids: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """MAGMA Place view — retrieve co-located entities (EcphoryRAG).

        Retrieves co-located entities or places using a Cypher query from the backend.
        """
        if not self.backend:
            logger.debug("No backend configured for retrieve_place_view.")
            return []

        results = []
        if place_ids:
            # Match entities co-located at specified places
            cypher = (
                "MATCH (e:Entity)-[r:CO_LOCATED_AT|co_located_at]->(p:Place) "
                "WHERE p.id IN $place_ids "
                "RETURN e, p LIMIT $limit"
            )
            rows = self.backend.execute(
                cypher, {"place_ids": place_ids, "limit": top_k}
            )
            for row in rows:
                entity = row.get("e", row)
                place = row.get("p", {})
                if isinstance(entity, dict):
                    entity["_place"] = (
                        place.get("id") if isinstance(place, dict) else str(place)
                    )
                    results.append(entity)
        elif phase_ids:
            # Match entities associated with specific phases
            cypher = (
                "MATCH (e:Entity)-[r:ASSOCIATED_WITH|associated_with]->(p:Phase) "
                "WHERE p.id IN $phase_ids "
                "RETURN e, p LIMIT $limit"
            )
            rows = self.backend.execute(
                cypher, {"phase_ids": phase_ids, "limit": top_k}
            )
            for row in rows:
                entity = row.get("e", row)
                phase = row.get("p", {})
                if isinstance(entity, dict):
                    entity["_phase"] = (
                        phase.get("id") if isinstance(phase, dict) else str(phase)
                    )
                    results.append(entity)
        else:
            # Query based search for co-located entities matching query
            cypher = (
                "MATCH (e:Entity)-[r:CO_LOCATED_AT|co_located_at]->(p:Place) "
                "WHERE e.id CONTAINS $q OR p.id CONTAINS $q "
                "RETURN e, p LIMIT $limit"
            )
            rows = self.backend.execute(cypher, {"q": query, "limit": top_k})
            for row in rows:
                entity = row.get("e", row)
                place = row.get("p", {})
                if isinstance(entity, dict):
                    entity["_place"] = (
                        place.get("id") if isinstance(place, dict) else str(place)
                    )
                    results.append(entity)

        return results

    def retrieve_epistemic_view(
        self,
        query: str,
        include_contradictions: bool = True,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """MAGMA Epistemic view — beliefs + supporting / contradicting evidence.

        CONCEPT:AU-KG.enrichment.entity-claim-extraction — Entity-Claim Extraction / MAGMA Completion

        Queries the KG for ClaimNodes matching the query, then traverses
        BUILDS_ON, EXEMPLIFIES, CITES edges for supporting evidence and
        CONTRADICTS edges for contradicting evidence.

        Args:
            query: Search query for relevant claims.
            include_contradictions: Whether to include contradicting claims.
            top_k: Maximum number of beliefs to return.

        Returns:
            Dict with ``beliefs``, ``supporting``, and ``contradicting`` lists.
        """
        beliefs: list[dict[str, Any]] = []
        supporting: list[dict[str, Any]] = []
        contradicting: list[dict[str, Any]] = []

        if self.backend:
            # 1. Find claims matching the query (beliefs)
            belief_results = self.backend.execute(
                "MATCH (c:Claim) WHERE c.claim_text CONTAINS $q "
                "OR c.name CONTAINS $q "
                "RETURN c ORDER BY c.confidence DESC LIMIT $limit",
                {"q": query, "limit": top_k},
            )
            for row in belief_results:
                claim = row.get("c", row)
                beliefs.append(claim)
                claim_id = claim.get("id", "")

                # 2. Find supporting evidence (BUILDS_ON, EXEMPLIFIES, CITES)
                support_results = self.backend.execute(
                    "MATCH (s)-[r]->(c {id: $cid}) "
                    "WHERE type(r) IN ['BUILDS_ON', 'EXEMPLIFIES', 'CITES', "
                    "'builds_on', 'exemplifies', 'cites'] "
                    "RETURN s, type(r) as rel_type",
                    {"cid": claim_id},
                )
                for s_row in support_results:
                    support_node = s_row.get("s", s_row)
                    support_node["_relationship"] = s_row.get("rel_type", "supports")
                    support_node["_target_claim"] = claim_id
                    supporting.append(support_node)

                # 3. Find contradicting evidence (CONTRADICTS)
                if include_contradictions:
                    contradict_results = self.backend.execute(
                        "MATCH (s)-[r]->(c {id: $cid}) "
                        "WHERE type(r) IN ['CONTRADICTS', 'contradicts', "
                        "'CONTRADICTS_BELIEF', 'contradicts_belief'] "
                        "RETURN s, type(r) as rel_type",
                        {"cid": claim_id},
                    )
                    for c_row in contradict_results:
                        contra_node = c_row.get("s", c_row)
                        contra_node["_relationship"] = c_row.get(
                            "rel_type", "contradicts"
                        )
                        contra_node["_target_claim"] = claim_id
                        contradicting.append(contra_node)
        else:
            # GCE fallback — every node's type must be inspected to match, so
            # hydrate the WHOLE graph in ONE round-trip
            # (CONCEPT:AU-KG.retrieval.batch-hydrate) rather than one
            # `_get_node_properties` point-read per node scanned.
            query_lower = query.lower()
            for node_id, data in self.graph._get_all_nodes_with_properties():
                node_type = str(data.get("type", "")).lower()
                if node_type in ("claim", "evidence"):
                    claim_text = str(
                        data.get("claim_text", data.get("claim", ""))
                    ).lower()
                    name = str(data.get("name", "")).lower()
                    if query_lower in claim_text or query_lower in name:
                        belief = dict(data)
                        belief["id"] = node_id
                        beliefs.append(belief)

                        if len(beliefs) >= top_k:
                            break

            # Find supporting/contradicting edges for found beliefs — collect
            # every (belief, predecessor) pair FIRST, then hydrate every
            # predecessor in ONE batched round-trip
            # (CONCEPT:AU-KG.retrieval.batch-hydrate) instead of one point-read
            # per predecessor per belief.
            pending_preds: list[tuple[str, str]] = []
            for belief in beliefs:
                belief_id = belief.get("id", "")
                for u in self.graph.get_predecessors(belief_id):
                    pending_preds.append((belief_id, u))
            hydrated_preds = self.graph._get_node_properties_batch(
                [u for _belief_id, u in pending_preds]
            )
            for belief_id, u in pending_preds:
                edge_type = "related"  # Default when edge props unavailable
                source_data = dict(hydrated_preds.get(u, {}))
                source_data["id"] = u
                source_data["_target_claim"] = belief_id

                if edge_type in ("builds_on", "exemplifies", "cites"):
                    source_data["_relationship"] = edge_type
                    supporting.append(source_data)
                elif edge_type in (
                    "contradicts",
                    "contradicts_belief",
                    "contradicts_kb",
                ):
                    if include_contradictions:
                        source_data["_relationship"] = edge_type
                        contradicting.append(source_data)

        logger.debug(
            "Epistemic view for %r: %d beliefs, %d supporting, %d contradicting",
            query,
            len(beliefs),
            len(supporting),
            len(contradicting),
        )

        return {
            "beliefs": beliefs,
            "supporting": supporting,
            "contradicting": contradicting,
        }

    def find_relevant_policies(self, query: str) -> list[dict[str, Any]]:
        """Search for policies that apply to the current query context."""
        # Hybrid search for policies
        results = self.query_cypher(
            "MATCH (p:Policy) WHERE p.name CONTAINS $q OR p.description CONTAINS $q RETURN p",
            {"q": query},
        )
        # Note: If LadybugDB supports vector search, we would use it here as well
        return [r.get("p", r) for r in results]

    def find_relevant_processes(self, query: str) -> list[dict[str, Any]]:
        """Search for process flows that match the current query goal."""
        results = self.query_cypher(
            "MATCH (f:ProcessFlow) WHERE f.goal CONTAINS $q OR f.name CONTAINS $q RETURN f",
            {"q": query},
        )
        return [r.get("f", r) for r in results]

    # ── Native Innovation Discovery (CONCEPT:AU-KG.query.object-graph-mapper) ──────────────

    # Signal dictionaries for innovation extraction — embedded in the engine
    # so that all search modes can reuse them without external scripts.
    _BIOMIMICRY_KEYWORDS: dict[str, dict[str, str]] = {
        "swarm": {"analogy": "multi-agent coordination", "domain": "orchestration"},
        "colony": {"analogy": "distributed task allocation", "domain": "scheduling"},
        "pheromone": {
            "analogy": "stigmergic communication channels",
            "domain": "signal_boards",
        },
        "neural": {"analogy": "adaptive model routing", "domain": "model_selection"},
        "immune": {
            "analogy": "anomaly detection and self-healing",
            "domain": "circuit_breakers",
        },
        "evolution": {
            "analogy": "genetic algorithm optimization",
            "domain": "prompt_evolution",
        },
        "mutation": {"analogy": "parametric exploration", "domain": "variant_pool"},
        "symbiosis": {
            "analogy": "plugin ecosystem composition",
            "domain": "mcp_composition",
        },
        "quorum": {
            "analogy": "distributed consensus thresholds",
            "domain": "bft_voting",
        },
        "mycelium": {"analogy": "knowledge graph topology", "domain": "kg_routing"},
        "plasticity": {
            "analogy": "continual learning without forgetting",
            "domain": "ewc",
        },
        "homeostasis": {
            "analogy": "resource equilibrium maintenance",
            "domain": "cost_governors",
        },
        "emergent": {
            "analogy": "complex behavior from simple rules",
            "domain": "swarm_presets",
        },
        "stigmergy": {
            "analogy": "indirect coordination via environment",
            "domain": "signal_boards",
        },
    }

    _TECH_KEYWORDS: dict[str, dict[str, str]] = {
        "attention": {
            "analogy": "context-aware retrieval weighting",
            "domain": "hybrid_retriever",
        },
        "transformer": {
            "analogy": "parallel sequence processing",
            "domain": "batch_processing",
        },
        "embedding": {
            "analogy": "semantic vector representation",
            "domain": "knowledge_graph",
        },
        "reinforcement": {
            "analogy": "reward-driven routing optimization",
            "domain": "confidence_gating",
        },
        "chain-of-thought": {
            "analogy": "rationale persistence",
            "domain": "quiet_star",
        },
        "tree search": {"analogy": "MCTS planning fallback", "domain": "lats"},
        "consensus": {"analogy": "multi-agent agreement", "domain": "bft"},
        "knowledge graph": {"analogy": "structured relational memory", "domain": "kg"},
        "ontology": {"analogy": "formal domain modeling", "domain": "owl_reasoning"},
        "retrieval": {
            "analogy": "context-aware document fetching",
            "domain": "hybrid_retriever",
        },
        "rag": {
            "analogy": "retrieval-augmented generation",
            "domain": "hybrid_retriever",
        },
        "multi-agent": {
            "analogy": "coordinated specialist teams",
            "domain": "orchestration",
        },
        "orchestration": {
            "analogy": "workflow coordination engine",
            "domain": "orchestration",
        },
        "planning": {
            "analogy": "task decomposition and scheduling",
            "domain": "htn_planning",
        },
        "tool use": {
            "analogy": "dynamic capability invocation",
            "domain": "tool_interface",
        },
        "mcp": {
            "analogy": "model context protocol integration",
            "domain": "mcp_factory",
        },
        "safety": {
            "analogy": "guardrails and constraint enforcement",
            "domain": "guardrails",
        },
        "evaluation": {
            "analogy": "quality assessment framework",
            "domain": "evaluation_engine",
        },
        "telemetry": {
            "analogy": "runtime observability signals",
            "domain": "telemetry",
        },
        "scheduling": {
            "analogy": "resource allocation optimization",
            "domain": "cognitive_scheduler",
        },
        "memory": {
            "analogy": "persistent experience storage",
            "domain": "tiered_memory",
        },
        "federated": {
            "analogy": "distributed graph querying",
            "domain": "external_federation",
        },
        "hypergraph": {
            "analogy": "n-ary relationship modeling",
            "domain": "hyperedges",
        },
        "topology": {"analogy": "structural graph analysis", "domain": "partitioning"},
        "composability": {
            "analogy": "modular building-block assembly",
            "domain": "mcp_composition",
        },
        "middleware": {
            "analogy": "cross-cutting concern interception",
            "domain": "middleware",
        },
        "inference": {
            "analogy": "derived knowledge from assertions",
            "domain": "owl_reasoning",
        },
        "security": {"analogy": "threat defense mechanisms", "domain": "security"},
        "context window": {
            "analogy": "adaptive context management",
            "domain": "context_management",
        },
    }

    _INNOVATION_SIGNALS = [
        "novel",
        "new",
        "propose",
        "introduce",
        "demonstrate",
        "improve",
        "outperform",
        "achieve",
        "enable",
        "first",
        "state-of-the-art",
        "sota",
        "surpass",
        "advance",
    ]

    def _extract_signals(self, content: str) -> dict[str, Any]:
        """Extract biomimicry and tech innovation signals from text content."""
        import re

        content_lower = content.lower()
        bio_hits = []
        tech_hits = []

        for kw, info in self._BIOMIMICRY_KEYWORDS.items():
            count = content_lower.count(kw)
            if count > 0:
                bio_hits.append(
                    {
                        "keyword": kw,
                        "count": count,
                        "analogy": info["analogy"],
                        "domain": info["domain"],
                    }
                )

        for kw, info in self._TECH_KEYWORDS.items():
            count = content_lower.count(kw)
            if count > 0:
                tech_hits.append(
                    {
                        "keyword": kw,
                        "count": count,
                        "analogy": info["analogy"],
                        "domain": info["domain"],
                    }
                )

        # Extract key innovation claims
        claims = []
        for sent in re.split(r"[.!?]\s+", content):
            if any(sig in sent.lower() for sig in self._INNOVATION_SIGNALS):
                words = sent.split()
                if 5 < len(words) < 50:
                    claims.append(sent.strip())
        claims = claims[:5]

        return {
            "biomimicry_signals": sorted(bio_hits, key=lambda x: -x["count"]),
            "tech_signals": sorted(tech_hits, key=lambda x: -x["count"]),
            "innovation_claims": claims,
            "total_signal_count": len(bio_hits) + len(tech_hits),
        }

    def discover_innovations(
        self,
        query: str,
        top_k: int = 20,
        relevance_threshold: float = 0.2,
        exclude_assimilated: bool = False,
        target_codebase: str = "",
    ) -> dict[str, Any]:
        """Native Layer 1 cross-reference: vector search + innovation signal extraction.

        CONCEPT:AU-KG.query.object-graph-mapper — Active Knowledge Graph Discovery

        Performs hybrid search, then enriches each result with biomimicry/tech
        innovation signals extracted from the node content. Results include
        signal counts, concept mappings, and innovation claims.

        This is the backend-native equivalent of the Layer 1 cross-reference
        pipeline, computed entirely on the KG without LLM calls.

        Args:
            query: Search query (concept name, research topic, etc.)
            top_k: Maximum enriched results to return.
            relevance_threshold: Minimum similarity score.
            exclude_assimilated: If True, filter out Article nodes that have
                ASSIMILATED_INTO edges with status='implemented' for the
                given target_codebase. CONCEPT:AU-KG.query.skip-assimilated-papers
            target_codebase: Codebase path to check assimilation against.
                Only used when exclude_assimilated=True.

        Returns:
            Dict with enriched results, signal summary, and top recommendations.
        """
        # 0. Build set of assimilated target_paths to exclude
        assimilated_paths: set[str] = set()
        if exclude_assimilated and self.backend:
            try:
                if target_codebase:
                    rows = self.query_cypher(
                        "MATCH (a:Article)-[r:ASSIMILATED_INTO]->(c) "
                        "WHERE r.status = 'implemented' AND r.codebase = $cb "
                        "RETURN DISTINCT a.target_path AS path",
                        {"cb": target_codebase},
                    )
                else:
                    rows = self.query_cypher(
                        "MATCH (a:Article)-[r:ASSIMILATED_INTO]->(c) "
                        "WHERE r.status = 'implemented' "
                        "RETURN DISTINCT a.target_path AS path",
                    )
                for row in rows:
                    p = row.get("path", "")
                    if p:
                        assimilated_paths.add(p)
                if assimilated_paths:
                    logger.info(
                        "Excluding %d assimilated paper paths from discovery",
                        len(assimilated_paths),
                    )
            except Exception as exc:
                logger.warning("Failed to load assimilated paths: %s", exc)

        # 1. Run hybrid vector search with low threshold to cast wide net
        raw_results = self.search_hybrid(
            query,
            top_k=top_k * 3,
            skip_quality_gate=True,
            relevance_threshold=relevance_threshold,
        )

        # 2. Enrich each result with innovation signals
        enriched = []
        domain_accumulator: dict[str, list[dict[str, Any]]] = {}

        for r in raw_results:
            # CONCEPT:AU-KG.query.skip-assimilated-papers — Skip assimilated papers
            if assimilated_paths:
                r_path = r.get("target_path", "")
                if r_path and r_path in assimilated_paths:
                    continue

            # Build content from available fields
            content_parts = []
            for field in (
                "content",
                "description",
                "name",
                "summary",
                "text",
                "claim_text",
            ):
                val = r.get(field)
                if val and isinstance(val, str):
                    content_parts.append(val)
            content = " ".join(content_parts)

            if not content or len(content) < 20:
                continue

            signals = self._extract_signals(content)

            entry = {
                "id": r.get("id", ""),
                "name": r.get("name", r.get("title", "")),
                "type": r.get("type", ""),
                "target_path": r.get("target_path", ""),
                "score": round(r.get("_score", 0.0), 4),
                "concept_id": r.get("concept_id", ""),
                **signals,
            }

            # Accumulate by domain for recommendations
            for sig in signals["tech_signals"] + signals["biomimicry_signals"]:
                domain = sig["domain"]
                if domain not in domain_accumulator:
                    domain_accumulator[domain] = []
                domain_accumulator[domain].append(
                    {
                        "source": entry["name"] or entry["id"],
                        "keyword": sig["keyword"],
                        "analogy": sig["analogy"],
                        "score": entry["score"],
                    }
                )

            if signals["total_signal_count"] > 0:
                enriched.append(entry)

        # Sort by (score * signal_count) for emergent value ranking
        enriched.sort(
            key=lambda x: x["score"] * (1 + x["total_signal_count"]),
            reverse=True,
        )
        enriched = enriched[:top_k]

        # 3. Build domain-level recommendations
        recommendations = []
        for domain, sources in sorted(
            domain_accumulator.items(),
            key=lambda x: -len(x[1]),
        ):
            best = max(sources, key=lambda s: s["score"])
            recommendations.append(
                {
                    "domain": domain,
                    "analogy": best["analogy"],
                    "source_count": len(sources),
                    "top_source": best["source"],
                    "top_score": best["score"],
                    "priority": "high"
                    if len(sources) >= 3
                    else "medium"
                    if len(sources) >= 2
                    else "low",
                }
            )

        return {
            "query": query,
            "total_matches": len(enriched),
            "results": enriched,
            "domain_recommendations": recommendations[:20],
            "summary": {
                "total_signals": sum((e["total_signal_count"] for e in enriched), 0),
                "unique_domains": len(domain_accumulator),
                "high_priority_domains": len(
                    [r for r in recommendations if r["priority"] == "high"]
                ),
            },
        }
