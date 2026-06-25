"""Auto-extracted graph-os MCP tools: query_tools (register_query_tools).

Split out of kg_server._build_server to deepen the MCP surface into focused
modules without changing tool behavior or names.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server

#: Code-symbol nav actions over the resolved graph (CONCEPT:KG-2.9g).
CODE_NAV_ACTIONS = frozenset(
    {"find_definition", "find_references", "trace_call_graph", "impact_of_change"}
)

#: Symbol→symbol path action (CONCEPT:KG-2.211) — handled outside the single-anchor
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
    """CONCEPT:KG-2.211 — "what connects A to B": the shortest path between two
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

    # Annotate each hop with the connecting edge (undirected match for the relation).
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

    return {
        "source": src_id,
        "target": dst_id,
        "connected": True,
        "length": len(path) - 1,
        "path": path,
        "hops": hops,
    }


def register_query_tools(mcp):
    """Register the query_tools group on the given FastMCP server."""

    @mcp.tool(
        name="graph_query",
        description="Execute a read-only Cypher query against the Knowledge Graph.",
        tags=["graph-os", "query"],
    )
    def graph_query(
        cypher: str = Field(
            description="A Cypher query string (read-only — no CREATE/MERGE/DELETE)."
        ),
        params: str = Field(default="{}", description="JSON-encoded query parameters."),
        scope: str = Field(
            default="local",
            description=(
                "'local' for the internal KG (Cypher), 'sql' to run read-only SQL over the "
                "KG via the engine's DataFusion surface (e.g. SELECT ... FROM nodes — "
                "CONCEPT:KG-2.243, same path as the pg-wire listener), or 'federated' to "
                "query an external graph endpoint."
            ),
        ),
        reference_id: str = Field(
            default="",
            description="Required when scope='federated'. The ExternalGraphReference node ID.",
        ),
        as_of: str = Field(
            default="",
            description=(
                "CONCEPT:KG-2.11 — optional ISO-8601 instant. When set, rows are filtered to "
                "those whose bi-temporal validity (valid_from <= as_of < valid_to) holds, "
                "answering 'what was true as of date T'."
            ),
        ),
        target: str = Field(
            default="",
            description=(
                "CONCEPT:KG-2.63 — named graph connection to query (default = primary). "
                "Use a registered connection name (e.g. 'prod-neo4j'), or 'all' (or a "
                "comma-separated list) to fan out the same query to several backends and "
                "get per-connection labeled results."
            ),
        ),
    ) -> str:
        """Execute a read-only Cypher query against the Knowledge Graph. Use this to fetch graph data, explore relationships, and read node properties."""
        parsed_params = json.loads(params) if params else {}

        if scope == "sql":
            # CONCEPT:KG-2.243 — read-only SQL over the KG via the engine's
            # DataFusion surface (the same path the pg-wire listener uses). The
            # `cypher` arg carries the SQL string. RLS-governed + read-path-first
            # (engine.sql refuses non-SELECT). Honors `target` fan-out like Cypher.
            try:
                entries, errors, fanout = kg_server._resolve_target_engines(target)
            except Exception as e:
                return json.dumps({"error": str(e)})
            if not fanout:
                _name, engine = entries[0]
                try:
                    return json.dumps(engine.sql(cypher), default=str)
                except Exception as e:
                    return json.dumps({"error": str(e)})
            results, fan_errors = kg_server.fanout_execute(
                entries, lambda name, engine: engine.sql(cypher)
            )
            return json.dumps(
                {"targets": results, "errors": {**errors, **fan_errors}},
                default=str,
            )

        if scope == "federated":
            engine = kg_server._get_engine()
            if not reference_id:
                return json.dumps(
                    {"error": "reference_id required for federated queries"}
                )
            try:
                results = engine.execute_federated_query(
                    reference_id, cypher, parsed_params
                )
                return json.dumps(results, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})

        # Local query — block writes. Match write keywords only as whole Cypher
        # clause keywords (word boundaries) and ignore quoted string literals, so
        # identifiers like ``created_at``/``offset`` or a literal ``'CREATE …'`` are
        # not misread as a mutation. (CONCEPT:KG-2.63)
        import re

        _cypher_no_literals = re.sub(r"'[^']*'|\"[^\"]*\"", "", cypher)
        _write_kw = re.search(
            r"\b(CREATE|MERGE|DELETE|REMOVE|DROP|SET)\b",
            _cypher_no_literals,
            re.IGNORECASE,
        )
        if _write_kw:
            return json.dumps(
                {
                    "error": f"Write operation '{_write_kw.group(1).upper()}' not allowed. Use kg_write for mutations."
                }
            )

        # CONCEPT:KG-2.63 — resolve the target connection(s).
        try:
            entries, errors, fanout = kg_server._resolve_target_engines(target)
        except Exception as e:
            return json.dumps({"error": str(e)})

        if not fanout:
            # Single connection (default or one named) — identical shape to legacy.
            _name, engine = entries[0]
            try:
                results = engine.query_cypher(
                    cypher, parsed_params, as_of=as_of or None
                )
                return json.dumps(results, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})

        # Fan-out — per-target timeout so one slow backend can't stall the set.
        results, fan_errors = kg_server.fanout_execute(
            entries,
            lambda name, engine: engine.query_cypher(
                cypher, parsed_params, as_of=as_of or None
            ),
        )
        return json.dumps(
            {"targets": results, "errors": {**errors, **fan_errors}}, default=str
        )

    kg_server.REGISTERED_TOOLS["graph_query"] = graph_query

    # ══════════════════════════════════════════════════════════════════
    # 1b. graph_context — CONCEPT:ORCH-1.39 cross-process curated-context store
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_context",
        description=(
            "CONCEPT:ORCH-1.39 — store/fetch curated context for invoker→spawned-agent "
            "handoff, persisted in the epistemic-graph so a SEPARATELY-spawned agent can read "
            "it by id. Actions: 'put' (store content, returns context_id), 'get' (fetch by "
            "context_id), 'list' (by session_id). Pass the returned context_id to "
            "graph_orchestrate(action='execute_agent', context_ref=...)."
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
        import contextlib
        import time
        import uuid as _uuid

        engine = kg_server._get_engine()
        if not engine:
            return json.dumps({"error": "IntelligenceGraphEngine not active."})
        if action == "put":
            if not content:
                return json.dumps({"error": "content required for put"})
            sid = session_id or _uuid.uuid4().hex[:8]
            cid = context_id or f"ctx:{sid}:{key or _uuid.uuid4().hex[:6]}"
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
            # CONCEPT:ORCH-1.40 — session-anchored collection: upsert the id-addressable
            # Session node and link it, so "list by session" is a reliable id-anchored
            # traversal (the engine has no property index; property scans are unreliable).
            snode = f"session:{sid}"
            with contextlib.suppress(Exception):
                engine.add_node(
                    snode, "Session", properties={"id": snode, "session_id": sid}
                )
                engine.add_edge(snode, cid, "HAS_CONTEXT")
            return json.dumps({"context_id": cid, "session_id": sid})
        if action == "get":
            if not context_id:
                return json.dumps({"error": "context_id required for get"})
            try:
                rows = engine.query_cypher(
                    "MATCH (c:ContextBlob) WHERE c.id = $id "
                    "RETURN c.content AS content, c.session_id AS session_id, "
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
                return json.dumps({"error": str(exc)})
        if action == "prune":
            # Delete expired ContextBlobs (CONCEPT:ORCH-1.39 lifecycle).
            try:
                rows = engine.query_cypher(
                    "MATCH (c:ContextBlob) WHERE c.ttl_s > 0 AND "
                    "(c.created_at + c.ttl_s) < $now RETURN c.id AS id",
                    {"now": time.time()},
                )
                pruned = 0
                _del = getattr(engine, "delete_node", None) or getattr(
                    getattr(engine, "backend", None), "delete_node", None
                )
                for r in rows or []:
                    if callable(_del):
                        with contextlib.suppress(Exception):
                            _del(r["id"])
                            pruned += 1
                return json.dumps({"pruned": pruned, "expired": len(rows or [])})
            except Exception as exc:  # noqa: BLE001
                return json.dumps({"error": str(exc)})
        if action == "list":
            try:
                # CONCEPT:ORCH-1.40 — id-anchored traversal from the Session node (the engine's
                # reliable, fast O(degree) path; the index-less backend can't serve property
                # scans). The traversal reader returns whole nodes (`RETURN c`), so project +
                # sort + limit client-side.
                rows = engine.query_cypher(
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
                return json.dumps({"error": str(exc)})
        return json.dumps({"error": f"unknown action: {action}"})

    kg_server.REGISTERED_TOOLS["graph_context"] = graph_context

    # ══════════════════════════════════════════════════════════════════
    # 1c. graph_message — CONCEPT:ORCH-1.40 invoker↔spawned-agent message channel
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_message",
        description=(
            "CONCEPT:ORCH-1.40 — bidirectional, cross-process, ordered message channel between "
            "an invoking agent and a spawned agent, over the epistemic-graph native channels. "
            "Actions: 'open' (session_id+run_id → channel_id), 'send' (channel_id+sender+payload "
            "[+durable]), 'receive' (channel_id [+since cursor] → new messages + cursor), "
            "'history' (durable replay, survives restart), 'close'. Use the channel_id returned "
            "by graph_orchestrate(execute_agent, open_channel=True) to talk to the spawned agent."
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
            msgs, cursor = agent_channel.receive(engine, channel_id, since=since)
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
    def graph_search(
        query: str = Field(description="Natural language search query or concept ID."),
        mode: str = Field(
            default="hybrid",
            description="Search strategy:\n- 'hybrid': Semantic + keyword weighted search (default).\n- 'hyde': Memory-first HyDE multi-query plan + dual threshold (CONCEPT:KG-2.12).\n- 'deep': Wide-recall single query at the 0.28 deep threshold.\n- 'concept': Look up a CONCEPT:ID (e.g. 'KG-2.7', 'ORCH-1.0').\n- 'analogy': Find structurally similar concepts.\n- 'memory': Search tiered memory (episodic/semantic/procedural).\n- 'discover': Cross-reference query against all ingested content.\n- 'dci': Direct Corpus Interaction.\n- 'latent': Latent-topology hierarchical routing (CONCEPT:KG-2.3).\n- 'sira': Single-shot SIRA sparsity-aligned context.\n- 'hard_negatives': Mine hard negatives for the query (CONCEPT:KG-2.3).\n- 'rerank': Hybrid semantic+keyword re-scoring of candidates.\n- 'adore': Iterative query expansion with retrieval-grounded graded relevance feedback + training-free stopping (CONCEPT:KG-2.88/2.87).\n- 'chrono_ids': Attach an explicit temporal semantic ID (+recency bucket) to each result for generative retrieval (CONCEPT:KG-2.86).",
        ),
        top_k: int = Field(default=10, description="Maximum results to return."),
        self_correct: bool = Field(
            default=False,
            description="CONCEPT:KG-2.12 — run a self-correcting second retrieval pass at the deep threshold when the quality gate fails.",
        ),
        as_of: str = Field(
            default="",
            description="Optional ISO-8601 instant. Pack-driven recency decay is measured relative to this time, enabling knowledge-state-as-of-date-D retrieval such as an academic literature state. Defaults to now (CONCEPT:KG-2.22).",
        ),
        target: str = Field(
            default="",
            description=(
                "CONCEPT:KG-2.63 — named graph connection to search (default = primary). "
                "Use a registered connection name, or 'all' (or a comma-separated list) to "
                "fan out and get per-connection labeled results."
            ),
        ),
    ) -> str:
        """Search the Knowledge Graph using multiple strategies. Useful for finding context, concepts, memories, and capabilities across the ecosystem."""

        def _run_search(engine: Any) -> str:
            if not engine:
                return "Error: IntelligenceGraphEngine not active."
            try:
                return _search_with_engine(engine)
            except Exception as e:
                return f"Search error: {str(e)}"

        def _search_with_engine(engine: Any) -> str:
            if mode in ("hyde", "deep"):
                results = engine.search_hybrid(
                    query=query, top_k=top_k, mode=mode, self_correct=self_correct
                )
            elif mode == "hybrid":
                results = engine.search_hybrid(
                    query=query,
                    top_k=top_k,
                    self_correct=self_correct,
                    as_of=as_of or None,
                )
            elif mode == "concept":
                results = engine.search_hybrid(query=query, top_k=top_k)
            elif mode == "analogy":
                results = engine.search_hybrid(query=query, top_k=top_k)
            elif mode == "adore":
                results = engine.search_adore(query=query, top_k=top_k)
            elif mode == "chrono_ids":
                results = engine.temporal_semantic_ids(query=query, top_k=top_k)
            elif mode == "dci":
                results = engine.search_dci(query=query, top_k=top_k)
            elif mode == "memory":
                results = engine.search_memories(query=query, top_k=top_k)
            elif mode == "discover":
                try:
                    from agent_utilities.capabilities.manager import CapabilityManager

                    manager = CapabilityManager(engine)
                    results = manager.discover_capabilities(query)
                    if not results:
                        return f"No capabilities found for '{query}'"
                    return "\n".join([f"- {r.name}: {r.description}" for r in results])
                except ImportError:
                    return "Error: capabilities module not available"
            elif mode == "latent":
                # KG-2.3 — route through the latent topology hierarchy.
                from agent_utilities.knowledge_graph.retrieval.latent_topology_rag import (  # noqa: E501
                    LatentTopologicalRAG,
                )

                results = LatentTopologicalRAG(engine).retrieve(query, top_k=top_k)
            elif mode == "sira":
                # Single-shot SIRA: hybrid-retrieve, then sparsity-align the set.
                from agent_utilities.knowledge_graph.retrieval.single_shot_sira import (
                    SingleShotSIRA,
                )

                base = engine.search_hybrid(query=query, top_k=top_k) or []
                results = SingleShotSIRA(engine).align_context(base)
            elif mode == "hard_negatives":
                # KG-2.3 — mine hard negatives via the engine's hybrid retriever.
                from agent_utilities.knowledge_graph.retrieval.hard_negative_miner import (  # noqa: E501
                    HardNegativeMiner,
                )

                retriever = getattr(engine, "hybrid_retriever", None)
                if retriever is None:
                    return (
                        "Error: hybrid retriever unavailable for hard-negative mining."
                    )
                negs = HardNegativeMiner(retriever).mine(query)
                if not negs:
                    return f"No hard negatives mined for: '{query}'"
                return "\n".join(
                    f"- {n.doc_id}: {getattr(n, 'reason', '')}" for n in negs
                )
            elif mode == "rerank":
                # Semantic-retrieval hybrid re-scoring of a candidate set.
                from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (  # noqa: E501
                    HybridSearchScorer,
                )

                base = engine.search_hybrid(query=query, top_k=top_k) or []
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
                results = HybridSearchScorer().score_documents(query, qemb, docs)
            else:
                return f"Error: Unknown search mode '{mode}'"

            if not results:
                return f"No results found for query: '{query}'"

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

        # CONCEPT:KG-2.63 — resolve target connection(s).
        try:
            entries, errors, fanout = kg_server._resolve_target_engines(target)
        except Exception as e:
            return f"Search error: {str(e)}"

        if not fanout:
            return _run_search(entries[0][1])

        # Fan-out — per-target timeout so one slow backend can't stall the set.
        results, fan_errors = kg_server.fanout_execute(
            entries, lambda name, engine: _run_search(engine)
        )
        out_lines = [f"=== {name} ===\n{results[name]}" for name in results]
        out_lines += [
            f"=== {name} (error) ===\n{err}"
            for name, err in {**errors, **fan_errors}.items()
        ]
        return "\n\n".join(out_lines)

    kg_server.REGISTERED_TOOLS["graph_search"] = graph_search

    @mcp.tool(
        name="graph_search_synthesis",
        description=(
            "Synthesize a shortcut-resistant deep-search task from the evidence graph, "
            "or diagnose realized search difficulty of solver trajectories "
            "(CONCEPT:KG-2.70/2.71/2.72, AHE-3.30; distills arXiv:2606.12087)."
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
                return f"Error: invalid trajectories JSON: {e}"
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
                backend = getattr(self._eng, "backend", None)
                if backend is not None and hasattr(backend, "execute"):
                    return backend.execute(cypher, params or {}) or []
                if hasattr(self._eng, "query"):
                    return self._eng.query(cypher, params or {}) or []
                return []

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
            return f"Synthesis error: {str(e)}"
        return _json.dumps(task.to_dict())

    kg_server.REGISTERED_TOOLS["graph_search_synthesis"] = graph_search_synthesis

    # ══════════════════════════════════════════════════════════════════
    # graph_code_nav — CONCEPT:KG-2.9g code-symbol navigation over the
    # RESOLVED graph (`:Code` + `calls`/`depends_on`/`IMPLEMENTS`). Templated
    # so agents (and Duo-style flows) get gkg's find-def / find-refs / trace /
    # impact as first-class tools instead of hand-written Cypher.
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_code_nav",
        description=(
            "Navigate the resolved code graph (CONCEPT:KG-2.9g). action: "
            "'find_definition' (locate a symbol's :Code node), 'find_references' "
            "(callers of a symbol), 'trace_call_graph' (transitive callees), "
            "'impact_of_change' (transitive callers = blast radius), 'connects' "
            "(shortest path between TWO symbols — set symbol/node_id AND "
            "target_symbol/target_node_id — rendered hop-by-hop with each edge's "
            "relation + confidence, CONCEPT:KG-2.211). Start from a symbol name or "
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
                return json.dumps({"error": str(e)})

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
            return json.dumps({"error": str(e)})

        try:
            rows = engine.query_cypher(cypher, qparams)
            return json.dumps({"action": action, "results": rows}, default=str)
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["graph_code_nav"] = graph_code_nav
