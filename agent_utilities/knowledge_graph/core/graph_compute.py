# CONCEPT:KG-2.2 - High-Performance Graph Compute Engine
# CONCEPT:ORCH-1.11 - Compiled Orchestration Kernel
# CONCEPT:KG-2.7 - Tokio Service Layer (Tokio-first)
# CONCEPT:KG-2.58 - Tenant-Partitioned Engine Sharding (HRW over GRAPH_SERVICE_ENDPOINTS)

import json
import logging
import os
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)


def _load_or_create_engine_secret() -> str:
    """Load (or generate once) the per-install engine HMAC secret.

    CONCEPT:OS-5.14 — Authenticated Identity Enforcement. The secret lives at
    ``data_dir()/engine_secret`` with mode 0600 so every local process — and
    every engine this launcher spawns — shares it. Creation is race-safe
    (``O_EXCL``; the loser re-reads the winner's secret). If the data dir is
    unwritable a process-local secret is used (warned: siblings won't share it).
    """
    import secrets as _secrets

    from agent_utilities.core.paths import data_dir

    path = data_dir() / "engine_secret"
    try:
        if path.exists():
            existing = path.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        path.parent.mkdir(parents=True, exist_ok=True)
        secret = _secrets.token_hex(32)
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return path.read_text(encoding="utf-8").strip() or secret
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(secret)
        return secret
    except OSError as exc:
        logger.warning(
            "Could not persist engine secret at %s (%s); using a process-local "
            "secret — sibling processes will not share it.",
            path,
            exc,
        )
        return _secrets.token_hex(32)


def resolve_engine_auth(config: Any) -> tuple[str | None, bool]:
    """Resolve the engine HMAC auth material as ``(secret, insecure)``.

    CONCEPT:OS-5.14 — secure by default:

    * ``KG_ENGINE_INSECURE=1`` → ``(None, True)``: no client auth token, and a
      spawned engine gets ``EPISTEMIC_GRAPH_ALLOW_INSECURE=1`` so binaries that
      refuse to start without a secret still come up for dev.
    * ``GRAPH_SERVICE_AUTH_SECRET`` set → use it verbatim.
    * Otherwise → the persisted per-install secret
      (:func:`_load_or_create_engine_secret`).

    Always sending a secret is backward-compatible: an engine running without
    one ignores auth tokens entirely, so this works with both old (empty-secret
    tolerant) and new (refuse-insecure) engine binaries.
    """
    if getattr(config, "kg_engine_insecure", False):
        return None, True
    if config.graph_service_auth_secret:
        return config.graph_service_auth_secret, False
    return _load_or_create_engine_secret(), False


class GraphComputeEngine:
    """Graph compute engine backed by the epistemic-graph Tokio service.

    All graph operations route through the Tokio service layer via UDS/TCP
    (length-prefixed MessagePack, HMAC-authenticated). There is **no PyO3 /
    in-process mode** — the service is a separate process and must be running
    before this engine is instantiated.
    """

    def __init__(self, graph_name: str | None = None, **kwargs: Any) -> None:
        from epistemic_graph.client import SyncEpistemicGraphClient

        from agent_utilities.core.config import AgentConfig

        from .shard_topology import (
            is_local_endpoint,
            record_shard_connect,
            resolve_endpoints,
            resolve_routing_graph,
            shard_endpoint_for,
        )

        self.graph: dict[str, Any] = {}
        # SyncEpistemicGraphClient wrapped in a BreakerClientProxy
        # — attribute-transparent; raw client at
        # ``self._client.__wrapped__``. (CONCEPT:OS-5.23)
        self._client: Any
        self._mode: str = "service"

        config = AgentConfig()
        endpoints = resolve_endpoints(config)
        sharded = len(endpoints) > 1
        if sharded:
            # Tenant-partitioned sharding (CONCEPT:KG-2.58): tenant → named
            # graph → HRW → shard. An explicit non-default graph routes by its
            # own name; the default graph maps to the ambient ActorContext
            # tenant's graph (tenant__<t>__<base>) when one is in scope.
            graph_name = resolve_routing_graph(graph_name, config)
        elif graph_name is None:
            # Per-tenant named-graph isolation must NOT require multiple shards
            # (CONCEPT:KG-2.60): with enforcement on, route the ambient tenant to
            # its own named graph even on a single endpoint (HRW over one
            # endpoint is the identity). With enforcement off this is byte-for-byte
            # the legacy default-graph behaviour.
            from .company_brain_runtime import brain_enforcement_enabled

            if brain_enforcement_enabled():
                graph_name = resolve_routing_graph(None, config)
            else:
                graph_name = config.kg_default_graph
        # Retained so downstream consumers (e.g. the delta-ingestion manifest)
        # can key state by tenant graph. (CONCEPT:KG-2.8)
        self.graph_name = graph_name

        # Note: Since GraphComputeEngine is synchronous and often used as a long-lived
        # wrapper, we still use the standard SyncEpistemicGraphClient but point it at
        # the graph's HRW-owning shard (identity with one endpoint). True async
        # connection pooling callers should use epistemic_graph.pool.ShardRouter,
        # which shares this exact placement function. (CONCEPT:KG-2.58)
        endpoint = shard_endpoint_for(graph_name, endpoints)
        # Explicit endpoint override (CONCEPT:KG-2.58 / Phase D — dedicated ingest
        # engine): the ingest path pins its parse + community-scratch work to a
        # SEPARATE engine process, isolated from the query engine and the background
        # daemons (embedding backfill / reconcile / poll). Bypasses HRW so the caller
        # controls placement; the caller is responsible for only passing a reachable
        # endpoint (it health-gates + falls back to the query engine otherwise).
        endpoint_override = kwargs.get("endpoint")
        if endpoint_override:
            endpoint = str(endpoint_override)
        # Engine auth (CONCEPT:OS-5.14): resolve the shared HMAC secret —
        # configured, or generated once and persisted under the XDG data dir —
        # and export it so sibling clients (the epistemic_graph pool and any
        # direct SyncEpistemicGraphClient user falls back to this env var) and
        # spawned engines agree. KG_ENGINE_INSECURE=1 opts out for dev.
        auth_secret, engine_insecure = resolve_engine_auth(config)
        if auth_secret:
            os.environ.setdefault("GRAPH_SERVICE_AUTH_SECRET", auth_secret)
        connect_kwargs = {
            "auth_secret": auth_secret,
            "graph_name": graph_name,
        }
        if endpoint.startswith("tcp://"):
            connect_kwargs["tcp_addr"] = endpoint[6:]
        elif endpoint.startswith("unix://"):
            connect_kwargs["socket_path"] = endpoint[7:]
        else:
            connect_kwargs["socket_path"] = endpoint

        # Circuit breaker — ONE shared breaker per endpoint (CONCEPT:OS-5.23).
        # When the engine is down, N consecutive connect/timeout failures open
        # the circuit and every caller fails fast with the typed
        # EngineCircuitOpenError (a ConnectionError) instead of hammering a
        # dead socket; a half-open probe after the cooldown heals it.
        from agent_utilities.knowledge_graph.core.engine_breaker import (
            get_breaker,
            wrap_client_with_breaker,
        )

        breaker = get_breaker(endpoint)
        breaker.before_call()  # fast-fail BEFORE attempting a connect when open

        # Autostart governs only the LOCAL engine (CONCEPT:KG-2.58): in sharded
        # mode a remote (tcp://) shard is a hard contract — auto-spawning a
        # local stand-in would silently split that shard's graphs into
        # invisible islands (same fail-loud convention as KG-2.55). The flock
        # host role (host_lock.py) likewise elects a daemon owner for the
        # local engine only.
        autostart_allowed = os.environ.get("EPISTEMIC_GRAPH_AUTOSTART") == "1" and (
            not sharded or is_local_endpoint(endpoint)
        )

        try:
            self._client = SyncEpistemicGraphClient.connect(**connect_kwargs)
        except Exception as initial_e:
            if isinstance(initial_e, OSError | EOFError):
                breaker.record_failure()
            if autostart_allowed:
                import subprocess
                import sys
                import time
                from pathlib import Path

                from .engine_lock import engine_spawn_guard

                sock = connect_kwargs.get("socket_path")
                try:
                    # Single-instance spawn (CONCEPT:KG-2.8 / OS-5.9): serialize all
                    # autostart spawners for this socket behind a flock and
                    # double-check connectivity before spawning. Without this, two
                    # connects racing — or a client spawning while a displaced engine
                    # still holds the socket — produce a split-brain (two engines on
                    # one socket, clobbering the same --persist-dir). The guard is
                    # held across spawn+wait so a concurrent spawner finds the engine
                    # already up on re-check instead of spawning a second one.
                    with engine_spawn_guard(sock):
                        try:
                            # Double check: a peer may have brought it up while we
                            # waited for the guard.
                            self._client = SyncEpistemicGraphClient.connect(
                                **connect_kwargs
                            )
                        except Exception:  # noqa: BLE001 - still down; we spawn
                            self._client = self._autostart_engine(
                                connect_kwargs,
                                sock,
                                engine_insecure,
                                auth_secret,
                                subprocess,
                                sys,
                                time,
                                Path,
                            )
                except ConnectionError:
                    record_shard_connect(endpoint, False)
                    raise
                except Exception as retry_e:
                    if isinstance(retry_e, OSError | EOFError):
                        breaker.record_failure()
                    record_shard_connect(endpoint, False)
                    raise ConnectionError(
                        f"Cannot connect to epistemic-graph Tokio service after auto-start: {retry_e}. "
                        "Ensure the epistemic-graph-server daemon is running."
                    ) from retry_e
            elif sharded:
                # Fail-loud per-shard contract (CONCEPT:KG-2.58, KG-2.55-style):
                # name the shard so the operator fixes the topology instead of
                # half the keyspace quietly degrading.
                record_shard_connect(endpoint, False)
                raise ConnectionError(
                    f"Configured engine shard {endpoint!r} (owner of graph "
                    f"{graph_name!r} by HRW over GRAPH_SERVICE_ENDPOINTS) is "
                    f"unreachable: {initial_e}. Start that shard's "
                    "epistemic-graph-server (or remove it from "
                    "GRAPH_SERVICE_ENDPOINTS — moving a graph between shards "
                    "requires a manual snapshot export/import). Autostart "
                    "applies only to the local unix:// endpoint, never to "
                    "remote shards."
                ) from initial_e
            else:
                record_shard_connect(endpoint, False)
                raise ConnectionError(
                    f"Cannot connect to epistemic-graph Tokio service: {initial_e}. "
                    "Ensure the epistemic-graph-server daemon is running, or set EPISTEMIC_GRAPH_AUTOSTART=1."
                ) from initial_e

        # Connected: close/reset the breaker and guard every subsequent call
        # with it. The proxy is attribute-transparent, and the raw client
        # stays reachable via ``self._client.__wrapped__``. (CONCEPT:OS-5.23)
        breaker.record_success()
        record_shard_connect(endpoint, True)
        self._client = wrap_client_with_breaker(self._client, breaker)

        logger.info(
            "Connected to epistemic-graph Tokio service (graph: %s, endpoint: %s).",
            graph_name,
            endpoint,
        )

        try:
            if self._client:
                # Try to create the graph so tests and dynamic instances don't fail
                # if the graph doesn't exist in the Rust backend yet.
                self._client.tenants.create(graph_name)
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug(f"Tenant graph {graph_name} already exists.")
            else:
                logger.warning(f"Failed to create tenant graph {graph_name}: {repr(e)}")
            pass

        # Bridging local events to the rust service when kafka isn't running
        if (
            os.environ.get("KAFKA_BOOTSTRAP_SERVERS") is None
            or os.environ.get("KAFKA_BOOTSTRAP_SERVERS") == ""
        ):
            self._start_event_bridge()

    def _autostart_engine(
        self,
        connect_kwargs: dict[str, Any],
        sock: str | None,
        engine_insecure: bool,
        auth_secret: str | None,
        subprocess: Any,
        sys: Any,
        time: Any,
        Path: Any,
    ) -> Any:
        """Spawn the local epistemic-graph engine and return a connected client.

        Called ONLY while holding the per-socket spawn guard (see
        :func:`engine_lock.engine_spawn_guard`) and only after a double-checked
        connect confirmed the engine is still down — so this is the sole spawner
        for ``sock``. Mirrors the prior inline autostart: durable ``--persist-dir``
        + checkpoint, the same auth secret the client uses (CONCEPT:OS-5.14).
        """
        from epistemic_graph.client import SyncEpistemicGraphClient

        logger.info(
            "epistemic-graph Tokio service not running. Auto-starting daemon (single-instance guard held)..."
        )
        server_path = str(Path(sys.executable).parent / "epistemic-graph-server")
        cmd = [server_path]
        if sock:
            cmd += ["--socket-path", str(sock)]
        # Durable by default (CONCEPT:KG-2.8 / OS-5.9): snapshot the graphs to disk
        # so an auto-spawned engine warm-restarts from the last checkpoint instead
        # of starting empty. pggraph stays the durable system-of-record; this is
        # the fast local cache.
        persist_dir = os.environ.get("GRAPH_SERVICE_PERSIST_DIR")
        if persist_dir is None:
            try:
                from agent_utilities.core.paths import data_dir

                persist_dir = str(data_dir() / "graph_snapshots")
            except Exception:
                persist_dir = None
        if persist_dir:
            cmd += [
                "--persist-dir",
                persist_dir,
                "--checkpoint-interval",
                os.environ.get("GRAPH_SERVICE_CHECKPOINT_INTERVAL", "60"),
            ]
        # Engine auth (CONCEPT:OS-5.14): the spawned engine gets the SAME secret
        # this client authenticates with (the engine reads GRAPH_SERVICE_AUTH_SECRET).
        # With KG_ENGINE_INSECURE the explicit allow flag keeps refuse-insecure
        # binaries bootable for dev.
        child_env = dict(os.environ)
        if engine_insecure:
            child_env["EPISTEMIC_GRAPH_ALLOW_INSECURE"] = "1"
            child_env.pop("GRAPH_SERVICE_AUTH_SECRET", None)
        else:
            child_env["GRAPH_SERVICE_AUTH_SECRET"] = auth_secret or ""
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=child_env,
        )
        time.sleep(1.0)
        return SyncEpistemicGraphClient.connect(**connect_kwargs)

    def _start_event_bridge(self) -> None:
        """Starts a background bridge to forward local EventBus events to the Rust service."""
        import asyncio
        import threading

        from agent_utilities.knowledge_graph.core.event_backend import get_event_backend

        def bridge_worker() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            eb = get_event_backend()

            async def handle_mutation(topic: str, payload: dict) -> None:
                if "event_type" in payload and "query" in payload:
                    try:
                        if self._client:
                            self._client.apply_mutation(
                                payload["event_type"], payload["query"]
                            )
                    except Exception as exc:
                        logger.error(
                            "Failed to forward mutation to epistemic-graph: %s", exc
                        )

            async def run_subscriber() -> None:
                await eb.subscribe("kg.mutations", "epistemic-bridge", handle_mutation)
                # Keep loop alive to process events
                while True:
                    await asyncio.sleep(3600)

            try:
                loop.run_until_complete(run_subscriber())
            except Exception as e:
                logger.error("Event bridge worker failed: %s", e)

        t = threading.Thread(
            target=bridge_worker, daemon=True, name="EventBridgeWorker"
        )
        t.start()
        logger.info("Started Local-First EventBus bridge to epistemic-graph")

    # ── Node CRUD ────────────────────────────────────────────────────────

    def add_node(self, node_id: str, properties: Any = None, **kwargs: Any) -> None:
        """Add a node with properties to the graph.

        Supports both explicit dict and NX-style kwargs::

            engine.add_node("n1", {"type": "Agent"})
            engine.add_node("n1", type="Agent", name="foo")
        """

        def clean_props(d: Mapping[str, Any]) -> dict[str, Any]:
            import datetime

            from pydantic import BaseModel

            def serialize(val: Any) -> Any:
                if hasattr(val, "model_dump"):
                    try:
                        return val.model_dump(mode="json")
                    except Exception:
                        pass
                if isinstance(val, BaseModel):
                    return val.model_dump(mode="json")
                if isinstance(val, dict):
                    return {k: serialize(v) for k, v in val.items()}
                if isinstance(val, list | tuple | set):
                    return [serialize(v) for v in val]
                if isinstance(val, datetime.datetime):
                    return val.isoformat()
                return val

            return {k: serialize(v) for k, v in d.items()}

        props = dict(properties or {})
        props.update(kwargs)
        props = clean_props(props)
        self._client.nodes.add(node_id, props)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        properties: Any = None,
        **kwargs: Any,
    ) -> None:
        """Add a directed edge between two nodes with properties.

        Supports both explicit dict and NX-style kwargs::

            engine.add_edge("a", "b", {"type": "DEPENDS_ON"})
            engine.add_edge("a", "b", type="DEPENDS_ON")
        """

        def clean_props(d: dict[str, Any]) -> dict[str, Any]:
            import datetime

            from pydantic import BaseModel

            def serialize(val: Any) -> Any:
                if hasattr(val, "model_dump"):
                    try:
                        return val.model_dump(mode="json")
                    except Exception:
                        pass
                if isinstance(val, BaseModel):
                    return val.model_dump(mode="json")
                if isinstance(val, dict):
                    return {k: serialize(v) for k, v in val.items()}
                if isinstance(val, list | tuple | set):
                    return [serialize(v) for v in val]
                if isinstance(val, datetime.datetime):
                    return val.isoformat()
                return val

            return {k: serialize(v) for k, v in d.items()}

        props = dict(properties or {})
        props.update(kwargs)
        props = clean_props(props)

        if self.has_edge(source_id, target_id):
            self.remove_edge(source_id, target_id)

        try:
            self._client.edges.add(source_id, target_id, props)
        except Exception:
            # Ensure nodes exist without overwriting their existing properties
            if not self.has_node(source_id):
                self._client.nodes.add(source_id, {})
            if not self.has_node(target_id):
                self._client.nodes.add(target_id, {})
            self._client.edges.add(source_id, target_id, props)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all of its associated edges."""
        self._client.nodes.remove(node_id)

    def remove_edge(self, source_id: str, target_id: str, key: Any = None) -> None:
        """Remove a directed edge between source and target."""
        self._client.edges.remove(source_id, target_id)

    def has_node(self, node_id: str) -> bool:
        """Check if node_id exists in the graph."""
        return self._client.nodes.has(node_id)

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Check if a directed edge exists between source and target."""
        return self._client.edges.has(source_id, target_id)

    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return self._client.nodes.count()

    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        return self._client.edges.count()

    # ── Graph Algorithms ─────────────────────────────────────────────────

    def topological_sort(self) -> list[str]:
        """Perform topological sort across the graph.

        Raises:
            ValueError: If the graph contains dependency cycles.
        """
        try:
            return self._client.graph.topological_sort()
        except Exception as e:
            raise ValueError("Graph contains cycles") from e

    def find_cycle(self) -> list[str] | None:
        """Detect and return any cycles found within the graph."""
        return self._client.graph.find_cycle()

    def get_shortest_path(self, source_id: str, target_id: str) -> list[str] | None:
        """Get the shortest path between source and target nodes."""
        return self._client.graph.shortest_path(source_id, target_id)

    @staticmethod
    def _bfs_collect(
        start: Any,
        max_depth: int,
        neighbors_fn: Any,
        node_info_fn: Any,
    ) -> list[dict[str, Any]]:
        """Backend-agnostic BFS traversal collecting blast radius results.

        Args:
            start: Starting node identifier (backend-specific).
            max_depth: Maximum traversal depth.
            neighbors_fn: Callable(node) -> iterable of neighbor nodes.
            node_info_fn: Callable(node) -> dict with 'id' and 'type' keys.
        """
        visited: set = {start}
        queue: list[tuple[Any, int]] = [(start, 0)]
        results: list[dict[str, Any]] = []
        while queue:
            curr, depth = queue.pop(0)
            if curr != start:
                info = node_info_fn(curr)
                info["depth"] = depth
                results.append(info)
            if depth < max_depth:
                for neighbor in neighbors_fn(curr):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
        return results

    def get_blast_radius(self, node_id: str, max_depth: int) -> list[dict[str, Any]]:
        """Compute the blast radius dependencies from a starting node.

        Returns a list of dicts: [{'id': str, 'type': str, 'depth': int}]
        """
        nodes = self._client.graph.blast_radius(node_id, max_depth)
        res = []
        for i, nid in enumerate(nodes, start=1):
            res.append({"id": nid, "type": "Node", "depth": min(i, max_depth)})
        return res

    def parse_repository(self, root_path: str) -> None:
        """Parse repository AST natively using the Rust backend."""
        self._client.graph.parse_repository(root_path)

    def parse_file(self, file_path: str, source: bytes) -> dict[str, Any]:
        """Parse one source file's AST natively via the Rust engine.

        Returns the ``ParseFile`` result (symbols + native test-quality metrics
        for Python). The compute layer — not Python — does the AST work.
        """
        return self._client.graph.parse_file(file_path, source)

    def parse_files(self, files: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
        """Batch-parse many files in ONE engine round-trip (CONCEPT:KG-2.16).

        Returns one ``ParseFile``-shaped result per input file, in input order.
        Collapses a per-file parse storm into a single RPC. Requires an engine
        that advertises ``ParseFiles`` — gate on :attr:`supports_batch_parse`.
        """
        return self._client.graph.parse_files(files)

    @property
    def supports_batch_parse(self) -> bool:
        """Whether the connected engine supports the batched ``ParseFiles`` op.

        Cached. Lets callers fall back to per-file ``parse_file`` against an
        engine built before ``ParseFiles`` existed (backward-compatible rollout).
        """
        cached = getattr(self, "_supports_batch_parse", None)
        if cached is None:
            try:
                cached = bool(self._client.supports("ParseFiles"))
            except Exception:
                cached = False
            self._supports_batch_parse = cached
        return cached

    def vf2_subgraph_match(self, pattern: "GraphComputeEngine") -> list[dict[str, str]]:
        """Find all subgraph isomorphism matches from pattern to target graph."""
        from agent_utilities.knowledge_graph.core.engine_breaker import unwrap_client

        # The wire call needs the RAW client of the pattern engine, not its
        # breaker proxy (CONCEPT:OS-5.23).
        return self._client.graph.vf2_subgraph_match(unwrap_client(pattern._client))

    # ── Ledger Operations ────────────────────────────────────────────────

    def get_ledger(self) -> list[str]:
        """Retrieve the mutation transaction ledger log."""
        return self._client.ledger.get()

    def clear_ledger(self) -> None:
        """Clear the mutation transaction ledger log."""
        self._client.ledger.clear()

    @staticmethod
    def _parse_ledger_entry(tx: str) -> tuple[str, list[str]]:
        """Parse a ledger transaction string into (operation, args).

        Shared parser to ensure Rust and Python ledger formats stay in sync.
        """
        parts = tx.split("|")
        if not parts:
            return ("", [])
        return (parts[0], parts[1:])

    def apply_ledger(self, transactions: list[str]) -> None:
        """Replay mutations from a transaction ledger log."""
        self._client.ledger.apply(transactions)

    def flush_ledger_to_backend(self, backend: Any) -> int:
        """Flush the epistemic-graph mutation ledger to a persistent backend.

        Args:
            backend: A GraphBackend instance (e.g., LadybugBackend)

        Returns:
            int: The number of transactions flushed.
        """
        txs = self.get_ledger()
        if not txs:
            return 0

        count = 0
        for tx in txs:
            op, args = self._parse_ledger_entry(tx)
            if op == "AddNode" and len(args) >= 2:
                node_id = args[0]
                props_str = args[1]
                try:
                    props = json.loads(props_str)
                except Exception:
                    props = {}

                node_type = props.get("type", props.get("node_type", "Entity"))
                if node_type == "SYMBOL":
                    symbol_type = props.get("symbol_type", "Unknown")
                    file_path = props.get("file_path", "")
                    ast_hash = props.get("ast_hash", "")
                    name = props.get("name", node_id)
                    metadata_str = json.dumps(props)

                    query = (
                        "MERGE (n:Symbol {id: $id}) "
                        "SET n.type = 'SYMBOL', n.name = $name, "
                        "n.symbol_type = $sym_type, n.file_path = $fp, "
                        "n.ast_hash = $ast_hash, n.metadata = $meta"
                    )
                    try:
                        backend.execute_write(
                            query,
                            parameters={
                                "id": node_id,
                                "name": name,
                                "sym_type": symbol_type,
                                "fp": file_path,
                                "ast_hash": ast_hash,
                                "meta": metadata_str,
                            },
                        )
                    except Exception as e:
                        logger.error(f"Failed to sync Symbol node {node_id}: {e}")
                else:
                    # Generic node fallback
                    query = f"MERGE (n:{node_type} {{id: $id}}) SET n.metadata = $meta"
                    try:
                        backend.execute_write(
                            query,
                            parameters={
                                "id": node_id,
                                "meta": props_str,
                            },
                        )
                    except Exception as e:
                        logger.error(f"Failed to sync Node {node_id}: {e}")
                count += 1
            elif op == "AddEdge" and len(args) >= 3:
                src = args[0]
                tgt = args[1]
                props_str = args[2]
                try:
                    props = json.loads(props_str)
                except Exception:
                    props = {}

                edge_type = props.get("type") or props.get("edge_type") or "RELATED_TO"
                # Sanitize edge type for cypher
                edge_type = edge_type.replace(" ", "_").upper()
                query = (
                    f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) "
                    f"MERGE (a)-[r:{edge_type}]->(b) "
                    "SET r.metadata = $meta"
                )
                try:
                    backend.execute_write(
                        query,
                        parameters={
                            "src": src,
                            "tgt": tgt,
                            "meta": props_str,
                        },
                    )
                except Exception as e:
                    logger.error(f"Failed to sync Edge {src}->{tgt}: {e}")
                count += 1

        self.clear_ledger()
        return count

    # ── Serialization ────────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize the graph to a JSON string representation."""
        nodes = []
        for nid in self._get_all_nodes():
            props = self._get_node_properties(nid)
            nodes.append({"id": nid, "properties": props})

        edges = []
        for src, tgt in self._get_all_edges():
            props = self._get_edge_properties(src, tgt)
            edges.append({"source": src, "target": tgt, "properties": props})

        return json.dumps({"nodes": nodes, "edges": edges}, default=str)

    def from_json(self, json_str: str) -> None:
        """Deserialize and rebuild the graph from a JSON string."""
        data = json.loads(json_str)
        # Clear existing graph nodes/edges via client if possible or just rebuild
        for nid in self._get_all_nodes():
            try:
                self.remove_node(nid)
            except Exception:
                pass

        # Re-add nodes
        for node_data in data.get("nodes", []):
            nid = node_data["id"]
            props = node_data.get("properties", {})
            self.add_node(nid, props)

        # Re-add edges
        for edge_data in data.get("edges", []):
            src = edge_data["source"]
            tgt = edge_data["target"]
            props = edge_data.get("properties", {})
            self.add_edge(src, tgt, props)

    def drop_graph(self) -> bool:
        """Unload this engine's named graph from the running engine (free L1 memory).

        The engine-side per-graph unload behind the KG-2.62 pool eviction hook:
        deletes the tenant's named graph from the engine process. **Lossy unless
        the data is durably mirrored to L3** (tiered backend), so the pool only
        calls this when ``KG_ENGINE_POOL_DROP_ON_EVICT`` is set. Returns True on
        success. Never raises — eviction must not crash a request.
        """
        try:
            self._client.tenants.delete(self.graph_name)
            return True
        except Exception as exc:  # noqa: BLE001 — best-effort unload
            logger.debug("drop_graph(%s) failed: %s", self.graph_name, exc)
            return False

    def to_msgpack(self) -> bytes:
        """Serialize graph to MsgPack binary representation."""
        return self._client.lifecycle.to_msgpack()

    def from_msgpack(self, msgpack_bytes: bytes) -> None:
        """Deserialize graph from MsgPack binary representation."""
        self._client.lifecycle.from_msgpack(msgpack_bytes)

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _get_all_nodes(self) -> list[str]:
        return [nid for nid, _ in self._client.nodes.list()]

    def _get_all_nodes_with_properties(self) -> list[tuple[str, dict[str, Any]]]:
        """Return every ``(node_id, properties)`` pair in a SINGLE round-trip.

        ``nodes.list()`` already returns the full properties alongside each id, so
        a full-graph scan must consume them here rather than issuing one
        ``_get_node_properties`` round-trip per node (an N+1 that cost ~45s on a
        40K-node graph and held the GIL, starving foreground ingestion).
        (CONCEPT:KG-2.8 ingestion throughput)
        """
        out: list[tuple[str, dict[str, Any]]] = []
        for nid, props in self._client.nodes.list():
            if isinstance(props, dict):
                out.append((nid, props))
            elif isinstance(props, str):
                try:
                    parsed = json.loads(props)
                    out.append((nid, parsed if isinstance(parsed, dict) else {}))
                except Exception:
                    out.append((nid, {}))
            else:
                out.append((nid, {}))
        return out

    def _get_node_properties(self, node_id: str) -> dict[str, Any]:
        props = self._client.nodes.properties(node_id)
        if isinstance(props, dict):
            return props
        if isinstance(props, str):
            try:
                parsed = json.loads(props)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _get_all_edges(self) -> list[tuple[str, str]]:
        return [(src, tgt) for src, tgt, _ in self._client.edges.list()]

    def _get_all_edges_with_properties(
        self,
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Return every ``(src, tgt, properties)`` triple in a SINGLE round-trip.

        ``edges.list()`` already ships each edge's (msgpack) properties alongside
        its endpoints, so a full-graph edge scan must decode them locally rather
        than issuing one ``_get_edge_properties`` round-trip per edge — the same
        N+1 the bulk node scan avoids. On a 67K-edge graph the per-edge path cost
        ~100s/scan and was re-run once per assimilation stage.
        (CONCEPT:KG-2.8 throughput)
        """
        import msgpack

        out: list[tuple[str, str, dict[str, Any]]] = []
        for src, tgt, raw in self._client.edges.list():
            props: Any = raw
            if isinstance(raw, bytes | bytearray | list):
                try:
                    props = msgpack.unpackb(bytes(raw), raw=False)
                except Exception:
                    props = {}
            out.append((src, tgt, props if isinstance(props, dict) else {}))
        return out

    # ── Rust-native API wrappers ─────────────────────────────────────────

    def in_degree(self, node_id: str) -> int:
        """Return the in-degree of a node."""
        try:
            return self._client.nodes.in_degree(node_id)
        except Exception:
            return 0

    def out_degree(self, node_id: str) -> int:
        """Return the out-degree of a node."""
        try:
            return self._client.nodes.out_degree(node_id)
        except Exception:
            return 0

    def get_predecessors(self, node_id: str) -> list[str]:
        """Return predecessor node IDs."""
        try:
            return self._client.nodes.predecessors(node_id)
        except Exception:
            return []

    def get_successors(self, node_id: str) -> list[str]:
        """Return successor node IDs."""
        try:
            return self._client.nodes.successors(node_id)
        except Exception:
            return []

    def get_neighbors(self, node_id: str) -> list[str]:
        """Return all neighbor node IDs (predecessors + successors, deduplicated)."""
        try:
            return self._client.nodes.neighbors(node_id)
        except Exception:
            return []

    def node_ids(self) -> list[str]:
        """Return all node IDs in the graph."""
        return self._client.nodes.ids()

    def degree_centrality_all(self) -> list[tuple[str, float]]:
        """Compute degree centrality for all nodes."""
        return self._client.analytics.degree_centrality_all()

    def pagerank(
        self, damping: float = 0.85, iterations: int = 100
    ) -> list[tuple[str, float]]:
        """Compute PageRank scores for all nodes."""
        return self._client.analytics.pagerank(damping, iterations)

    def connected_components(self) -> list[list[str]]:
        """Return weakly connected components as lists of node IDs."""
        return self._client.graph.connected_components()

    def strongly_connected_components(self) -> list[list[str]]:
        """Return strongly connected components via Tarjan's algorithm.

        CONCEPT:KG-2.7 — Tarjan's SCC via Tokio service (GIL-free).
        """
        return self._client.graph.strongly_connected_components()

    def minimum_spanning_tree(self) -> list[tuple[str, str, float]]:
        """Return the minimum spanning tree as (source, target, weight) edges.

        CONCEPT:KG-2.7 — Kruskal's MST via Tokio service (GIL-free).
        """
        return self._client.graph.minimum_spanning_tree()

    def community_detection(self, resolution: float = 1.0) -> list[list[str]]:
        """Detect communities using label propagation."""
        return self._client.graph.community_detection(resolution)

    def community_detect_ephemeral(
        self,
        node_ids: list[str],
        edges: list[tuple[str, str]],
        resolution: float = 1.0,
    ) -> list[list[str]]:
        """Stateless community detection over an inline call graph (KG-2.58).

        Runs detection on the passed nodes/edges in an in-memory throwaway graph on
        the engine — no tenant load, no persistence. Eliminates the bulk-load
        round-trip + comm-tenant churn of the load-then-detect pattern.
        """
        return self._client.graph.community_detect_ephemeral(node_ids, edges, resolution)

    def betweenness_centrality(self) -> list[tuple[str, float]]:
        """Compute betweenness centrality via Brandes' algorithm."""
        return self._client.analytics.betweenness_centrality()

    def graph_coloring(self) -> list[tuple[str, int]]:
        """Greedy graph coloring — assigns colors so no adjacent nodes share a color."""
        return self._client.graph.graph_coloring()

    def compute_similarity_edges(
        self, threshold: float = 0.8
    ) -> list[tuple[str, str, float]]:
        """Compute similarity edges between nodes with embeddings."""
        return self._client.graph.compute_similarity_edges(threshold)

    def prune_by_lifecycle(
        self, max_age_secs: int = 0, min_score: float = 0.0
    ) -> dict[str, Any]:
        """Lifecycle-aware pruning: remove nodes past max_age or below min_score."""
        result_json = self._client.lifecycle.prune(max_age_secs, min_score)
        return json.loads(result_json)

    def get_context_view(self, agent_id: str, max_tokens: int = 8000) -> dict[str, Any]:
        """Get an optimized context view for an agent within a token budget."""
        result_json = self._client.lifecycle.get_context_view(agent_id, max_tokens)
        return json.loads(result_json)

    def batch_update(self, operations: list[dict[str, Any]]) -> dict[str, Any]:
        """Batch update: apply multiple operations in a single service call."""
        result = self._client.lifecycle.batch_update(operations)
        # The client already decodes the MessagePack response to a dict; only
        # decode if a raw JSON string/bytes came back (older transports).
        if isinstance(result, str | bytes | bytearray):
            return json.loads(result)
        return result

    def metrics(self) -> dict[str, Any]:
        """Runtime metrics for monitoring and observability."""
        result_json = self._client.lifecycle.metrics()
        return json.loads(result_json)

    def personalized_pagerank(
        self,
        seed_nodes: dict[str, float] | None = None,
        damping: float = 0.85,
        iterations: int = 100,
    ) -> dict[str, float]:
        """Personalized PageRank with seed teleport nodes."""
        seeds = list((seed_nodes or {}).items())
        result = self._client.analytics.personalized_pagerank(
            seeds, damping, iterations
        )
        return dict(result)

    # ── Batch Operations ─────────────────────────────────────────────────

    def bulk_mutate(self, operations: list[dict[str, Any]]) -> Any:
        """Send a batch of mutations in a single service call.

        Each operation dict should have a ``method`` key and any required
        parameters.  Example::

            engine.bulk_mutate([
                {"method": "AddNode", "node_id": "A", "properties_json": "{}"},
                {"method": "AddEdge", "source_id": "A", "target_id": "B", ...},
            ])
        """
        return self._client.lifecycle.batch_update(operations)

    def evict_lru(self, max_nodes: int = 50_000) -> int:
        """Evict oldest nodes to enforce an in-memory cap.

        Returns the number of evicted nodes.
        """
        return self._client.lifecycle.evict_lru(max_nodes)

    # ── Graph Traversal API ──────────────────────────────────────────────
    # These provide the standard graph traversal interface used across
    # the codebase (owl_bridge, graph_validator, memory_retriever, etc).
    # All hot paths route to the Rust Tokio service; these are thin wrappers.

    @property
    def nodes(self) -> "_NodeView":
        """NX-compatible node view.  Supports iteration, ``in``, and ``[id]``."""
        return _NodeView(self)

    @property
    def edges(self) -> "_EdgeView":
        """NX-compatible edge view.  Supports iteration and ``data=True``."""
        return _EdgeView(self)

    def number_of_nodes(self) -> int:
        """Alias for ``node_count()``."""
        return self.node_count()

    def number_of_edges(self) -> int:
        """Alias for ``edge_count()``."""
        return self.edge_count()

    def degree(self, node_id: str) -> int:
        """Total degree (in + out) of *node_id*."""
        return self.in_degree(node_id) + self.out_degree(node_id)

    def successors(self, node_id: str) -> list[str]:
        """Return successors of *node_id*."""
        return self.get_successors(node_id)

    def predecessors(self, node_id: str) -> list[str]:
        """Return predecessors of *node_id*."""
        return self.get_predecessors(node_id)

    def neighbors(self, node_id: str) -> list[str]:
        """Return neighbors (successors + predecessors) of *node_id*."""
        return self.get_neighbors(node_id)

    def get_edge_data(self, source_id: str, target_id: str, default: Any = None) -> Any:
        """NX-compatible edge data lookup."""
        props = self._get_edge_properties(source_id, target_id)
        if not props:
            return default if not self.has_edge(source_id, target_id) else {0: {}}

        class MultiDiGraphCompatDict(dict):
            def __init__(self, p: dict[str, Any]):
                super().__init__({0: p})
                self._props = p

            def __getitem__(self, key: Any) -> Any:
                if key == 0:
                    return self._props
                return self._props[key]

            def get(self, key: Any, default_val: Any = None) -> Any:
                if key == 0:
                    return self._props
                return self._props.get(key, default_val)

        return MultiDiGraphCompatDict(props)

    def out_edges(self, node_id: str, data: bool = False) -> list:
        """Return outgoing edges from *node_id*.

        When *data* is True, returns ``(src, tgt, props)`` triples.
        """
        succs = self.get_successors(node_id)
        if data:
            return [(node_id, s, self._get_edge_properties(node_id, s)) for s in succs]
        return [(node_id, s) for s in succs]

    def in_edges(self, node_id: str, data: bool = False) -> list:
        """Return incoming edges to *node_id*.

        When *data* is True, returns ``(src, tgt, props)`` triples.
        """
        preds = self.get_predecessors(node_id)
        if data:
            return [(p, node_id, self._get_edge_properties(p, node_id)) for p in preds]
        return [(p, node_id) for p in preds]

    def _get_edge_properties(self, source_id: str, target_id: str) -> dict[str, Any]:
        """Retrieve edge properties between two nodes."""
        props_list = self._client.edges.properties(source_id, target_id)
        if props_list:
            props = props_list[0]
            if isinstance(props, dict):
                return props
            if isinstance(props, str):
                try:
                    parsed = json.loads(props)
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}
        return {}

    def __contains__(self, node_id: str) -> bool:
        """Support ``node_id in engine`` syntax."""
        return self.has_node(node_id)

    def __getitem__(self, node_id: str) -> dict[str, Any]:
        """Support ``engine[node_id]`` to get node properties."""
        return self._get_node_properties(node_id)


class _NodePropertiesProxy(dict):
    def __init__(
        self, engine: GraphComputeEngine, node_id: str, properties: dict[str, Any]
    ):
        super().__init__(properties)
        self._engine = engine
        self._node_id = node_id

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self._engine.add_node(self._node_id, properties=dict(self))

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._engine.add_node(self._node_id, properties=dict(self))

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._engine.add_node(self._node_id, properties=dict(self))


class _EdgePropertiesProxy(dict):
    def __init__(
        self,
        engine: GraphComputeEngine,
        source_id: str,
        target_id: str,
        properties: dict[str, Any],
    ):
        super().__init__(properties)
        self._engine = engine
        self._source_id = source_id
        self._target_id = target_id

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self._engine.add_edge(self._source_id, self._target_id, properties=dict(self))

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._engine.add_edge(self._source_id, self._target_id, properties=dict(self))

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._engine.add_edge(self._source_id, self._target_id, properties=dict(self))


class _NodeView:
    """Lightweight proxy providing NX-style ``graph.nodes`` access."""

    __slots__ = ("_engine",)

    def __init__(self, engine: GraphComputeEngine) -> None:
        self._engine = engine

    def __iter__(self):
        return iter(self._engine.node_ids())

    def __len__(self) -> int:
        return self._engine.node_count()

    def __contains__(self, node_id: str) -> bool:
        return self._engine.has_node(node_id)

    def __getitem__(self, node_id: str) -> dict[str, Any]:
        props = self._engine._get_node_properties(node_id)
        return _NodePropertiesProxy(self._engine, node_id, props)

    def get(self, node_id: str, default: Any = None) -> Any:
        """Support ``graph.nodes.get(id, default)`` pattern."""
        if self._engine.has_node(node_id):
            props = self._engine._get_node_properties(node_id)
            return _NodePropertiesProxy(self._engine, node_id, props)
        return default

    def __call__(self, data: bool = False):
        """Support ``graph.nodes(data=True)`` iteration."""
        if data:
            # One bulk round-trip (props ship with the node list) instead of an
            # ``_get_node_properties`` round-trip per node. (CONCEPT:KG-2.8)
            return [
                (nid, _NodePropertiesProxy(self._engine, nid, props))
                for nid, props in self._engine._get_all_nodes_with_properties()
            ]
        return self._engine.node_ids()


class _EdgeView:
    """Lightweight proxy providing NX-style ``graph.edges`` access."""

    __slots__ = ("_engine",)

    def __init__(self, engine: GraphComputeEngine) -> None:
        self._engine = engine

    def __iter__(self):
        return iter(self._engine._get_all_edges())

    def __len__(self) -> int:
        return self._engine.edge_count()

    def __call__(
        self, data: bool = False, keys: bool = False, default: Any = None, **kwargs: Any
    ):
        """Support ``graph.edges(data=True, keys=True)`` iteration."""
        result: list[Any] = []
        # One bulk round-trip (props ship with the edge list, decoded locally)
        # instead of an ``_get_edge_properties`` round-trip per edge. (KG-2.8)
        for src, tgt, props in self._engine._get_all_edges_with_properties():
            proxy = _EdgePropertiesProxy(self._engine, src, tgt, props)
            if data and keys:
                result.append((src, tgt, 0, proxy))
            elif data:
                result.append((src, tgt, proxy))
            elif keys:
                result.append((src, tgt, 0))
            else:
                result.append((src, tgt))
        return result

    def __getitem__(self, key: Any) -> Any:
        """Support edge properties lookup by tuple key."""
        if not isinstance(key, tuple) or len(key) < 2:
            raise KeyError(key)
        src, tgt = key[0], key[1]
        props = self._engine._get_edge_properties(src, tgt)
        if props is None:
            raise KeyError(key)

        proxy = _EdgePropertiesProxy(self._engine, src, tgt, props)
        if len(key) >= 3:
            return proxy

        class MultiDiGraphCompatEdgeDict(dict):
            def __getitem__(self, k):
                return proxy

            def get(self, k, default=None):
                return proxy

        return MultiDiGraphCompatEdgeDict({0: proxy})
