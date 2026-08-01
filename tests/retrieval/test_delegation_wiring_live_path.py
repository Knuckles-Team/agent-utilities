from __future__ import annotations

"""Wiring regression tests for the delegation path — reachable is not enough.

Every delegation entrypoint (the graph-os MCP ``graph_orchestrate`` tool,
agent-webui/REST, and Telegram/Mattermost via ``messaging/router.py``) funnels
into ``Orchestrator.execute_agent``. Three defects, each in a *different* layer,
independently reduced that path to zero — and every one of them was invisible to
the existing suite because each layer's own unit tests passed in isolation:

1. ``agent_utilities.mcp.child_resilience`` hard-imported the MCP **SDK v2**
   spelling ``MCPError``. On an SDK-v1 runtime (measured: ``mcp`` 1.29.0 in the
   deployed pod) that raises at import time, which makes
   ``agent_utilities.mcp.multiplexer`` unimportable, which takes out
   ``agent_runner._fleet_server_url`` — so NO delegation can resolve a fleet tool
   server at all, and the fleet meta-tools never register.
2. The mandatory evidence compiler assembled multi-hop context with O(N) *serial*
   engine point-reads. An engine round-trip costs the same regardless of payload,
   so the compile blew its fixed wall-clock budget and, under the default
   ``grounding='required'``, failed every run closed before it reached the model.
3. ``_degrade_for_policy`` raised ``GroundingUnavailableError`` with the causing
   exception discarded, so a grounding failure reported a type name and nothing
   else.

These are live-path tests (AGENTS.md "Wire-First"): they exercise the real
entrypoints and assert the wiring holds, rather than re-testing each helper.
"""

import contextvars
import threading

import pytest


class TestFleetToolServerWiring:
    """Delegation must be able to resolve a fleet MCP tool server."""

    def test_child_resilience_binds_a_real_protocol_error_type(self):
        """``MCPError`` must name an exception class under EITHER SDK spelling.

        Guards both failure modes at once: the old ``try/except ImportError``
        left the name bound to ``()`` (silently disabling session-death
        recovery), and the v2-only hard import broke the whole module on v1.
        """
        from agent_utilities.mcp import child_resilience

        assert isinstance(child_resilience.MCPError, type)
        assert issubclass(child_resilience.MCPError, BaseException)

    def test_multiplexer_is_importable(self):
        """The multiplexer must import — the fleet catalog and meta-tools ride on it."""
        import importlib

        module = importlib.import_module("agent_utilities.mcp.multiplexer")
        assert hasattr(module, "MCPMultiplexer")

    def test_fleet_server_url_resolver_is_reachable(self):
        """``_fleet_server_url`` is the seam delegation uses to bind a tool server.

        It must be callable without an ImportError escaping from the multiplexer
        it imports lazily. An unknown server legitimately resolves to "" or
        raises RuntimeError; an ImportError is the regression.
        """
        from agent_utilities.orchestration.agent_runner import _fleet_server_url

        try:
            _fleet_server_url("servicenow-mcp")
        except ImportError as exc:  # pragma: no cover - the regression itself
            pytest.fail(f"fleet URL resolution broke on an import: {exc!r}")
        except RuntimeError:
            pass  # no template / no catalog entry configured in this env


class _CountingGraph:
    """Engine-graph double that counts round-trips per operation.

    Models the property that actually matters: every method here is one
    inter-process round-trip whose cost is independent of payload size.
    """

    def __init__(self, hits, props, edges=None):
        self._hits = hits
        self._props = props
        self._edges = edges or {}
        self.calls: dict[str, int] = {}
        self.threads: set[int] = set()

    def _tick(self, op: str) -> None:
        self.calls[op] = self.calls.get(op, 0) + 1
        self.threads.add(threading.get_ident())

    def query_unified(self, _plan, **_k):
        return []

    def semantic_search(self, _emb, _n=5):
        self._tick("semantic_search")
        return list(self._hits)

    def _get_node_properties(self, nid):
        self._tick("properties")
        return dict(self._props.get(nid, {}))

    def has_node(self, nid):
        self._tick("has_node")
        return nid in self._props

    def has_batch(self, ids):
        self._tick("has_batch")
        return {nid: nid in self._props for nid in ids}

    def get_neighbors(self, nid):
        self._tick("neighbors")
        return list(self._edges.get(nid, []))


def _retriever(graph, monkeypatch):
    """Build a real ``HybridRetriever`` over ``graph``.

    ``__init__`` also constructs the EncPI positional-interaction encoder, which
    plays no part in context assembly and is stubbed here so these tests assert
    only the round-trip contract they are about. It is a genuinely unrelated
    subsystem, not an inconvenient dependency being skipped past.
    """
    from unittest.mock import MagicMock

    from agent_utilities.knowledge_graph.retrieval import hybrid_retriever as hr

    monkeypatch.setattr(hr, "PositionalInteractionEncoder", MagicMock)

    class _Engine:
        pass

    engine = _Engine()
    engine.graph = graph
    engine.backend = None
    engine._search_keyword = lambda q, top_k=10: [  # noqa: ARG005
        {"id": nid, "_score": 1.0} for nid, _ in graph._hits
    ]
    return hr.HybridRetriever(engine)


class TestContextAssemblyRoundTrips:
    """The compile has a FIXED wall-clock budget, so its round-trips must be bounded."""

    @staticmethod
    def _graph(n: int):
        ids = [f"n{i}" for i in range(n)]
        props = {nid: {"id": nid, "content": nid} for nid in ids}
        return _CountingGraph([(nid, 1.0) for nid in ids], props)

    def test_existence_is_one_round_trip_not_one_per_base_node(self, monkeypatch):
        """N base nodes must cost ONE existence call, not N.

        This is the regression that made grounding un-satisfiable: with a ~2.8s
        round-trip floor, ten serial ``has_node`` calls alone exceed the compile
        budget.
        """
        graph = self._graph(10)
        retriever = _retriever(graph, monkeypatch)
        retriever.retrieve_hybrid("anything", context_window=10, skip_quality_gate=True)

        assert graph.calls.get("has_batch", 0) <= 1, graph.calls
        assert graph.calls.get("has_node", 0) == 0, (
            "per-base-node has_node point reads are back: "
            f"{graph.calls} — this is O(N) serial round-trips inside a fixed budget"
        )

    def test_neighbour_expansion_is_capped(self, monkeypatch):
        """A wide frontier must not translate into unbounded round-trips."""
        from agent_utilities.knowledge_graph.retrieval import hybrid_retriever as hr

        graph = self._graph(40)
        retriever = _retriever(graph, monkeypatch)
        retriever.retrieve_hybrid("anything", context_window=40, skip_quality_gate=True)

        assert graph.calls.get("neighbors", 0) <= hr._MAX_NEIGHBOR_EXPANSIONS, (
            f"neighbour expansion exceeded its cap: {graph.calls}"
        )

    def test_neighbour_fetches_are_overlapped(self, monkeypatch):
        """Independent neighbour reads must not be serialized on one thread."""
        graph = self._graph(12)
        retriever = _retriever(graph, monkeypatch)
        retriever.retrieve_hybrid("anything", context_window=12, skip_quality_gate=True)

        if graph.calls.get("neighbors", 0) > 1:
            assert len(graph.threads) > 1, (
                "every neighbour read ran on the calling thread — the round-trips "
                "are still serialized"
            )

    def test_hydration_is_batched_across_base_nodes_not_per_base_node(
        self, monkeypatch
    ):
        """N base nodes falling through to the BFS fallback must hydrate their
        discovered ids in ONE ``properties_batch`` round-trip TOTAL, not one
        per base node (CONCEPT:AU-KG.retrieval.batch-hydrate).

        Regression target: ``_batch_node_properties`` is itself already a
        single-RPC batch, but it used to sit INSIDE the per-base-node loop —
        called once per base node — so N base nodes cost N round-trips for
        hydration that the engine can answer in one. Live-pod measurement
        (``scripts/delegation_probe.py --profile``) showed exactly this
        signature: ``nodes.properties_batch`` called 78 times across one
        delegation, ~5.6s each, dominating the grounding compile's wall time.
        """

        class _FakeNodesNS:
            def __init__(self, props):
                self._props = props
                self.batch_calls = 0
                self.batch_call_sizes: list[int] = []

            def properties_batch(self, ids):
                self.batch_calls += 1
                self.batch_call_sizes.append(len(ids))
                return {nid: dict(self._props.get(nid, {})) for nid in ids}

        class _FakeClient:
            def __init__(self, nodes_ns):
                self.nodes = nodes_ns

        n = 8
        graph = self._graph(n)
        nodes_ns = _FakeNodesNS(graph._props)
        graph._client = _FakeClient(nodes_ns)
        retriever = _retriever(graph, monkeypatch)
        retriever.retrieve_hybrid("anything", context_window=n, skip_quality_gate=True)

        assert nodes_ns.batch_calls == 1, (
            f"properties_batch was called {nodes_ns.batch_calls} times across "
            f"{n} base nodes — hydration is no longer batched across base "
            f"nodes and the per-base-node N+1 is back"
        )
        # The one call carries every base node's discovered id — nothing was
        # dropped by collapsing the per-node calls into one.
        assert nodes_ns.batch_call_sizes == [n]


class TestNeighbourBatchContract:
    """``_neighbors_batch`` must preserve authority and never fake an answer."""

    def test_ambient_context_reaches_the_workers(self, monkeypatch):
        """The verified GraphSession rides a contextvar; workers must inherit it.

        Without a per-task context copy the workers either lose the ambient
        authority (``SessionRequiredError``) or collide on a shared Context
        ("cannot enter context: ... is already entered") — and a collision would
        otherwise look exactly like a graph with no edges.
        """
        marker: contextvars.ContextVar[str] = contextvars.ContextVar(
            "delegation_probe_marker", default=""
        )
        seen: list[str] = []

        class _G:
            def get_neighbors(self, nid):
                seen.append(marker.get())
                return []

        retriever = _retriever(_CountingGraph([], {}), monkeypatch)
        retriever.engine.graph = _G()

        marker.set("verified")
        retriever._neighbors_batch([f"n{i}" for i in range(8)])

        assert seen and all(v == "verified" for v in seen), seen

    def test_a_failed_read_is_not_reported_as_no_neighbours(self, monkeypatch):
        """A raising node must be OMITTED, never mapped onto an empty list.

        Mapping a failure onto ``[]`` is indistinguishable from an isolated node
        and silently corrupts the assembled subgraph.
        """

        class _G:
            def get_neighbors(self, nid):
                if nid == "boom":
                    raise RuntimeError("engine unavailable")
                return ["x"]

        retriever = _retriever(_CountingGraph([], {}), monkeypatch)
        retriever.engine.graph = _G()

        out = retriever._neighbors_batch(["ok1", "boom", "ok2"])
        assert "boom" not in out, "a failed read was recorded as an empty neighbourhood"
        assert out.get("ok1") == ["x"]


class TestGroundingFailureIsDiagnosable:
    """A fail-closed grounding error must carry its cause."""

    def test_degrade_for_policy_chains_the_original_exception(self):
        from agent_utilities.core import contextual_model as cm

        cause = ValueError("the real reason")
        token = cm._grounding_policy.set("required")
        try:
            with pytest.raises(cm.GroundingUnavailableError) as caught:
                cm._degrade_for_policy(
                    [], "test-model", reason="error:ValueError", cause=cause
                )
        finally:
            cm._grounding_policy.reset(token)

        assert caught.value.__cause__ is cause, (
            "GroundingUnavailableError discarded __cause__ — the caller sees a "
            "type name and no traceback"
        )
