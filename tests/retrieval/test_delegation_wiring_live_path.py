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

import threading

import pytest


class TestFleetToolServerWiring:
    """Delegation must be able to resolve a fleet MCP tool server."""

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
