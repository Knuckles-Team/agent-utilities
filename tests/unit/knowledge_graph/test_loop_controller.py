"""Unit tests for the propose-only self-evolution golden loop (CONCEPT:AU-KG.query.vendor-agnostic-traversal)."""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.adaptation.topic_resolver import (
    mark_addressed,
    unresolved_topics,
)
from agent_utilities.knowledge_graph.research.loop_controller import LoopController


class _StubEngine:
    """Minimal engine: canned cypher results + records link_nodes calls."""

    def __init__(self, concepts, addressed):
        self._concepts = concepts  # list[(id, name)]
        self._addressed = set(addressed)  # ids with ADDRESSED_BY
        self.links: list[tuple[str, str, str]] = []
        self.backend = object()  # no semantic_search → acquire returns []

    def query_cypher(self, q: str, params: dict | None = None) -> list[dict[str, Any]]:
        if "ADDRESSED_BY" in q and "RETURN c.id AS id" in q and "name" not in q:
            return [{"id": i} for i in self._addressed]
        if "MATCH (c:Concept) RETURN c.id AS id, c.name AS name" in q:
            return [{"id": i, "name": n} for i, n in self._concepts]
        return []

    def link_nodes(self, source_id, target_id, rel_type, properties=None):
        self.links.append((source_id, target_id, rel_type))


def test_unresolved_topics_subtracts_addressed():
    eng = _StubEngine(
        concepts=[("c:1", "A"), ("c:2", "B"), ("c:3", "C")],
        addressed=["c:2"],
    )
    topics = unresolved_topics(eng, limit=10)
    ids = {t["id"] for t in topics}
    assert ids == {"c:1", "c:3"}  # c:2 is already addressed → excluded


def test_mark_addressed_writes_both_directions():
    eng = _StubEngine([], [])
    n = mark_addressed(eng, "c:1", ["src:a", "src:b", "c:1"], source="t")
    assert n == 2  # self-link (c:1) skipped
    rels = {(s, t, r) for s, t, r in eng.links}
    assert ("src:a", "c:1", "ADDRESSES") in rels
    assert ("c:1", "src:a", "ADDRESSED_BY") in rels


def test_run_breadth_self_configures_from_workspace_yml(monkeypatch):
    """Live path: with no KG_BREADTH_* roots, breadth auto-discovers the ecosystem
    from the XDG workspace.yml — so assimilate always has a codebase to compare
    research against, zero-config (CONCEPT:AU-KG.query.vendor-agnostic-traversal)."""
    import agent_utilities.core.workspace_config as wc
    import agent_utilities.knowledge_graph.assimilation as assim
    from agent_utilities.knowledge_graph.assimilation.breadth_ingest import (
        BreadthReport,
    )

    monkeypatch.delenv("KG_BREADTH_LIBRARY_ROOTS", raising=False)
    monkeypatch.delenv("KG_BREADTH_REPO_ROOTS", raising=False)
    monkeypatch.setattr(
        wc, "workspace_project_roots", lambda *a, **k: ["/eco/repo-a", "/eco/repo-b"]
    )
    captured: dict = {}

    def fake_run(engine, *, library_roots=None, repo_roots=None, **kw):
        captured["repos"] = repo_roots
        return BreadthReport()

    monkeypatch.setattr(assim, "run_breadth_ingest", fake_run)

    rep = LoopController(_StubEngine([], []))._run_breadth()
    assert captured["repos"] == ["/eco/repo-a", "/eco/repo-b"]
    assert not rep.get("skipped")


def test_run_one_cycle_intake_only_propose_only():
    eng = _StubEngine(concepts=[("c:1", "A"), ("c:2", "B")], addressed=[])
    # acquire returns [] (no semantic_search) → resolve does nothing, but the
    # cycle must complete cleanly and stay propose-only.
    rep = LoopController(eng).run_one_cycle(synthesize=False, distill=False)
    assert rep["propose_only"] is True
    assert rep["topics_intake"] == 2
    assert rep["topics_resolved"] == 0
    assert rep["errors"] == []


def test_run_one_cycle_intake_papers_runs_research_pipeline(monkeypatch):
    """Caller-supplied papers trigger the unified intake stage (research-pipeline
    runner) before assimilate (CONCEPT:AU-KG.research.research-intelligence-loop)."""
    from types import SimpleNamespace

    import agent_utilities.automation.research_pipeline as rp

    seen: dict = {}

    class _FakeRunner:
        def __init__(self, engine=None, **kw):
            seen["engine"] = engine

        async def run_daily_pipeline(self, papers=None):
            seen["papers"] = papers
            return SimpleNamespace(
                papers_discovered=len(papers or []),
                papers_relevant=1,
                papers_marginal=0,
                papers_already_known=0,
                owl_inferences=0,
                errors=[],
            )

    monkeypatch.setattr(rp, "ResearchPipelineRunner", _FakeRunner)

    eng = _StubEngine([], [])
    rep = LoopController(eng).run_one_cycle(
        papers=[{"id": "2606.09498", "title": "Self-Harness"}],
        assimilate=False,
        synthesize=False,
        breadth=False,
    )
    assert rep["intake_papers"]["papers_discovered"] == 1
    assert seen["papers"][0]["id"] == "2606.09498"
    assert rep["errors"] == []


# ── Discovery-flywheel mining stage (CONCEPT:AU-KG.evolution.mining-flywheel) ──────────


def test_mine_discovery_association_rule_and_link_prediction(monkeypatch):
    """The mining stage mirrors epistemic-graph's docs/mining.md concept↔capability
    example: mocking the ``_invoke`` call boundary (the same one graph_mine/graph_learn
    MCP tools use) to return a concept↔capability association rule + a predicted
    concept↔concept edge, asserting the compact summary carries both through and
    the cycle never raises."""
    import json as _json

    import agent_utilities.mcp.tools.engine_surface_tools as engine_surface_tools

    def fake_invoke(*, surface, action, graph, candidates, params):
        if surface == "mining" and action == "associate":
            return _json.dumps(
                {
                    "surface": "mining",
                    "action": "associate",
                    "result": {
                        "rules": [
                            {
                                "antecedent": ["concept:cA", "concept:cB"],
                                "consequent": ["capability:capZ"],
                                "confidence": 1.0,
                                "lift": 1.67,
                            }
                        ],
                        "written_back": 1,
                    },
                }
            )
        if surface == "mining" and action == "anomaly":
            return _json.dumps(
                {"surface": "mining", "action": "anomaly", "result": {"rows": []}}
            )
        if surface == "graphlearn" and action == "fit":
            return _json.dumps(
                {
                    "surface": "graphlearn",
                    "action": "fit",
                    "result": {"model": {"basis": "chebyshev"}, "n_nodes": 3},
                }
            )
        if surface == "graphlearn" and action == "predict":
            return _json.dumps(
                {
                    "surface": "graphlearn",
                    "action": "predict",
                    "result": {
                        "predicted": [
                            {"src": "concept:cA", "dst": "concept:cC", "score": 0.9}
                        ]
                    },
                }
            )
        raise AssertionError(f"unexpected invoke {surface}/{action}")

    monkeypatch.setattr(engine_surface_tools, "_invoke", fake_invoke)

    eng = _StubEngine([], [])  # query_cypher falls through to [] → anomaly skipped
    rep = LoopController(eng)._run_mine_discovery()

    assert rep["association_rules"]["count"] == 1
    rule = rep["association_rules"]["examples"][0]
    assert rule["antecedent"] == ["concept:cA", "concept:cB"]
    assert rule["consequent"] == ["capability:capZ"]
    assert rule["confidence"] == 1.0

    assert rep["predicted_edges"]["count"] == 1
    assert rep["predicted_edges"]["examples"][0]["dst"] == "concept:cC"

    assert rep["anomalies"] == {"count": 0, "examples": []}
    assert rep["errors"] == []


def test_mine_discovery_degrades_cleanly_on_mining_error(monkeypatch):
    """When the engine build has no mining surface (or the call otherwise fails),
    ``_invoke`` returns an error payload as data (never raises) — the mining stage
    must surface that as a captured error and an empty summary, never blow up the
    cycle."""
    import json as _json

    import agent_utilities.mcp.tools.engine_surface_tools as engine_surface_tools

    def fake_invoke(*, surface, action, graph, candidates, params):
        return _json.dumps(
            {
                "surface": surface,
                "action": action,
                "degraded": True,
                "error": f"engine surface {surface!r} is not available in this build",
            }
        )

    monkeypatch.setattr(engine_surface_tools, "_invoke", fake_invoke)

    eng = _StubEngine([], [])
    rep = LoopController(eng)._run_mine_discovery()

    assert rep["association_rules"] == {"count": 0, "examples": []}
    assert rep["anomalies"] == {"count": 0, "examples": []}
    assert rep["predicted_edges"] == {"count": 0, "examples": []}
    assert len(rep["errors"]) == 2  # association + predicted_edges:fit both degraded


def test_run_one_cycle_mine_discovery_defaults_true_and_can_disable(monkeypatch):
    """``mine_discovery`` defaults to ``config.kg_loop_mine_discovery`` (True) and can
    be explicitly disabled per-call, mirroring the other ``kg_loop_*`` gated stages."""
    import agent_utilities.knowledge_graph.research.loop_controller as loop_controller

    calls: list[bool] = []

    def fake_mine_discovery(self):
        calls.append(True)
        return {
            "association_rules": {"count": 0, "examples": []},
            "anomalies": {"count": 0, "examples": []},
            "predicted_edges": {"count": 0, "examples": []},
            "errors": [],
        }

    monkeypatch.setattr(
        loop_controller.LoopController, "_run_mine_discovery", fake_mine_discovery
    )

    eng = _StubEngine([], [])
    rep_default = LoopController(eng).run_one_cycle(
        assimilate=False, synthesize=False, distill=False, reason=False, breadth=False
    )
    assert len(calls) == 1
    assert rep_default["mine_discovery"] is not None

    calls.clear()
    rep_disabled = LoopController(eng).run_one_cycle(
        assimilate=False,
        synthesize=False,
        distill=False,
        reason=False,
        breadth=False,
        mine_discovery=False,
    )
    assert len(calls) == 0
    assert rep_disabled["mine_discovery"] is None


# ── Belief revision / confidence propagation stage (CONCEPT:AU-KG.maintenance.
# confidence-propagation-belief-revision, workstream C2) ──────────────────────────


class _BeliefStubEngine:
    """Minimal engine stub: canned ``Belief`` rows + records ``add_node`` calls.

    ``query_cypher`` raises when ``fail_query`` is set (degrade-path test);
    ``add_node`` raises when a node id is in ``fail_add_node_ids`` (per-item
    persistence-failure tolerance test).
    """

    def __init__(
        self,
        belief_rows: list[dict[str, Any]],
        *,
        fail_query: bool = False,
        fail_add_node_ids: frozenset[str] = frozenset(),
    ) -> None:
        self._belief_rows = belief_rows
        self._fail_query = fail_query
        self._fail_add_node_ids = fail_add_node_ids
        self.added_nodes: list[tuple[str, dict[str, Any]]] = []
        self.backend = object()

    def query_cypher(self, q: str, params: dict | None = None) -> list[dict[str, Any]]:
        if self._fail_query:
            raise RuntimeError("engine unreachable")
        if "MATCH (b:Belief)" in q:
            return list(self._belief_rows)
        return []

    def add_node(self, node_id: str, properties: dict[str, Any]) -> None:
        if any(node_id.startswith(f) for f in self._fail_add_node_ids):
            raise RuntimeError(f"persist failed for {node_id}")
        self.added_nodes.append((node_id, properties))


def _belief_row(
    belief_id: str,
    statement: str,
    confidence: float,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "id": belief_id,
        "statement": statement,
        "confidence": confidence,
        "evidence_node_ids": [],
        "supported_by_node_ids": [],
        "contradicted_by_node_ids": [],
        "last_reviewed": "2020-01-01T00:00:00+00:00",
    }
    row.update(extra)
    return row


def test_run_belief_revision_recomputes_and_persists_proposals():
    eng = _BeliefStubEngine(
        [
            _belief_row(
                "belief:a",
                "the new caching layer clearly improves database performance",
                0.8,
            ),
            _belief_row(
                "belief:b",
                "the new caching layer clearly degrades database performance",
                0.8,
            ),
        ]
    )
    rep = LoopController(eng)._run_belief_revision()

    assert rep["skipped"] is False
    assert rep["beliefs_scanned"] == 2
    assert rep["revisions"] == 2
    assert rep["persisted_nodes"] == 2
    assert rep["errors"] == []
    assert len(eng.added_nodes) == 2

    node_id, props = eng.added_nodes[0]
    assert node_id.startswith("BeliefRevisionProposal:")
    assert props["type"] == "BeliefRevisionProposal"
    assert props["status"] == "proposal"
    assert props["belief_id"] in {"belief:a", "belief:b"}
    assert "reasoning_trace" in props
    assert props["reasoning_trace"]  # non-empty explainability record

    # Mutually-contradicting, similar-strength beliefs both lose confidence —
    # never a mutation of the live Belief node (only new proposal nodes exist).
    example_by_id = {ex["belief_id"]: ex for ex in rep["examples"]}
    assert example_by_id["belief:a"]["new_confidence"] < 0.8
    assert example_by_id["belief:b"]["new_confidence"] < 0.8


def test_run_belief_revision_skips_with_fewer_than_two_beliefs():
    eng = _BeliefStubEngine([_belief_row("belief:a", "x is true", 0.6)])
    rep = LoopController(eng)._run_belief_revision()
    assert rep["skipped"] is True
    assert rep["reason"] == "fewer than 2 Belief nodes"
    assert rep["beliefs_scanned"] == 1
    assert eng.added_nodes == []


def test_run_belief_revision_skips_cleanly_with_zero_beliefs():
    eng = _BeliefStubEngine([])
    rep = LoopController(eng)._run_belief_revision()
    assert rep["skipped"] is True
    assert rep["beliefs_scanned"] == 0


def test_run_belief_revision_degrades_cleanly_on_query_error():
    eng = _BeliefStubEngine([], fail_query=True)
    rep = LoopController(eng)._run_belief_revision()  # must not raise
    assert rep["skipped"] is True
    assert rep["reason"] == "query failed"
    assert len(rep["errors"]) == 1


def test_run_belief_revision_tolerates_one_malformed_row():
    eng = _BeliefStubEngine(
        [
            _belief_row("belief:a", "Caching improves performance", 0.7),
            _belief_row("belief:b", "Caching degrades performance", 0.7),
            # Malformed: confidence cannot be parsed as a float.
            _belief_row("belief:bad", "some claim", "not-a-number"),
        ]
    )
    rep = LoopController(eng)._run_belief_revision()
    assert rep["skipped"] is False
    assert rep["beliefs_scanned"] == 2  # the malformed row was dropped, not fatal
    assert any("belief_revision:parse" in e for e in rep["errors"])


def test_run_belief_revision_tolerates_persist_failure_per_item():
    eng = _BeliefStubEngine(
        [
            _belief_row("belief:a", "Caching improves performance", 0.7),
            _belief_row("belief:b", "Caching degrades performance", 0.7),
        ],
        fail_add_node_ids=frozenset({"BeliefRevisionProposal:belief:a"}),
    )
    rep = LoopController(eng)._run_belief_revision()  # must not raise
    assert rep["skipped"] is False
    assert rep["revisions"] == 2
    assert rep["persisted_nodes"] == 1  # 'a' failed to persist, 'b' still did
    assert any("belief_revision:persist" in e for e in rep["errors"])


def test_run_belief_revision_never_calls_update_on_the_live_belief():
    """Propose-only doctrine: the stage must only ever ADD new
    ``BeliefRevisionProposal`` nodes, never attempt to mutate the canonical
    ``Belief`` node (no ``update_node``/similar call on the original id)."""
    eng = _BeliefStubEngine(
        [
            _belief_row("belief:a", "Caching improves performance", 0.7),
            _belief_row("belief:b", "Caching degrades performance", 0.7),
        ]
    )
    LoopController(eng)._run_belief_revision()
    assert not hasattr(eng, "update_node")
    assert all(
        node_id.startswith("BeliefRevisionProposal:")
        for node_id, _ in eng.added_nodes
    )
    assert all("belief:a" != node_id and "belief:b" != node_id for node_id, _ in eng.added_nodes)


def test_run_belief_revision_respects_propose_only_false():
    eng = _BeliefStubEngine(
        [
            _belief_row("belief:a", "Caching improves performance", 0.7),
            _belief_row("belief:b", "Caching degrades performance", 0.7),
        ]
    )
    rep = LoopController(eng, propose_only=False)._run_belief_revision()
    assert rep["skipped"] is False
    assert rep["revisions"] == 2
    assert rep["persisted_nodes"] == 0  # nothing written when propose_only=False
    assert eng.added_nodes == []


def test_run_one_cycle_belief_revision_defaults_true_and_can_disable(monkeypatch):
    """``belief_revision`` defaults to ``config.kg_loop_belief_revision`` (True)
    and can be explicitly disabled per-call, mirroring ``mine_discovery``."""
    import agent_utilities.knowledge_graph.research.loop_controller as loop_controller

    calls: list[bool] = []

    def fake_belief_revision(self):
        calls.append(True)
        return {"skipped": True, "reason": "fewer than 2 Belief nodes", "errors": []}

    monkeypatch.setattr(
        loop_controller.LoopController, "_run_belief_revision", fake_belief_revision
    )

    eng = _StubEngine([], [])
    rep_default = LoopController(eng).run_one_cycle(
        assimilate=False,
        synthesize=False,
        distill=False,
        reason=False,
        breadth=False,
        mine_discovery=False,
    )
    assert len(calls) == 1
    assert rep_default["belief_revision"] is not None

    calls.clear()
    rep_disabled = LoopController(eng).run_one_cycle(
        assimilate=False,
        synthesize=False,
        distill=False,
        reason=False,
        breadth=False,
        mine_discovery=False,
        belief_revision=False,
    )
    assert len(calls) == 0
    assert rep_disabled["belief_revision"] is None
