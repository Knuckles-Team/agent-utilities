"""Unit tests for the propose-only self-evolution golden loop (CONCEPT:AU-KG.query.vendor-agnostic-traversal)."""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.adaptation.topic_resolver import (
    mark_addressed,
    unresolved_topics,
)
from agent_utilities.knowledge_graph.research.loop_controller import LoopController

# The two tests below run a golden-loop cycle whose ``distill_skills`` stage
# imports ``agent_utilities.numeric`` (the compiled ``epistemic_graph.numeric``
# kernel) transitively; when the kernel is absent the stage's own ImportError
# is caught and appended to the cycle's ``errors`` list rather than raised, so
# ``rep["errors"] == []`` fails with a real, environment-caused non-empty list
# instead of an exception the conftest auto-skip hook could classify. Matches
# the same "package genuinely absent" contract used elsewhere (e.g.
# tests/unit/test_engine_api_coverage.py).
try:
    import epistemic_graph.numeric as _numeric_kernel  # noqa: F401
except ImportError:
    _numeric_kernel = None

_NEEDS_NUMERIC_KERNEL = pytest.mark.skipif(
    _numeric_kernel is None,
    reason=(
        "epistemic_graph.numeric kernel not installed in this environment -- "
        "the golden loop's distill_skills stage cannot import "
        "agent_utilities.numeric."
    ),
)


class _StubEngine:
    """Minimal engine: canned cypher results + records link_nodes calls."""

    def __init__(self, concepts, addressed):
        self._concepts = concepts  # list[(id, name)]
        self._addressed = set(addressed)  # ids with ADDRESSED_BY
        self.links: list[tuple[str, str, str]] = []
        self.backend = object()  # no semantic_search → acquire returns []

    def query_cypher(self, q: str, params: dict | None = None) -> list[dict[str, Any]]:
        params = params or {}
        if "ADDRESSED_BY" in q and "RETURN c.id AS id" in q and "name" not in q:
            return [{"id": i} for i in self._addressed]
        if "MATCH (c:Concept) RETURN c.id AS id, c.name AS name" in q:
            # active_loops() (research/loops.py) generalizes unresolved_topics()
            # and requires each Concept row to carry loop_kind so it can
            # classify research/develop/skill loops — every concept here is a
            # plain research topic.
            return [
                {"id": i, "name": n, "loop_kind": "research"} for i, n in self._concepts
            ]
        if "MATCH (w:WorkItem" in q:
            # active_loops() backs every Concept with a WorkItem
            # (orchestration/work_item.py get_work_item) to track its
            # goal_loop lifecycle — a bare id/name Concept has no backing
            # WorkItem of its own, so synthesize a minimal always-active one
            # keyed by the SAME deterministic id get_work_item looks up
            # (loop_work_item_id: "workitem:loop:<concept id>").
            item_id = str(params.get("id") or "")
            if item_id.startswith("workitem:loop:"):
                return [{"id": item_id, "kind": "goal_loop", "status": "active"}]
            return []
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


@_NEEDS_NUMERIC_KERNEL
def test_run_one_cycle_intake_only_propose_only(monkeypatch):
    # _acquire_resolve (loop_controller.py) builds one embedder per cycle and
    # pings it via bounded_embed BEFORE the per-topic loop. The hermetic unit
    # suite has no reachable embedding endpoint (tests/unit/conftest.py's
    # autouse _hermetic_embeddings fixture blocks create_embedding_model), so
    # the real ping fails closed and appends an "embedding endpoint
    # unavailable" entry to report["errors"] — never reaching the deeper path
    # this test means to exercise (resolve is a no-op because THIS FAKE
    # ENGINE's backend has no semantic_search, a separate, later check inside
    # acquire_for_topic). Wire in a reachable-but-inert embedder — the same seam
    # test_workflow_compiler_embed_resilience.py's ``_patch_embed`` uses — so
    # the ping succeeds and semantic_search's absence is what makes resolve
    # a no-op, not an unreachable endpoint.
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.semantic.make_embed_fn",
        lambda *_a, **_k: lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.research.search.bounded_embed",
        lambda _embed_fn, _text, _timeout: [0.1, 0.2, 0.3],
    )

    eng = _StubEngine(concepts=[("c:1", "A"), ("c:2", "B")], addressed=[])
    # acquire returns [] (no semantic_search) → resolve does nothing, but the
    # cycle must complete cleanly and stay propose-only.
    rep = LoopController(eng).run_one_cycle(synthesize=False, distill=False)
    assert rep["propose_only"] is True
    assert rep["topics_intake"] == 2
    assert rep["topics_resolved"] == 0
    assert rep["errors"] == []


@_NEEDS_NUMERIC_KERNEL
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

    # The predicted-edge pass now commits its ObjectCentricGraphSlice through
    # ingest_graph_slice (D-61-4), which correctly refuses a bare stub engine.
    # This test is about the mining SUMMARY, not the writeback, so stub the
    # writer rather than let an unrelated EngineUnavailable land in errors --
    # it previously only "passed" because ingest_envelope silently accepted the
    # wrong payload shape, which is the defect itself.
    from agent_utilities.knowledge_graph.ingestion import envelope_ingest

    monkeypatch.setattr(
        envelope_ingest,
        "ingest_graph_slice",
        lambda *a, **k: {"status": "success", "envelope_id": "env:test"},
    )

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


def test_mine_predicted_edges_emits_real_neural_relation_prediction_live_path(
    monkeypatch,
):
    """CONCEPT:AU-KG.ingest.semantic-event-contract — before this wiring,
    ``NeuralRelationPrediction`` had a model class, an ontology concept, and a
    SHACL shape but ZERO producers: nothing outside ``tests/`` constructed one
    and fed it into the ingestion pipeline. This drives the REAL
    ``_mine_predicted_edges`` link-prediction pass end to end (mocking only the
    ``_invoke``/``ingest_envelope`` boundaries, same as the sibling test above)
    and asserts a real ``NeuralRelationPrediction`` was constructed, wrapped in
    a real ``ObjectCentricGraphSlice``, and committed as a real
    ``ChangeEnvelope`` — not a standalone unit test of the semantic-event
    model."""
    import json as _json

    import agent_utilities.mcp.tools.engine_surface_tools as engine_surface_tools
    from agent_utilities.knowledge_graph.ingestion import envelope_ingest

    def fake_invoke(*, surface, action, graph, candidates, params):
        if surface == "mining" and action == "associate":
            return _json.dumps(
                {"surface": "mining", "action": "associate", "result": {"rules": []}}
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

    captured_slices = []

    # Patch the writer this path ACTUALLY uses. An ObjectCentricGraphSlice
    # envelope carries a {entities, relationships} typed_payload, so it must be
    # committed with ingest_graph_slice: handing it to ingest_envelope collapses
    # every entity onto ONE untyped node while still reporting success (the same
    # defect D-61-4 fixed in graph_mine's OCEL commit). Patching the writer the
    # code no longer calls would silently stop covering this path.
    def fake_ingest_graph_slice(engine, connector, entities, relationships, **kwargs):
        captured_slices.append((connector, entities, relationships))
        return {"status": "success", "envelope_id": "env:test"}

    monkeypatch.setattr(envelope_ingest, "ingest_graph_slice", fake_ingest_graph_slice)

    eng = _StubEngine([], [])
    rep = LoopController(eng)._run_mine_discovery()

    events = rep["predicted_edges"]["semantic_events"]
    assert events["emitted"] == 1
    assert events["status"] == "success"
    assert rep["errors"] == []

    assert len(captured_slices) == 1
    connector, entities, _relationships = captured_slices[0]
    assert connector == "ocel"
    neural_entities = [
        e for e in entities if e["node_type"] == "NeuralRelationPrediction"
    ]
    assert len(neural_entities) == 1
    neural_entity = neural_entities[0]
    assert neural_entity["prediction_score"] == 0.9
    assert neural_entity["decision_status"] == "proposed"
    assert neural_entity["evidence_refs"] == ["concept:cA", "concept:cC"]
    business_objects = {
        e["source_record_id"] for e in entities if e["node_type"] == "BusinessObject"
    }
    assert business_objects == {"concept:cA", "concept:cC"}


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


def test_run_one_cycle_placement_control_uses_typed_opt_in(monkeypatch):
    import agent_utilities.knowledge_graph.research.placement_mining as placement
    from agent_utilities.core.config import config

    calls: list[tuple[object, bool]] = []

    def fake_placement_control(engine, *, enabled, **_kwargs):
        calls.append((engine, enabled))
        return {"enabled": True, "proposals": 0}

    monkeypatch.setattr(placement, "placement_control_loop", fake_placement_control)
    monkeypatch.setattr(config, "placement_control_loop_enabled", True)

    eng = _StubEngine([], [])
    report = LoopController(eng).run_one_cycle(
        assimilate=False,
        synthesize=False,
        distill=False,
        reason=False,
        breadth=False,
        mine_discovery=False,
        belief_revision=False,
        insight_validation=False,
        trace_mining=False,
    )
    assert calls == [(eng, True)]
    assert report["placement_control"] == {"enabled": True, "proposals": 0}


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

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
        *,
        session: Any = None,
    ) -> None:
        # Matches IntelligenceGraphEngine.add_node's real signature
        # (core/engine.py): node_type is a required positional arg between
        # node_id and properties, not a 2-arg (node_id, properties) call —
        # _run_belief_revision calls add_node(node_id, "BeliefRevisionProposal",
        # properties=...), which this stub's previous 2-arg signature couldn't
        # accept (TypeError, silently swallowed by the caller's best-effort
        # per-item persistence try/except, so persisted_nodes stayed 0).
        if any(node_id.startswith(f) for f in self._fail_add_node_ids):
            raise RuntimeError(f"persist failed for {node_id}")
        # Mirrors IntelligenceGraphEngine.add_node's own real behavior: it
        # folds node_type into the persisted properties under the canonical
        # 'node_type' key (the retired 'type' key raises ValueError there).
        stored = dict(properties or {})
        stored["node_type"] = node_type
        self.added_nodes.append((node_id, stored))


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
    assert props["node_type"] == "BeliefRevisionProposal"
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
        node_id.startswith("BeliefRevisionProposal:") for node_id, _ in eng.added_nodes
    )
    assert all(
        "belief:a" != node_id and "belief:b" != node_id
        for node_id, _ in eng.added_nodes
    )


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
