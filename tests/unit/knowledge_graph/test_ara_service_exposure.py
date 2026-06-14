"""ARA exposure — service dispatch + MCP tool + REST router single-SoT (KG-2.80)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.research.ara import ARAService


class _Engine:
    def __init__(self, concepts=None, artifacts=None):
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []
        self._concepts = concepts or []
        self._artifacts = artifacts or []

    def add_node(self, nid, ntype, properties=None):
        self.nodes[nid] = {"type": ntype, **(properties or {})}

    def add_edge(self, src, dst, rel_type="", **props):
        self.edges.append((src, dst, rel_type))

    def query_cypher(self, q, params=None):
        if "research_artifact" in q and "claim" in q:
            return [{"id": "claim:p:0", "statement": "c"}]
        if "research_artifact" in q:
            return self._artifacts
        return self._concepts


class _Generator:
    class _Legacy:
        article_id = "p"
        title = "Paper"
        summary = "s"
        key_contributions = ["a claim about owl reasoning"]
        methods = ["a method"]
        suggested_experiments = ["an experiment"]
        authors: list[str] = []
        source_url = ""

    def generate_paper_artifact(self, article_id, target_codebase=None):
        return self._Legacy()


# ── service dispatch ────────────────────────────────────────────────────────


def test_unknown_action_returns_error():
    out = ARAService(_Engine()).run("frobnicate")
    assert out["error"].startswith("unknown action")


def test_capture_dispatch_flushes_event_with_provenance():
    eng = _Engine()
    out = ARAService(eng).run(
        "capture", article_id="p", text="we find X", provenance="user", actor="u1"
    )
    assert out["action"] == "capture" and out["flushed"] == 1
    assert out["event"]["provenance"] == "user"
    assert any(node["type"] == "exploration_node" for node in eng.nodes.values())


def test_compile_then_review_via_service(monkeypatch):
    eng = _Engine()
    # patch the compiler's default generator so no live extractor is needed
    import agent_utilities.knowledge_graph.research.ara.compiler as comp

    monkeypatch.setattr(
        comp.ARACompiler, "_default_generator", lambda self: _Generator()
    )
    svc = ARAService(eng, ground_fn=lambda s: ["concept:owl"])
    compiled = svc.run("compile", article_id="p")
    assert compiled["action"] == "compile"
    assert compiled["report"]["n_claims"] == 1
    # review recompiles (no materialize) then seals; claim is grounded → L1 passes
    reviewed = svc.run("review", article_id="p", level="L1")
    assert reviewed["report"]["passed"] is True


def test_reason_dispatch_returns_harvest_shape():
    out = ARAService(_Engine()).run("reason", query="owl")
    # no graph edges → empty harvest, but the shape is stable and error-free-ish
    assert out["action"] == "reason"
    assert "new_topics" in out and "inferred_edges" in out


def test_list_and_get_reads():
    eng = _Engine(artifacts=[{"id": "research_artifact:p", "name": "Paper"}])
    svc = ARAService(eng)
    listed = svc.run("list")
    assert listed["count"] == 1
    got = svc.run("get", article_id="p")
    assert got["action"] == "get" and got["artifact"]["id"] == "research_artifact:p"


# ── single source of truth: MCP tool + REST router both dispatch the service ─


def test_mcp_tool_and_route_registered():
    from agent_utilities.mcp import kg_server

    # ACTION_TOOL_ROUTES carries the REST path → /research/artifact is auto-mounted
    assert kg_server.ACTION_TOOL_ROUTES.get("research_artifact") == "/research/artifact"


def test_gateway_router_exposes_research_paths():
    from agent_utilities.gateway.research_api import research_router

    paths = {r.path for r in research_router.routes}
    assert "/research/reason" in paths
    assert "/research/compile" in paths
    assert "/research/review" in paths
    assert "/research/artifacts" in paths
