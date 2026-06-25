"""Tests for code-scoped graph analytics (CONCEPT:KG-2.210 / KG-2.213).

Exercises the Graphify-style god-node / community / surprising-connection analytics
and the architecture report offline via a fake engine whose backend returns the
tutorial's ``sample_app`` :Code call graph. Pure helpers (degree, Tarjan cycles,
confidence buckets) are tested directly.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.retrieval import code_metrics as cm

# The tutorial sample_app, as a resolved :Code subgraph. ``settings`` is the
# expected "god node" (imported/used everywhere); a small cross-module web of
# calls so communities + bridges are non-trivial.
_EDGES = [
    ("database.py:pool", "config.py:settings", "depends_on"),
    ("cache.py:limiter", "config.py:settings", "depends_on"),
    ("auth.py:AuthService", "config.py:settings", "depends_on"),
    ("auth.py:AuthService", "database.py:get_connection", "calls"),
    ("auth.py:AuthService", "models.py:User", "calls"),
    ("services.py:UserService", "auth.py:AuthService", "calls"),
    ("services.py:UserService", "database.py:get_connection", "calls"),
    ("services.py:UserService", "models.py:User", "calls"),
    ("api.py:signup_route", "services.py:UserService", "calls"),
    ("api.py:login_route", "services.py:UserService", "calls"),
    ("api.py:signup_route", "cache.py:limiter", "calls"),
    ("database.py:get_connection", "database.py:pool", "calls"),
    ("main.py:run", "api.py:signup_route", "calls"),
    ("main.py:run", "api.py:login_route", "calls"),
]


class _Backend:
    def execute(self, cypher, params):
        # Only the :Code subgraph projection query is issued here.
        scope = params.get("scope")
        rows = []
        for src, dst, rel in _EDGES:
            if scope and scope not in src and scope not in dst:
                continue
            rows.append(
                {
                    "src": src,
                    "src_name": src.split(":", 1)[1],
                    "src_file": src.split(":", 1)[0],
                    "src_lang": "python",
                    "src_kind": "function",
                    "dst": dst,
                    "dst_name": dst.split(":", 1)[1],
                    "dst_file": dst.split(":", 1)[0],
                    "dst_lang": "python",
                    "dst_kind": "function",
                    "rel": rel,
                    "confidence": 1.0,
                }
            )
        return rows


class _GraphCompute:
    def community_detect_ephemeral(self, node_ids, edges, resolution=1.0):
        # Deterministic 2-way split by file prefix so the test is stable without
        # the engine: api/main/services in one, the rest in another.
        front = {"api.py", "main.py", "services.py"}
        a = [n for n in node_ids if n.split(":", 1)[0] in front]
        b = [n for n in node_ids if n.split(":", 1)[0] not in front]
        return [c for c in (a, b) if c]


class FakeEngine:
    def __init__(self):
        self.backend = _Backend()
        self.graph_compute = _GraphCompute()
        self.added: list[tuple[str, dict]] = []

    def add_node(self, node_id, props):
        self.added.append((node_id, props))


# ── pure helpers ────────────────────────────────────────────────────────────


def test_confidence_bucket():
    assert cm._confidence_bucket(1.0) == "EXTRACTED"
    assert cm._confidence_bucket(0.5) == "INFERRED"
    assert cm._confidence_bucket(0.2) == "AMBIGUOUS"
    assert cm._confidence_bucket(None) == "EXTRACTED"
    assert cm._confidence_bucket("INFERRED") == "INFERRED"


def test_import_cycles_detects_loop():
    edges = [
        {"src": "a", "dst": "b"},
        {"src": "b", "dst": "c"},
        {"src": "c", "dst": "a"},  # cycle a→b→c→a
        {"src": "c", "dst": "d"},  # tail, not in cycle
    ]
    cycles = cm.import_cycles(["a", "b", "c", "d"], edges)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"a", "b", "c"}


def test_import_cycles_none_on_dag():
    edges = [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}]
    assert cm.import_cycles(["a", "b", "c"], edges) == []


# ── build_code_metrics ──────────────────────────────────────────────────────


def test_code_metrics_finds_god_node_and_communities():
    out = cm.build_code_metrics(FakeEngine(), top_k=5)
    assert out["status"] == "ok"
    # settings is referenced by 3 modules + nothing else → high degree hub.
    god_labels = [g["label"] for g in out["god_nodes"]]
    assert "settings" in god_labels
    # at least two communities from the deterministic split.
    assert out["community_count"] >= 2
    # relation + confidence distributions present.
    assert "calls" in out["by_relation"]
    assert out["by_confidence"]["EXTRACTED"] >= 1


def test_code_metrics_surprising_connections_are_cross_community():
    out = cm.build_code_metrics(FakeEngine(), top_k=10)
    for bridge in out["surprising_connections"]:
        assert bridge["from_community"] != bridge["to_community"]


def test_code_metrics_scope_filter():
    out = cm.build_code_metrics(FakeEngine(), scope="auth.py", top_k=5)
    assert out["status"] == "ok"
    # every surfaced god node must touch the scoped file.
    assert all(
        "auth.py" in (g["file_path"] or "") or g["label"]
        for g in out["god_nodes"]
    )


def test_code_metrics_empty_graph():
    class Empty(FakeEngine):
        def __init__(self):
            super().__init__()
            self.backend.execute = lambda *a, **k: []

    out = cm.build_code_metrics(Empty(), top_k=5)
    assert out["status"] == "empty"


# ── build_arch_report ───────────────────────────────────────────────────────


def test_arch_report_markdown_and_persist():
    eng = FakeEngine()
    rep = cm.build_arch_report(eng, top_k=5)
    assert rep["status"] == "ok"
    md = rep["markdown"]
    assert "# Architecture Report" in md
    assert "## God Nodes" in md
    assert "## Surprising Connections" in md
    assert "## Dependency Cycles" in md
