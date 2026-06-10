"""Tests for the durable delta-ingestion manifest + centralized skip (CONCEPT:KG-2.8).

Service-free: the manifest runs in its SQLite fallback mode and the engine is
driven with a lightweight stub KG, so these validate delta-skip, durability
across "restart", upsert-no-duplicate, and the per-category coverage gauge
without needing the epistemic-graph daemon.
"""

import pytest

from agent_utilities.knowledge_graph.enrichment.query import enrichment_coverage
from agent_utilities.knowledge_graph.ingestion.engine import (
    ContentType,
    IngestionEngine,
    IngestionManifest,
)
from agent_utilities.knowledge_graph.ingestion.manifest import DeltaManifest

# ── DeltaManifest (SQLite mode) ────────────────────────────────────────


class TestDeltaManifest:
    def _mk(self, tmp_path):
        return DeltaManifest(backend=None, db_path=str(tmp_path / "m.db"))

    def test_record_seen_get_roundtrip(self, tmp_path):
        m = self._mk(tmp_path)
        g, c, s = "__bus__", "codebase", "/x/y.py"
        assert m.mode == "sqlite"
        assert m.get(g, c, s) is None
        m.record(g, c, s, "h1")
        assert m.seen(g, c, s, "h1") is True
        assert m.seen(g, c, s, "h2") is False
        assert m.get(g, c, s) == "h1"

    def test_upsert_no_duplicate(self, tmp_path):
        m = self._mk(tmp_path)
        g, c, s = "__bus__", "codebase", "/x/y.py"
        m.record(g, c, s, "h1")
        m.record(g, c, s, "h2")  # upsert, not a second row
        assert m.get(g, c, s) == "h2"
        assert m.load_for_graph(g, c) == {s: "h2"}

    def test_durable_across_restart(self, tmp_path):
        db = str(tmp_path / "m.db")
        DeltaManifest(backend=None, db_path=db).record(
            "__bus__", "codebase", "/a.py", "h9"
        )
        # Fresh instance over the same store (simulates a process restart).
        m2 = DeltaManifest(backend=None, db_path=db)
        assert m2.seen("__bus__", "codebase", "/a.py", "h9") is True

    def test_clear_scoped(self, tmp_path):
        m = self._mk(tmp_path)
        m.record("__bus__", "codebase", "/a.py", "h1")
        m.record("__bus__", "document", "/b.md", "h2")
        m.clear("__bus__", "codebase")
        assert m.get("__bus__", "codebase", "/a.py") is None
        assert m.get("__bus__", "document", "/b.md") == "h2"

    def test_load_for_graph_is_scoped(self, tmp_path):
        m = self._mk(tmp_path)
        m.record("__bus__", "codebase_file", "/a.py", "h1")
        m.record("__bus__", "codebase_file", "/b.py", "h2")
        m.record("other", "codebase_file", "/c.py", "h3")
        assert m.load_for_graph("__bus__", "codebase_file") == {
            "/a.py": "h1",
            "/b.py": "h2",
        }


# ── Centralized engine-level delta skip ────────────────────────────────


class _StubKG:
    """Minimal KG: PROMPT adaptor returns success without a real graph."""

    backend = None
    graph = None  # prompt adaptor: no graph → no write, still status=success


@pytest.fixture
def prompt_engine(tmp_path):
    eng = IngestionEngine(kg_engine=_StubKG())
    eng.manifest = DeltaManifest(backend=None, db_path=str(tmp_path / "m.db"))
    return eng


class TestEngineDeltaSkip:
    @pytest.mark.anyio
    async def test_unchanged_source_skipped(self, prompt_engine, tmp_path):
        f = tmp_path / "p.md"
        f.write_text("hello prompt")
        man = IngestionManifest(content_type=ContentType.PROMPT, source_uri=str(f))

        r1 = await prompt_engine.ingest(man)
        r2 = await prompt_engine.ingest(man)
        assert r1.status == "success"
        assert r2.status == "skipped"

    @pytest.mark.anyio
    async def test_changed_source_reingested(self, prompt_engine, tmp_path):
        f = tmp_path / "p.md"
        f.write_text("v1")
        man = IngestionManifest(content_type=ContentType.PROMPT, source_uri=str(f))
        assert (await prompt_engine.ingest(man)).status == "success"
        assert (await prompt_engine.ingest(man)).status == "skipped"
        f.write_text("v2 changed")
        assert (await prompt_engine.ingest(man)).status == "success"

    @pytest.mark.anyio
    async def test_force_bypasses_skip(self, prompt_engine, tmp_path):
        f = tmp_path / "p.md"
        f.write_text("hello")
        man = IngestionManifest(content_type=ContentType.PROMPT, source_uri=str(f))
        forced = IngestionManifest(
            content_type=ContentType.PROMPT, source_uri=str(f), force=True
        )
        assert (await prompt_engine.ingest(man)).status == "success"
        assert (await prompt_engine.ingest(forced)).status == "success"

    @pytest.mark.anyio
    async def test_failure_not_recorded(self, prompt_engine, tmp_path):
        # A missing prompt file fails; a failure must NOT be recorded, so a
        # retry is attempted again (not skipped).
        man = IngestionManifest(
            content_type=ContentType.PROMPT, source_uri=str(tmp_path / "missing.md")
        )
        assert (await prompt_engine.ingest(man)).status == "failed"
        assert (await prompt_engine.ingest(man)).status == "failed"


# ── ContentType.classify ───────────────────────────────────────────────


class TestClassify:
    def test_doc_extension(self):
        assert ContentType.classify("/a/notes.md") == ContentType.DOCUMENT
        assert ContentType.classify("/a/paper.pdf") == ContentType.DOCUMENT

    def test_url_is_document(self):
        assert ContentType.classify("https://example.com/x") == ContentType.DOCUMENT

    def test_mcp_config(self):
        assert ContentType.classify("/a/mcp_config.json") == ContentType.MCP_SERVER

    def test_default_codebase(self, tmp_path):
        assert ContentType.classify(str(tmp_path)) == ContentType.CODEBASE
        assert ContentType.classify("/a/module.py") == ContentType.CODEBASE

    def test_directory_of_documents(self, tmp_path):
        # A folder of papers/notes is a DOCUMENT corpus, not a codebase.
        (tmp_path / "paper1.pdf").write_bytes(b"%PDF-1.7 fake")
        (tmp_path / "paper2.pdf").write_bytes(b"%PDF-1.7 fake")
        (tmp_path / "notes.md").write_text("# notes")
        assert ContentType.classify(str(tmp_path)) == ContentType.DOCUMENT

    def test_directory_codebase_marker_wins(self, tmp_path):
        # Packaging marker is definitive even when docs are also present.
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        (tmp_path / "README.md").write_text("# x")
        assert ContentType.classify(str(tmp_path)) == ContentType.CODEBASE

    def test_directory_of_source(self, tmp_path):
        (tmp_path / "mod.py").write_text("x = 1\n")
        (tmp_path / "util.py").write_text("y = 2\n")
        assert ContentType.classify(str(tmp_path)) == ContentType.CODEBASE

    def test_directory_documents_ignore_vendored(self, tmp_path):
        # A document corpus carrying a bundled virtualenv must still be a
        # DOCUMENT corpus — vendored dirs are pruned before counting.
        (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.7 fake")
        venv = tmp_path / ".venv" / "lib" / "site-packages"
        venv.mkdir(parents=True)
        for i in range(25):
            (venv / f"m{i}.py").write_text("x = 1\n")
        assert ContentType.classify(str(tmp_path)) == ContentType.DOCUMENT


# ── enrichment_coverage gauge ──────────────────────────────────────────


class _FakeBackend:
    """execute() returns canned ``{'n': node}`` rows keyed by the queried label."""

    def __init__(self, nodes):
        self.nodes = nodes

    def execute(self, query, params=None):
        for label in ("Code", "Test", "Document", "Concept", "Feature"):
            if f"(n:{label})" in query:
                return [{"n": n} for n in self.nodes.get(label, [])]
        return []


class _RecordingKG:
    """KG stub that records add_node calls (for the CONFIG adaptor)."""

    backend = None

    def __init__(self):
        self.nodes = []

    def add_node(self, node_id, node_type, properties):
        self.nodes.append((node_id, node_type, properties))


class TestConfigAdaptor:
    @pytest.mark.anyio
    async def test_ingest_config_models(self, tmp_path):
        import json

        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps(
                {
                    "chat_models": [
                        {"id": "qwen/q9", "base_url": "http://x/v1", "can_kg": True},
                        {"id": "qwen-lite", "api_key": "secret", "can_route": True},
                    ],
                    "embedding_models": [{"id": "bge-m3", "base_url": "http://e/v1"}],
                    "routing_strategy": "hybrid",
                    "max_concurrent_agents": 12,
                }
            )
        )
        kg = _RecordingKG()
        eng = IngestionEngine(kg_engine=kg)
        eng.manifest = DeltaManifest(backend=None, db_path=str(tmp_path / "m.db"))
        assert ContentType.classify(str(cfg)) == ContentType.CONFIG

        r = await eng.ingest(
            IngestionManifest(content_type=ContentType.CONFIG, source_uri=str(cfg))
        )
        assert r.status == "success"
        types = [t for _, t, _ in kg.nodes]
        assert types.count("LanguageModel") == 2
        assert types.count("EmbeddingModel") == 1
        assert "SystemConfig" in types
        # Secrets dropped from persisted props.
        for _, t, props in kg.nodes:
            assert "api_key" not in props and "base_url" not in props
        # Re-ingest unchanged → skipped (delta manifest).
        r2 = await eng.ingest(
            IngestionManifest(content_type=ContentType.CONFIG, source_uri=str(cfg))
        )
        assert r2.status == "skipped"


class TestEnrichmentCoverage:
    def test_codebase_coverage_math(self):
        be = _FakeBackend(
            {
                "Code": [
                    {"ast_hash": "a", "summary": "documented"},
                    {"ast_hash": "b", "summary": ""},
                    {"ast_hash": "c", "summary": ""},
                    {"summary": "no-hash phantom"},  # excluded (no ast_hash)
                ],
                "Test": [
                    {"needs_work": True},
                    {"needs_work": False},
                ],
            }
        )
        cov = enrichment_coverage(be)["codebase"]
        assert cov["code_total"] == 3  # phantom excluded
        assert cov["code_with_cards"] == 1
        assert cov["cards_pending"] == 2
        assert cov["tests_total"] == 2
        assert cov["tests_needing_work"] == 1
        assert cov["coverage"] == round(1 / 3, 4)

    def test_empty_graph_zero_coverage(self):
        cov = enrichment_coverage(_FakeBackend({}))
        assert cov["codebase"]["coverage"] == 0.0
        assert cov["documents"]["coverage"] == 0.0
        assert cov["features"]["feature_total"] == 0
