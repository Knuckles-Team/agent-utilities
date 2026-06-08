"""Tests for IngestionEngine (CONCEPT:KG-2.7).

Validates that all ContentType adaptors route correctly through the
single ``IngestionEngine.ingest()`` entrypoint, and that batch ingestion,
history tracking, and error handling work as expected.
"""

import json
import uuid
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.ingestion.engine import (
    ContentType,
    IngestionEngine,
    IngestionManifest,
    IngestionResult,
)

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def mock_kg_engine():
    """Create a mock IntelligenceGraphEngine with standard attributes."""
    kg = MagicMock()
    kg.backend = MagicMock()
    kg.graph = MagicMock()
    kg.graph_compute = MagicMock()

    kg.ingest_episode = MagicMock(return_value=f"ep:{uuid.uuid4().hex[:8]}")
    kg.ingest_agent_skill = MagicMock()
    kg.ingest_mcp_server = MagicMock()
    kg.ingest_a2a_agent_card = MagicMock()
    kg.ingest_constitution = MagicMock(return_value={"policies_ingested": 5})
    kg.ingest_engineering_rules = MagicMock(return_value={"rules_ingested": 10})
    kg.ingest_all_policies = MagicMock(
        return_value={"policies_ingested": 5, "rules_ingested": 10}
    )

    return kg


@pytest.fixture
def engine(mock_kg_engine):
    """Create an IngestionEngine with mocked dependencies."""
    return IngestionEngine(kg_engine=mock_kg_engine)


# ── ContentType ───────────────────────────────────────────────────────


class TestContentType:
    """CONCEPT:KG-2.7 — ContentType enum completeness."""

    def test_all_content_types_registered(self):
        """Every ContentType value must have a name."""
        expected = {
            "codebase",
            "document",
            "conversation",
            "social",
            "kb",
            "sparql",
            "skill",
            "mcp_server",
            "policy",
            "event",
            "prompt",
            "config",
        }
        actual = {ct.value for ct in ContentType}
        assert actual == expected

    def test_content_type_is_str_enum(self):
        assert isinstance(ContentType.CODEBASE, str)
        assert ContentType.CODEBASE == "codebase"


# ── IngestionManifest ─────────────────────────────────────────────────


class TestIngestionManifest:
    """CONCEPT:KG-2.7 — IngestionManifest construction and defaults."""

    def test_minimal_manifest(self):
        m = IngestionManifest(
            content_type=ContentType.DOCUMENT,
            source_uri="/path/to/doc.md",
        )
        assert m.content_type == ContentType.DOCUMENT
        assert m.source_uri == "/path/to/doc.md"
        assert m.metadata == {}
        assert m.max_depth == 3
        assert m.force is False

    def test_manifest_with_metadata(self):
        m = IngestionManifest(
            content_type=ContentType.SOCIAL,
            source_uri='{"post_id": "123"}',
            metadata={"kg_context": ["AI", "ML"]},
            force=True,
        )
        assert m.force is True
        assert m.metadata["kg_context"] == ["AI", "ML"]


# ── Directory content hash (delta-skip identity) ─────────────────────


class TestDirectorySourceHash:
    """CONCEPT:KG-2.7 — directory content-identity hashing.

    Regression: ``_default_source_hash`` must prune ``_SKIP_DIRS`` *during*
    traversal (``os.walk`` in-place prune), never descending into vendored/build
    trees. The old ``rglob("*")``-then-filter walked every file under
    ``node_modules``/``.git``/``.venv`` first — minutes of CPU on real repos.
    """

    def test_skips_vendored_dirs_and_is_stable(self, tmp_path):
        from agent_utilities.knowledge_graph.ingestion.engine import (
            _default_source_hash,
        )

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')")
        (tmp_path / "README.md").write_text("docs")

        baseline = _default_source_hash(str(tmp_path))
        assert baseline is not None

        # Adding files *inside* a skip-dir must not change the digest, proving we
        # never descend into it (and never pay to walk it).
        nm = tmp_path / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        for i in range(50):
            (nm / f"f{i}.js").write_text("x" * 100)
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main")

        assert _default_source_hash(str(tmp_path)) == baseline

        # A real source change *does* change the digest.
        (tmp_path / "src" / "extra.py").write_text("x = 1")
        assert _default_source_hash(str(tmp_path)) != baseline


# ── IngestionResult ──────────────────────────────────────────────────


class TestIngestionResult:
    """CONCEPT:KG-2.7 — IngestionResult construction and defaults."""

    def test_success_result(self):
        m = IngestionManifest(
            content_type=ContentType.CODEBASE,
            source_uri="/path/to/project",
        )
        r = IngestionResult(manifest=m, status="success", nodes_created=42)
        assert r.status == "success"
        assert r.nodes_created == 42
        assert r.error is None

    def test_failed_result(self):
        m = IngestionManifest(
            content_type=ContentType.CODEBASE,
            source_uri="/nonexistent",
        )
        r = IngestionResult(manifest=m, status="failed", error="Path not found")
        assert r.status == "failed"
        assert r.error == "Path not found"


# ── Codebase Adaptor ─────────────────────────────────────────────────


class TestCodebaseIngestion:
    """CONCEPT:KG-2.7 — CODEBASE content type adaptor."""

    @pytest.mark.anyio
    async def test_nonexistent_path(self, engine):
        result = await engine.ingest(
            IngestionManifest(
                content_type=ContentType.CODEBASE,
                source_uri="/definitely/does/not/exist/12345",
            )
        )
        assert result.status == "failed"
        assert "does not exist" in result.error

    @pytest.mark.anyio
    async def test_routes_through_enrichment_pipeline(self, engine, tmp_path):
        """Structural codebase ingest runs the per-file Rust parse path
        (EnrichmentPipeline), not the old whole-repo parse_repository (CONCEPT:KG-2.8).
        """
        (tmp_path / "main.py").write_text("def hello():\n    return 1\n")
        # Fake the Rust parser with a benign empty parse so no service is needed.
        engine.kg.graph_compute.parse_file = MagicMock(return_value={})

        result = await engine.ingest(
            IngestionManifest(
                content_type=ContentType.CODEBASE,
                source_uri=str(tmp_path),
                metadata={"features": False},  # avoid spawning a community engine
            )
        )

        assert result.status == "success"
        # New path uses parse_file per discovered file …
        engine.kg.graph_compute.parse_file.assert_called()
        # … and the old whole-repo parser is gone (strangler).
        engine.kg.graph_compute.parse_repository.assert_not_called()


# ── Conversation Adaptor ─────────────────────────────────────────────


class TestConversationIngestion:
    """CONCEPT:KG-2.7 — CONVERSATION content type adaptor."""

    @pytest.mark.anyio
    async def test_creates_episode_node(self, engine):
        result = await engine.ingest(
            IngestionManifest(
                content_type=ContentType.CONVERSATION,
                source_uri="User asked about deployment",
                metadata={"source": "chat"},
            )
        )
        assert result.status == "success"
        assert result.nodes_created == 1
        engine.kg.ingest_episode.assert_called_once()


# ── Skill Adaptor ────────────────────────────────────────────────────


class TestSkillIngestion:
    """CONCEPT:KG-2.7 — SKILL content type adaptor."""

    @pytest.mark.anyio
    async def test_missing_skill_md(self, engine, tmp_path):
        result = await engine.ingest(
            IngestionManifest(
                content_type=ContentType.SKILL,
                source_uri=str(tmp_path),
            )
        )
        assert result.status == "failed"
        assert "SKILL.md" in result.error

    @pytest.mark.anyio
    async def test_parses_frontmatter(self, engine, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            "---\nname: test-skill\ndescription: A test\n---\n# Instructions\n"
        )

        result = await engine.ingest(
            IngestionManifest(
                content_type=ContentType.SKILL,
                source_uri=str(tmp_path),
            )
        )
        assert result.status == "success"
        assert result.nodes_created == 1
        assert result.details["skill_name"] == "test-skill"
        engine.kg.ingest_agent_skill.assert_called_once()


# ── Policy Adaptor ───────────────────────────────────────────────────


class TestPolicyIngestion:
    """CONCEPT:KG-2.7 — POLICY content type adaptor."""

    @pytest.mark.anyio
    async def test_ingest_all_policies(self, engine, tmp_path):
        result = await engine.ingest(
            IngestionManifest(
                content_type=ContentType.POLICY,
                source_uri=str(tmp_path),
                metadata={"policy_type": "all"},
            )
        )
        assert result.status == "success"
        engine.kg.ingest_all_policies.assert_called_once()

    @pytest.mark.anyio
    async def test_ingest_constitution(self, engine, tmp_path):
        result = await engine.ingest(
            IngestionManifest(
                content_type=ContentType.POLICY,
                source_uri=str(tmp_path),
                metadata={"policy_type": "constitution"},
            )
        )
        assert result.status == "success"
        engine.kg.ingest_constitution.assert_called_once()


# ── MCP Server Adaptor ───────────────────────────────────────────────


class TestMCPServerIngestion:
    """CONCEPT:KG-2.7 — MCP_SERVER content type adaptor."""

    @pytest.mark.anyio
    async def test_local_mcp_config(self, engine, tmp_path):
        config = {
            "mcpServers": {"my-srv": {"command": "uvx", "args": ["x"], "env": {}}}
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config))

        # Fake the live-discovery hooks: parse → one real entry; discovery →
        # no tools (avoid spawning a real server in unit tests).
        engine.kg.parse_mcp_config = lambda data: [
            {"name": "my-srv", "command": "uvx", "args": ["x"], "env": {}}
        ]

        async def _no_tools(entry, timeout=15.0):
            return []

        engine.kg.discover_mcp_tools = _no_tools

        result = await engine.ingest(
            IngestionManifest(
                content_type=ContentType.MCP_SERVER,
                source_uri=str(config_file),
            )
        )
        assert result.status == "success"
        assert result.details["servers_ingested"] == 1
        engine.kg.ingest_mcp_server.assert_called_once()

    @pytest.mark.anyio
    async def test_missing_config(self, engine):
        result = await engine.ingest(
            IngestionManifest(
                content_type=ContentType.MCP_SERVER,
                source_uri="/nonexistent/config.json",
            )
        )
        assert result.status == "failed"


# ── Prompt Adaptor ───────────────────────────────────────────────────


class TestPromptIngestion:
    """CONCEPT:KG-2.7 — PROMPT content type adaptor."""

    @pytest.mark.anyio
    async def test_prompt_success(self, engine, tmp_path):
        prompt = tmp_path / "system_prompt.md"
        prompt.write_text("You are a helpful assistant.\n")

        result = await engine.ingest(
            IngestionManifest(
                content_type=ContentType.PROMPT,
                source_uri=str(prompt),
                # Structural only — don't make a real LLM concept-extraction call
                # in a unit test (concept extraction is covered elsewhere).
                metadata={"extract_concepts": False},
            )
        )
        assert result.status == "success"
        assert result.nodes_created == 1
        assert "prompt_id" in result.details

    @pytest.mark.anyio
    async def test_prompt_missing_file(self, engine):
        result = await engine.ingest(
            IngestionManifest(
                content_type=ContentType.PROMPT,
                source_uri="/nonexistent/prompt.md",
            )
        )
        assert result.status == "failed"


# ── Batch Ingestion ──────────────────────────────────────────────────


class TestBatchIngestion:
    """CONCEPT:KG-2.7 — Batch ingestion via ingest_batch()."""

    @pytest.mark.anyio
    async def test_mixed_results(self, engine, tmp_path):
        prompt = tmp_path / "ok.md"
        prompt.write_text("content\n")

        manifests = [
            IngestionManifest(
                content_type=ContentType.PROMPT,
                source_uri=str(prompt),
            ),
            IngestionManifest(
                content_type=ContentType.PROMPT,
                source_uri="/nonexistent.md",
            ),
        ]
        results = await engine.ingest_batch(manifests)
        assert len(results) == 2
        assert results[0].status == "success"
        assert results[1].status == "failed"


# ── History Tracking ─────────────────────────────────────────────────


class TestHistoryTracking:
    """CONCEPT:KG-2.7 — Ingestion history tracking."""

    @pytest.mark.anyio
    async def test_history_populated(self, engine, tmp_path):
        prompt = tmp_path / "test.md"
        prompt.write_text("test\n")

        assert len(engine.history) == 0
        await engine.ingest(
            IngestionManifest(
                content_type=ContentType.PROMPT,
                source_uri=str(prompt),
            )
        )
        assert len(engine.history) == 1
        assert engine.history[0].status == "success"

    @pytest.mark.anyio
    async def test_history_tracks_failures(self, engine):
        await engine.ingest(
            IngestionManifest(
                content_type=ContentType.PROMPT,
                source_uri="/nonexistent.md",
            )
        )
        assert len(engine.history) == 1
        assert engine.history[0].status == "failed"


# ── Module Exports ───────────────────────────────────────────────────


class TestModuleExports:
    """CONCEPT:KG-2.7 — Verify public API surface from ingestion module."""

    def test_engine_importable(self):
        from agent_utilities.knowledge_graph.ingestion import IngestionEngine

        assert IngestionEngine is not None

    def test_content_type_importable(self):
        from agent_utilities.knowledge_graph.ingestion import ContentType

        assert ContentType is not None

    def test_manifest_importable(self):
        from agent_utilities.knowledge_graph.ingestion import IngestionManifest

        assert IngestionManifest is not None

    def test_result_importable(self):
        from agent_utilities.knowledge_graph.ingestion import IngestionResult

        assert IngestionResult is not None
