"""Tests for deterministic repo content classification + codebase-ingest routing.

CONCEPT:KG-2.284 / KG-2.285 — A single codebase ingest must natively recognise a
repo's Skills / Prompts / Specs / Documents and route each to its own KG type
instead of flattening everything into Code. These tests pin the classifier's
precedence (the edge cases the design calls out) and verify the engine router
fans each artifact out to the correct native adaptor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.ingestion.engine import (
    ContentType,
    IngestionEngine,
    IngestionManifest,
    IngestionResult,
)
from agent_utilities.knowledge_graph.ingestion.repo_classifier import classify_repo

# ── Fixture: a small mixed repo (code + skill + prompt + spec + docs) ──────


@pytest.fixture
def mixed_repo(tmp_path: Path) -> Path:
    root = tmp_path / "demo_repo"
    root.mkdir()

    # Packaging marker → definitively a codebase.
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n")

    # Code.
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("def main():\n    return 1\n")

    # Markdown — at root, in docs/, and (edge case) INSIDE a code dir.
    (root / "README.md").write_text("# Demo\nA demo repo.\n")
    (root / "docs").mkdir()
    (root / "docs" / "guide.md").write_text("# Guide\nUsage.\n")
    (root / "src" / "notes.md").write_text("# Notes\nIn a code dir.\n")

    # A skill: SKILL.md + a nested reference doc that must be CLAIMED by the
    # skill (NOT separately ingested as a Document).
    skill = root / "skills" / "myskill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: myskill\n---\n# Do things\n")
    (skill / "reference").mkdir()
    (skill / "reference" / "ref.md").write_text("# Reference\nDetails.\n")

    # A prompt JSON under prompts/ (must be PROMPT, not config).
    (root / "prompts").mkdir()
    (root / "prompts" / "greeting.json").write_text('{"system": "You are helpful."}')

    # Specs: one under .specify/, one *.spec.md at root.
    spec_dir = root / ".specify" / "specs" / "feature"
    spec_dir.mkdir(parents=True)
    (spec_dir / "spec.md").write_text("# Feature spec\nRequirements.\n")
    (root / "auth.spec.md").write_text("# Auth spec\nDesign.\n")

    # Config files.
    (root / "config.json").write_text('{"chat_models": []}')
    (root / "mcp_config.json").write_text('{"mcpServers": {}}')

    # A plain data JSON with no prompt markers → must stay UNCLASSIFIED.
    (root / "data.json").write_text('{"rows": [1, 2, 3]}')

    return root


# ── Classifier precedence (pure, deterministic) ────────────────────────────


class TestRepoClassifier:
    """CONCEPT:KG-2.284 — extension + path + sniff, explicit precedence."""

    def test_buckets(self, mixed_repo: Path):
        plan = classify_repo(mixed_repo)
        skill_dirs = {fc.path for fc in plan.skills}
        doc_paths = {fc.path for fc in plan.documents}
        prompt_paths = {fc.path for fc in plan.prompts}
        spec_paths = {fc.path for fc in plan.specs}
        config_paths = {fc.path for fc in plan.configs}
        code_paths = {fc.path for fc in plan.code}

        # Skill dir recognised; its nested reference/*.md is CLAIMED (not a Document).
        assert mixed_repo / "skills" / "myskill" in skill_dirs
        assert (
            mixed_repo / "skills" / "myskill" / "reference" / "ref.md"
        ) not in doc_paths
        assert (mixed_repo / "skills" / "myskill" / "SKILL.md") not in doc_paths

        # Markdown → Document, including the .md INSIDE a code dir (edge case).
        assert mixed_repo / "README.md" in doc_paths
        assert mixed_repo / "docs" / "guide.md" in doc_paths
        assert mixed_repo / "src" / "notes.md" in doc_paths

        # Prompt JSON → Prompt (NOT config), even though it ends in .json.
        assert mixed_repo / "prompts" / "greeting.json" in prompt_paths

        # Specs → Spec (NOT Document), both under .specify/ and *.spec.md.
        assert mixed_repo / ".specify" / "specs" / "feature" / "spec.md" in spec_paths
        assert mixed_repo / "auth.spec.md" in spec_paths
        # A spec is never also a Document.
        assert spec_paths.isdisjoint(doc_paths)

        # Config / mcp recognised.
        assert mixed_repo / "config.json" in config_paths
        assert mixed_repo / "mcp_config.json" in config_paths

        # Code reported for coverage.
        assert mixed_repo / "src" / "app.py" in code_paths

        # A plain data JSON is left unclassified (no LLM guess).
        all_paths = doc_paths | prompt_paths | spec_paths | config_paths | code_paths
        assert mixed_repo / "data.json" not in all_paths

    def test_config_json_is_not_a_prompt(self, mixed_repo: Path):
        plan = classify_repo(mixed_repo)
        prompt_paths = {fc.path for fc in plan.prompts}
        assert mixed_repo / "config.json" not in prompt_paths
        assert mixed_repo / "mcp_config.json" not in prompt_paths

    def test_summary_counts(self, mixed_repo: Path):
        s = classify_repo(mixed_repo).summary()
        assert s["skills"] == 1
        assert s["prompts"] == 1
        assert s["specs"] == 2
        assert s["documents"] == 3  # README, docs/guide, src/notes
        assert s["code"] == 1

    def test_repo_root_skill_does_not_claim_everything(self, tmp_path: Path):
        """A repo-root SKILL.md must NOT suppress the repo's docs/prompts."""
        root = tmp_path / "skillish_repo"
        (root / "docs").mkdir(parents=True)
        (root / "SKILL.md").write_text("---\nname: top\n---\nbody\n")
        (root / "docs" / "d.md").write_text("# doc\n")
        plan = classify_repo(root)
        # Root not treated as a claiming skill dir → docs still route.
        assert (root / "docs" / "d.md") in {fc.path for fc in plan.documents}


# ── Router: each artifact fans out to the right native adaptor ─────────────


class _FakeBackend:
    def __init__(self):
        self.nodes: list[tuple[str, dict]] = []
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id, **props):
        self.nodes.append((node_id, props))

    def add_edge(self, src, dst, rel_type=None, **_):
        self.edges.append((src, dst, rel_type))


class _FakeManifest:
    """Delta ledger stub: nothing seen, records are no-ops (CONCEPT:KG-2.295)."""

    def __init__(self):
        self.recorded: list[tuple] = []

    def seen(self, *_a, **_k) -> bool:
        return False

    def record(self, *a, **_k) -> None:
        self.recorded.append(a)


class TestCodebaseArtifactRouting:
    """CONCEPT:KG-2.285 — live-path: routed manifests + Spec/Repo writes."""

    @pytest.mark.asyncio
    async def test_routes_each_type(self, mixed_repo: Path, monkeypatch):
        engine = IngestionEngine.__new__(IngestionEngine)
        backend = _FakeBackend()
        engine.kg = None
        engine.backend = backend
        engine.graph_name = "__commons__"
        engine._routed_backends = {}
        engine.manifest = _FakeManifest()

        # Spy on the central ingest seam (skills/prompts still route through it):
        # record each routed sub-manifest and return a success result carrying a
        # stable source_id (for repo linking).
        seen: list[IngestionManifest] = []

        async def _fake_ingest(manifest, **_):
            seen.append(manifest)
            return IngestionResult(
                manifest=manifest,
                status="success",
                nodes_created=2,
                edges_created=1,
                enrichable=[{"source_id": f"{manifest.content_type.value}:x"}],
            )

        monkeypatch.setattr(engine, "ingest", _fake_ingest)

        # CONCEPT:KG-2.295 — documents take the batched, enrich-deferred path
        # (direct unit + per-doc _BatchedBackend), NOT self.ingest. Spy on the
        # canonical unit to record each routed document path.
        doc_seen: list[Path] = []

        def _fake_doc_file(manifest, path_obj, backend=None):
            doc_seen.append(Path(path_obj))
            return IngestionResult(
                manifest=manifest,
                status="success",
                nodes_created=2,
                edges_created=1,
                enrichable=[
                    {
                        "source_id": f"document:{path_obj.name}",
                        "text": "body",
                        "source_type": "document",
                        "concepts_done": True,
                    }
                ],
            )

        monkeypatch.setattr(engine, "_ingest_document_file", _fake_doc_file)

        result = IngestionResult(
            manifest=IngestionManifest(
                content_type=ContentType.CODEBASE, source_uri=str(mixed_repo)
            ),
            status="success",
        )
        await engine._route_classified_artifacts(
            result.manifest, str(mixed_repo), result
        )

        by_type: dict[str, list[Path]] = {"skill": [], "prompt": []}
        for m in seen:
            by_type.setdefault(m.content_type.value, []).append(Path(m.source_uri))
        documents = set(doc_seen)

        # Skill dir routed to SKILL adaptor.
        assert (mixed_repo / "skills" / "myskill") in by_type["skill"]
        # Prompt JSON routed to PROMPT adaptor.
        assert (mixed_repo / "prompts" / "greeting.json") in by_type["prompt"]
        # Skills/prompts do NOT take the document path.
        assert not doc_seen or all(
            p.suffix in {".md", ".markdown", ".txt", ".rst"} for p in doc_seen
        )
        # Each markdown routed to the DOCUMENT unit (incl. the one in a code dir).
        assert (mixed_repo / "README.md") in documents
        assert (mixed_repo / "src" / "notes.md") in documents
        # The skill's nested ref.md was claimed → never routed as a Document.
        assert (
            mixed_repo / "skills" / "myskill" / "reference" / "ref.md"
        ) not in documents
        # Documents' enrichable bubbled up to the PARENT result for one central pass.
        assert any(e.get("source_type") == "document" for e in result.enrichable)

        # Specs written inline as Spec nodes (no SPEC adaptor).
        spec_nodes = [n for n in backend.nodes if n[1].get("type") == "Spec"]
        assert len(spec_nodes) == 2
        # A Repo node was created and links the artifacts via CONTAINS.
        assert any(n[1].get("type") == "Repo" for n in backend.nodes)
        assert any(rel == "CONTAINS" for _s, _d, rel in backend.edges)

        # Counts surfaced for observability.
        classified = result.details["classified"]
        assert classified["skill"] == 1
        assert classified["prompt"] == 1
        assert classified["document"] == 3
        assert classified["spec"] == 2

    @pytest.mark.asyncio
    async def test_documents_batch_writes_instead_of_per_node_round_trips(
        self, mixed_repo: Path, monkeypatch
    ):
        """CONCEPT:KG-2.295 — a repo's docs flush as a few BULK RPCs, not one
        engine round-trip per Document/chunk/concept node.

        Simulates the unit writing 13 nodes + 12 edges per doc (1 Document + 12
        chunks + edges). Per-file that would be 3×25 = 75 socket round-trips; the
        batched path collapses each doc to ~2 bulk RPCs (nodes then edges).
        """

        class _BulkGraph:
            def __init__(self):
                self.bulk_calls = 0
                self.ops = 0

            def batch_update(self, ops):
                self.bulk_calls += 1
                self.ops += len(ops)

        class _BulkBackend:
            def __init__(self):
                self._graph = _BulkGraph()
                self.per_item_doc_writes = 0
                self.nodes: list[tuple[str, dict]] = []
                self.edges: list[tuple[str, str, str]] = []

            def add_node(self, node_id, **props):
                if str(props.get("type")) in {"Document", "idea_block"}:
                    self.per_item_doc_writes += 1
                self.nodes.append((node_id, props))

            def add_edge(self, src, dst, rel_type=None, **_):
                self.edges.append((src, dst, rel_type))

        engine = IngestionEngine.__new__(IngestionEngine)
        backend = _BulkBackend()
        engine.kg = None
        engine.backend = backend
        engine.graph_name = "__commons__"
        engine._routed_backends = {}
        engine.manifest = _FakeManifest()

        async def _fake_ingest(manifest, **_):  # skills/prompts
            return IngestionResult(
                manifest=manifest,
                status="success",
                enrichable=[{"source_id": f"{manifest.content_type.value}:x"}],
            )

        monkeypatch.setattr(engine, "ingest", _fake_ingest)

        # The unit writes its nodes/edges through the supplied (batched) backend.
        def _fake_doc_file(manifest, path_obj, backend=None):
            b = backend if backend is not None else engine.backend
            did = f"doc:{path_obj.name}"
            b.add_node(did, type="Document")
            for i in range(12):
                cid = f"{did}:chunk:{i}"
                b.add_node(cid, type="idea_block")
                b.add_edge(cid, did, rel_type="PART_OF")
            return IngestionResult(
                manifest=manifest,
                status="success",
                nodes_created=13,
                edges_created=12,
                enrichable=[
                    {
                        "source_id": did,
                        "text": "b",
                        "source_type": "document",
                        "concepts_done": True,
                    }
                ],
            )

        monkeypatch.setattr(engine, "_ingest_document_file", _fake_doc_file)

        result = IngestionResult(
            manifest=IngestionManifest(
                content_type=ContentType.CODEBASE, source_uri=str(mixed_repo)
            ),
            status="success",
        )
        await engine._route_classified_artifacts(
            result.manifest, str(mixed_repo), result
        )

        # All 3 docs' writes went through the engine BULK path…
        assert backend._graph.ops == 3 * (13 + 12)  # 75 ops total
        # …as a handful of RPCs (≤2 per doc), NOT 75 per-node round-trips.
        assert backend._graph.bulk_calls <= 2 * 3
        assert backend._graph.bulk_calls < backend._graph.ops // 4
        # No Document/chunk node hit the engine as a per-item write.
        assert backend.per_item_doc_writes == 0
