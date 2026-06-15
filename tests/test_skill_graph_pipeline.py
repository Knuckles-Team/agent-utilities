#!/usr/bin/python
from __future__ import annotations

"""Tests for the unified skill-graph pipeline (CONCEPT:KG-2.7).

Offline by construction: web/generated acquisition are exercised via injected
``crawler_fn``/``generator_fn`` and the KG-enrichment graceful-degrade path is
forced without a live daemon — so the whole suite runs with no network or engine.
"""

import json
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.distillation import (
    SkillGraphPipeline,
    SourceSpec,
    validate_skill_graph,
)
from agent_utilities.knowledge_graph.distillation.skill_graph_pipeline import (
    AcquiredDoc,
)
from agent_utilities.knowledge_graph.distillation.skill_graph_schema import (
    SOURCES_SCHEMA,
    parse_frontmatter,
    sha256_text,
)


@pytest.fixture
def src_dir(tmp_path: Path) -> Path:
    d = tmp_path / "src"
    d.mkdir()
    (d / "widget.md").write_text("# Widget API\n\nCreate widgets.\n", encoding="utf-8")
    (d / "auth.md").write_text("# Auth\n\nBearer tokens.\n", encoding="utf-8")
    return d


def _pipe() -> SkillGraphPipeline:
    return SkillGraphPipeline(kg_enrich=False)


# ── schema / validator ──────────────────────────────────────────────────────


def test_sha256_text_is_newline_normalized():
    assert sha256_text("a\r\nb") == sha256_text("a\nb")


def test_parse_frontmatter_scalars_lists_bools():
    fm = parse_frontmatter(
        "---\nname: x\nkg_ingested: false\nsource_types: [web, pdf]\n---\nbody\n"
    )
    assert fm["name"] == "x"
    assert fm["kg_ingested"] is False
    assert fm["source_types"] == ["web", "pdf"]


def test_validate_clean_graph(src_dir, tmp_path):
    res = _pipe().build(name="widget-docs", specs=[SourceSpec("dir", str(src_dir))],
                        out_dir=tmp_path / "out")
    assert res["validation_errors"] == []
    assert validate_skill_graph(tmp_path / "out" / "widget-docs") == []


def test_validate_missing_skill_md(tmp_path):
    (tmp_path / "empty").mkdir()
    errors = validate_skill_graph(tmp_path / "empty")
    assert errors and "missing SKILL.md" in errors[0]


def test_validate_name_mismatch(src_dir, tmp_path):
    _pipe().build(name="widget-docs", specs=[SourceSpec("dir", str(src_dir))],
                  out_dir=tmp_path / "out")
    graph = tmp_path / "out" / "widget-docs"
    renamed = graph.parent / "other-docs"
    graph.rename(renamed)
    errors = validate_skill_graph(renamed)
    assert any("!= directory name" in e for e in errors)


def test_validate_flags_missing_manifest(src_dir, tmp_path):
    _pipe().build(name="widget-docs", specs=[SourceSpec("dir", str(src_dir))],
                  out_dir=tmp_path / "out")
    graph = tmp_path / "out" / "widget-docs"
    (graph / "sources.json").unlink()
    errors = validate_skill_graph(graph)
    assert any("missing sources.json" in e for e in errors)


# ── source kinds ──────────────────────────────────────────────────────────────


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        SourceSpec("bogus", "x")


def test_build_dir_source(src_dir, tmp_path):
    res = _pipe().build(name="widget-docs", specs=[SourceSpec("dir", str(src_dir))],
                        out_dir=tmp_path / "out")
    assert res["file_count"] == 2
    assert res["kg_ingested"] is False
    manifest = json.loads(
        (tmp_path / "out" / "widget-docs" / "sources.json").read_text()
    )
    assert manifest["schema"] == SOURCES_SCHEMA
    assert manifest["sources"][0]["kind"] == "dir"
    assert manifest["sources"][0]["content_hash"].startswith("sha256:")
    assert len(manifest["files"]) == 2


def test_build_web_via_injected_crawler(tmp_path):
    def crawler_fn(spec):
        assert spec.kind == "web"
        return [AcquiredDoc(rel_path="index.md", text="# Home\n\nhi\n",
                            source_uri=spec.uri)]

    pipe = SkillGraphPipeline(crawler_fn=crawler_fn, kg_enrich=False)
    res = pipe.build(name="site-docs", specs=[SourceSpec("web", "https://x")],
                     out_dir=tmp_path / "out")
    assert res["file_count"] == 1
    fm = parse_frontmatter((tmp_path / "out" / "site-docs" / "SKILL.md").read_text())
    assert fm["source_types"] == ["web"]


def test_build_generated_via_injected_generator(tmp_path):
    def generator_fn(spec):
        return [AcquiredDoc(rel_path="topic.md", text=f"# {spec.uri}\n\ngenerated\n",
                            source_uri="generated://topic")]

    pipe = SkillGraphPipeline(generator_fn=generator_fn, kg_enrich=False)
    res = pipe.build(name="topic-docs",
                     specs=[SourceSpec("generated", "Kubernetes networking")],
                     out_dir=tmp_path / "out")
    assert res["file_count"] == 1


def test_multi_source_merge_dedupes_paths(src_dir, tmp_path):
    # Two dir sources that both yield widget.md/auth.md → names disambiguated.
    res = _pipe().build(
        name="merged-docs",
        specs=[SourceSpec("dir", str(src_dir)), SourceSpec("dir", str(src_dir))],
        out_dir=tmp_path / "out",
    )
    assert res["file_count"] == 4
    assert res["source_count"] == 2


# ── freshness + rebuild ───────────────────────────────────────────────────────


def test_status_fresh_then_stale_then_rebuild(src_dir, tmp_path):
    pipe = _pipe()
    pipe.build(name="widget-docs", specs=[SourceSpec("dir", str(src_dir))],
               out_dir=tmp_path / "out")
    graph = tmp_path / "out" / "widget-docs"

    assert pipe.status(graph, quick=True)["status"] == "fresh"

    (src_dir / "auth.md").write_text("# Auth\n\nNow OAuth.\n", encoding="utf-8")
    assert pipe.status(graph, quick=True)["status"] == "stale"

    res = pipe.rebuild(graph)
    assert res["version"] == "0.1.1"
    assert pipe.status(graph, quick=True)["status"] == "fresh"


def test_status_unknown_without_manifest(tmp_path):
    (tmp_path / "x").mkdir()
    assert _pipe().status(tmp_path / "x")["status"] == "unknown"


# ── hybrid-auto KG: graceful degrade ───────────────────────────────────────────


def test_kg_enrichment_degrades_when_daemon_unreachable(src_dir, tmp_path, monkeypatch):
    """kg_enrich=True but the ingest subprocess fails → kg_ingested False, graph still built."""
    import agent_utilities.knowledge_graph.distillation.skill_graph_pipeline as mod

    def boom(*a, **k):
        raise FileNotFoundError("no interpreter")

    monkeypatch.setattr(mod.subprocess, "run", boom)
    pipe = SkillGraphPipeline(kg_enrich=True)
    res = pipe.build(name="widget-docs", specs=[SourceSpec("dir", str(src_dir))],
                     out_dir=tmp_path / "out")
    assert res["kg_ingested"] is False
    assert res["file_count"] == 2
    assert res["validation_errors"] == []


# ── ontology synergy ──────────────────────────────────────────────────────────


def test_skillgraph_ontology_interface_registered_and_owl():
    from agent_utilities.knowledge_graph.ontology.interfaces import (
        DEFAULT_INTERFACE_REGISTRY,
    )

    iface = DEFAULT_INTERFACE_REGISTRY.get("SkillGraph")
    assert iface is not None
    link_names = {lc.name for lc in iface.link_constraints}
    assert {"contains", "covers", "derived_from"} <= link_names
    assert "SkillGraph" in DEFAULT_INTERFACE_REGISTRY.to_owl()
