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
    res = _pipe().build(
        name="widget-docs",
        specs=[SourceSpec("dir", str(src_dir))],
        out_dir=tmp_path / "out",
    )
    assert res["validation_errors"] == []
    assert validate_skill_graph(tmp_path / "out" / "widget-docs") == []


def test_validate_missing_skill_md(tmp_path):
    (tmp_path / "empty").mkdir()
    errors = validate_skill_graph(tmp_path / "empty")
    assert errors and "missing SKILL.md" in errors[0]


def test_validate_name_mismatch(src_dir, tmp_path):
    _pipe().build(
        name="widget-docs",
        specs=[SourceSpec("dir", str(src_dir))],
        out_dir=tmp_path / "out",
    )
    graph = tmp_path / "out" / "widget-docs"
    renamed = graph.parent / "other-docs"
    graph.rename(renamed)
    errors = validate_skill_graph(renamed)
    assert any("!= directory name" in e for e in errors)


def test_validate_flags_missing_manifest(src_dir, tmp_path):
    _pipe().build(
        name="widget-docs",
        specs=[SourceSpec("dir", str(src_dir))],
        out_dir=tmp_path / "out",
    )
    graph = tmp_path / "out" / "widget-docs"
    (graph / "sources.json").unlink()
    errors = validate_skill_graph(graph)
    assert any("missing sources.json" in e for e in errors)


# ── source kinds ──────────────────────────────────────────────────────────────


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        SourceSpec("bogus", "x")


def test_build_dir_source(src_dir, tmp_path):
    res = _pipe().build(
        name="widget-docs",
        specs=[SourceSpec("dir", str(src_dir))],
        out_dir=tmp_path / "out",
    )
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
        return [
            AcquiredDoc(rel_path="index.md", text="# Home\n\nhi\n", source_uri=spec.uri)
        ]

    pipe = SkillGraphPipeline(crawler_fn=crawler_fn, kg_enrich=False)
    res = pipe.build(
        name="site-docs",
        specs=[SourceSpec("web", "https://x")],
        out_dir=tmp_path / "out",
    )
    assert res["file_count"] == 1
    fm = parse_frontmatter((tmp_path / "out" / "site-docs" / "SKILL.md").read_text())
    assert fm["source_types"] == ["web"]


def test_build_generated_via_injected_generator(tmp_path):
    def generator_fn(spec):
        return [
            AcquiredDoc(
                rel_path="topic.md",
                text=f"# {spec.uri}\n\ngenerated\n",
                source_uri="generated://topic",
            )
        ]

    pipe = SkillGraphPipeline(generator_fn=generator_fn, kg_enrich=False)
    res = pipe.build(
        name="topic-docs",
        specs=[SourceSpec("generated", "Kubernetes networking")],
        out_dir=tmp_path / "out",
    )
    assert res["file_count"] == 1


def test_multi_source_merge_dedupes_identical_content(src_dir, tmp_path):
    # Two dir sources yielding the SAME widget.md/auth.md → content optimization drops
    # the exact-duplicate pages, so the corpus is 2 files (not 4) across 2 sources.
    res = _pipe().build(
        name="merged-docs",
        specs=[SourceSpec("dir", str(src_dir)), SourceSpec("dir", str(src_dir))],
        out_dir=tmp_path / "out",
    )
    assert res["file_count"] == 2
    assert res["source_count"] == 2


# ── freshness + rebuild ───────────────────────────────────────────────────────


def test_index_json_and_polished_skill_md(src_dir, tmp_path):
    _pipe().build(
        name="widget-docs",
        specs=[SourceSpec("dir", str(src_dir))],
        out_dir=tmp_path / "out",
    )
    graph = tmp_path / "out" / "widget-docs"
    idx = json.loads((graph / "index.json").read_text())
    assert idx["schema"] == "skill-graph-index/v1"
    assert idx["file_count"] == 2
    assert {s["path"] for s in idx["sections"]} == {
        "reference/auth.md",
        "reference/widget.md",
    }
    assert any(s["headings"] for s in idx["sections"])  # headings extracted
    skill_md = (graph / "SKILL.md").read_text()
    assert "## 🧭 How to use this skill-graph" in skill_md
    assert "index: index.json" in skill_md
    assert "| **Files** |" in skill_md  # badge table


def test_distill_writes_overview_and_links_it(src_dir, tmp_path):
    def distiller_fn(name, digest):
        assert "FILE:" in digest  # corpus digest fed to the distiller
        return f"# {name} — Distilled Overview\n\n## What it is\nA test corpus.\n"

    pipe = SkillGraphPipeline(distiller_fn=distiller_fn, kg_enrich=False)
    pipe.build(
        name="widget-docs",
        specs=[SourceSpec("dir", str(src_dir))],
        out_dir=tmp_path / "out",
    )
    graph = tmp_path / "out" / "widget-docs"
    res = pipe.distill_one(graph)
    assert res["status"] == "distilled"
    assert (graph / "OVERVIEW.md").exists()
    skill_md = (graph / "SKILL.md").read_text()
    assert "overview: OVERVIEW.md" in skill_md
    assert "[OVERVIEW.md](OVERVIEW.md)" in skill_md
    assert json.loads((graph / "sources.json").read_text())["distilled"] is True


def test_restyle_rerenders_without_recrawl(src_dir, tmp_path):
    pipe = _pipe()
    pipe.build(
        name="widget-docs",
        specs=[SourceSpec("dir", str(src_dir))],
        out_dir=tmp_path / "out",
    )
    graph = tmp_path / "out" / "widget-docs"
    (graph / "SKILL.md").write_text("---\nname: widget-docs\n---\nstale\n")  # clobber
    (graph / "index.json").unlink()
    res = pipe.restyle_one(graph)
    assert res["status"] == "restyled" and res["file_count"] == 2
    assert (graph / "index.json").exists()
    assert "## 🧭 How to use this skill-graph" in (graph / "SKILL.md").read_text()


def test_status_fresh_then_stale_then_rebuild(src_dir, tmp_path):
    pipe = _pipe()
    pipe.build(
        name="widget-docs",
        specs=[SourceSpec("dir", str(src_dir))],
        out_dir=tmp_path / "out",
    )
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


def test_kg_enrichment_degrades_when_ingest_fails(src_dir, tmp_path, monkeypatch):
    """kg_enrich=True but the ingest subprocess fails → kg_ingested False, graph still built."""
    import agent_utilities.knowledge_graph.distillation.skill_graph_pipeline as mod

    # Embedder probe says "unknown" (proceed), then the bounded ingest run fails.
    monkeypatch.setattr(mod, "_embedder_responsive", lambda *a, **k: None)
    monkeypatch.setattr(
        mod, "_run_bounded", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError())
    )
    pipe = SkillGraphPipeline(kg_enrich=True)
    res = pipe.build(
        name="widget-docs",
        specs=[SourceSpec("dir", str(src_dir))],
        out_dir=tmp_path / "out",
    )
    assert res["kg_ingested"] is False
    assert res["file_count"] == 2
    assert res["validation_errors"] == []


def test_kg_health_gate_skips_fast_when_embedder_down(src_dir, tmp_path, monkeypatch):
    """A definitively-down embedder is skipped without ever shelling out to ingest."""
    import agent_utilities.knowledge_graph.distillation.skill_graph_pipeline as mod

    monkeypatch.setattr(mod, "_embedder_responsive", lambda *a, **k: False)

    def fail_run(*a, **k):  # must NOT be reached
        raise AssertionError("ingest subprocess should not run when embedder is down")

    monkeypatch.setattr(mod, "_run_bounded", fail_run)
    pipe = SkillGraphPipeline(kg_enrich=True)
    res = pipe.build(
        name="widget-docs",
        specs=[SourceSpec("dir", str(src_dir))],
        out_dir=tmp_path / "out",
    )
    assert res["kg_ingested"] is False


def test_run_bounded_returns_and_kills_on_timeout():
    import time as _t

    from agent_utilities.knowledge_graph.distillation.skill_graph_pipeline import (
        _run_bounded,
    )

    rc, out, _err = _run_bounded(["printf", "hello"], timeout=10)
    assert rc == 0 and out == "hello"

    # A sleep that exceeds the timeout is killed (process group) and raises.
    import subprocess

    start = _t.time()
    with pytest.raises(subprocess.TimeoutExpired):
        _run_bounded(["sleep", "30"], timeout=1)
    assert _t.time() - start < 10  # returned promptly, did not wait out the sleep


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


def test_refresh_writes_only_the_delta(tmp_path):
    # A multi-file web graph; refresh must touch ONLY the changed files (not rewrite
    # the whole tree), preserving unchanged files' bytes/mtime.
    files = {
        "a.md": "# A\n\nalpha\n",
        "b.md": "# B\n\nbeta\n",
        "c.md": "# C\n\ngamma\n",
    }

    def crawler_fn(spec):
        return [
            AcquiredDoc(rel_path=k, text=v, source_uri=spec.uri)
            for k, v in files.items()
        ]

    pipe = SkillGraphPipeline(crawler_fn=crawler_fn, kg_enrich=False)
    pipe.build(
        name="site-docs",
        specs=[SourceSpec("web", "https://x")],
        out_dir=tmp_path / "out",
    )
    graph = tmp_path / "out" / "site-docs"
    ref = graph / "reference"
    mtimes = {p.name: p.stat().st_mtime_ns for p in ref.glob("*.md")}

    # change b.md, add d.md, remove c.md
    files["b.md"] = "# B\n\nBETA CHANGED\n"
    files["d.md"] = "# D\n\ndelta\n"
    del files["c.md"]
    res = pipe.refresh_one(graph)

    assert res["status"] == "refreshed"
    assert res["delta"] == {"added": 1, "changed": 1, "removed": 1, "unchanged": 1}
    after = {p.name: p.stat().st_mtime_ns for p in ref.glob("*.md")}
    assert after["a.md"] == mtimes["a.md"]  # unchanged → not rewritten
    assert after["b.md"] != mtimes["b.md"]  # changed → rewritten
    assert not (ref / "c.md").exists()  # removed
    assert (ref / "d.md").exists()  # added
    assert "BETA CHANGED" in (ref / "b.md").read_text()


def test_refresh_no_file_delta_is_fresh(tmp_path):
    # Source bytes differ (trailing whitespace) but the optimized files are identical
    # → no file delta → fresh, no version bump.
    state = {"text": "# A\n\nbody\n"}

    def crawler_fn(spec):
        return [AcquiredDoc(rel_path="a.md", text=state["text"], source_uri=spec.uri)]

    pipe = SkillGraphPipeline(crawler_fn=crawler_fn, kg_enrich=False)
    pipe.build(
        name="s-docs", specs=[SourceSpec("web", "https://x")], out_dir=tmp_path / "out"
    )
    graph = tmp_path / "out" / "s-docs"
    state["text"] = "# A\n\nbody\n\n\n   "  # only trailing whitespace differs
    res = pipe.refresh_one(graph)
    assert res["status"] == "fresh"
    assert res["delta"]["changed"] == 0


# ── legacy migration ──────────────────────────────────────────────────────────


def _legacy_graph(root: Path, name: str, *, source_url: str | None, files: int) -> Path:
    """Write a legacy-format skill-graph (old frontmatter, no sources.json)."""
    d = root / name
    (d / "reference").mkdir(parents=True)
    for i in range(files):
        (d / "reference" / f"f{i}.md").write_text(
            f"# F{i}\n\nbody {i}\n", encoding="utf-8"
        )
    fm = [f"name: {name}", "description: Legacy docs.", "crawl_depth: 3"]
    if source_url:
        fm.append(f"source_url: {source_url}")
    (d / "SKILL.md").write_text(
        "---\n" + "\n".join(fm) + "\n---\n# legacy\n", encoding="utf-8"
    )
    return d


def test_classify_legacy_modes(tmp_path):
    pipe = _pipe()
    reacq = _legacy_graph(tmp_path, "a-docs", source_url="https://x/docs", files=2)
    wrap = _legacy_graph(tmp_path, "b-docs", source_url=None, files=2)
    native = tmp_path / "c"
    (native).mkdir()
    (native / "SKILL.md").write_text("---\nname: c\n---\n# native\n", encoding="utf-8")

    assert pipe.classify_legacy(reacq)["mode"] == "reacquire"
    assert pipe.classify_legacy(wrap)["mode"] == "wrap"
    assert pipe.classify_legacy(native)["mode"] == "native"


def test_migrate_wrap_preserves_content_and_standardizes(tmp_path):
    g = _legacy_graph(tmp_path, "b-docs", source_url=None, files=3)
    res = _pipe().migrate_legacy(g, mode="auto")
    assert res["migrated_mode"] == "wrap"
    assert res["version"] == "1.0.0"
    assert res["file_count"] == 3
    assert res["validation_errors"] == []
    manifest = json.loads((g / "sources.json").read_text())
    assert manifest["schema"] == SOURCES_SCHEMA
    fm = parse_frontmatter((g / "SKILL.md").read_text())
    assert fm["skill_graph_version"] == "1.0.0"


def test_migrate_wrap_records_upstream_web_provenance(tmp_path):
    # A crawled legacy graph (has source_url) wrapped offline: content is adopted from
    # the existing reference/, but the durable sources are the web URLs so it stays
    # re-crawlable. No temp path leaks into the manifest.
    g = _legacy_graph(tmp_path, "site-docs", source_url="https://site/docs", files=2)
    res = _pipe().migrate_legacy(g, mode="wrap")
    assert res["migrated_mode"] == "wrap" and res["file_count"] == 2
    manifest = json.loads((g / "sources.json").read_text())
    assert [s["kind"] for s in manifest["sources"]] == ["web"]
    assert manifest["sources"][0]["uri"] == "https://site/docs"
    assert "/tmp/" not in json.dumps(manifest["sources"])
    fm = parse_frontmatter((g / "SKILL.md").read_text())
    assert fm["source_types"] == ["web"]
    assert fm["source_url"] == "https://site/docs"


def test_migrate_reacquire_uses_source_url(tmp_path):
    seen = {}

    def crawler_fn(spec):
        seen["uri"] = spec.uri
        return [
            AcquiredDoc(
                rel_path="page.md", text="# Fresh\n\nnew\n", source_uri=spec.uri
            )
        ]

    pipe = SkillGraphPipeline(crawler_fn=crawler_fn, kg_enrich=False)
    g = _legacy_graph(tmp_path, "a-docs", source_url="https://x/docs", files=2)
    res = pipe.migrate_legacy(g, mode="auto")
    assert res["migrated_mode"] == "reacquire"
    assert seen["uri"] == "https://x/docs"
    fm = parse_frontmatter((g / "SKILL.md").read_text())
    assert fm["source_types"] == ["web"]


def test_refresh_skips_unchanged_and_rebuilds_changed(tmp_path):
    # A controllable crawler whose output we can flip to simulate upstream change.
    state = {"text": "# Home\n\nv1\n", "calls": 0}

    def crawler_fn(spec):
        state["calls"] += 1
        return [
            AcquiredDoc(rel_path="index.md", text=state["text"], source_uri=spec.uri)
        ]

    pipe = SkillGraphPipeline(crawler_fn=crawler_fn, kg_enrich=False)
    pipe.build(
        name="site-docs",
        specs=[SourceSpec("web", "https://x")],
        out_dir=tmp_path / "out",
    )
    graph = tmp_path / "out" / "site-docs"

    # Unchanged upstream → refresh is a no-op (fresh), no version bump.
    r1 = pipe.refresh_one(graph)
    assert r1["status"] == "fresh"

    # Upstream changes → refresh rewrites + bumps version.
    state["text"] = "# Home\n\nv2 CHANGED\n"
    r2 = pipe.refresh_one(graph)
    assert r2["status"] == "refreshed"
    assert r2["version"] == "0.1.1"
    assert "v2 CHANGED" in (graph / "reference" / "index.md").read_text()

    # Now unchanged again.
    assert pipe.refresh_one(graph)["status"] == "fresh"


def test_refresh_force_rewrites_unchanged(tmp_path):
    def crawler_fn(spec):
        return [
            AcquiredDoc(
                rel_path="index.md", text="# Home\n\nsame\n", source_uri=spec.uri
            )
        ]

    pipe = SkillGraphPipeline(crawler_fn=crawler_fn, kg_enrich=False)
    pipe.build(
        name="site-docs",
        specs=[SourceSpec("web", "https://x")],
        out_dir=tmp_path / "out",
    )
    graph = tmp_path / "out" / "site-docs"
    assert pipe.refresh_one(graph, force=True)["status"] == "refreshed"


def test_refresh_shrink_guard_keeps_content_on_moved_url(tmp_path):
    # Build a content-rich graph, then simulate the source_url having moved (re-crawl
    # returns a tiny landing page). The guard keeps the rich content + flags stale_url.
    big = "# Docs\n\n" + ("lorem ipsum dolor sit amet " * 1000) + "\n"
    state = {"text": big}

    def crawler_fn(spec):
        return [
            AcquiredDoc(rel_path="index.md", text=state["text"], source_uri=spec.uri)
        ]

    pipe = SkillGraphPipeline(crawler_fn=crawler_fn, kg_enrich=False)
    pipe.build(
        name="site-docs",
        specs=[SourceSpec("web", "https://x")],
        out_dir=tmp_path / "out",
    )
    graph = tmp_path / "out" / "site-docs"
    before = (graph / "reference").rglob("*.md")
    before_bytes = sum(p.stat().st_size for p in before)

    state["text"] = "# Moved\n\nThis page has relocated.\n"  # tiny re-crawl
    res = pipe.refresh_one(graph, force=True)
    assert res["status"] == "stale_url"
    # Existing rich content is untouched.
    kept = sum(p.stat().st_size for p in (graph / "reference").rglob("*.md"))
    assert kept == before_bytes

    # With the guard disabled, it overwrites.
    res2 = pipe.refresh_one(graph, force=True, shrink_guard=False)
    assert res2["status"] == "refreshed"


def test_refresh_one_unmanaged_is_skipped(tmp_path):
    (tmp_path / "g").mkdir()
    assert _pipe().refresh_one(tmp_path / "g")["status"] == "skipped"


def test_refresh_all_reports_per_graph(tmp_path):
    def crawler_fn(spec):
        return [AcquiredDoc(rel_path="i.md", text="# x\n", source_uri=spec.uri)]

    pipe = SkillGraphPipeline(crawler_fn=crawler_fn, kg_enrich=False)
    root = tmp_path / "root"
    for n in ("a-docs", "b-docs"):
        pipe.build(name=n, specs=[SourceSpec("web", f"https://{n}")], out_dir=root)
    report = pipe.refresh_all(root)
    assert {r["name"] for r in report["results"]} == {"a-docs", "b-docs"}
    assert all(r["status"] == "fresh" for r in report["results"])


def test_migrate_skips_native(tmp_path):
    native = tmp_path / "c"
    native.mkdir()
    (native / "SKILL.md").write_text("---\nname: c\n---\n# native\n", encoding="utf-8")
    res = _pipe().migrate_legacy(native, mode="auto")
    assert res["skipped"] is True and res["reason"] == "native"
