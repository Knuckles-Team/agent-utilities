"""Track 2 — OKF round-trip + openwiki, standardized on skill-graphs.

Covers the five Track-2 capabilities:
* AU-KG.research.okf-overlay-mode — permissive consumer + thin-overlay concept.
* AU-ECO.connector.okf-roundtrip-sync — .catalog.state push / conflict / delete.
* AU-ECO.connector.openwiki-preset — filesystem preset + snapshot watermark + OKF stamp.
* AU-KG.ingest.okf-type-mapping — external type → governed domain + review queue.
* AU-KG.ingest.broken-link-tolerance — dangling-node placeholder for broken links.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.distillation import okf_bundle as ob


# ── permissive consumer (SPEC §9) + REQUIRED-key alignment (§4.1) ────────────
def test_read_frontmatter_tolerates_unknown_type_and_keys():
    fm, body = ob.read_frontmatter(
        '---\ntype: Totally New Type\ntitle: "X"\nweird: [a, b]\n---\n# Body\ntext'
    )
    assert fm["type"] == "Totally New Type"
    assert fm["title"] == "X"
    assert fm["weird"] == ["a", "b"]
    assert body.startswith("# Body")
    assert ob.frontmatter_conforms(fm)  # only `type` is required


def test_read_frontmatter_no_block_returns_whole_text():
    fm, body = ob.read_frontmatter("no frontmatter here")
    assert fm == {}
    assert body == "no frontmatter here"


def test_frontmatter_text_is_non_destructive_when_present():
    already = "---\ntype: Wiki\n---\nbody"
    assert ob.frontmatter_text(already, ftype="Reference") == already


def test_required_keys_is_only_type():
    assert ob.REQUIRED_KEYS == ("type",)


# ── thin-overlay concept mode ────────────────────────────────────────────────
def test_render_overlay_concept_defers_to_source():
    md = ob.render_overlay_concept(
        title="Users Table",
        ftype="BigQuery Table",
        resource="bq://proj/ds/users",
        concept_id="AU-KG.domains.users",
        links=[("orders", "orders.md")],
    )
    fm, body = ob.read_frontmatter(md)
    assert fm["type"] == "BigQuery Table"
    assert fm["resource"] == "bq://proj/ds/users"
    assert fm["overlay"] == "true"
    assert fm["id"] == "AU-KG.domains.users"
    assert "Overlay concept" in body and "orders.md" in body
    # No source body duplicated — just the pointer + links.
    assert "bq://proj/ds/users" in body


# ── round-trip sync: create → conflict → intent-to-delete ────────────────────
def _make_skill_graph(root: Path) -> Path:
    sg = root / "sg"
    ref = sg / "reference"
    ref.mkdir(parents=True)
    (ref / "a.md").write_text("# A\nalpha", encoding="utf-8")
    (ref / "b.md").write_text("# B\nbeta", encoding="utf-8")
    (sg / "SKILL.md").write_text("---\nname: sg\n---\n# sg", encoding="utf-8")
    (sg / "sources.json").write_text('{"name": "sg"}', encoding="utf-8")
    return sg


def test_roundtrip_push_creates_then_is_idempotent(tmp_path):
    sg = _make_skill_graph(tmp_path)
    cat = tmp_path / "cat"
    sync = ob.OkfRoundTripSync(sg, cat)

    preview = sync.push(dry_run=True)
    assert preview["status"] == "previewed"
    assert "reference/a.md" in preview["creates"]

    pushed = sync.push(dry_run=False)
    assert pushed["status"] == "pushed"
    assert (cat / "reference" / "a.md").read_text(encoding="utf-8") == "# A\nalpha"
    assert (cat / ob.STATE_FILENAME).exists()

    # A second push with no source change → everything unchanged.
    again = sync.push(dry_run=True)
    assert again["creates"] == [] and again["updates"] == []
    assert "reference/a.md" in again["unchanged"]


def test_roundtrip_fail_fast_on_interim_conflict(tmp_path):
    sg = _make_skill_graph(tmp_path)
    cat = tmp_path / "cat"
    sync = ob.OkfRoundTripSync(sg, cat)
    sync.push(dry_run=False)

    # Someone edits the catalog copy out-of-band after the last sync.
    (cat / "reference" / "a.md").write_text("# A\nHAND EDITED", encoding="utf-8")

    plan = sync.push(dry_run=True)
    assert "reference/a.md" in plan["conflicts"]
    assert plan["status"] == "conflict"

    with pytest.raises(ob.OkfConflictError):
        sync.push(dry_run=False)

    # force overrides the conflict and rewrites from source.
    forced = sync.push(dry_run=False, force=True)
    assert forced["status"] == "pushed"
    assert (cat / "reference" / "a.md").read_text(encoding="utf-8") == "# A\nalpha"


def test_roundtrip_intent_to_delete(tmp_path):
    sg = _make_skill_graph(tmp_path)
    cat = tmp_path / "cat"
    sync = ob.OkfRoundTripSync(sg, cat)
    sync.push(dry_run=False)

    # Remove a source file → it becomes intent-to-delete (only with allow_delete).
    (sg / "reference" / "b.md").unlink()
    no_del = sync.push(dry_run=True, allow_delete=False)
    assert no_del["deletes"] == []
    with_del = sync.push(dry_run=True, allow_delete=True)
    assert "reference/b.md" in with_del["deletes"]

    sync.push(dry_run=False, allow_delete=True)
    assert not (cat / "reference" / "b.md").exists()
    assert "reference/b.md" not in ob.read_catalog_state(cat)


# ── external type → governed domain + review queue ───────────────────────────
def test_map_external_type_seed_and_signal():
    assert ob.map_external_type("Reference") == ("KG", "research")
    assert ob.map_external_type("API Endpoint") == ("KG", "query")
    # signal match against the closed domain vocab (sparql is a `query` signal)
    assert ob.map_external_type("sparql") == ("KG", "query")


def test_resolve_type_domain_queues_unmapped(tmp_path):
    q = tmp_path / "queue.json"
    dom = ob.resolve_type_domain("Zorblax Widget", queue_path=q, provenance="repoX")
    assert dom == ob.DEFAULT_TYPE_DOMAIN
    queued = ob.list_type_review_queue(queue_path=q)
    assert queued[0]["type"] == "Zorblax Widget"
    assert "repoX" in queued[0]["provenance"]
    # re-queue with a new provenance dedups on type
    ob.resolve_type_domain("Zorblax Widget", queue_path=q, provenance="repoY")
    queued = ob.list_type_review_queue(queue_path=q)
    assert len(queued) == 1 and set(queued[0]["provenance"]) == {"repoX", "repoY"}


# ── openwiki filesystem preset ───────────────────────────────────────────────
def _make_openwiki(repo: Path, last_update: str) -> None:
    wiki = repo / "openwiki"
    wiki.mkdir(parents=True)
    (wiki / "Home.md").write_text("# Home\nWelcome to the wiki", encoding="utf-8")
    (wiki / "Guide.md").write_text("# Guide\nHow to use it", encoding="utf-8")
    (wiki / ".last-update.json").write_text(
        json.dumps({"updated": last_update}), encoding="utf-8"
    )


def test_openwiki_preset_stamps_okf_and_snapshot_delta(tmp_path):
    from agent_utilities.protocols.source_connectors.registry import build_connector

    repo = tmp_path / "my-repo"
    _make_openwiki(repo, "2026-01-01")
    conn = build_connector("filesystem", {"preset": "openwiki", "root": str(repo)})

    # First poll ingests both pages, OKF-stamped, with per-repo SLUG provenance.
    batch = conn.poll(None)
    assert len(batch.documents) == 2
    doc = batch.documents[0]
    fm, _ = ob.read_frontmatter(doc.text)
    assert fm["type"] == "wiki"
    assert doc.metadata["slug"] == "my-repo"
    assert doc.metadata["okf_domain"]  # mapped to a governed domain

    # Re-poll with the SAME .last-update.json snapshot → zero docs (delta skip).
    again = conn.poll(batch.checkpoint)
    assert again.documents == []
    assert again.checkpoint.watermark == batch.checkpoint.watermark

    # Bump the snapshot → the corpus re-ingests.
    _make_openwiki_bump(repo, "2026-02-02")
    changed = conn.poll(batch.checkpoint)
    assert len(changed.documents) == 2


def _make_openwiki_bump(repo: Path, last_update: str) -> None:
    (repo / "openwiki" / ".last-update.json").write_text(
        json.dumps({"updated": last_update}), encoding="utf-8"
    )


def test_openwiki_preset_unknown_preset_raises(tmp_path):
    from agent_utilities.protocols.source_connectors.registry import build_connector

    with pytest.raises(ValueError):
        build_connector("filesystem", {"preset": "nope", "root": str(tmp_path)})


# ── broken-link tolerance → dangling placeholder ─────────────────────────────
def _fake_embed(texts):
    """Deterministic offline embedder — keeps these tests hermetic (no network).

    Link extraction / byte-identity have nothing to do with embeddings; injecting
    a stub via the ``embed_fn`` seam avoids the live vLLM embedder retry-backoff.
    """
    return [[0.0] for _ in texts]


def test_broken_link_creates_dangling_placeholder_and_keeps_edge():
    from agent_utilities.knowledge_graph.ontology.document_processing import (
        DocumentProcessor,
    )

    proc = DocumentProcessor(graph=None, extract_links=True, embed_fn=_fake_embed)
    text = "# Doc\nSee [Other](other.md) and [Site](https://x/y).\n"
    result = proc.process(text, text=text, source="home.md", persist=False)

    link_edges = [e for e in result.edges if e["type"] == "LINKS_TO"]
    assert {e["href"] for e in link_edges} == {"other.md", "https://x/y"}
    # broken/forward link target → a dangling placeholder node (edge never dropped)
    dangling = {n["id"]: n for n in result.link_nodes}
    assert any(n["dangling"] for n in dangling.values())
    ext = [n for n in result.link_nodes if n["external"]]
    assert ext and ext[0]["source"] == "https://x/y"


def test_extract_links_off_by_default_is_byte_identical():
    from agent_utilities.knowledge_graph.ontology.document_processing import (
        DocumentProcessor,
    )

    proc = DocumentProcessor(graph=None, embed_fn=_fake_embed)
    text = "# Doc\nSee [Other](other.md).\n"
    result = proc.process(text, text=text, source="home.md", persist=False)
    assert result.link_nodes == []
    assert all(e["type"] in ("HAS_CHUNK", "CHUNK_OF") for e in result.edges)


# ── OKF writeback sink on the existing graph_writeback surface ───────────────
def test_okf_sink_dry_run_previews_plan(tmp_path):
    from agent_utilities.knowledge_graph.enrichment.writeback import run_writeback

    sg = _make_skill_graph(tmp_path)
    cat = tmp_path / "cat"
    out = run_writeback(
        "okf",
        dry_run=True,
        creations=[{"skill_dir": str(sg), "catalog_dir": str(cat)}],
    )
    assert out["status"] == "completed"
    assert out["proposals"] and out["proposals"][0]["op"] == "okf_push"
    assert "reference/a.md" in out["proposals"][0]["creates"]
