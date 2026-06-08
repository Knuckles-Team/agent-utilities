"""CONCEPT:KG-2.19 — Self-Curating Wiki.

Covers SHA-256 delta detection (only changed pages re-ingested), atomic state persistence, and the
dry-run path — using an injected ingest_fn so no real IngestionEngine is needed.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ingestion.wiki_curator import (
    WikiCurator,
    file_hash,
)


def _write(d, name, text):
    p = d / name
    p.write_text(text)
    return p


@pytest.mark.concept(id="KG-2.19")
def test_file_hash_changes_with_content(tmp_path):
    p = _write(tmp_path, "a.md", "hello")
    h1 = file_hash(p)
    p.write_text("hello world")
    assert file_hash(p) != h1


@pytest.mark.concept(id="KG-2.19")
def test_first_run_ingests_all_then_delta_skips(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    _write(wiki, "a.md", "alpha")
    _write(wiki, "b.md", "beta")
    state = tmp_path / "state.json"
    ingested = []
    curator = WikiCurator(state)

    # First run: both pages are new → ingested.
    s1 = curator.curate(wiki, lambda p: ingested.append(p.name))
    assert s1["ingested"] == 2 and s1["skipped"] == 0
    assert sorted(ingested) == ["a.md", "b.md"]

    # Second run, nothing changed → all skipped (delta).
    ingested.clear()
    s2 = curator.curate(wiki, lambda p: ingested.append(p.name))
    assert s2["ingested"] == 0 and s2["skipped"] == 2
    assert ingested == []


@pytest.mark.concept(id="KG-2.19")
def test_only_changed_page_reingested(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    _write(wiki, "a.md", "alpha")
    b = _write(wiki, "b.md", "beta")
    state = tmp_path / "state.json"
    curator = WikiCurator(state)
    curator.curate(wiki, lambda p: None)  # prime state

    b.write_text("beta v2")  # change only b
    ingested = []
    s = curator.curate(wiki, lambda p: ingested.append(p.name))
    assert ingested == ["b.md"] and s["ingested"] == 1 and s["skipped"] == 1


@pytest.mark.concept(id="KG-2.19")
def test_changed_files_lists_new_and_modified(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    _write(wiki, "a.md", "alpha")
    curator = WikiCurator(tmp_path / "state.json")
    assert [p.name for p in curator.changed_files(wiki)] == ["a.md"]


@pytest.mark.concept(id="KG-2.19")
def test_dry_run_does_not_ingest_or_write_state(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    _write(wiki, "a.md", "alpha")
    state = tmp_path / "state.json"
    curator = WikiCurator(state)
    ingested = []
    s = curator.curate(wiki, lambda p: ingested.append(p.name), dry_run=True)
    assert s["dry_run"] is True and ingested == []
    assert s["would_ingest"] == [str(wiki / "a.md")]
    assert not state.exists()  # no state committed


@pytest.mark.concept(id="KG-2.19")
def test_state_persisted_atomically(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    _write(wiki, "a.md", "alpha")
    state = tmp_path / "state.json"
    WikiCurator(state).curate(wiki, lambda p: None)
    assert state.is_file()
    import json

    assert str(wiki / "a.md") in json.loads(state.read_text())
