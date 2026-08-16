"""Tests for the git-markdown source connector (CONCEPT:AU-ECO.connector.git-markdown-revision-connector).

Offline + deterministic: every test drives a REAL, throwaway local ``git``
repository under ``tmp_path`` (no network, no external service) — the
connector's whole point is git-revision semantics, so a fake filesystem walk
would not exercise the thing being tested.
"""

from __future__ import annotations

import subprocess

import pytest

from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope
from agent_utilities.protocols.source_connectors import (
    ConnectorCheckpoint,
    LoadConnector,
    PollConnector,
    build_connector,
    list_sources,
)
from agent_utilities.protocols.source_connectors.connectors.git_markdown import (
    GIT_MARKDOWN_PRESETS,
    GitMarkdownError,
    fragment_markdown,
)


def _git(repo, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()


def _init_repo(tmp_path) -> object:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.com")
    _git(repo, "config", "user.name", "t")
    return repo


def _commit(repo, message: str = "commit") -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.mark.concept("AU-ECO.connector.git-markdown-revision-connector")
def test_registry_discovers_git_markdown():
    assert "git_markdown" in list_sources()


@pytest.mark.concept("AU-ECO.connector.git-markdown-revision-connector")
def test_configure_requires_a_git_working_tree(tmp_path):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    with pytest.raises(GitMarkdownError):
        build_connector("git_markdown", {"root": str(not_a_repo)})


@pytest.mark.concept("AU-ECO.connector.git-markdown-revision-connector")
def test_configure_unknown_preset_lists_available(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "a.md").write_text("# A\n")
    _commit(repo)
    with pytest.raises(ValueError, match="Available"):
        build_connector("git_markdown", {"root": str(repo), "preset": "nope"})


@pytest.mark.concept("AU-ECO.connector.git-markdown-revision-connector")
def test_presets_carry_no_local_root(tmp_path):
    # Mirrors FilesystemConnector.FILESYSTEM_PRESETS: a preset supplies
    # everything EXCEPT `root` (the deployment-local checkout path).
    for preset in GIT_MARKDOWN_PRESETS.values():
        assert "root" not in preset


@pytest.mark.concept("AU-ECO.connector.git-markdown-revision-connector")
def test_load_reads_content_at_head_revision_verbatim(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "docs").mkdir()
    (repo / "docs" / "a.md").write_text("# Title A\nalpha body\n")
    (repo / "docs" / "b.md").write_text("# Title B\nbeta body\n")
    (repo / "docs" / "c.txt").write_text("not markdown\n")
    sha = _commit(repo)

    conn = build_connector("git_markdown", {"root": str(repo), "subdir": "docs"})
    assert isinstance(conn, LoadConnector)
    docs = list(conn.load())
    assert {d.title for d in docs} == {"Title A", "Title B"}
    assert all(d.metadata["git_commit"] == sha for d in docs)
    assert all(d.source_uri.startswith("git-markdown://") for d in docs)
    assert all(str(repo) not in d.source_uri for d in docs)


@pytest.mark.concept("AU-ECO.connector.incremental-poll-watermark")
def test_poll_first_run_then_unchanged_is_a_no_op(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "a.md").write_text("# A\ncontent\n")
    _commit(repo)

    conn = build_connector("git_markdown", {"root": str(repo)})
    assert isinstance(conn, PollConnector)
    first = conn.poll(None)
    assert len(first.documents) == 1
    assert conn.last_envelopes and conn.last_envelopes[0].operation == "upsert"

    again = conn.poll(first.checkpoint)
    assert again.documents == []
    assert conn.last_envelopes == []
    assert again.checkpoint.watermark == first.checkpoint.watermark


@pytest.mark.concept("AU-ECO.connector.git-markdown-revision-connector")
def test_poll_diff_feed_covers_add_modify_delete_and_stable_ids(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "a.md").write_text("# A\n")
    (repo / "b.md").write_text("# B\n")
    sha1 = _commit(repo)

    conn = build_connector("git_markdown", {"root": str(repo)})
    a_uri_before = conn._portable_uri("a.md")

    (repo / "a.md").write_text("# A changed\n")
    (repo / "b.md").unlink()
    (repo / "c.md").write_text("# C new\n")
    sha2 = _commit(repo)

    batch = conn.poll(ConnectorCheckpoint(has_more=False, watermark=sha1))
    assert batch.checkpoint.watermark == sha2
    changed_paths = {d.metadata["relpath"] for d in batch.documents}
    assert changed_paths == {"a.md", "c.md"}
    ops = {e.provenance.get("relpath"): e.operation for e in conn.last_envelopes}
    assert ops == {"a.md": "upsert", "b.md": "delete", "c.md": "upsert"}
    # a.md's portable URI (and therefore its DocumentProcessor-owned node id)
    # never changes across a content edit — stable ids for the untouched shape.
    assert conn._portable_uri("a.md") == a_uri_before


@pytest.mark.concept("AU-ECO.connector.git-markdown-revision-connector")
def test_poll_treats_a_rename_as_delete_plus_upsert(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "old.md").write_text("# Renamed doc\nsome real content here\n")
    sha1 = _commit(repo)

    _git(repo, "mv", "old.md", "new.md")
    sha2 = _commit(repo, "rename")

    conn = build_connector("git_markdown", {"root": str(repo)})
    batch = conn.poll(ConnectorCheckpoint(has_more=False, watermark=sha1))
    assert batch.checkpoint.watermark == sha2
    ops = {e.provenance.get("relpath"): e.operation for e in conn.last_envelopes}
    assert ops == {"old.md": "delete", "new.md": "upsert"}


@pytest.mark.concept("AU-ECO.connector.git-markdown-revision-connector")
def test_filenames_filter_restricts_to_exact_basename(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "skills").mkdir()
    (repo / "skills" / "a").mkdir()
    (repo / "skills" / "a" / "SKILL.md").write_text("---\nname: a\n---\n# A\n")
    (repo / "skills" / "a" / "README.md").write_text("# not a skill\n")
    _commit(repo)

    conn = build_connector(
        "git_markdown",
        {"root": str(repo), "subdir": "skills", "filenames": ["SKILL.md"]},
    )
    docs = list(conn.load())
    assert [d.metadata["relpath"] for d in docs] == ["skills/a/SKILL.md"]


@pytest.mark.concept("AU-KG.ingest.change-envelope")
def test_upsert_and_delete_envelopes_carry_the_contract_fields(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "a.md").write_text("# A\nbody\n")
    sha = _commit(repo)

    conn = build_connector(
        "git_markdown", {"root": str(repo), "corpus": "demo", "public": True}
    )
    upsert = conn._upsert_envelope(sha, "a.md")
    assert isinstance(upsert, ChangeEnvelope)
    assert upsert.connector == "git_markdown"
    assert upsert.source_version == sha  # revision
    assert upsert.source_acl is not None and upsert.source_acl.is_public  # access
    assert upsert.typed_payload is not None  # content-bearing payload
    assert upsert.operation == "upsert"

    delete = conn._delete_envelope(sha, "a.md")
    assert delete.operation == "delete"
    assert delete.source_object_id == upsert.source_object_id  # same revision record
    assert delete.typed_payload is None


@pytest.mark.concept("AU-ECO.connector.git-markdown-revision-connector")
def test_revision_record_id_is_opaque_but_document_node_id_may_not_be(tmp_path):
    # D-GM-3: envelope_ingest's opaque-identifier exemption only recognizes a
    # bare/namespaced 32-or-64-hex digest. `_revision_record_id` uses a full
    # 32-hex digest (opaque); `_document_node_id` mirrors engine.py's own
    # 24-hex truncation (NOT opaque) and exists only for cross-reference.
    repo = _init_repo(tmp_path)
    conn = build_connector("git_markdown", {"root": str(repo)})
    revision_id = conn._revision_record_id("a.md")
    assert revision_id.startswith("gitmd-revision:")
    digest = revision_id.split(":", 1)[1]
    assert len(digest) == 32 and all(c in "0123456789abcdef" for c in digest)

    document_id = conn._document_node_id("a.md")
    assert document_id.startswith("doc:git_markdown:")
    assert len(document_id.split(":")[-1]) == 24


@pytest.mark.concept("AU-ECO.connector.git-markdown-revision-connector")
def test_title_extraction_prefers_frontmatter_then_heading_then_filename(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "front.md").write_text("---\ntitle: From Frontmatter\n---\n# Heading\n")
    (repo / "heading-only.md").write_text("# Heading Title\nbody\n")
    (repo / "plain.md").write_text("just prose, no heading\n")
    sha = _commit(repo)

    conn = build_connector("git_markdown", {"root": str(repo)})
    docs = {d.metadata["relpath"]: d for d in conn.load()}
    assert docs["front.md"].title == "From Frontmatter"
    assert docs["heading-only.md"].title == "Heading Title"
    assert docs["plain.md"].title == "plain"
    del sha


@pytest.mark.concept("AU-KG.ontology.connector-manifest-gate")
def test_privacy_safe_text_redacts_before_persistence(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "a.md").write_text(
        "# Doc\nSee /home/alice/project for the config and contact a@b.com.\n"
    )
    _commit(repo)

    conn = build_connector("git_markdown", {"root": str(repo)})
    (doc,) = list(conn.load())
    assert "/home/alice" not in doc.text
    assert "a@b.com" not in doc.text
    assert "REDACTED" in doc.text


@pytest.mark.concept("AU-KG.ingest.stable-fragment-address")
def test_fragment_markdown_builds_stable_addressable_fragments():
    text = "# Top\nintro paragraph\n\n## Sub\nsub paragraph one\n\nsub paragraph two\n"
    fragments = fragment_markdown("artifact:test", text)
    kinds = [(f.kind, f.address) for f in fragments]
    assert ("heading", "heading:top") in kinds
    assert ("paragraph", "heading:top/paragraph:0") in kinds
    assert ("heading", "heading:top/heading:sub") in kinds
    assert ("paragraph", "heading:top/heading:sub/paragraph:0") in kinds
    assert ("paragraph", "heading:top/heading:sub/paragraph:1") in kinds
    # A typo fix in one paragraph changes only that fragment's content_hash;
    # every fragment_id (address) is unaffected.
    edited = text.replace("sub paragraph one", "sub paragraph ONE")
    edited_fragments = {
        f.fragment_id: f.content_hash
        for f in fragment_markdown("artifact:test", edited)
    }
    original_fragments = {f.fragment_id: f.content_hash for f in fragments}
    assert set(edited_fragments) == set(original_fragments)
    changed = [
        fid
        for fid in original_fragments
        if original_fragments[fid] != edited_fragments[fid]
    ]
    assert len(changed) == 1


@pytest.mark.concept("AU-KG.ingest.evidence-spine-artifact")
def test_build_artifact_keys_to_the_file_not_its_content(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "a.md").write_text("# A\nfirst version\n")
    sha1 = _commit(repo)
    conn = build_connector("git_markdown", {"root": str(repo), "corpus": "demo"})
    art1 = conn.build_artifact(sha1, "a.md")
    assert art1 is not None
    assert art1.fragments

    (repo / "a.md").write_text("# A\nsecond version\n")
    sha2 = _commit(repo)
    art2 = conn.build_artifact(sha2, "a.md")
    assert art2.artifact_id == art1.artifact_id  # same file -> same artifact
    assert (
        art2.content_hash != art1.content_hash
    )  # different revision -> different hash
