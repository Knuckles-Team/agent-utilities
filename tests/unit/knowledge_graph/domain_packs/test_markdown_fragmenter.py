"""Reference markdown fragmenter: a real file's frontmatter/table/heading/link ->
real evidence-spine Fragment objects (CONCEPT:AU-KG.ingest.mapping-dsl,
CONCEPT:AU-KG.ingest.stable-fragment-address, D-GP2-2)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.domain_packs.markdown_fragmenter import (
    fragment_markdown_file,
    fragment_markdown_text,
)

RUNBOOK_MD = """---
status: active
owner: alice
---

# Steps

| step | assignee |
| --- | --- |
| Provision VM | bob |
| Configure network | carol |

See also [reference](other.md) for background.
"""


def test_frontmatter_keys_become_frontmatter_fragments():
    artifact, fragments = fragment_markdown_text(RUNBOOK_MD, source_path="runbook.md")

    frontmatter = {
        f.attributes["key"]: f for f in fragments if f.kind == "frontmatter_key"
    }
    assert frontmatter["status"].attributes["value"] == "active"
    assert frontmatter["owner"].attributes["value"] == "alice"
    assert all(f.artifact_id == artifact.artifact_id for f in frontmatter.values())


def test_table_rows_become_table_row_fragments_scoped_to_their_heading():
    _artifact, fragments = fragment_markdown_text(RUNBOOK_MD, source_path="runbook.md")

    rows = [f for f in fragments if f.kind == "table_row"]
    assert len(rows) == 2
    assert rows[0].attributes["row"] == {"step": "Provision VM", "assignee": "bob"}
    assert rows[1].attributes["row"] == {
        "step": "Configure network",
        "assignee": "carol",
    }
    assert all(r.attributes["heading_path"] == "Steps" for r in rows)
    assert [r.attributes["row_index"] for r in rows] == [0, 1]


def test_heading_becomes_heading_fragment():
    _artifact, fragments = fragment_markdown_text(RUNBOOK_MD, source_path="runbook.md")

    headings = [f for f in fragments if f.kind == "heading"]
    assert len(headings) == 1
    assert headings[0].attributes == {
        "heading": "Steps",
        "heading_path": "Steps",
        "level": 1,
    }


def test_link_becomes_link_fragment():
    _artifact, fragments = fragment_markdown_text(RUNBOOK_MD, source_path="runbook.md")

    links = [f for f in fragments if f.kind == "link"]
    assert len(links) == 1
    assert links[0].attributes == {"text": "reference", "href": "other.md"}


def test_artifact_id_is_deterministic_from_source_path():
    """The evidence spine's own identity scheme (CONCEPT:AU-KG.ingest.
    evidence-spine-artifact): an artifact is keyed to SOURCE IDENTITY, not
    content — so re-fragmenting the SAME path after an edit updates the SAME
    artifact (and every existing citation into it survives) rather than
    forking a new one on every edit."""
    artifact_a, _ = fragment_markdown_text(RUNBOOK_MD, source_path="runbook.md")
    artifact_b, _ = fragment_markdown_text(
        RUNBOOK_MD.replace("active", "deprecated"), source_path="runbook.md"
    )

    assert artifact_a.artifact_id == artifact_b.artifact_id
    assert artifact_a.content_hash != artifact_b.content_hash

    # A DIFFERENT path is a DIFFERENT artifact, even with byte-identical content.
    artifact_c, _ = fragment_markdown_text(
        RUNBOOK_MD, source_path="a-different-path.md"
    )
    assert artifact_c.artifact_id != artifact_a.artifact_id
    assert artifact_c.content_hash == artifact_a.content_hash


def test_fragment_markdown_file_reads_from_disk(tmp_path):
    path = tmp_path / "runbook.md"
    path.write_text(RUNBOOK_MD, encoding="utf-8")

    artifact, fragments = fragment_markdown_file(path)

    assert artifact.source_object_id == str(path)
    assert any(f.kind == "table_row" for f in fragments)
