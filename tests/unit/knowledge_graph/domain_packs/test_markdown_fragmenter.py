"""Reference markdown fragmenter: a real file's frontmatter/table/heading/link ->
Fragment objects (CONCEPT:AU-KG.ingest.mapping-dsl)."""

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
        f.metadata["key"]: f for f in fragments if f.fragment_type == "frontmatter_key"
    }
    assert frontmatter["status"].value == "active"
    assert frontmatter["owner"].value == "alice"
    assert all(f.artifact_id == artifact.artifact_id for f in frontmatter.values())


def test_table_rows_become_table_row_fragments_scoped_to_their_heading():
    _artifact, fragments = fragment_markdown_text(RUNBOOK_MD, source_path="runbook.md")

    rows = [f for f in fragments if f.fragment_type == "table_row"]
    assert len(rows) == 2
    assert rows[0].value == {"step": "Provision VM", "assignee": "bob"}
    assert rows[1].value == {"step": "Configure network", "assignee": "carol"}
    assert all(r.metadata["heading_path"] == "Steps" for r in rows)
    assert [r.metadata["row_index"] for r in rows] == [0, 1]


def test_heading_becomes_heading_section_fragment():
    _artifact, fragments = fragment_markdown_text(RUNBOOK_MD, source_path="runbook.md")

    headings = [f for f in fragments if f.fragment_type == "heading_section"]
    assert len(headings) == 1
    assert headings[0].metadata == {
        "heading": "Steps",
        "heading_path": "Steps",
        "level": 1,
    }


def test_link_becomes_link_fragment():
    _artifact, fragments = fragment_markdown_text(RUNBOOK_MD, source_path="runbook.md")

    links = [f for f in fragments if f.fragment_type == "link"]
    assert len(links) == 1
    assert links[0].value == {"text": "reference", "href": "other.md"}


def test_artifact_id_is_deterministic_from_content():
    artifact_a, _ = fragment_markdown_text(RUNBOOK_MD, source_path="runbook.md")
    artifact_b, _ = fragment_markdown_text(
        RUNBOOK_MD, source_path="a-different-path.md"
    )

    # Same bytes -> same content hash -> same derived artifact_id, regardless
    # of source_path (content-addressed identity).
    assert artifact_a.artifact_id == artifact_b.artifact_id
    assert artifact_a.content_hash == artifact_b.content_hash


def test_fragment_markdown_file_reads_from_disk(tmp_path):
    path = tmp_path / "runbook.md"
    path.write_text(RUNBOOK_MD, encoding="utf-8")

    artifact, fragments = fragment_markdown_file(path)

    assert artifact.source_path == str(path)
    assert any(f.fragment_type == "table_row" for f in fragments)
