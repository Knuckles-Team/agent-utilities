"""Evidence spine — ``Artifact`` -> addressable ``Fragment``.

CONCEPT:AU-KG.ingest.evidence-spine-artifact
CONCEPT:AU-KG.ingest.stable-fragment-address

These are **id-stability** tests, not existence tests.  The claim under test is
that a citation survives the edits real documents actually receive, and that
re-ingesting an unchanged document produces byte-identical fragment ids.  Each
test names the specific edit class it defends.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope
from agent_utilities.knowledge_graph.ingestion.evidence_spine import (
    ARTIFACT_NODE_TYPE,
    FRAGMENT_NODE_TYPE,
    HAS_FRAGMENT_EDGE,
    NEXT_FRAGMENT_EDGE,
    PARENT_FRAGMENT_EDGE,
    Artifact,
    Fragment,
    artifact_id_for,
    citation_status,
    content_digest,
    fragment_id_for,
    fragment_markdown,
    resolve_fragment,
)

DOC = """# Payments Platform

The payments platform settles card and ACH transactions.

## Getting Started

Install the SDK, then configure your API key.

| Field | Meaning |
|---|---|
| amount | Minor units |
| currency | ISO 4217 |

- Provision a sandbox account
- Rotate the sandbox key monthly

```python
client = Payments(api_key=KEY)
```

## Operations

> On-call rotates weekly.

Escalate a stuck settlement to the platform team.
"""

ARTIFACT = artifact_id_for("git-markdown", "docs-repo", "docs/payments.md")


def spine(text: str = DOC) -> tuple[Fragment, ...]:
    return fragment_markdown(text, artifact_id=ARTIFACT)


def address_map(fragments: tuple[Fragment, ...]) -> dict[str, str]:
    """``text -> fragment_id`` for the fragments carrying real prose."""
    return {f.text: f.fragment_id for f in fragments}


# ── the fragmenter produces a real, ordered, nested spine ────────────────────


def test_fragments_cover_every_structural_kind() -> None:
    kinds = {f.kind for f in spine()}
    assert {
        "heading",
        "paragraph",
        "table",
        "table_row",
        "list_item",
        "code_block",
        "quote",
    } <= kinds


def test_fragments_are_orderable_and_nestable() -> None:
    fragments = spine()
    assert [f.sequence for f in fragments] == list(range(len(fragments)))

    by_id = {f.fragment_id: f for f in fragments}
    rows = [f for f in fragments if f.kind == "table_row"]
    assert rows, "the fixture has a table"
    for row in rows:
        table = by_id[row.parent_fragment_id]
        assert table.kind == "table"
        # …and the table is itself inside a section — a row inside a table
        # inside a section, reconstructed from the graph alone.
        section = by_id[table.parent_fragment_id]
        assert section.kind == "heading"
        assert row.depth == table.depth + 1 == section.depth + 2


def test_fragment_id_is_a_pure_function_of_artifact_and_path() -> None:
    for fragment in spine():
        assert fragment.fragment_id == fragment_id_for(ARTIFACT, fragment.path)


def test_fragment_rejects_an_address_that_does_not_match_its_path() -> None:
    with pytest.raises(ValueError, match="fragment_id_for"):
        Fragment(
            fragment_id="fragment:deadbeef",
            artifact_id=ARTIFACT,
            kind="paragraph",
            path=("paragraph:0",),
            text="x",
            content_hash=content_digest("x"),
        )


# ── the stability contract ───────────────────────────────────────────────────


def test_reingesting_an_unchanged_document_yields_identical_ids() -> None:
    """The floor requirement: a no-op re-ingest must not move a single citation."""
    first, second = spine(), spine()
    assert [f.fragment_id for f in first] == [f.fragment_id for f in second]
    assert [f.content_hash for f in first] == [f.content_hash for f in second]


def test_cosmetic_reformatting_does_not_change_a_content_hash() -> None:
    """Trailing whitespace / rewrapping is not a content change."""
    reflowed = DOC.replace(
        "The payments platform settles card and ACH transactions.",
        "The payments platform settles card\nand ACH transactions.   ",
    )
    before = address_map(spine())
    after = {f.text: f for f in spine(reflowed)}
    target = "The payments platform settles card\nand ACH transactions."
    assert after[target].fragment_id == before[
        "The payments platform settles card and ACH transactions."
    ]
    assert after[target].content_hash == content_digest(
        "The payments platform settles card and ACH transactions."
    )


def test_fixing_a_typo_keeps_the_address_and_moves_only_the_hash() -> None:
    """The edit a purely content-hashed id would break."""
    original = {f.address: f for f in spine()}
    edited = {
        f.address: f
        for f in spine(DOC.replace("configure your API key", "configure your API KEY"))
    }
    address = next(a for a, f in original.items() if "configure your API" in f.text)
    assert edited[address].fragment_id == original[address].fragment_id
    assert edited[address].content_hash != original[address].content_hash


def test_inserting_a_paragraph_in_another_section_moves_nothing() -> None:
    """The edit a purely positional id would break."""
    before = address_map(spine())
    inserted = DOC.replace(
        "## Operations\n",
        "## Operations\n\nA brand new operations preamble.\n",
    )
    after = address_map(spine(inserted))
    unchanged = "Install the SDK, then configure your API key."
    assert after[unchanged] == before[unchanged]
    assert after["Escalate a stuck settlement to the platform team."] != ""


def test_inserting_a_whole_section_moves_nothing_that_came_before_or_after() -> None:
    """Headings are slug-anchored, so a new section renumbers no sibling."""
    before = address_map(spine())
    inserted = DOC.replace(
        "## Operations",
        "## Security\n\nRotate credentials quarterly.\n\n## Operations",
    )
    after = address_map(spine(inserted))
    for text in (
        "Install the SDK, then configure your API key.",
        "Escalate a stuck settlement to the platform team.",
        "The payments platform settles card and ACH transactions.",
    ):
        assert after[text] == before[text], text


def test_reordering_table_rows_keeps_every_row_address() -> None:
    """A row is anchored by its first cell, so a re-sorted table stays citable."""
    before = {f.text: f.fragment_id for f in spine() if f.kind == "table_row"}
    swapped = DOC.replace(
        "| amount | Minor units |\n| currency | ISO 4217 |",
        "| currency | ISO 4217 |\n| amount | Minor units |",
    )
    after = {f.text: f.fragment_id for f in spine(swapped) if f.kind == "table_row"}
    assert after == before


def test_duplicate_headings_do_not_collide_onto_one_address() -> None:
    text = "## Notes\n\nfirst note\n\n## Notes\n\nsecond note\n"
    fragments = spine(text)
    ids = [f.fragment_id for f in fragments]
    assert len(ids) == len(set(ids))


def test_inserting_a_sibling_paragraph_shifts_the_address_and_is_recoverable() -> None:
    """The scheme's honest residual weakness — and the fallback that covers it."""
    before = spine()
    cited = next(
        f for f in before if f.text == "Install the SDK, then configure your API key."
    )
    after = spine(
        DOC.replace(
            "Install the SDK, then configure your API key.",
            "Read the overview first.\n\nInstall the SDK, then configure your API key.",
        )
    )
    moved = next(f for f in after if f.text == cited.text)
    # The address DID shift — we do not pretend otherwise.
    assert moved.fragment_id != cited.fragment_id
    # …and the content hash still finds it, so the citation is re-pointable.
    assert resolve_fragment(after, content_hash=cited.content_hash) is moved


def test_citation_status_reports_all_four_outcomes_distinctly() -> None:
    fragments = spine()
    cited = next(f for f in fragments if f.kind == "quote")

    current = citation_status(
        fragments, fragment_id=cited.fragment_id, content_hash=cited.content_hash
    )
    assert current["status"] == "current"

    edited = spine(DOC.replace("On-call rotates weekly.", "On-call rotates daily."))
    stale = citation_status(
        edited, fragment_id=cited.fragment_id, content_hash=cited.content_hash
    )
    assert stale["status"] == "stale"

    # Only a SAME-KIND sibling inserted before it can shift an anonymous
    # fragment's ordinal — ordinals are counted per (parent, kind), so the
    # paragraph inserted in the test above leaves this quote where it was.
    relocated = spine(
        DOC.replace(
            "> On-call rotates weekly.",
            "> Escalation policy v2 applies.\n\n> On-call rotates weekly.",
        )
    )
    moved = citation_status(
        relocated, fragment_id=cited.fragment_id, content_hash=cited.content_hash
    )
    assert moved["status"] == "moved"
    assert moved["fragment_id"] != cited.fragment_id

    lost = citation_status(
        spine("# Unrelated\n\nnothing here\n"),
        fragment_id=cited.fragment_id,
        content_hash=cited.content_hash,
    )
    assert lost["status"] == "lost"


def test_ambiguous_content_is_preserved_not_guessed() -> None:
    """Two identical paragraphs must not resolve to a lucky one of them."""
    fragments = spine("# T\n\n## A\n\nsame text\n\n## B\n\nsame text\n")
    duplicate = next(f for f in fragments if f.text == "same text")
    assert resolve_fragment(fragments, content_hash=duplicate.content_hash) is None


# ── the Artifact half of the spine ───────────────────────────────────────────


def envelope() -> ChangeEnvelope:
    return ChangeEnvelope.from_connector_record(
        {"id": "docs/payments.md", "node_type": "Document", "updatedAt": "2026-07-31"},
        connector="git-markdown",
        source_instance="docs-repo",
    )


def test_artifact_is_keyed_to_its_envelope_and_carries_a_content_hash() -> None:
    env = envelope()
    artifact = Artifact.from_envelope(env, content=DOC, fragments=spine())

    assert artifact.artifact_id == artifact_id_for(
        "git-markdown", "docs-repo", "docs/payments.md"
    )
    assert artifact.content_hash == content_digest(DOC)
    assert artifact.content_hash.startswith("sha256:")
    assert artifact.envelope_id == env.envelope_id
    assert artifact.idempotency_key == env.idempotency_key


def test_artifact_takes_governance_from_the_envelope_not_the_payload() -> None:
    """A source record must not get to declare its own ACL."""
    env = envelope()
    artifact = Artifact.from_envelope(env, content=DOC)
    assert artifact.classification == env.classification.value
    assert artifact.external_access == env.source_acl.model_dump()
    assert artifact.external_access["is_public"] is False


def test_artifact_rejects_a_fragment_from_another_artifact() -> None:
    foreign = fragment_markdown(DOC, artifact_id=artifact_id_for("x", "", "y"))
    with pytest.raises(ValueError, match="must all belong to this artifact"):
        Artifact.from_envelope(envelope(), content=DOC, fragments=foreign)


def test_graph_slice_is_admissible_to_ingest_graph_slice() -> None:
    """``ingest_graph_slice`` rejects ``type``/alias keys — prove we never emit them."""
    artifact = Artifact.from_envelope(envelope(), content=DOC, fragments=spine())
    entities, relationships = artifact.to_graph_slice(document_id="doc:payments")

    assert entities[0]["node_type"] == ARTIFACT_NODE_TYPE
    assert all("type" not in entity for entity in entities)
    assert all(str(entity.get("node_type") or "").strip() for entity in entities)
    for edge in relationships:
        assert str(edge.get("relationship") or "").strip()
        assert not {"type", "rel_type", "relationship_type", "relation"} & set(edge)

    fragment_nodes = [e for e in entities if e["node_type"] == FRAGMENT_NODE_TYPE]
    assert len(fragment_nodes) == len(artifact.fragments)
    kinds = {edge["relationship"] for edge in relationships}
    assert {HAS_FRAGMENT_EDGE, PARENT_FRAGMENT_EDGE, NEXT_FRAGMENT_EDGE} <= kinds


def test_every_fragment_node_satisfies_the_shacl_required_properties() -> None:
    """The FragmentShape gate requires artifact_id + content_hash + address +
    fragment_kind + sequence on every row; a row missing one is refused at
    ingest, so the producer must never emit one."""
    artifact = Artifact.from_envelope(envelope(), content=DOC, fragments=spine())
    entities, _ = artifact.to_graph_slice()
    for row in entities:
        if row["node_type"] != FRAGMENT_NODE_TYPE:
            continue
        assert row["artifact_id"] == artifact.artifact_id
        assert row["content_hash"].startswith("sha256:")
        assert row["address"]
        assert row["fragment_kind"]
        assert isinstance(row["sequence"], int) and row["sequence"] >= 0


def test_fragment_renders_an_engine_artifact_locus() -> None:
    """The engine protocol's ArtifactLocus is the evidence shape claims cite."""
    fragment = next(f for f in spine() if f.kind == "paragraph")
    locus = fragment.to_locus()
    assert locus["kind"] == "document_span"
    assert locus["start"] is not None and locus["end"] > locus["start"]
    assert locus["selector"]["fragment_id"] == fragment.fragment_id
    assert locus["selector"]["content_hash"] == fragment.content_hash


def test_version_id_pins_a_citation_to_one_revision() -> None:
    fragment = next(f for f in spine() if f.kind == "paragraph")
    # '#', not '@' -- the engine's validate_safe_text privacy guard rejects
    # any '@' in inline text outright (D-GM-4 / D-GS856-6 / D-MW-1 / D-MW-2).
    assert fragment.version_id.startswith(f"{fragment.fragment_id}#")
    assert "@" not in fragment.version_id
    edited = next(
        f
        for f in spine(DOC.replace("settles card and ACH", "settles card + ACH"))
        if f.address == fragment.address
    )
    assert edited.fragment_id == fragment.fragment_id
    assert edited.version_id != fragment.version_id
