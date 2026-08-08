"""D-ES-3: producers for the evidence spine beyond markdown.

``fragment_markdown`` was the only producer even though ``Fragment``/
``Artifact`` already carry the ``page``/``record``/``field``/``table_cell``
kinds and the engine's ``ArtifactLocus`` vocabulary for the rest — the gap was
producers, not contract. These tests exercise the three new ones:
``fragment_pdf`` (page + block), ``fragment_record`` (one JSON object, keyed
by JSON-pointer-shaped path), and ``fragment_rowset`` (DB rows, keyed by
primary key).
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.ingestion.evidence_spine import (
    Fragment,
    artifact_id_for,
    fragment_id_for,
    fragment_pdf,
    fragment_record,
    fragment_rowset,
)

ARTIFACT = artifact_id_for("filesystem", "reports-bucket", "reports/q3.pdf")


# ── fragment_pdf ──────────────────────────────────────────────────────────────


def test_fragment_pdf_emits_one_page_fragment_per_page() -> None:
    pages = ["Page one text.", "Page two text.", "Page three text."]
    fragments = fragment_pdf(pages, artifact_id=ARTIFACT)

    page_fragments = [f for f in fragments if f.kind == "page"]
    assert len(page_fragments) == 3
    assert [f.attributes["page_number"] for f in page_fragments] == [1, 2, 3]
    assert all(f.locus_kind == "page_box" for f in page_fragments)
    # Sequence is a total document order across pages, not reset per page.
    assert [f.sequence for f in fragments] == list(range(len(fragments)))


def test_fragment_pdf_splits_a_page_into_paragraph_blocks() -> None:
    pages = ["First block.\n\nSecond block.\n\nThird block."]
    fragments = fragment_pdf(pages, artifact_id=ARTIFACT)

    page = next(f for f in fragments if f.kind == "page")
    blocks = [f for f in fragments if f.kind == "paragraph"]
    assert [b.text for b in blocks] == [
        "First block.",
        "Second block.",
        "Third block.",
    ]
    assert all(b.parent_fragment_id == page.fragment_id for b in blocks)


def test_fragment_pdf_reingesting_unchanged_pages_yields_identical_ids() -> None:
    pages = ["Alpha.\n\nBeta.", "Gamma."]
    first = fragment_pdf(pages, artifact_id=ARTIFACT)
    second = fragment_pdf(pages, artifact_id=ARTIFACT)

    assert [f.fragment_id for f in first] == [f.fragment_id for f in second]


def test_fragment_pdf_empty_page_contributes_no_block_fragments() -> None:
    fragments = fragment_pdf(["", "   \n\n  ", "real text"], artifact_id=ARTIFACT)

    pages = [f for f in fragments if f.kind == "page"]
    blocks = [f for f in fragments if f.kind == "paragraph"]
    assert len(pages) == 3
    assert len(blocks) == 1
    assert blocks[0].text == "real text"


def test_fragment_pdf_every_fragment_belongs_to_the_artifact() -> None:
    fragments = fragment_pdf(["one", "two"], artifact_id=ARTIFACT)
    assert all(f.artifact_id == ARTIFACT for f in fragments)
    assert all(isinstance(f, Fragment) for f in fragments)


# ── fragment_record ───────────────────────────────────────────────────────────


def test_fragment_record_emits_a_field_per_scalar_key() -> None:
    record = {"id": "acct_123", "currency": "USD", "amount": 4200}
    fragments = fragment_record(record, artifact_id=ARTIFACT)

    root = fragments[0]
    assert root.kind == "record"
    fields = {f.label: f.text for f in fragments if f.kind == "field"}
    assert fields == {"id": "acct_123", "currency": "USD", "amount": "4200"}
    assert all(f.parent_fragment_id == root.fragment_id for f in fragments[1:])


def test_fragment_record_recurses_into_nested_objects_and_arrays() -> None:
    record = {
        "customer": {"name": "Ada", "vip": True},
        "line_items": ["widget", "gadget"],
    }
    fragments = fragment_record(record, artifact_id=ARTIFACT)
    by_kind = {}
    for f in fragments:
        by_kind.setdefault(f.kind, []).append(f)

    assert len(by_kind["record"]) == 2  # the root + the nested "customer" object
    assert len(by_kind["list"]) == 1  # "line_items"
    assert {f.text for f in by_kind["field"]} == {"Ada", "True"}
    assert {f.text for f in by_kind["list_item"]} == {"widget", "gadget"}

    customer = next(f for f in by_kind["record"] if f.label == "customer")
    name_field = next(f for f in by_kind["field"] if f.text == "Ada")
    assert name_field.parent_fragment_id == customer.fragment_id


def test_fragment_record_renaming_a_sibling_key_does_not_move_another_fields_address() -> (
    None
):
    before = fragment_record({"a": "1", "b": "2"}, artifact_id=ARTIFACT)
    after = fragment_record({"a": "1", "renamed_b": "2"}, artifact_id=ARTIFACT)

    a_before = next(f for f in before if f.label == "a")
    a_after = next(f for f in after if f.label == "a")
    assert a_before.fragment_id == a_after.fragment_id


def test_fragment_record_reingesting_unchanged_record_yields_identical_ids() -> None:
    record = {"a": 1, "b": {"c": 2}}
    first = fragment_record(record, artifact_id=ARTIFACT)
    second = fragment_record(record, artifact_id=ARTIFACT)
    assert [f.fragment_id for f in first] == [f.fragment_id for f in second]


# ── fragment_rowset ───────────────────────────────────────────────────────────


def test_fragment_rowset_keys_each_row_by_its_primary_key() -> None:
    rows = [
        {"id": "row-1", "status": "open"},
        {"id": "row-2", "status": "closed"},
    ]
    fragments = fragment_rowset(rows, artifact_id=ARTIFACT, key_field="id")

    records = [f for f in fragments if f.kind == "record"]
    assert {r.label for r in records} == {"row-1", "row-2"}
    # Addresses are content-of-the-key-derived, not positional.
    assert all(r.fragment_id == fragment_id_for(ARTIFACT, r.path) for r in records)


def test_fragment_rowset_survives_a_resort() -> None:
    rows = [{"id": "row-1"}, {"id": "row-2"}, {"id": "row-3"}]
    original = fragment_rowset(rows, artifact_id=ARTIFACT)
    resorted = fragment_rowset(list(reversed(rows)), artifact_id=ARTIFACT)

    ids_before = {f.label: f.fragment_id for f in original if f.kind == "record"}
    ids_after = {f.label: f.fragment_id for f in resorted if f.kind == "record"}
    assert ids_before == ids_after


def test_fragment_rowset_supports_a_composite_key() -> None:
    rows = [{"tenant": "t1", "id": "1", "name": "a"}]
    fragments = fragment_rowset(rows, artifact_id=ARTIFACT, key_field=("tenant", "id"))
    record = next(f for f in fragments if f.kind == "record")
    assert record.label == "t1/1"


def test_fragment_rowset_missing_key_falls_back_to_a_unique_ordinal() -> None:
    rows = [{"name": "no pk here"}, {"name": "also no pk"}]
    fragments = fragment_rowset(rows, artifact_id=ARTIFACT, key_field="id")

    records = [f for f in fragments if f.kind == "record"]
    assert len({r.fragment_id for r in records}) == 2, (
        "two rows missing their key field must still get distinct addresses"
    )
