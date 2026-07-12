#!/usr/bin/python
"""Pure unit tests for ``epistemic_row.py`` (CONCEPT:AU-KB-CURRENCY).

Covers the pieces that don't need a live engine: :class:`EvidenceSpan`'s
wire-parsing (the typed evidence-span variant, item 3 of the
``epistemic-columns-currency.md`` follow-ups), :attr:`EpistemicRow.
typed_evidence_refs`, and :func:`attach_epistemic_rows` — the shared helper
every ``include_epistemic`` read surface (facade ``query``, ``query_unified``,
``uql``, ``store.execute``) adopts. The real-engine proof that these columns
originate server-side (not fabricated) lives in the ``tests/integration/``
suite; these tests exercise the AU-side parsing/plumbing in isolation with
synthetic wire-shaped dicts.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.epistemic_row import (
    CONTESTED_LABEL,
    NEUTRAL_CONFIDENCE,
    EpistemicRow,
    EvidenceSpan,
    attach_epistemic_columns,
    attach_epistemic_rows,
    epistemic_status,
    is_contested_row,
    row_ids_from_plain_rows,
    should_attach_epistemic_columns,
)


class TestEvidenceSpanFromWire:
    def test_parses_page_box_variant(self) -> None:
        raw = {
            "PageBox": {
                "document_id": "doc-1",
                "page": 3,
                "x": 10.0,
                "y": 20.0,
                "width": 100.0,
                "height": 50.0,
            }
        }
        span = EvidenceSpan.from_wire(raw)
        assert span is not None
        assert span.kind == "PageBox"
        assert span.document_id == "doc-1"
        assert span.page == 3
        assert span.x == 10.0
        assert span.y == 20.0
        assert span.width == 100.0
        assert span.height == 50.0
        # Unrelated-variant fields stay None.
        assert span.trace_id is None
        # The verbatim field map is preserved.
        assert span.raw == raw["PageBox"]

    def test_parses_trace_span_variant(self) -> None:
        raw = {"TraceSpan": {"trace_id": "t1", "span_id": "s1"}}
        span = EvidenceSpan.from_wire(raw)
        assert span is not None
        assert span.kind == "TraceSpan"
        assert span.trace_id == "t1"
        assert span.span_id == "s1"

    def test_parses_code_symbol_variant(self) -> None:
        raw = {
            "CodeSymbol": {
                "file_path": "src/lib.rs",
                "symbol": "foo",
                "start_line": 10,
                "end_line": 20,
            }
        }
        span = EvidenceSpan.from_wire(raw)
        assert span is not None
        assert span.kind == "CodeSymbol"
        assert span.file_path == "src/lib.rs"
        assert span.symbol == "foo"
        assert span.start_line == 10
        assert span.end_line == 20

    def test_preserves_unrecognized_field_in_raw(self) -> None:
        raw = {
            "DocumentSpan": {
                "document_id": "d1",
                "start": 0,
                "end": 10,
                "future_field": 1,
            }
        }
        span = EvidenceSpan.from_wire(raw)
        assert span is not None
        assert span.document_id == "d1"
        assert span.start == 0
        assert span.end == 10
        # A field this dataclass has no named attribute for is still in `raw`.
        assert span.raw["future_field"] == 1

    def test_returns_none_for_malformed_shapes(self) -> None:
        assert EvidenceSpan.from_wire(None) is None
        assert EvidenceSpan.from_wire("not-a-dict") is None
        assert EvidenceSpan.from_wire({}) is None
        assert EvidenceSpan.from_wire({"A": 1, "B": 2}) is None  # not single-key
        assert EvidenceSpan.from_wire({"PageBox": "not-a-dict"}) is None


class TestEpistemicRowTypedEvidenceRefs:
    def test_typed_view_parses_recognized_entries(self) -> None:
        row = EpistemicRow(
            id="n1",
            kind="Claim",
            score=None,
            confidence=0.9,
            evidence_refs=[
                {
                    "PageBox": {
                        "document_id": "d1",
                        "page": 1,
                        "x": 0.0,
                        "y": 0.0,
                        "width": 1.0,
                        "height": 1.0,
                    }
                },
                {"garbage": "entry", "extra": "key"},  # not single-key -> skipped
            ],
        )
        typed = row.typed_evidence_refs
        assert len(typed) == 1
        assert typed[0].kind == "PageBox"
        assert typed[0].document_id == "d1"

    def test_typed_view_empty_when_no_evidence(self) -> None:
        row = EpistemicRow(id="n1", kind="Claim", score=None, confidence=1.0)
        assert row.typed_evidence_refs == []

    def test_calibration_aliases_confidence(self) -> None:
        row = EpistemicRow(id="n1", kind="Claim", score=None, confidence=0.42)
        assert row.calibration == row.confidence == 0.42


class TestAttachEpistemicRows:
    def test_degrades_to_empty_when_fetch_is_none(self) -> None:
        rows = [{"id": "n1"}]
        assert attach_epistemic_rows(rows, None) == []

    def test_degrades_to_empty_when_no_resolvable_ids(self) -> None:
        def fetch(ids: list[str]) -> list[dict]:
            raise AssertionError("fetch should not be called with no ids")

        assert attach_epistemic_rows([{"no_id_here": 1}], fetch) == []

    def test_preserves_original_row_order_not_fetch_order(self) -> None:
        # Two plain [{"id","score"}] rows in rank order n2, n1 — the engine's
        # explain_provenance_by_ids response comes back in a DIFFERENT order
        # (n1, n2); the zipped result must follow the ORIGINAL rows' order.
        rows = [{"id": "n2", "score": 0.9}, {"id": "n1", "score": 0.5}]

        def fetch(ids: list[str]) -> list[dict]:
            assert set(ids) == {"n1", "n2"}
            return [
                {"id": "n1", "kind": "Claim", "confidence": 0.7},
                {"id": "n2", "kind": "Claim", "confidence": 0.8},
            ]

        result = attach_epistemic_rows(rows, fetch)
        assert [r.id for r in result] == ["n2", "n1"]
        assert result[0].confidence == 0.8
        assert result[1].confidence == 0.7

    def test_drops_row_the_engine_could_not_resolve(self) -> None:
        rows = [{"id": "n1"}, {"id": "missing"}]

        def fetch(ids: list[str]) -> list[dict]:
            return [{"id": "n1", "kind": "Claim", "confidence": 1.0}]

        result = attach_epistemic_rows(rows, fetch)
        assert [r.id for r in result] == ["n1"]

    def test_zips_plain_row_properties_through(self) -> None:
        rows = [{"n": {"id": "n1", "name": "hello"}}]

        def fetch(ids: list[str]) -> list[dict]:
            return [{"id": "n1", "kind": "Claim", "confidence": 1.0}]

        result = attach_epistemic_rows(rows, fetch)
        assert result[0].properties == {"id": "n1", "name": "hello"}


def test_row_ids_from_plain_rows_still_used_by_attach_epistemic_rows() -> None:
    """Sanity: :func:`attach_epistemic_rows` builds on the SAME id-extraction
    primitive :meth:`KnowledgeGraph._attach_epistemic` always used — no second,
    divergent id-extraction path was introduced for the new read surfaces."""
    rows = [{"id": "n1"}, {"n": {"id": "n2"}}]
    assert {ip["id"] for ip in row_ids_from_plain_rows(rows)} == {"n1", "n2"}


class TestEpistemicStatus:
    def test_unresolved_when_not_resolved(self) -> None:
        assert (
            epistemic_status(resolved=False, confidence=0.99, policy_labels=[])
            == "unresolved"
        )

    def test_contested_from_label(self) -> None:
        assert (
            epistemic_status(
                resolved=True, confidence=0.99, policy_labels=[CONTESTED_LABEL]
            )
            == "contested"
        )

    def test_contested_from_contradiction_count(self) -> None:
        assert (
            epistemic_status(
                resolved=True,
                confidence=0.99,
                policy_labels=[],
                contradiction_count=1,
            )
            == "contested"
        )

    def test_low_confidence_below_threshold(self) -> None:
        assert (
            epistemic_status(resolved=True, confidence=0.1, policy_labels=[])
            == "low_confidence"
        )

    def test_confirmed_when_resolved_and_confident(self) -> None:
        assert (
            epistemic_status(resolved=True, confidence=0.95, policy_labels=[])
            == "confirmed"
        )


class TestIsContestedRow:
    def test_true_for_contested_label(self) -> None:
        assert is_contested_row({"policy_labels": [CONTESTED_LABEL]}) is True

    def test_true_for_low_confidence_property(self) -> None:
        assert is_contested_row({"confidence": 0.1}) is True

    def test_true_for_nested_node_dict(self) -> None:
        assert is_contested_row({"n": {"confidence": 0.1}}) is True

    def test_false_for_ordinary_row(self) -> None:
        assert is_contested_row({"id": "n1", "name": "hello"}) is False


class TestShouldAttachEpistemicColumns:
    def test_always_true_when_default_is_true(self) -> None:
        assert should_attach_epistemic_columns([{"id": "n1"}], default=True) is True

    def test_false_when_default_off_and_nothing_contested(self) -> None:
        rows = [{"id": "n1", "name": "hello"}]
        assert should_attach_epistemic_columns(rows, default=False) is False

    def test_auto_on_when_default_off_but_row_contested(self) -> None:
        rows = [{"id": "n1", "policy_labels": [CONTESTED_LABEL]}]
        assert should_attach_epistemic_columns(rows, default=False) is True


class TestAttachEpistemicColumns:
    def test_never_changes_return_type_or_row_count(self) -> None:
        rows = [{"id": "n1"}, {"id": "n2"}]
        result = attach_epistemic_columns(rows, None)
        assert result is rows  # same list object, in place
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_degrades_to_neutral_prior_when_fetch_is_none(self) -> None:
        rows = [{"id": "n1", "name": "hello"}]
        result = attach_epistemic_columns(rows, None)
        assert result[0]["name"] == "hello"  # original data untouched
        assert result[0]["confidence"] == NEUTRAL_CONFIDENCE
        assert result[0]["source_refs"] == []
        assert result[0]["evidence_refs"] == []
        assert result[0]["policy_labels"] == []
        assert result[0]["provenance"] == {
            "resolved": False,
            "valid_time": None,
            "tx_time": None,
        }

    def test_degrades_to_neutral_prior_when_fetch_raises(self) -> None:
        def fetch(ids: list[str]) -> list[dict]:
            raise RuntimeError("engine unreachable")

        rows = [{"id": "n1"}]
        result = attach_epistemic_columns(rows, fetch)
        assert result[0]["confidence"] == NEUTRAL_CONFIDENCE
        assert result[0]["provenance"]["resolved"] is False

    def test_merges_resolved_envelope_onto_matching_row(self) -> None:
        rows = [{"id": "n1", "name": "hello"}]

        def fetch(ids: list[str]) -> list[dict]:
            assert ids == ["n1"]
            return [
                {
                    "id": "n1",
                    "confidence": 0.42,
                    "source_refs": ["doc:1"],
                    "evidence_spans": [{"DocumentSpan": {"document_id": "d1"}}],
                    "policy_labels": [CONTESTED_LABEL],
                    "valid_time": [1, 2],
                    "tx_time": [3, 4],
                }
            ]

        result = attach_epistemic_columns(rows, fetch)
        row = result[0]
        assert row["name"] == "hello"
        assert row["confidence"] == 0.42
        assert row["source_refs"] == ["doc:1"]
        assert row["evidence_refs"] == [{"DocumentSpan": {"document_id": "d1"}}]
        assert row["policy_labels"] == [CONTESTED_LABEL]
        assert row["provenance"] == {
            "resolved": True,
            "valid_time": [1, 2],
            "tx_time": [3, 4],
        }

    def test_never_clobbers_a_property_the_row_already_carries(self) -> None:
        # A caller-selected `RETURN n.confidence AS confidence` column is a
        # real property, not the injected epistemic default — must survive.
        rows = [{"id": "n1", "confidence": 0.99}]

        def fetch(ids: list[str]) -> list[dict]:
            return [{"id": "n1", "confidence": 0.1}]

        result = attach_epistemic_columns(rows, fetch)
        assert result[0]["confidence"] == 0.99

    def test_empty_rows_is_a_pure_noop(self) -> None:
        assert attach_epistemic_columns([], None) == []
