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
    epistemic_row_from_stream_row,
    epistemic_status,
    is_contested_row,
    row_ids_from_plain_rows,
    should_attach_epistemic_columns,
    stream_epistemic_rows_by_label,
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


class TestEpistemicRowFromWireContradictionAndProofIds:
    """SURPASS gap-closure: `proof_ids`/`contradiction_ids` exist on the Arrow
    `KnowledgeBatch` server-side but weren't threaded into `EpistemicRow` -- these
    prove `from_wire` now picks up both columns from an `EvidenceClaim` projection,
    exactly like every other column."""

    def test_from_wire_populates_both_new_columns(self) -> None:
        row = EpistemicRow.from_wire(
            {
                "id": "claim:1",
                "kind": "Claim",
                "score": 0.5,
                "confidence": 0.8,
                "contradiction_ids": ["claim:2"],
                "proof_ids": ["evidence:1", "claim:base"],
            }
        )
        assert row.contradiction_ids == ["claim:2"]
        assert row.proof_ids == ["evidence:1", "claim:base"]

    def test_from_wire_defaults_both_to_empty_when_absent(self) -> None:
        # An older/non-epistemic engine build's wire dict simply lacks these keys
        # -- from_wire must default cleanly, never raise a KeyError.
        row = EpistemicRow.from_wire({"id": "n1", "kind": "Claim", "confidence": 1.0})
        assert row.contradiction_ids == []
        assert row.proof_ids == []


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
        assert result[0]["contradiction_ids"] == []
        assert result[0]["proof_ids"] == []
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
                    "contradiction_ids": ["claim:rival"],
                    "proof_ids": ["evidence:e1", "claim:base"],
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
        # SURPASS gap-closure: contradiction_ids/proof_ids now thread through the
        # SAME merge path as every other epistemic column above.
        assert row["contradiction_ids"] == ["claim:rival"]
        assert row["proof_ids"] == ["evidence:e1", "claim:base"]
        assert row["provenance"] == {
            "resolved": True,
            "valid_time": [1, 2],
            "tx_time": [3, 4],
        }

    def test_contradiction_count_uses_real_ids_not_just_the_label_proxy(
        self, monkeypatch
    ) -> None:
        """SURPASS gap-closure: before `contradiction_ids` reached this far, the
        OTel `contradiction_count` annotation was a boolean proxy (1 iff
        `CONTESTED_LABEL` was present, 0 otherwise). Now that the real column is
        threaded through, a row with a NON-EMPTY `contradiction_ids` list but NO
        `CONTESTED_LABEL` must still report its real count."""
        captured: dict[str, object] = {}

        class _FakeTelemetry:
            def annotate_epistemic(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(
            "agent_utilities.observability.get_telemetry_engine",
            lambda: _FakeTelemetry(),
        )

        rows = [{"id": "n1"}]

        def fetch(ids: list[str]) -> list[dict]:
            return [
                {
                    "id": "n1",
                    "confidence": 0.9,
                    "policy_labels": [],  # NOT contested by label
                    "contradiction_ids": ["claim:a", "claim:b"],
                }
            ]

        attach_epistemic_columns(rows, fetch)
        assert captured["contradiction_count"] == 2
        assert captured["status"] == "contested"

    def test_reaches_a_real_exported_span_with_the_correct_epistemic_status(
        self, monkeypatch
    ) -> None:
        """X2 live-path proof: unlike the fake-telemetry test above (which
        only pins the CALL ARGS), this drives the REAL ``TelemetryEngine.
        annotate_epistemic`` through an in-memory exporter — catching the
        real bug this closed: ``status`` was validated against the
        run-status vocabulary, silently collapsing every real "contested"/
        "confirmed"/"low_confidence" epistemic status to "unknown" on export."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test-epistemic-row")

        rows = [{"id": "n1"}]

        def fetch(ids: list[str]) -> list[dict]:
            return [
                {
                    "id": "n1",
                    "confidence": 0.9,
                    "policy_labels": [],
                    "contradiction_ids": ["claim:a", "claim:b"],
                }
            ]

        with tracer.start_as_current_span("kg.query") as span:
            attach_epistemic_columns(rows, fetch)

        assert span.attributes["epistemic.status"] == "contested"
        assert span.attributes["epistemic.contradiction_count"] == 2
        assert span.attributes["epistemic.confidence"] == 0.9

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


class TestEpistemicRowFromStreamRow:
    """CONCEPT:AU-KG.query.knowledge-stream-consumer (report §9 #3) — the bulk,
    ``Method::KnowledgeStream``-sourced sibling of :meth:`EpistemicRow.from_wire`.
    """

    def _stream_row(self, **overrides) -> dict:
        row = {
            "id": "opaque:ref:abc",
            "kind": "graph_row",
            "scores": {"score": None},
            "confidence": 0.72,
            "evidence_kind": None,
            "evidence_refs_json": [],
            "valid_time": (100, 200),
            "tx_time": (10, None),
            "source_refs": ["src:1"],
            "policy_labels": [CONTESTED_LABEL],
            "transformation_ids": [],
            "proof_ids": ["proof:1"],
            "alternative_ids": [],
            "contradiction_ids": ["claim:x"],
            "blob_handle": None,
            "has_payload": False,
        }
        row.update(overrides)
        return row

    def test_maps_every_field_honestly(self) -> None:
        row = epistemic_row_from_stream_row(self._stream_row())
        assert row.id == "opaque:ref:abc"
        assert row.kind == "graph_row"
        assert row.score is None
        assert row.confidence == 0.72
        assert row.valid_time == (100, 200)
        assert row.tx_time == (10, None)
        assert row.source_refs == ["src:1"]
        assert row.policy_labels == [CONTESTED_LABEL]
        assert row.proof_ids == ["proof:1"]
        assert row.contradiction_ids == ["claim:x"]
        # The wire genuinely carries no evidence-locus payload or node
        # properties for this family — never fabricated onto these fields.
        assert row.evidence_refs == []
        assert row.properties == {}

    def test_reads_the_named_score(self) -> None:
        row = epistemic_row_from_stream_row(self._stream_row(scores={"score": 0.5}))
        assert row.score == 0.5

    def test_missing_optional_fields_degrade_to_neutral_defaults(self) -> None:
        row = epistemic_row_from_stream_row({"id": "opaque:ref:bare"})
        assert row.id == "opaque:ref:bare"
        assert row.kind == ""
        assert row.score is None
        assert row.confidence == NEUTRAL_CONFIDENCE
        assert row.valid_time == (None, None)
        assert row.tx_time == (None, None)
        assert row.policy_labels == []


class TestStreamEpistemicRowsByLabel:
    """Live-path: the facade-adjacent bulk sweep actually converts every
    streamed row, and degrades to ``None`` exactly like the underlying
    ``knowledge_stream`` primitive when no engine surface is reachable."""

    def test_none_when_stream_unavailable(self) -> None:
        class _NoStream:
            pass

        assert stream_epistemic_rows_by_label(_NoStream(), "Claim") is None

    def test_converts_every_streamed_row(self, monkeypatch) -> None:
        calls: list[tuple] = []

        def fake_stream(compute, label, *, batch_size=512, limit=0):
            calls.append((compute, label, batch_size, limit))
            yield {
                "id": "opaque:ref:1",
                "kind": "graph_row",
                "scores": {"score": None},
                "confidence": 0.3,
                "source_refs": [],
                "valid_time": (None, None),
                "tx_time": (None, None),
                "policy_labels": [],
                "contradiction_ids": [],
                "proof_ids": [],
            }
            yield {
                "id": "opaque:ref:2",
                "kind": "graph_row",
                "scores": {"score": None},
                "confidence": 0.95,
                "source_refs": [],
                "valid_time": (None, None),
                "tx_time": (None, None),
                "policy_labels": [],
                "contradiction_ids": [],
                "proof_ids": [],
            }

        monkeypatch.setattr(
            "agent_utilities.knowledge_graph.core.knowledge_stream.stream_graph_confidence",
            fake_stream,
        )

        compute = object()
        result = stream_epistemic_rows_by_label(
            compute, "Claim", batch_size=64, limit=10
        )
        rows = list(result)
        assert [r.confidence for r in rows] == [0.3, 0.95]
        assert all(isinstance(r, EpistemicRow) for r in rows)
        assert calls == [(compute, "Claim", 64, 10)]
