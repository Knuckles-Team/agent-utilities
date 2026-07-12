#!/usr/bin/python
from __future__ import annotations

"""``EpistemicRow`` — the typed, epistemic-columns-carrying query result (CONCEPT:AU-KB-CURRENCY).

**Seam 1 (the keystone cross-repo currency).** ``epistemic-graph``'s engine-native
query surfaces (``eg_plan::KnowledgeSet`` / ``KnowledgeBatch``, CONCEPT:EG-P1-2) resolve
a per-row epistemic envelope — score, belief confidence, a bitemporal valid/tx
window, evidence provenance, policy labels — for every result row. Before this
module, agent-utilities' primary read path (:meth:`KnowledgeGraph.query`) flattened
every result to a plain ``dict`` and DROPPED all of that: a caller got back node
properties and nothing else, no matter how much epistemic metadata the engine had
already computed for the exact same rows.

This module is the **AU-side half** of the seam: it defines :class:`EpistemicRow`,
the typed carrier a caller gets back when it opts in
(``KnowledgeGraph.query(cypher, include_epistemic=True)``), and the small
plumbing (:func:`row_ids_from_plain_rows`) that bridges a plain-dict Cypher result
to the engine's id-seeded epistemic surface
(``Method::ExplainProvenanceByIds`` / ``client.query.explain_provenance_by_ids``,
CONCEPT:EG-KB-CURRENCY) — the ONE new, minimal wire surface added on the engine side
for this feature. See ``docs/architecture/epistemic-columns-currency.md`` for the
full seam write-up (both repos, the wire shape, and how another AU consumer adopts
the same pattern for its own read path).

Every field on :class:`EpistemicRow` is a straight copy of a value the engine's
``KnowledgeSet``/``KnowledgeBatch`` ALREADY computed server-side for that row — this
module fabricates nothing. ``score``/``confidence``/``valid_time``/``tx_time`` are
populated regardless of the engine's ``epistemic`` build feature;
``source_refs``/``policy_labels``/``evidence_refs`` are empty (never fabricated,
honestly reported via the wire's ``resolved`` flag) when that feature is off.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "EpistemicRow",
    "EvidenceSpan",
    "attach_epistemic_rows",
    "row_ids_from_plain_rows",
]

# The field names every ``eg_modality::EvidenceSpan`` variant contributes, unioned —
# see ``EvidenceSpanWire`` in ``epistemic-graph/crates/eg-types/src/protocol.rs`` for
# the authoritative Rust enum this mirrors. Used by :meth:`EvidenceSpan.from_wire` to
# pick out the recognized keys from a variant's field map; anything else on the map is
# preserved verbatim in :attr:`EvidenceSpan.raw` rather than dropped.
_EVIDENCE_SPAN_FIELDS = frozenset(
    {
        "document_id",
        "start",
        "end",
        "table_id",
        "row_start",
        "row_end",
        "col_start",
        "col_end",
        "image_id",
        "x",
        "y",
        "width",
        "height",
        "page",
        "audio_id",
        "start_ms",
        "end_ms",
        "video_id",
        "start_frame",
        "end_frame",
        "metric",
        "table",
        "row_id",
        "version",
        "file_path",
        "symbol",
        "start_line",
        "end_line",
        "trace_id",
        "span_id",
    }
)


@dataclass(frozen=True)
class EvidenceSpan:
    """Typed view of one wire evidence-span locus (CONCEPT:AU-KB-CURRENCY follow-up).

    Mirrors ``eg_modality::EvidenceSpan`` / its wire twin ``EvidenceSpanWire``
    (``epistemic-graph/crates/eg-types/src/protocol.rs``) — an externally-tagged
    Rust enum with 11 variants (``DocumentSpan``/``TableCellRange``/``ImageRegion``/
    ``PageBox``/``AudioSegment``/``VideoShot``/``VideoFrameRange``/``MetricWindow``/
    ``RowVersion``/``CodeSymbol``/``TraceSpan``), which msgpack serializes as a
    single-key map ``{"<Variant>": {field: value, ...}}`` (the same shape
    ``media_store.py``'s ``BeliefGraph::from_graph_view`` already decodes for a
    ``PageBox`` locus). ``EpistemicRow.evidence_refs`` carries that raw wire dict
    verbatim; this dataclass is the typed, attribute-accessible view a caller gets
    via :meth:`EpistemicRow.typed_evidence_refs` instead of hand-parsing the map.

    ``kind`` is the Rust variant name; every OTHER field is ``None`` unless that
    variant's own field map set it (a straight copy off the wire, never fabricated
    or defaulted to a guessed value) — a caller checks ``kind`` to know which
    fields are meaningful, exactly as matching the Rust enum would. :attr:`raw`
    keeps the variant's full field map for a caller that wants the untouched
    payload (e.g. an unrecognized future field this dataclass hasn't been
    widened for yet).
    """

    kind: str
    document_id: str | None = None
    start: int | None = None
    end: int | None = None
    table_id: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    col_start: int | None = None
    col_end: int | None = None
    image_id: str | None = None
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    page: int | None = None
    audio_id: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    video_id: str | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    metric: str | None = None
    table: str | None = None
    row_id: str | None = None
    version: int | None = None
    file_path: str | None = None
    symbol: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    trace_id: str | None = None
    span_id: str | None = None
    #: The variant's full field map exactly as the wire carried it (never dropped,
    #: even for a key this dataclass doesn't have a named attribute for).
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, raw: Any) -> EvidenceSpan | None:
        """Parse one externally-tagged ``{"<Variant>": {...}}`` wire entry.

        Returns ``None`` (never raises, never fabricates) for anything that
        doesn't match the expected single-key-map shape — a malformed or
        future/unrecognized entry is skipped rather than guessed at.
        """
        if not isinstance(raw, dict) or len(raw) != 1:
            return None
        ((kind, fields),) = raw.items()
        if not isinstance(fields, dict):
            return None
        known = {k: v for k, v in fields.items() if k in _EVIDENCE_SPAN_FIELDS}
        return cls(kind=kind, raw=dict(fields), **known)


@dataclass(frozen=True)
class EpistemicRow:
    """One query-result row, widened with the engine's epistemic envelope.

    Constructed via :meth:`from_wire` from the raw dict
    ``client.query.explain_provenance_by_ids(...)`` (or ``explain_provenance``)
    returns — see ``ExplainProvenanceRowWire`` in
    ``epistemic-graph/crates/eg-types/src/protocol.rs`` for the authoritative wire
    shape this mirrors field-for-field.
    """

    id: str
    kind: str
    score: float | None
    confidence: float
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    valid_time: tuple[int | None, int | None] = (None, None)
    tx_time: tuple[int | None, int | None] = (None, None)
    policy_labels: list[str] = field(default_factory=list)
    #: The plain node-property dict the ORIGINAL (non-epistemic) query projected
    #: for this id, when the caller supplied one (see
    #: :meth:`KnowledgeGraph.query`'s ``include_epistemic`` path) — so opting into
    #: the epistemic envelope never loses the properties a plain ``dict`` row
    #: would have carried. Empty when no matching plain row was found/supplied.
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def calibration(self) -> float:
        """The row's calibrated belief signal.

        The engine substrate's ``confidence`` (``KnowledgeRow::confidence``,
        `eg_plan::knowledge`) IS the calibration signal it produces today — there is
        no second, separate "calibration" column in the wire shape. This alias
        exists so a caller using either vocabulary (confidence-scoring vs.
        calibration-scoring literature both use "confidence"/"calibration"
        interchangeably for a belief-strength scalar in ``[0, 1]``) reads the same
        value under its preferred name, without this module fabricating a second,
        independently-computed number that does not exist server-side.
        """
        return self.confidence

    @property
    def typed_evidence_refs(self) -> list[EvidenceSpan]:
        """Typed view of :attr:`evidence_refs` (CONCEPT:AU-KB-CURRENCY follow-up).

        One :class:`EvidenceSpan` per wire entry that parses as a recognized
        externally-tagged variant; an entry that doesn't (malformed, or a future
        variant this dataclass hasn't been widened for) is skipped — never
        fabricated. Empty whenever :attr:`evidence_refs` is (e.g. the engine's
        ``epistemic`` feature is off, see the class docstring).
        """
        spans = (EvidenceSpan.from_wire(r) for r in self.evidence_refs)
        return [s for s in spans if s is not None]

    @classmethod
    def from_wire(
        cls, row: dict[str, Any], *, properties: dict[str, Any] | None = None
    ) -> EpistemicRow:
        """Build one :class:`EpistemicRow` from a raw
        ``ExplainProvenanceRowWire``-shaped dict (a single entry of
        ``client.query.explain_provenance_by_ids(ids)["rows"]``).

        ``valid_time``/``tx_time`` arrive over msgpack as 2-element lists (Rust
        tuples have no native msgpack tuple type) — normalized to a real Python
        ``tuple`` here so callers get the documented ``(from, until)`` pair.
        """

        def _pair(v: Any) -> tuple[int | None, int | None]:
            if isinstance(v, list | tuple) and len(v) == 2:
                return (v[0], v[1])
            return (None, None)

        return cls(
            id=row.get("id", ""),
            kind=row.get("kind", ""),
            score=row.get("score"),
            confidence=row.get("confidence", 1.0),
            evidence_refs=list(row.get("evidence_spans") or []),
            source_refs=list(row.get("source_refs") or []),
            valid_time=_pair(row.get("valid_time")),
            tx_time=_pair(row.get("tx_time")),
            policy_labels=list(row.get("policy_labels") or []),
            properties=dict(properties or {}),
        )


def row_ids_from_plain_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract ``(id, properties)`` pairs out of plain Cypher result ``rows``.

    A plain-dict row from :meth:`KnowledgeGraph.query` is keyed by RETURN
    alias; a bare ``RETURN n`` column holds the FULL node dict with an injected
    ``"id"`` key (see ``EpistemicGraphBackend._project``'s ``_project_item``), while
    a projection like ``RETURN n.id AS id`` yields a top-level ``"id"`` key whose
    value is the bare id string. Both shapes are recognized here; a row matching
    neither contributes nothing (never a guess). Returns one dict per DISTINCT id
    found, first-occurrence order preserved, each shaped
    ``{"id": str, "properties": dict}`` — ``properties`` is the nested node dict
    when found, else ``{}`` (the ``RETURN n.id AS id`` shape has no properties to
    carry).
    """
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for value in row.values():
            if (
                isinstance(value, dict)
                and isinstance(value.get("id"), str)
                and value["id"]
            ):
                nid = value["id"]
                seen.setdefault(nid, {"id": nid, "properties": value})
        top_id = row.get("id")
        if isinstance(top_id, str) and top_id:
            seen.setdefault(top_id, {"id": top_id, "properties": {}})
    return list(seen.values())


def attach_epistemic_rows(
    rows: list[dict[str, Any]],
    fetch: Callable[[list[str]], list[dict[str, Any]]] | None,
) -> list[EpistemicRow]:
    """Currency-upgrade plain ``rows`` into :class:`EpistemicRow` results (CONCEPT:AU-KB-CURRENCY).

    The ONE shared implementation of the Seam 1 pattern every AU read surface's
    ``include_epistemic=True`` opt-in adopts (:meth:`KnowledgeGraph.query`,
    ``GraphComputeEngine.query_unified``, ``IntelligenceGraphEngine.uql``,
    ``GraphBackend.execute``): extract the distinct ids ``rows`` project (via
    :func:`row_ids_from_plain_rows` — recognizes both a nested node-dict
    projection and a bare ``id`` scalar column), resolve their epistemic envelope
    in ONE round-trip via ``fetch`` (an id-seeded engine primitive shaped like
    ``GraphComputeEngine.explain_provenance_by_ids`` /
    ``Method::ExplainProvenanceByIds``), and zip each engine row back with the
    plain row's own properties — in the ORIGINAL rows' order — so opting in never
    loses information the plain row carried, and a plan's rank order (e.g.
    ``query_unified``'s post-``Rank`` order) survives the upgrade.

    Degrades to ``[]`` (never raises, never fabricates) when ``fetch`` is
    ``None`` (no compute engine exposes the primitive) or nothing in ``rows``
    carries a resolvable id — the documented ``include_epistemic`` contract every
    adopting read surface shares.
    """
    if fetch is None:
        return []
    id_props = row_ids_from_plain_rows(rows)
    if not id_props:
        return []
    props_by_id = {ip["id"]: ip["properties"] for ip in id_props}
    wire_rows = fetch([ip["id"] for ip in id_props]) or []
    wire_by_id = {wr.get("id"): wr for wr in wire_rows if isinstance(wr, dict)}
    result: list[EpistemicRow] = []
    for ip in id_props:
        wr = wire_by_id.get(ip["id"])
        if wr is None:
            continue
        result.append(
            EpistemicRow.from_wire(wr, properties=props_by_id.get(ip["id"], {}))
        )
    return result
