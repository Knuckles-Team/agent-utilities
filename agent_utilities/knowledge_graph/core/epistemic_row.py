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

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "EpistemicRow",
    "EvidenceSpan",
    "attach_epistemic_rows",
    "row_ids_from_plain_rows",
    "attach_epistemic_columns",
    "should_attach_epistemic_columns",
    "is_contested_row",
    "epistemic_status",
    "NEUTRAL_CONFIDENCE",
    "CONTESTED_LABEL",
]

#: Neutral confidence prior used when no epistemic envelope resolves for a row
#: (mirrors ``retrieval/context_compiler.py``'s ``_NEUTRAL_CONFIDENCE`` — the
#: SAME "no information yet" belief-strength default; kept as its own constant
#: here, rather than imported, because ``knowledge_graph/core`` sits BELOW
#: ``knowledge_graph/retrieval`` in the layering and must not import it).
NEUTRAL_CONFIDENCE = 0.5

#: Policy label the engine (and ``retrieval/context_compiler.py``) uses to
#: flag a disputed/contested claim.
CONTESTED_LABEL = "epistemic:contested"

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
    #: SURPASS gap-closure ("wire `proof_ids`/`contradiction_ids` from the Arrow
    #: `KnowledgeBatch` into `EpistemicRow`") — ids this row CONTRADICTS/ATTACKS or is
    #: contradicted/attacked BY (SYMMETRIC), straight off the wire's
    #: ``ExplainProvenanceRowWire.contradiction_ids`` (same column
    #: ``eg_plan::KnowledgeBatch``'s Arrow-columnar surface already carries — this
    #: row-shaped path is the one that actually reaches a Python caller). Empty when
    #: the engine's ``epistemic`` feature is off, or the row has no classified
    #: contradiction/attack edge — never fabricated.
    contradiction_ids: list[str] = field(default_factory=list)
    #: SURPASS gap-closure (see ``contradiction_ids`` above) — the transitive
    #: justification/premise chain underneath this row's belief, deduped, excluding
    #: the row's own id, off ``ExplainProvenanceRowWire.proof_ids``. Empty when
    #: ``epistemic`` is off or the row has no evidence neighbourhood.
    proof_ids: list[str] = field(default_factory=list)
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
            contradiction_ids=list(row.get("contradiction_ids") or []),
            proof_ids=list(row.get("proof_ids") or []),
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


def epistemic_status(
    *,
    resolved: bool,
    confidence: float,
    policy_labels: list[str] | tuple[str, ...],
    contradiction_count: int = 0,
    low_confidence_threshold: float = NEUTRAL_CONFIDENCE,
) -> str:
    """Derive a coarse, human/OTel-friendly status label for one row's light
    epistemic envelope (CONCEPT:AU-KB-CURRENCY light layer).

    A DERIVED label, never a wire field — the engine has no ``status``
    concept of its own at this primitive; this just names the same signal
    :func:`attach_epistemic_columns` already carries so a caller (or an OTel
    span attribute, see ``observability.TelemetryEngine.annotate_epistemic``)
    doesn't have to re-derive it:

    - ``"unresolved"``: no envelope was found for this row's id — the
      neutral-prior case (no compute engine, id not found, request failed).
    - ``"contested"``: the engine tagged the row :data:`CONTESTED_LABEL`, or
      the caller supplied a positive ``contradiction_count``.
    - ``"low_confidence"``: resolved, not contested, confidence below
      ``low_confidence_threshold``.
    - ``"confirmed"``: resolved, not contested, confidence at/above
      threshold.
    """
    if not resolved:
        return "unresolved"
    if contradiction_count > 0 or CONTESTED_LABEL in policy_labels:
        return "contested"
    if confidence < low_confidence_threshold:
        return "low_confidence"
    return "confirmed"


def is_contested_row(row: dict[str, Any]) -> bool:
    """Cheap, LOCAL (no RPC) contested/low-confidence check on a plain row's
    OWN already-materialized properties — e.g. a ``confidence``/
    ``policy_labels`` property some write path already stamped on the node,
    independent of any ``explain_provenance_by_ids`` round trip.

    Used by :func:`should_attach_epistemic_columns` to auto-force the light
    epistemic attach even when a deployment has opted the default off
    (CONCEPT:AU-KB-CURRENCY — the "auto-on for contested/low-confidence
    results" contract): a row that already LOOKS contested/uncertain from
    data it already carries is never silently served without resolving its
    full epistemic context.
    """
    candidates: list[dict[str, Any]] = [row]
    for value in row.values():
        if isinstance(value, dict):
            candidates.append(value)
    for node in candidates:
        labels = node.get("policy_labels") or []
        if isinstance(labels, list | tuple) and CONTESTED_LABEL in labels:
            return True
        conf = node.get("confidence")
        if conf is not None:
            try:
                if float(conf) < NEUTRAL_CONFIDENCE:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def should_attach_epistemic_columns(
    rows: list[dict[str, Any]], *, default: bool
) -> bool:
    """Whether the light epistemic attach (:func:`attach_epistemic_columns`)
    should run for this batch of plain rows (CONCEPT:AU-KB-CURRENCY light
    layer — the "Native by default" resolution for ``config.
    epistemic_light_default``).

    ``default`` is the deployment's configured default. When it is ``True``
    (the platform default), this always returns ``True`` — the light
    envelope attaches on every read. When an operator has opted it ``False``
    (a deployment that must skip the extra batched
    ``explain_provenance_by_ids`` round trip on every read), this STILL
    returns ``True`` if any row already shows a contested/low-confidence
    signal in its own properties (:func:`is_contested_row`, a cheap, local,
    no-RPC check) — a disputed/uncertain result is never silently served
    without its epistemic context.
    """
    if default:
        return True
    return any(is_contested_row(row) for row in rows if isinstance(row, dict))


def _row_id(row: dict[str, Any]) -> str | None:
    """Best-effort id extraction for ONE plain row — the single-row sibling
    of :func:`row_ids_from_plain_rows`, used to zip the light-attach
    envelope back onto its originating row without disturbing row
    order/identity."""
    for value in row.values():
        if isinstance(value, dict) and isinstance(value.get("id"), str) and value["id"]:
            return value["id"]
    top_id = row.get("id")
    if isinstance(top_id, str) and top_id:
        return top_id
    return None


def attach_epistemic_columns(
    rows: list[dict[str, Any]],
    fetch: Callable[[list[str]], list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    """LIGHT, additive epistemic-columns attach (CONCEPT:AU-KB-CURRENCY light
    layer) — the non-breaking counterpart to :func:`attach_epistemic_rows`.

    Merges ``confidence``/``source_refs``/``evidence_refs``/
    ``policy_labels``/``provenance`` onto each row IN PLACE via
    ``dict.setdefault`` (never clobbers a property the row already carries
    under one of those names) and returns the SAME ``list[dict]`` — unlike
    :func:`attach_epistemic_rows`, this NEVER changes the caller's return
    type, so it is safe to run BY DEFAULT on every plain read path
    (:meth:`~agent_utilities.knowledge_graph.facade.KnowledgeGraph.query`,
    ``GraphComputeEngine.query_unified``, ``IntelligenceGraphEngine.uql``)
    without breaking any existing caller expecting ``list[dict]`` rows.

    Degrades cleanly (the documented neutral-prior contract): when ``fetch``
    is unavailable, raises, or a specific row's id can't be resolved, that
    row keeps 100% of its original data and gets a neutral prior
    (``confidence=NEUTRAL_CONFIDENCE``, empty ref/label lists,
    ``provenance={"resolved": False, ...}``) instead of being dropped or the
    whole call raising/emptying — a backend with no epistemic support
    degrades to "unknown", never to an error or a truncated result set.

    Also stamps the resolved envelope onto the current OTel span (best
    effort, no-op when tracing isn't configured — CONCEPT:AU-KB-CURRENCY OTel
    projection) via ``observability.get_telemetry_engine().annotate_epistemic``,
    using the highest ``contradiction_count``/lowest ``confidence`` seen
    across the batch so a contested/low-confidence result is visible on the
    span that produced it.
    """
    if not rows:
        return rows
    wire_by_id: dict[str, dict[str, Any]] = {}
    if fetch is not None:
        try:
            id_props = row_ids_from_plain_rows(rows)
            ids = [ip["id"] for ip in id_props]
            if ids:
                wire_rows = fetch(ids) or []
                wire_by_id = {
                    wr["id"]: wr
                    for wr in wire_rows
                    if isinstance(wr, dict) and isinstance(wr.get("id"), str)
                }
        except Exception as exc:  # pragma: no cover - never break a plain read
            logger.debug("light epistemic-columns attach skipped: %s", exc)
            wire_by_id = {}

    worst_confidence = 1.0
    any_contested = False
    max_contradiction_count = 0
    all_policy_labels: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        rid = _row_id(row)
        wr = wire_by_id.get(rid) if rid else None
        resolved = wr is not None
        if wr is not None:
            confidence = float(
                wr.get("confidence", NEUTRAL_CONFIDENCE) or NEUTRAL_CONFIDENCE
            )
            source_refs = list(wr.get("source_refs") or [])
            evidence_refs = list(wr.get("evidence_spans") or [])
            policy_labels = list(wr.get("policy_labels") or [])
            # SURPASS gap-closure ("wire proof_ids/contradiction_ids into
            # EpistemicRow"): straight off the wire, same as every other column
            # above -- a row's REAL contradiction count, not just the
            # `CONTESTED_LABEL` policy-label proxy `any_contested` used before
            # these columns reached here.
            contradiction_ids = list(wr.get("contradiction_ids") or [])
            proof_ids = list(wr.get("proof_ids") or [])
            provenance: dict[str, Any] = {
                "resolved": True,
                "valid_time": wr.get("valid_time"),
                "tx_time": wr.get("tx_time"),
            }
        else:
            confidence = NEUTRAL_CONFIDENCE
            source_refs = []
            evidence_refs = []
            policy_labels = []
            contradiction_ids = []
            proof_ids = []
            provenance = {"resolved": False, "valid_time": None, "tx_time": None}
        row.setdefault("confidence", confidence)
        row.setdefault("source_refs", source_refs)
        row.setdefault("evidence_refs", evidence_refs)
        row.setdefault("policy_labels", policy_labels)
        row.setdefault("contradiction_ids", contradiction_ids)
        row.setdefault("proof_ids", proof_ids)
        row.setdefault("provenance", provenance)

        worst_confidence = min(worst_confidence, confidence)
        all_policy_labels.update(policy_labels)
        max_contradiction_count = max(max_contradiction_count, len(contradiction_ids))
        if CONTESTED_LABEL in policy_labels:
            any_contested = True

    # A row can be contested via either signal: the engine's own `CONTESTED_LABEL`
    # policy label, OR a non-empty `contradiction_ids` list now that it actually
    # reaches this far -- take the real count when present, else fall back to the
    # boolean label-only proxy so a build/row still missing the column (e.g. an
    # older engine) behaves exactly as it did before this gap-closure.
    contradiction_count = max_contradiction_count or (1 if any_contested else 0)

    try:
        from agent_utilities.observability import get_telemetry_engine

        get_telemetry_engine().annotate_epistemic(
            confidence=worst_confidence,
            status=epistemic_status(
                resolved=bool(wire_by_id),
                confidence=worst_confidence,
                policy_labels=tuple(all_policy_labels),
                contradiction_count=contradiction_count,
            ),
            contradiction_count=contradiction_count,
            policy_labels=sorted(all_policy_labels),
            source_count=len(rows),
        )
    except Exception as exc:  # pragma: no cover - tracing must never break a read
        logger.debug("epistemic OTel span annotation skipped: %s", exc)

    return rows
