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

from dataclasses import dataclass, field
from typing import Any

__all__ = ["EpistemicRow", "row_ids_from_plain_rows"]


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
