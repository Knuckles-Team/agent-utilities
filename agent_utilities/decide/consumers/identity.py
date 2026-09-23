"""EH-032 / EH-362: entity resolution (``owl:sameAs`` proposals) through EG ``Decide``.

``resolve_identity_candidates`` flags pairs whose combined evidence clears a
threshold; it never merges. Each flagged pair is now a full-label
``resolve_entity`` question over two declared options, ``same_as`` and
``distinct`` -- the regime where a gold-set-fitted head carries EG's
conformal act-risk guarantee (DECIDE-LAYER-DESIGN §6.3). The answer is still
only a PROPOSAL: ``same_as`` keeps the candidate for governed review,
``distinct`` withdraws it; :func:`confirm_merge` stays the one merge path.

Wikidata keys (EH-362): world-reference-mcp's alignment stream maps a
connector's external id (``ncbi_taxon_id``, ``chebi_id``, ...) to a Wikidata
QID. Aligned records gain a ``wikidata_id`` identifier, so a shared QID is
exact-identifier evidence in the resolver, and every pair declares
``wikidata_match``: +1 same item, -1 different items, absent when either side
is unaligned (the schema's declared imputation, never a silent zero). The
deterministic fallback keeps a flagged pair unless both sides are aligned to
DIFFERENT items -- two Wikidata items are two entities.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_utilities import decide
from agent_utilities.decide.options import Option, text_param

WIKIDATA_ID = "wikidata_id"
SAME_AS = "same_as"
DISTINCT = "distinct"


@dataclass(frozen=True, slots=True)
class WikidataAlignment:
    """``(external_id_field, external_id) -> QID`` from the alignment stream."""

    by_key: Mapping[tuple[str, str], str] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> WikidataAlignment:
        return cls(
            {
                (str(r["external_id_field"]), str(r["external_id"])): str(
                    r["wikidata_id"]
                )
                for r in rows
                if r.get("wikidata_id") and r.get("external_id_field")
            }
        )

    def qid_for(self, identifiers: Mapping[str, str]) -> str | None:
        if identifiers.get(WIKIDATA_ID):
            return str(identifiers[WIKIDATA_ID])
        for key in sorted(identifiers):
            qid = self.by_key.get((key, str(identifiers[key])))
            if qid:
                return qid
        return None


def with_wikidata(records: Sequence[Any], alignment: WikidataAlignment) -> list[Any]:
    """``records`` with a ``wikidata_id`` identifier wherever the alignment knows one."""
    out = []
    for record in records:
        qid = alignment.qid_for(record.identifiers)
        if qid and record.identifiers.get(WIKIDATA_ID) != qid:
            identifiers = {**record.identifiers, WIKIDATA_ID: qid}
            record = dataclasses.replace(record, identifiers=identifiers)
        out.append(record)
    return out


def wikidata_match(a: Any, b: Any) -> float | None:
    """+1 same QID, -1 different QIDs, ``None`` when either side is unaligned."""
    qa, qb = a.identifiers.get(WIKIDATA_ID), b.identifiers.get(WIKIDATA_ID)
    if not qa or not qb:
        return None
    return 1.0 if qa == qb else -1.0


def _options(confidence: float, match: float | None) -> list[Option]:
    same = {"similarity": confidence}
    other = {"similarity": 1.0 - confidence}
    if match is not None:
        same["wikidata_match"], other["wikidata_match"] = match, -match
    return [Option(SAME_AS, same), Option(DISTINCT, other)]


def decide_pair(candidate: Any, a: Any, b: Any) -> decide.Choice:
    """The proposal for one flagged pair: ``same_as`` (keep) or ``distinct``."""
    match = wikidata_match(a, b)
    fallback = DISTINCT if match == -1.0 else SAME_AS
    return decide.choose(
        "au.entity.same_as",
        _options(float(candidate.confidence), match),
        lambda: fallback,
        params=[text_param("kind", a.kind or b.kind or "entity")],
    )


def decided_candidates(
    records: Sequence[Any], candidates: Sequence[Any]
) -> list[tuple[Any, decide.Choice]]:
    """The flagged pairs EG (or the fallback) proposes as ``same_as``."""
    by_id = {r.id: r for r in records}
    kept = []
    for candidate in candidates:
        choice = decide_pair(
            candidate, by_id[candidate.entity_a], by_id[candidate.entity_b]
        )
        if choice.option_id == SAME_AS:
            kept.append((candidate, choice))
    return kept


__all__ = [
    "DISTINCT",
    "SAME_AS",
    "WIKIDATA_ID",
    "WikidataAlignment",
    "decide_pair",
    "decided_candidates",
    "wikidata_match",
    "with_wikidata",
]
