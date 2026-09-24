"""EH-032 / EH-362: sameAs proposals are decisions; Wikidata keys are evidence."""

from __future__ import annotations

from agent_utilities.decide.consumers.identity import (
    WikidataAlignment,
    decided_candidates,
    with_wikidata,
)
from agent_utilities.knowledge_graph.assimilation.identity_candidates import (
    EntityRecord,
    resolve_identity_candidates,
)
from tests.unit.decide.fakes import FakeTransport, acted

ROWS = [
    {
        "wikidata_id": "Q140",
        "external_id_field": "ncbi_taxon_id",
        "external_id": "9689",
    },
    {
        "wikidata_id": "Q146",
        "external_id_field": "gbif_taxon_key",
        "external_id": "5219",
    },
]


def _records() -> list[EntityRecord]:
    return with_wikidata(
        [
            EntityRecord("n-1", "Panthera leo", "Taxon", {"ncbi_taxon_id": "9689"}),
            EntityRecord("g-1", "Lion", "Taxon", {"wikidata_id": "Q140"}),
            EntityRecord("g-2", "Panthera leo", "Taxon", {"gbif_taxon_key": "5219"}),
        ],
        WikidataAlignment.from_rows(ROWS),
    )


def test_an_aligned_qid_is_exact_identifier_evidence() -> None:
    records = _records()
    assert records[0].identifiers["wikidata_id"] == "Q140"
    pairs = {(c.entity_a, c.entity_b) for c in resolve_identity_candidates(records)}
    assert ("n-1", "g-1") in pairs, "a shared QID flags the pair"


def test_the_fallback_keeps_flagged_pairs_but_never_two_wikidata_items(
    eg: FakeTransport,
) -> None:
    records = _records()
    candidates = resolve_identity_candidates(records)
    kept = {
        (c.entity_a, c.entity_b) for c, _ in decided_candidates(records, candidates)
    }
    assert ("n-1", "g-1") in kept
    assert ("n-1", "g-2") not in kept, "Q140 and Q146 are two entities"
    request = eg.requests[0]
    assert request["question"]["kind"] == "resolve_entity"
    assert request["question"]["safety"] == "irreversible"


def test_eg_s_decision_is_only_a_proposal(eg: FakeTransport) -> None:
    eg.answer = acted("distinct")
    records = _records()
    candidates = resolve_identity_candidates(records)
    assert decided_candidates(records, candidates) == []
    assert all(c.status == "candidate" for c in candidates), "nothing ever merges"
