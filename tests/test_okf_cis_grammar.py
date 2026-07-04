"""OKF-CIS grammar + vocab tests (CONCEPT:AU-OS.governance.concept-2).

Covers the additive standard: id<->path<->IRI round-trip, the >=2-segment rule,
rejection of legacy/malformed ids, the shared-pillar federation invariant, the
closed domain vocab, and the 2-letter slug registry uniqueness.
"""

from __future__ import annotations

import pytest

from agent_utilities.governance import concept_hierarchy as ch


@pytest.mark.parametrize(
    "cid,path",
    [
        ("AU-KG.ingest.entropy-dedup", "AU/KG/ingest/entropy-dedup"),
        ("EG-KG.storage.redb", "EG/KG/storage/redb"),
        ("DS-AHE.trainer.gpu-slot", "DS/AHE/trainer/gpu-slot"),
        ("AU-OS.governance.concept-allocator.v2", "AU/OS/governance/concept-allocator/v2"),
    ],
)
def test_id_path_iri_roundtrip(cid: str, path: str) -> None:
    parsed = ch.parse_okf_id(cid)
    assert parsed.canonical == cid
    assert parsed.path == path
    assert ch.okf_id_to_path(cid) == path
    assert ch.path_to_okf_id(path) == cid
    assert ch.path_to_okf_id(path + ".md") == cid  # .md suffix tolerated
    assert parsed.iri == f"{ch.CONCEPT_IRI_BASE}/{path}"
    assert ch.concept_iri(cid) == parsed.iri


def test_shared_pillar_federates_across_repos() -> None:
    # Two repos, same pillar -> the SAME pillar IRI (no slug): cross-repo linking.
    au = ch.parse_okf_id("AU-KG.ingest.a")
    eg = ch.parse_okf_id("EG-KG.storage.b")
    assert au.pillar_iri == eg.pillar_iri == f"{ch.PILLAR_IRI_BASE}/KG"
    assert au.scheme_iri != eg.scheme_iri  # but distinct owning schemes


@pytest.mark.parametrize(
    "bad",
    [
        "AU-KG.ingest",  # only 1 segment -> a domain node, not a concept
        "AU-KG.retrieval.synthesized-cited-answer",  # legacy numeric grammar
        "AUX-KG.a.b",  # slug not 2 letters
        "AU-XX.a.b",  # pillar not in the closed set
        "AU-KG.A.b",  # uppercase segment
        "au-kg.a.b",  # lowercase slug/pillar
        "AU-KG..b",  # empty segment
    ],
)
def test_rejects_malformed(bad: str) -> None:
    assert not ch.is_okf_id(bad)
    with pytest.raises(ValueError):
        ch.parse_okf_id(bad)


def test_legacy_parser_still_works() -> None:
    # Additive: the legacy grammar must keep parsing during migration.
    assert ch.parse_concept_id("AU-KG.retrieval.synthesized-cited-answer").canonical == "AU-KG.retrieval.synthesized-cited-answer"
    assert ch.canonicalize("AU-KG.compute.numeric-kernel") == "AU-KG.ontology.concept-hierarchy"


def test_domain_vocab_closed_and_covers_all_pillars() -> None:
    vocab = ch.load_domain_vocab()
    assert set(vocab) == set(ch.PILLARS)
    assert ch.is_valid_domain("KG", "ingest")
    assert not ch.is_valid_domain("KG", "not-a-real-domain")
    # every listed domain is a valid concept-slug segment shape
    for pillar, domains in vocab.items():
        for dom in domains:
            assert ch.is_okf_id(f"AU-{pillar}.{dom}.x"), (pillar, dom)


def test_slug_registry_unique_and_two_letter() -> None:
    reg = ch.load_slug_registry()
    assert ch.slug_for_repo("agent-utilities") == "AU"
    assert ch.slug_for_repo("epistemic-graph") == "EG"
    assert len(set(reg.values())) == len(reg)  # globally unique
    assert all(len(s) == 2 and s.isupper() for s in reg.values())


def test_part_of_edges_chain_to_shared_pillar() -> None:
    parsed = [ch.parse_okf_id("AU-KG.ingest.dedup"), ch.parse_okf_id("EG-KG.storage.redb")]
    edges = dict(ch.okf_part_of_edges(parsed))
    # concept -> domain -> pillar; both repos' KG concepts reach one pillar node
    assert edges[parsed[0].iri] == parsed[0].domain_iri
    assert edges[parsed[0].domain_iri] == parsed[0].pillar_iri
    assert parsed[0].pillar_iri == parsed[1].pillar_iri
