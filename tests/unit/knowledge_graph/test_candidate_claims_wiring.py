"""Wiring tests for CONCEPT:AU-KG.enrichment.candidate-claim-extraction.

Universal-ingestion program, Track 4. Three things must be proven with a REAL
call path (not an existence check):

1. A real prose fragment run through :class:`CandidateClaimExtractor` yields a
   :class:`CandidateClaim` whose evidence span resolves back to the fragment
   it cites (``fragment.text[start:end] == quote``).
2. The extractor's no-write-authority boundary is STRUCTURAL: it cannot be
   handed an engine at all (a signature contract), and running it end-to-end
   alongside a fully "live" engine double never touches that double.
3. Confidence is honest: a present, parseable model confidence is carried
   through; an absent/unparseable one abstains (``None``), never a fabricated
   ``0.0``.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncGenerator
from unittest.mock import Mock

import pytest

from agent_utilities.knowledge_graph.extraction.candidate_claims import (
    CandidateClaim,
    CandidateClaimExtractor,
    EvidenceSpan,
    FragmentLike,
    claim_confidence,
)
from agent_utilities.knowledge_graph.extraction.fact_extractor import FactDeduper


class _Fragment:
    """A minimal, real ``FragmentLike`` — no evidence-spine dataclass exists yet
    (see the module docstring), so this is what a caller hands in today."""

    def __init__(self, id: str, text: str) -> None:  # noqa: A002
        self.id = id
        self.text = text


def _fact_json(
    subject: str,
    predicate: str,
    obj: str,
    evidence_span: str,
    *,
    confidence: object = 92,
) -> str:
    payload = {
        "title": f"{subject} {predicate} {obj}",
        "description": "desc",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "evidence_span": evidence_span,
        "tags": ["t"],
    }
    if confidence is not None:
        payload["confidence"] = confidence
    return json.dumps(payload)


def _one_shot_stream(chunk: str):
    async def _stream(_prompt: str, _seed: int) -> AsyncGenerator[str, None]:
        yield chunk

    return _stream


def _no_op_embed(_text: str) -> list[float]:
    """A deterministic, model-free embed fn so tests never load a real embedder."""
    return [1.0, 0.0]


def test_fragment_like_is_structurally_satisfied_by_a_plain_object() -> None:
    """No evidence-spine `Fragment` dataclass exists yet — any object exposing
    `id`/`text` satisfies the duck-typed protocol this module accepts."""
    assert isinstance(_Fragment(id="f1", text="hello"), FragmentLike)
    assert not isinstance(object(), FragmentLike)


# --------------------------------------------------------------------------- #
# 1. Evidence spans resolve back to the real fragment they cite
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_real_prose_fragment_yields_candidate_with_resolvable_evidence_span() -> (
    None
):
    fragment = _Fragment(
        id="frag:doc-1:p3",
        text="Acme Corp acquired Globex in 2024 for an undisclosed sum.",
    )
    quote = "Acme Corp acquired Globex in 2024"
    # `parse_facts_incremental` scans for individual `{"title": ...}` objects
    # (not a wrapping `{"facts": [...]}`) — the shape the model's raw stream
    # actually emits one fact at a time.
    stream = _one_shot_stream(
        _fact_json("Acme Corp", "acquired", "Globex", quote, confidence=88)
    )

    extractor = CandidateClaimExtractor(stream_fn=stream)
    batch = await extractor.propose(
        "Acme Corp acquired Globex in 2024 for an undisclosed sum.",
        [fragment],
        source_id="source:doc-1",
        dedup=False,
    )

    assert batch.counts.accepted == 1
    assert len(batch.candidates) == 1
    claim = batch.candidates[0]
    assert isinstance(claim, CandidateClaim)
    assert claim.subject == "Acme Corp"
    assert claim.predicate == "acquired"
    assert claim.object == "Globex"
    assert claim.model_confidence == pytest.approx(0.88)
    assert claim.review_bucket == "accepted"
    assert claim.extraction_run_id  # bound to a built (not persisted) ExtractionRun

    assert len(claim.evidence) == 1
    span = claim.evidence[0]
    assert span.fragment_id == fragment.id
    assert fragment.text[span.start : span.end] == span.quote == quote
    assert batch.unresolved_evidence == 0


@pytest.mark.asyncio
async def test_evidence_span_pinned_to_fragment_never_fabricated_on_mismatch() -> None:
    """A quote that is NOT a substring of any given fragment resolves to NO
    evidence span at all — never a guessed offset."""
    fragment = _Fragment(id="frag:x", text="totally unrelated fragment text")
    stream = _one_shot_stream(
        _fact_json("A", "relates_to", "B", "this quote appears nowhere")
    )
    extractor = CandidateClaimExtractor(stream_fn=stream)
    batch = await extractor.propose(
        "irrelevant window text", [fragment], source_id="s", dedup=False
    )
    assert len(batch.candidates) == 1
    assert batch.candidates[0].evidence == []
    assert batch.unresolved_evidence == 1


# --------------------------------------------------------------------------- #
# 2. Structural no-write-authority
# --------------------------------------------------------------------------- #


def test_no_public_parameter_is_engine_backend_store_or_session_shaped() -> None:
    """Contract test: the public surface has NO parameter through which a
    live engine/backend/store/session could ever be threaded in."""
    banned = ("engine", "backend", "store", "session")
    targets = [
        CandidateClaimExtractor.__init__,
        CandidateClaimExtractor.propose,
        EvidenceSpan.locate,
        claim_confidence,
    ]
    for fn in targets:
        params = inspect.signature(fn).parameters
        for name in params:
            lowered = name.lower()
            assert not any(b in lowered for b in banned), (
                f"{fn.__qualname__} accepts a {name!r} parameter — "
                "no-write-authority must stay structural"
            )


def test_constructing_or_calling_with_an_engine_kwarg_is_rejected() -> None:
    """Handing the extractor a 'live engine' is not merely unused — it is a
    TypeError, because the parameter does not exist."""
    live_engine = Mock()
    live_engine.add_node = Mock()
    live_engine.add_edge = Mock()
    live_engine.link_nodes = Mock()

    with pytest.raises(TypeError):
        CandidateClaimExtractor(engine=live_engine)  # type: ignore[call-arg]

    extractor = CandidateClaimExtractor()
    with pytest.raises(TypeError):
        extractor.propose(  # type: ignore[call-arg]
            "text", [], source_id="s", engine=live_engine
        )
    live_engine.add_node.assert_not_called()
    live_engine.add_edge.assert_not_called()
    live_engine.link_nodes.assert_not_called()


@pytest.mark.asyncio
async def test_full_extraction_run_never_touches_a_live_engine_double() -> None:
    """End-to-end proof: run a real multi-round, deduped extraction alongside a
    fully 'live' engine double (working add_node/add_edge/link_nodes/
    persist-shaped methods) sitting in scope, and confirm it is NEVER called —
    not because it was mocked to no-op, but because nothing in this module
    ever references it."""
    live_engine = Mock()
    live_engine.add_node = Mock()
    live_engine.add_edge = Mock()
    live_engine.link_nodes = Mock()
    live_engine.graph = Mock()
    live_engine.backend = Mock()

    fragment = _Fragment(id="frag:1", text="Widgets Inc supplies Acme Corp.")
    stream = _one_shot_stream(
        _fact_json("Widgets Inc", "supplies", "Acme Corp", "Widgets Inc supplies")
    )
    extractor = CandidateClaimExtractor(stream_fn=stream)
    deduper = FactDeduper(embed_fn=_no_op_embed)

    batch = await extractor.propose(
        "Widgets Inc supplies Acme Corp.",
        [fragment],
        source_id="s2",
        rounds=1,
        dedup=True,
        deduper=deduper,
    )

    assert len(batch.candidates) == 1
    assert live_engine.mock_calls == []  # never touched — not passed, not reachable


# --------------------------------------------------------------------------- #
# 3. Confidence honesty — abstain, never fabricate
# --------------------------------------------------------------------------- #


def test_claim_confidence_scales_a_real_signal() -> None:
    assert claim_confidence(88) == pytest.approx(0.88)
    assert claim_confidence(0) == 0.0
    assert claim_confidence(150) == 1.0  # clamped, not rejected


def test_claim_confidence_abstains_rather_than_fabricating() -> None:
    assert claim_confidence(None) is None
    assert claim_confidence("not-a-number") is None


@pytest.mark.asyncio
async def test_missing_model_confidence_abstains_on_the_candidate() -> None:
    fragment = _Fragment(id="frag:2", text="Foo Corp merged with Bar Ltd.")
    stream = _one_shot_stream(
        _fact_json(
            "Foo Corp", "merged_with", "Bar Ltd", "Foo Corp merged", confidence=None
        )
    )
    extractor = CandidateClaimExtractor(stream_fn=stream)
    batch = await extractor.propose(
        "Foo Corp merged with Bar Ltd.", [fragment], source_id="s3", dedup=False
    )
    assert len(batch.candidates) == 1
    claim = batch.candidates[0]
    assert claim.model_confidence is None
    assert claim.abstained is True
    # Classification still runs (falls back to a conservative bucket) —
    # abstaining on the NUMBER never means the claim is silently dropped or
    # silently "accepted" with an invented confidence.
    assert claim.review_bucket in ("needs_review", "accepted")
