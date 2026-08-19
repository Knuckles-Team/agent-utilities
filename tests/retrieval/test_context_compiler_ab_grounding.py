"""A/B grounding-quality harness: ContextCompiler ON vs OFF (W3.7, CONCEPT:AU-KG.retrieval.context-compiler).

A test-level harness that runs the SAME delegated-run context-assembly path
(:func:`agent_utilities.core.contextual_model._compiled_evidence_and_bundle` — the
exact function the mandatory model-transport wrapper calls for every
``execute_agent``/``execute_workflow`` model turn, see
``test_context_compiler_delegated_run_default.py``) over a small fixture corpus,
once with the compiler's default-on behavior and once with the
``MODEL_CONTEXT_COMPILER_ENABLED`` escape hatch off, and compares two grounding-
quality signals:

* **citation coverage** — the fraction of selected context items whose citation
  carries at least one evidence ``source_ref``.
* **provenance density** — the average count of source/evidence references plus
  proof-graph edges per selected item.

Both are computed straight off the real :class:`~agent_utilities.knowledge_graph.
retrieval.context_compiler.ContextBundle` the wrapper produces (or the absence of
one, when the escape hatch is off) — not a synthetic proxy. The numbers are logged
(``logger.info``, captured below via ``caplog`` as proof-of-logging) and the delta
is asserted directionally: compiler-ON must strictly exceed compiler-OFF on a
corpus that actually carries citeable evidence, and compiler-OFF must be exactly
zero on both axes (an escape-hatch run sends the model no evidence at all, by
construction — see the mandatory-behavior tests).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from agent_utilities.core.contextual_model import (
    _compiled_evidence_and_bundle,
    use_context_compiler_engine,
)
from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.ontology.permissioning import (
    clear_markings,
    use_marking_authority,
)
from agent_utilities.models.company_brain import ActorType, DataClassification, NodeACL
from agent_utilities.security.brain_context import ActorContext

logger = logging.getLogger(__name__)


class _FakeMarkingStore:
    @staticmethod
    def execute(_query, _params):
        return []


@pytest.fixture(autouse=True)
def _clean_state():
    reset_company_brain()
    clear_markings()
    with use_marking_authority(_FakeMarkingStore()):
        yield
    reset_company_brain()
    clear_markings()


def _session() -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id="test-principal",
            actor_type=ActorType.AI_AGENT,
            roles=(),
            tenant_id="test-tenant",
            authenticated=True,
        ),
        tenant="test-tenant",
        graph="test-graph",
        scopes=frozenset({"kg:read"}),
        policy_version="policy-v1",
    )


def _user_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        parts=[SimpleNamespace(part_kind="user-prompt", content=text)]
    )


# ---------------------------------------------------------------------------
# Fixture corpus — a mix of well-sourced and bare candidates, like a real KG
# retrieval result set, so the harness measures a real selection, not a
# single-item strawman.
# ---------------------------------------------------------------------------

_FIXTURE_CORPUS: list[dict[str, object]] = [
    {
        "id": "policy-4.2",
        "content": "Refunds require manager approval per policy 4.2.",
        "source_refs": ["policy-doc:4.2"],
        "confidence": 0.9,
        "score": 1.0,
    },
    {
        "id": "policy-4.3",
        "content": "Approval must be logged in the ticket within 24 hours.",
        "source_refs": ["policy-doc:4.3"],
        "evidence_refs": ["ticket-log-schema"],
        "confidence": 0.85,
        "score": 0.92,
    },
    {
        "id": "policy-4.4",
        "content": "Escalate to a director for refunds over $500.",
        "source_refs": ["policy-doc:4.4"],
        "proof_ids": ["policy-4.2"],  # SUPPORTS policy-4.2
        "confidence": 0.8,
        "score": 0.85,
    },
    {
        "id": "faq-refund-1",
        "content": "Customers can request a refund within 30 days of purchase.",
        "source_refs": ["faq:refund-window"],
        "confidence": 0.7,
        "score": 0.7,
    },
    {
        "id": "stale-note",
        # No source_refs/evidence_refs/proof_ids at all — a bare, unsourced node,
        # exactly the kind ContextCompiler's evidence-quality axis should not
        # crowd out the well-sourced ones for.
        "content": "Someone mentioned refunds are handled by support.",
        "confidence": 0.3,
        "score": 0.5,
    },
]

_FIXTURE_QUERIES: tuple[str, ...] = (
    "What is the refund policy?",
    "How do I escalate a large refund request?",
    "What is the refund window for customers?",
)


class _FixtureCorpusEngine:
    def __init__(self, corpus: list[dict[str, object]]) -> None:
        self._corpus = corpus
        self.calls = 0

    def search_hybrid(
        self,
        query: str,
        *,
        top_k: int = 8,
        as_of: str | None = None,
        session: object | None = None,
    ) -> list[dict[str, object]]:
        del query, top_k, as_of, session
        self.calls += 1
        return [dict(node) for node in self._corpus]


def _grant_public(*node_ids: str) -> None:
    for nid in node_ids:
        get_company_brain().permissions.set_acl(
            NodeACL(node_id=nid, classification=DataClassification.PUBLIC)
        )


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


@dataclass
class GroundingMetrics:
    """Aggregate grounding-quality signals over a query set (one A/B arm)."""

    queries: int
    items: int
    citations: int
    cited_items: int
    total_source_refs: int
    proof_edges: int

    @property
    def citation_coverage(self) -> float:
        """Fraction of selected items whose citation carries >=1 source_ref."""
        return self.cited_items / self.items if self.items else 0.0

    @property
    def provenance_density(self) -> float:
        """Average (source_refs + proof-graph edges) per selected item."""
        return (
            (self.total_source_refs + self.proof_edges) / self.items
            if self.items
            else 0.0
        )


def _measure_grounding(queries: tuple[str, ...]) -> GroundingMetrics:
    """Run the REAL delegated-run compile path for each query and aggregate.

    Uses :func:`_compiled_evidence_and_bundle` — the exact function every
    execute_agent/execute_workflow model call goes through — so the A/B delta is
    measured on the real wiring, not a bespoke re-derivation of ContextCompiler's
    scoring.
    """
    items = citations = cited_items = total_refs = proof_edges = 0
    for query in queries:
        _governed, bundle = _compiled_evidence_and_bundle(
            [_user_message(query)], "ab-harness-model"
        )
        if bundle is None:
            continue  # the escape-hatch arm: no compiled evidence at all
        items += len(bundle.items)
        citations += len(bundle.citations)
        cited_items += sum(1 for c in bundle.citations if c.source_refs)
        total_refs += sum(len(c.source_refs) for c in bundle.citations)
        proof_edges += len(bundle.proof_graph)
    return GroundingMetrics(
        queries=len(queries),
        items=items,
        citations=citations,
        cited_items=cited_items,
        total_source_refs=total_refs,
        proof_edges=proof_edges,
    )


def _run_ab(
    monkeypatch: pytest.MonkeyPatch, *, compiler_enabled: bool
) -> GroundingMetrics:
    monkeypatch.setenv(
        "MODEL_CONTEXT_COMPILER_ENABLED", "true" if compiler_enabled else "false"
    )
    _grant_public(*[str(n["id"]) for n in _FIXTURE_CORPUS])
    engine = _FixtureCorpusEngine(_FIXTURE_CORPUS)
    with use_context_compiler_engine(engine), use_session(_session()):
        metrics = _measure_grounding(_FIXTURE_QUERIES)
    if compiler_enabled:
        # The ON arm must have actually retrieved — otherwise this "harness" would
        # be silently comparing OFF against OFF.
        assert engine.calls == len(_FIXTURE_QUERIES)
    else:
        assert engine.calls == 0
    return metrics


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def test_ab_grounding_quality_compiler_on_vs_off(monkeypatch, caplog) -> None:
    """W3.7 acceptance: 'A/B harness demonstrates grounding-quality delta'."""
    caplog.set_level(logging.INFO, logger=__name__)

    on_metrics = _run_ab(monkeypatch, compiler_enabled=True)
    off_metrics = _run_ab(monkeypatch, compiler_enabled=False)

    logger.info(
        "[W3.7 A/B] compiler=ON  queries=%d items=%d citation_coverage=%.3f "
        "provenance_density=%.3f",
        on_metrics.queries,
        on_metrics.items,
        on_metrics.citation_coverage,
        on_metrics.provenance_density,
    )
    logger.info(
        "[W3.7 A/B] compiler=OFF queries=%d items=%d citation_coverage=%.3f "
        "provenance_density=%.3f",
        off_metrics.queries,
        off_metrics.items,
        off_metrics.citation_coverage,
        off_metrics.provenance_density,
    )
    logger.info(
        "[W3.7 A/B] delta citation_coverage=%+.3f provenance_density=%+.3f",
        on_metrics.citation_coverage - off_metrics.citation_coverage,
        on_metrics.provenance_density - off_metrics.provenance_density,
    )

    # Numbers were genuinely logged (not just printed to stdout and lost).
    logged_text = "\n".join(caplog.messages)
    assert "compiler=ON" in logged_text
    assert "compiler=OFF" in logged_text
    assert "delta citation_coverage=" in logged_text

    # The OFF arm sends the model no evidence at all, by construction.
    assert off_metrics.items == 0
    assert off_metrics.citation_coverage == 0.0
    assert off_metrics.provenance_density == 0.0

    # The ON arm actually selected items and grounded most of them (the fixture
    # corpus has one deliberately bare/unsourced node it may still select for
    # coverage/diversity, so this is >= a majority, not a strict 100%).
    assert on_metrics.items > 0
    assert on_metrics.citation_coverage >= 0.5

    # The directional delta the acceptance criterion asks for.
    assert on_metrics.citation_coverage > off_metrics.citation_coverage
    assert on_metrics.provenance_density > off_metrics.provenance_density


def test_ab_grounding_harness_is_a_fair_comparison(monkeypatch) -> None:
    """Guards the harness itself: the ON arm must genuinely retrieve (prove the
    delta isn't an artifact of a broken fixture), and the OFF arm must genuinely
    skip retrieval (prove it isn't silently falling back to a cached ON result)."""
    on_metrics = _run_ab(monkeypatch, compiler_enabled=True)
    assert on_metrics.citations > 0

    off_metrics = _run_ab(monkeypatch, compiler_enabled=False)
    assert off_metrics.citations == 0
