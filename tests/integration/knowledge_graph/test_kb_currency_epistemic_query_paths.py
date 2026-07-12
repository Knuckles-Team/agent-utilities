#!/usr/bin/python
"""Seam 1 (CONCEPT:AU-KB-CURRENCY) — the remaining AU read surfaces adopting
``include_epistemic`` (the documented follow-ups in
``docs/architecture/epistemic-columns-currency.md``): ``GraphComputeEngine.
query_unified``, ``IntelligenceGraphEngine.uql``, and
``GraphBackend.execute`` (``EpistemicGraphBackend``). Each proves the SAME
thing the keystone facade test does — the confidence/bitemporal-window/
evidence-provenance/policy-label columns returned come from the REAL engine's
``KnowledgeSet`` resolution (via ``explain_provenance_by_ids``), not fabricated
AU-side — against a real, session-scoped ephemeral ``epistemic-graph-server``
(the ``engine_graph``/``tiny_engine`` fixtures; CONCEPT:AU-KG.memory.provides-real-ephemeral-one).

Skips (does not fail) when no real engine is reachable — ``tiny_engine``'s own
convention.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.engine]


def _seed_claim_evidence(graph: Any) -> tuple[str, str]:
    """Write a Claim + Evidence(+SUPPORTS) pair straight into the real engine.

    ``type`` (not ``node_type``) is the property key the engine's native label
    matcher (``scan_label``, ``eg-plan::exec.rs``) reads for ``Scan``/``MATCH
    (:Label)`` — the SAME key ``KnowledgeRow::kind`` resolution falls back to
    when ``node_type`` is absent (``eg-plan::knowledge.rs``), so a single
    ``type`` property drives both the plan's label match AND the epistemic
    envelope's ``kind`` column.
    """
    claim_id = f"claim-{uuid.uuid4().hex[:8]}"
    evidence_id = f"evidence-{uuid.uuid4().hex[:8]}"
    graph.add_node(
        claim_id,
        {
            "type": "Claim",
            "name": "kb-currency query-path test claim",
            "confidence": 0.83,
            "valid_from": 1_700_000_000,
            "valid_until": 1_800_000_000,
            "tx_from": 1_650_000_000,
        },
    )
    graph.add_node(evidence_id, {"type": "Evidence", "confidence": 0.95})
    graph.add_edge(evidence_id, claim_id, {"relationship_type": "SUPPORTS"})
    return claim_id, evidence_id


def _assert_currency_row(row: Any, claim_id: str, evidence_id: str) -> None:
    """Shared assertions: the confidence/bitemporal/evidence columns are the
    engine's OWN resolution for THIS exact node/edge pair, not AU-side echoes."""
    from agent_utilities.knowledge_graph.core.epistemic_row import EpistemicRow

    assert isinstance(row, EpistemicRow)
    assert row.id == claim_id
    assert row.kind == "Claim"
    assert row.confidence == pytest.approx(0.83)
    assert row.calibration == pytest.approx(0.83)
    assert row.valid_time == (1_700_000_000, 1_800_000_000)
    assert row.tx_time[0] == 1_650_000_000
    assert evidence_id in row.source_refs
    assert row.policy_labels, "engine should classify a SUPPORTS-only claim"


def test_query_unified_include_epistemic_carries_engine_envelope(
    engine_graph: Any,
) -> None:
    """``GraphComputeEngine.query_unified(..., include_epistemic=True)`` — the
    cross-modal ``[id, score]`` surface currency-upgraded (item 1 of the
    documented follow-ups)."""
    if not hasattr(engine_graph, "explain_provenance_by_ids"):
        pytest.skip(
            "installed epistemic_graph client predates explain_provenance_by_ids"
        )
    claim_id, evidence_id = _seed_claim_evidence(engine_graph)
    plan: list[dict[str, Any]] = [{"Scan": {"label": "Claim"}}]

    # Default path — byte-for-byte unaffected: plain [{"id", "score"}] rows.
    try:
        plain_rows = engine_graph.query_unified(plan)
    except Exception as exc:  # noqa: BLE001 - engine build without `query` feature
        pytest.skip(f"engine has no query_unified surface: {exc}")
    assert any(r.get("id") == claim_id for r in plain_rows)
    assert all(not hasattr(r, "confidence") for r in plain_rows)

    # Opt-in path — the Seam 1 currency upgrade.
    rows = engine_graph.query_unified(plan, include_epistemic=True)
    matches = [r for r in rows if r.id == claim_id]
    assert len(matches) == 1
    _assert_currency_row(matches[0], claim_id, evidence_id)


def test_uql_include_epistemic_carries_engine_envelope(engine_graph: Any) -> None:
    """``IntelligenceGraphEngine.uql(..., include_epistemic=True)`` — the UQL
    text-query surface currency-upgraded (item 1 of the documented follow-ups).

    ``QueryMixin.uql`` reads ``self.backend.graph`` — a minimal duck-typed host
    stands in for the full ``IntelligenceGraphEngine`` (mirroring
    ``test_unified_plan_retrieval.py``'s ``HybridRetriever.__new__`` convention),
    wired directly to the REAL ``engine_graph`` fixture's ``GraphComputeEngine``.
    """
    from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin

    if not hasattr(engine_graph, "explain_provenance_by_ids"):
        pytest.skip(
            "installed epistemic_graph client predates explain_provenance_by_ids"
        )
    claim_id, evidence_id = _seed_claim_evidence(engine_graph)

    class _Backend:
        graph = engine_graph

    class _Host:
        backend = _Backend()

    host = _Host()

    try:
        plain_rows = QueryMixin.uql(host, "MATCH (:Claim) |> LIMIT 50")
    except RuntimeError as exc:
        pytest.skip(f"engine has no UQL surface: {exc}")
    assert any(r.get("id") == claim_id for r in plain_rows)

    rows = QueryMixin.uql(host, "MATCH (:Claim) |> LIMIT 50", include_epistemic=True)
    matches = [r for r in rows if r.id == claim_id]
    assert len(matches) == 1
    _assert_currency_row(matches[0], claim_id, evidence_id)


def test_store_execute_include_epistemic_carries_engine_envelope(
    engine_graph: Any,
) -> None:
    """``EpistemicGraphBackend.execute(..., include_epistemic=True)`` — the
    unguarded/unaudited direct-backend path currency-upgraded (item 2 of the
    documented follow-ups), the ``store.execute`` counterpart of
    ``KnowledgeGraph.query(..., include_epistemic=True)``.

    Builds its OWN ``EpistemicGraphBackend`` (a fresh tenant graph, distinct
    from ``engine_graph``'s) bound to the SAME running session engine —
    ``engine_graph`` is requested purely to get the per-test engine wiring
    (``GRAPH_SERVICE_SOCKET``/``_AUTH_SECRET``) re-asserted, per its own
    docstring's convention.
    """
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )

    if not hasattr(engine_graph, "explain_provenance_by_ids"):
        pytest.skip(
            "installed epistemic_graph client predates explain_provenance_by_ids"
        )
    graph_name = f"store_execute_test_{uuid.uuid4().hex[:12]}"
    store = EpistemicGraphBackend(graph_name=graph_name)
    try:
        claim_id, evidence_id = _seed_claim_evidence(store.graph)
        cypher = f"MATCH (n:Claim) WHERE n.id = '{claim_id}' RETURN n"

        # Default path — byte-for-byte unaffected: plain dict rows.
        plain_rows = store.execute(cypher)
        assert len(plain_rows) == 1
        assert isinstance(plain_rows[0], dict)
        assert plain_rows[0]["n"]["id"] == claim_id

        # Opt-in path — the Seam 1 currency upgrade.
        rows = store.execute(cypher, include_epistemic=True)
        assert len(rows) == 1
        _assert_currency_row(rows[0], claim_id, evidence_id)
        assert rows[0].properties.get("name") == "kb-currency query-path test claim"
    finally:
        try:
            store.graph._client.tenants.delete(graph_name)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass


def test_query_cypher_include_epistemic_carries_engine_envelope(
    engine_graph: Any,
) -> None:
    """``IntelligenceGraphEngine.query_cypher(..., include_epistemic=True)`` — the
    MCP-facing Cypher surface (the ``graph_query`` tool / ``/graph/query`` REST
    twin) currency-upgraded (WS-1a: the remaining gap this file's own
    "documented follow-ups" list didn't yet cover — ``query_unified``/``uql``/
    ``store.execute`` were adopted, the MCP entry point itself was not).

    ``QueryMixin.query_cypher`` reads ``self.backend`` (needs ``.execute()`` +
    ``.graph``); a real ``EpistemicGraphBackend`` bound to a fresh tenant graph
    stands in for the full ``IntelligenceGraphEngine`` host, mirroring
    :func:`test_store_execute_include_epistemic_carries_engine_envelope`'s setup.
    """
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin

    if not hasattr(engine_graph, "explain_provenance_by_ids"):
        pytest.skip(
            "installed epistemic_graph client predates explain_provenance_by_ids"
        )
    graph_name = f"query_cypher_test_{uuid.uuid4().hex[:12]}"
    store = EpistemicGraphBackend(graph_name=graph_name)
    try:
        claim_id, evidence_id = _seed_claim_evidence(store.graph)
        cypher = f"MATCH (n:Claim) WHERE n.id = '{claim_id}' RETURN n"

        class _Host:
            backend = store
            control_backend = None

        host = _Host()

        # Default path — byte-for-byte unaffected: plain dict rows.
        plain_rows = QueryMixin.query_cypher(host, cypher)
        assert len(plain_rows) == 1
        assert plain_rows[0]["n"]["id"] == claim_id

        # Opt-in path — the Seam 1 currency upgrade.
        rows = QueryMixin.query_cypher(host, cypher, include_epistemic=True)
        assert len(rows) == 1
        _assert_currency_row(rows[0], claim_id, evidence_id)
    finally:
        try:
            store.graph._client.tenants.delete(graph_name)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass


def test_store_execute_include_epistemic_degrades_on_unsupported_backend() -> None:
    """A backend with no id-seeded epistemic primitive degrades to ``[]`` under
    ``include_epistemic=True`` — never raises, never silently returns plain
    ``dict`` rows under a ``True`` request (the documented ABC contract).

    Pure unit assertion, no engine required — exercises the SPARQL-tier
    backends' short-circuit directly.
    """
    from agent_utilities.knowledge_graph.backends.sparql.jena_fuseki_backend import (
        JenaFusekiBackend,
    )

    backend = JenaFusekiBackend.__new__(JenaFusekiBackend)
    assert backend.execute("SELECT * WHERE { ?s ?p ?o }", include_epistemic=True) == []
