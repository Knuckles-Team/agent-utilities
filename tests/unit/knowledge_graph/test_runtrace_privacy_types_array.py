"""D-CDX-68 regression: ``RunTrace.privacy_types`` must round-trip as a native
STRING[] on LadybugDB, not get JSON-stringified into a bare STRING.

``trace_properties`` (agent_utilities/observability/trace_ontology.py) builds
``privacy_types`` as a Python ``list[str]`` (``list(report.detected_types)``),
and the unified schema (agent_utilities/models/schema_definition.py) declares
the ``RunTrace.privacy_types`` column as ``STRING[]`` on every schema-backed
mirror. ``IntelligenceGraphEngine._prepare_node_props``/``_serialize_node``
JSON-encode any dict/list property that is NOT named in their respective
``ARRAY_FIELDS`` allowlists. ``privacy_types`` was missing from both lists, so
the property silently became a JSON string (e.g. ``'["email"]'``) instead of a
native array, and LadybugDB (Kuzu) rejects a STRING parameter bound against a
STRING[] column outright -- the RunTrace write failed and a canonical query
for it returned no row.

This test writes a real RunTrace node through the production
``trace_properties`` -> ``IntelligenceGraphEngine._upsert_node`` path against
an embedded LadybugDB (Kuzu needs no container -- just an isolated file, same
as ``tests/unit/knowledge_graph/test_ladybug_edge_binding.py``) and reads it
back through ``backend.execute`` (never the in-memory graph_compute cache) to prove
the array survives the schema-backed write path end to end.
"""

from __future__ import annotations

import pytest

from agent_utilities.observability.trace_ontology import trace_properties


def _ladybug_backend(tmp_path):
    try:
        from agent_utilities.knowledge_graph.backends import create_backend

        backend = create_backend(
            backend_type="ladybug", db_path=str(tmp_path / "runtrace_privacy.db")
        )
    except Exception as exc:  # noqa: BLE001 - driver optional in some envs
        pytest.skip(f"ladybug/kuzu unavailable: {exc}")
    if backend is None or type(backend).__name__ != "LadybugBackend":
        pytest.skip("ladybug backend not available")
    return backend


def _engine_for(backend):
    """Bind a bare ``IntelligenceGraphEngine`` to ``backend`` with no compute
    mirror -- mirrors the ``_engine`` helper in
    ``tests/unit/knowledge_graph/test_native_typed_ingestion.py``, so this
    test exercises exactly ``_upsert_node`` -> ``_prepare_node_props`` without
    requiring the separate epistemic-graph compute authority."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.__new__(IntelligenceGraphEngine)
    engine.backend = backend
    return engine


def test_runtrace_privacy_types_persists_as_native_array_on_ladybug(tmp_path):
    backend = _ladybug_backend(tmp_path)
    engine = _engine_for(backend)

    # A task string containing an email address is exactly what makes
    # PersistencePrivacyGuard populate a non-empty `privacy_types` list
    # (`report.detected_types` includes "email").
    props = trace_properties(
        run_id="run-cdx68-privacy-array",
        agent_name="privacy-array-probe",
        task="contact ops@example.com about the failing run",
        status="completed",
        timestamp="2026-08-02T00:00:00Z",
        event_sequence=1,
    )
    assert props["privacy_types"] == ["email"], (
        "fixture assumption broken: the task text must trigger email "
        f"detection for this regression to be meaningful, got {props['privacy_types']!r}"
    )

    node_id = props["run_id"]
    # This must not raise: pre-fix, LadybugBackend rejected the write because
    # `_prepare_node_props` JSON-stringified `privacy_types` into a bare
    # STRING bound against the declared STRING[] column.
    engine._upsert_node("RunTrace", node_id, {"id": node_id, **props})

    rows = backend.execute(
        "MATCH (n:RunTrace) WHERE n.id = $id RETURN n.privacy_types AS privacy_types",
        {"id": node_id},
    )
    assert rows, "RunTrace node not found after write -- the insert was silently lost"
    persisted = rows[0]["privacy_types"]

    # The defect's exact failure mode: a JSON-encoded string masquerading as
    # the array, e.g. '["email"]' instead of ["email"].
    assert not isinstance(persisted, str), (
        "privacy_types came back as a JSON string, not a native array -- "
        f"D-CDX-68 regressed: {persisted!r}"
    )
    assert list(persisted) == ["email"]


def test_outcome_evaluation_privacy_types_also_covered(tmp_path):
    """``OutcomeEvaluation.privacy_types`` shares the same STRING[] contract
    (agent_utilities/models/schema_definition.py) and the same
    ``_prepare_node_props`` seam -- prove the allowlist fix is not
    RunTrace-only."""
    backend = _ladybug_backend(tmp_path)
    engine = _engine_for(backend)

    node_id = "outcome:cdx68-privacy-array"
    engine._upsert_node(
        "OutcomeEvaluation",
        node_id,
        {
            "id": node_id,
            "node_type": "OutcomeEvaluation",
            "trace_id": "trace:cdx68",
            "status": "completed",
            "privacy_schema": "persistence-privacy-v1",
            "privacy_redactions": 1,
            "privacy_types": ["email"],
        },
    )

    rows = backend.execute(
        "MATCH (n:OutcomeEvaluation) WHERE n.id = $id "
        "RETURN n.privacy_types AS privacy_types",
        {"id": node_id},
    )
    assert rows, "OutcomeEvaluation node not found after write"
    persisted = rows[0]["privacy_types"]
    assert not isinstance(persisted, str), f"expected native array, got {persisted!r}"
    assert list(persisted) == ["email"]
