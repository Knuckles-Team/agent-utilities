"""The feature schema each decision point reads, as EG ``FeatureSchemaBody`` values.

These bodies are what an operator publishes (as ``FeatureSchema`` components
with id :attr:`DecisionPoint.schema_component_id`, encoded by EG's
``epistemic_graph.decision_stat.body_attributes_for_publish``) to bind a
point; a head is fitted and evaluated against the same digest. Keys match the
``numbers``/``texts`` each consumer declares on its options, so a schema and
its consumer cannot drift apart silently: an absent fact with
``missing=abstain`` makes EG abstain NAMING the fact (``UnknownFact``), never
read it as zero.

Dicts list keys in Rust field order: EG digests serde's field-order JSON.
"""

from __future__ import annotations

from typing import Any

FEATURE_SCHEMA_VERSION = 1
_ABSTAIN = {"missing": "abstain"}


def _number(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "kind": {"feature": "number", "key": name},
        "missing": _ABSTAIN,
    }


def _text(name: str, key: str, param: str) -> dict[str, Any]:
    return {
        "name": name,
        "kind": {"feature": "text_bm25", "key": key, "param": param},
        "missing": _ABSTAIN,
    }


#: question id -> the features its consumer declares (numbers by key, texts by key).
_FEATURES: dict[str, list[dict[str, Any]]] = {
    "au.retrieval.plan": [
        _number("threshold"),
        _number("passes"),
        _number("requested"),
    ],
    "au.ingestion.lane": [_number("classifier"), _number("llm_passes")],
    "au.enrichment.schedule": [_number("declared_cost"), _number("expected_yield")],
    "au.entity.same_as": [_number("similarity"), _number("wikidata_match")],
    "au.schema.mapping": [_text("label", "label", "column")],
    "au.tms.contradiction": [_number("confidence"), _number("support")],
    "au.route.choice": [_number("prior"), _number("reward")],
    "au.route.model": [_number("tier_rank"), _number("heuristic")],
    "au.tool.risk": [_number("sensitive"), _number("rule_verdict")],
    "au.route.cost": [_number("declared_cost"), _number("l5.observed_cost")],
    "au.connector.triage": [_number("severity_rank"), _number("heuristic")],
    "au.connector.tool": [_number("heuristic")],
    "au.connector.writeback": [_number("heuristic")],
}


def feature_schema_body(question_id: str) -> dict[str, Any]:
    """The ``FeatureSchemaBody`` for ``question_id`` (``KeyError`` names it)."""
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "features": list(_FEATURES[question_id]),
    }


def schema_questions() -> list[str]:
    """Every question id that has a schema, sorted."""
    return sorted(_FEATURES)


__all__ = ["FEATURE_SCHEMA_VERSION", "feature_schema_body", "schema_questions"]
