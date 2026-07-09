"""Unit tests for the typed ``EtlResult`` output contract (CONCEPT:AU-KG.etl.result-contract).

Covers the koheesio-assimilated pattern (typed/validated step output, see
``reports/koheesio-etl-analysis.md`` §3.1): construction, the ``count_of`` /
``coerce`` helpers that replace the old duck-typed ``_count()``, and that
``run_etl`` / ``sync_source`` / ``ingest_connector_to_table`` all still return a
plain, backward-compatible ``dict``.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.etl.result import EtlResult

pytestmark = pytest.mark.concept("AU-KG.etl.result-contract")


def test_default_status_and_counts():
    result = EtlResult()
    assert result.status == "ok"
    assert result.counts == {}
    assert result.source is None


def test_extra_handler_fields_pass_through():
    """A raw ``_sync_*`` handler dict's extra fields survive untouched."""
    result = EtlResult.coerce(
        {"status": "ok", "nodes_hydrated": 7, "instances": [{"name": "a"}]}
    )
    dumped = result.model_dump()
    assert dumped["nodes_hydrated"] == 7
    assert dumped["instances"] == [{"name": "a"}]
    # counts derived from the legacy duck-typed key when absent
    assert dumped["counts"] == {"nodes": 7}


def test_count_of_checks_legacy_keys_in_priority_order():
    assert EtlResult.count_of({"nodes": 3, "created": 9}) == 3
    assert EtlResult.count_of({"nodes_hydrated": 4}) == 4
    assert EtlResult.count_of({"created": 2}) == 2
    assert EtlResult.count_of({"rows_written": 5}) == 5
    assert EtlResult.count_of({"unrelated": 1}) == 0
    assert EtlResult.count_of(None) == 0
    assert EtlResult.count_of("not-a-dict") == 0  # type: ignore[arg-type]


def test_coerce_defaults_only_fill_gaps():
    """``coerce`` defaults never clobber a value the handler already set."""
    result = EtlResult.coerce({"status": "materialized", "source": "camunda"}, source="other")
    assert result.source == "camunda"
    assert result.status == "materialized"


def test_coerce_explicit_counts_skip_auto_derivation():
    result = EtlResult.coerce({"nodes": 100}, counts={"nodes": 1, "edges": 2})
    assert result.counts == {"nodes": 1, "edges": 2}


def test_coerce_passes_through_non_dict():
    result = EtlResult.coerce(None)
    assert result.status == "ok"


def test_to_dict_matches_model_dump():
    result = EtlResult(status="ok", source="jira")
    assert result.to_dict() == result.model_dump()
