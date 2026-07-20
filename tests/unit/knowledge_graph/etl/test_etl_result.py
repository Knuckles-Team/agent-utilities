"""Unit tests for the typed ``EtlResult`` output contract (CONCEPT:AU-KG.etl.result-contract).

Covers the koheesio-assimilated pattern (typed/validated step output, see
``reports/koheesio-etl-analysis.md`` §3.1): strict construction, explicit counts,
namespaced connector diagnostics, and typed nested ETL steps.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_utilities.knowledge_graph.etl.result import EtlResult

pytestmark = pytest.mark.concept("AU-KG.etl.result-contract")


def test_default_status_and_counts():
    result = EtlResult()
    assert result.status == "ok"
    assert result.counts == {}
    assert result.source is None


def test_connector_fields_are_namespaced_under_details():
    result = EtlResult(
        status="ok",
        counts={"nodes": 7},
        details={"instances": [{"name": "a"}]},
    )
    dumped = result.model_dump()
    assert dumped["counts"] == {"nodes": 7}
    assert dumped["details"] == {"instances": [{"name": "a"}]}


def test_unknown_top_level_fields_are_rejected():
    with pytest.raises(ValidationError):
        EtlResult.model_validate({"status": "ok", "nodes_hydrated": 4})


def test_nested_steps_use_the_same_contract():
    inbound = EtlResult(status="materialized", source="camunda", counts={"nodes": 4})
    result = EtlResult(status="ok", inbound=inbound)
    assert result.inbound == inbound
    assert result.model_dump()["inbound"]["counts"] == {"nodes": 4}


def test_counts_are_explicit():
    result = EtlResult(counts={"nodes": 1, "edges": 2})
    assert result.counts == {"nodes": 1, "edges": 2}
