"""The single resolver for EG's committed-GraphSchema SHACL validator (EH-385)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.core.committed_shacl import (
    CommittedShaclUnavailable,
    committed_shacl_authority,
    shacl_violation_summary,
    validate_committed,
)
from tests.committed_shacl_fakes import (
    CommittedShaclValidator,
    shacl_report,
    shacl_result,
)


@pytest.mark.spec("AU-QUAL-R003")
@pytest.mark.parametrize("attr", ["graph_compute", "graph", "compute"])
def test_resolves_the_wrapped_graph_compute(attr: str) -> None:
    inner = SimpleNamespace(shacl_validate_committed=CommittedShaclValidator())
    assert committed_shacl_authority(SimpleNamespace(**{attr: inner})) is inner


@pytest.mark.spec("AU-QUAL-R003")
def test_prefers_the_handle_itself_and_returns_none_without_a_validator() -> None:
    own = SimpleNamespace(shacl_validate_committed=CommittedShaclValidator())
    assert committed_shacl_authority(own) is own
    assert committed_shacl_authority(None) is None
    assert committed_shacl_authority(SimpleNamespace(graph=object())) is None
    assert (
        committed_shacl_authority(SimpleNamespace(shacl_validate_committed=None))
        is None
    )


@pytest.mark.spec("AU-QUAL-R003")
def test_an_intelligence_engine_shaped_wrapper_reaches_the_validator() -> None:
    """The production handle: the engine wraps GraphComputeEngine as graph_compute.

    Before EH-385, PromotionGovernanceValidator probed the wrapper itself for
    the method and so held every proposal as "authority unavailable".
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    assert not hasattr(IntelligenceGraphEngine, "shacl_validate_committed")
    assert callable(GraphComputeEngine.shacl_validate_committed)


def test_validate_committed_passes_only_the_data_graph() -> None:
    validator = CommittedShaclValidator()
    report = validate_committed(
        SimpleNamespace(
            graph_compute=SimpleNamespace(shacl_validate_committed=validator)
        ),
        "<a> <b> <c> .",
    )
    assert report.conforms is True
    assert validator.validations == ["<a> <b> <c> ."]


def test_validate_committed_raises_when_unreachable() -> None:
    with pytest.raises(CommittedShaclUnavailable):
        validate_committed(SimpleNamespace(), "")


def test_violation_summary_is_bounded_and_deduplicated() -> None:
    results = tuple(shacl_result(focus_node=f"n{i % 7}") for i in range(20))
    summary = shacl_violation_summary(shacl_report(conforms=False, results=results))
    assert summary.count("n") == 5
    assert summary.endswith("(+15 more)")
    assert shacl_violation_summary(shacl_report(conforms=False)) == (
        "no violation detail reported"
    )
