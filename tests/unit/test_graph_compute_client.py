"""AU-SEMANTIC-R010.1: typed EG-client composition and its refusal behavior."""

from __future__ import annotations

import pytest

from agent_utilities.api.graph_compute_client import (
    GraphComputeClient,
    GraphComputeClientUnavailable,
)


class _FullEGClient:
    def compute_graph(self, *args: object, **kwargs: object) -> str:
        return "computed"

    def get_session_row(self, *args: object, **kwargs: object) -> str:
        return "row"

    def resolve_object_mapping(self, *args: object, **kwargs: object) -> str:
        return "mapping"


class _PartialEGClient:
    def compute_graph(self, *args: object, **kwargs: object) -> str:
        return "computed"


def test_for_client_refuses_none() -> None:
    with pytest.raises(GraphComputeClientUnavailable):
        GraphComputeClient.for_client(None)


def test_for_client_refuses_client_missing_required_methods() -> None:
    with pytest.raises(GraphComputeClientUnavailable) as excinfo:
        GraphComputeClient.for_client(_PartialEGClient())
    assert "get_session_row" in str(excinfo.value)
    assert "resolve_object_mapping" in str(excinfo.value)


def test_for_client_accepts_full_surface_and_delegates() -> None:
    composed = GraphComputeClient.for_client(_FullEGClient())
    assert composed.compute_graph() == "computed"
    assert composed.get_session_row() == "row"
    assert composed.resolve_object_mapping() == "mapping"
