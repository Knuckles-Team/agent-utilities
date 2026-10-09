"""AU-SEMANTIC-R025.1: typed EG retrieval composition and its refusal behavior."""

from __future__ import annotations

import pytest

from agent_utilities.api.retrieval_client import (
    RetrievalClient,
    RetrievalClientUnavailable,
)


class _FullEGClient:
    def hybrid_search(self, *args: object, **kwargs: object) -> str:
        return "hybrid"

    def semantic_search(self, *args: object, **kwargs: object) -> str:
        return "semantic"

    def rerank(self, *args: object, **kwargs: object) -> str:
        return "reranked"


class _PartialEGClient:
    def hybrid_search(self, *args: object, **kwargs: object) -> str:
        return "hybrid"


def test_for_client_refuses_none() -> None:
    with pytest.raises(RetrievalClientUnavailable):
        RetrievalClient.for_client(None)


def test_for_client_refuses_client_missing_required_methods() -> None:
    with pytest.raises(RetrievalClientUnavailable) as excinfo:
        RetrievalClient.for_client(_PartialEGClient())
    assert "semantic_search" in str(excinfo.value)
    assert "rerank" in str(excinfo.value)


def test_for_client_accepts_full_surface_and_delegates() -> None:
    composed = RetrievalClient.for_client(_FullEGClient())
    assert composed.hybrid_search() == "hybrid"
    assert composed.semantic_search() == "semantic"
    assert composed.rerank() == "reranked"
