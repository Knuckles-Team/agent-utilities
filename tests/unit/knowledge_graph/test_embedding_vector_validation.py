"""Semantic requirement 028: explicit vector dimensions must not silently fall back."""

from unittest.mock import Mock

import pytest

from agent_utilities.knowledge_graph.enrichment import semantic


@pytest.mark.parametrize("dimension", [0, -1])
@pytest.mark.parametrize("vectors", [[], [[1, 2]]])
def test_explicit_nonpositive_dimension_is_rejected(monkeypatch, dimension, vectors):
    configured = Mock(return_value=2)
    monkeypatch.setattr(semantic, "configured_embedding_dimension", configured)
    with pytest.raises(
        RuntimeError, match="expected embedding dimension must be positive"
    ):
        semantic.validate_embedding_vectors(
            vectors, expected_count=len(vectors), expected_dimension=dimension
        )
    configured.assert_not_called()


def test_unspecified_dimension_uses_configured_dimension(monkeypatch):
    configured = Mock(return_value=2)
    monkeypatch.setattr(semantic, "configured_embedding_dimension", configured)
    assert semantic.validate_embedding_vectors([[1, "2"]], expected_count=1) == [
        [1.0, 2.0]
    ]
    configured.assert_called_once_with()


def test_explicit_positive_dimension_does_not_read_config(monkeypatch):
    configured = Mock(side_effect=AssertionError("unexpected config fallback"))
    monkeypatch.setattr(semantic, "configured_embedding_dimension", configured)
    assert semantic.validate_embedding_vectors(
        [[1, "2"]], expected_count=1, expected_dimension=2
    ) == [[1.0, 2.0]]
    configured.assert_not_called()


@pytest.mark.parametrize("dimension", [1, 3])
def test_explicit_dimension_mismatch_is_rejected(monkeypatch, dimension):
    configured = Mock(side_effect=AssertionError("unexpected config fallback"))
    monkeypatch.setattr(semantic, "configured_embedding_dimension", configured)
    with pytest.raises(RuntimeError, match="wrong vector dimension"):
        semantic.validate_embedding_vectors(
            [[1, 2]], expected_count=1, expected_dimension=dimension
        )
    configured.assert_not_called()
