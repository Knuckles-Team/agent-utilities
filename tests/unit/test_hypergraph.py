"""CONCEPT:KG-2.4"""

import pytest
import numpy as np

from agent_utilities.knowledge_graph.core.hypergraph import PositionalInteractionEncoder


def test_positional_interaction_encoder_initialization():
    """Test that EncPI initializes properly with seeded weights."""
    enc = PositionalInteractionEncoder(pos_dim=16, hidden_dim=32, out_dim=16, seed=42)
    assert enc.W1.shape == (32, 32)
    assert enc.W2.shape == (32, 16)

    # Check deterministic behavior
    enc2 = PositionalInteractionEncoder(pos_dim=16, hidden_dim=32, out_dim=16, seed=42)
    np.testing.assert_array_equal(enc.W1, enc2.W1)


def test_sinusoidal_encoding():
    """Test sinusoidal positional encoding generation."""
    enc = PositionalInteractionEncoder(pos_dim=4)
    pos_encoding = enc._sinusoidal_encoding(1)

    assert pos_encoding.shape == (4,)
    # Position 1, index 0 -> sin(1 / 10000^0) = sin(1)
    assert np.isclose(pos_encoding[0], np.sin(1))
    # Position 1, index 1 -> cos(1 / 10000^0) = cos(1)
    assert np.isclose(pos_encoding[1], np.cos(1))


def test_encode_interaction_shape_and_type():
    """Test that encode_interaction returns correct format and shape."""
    enc = PositionalInteractionEncoder(pos_dim=16, hidden_dim=32, out_dim=16)
    result = enc.encode_interaction(1, 2)

    assert isinstance(result, list)
    assert len(result) == 16
    assert all(isinstance(x, float) for x in result)


def test_encode_interaction_determinism():
    """Test that encoding the same interaction yields the same vector, but different interactions differ."""
    enc = PositionalInteractionEncoder(pos_dim=16, out_dim=16)

    vec1 = enc.encode_interaction(1, 2)
    vec2 = enc.encode_interaction(1, 2)

    # Same positions should yield exactly the same embedding
    assert vec1 == vec2

    # Different positions should yield different embeddings
    vec3 = enc.encode_interaction(2, 1)
    assert vec1 != vec3
