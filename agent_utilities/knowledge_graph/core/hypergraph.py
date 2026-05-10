from __future__ import annotations

"""Positional Interaction Encodings (EncPI) for Inductive Knowledge Hypergraphs.

CONCEPT:KG-2.4: Inductive Knowledge Hypergraphs
Implements the EncPI algorithm from "HYPER: A Foundation Model for Inductive
Knowledge Hypergraphs" to allow zero-shot inductive generalization across
novel edge intersections based purely on their structural positional interactions.
"""


import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


class PositionalInteractionEncoder:
    """Computes dense vector embeddings for positional interactions (EncPI).

    Uses a two-layer Multi-Layer Perceptron (MLP) over concatenated sinusoidal
    encodings of relation positions (e.g., position 1 and position 2) to generate
    a dense interaction embedding for inductive hypergraph reasoning.
    """

    def __init__(
        self,
        pos_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 64,
        seed: int = 42,
    ):
        """Initializes the EncPI MLP and positional encoding params.

        Args:
            pos_dim: Dimension for each positional sinusoidal encoding.
            hidden_dim: Hidden layer size for the MLP.
            out_dim: Final output embedding size.
            seed: Random seed for deterministic MLP weights.
        """
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.seed = seed

        # We use a fixed seed so the positional interactions are deterministic
        # across agent restarts, allowing them to be stably stored in the graph DB.
        rng = np.random.default_rng(self.seed)

        # 2-layer MLP weights
        # Input size is pos_dim * 2 (concatenated pos_a and pos_b)
        in_dim = pos_dim * 2

        # He initialization for ReLU
        self.W1 = rng.standard_normal((in_dim, hidden_dim)) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden_dim)

        self.W2 = rng.standard_normal((hidden_dim, out_dim)) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(out_dim)

    def _sinusoidal_encoding(self, pos: int) -> np.ndarray:
        """Generates sinusoidal positional encoding for a given integer position.

        Similar to Transformer positional encodings, maps an integer to a dense vector.
        """
        encoding = np.zeros(self.pos_dim)
        # We handle even and odd indices
        for i in range(0, self.pos_dim, 2):
            denominator = 10000 ** (i / self.pos_dim)
            encoding[i] = math.sin(pos / denominator)
            if i + 1 < self.pos_dim:
                encoding[i + 1] = math.cos(pos / denominator)
        return encoding

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def encode_interaction(self, pos_a: int, pos_b: int) -> list[float]:
        """Encodes the interaction between two positions in a relation graph.

        Args:
            pos_a: The position of the entity in the first relation (e.g. 1 for head)
            pos_b: The position of the entity in the second relation (e.g. 2 for tail)

        Returns:
            A dense vector embedding representing this specific structural interaction.
            (Returned as a standard Python list of floats for easy DB insertion)
        """
        pa = self._sinusoidal_encoding(pos_a)
        pb = self._sinusoidal_encoding(pos_b)

        # Concatenate [pa || pb]
        x = np.concatenate([pa, pb])

        # MLP forward pass
        # Layer 1
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self._relu(z1)

        # Layer 2
        z2 = np.dot(a1, self.W2) + self.b2

        # Return as list of floats
        return z2.tolist()
