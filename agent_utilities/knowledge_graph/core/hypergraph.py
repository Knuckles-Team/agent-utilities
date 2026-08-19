from __future__ import annotations

"""Positional Interaction Encodings (EncPI) for Inductive Knowledge Hypergraphs.

CONCEPT:AU-KG.compute.inductive-knowledge-hypergraphs: Inductive Knowledge Hypergraphs
Implements the EncPI algorithm from "HYPER: A Foundation Model for Inductive
Knowledge Hypergraphs" to allow zero-shot inductive generalization across
novel edge intersections based purely on their structural positional interactions.
"""


import logging
import math

from agent_utilities.numeric import NDArray, xp

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
        rng = xp.random.default_rng(self.seed)

        # 2-layer MLP weights
        # Input size is pos_dim * 2 (concatenated pos_a and pos_b)
        in_dim = pos_dim * 2

        # He initialization for ReLU
        scale_1 = math.sqrt(2.0 / in_dim)
        self.W1 = [
            [value * scale_1 for value in row]
            for row in rng.standard_normal((in_dim, hidden_dim))
        ]
        self.b1 = [0.0] * hidden_dim

        scale_2 = math.sqrt(2.0 / hidden_dim)
        self.W2 = [
            [value * scale_2 for value in row]
            for row in rng.standard_normal((hidden_dim, out_dim))
        ]
        self.b2 = [0.0] * out_dim

    def _sinusoidal_encoding(self, pos: int) -> NDArray:
        """Generates sinusoidal positional encoding for a given integer position.

        Similar to Transformer positional encodings, maps an integer to a dense vector.
        """
        encoding = [0.0] * self.pos_dim
        # We handle even and odd indices
        for i in range(0, self.pos_dim, 2):
            denominator = 10000 ** (i / self.pos_dim)
            encoding[i] = math.sin(pos / denominator)
            if i + 1 < self.pos_dim:
                encoding[i + 1] = math.cos(pos / denominator)
        return encoding

    def _relu(self, x: NDArray) -> NDArray:
        return [max(0.0, float(value)) for value in x]

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
        x = pa + pb

        # MLP forward pass
        # Layer 1
        z1 = xp.matmul([x], self.W1)[0]
        z1 = [value + bias for value, bias in zip(z1, self.b1, strict=True)]
        a1 = self._relu(z1)

        # Layer 2
        z2 = xp.matmul([a1], self.W2)[0]
        z2 = [value + bias for value, bias in zip(z2, self.b2, strict=True)]

        # Return as list of floats
        return [float(value) for value in z2]
