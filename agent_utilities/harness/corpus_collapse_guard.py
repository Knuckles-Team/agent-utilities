#!/usr/bin/python
from __future__ import annotations

"""Model-collapse / synthetic-degeneration guard for self-generated corpora.

CONCEPT:SAFE-1.4 — a model-collapse guard for the self-generated training corpus that rejects near-duplicate or distributionally-narrowing rows and caps the synthetic-to-human fraction so recursive distillation cannot quietly degenerate

The paper (§5.5/§7.4) warns that *naive iterated training on self-generated data leads
to a plateau and even degeneration* (Shumailov 2024). AU already detects collapse in the
agent **population** (AHE-3.2 `population_drift`), but nothing watched the **corpus** the
self-improvement loop distils from. This guards it: before a search-distilled row
(OS-5.36) enters the corpus it must clear three checks —

* **novelty** — not an exact duplicate, and (when an embedding is given) far enough from
  existing rows that it does not collapse diversity;
* **distributional health** — the admitted-score distribution must not narrow to a point
  (reuses the `population_drift` Wasserstein-1 / spread kernels); and
* **provenance cap** — the synthetic-to-total fraction must stay under a cap so
  human-grounded data is never fully crowded out.

It reuses the existing collapse-detection kernels rather than inventing a new metric, and
is pure/dependency-light so the harvester and any trainer can gate on it.
"""

import logging
import math
from dataclasses import dataclass, field

from agent_utilities.graph.population_drift import population_spread

logger = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class CorpusCollapseGuard:
    """Admit-or-reject gate that keeps a self-generated corpus from degenerating.

    Args:
        synthetic_cap: Max synthetic/total fraction (1.0 = no cap).
        min_novelty: Min ``1 − max_cosine`` to an existing row (embedding mode).
        spread_floor: Score-distribution spread below which the corpus has collapsed
            to a point (checked once ``min_samples`` scores are seen).
        min_samples: Scores required before the spread check engages.
        window: Trailing rows kept for the novelty/spread checks.
    """

    synthetic_cap: float = 0.9
    min_novelty: float = 0.05
    spread_floor: float = 0.02
    min_samples: int = 10
    window: int = 500
    _seen: set[str] = field(default_factory=set)
    _embeddings: list[list[float]] = field(default_factory=list)
    _scores: list[float] = field(default_factory=list)
    _total: int = 0
    _synthetic: int = 0
    rejected: int = 0

    def admit(
        self,
        key: str,
        *,
        embedding: list[float] | None = None,
        score: float | None = None,
        synthetic: bool = True,
    ) -> tuple[bool, str]:
        """Decide whether one row may enter the corpus; record it if admitted."""
        # 1. exact duplicate ⇒ zero novelty, the simplest collapse.
        if key in self._seen:
            self.rejected += 1
            return False, "duplicate (zero novelty)"
        # 2. provenance cap — never let synthetic data fully crowd out human data.
        if synthetic and self._total > 0:
            frac = (self._synthetic + 1) / (self._total + 1)
            if frac > self.synthetic_cap:
                self.rejected += 1
                return (
                    False,
                    f"synthetic fraction {frac:.2f} > cap {self.synthetic_cap:.2f}",
                )
        # 3. embedding novelty — too close to an existing row narrows the distribution.
        if embedding is not None and self._embeddings:
            novelty = 1.0 - max(_cosine(embedding, e) for e in self._embeddings)
            if novelty < self.min_novelty:
                self.rejected += 1
                return False, f"novelty {novelty:.3f} < floor {self.min_novelty:.3f}"

        self._seen.add(key)
        self._total += 1
        if synthetic:
            self._synthetic += 1
        if embedding is not None:
            self._embeddings.append(list(embedding))
            self._embeddings = self._embeddings[-self.window :]
        if score is not None:
            self._scores.append(float(score))
            self._scores = self._scores[-self.window :]
        return True, "admitted"

    @property
    def synthetic_fraction(self) -> float:
        return self._synthetic / self._total if self._total else 0.0

    def diversity(self) -> float:
        """Dispersion of the admitted-score distribution (population_drift reuse)."""
        return population_spread(self._scores)

    def is_collapsing(self) -> bool:
        """True once the corpus has narrowed: score spread vanished or synthetic-saturated."""
        if (
            len(self._scores) >= self.min_samples
            and self.diversity() < self.spread_floor
        ):
            return True
        return self._total > 0 and self.synthetic_fraction >= self.synthetic_cap

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "total": self._total,
            "synthetic": self._synthetic,
            "synthetic_fraction": round(self.synthetic_fraction, 4),
            "rejected": self.rejected,
            "distinct": len(self._seen),
            "diversity": round(self.diversity(), 4),
            "collapsing": self.is_collapsing(),
        }
