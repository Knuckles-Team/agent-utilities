"""OOLONG-Pairs: quadratic pairwise aggregation (CONCEPT:AU-AHE.rlm.long-context-benchmark).

Quadratic-complexity reasoning — the answer depends on relationships between *pairs* of records,
which a single forward pass cannot enumerate but programmatic RLM decomposition can. The synthetic
generator asks for the number of unordered record pairs sharing a category (the paper reports
near-zero base-model accuracy here, ~0.1%, vs RLM's large gains). Prefers a real export when staged.
"""

from __future__ import annotations

import random
from collections import Counter

from ..base import LongContextTask, TaskCase, register_task
from ._datasets import load_real_case

_CATEGORIES = ["alpha", "beta", "gamma", "delta"]


class OolongPairsTask(LongContextTask):
    name = "oolong_pairs"
    complexity = "O(n^2)"
    real_dataset = False

    def build(self, scale: int, *, seed: int = 0) -> TaskCase:
        real = load_real_case(self.name, index=seed)
        if real:
            return TaskCase(
                mode="real",
                grader_kind=real.get("grader_kind", "numeric"),
                **{k: real[k] for k in ("context", "question", "answer")},
            )

        rng = random.Random(seed)  # nosec B311 — deterministic synthetic benchmark data, not crypto
        records: list[str] = []
        cats: list[str] = []
        i = 0
        while sum(len(r) for r in records) < scale:
            cat = rng.choice(_CATEGORIES)
            cats.append(cat)
            records.append(
                f"node {i:06d} :: group={cat} :: weight={rng.randint(1, 100)}\n"
            )
            i += 1
        # Number of unordered pairs sharing a group: sum C(k,2) over group sizes.
        pairs = sum(k * (k - 1) // 2 for k in Counter(cats).values())
        rng.shuffle(records)
        context = "".join(records)
        return TaskCase(
            context=context,
            question=(
                "Count the number of unordered pairs of nodes that share the same group. "
                "Answer with only the integer count."
            ),
            answer=str(pairs),
            grader_kind="numeric",
            mode="synthetic",
            meta={"n_records": i, "context_chars": len(context)},
        )


register_task(OolongPairsTask())
