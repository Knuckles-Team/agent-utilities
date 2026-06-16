"""OOLONG: semantic labeling + linear aggregation (CONCEPT:AHE-3.32).

Linear-complexity long-context reasoning — every record must be read and classified, then
aggregated. The synthetic generator scatters labeled transaction records through the context and
asks for a count of one label (the paper's OOLONG measures exactly this read-all-then-aggregate
behavior, where vanilla long-context models collapse). Prefers a real OOLONG export when staged.
"""

from __future__ import annotations

import random

from ..base import LongContextTask, TaskCase, register_task
from ._datasets import load_real_case

_CATEGORIES = ["refund", "dispute", "upgrade", "cancellation", "inquiry", "complaint"]


class OolongTask(LongContextTask):
    name = "oolong"
    complexity = "O(n)"
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
        target = rng.choice(_CATEGORIES)
        records: list[str] = []
        count = 0
        i = 0
        # Emit records until the context reaches ~scale chars.
        while sum(len(r) for r in records) < scale:
            cat = rng.choice(_CATEGORIES)
            if cat == target:
                count += 1
            records.append(
                f"Ticket {i:06d} | customer={rng.randint(10000, 99999)} | "
                f"category={cat} | priority={rng.choice(['low', 'med', 'high'])}\n"
            )
            i += 1
        rng.shuffle(records)
        context = "".join(records)
        return TaskCase(
            context=context,
            question=(
                f"Count how many tickets have category={target}. "
                f"Answer with only the integer count."
            ),
            answer=str(count),
            grader_kind="numeric",
            mode="synthetic",
            meta={"target": target, "n_records": i, "context_chars": len(context)},
        )


register_task(OolongTask())
