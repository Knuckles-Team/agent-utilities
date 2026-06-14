"""S-NIAH: single needle-in-a-haystack (CONCEPT:AHE-3.32).

O(1) retrieval — a unique fact is buried in a large filler context and must be recovered. Purely
synthetic and self-contained, so it doubles as the 10M-token / "two-orders-beyond-context" stress
vehicle: turning ``scale`` up to ~40M chars (≈10M tokens) exercises the RLM's external-variable
navigation at extreme length with no external dataset.
"""

from __future__ import annotations

import random

from ..base import LongContextTask, TaskCase, register_task

_FILLER = (
    "The quarterly logistics review noted nominal throughput across all regional depots. "
    "Routine inventory reconciliation proceeded without material variance. "
)


def _haystack(n_chars: int, rng: random.Random) -> str:
    reps = max(1, n_chars // len(_FILLER) + 1)
    # Number the lines so the filler is non-degenerate and chunkable.
    lines = [f"[{i:08d}] {_FILLER}" for i in range(reps)]
    return "".join(lines)[:n_chars]


class SNiahTask(LongContextTask):
    name = "s_niah"
    complexity = "O(1)"
    real_dataset = False

    def build(self, scale: int, *, seed: int = 0) -> TaskCase:
        rng = random.Random(seed)
        tag = f"depot-{rng.randint(1000, 9999)}"
        code = f"{rng.randint(0, 9999999):07d}"
        needle = f"\nIMPORTANT RECORD: The secret passphrase for {tag} is {code}.\n"
        hay = _haystack(max(len(needle) + 200, scale), rng)
        # Insert the needle at a pseudo-random interior position (avoid the very edges).
        pos = rng.randint(len(hay) // 10, max(len(hay) // 10 + 1, 9 * len(hay) // 10))
        context = hay[:pos] + needle + hay[pos:]
        return TaskCase(
            context=context,
            question=f"What is the secret passphrase for {tag}? Answer with only the passphrase.",
            answer=code,
            grader_kind="substring",
            mode="synthetic",
            meta={"tag": tag, "needle_pos": pos, "context_chars": len(context)},
        )


register_task(SNiahTask())
