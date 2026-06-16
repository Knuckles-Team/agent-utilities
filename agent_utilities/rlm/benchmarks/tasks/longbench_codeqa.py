"""LongBench-v2 CodeQA: multiple-choice code-repository understanding (CONCEPT:AHE-3.32).

Multi-choice QA over a large synthetic code repository spread across many files — the model must
trace a call relationship it cannot see in a single window. The synthetic generator builds a call
graph and asks which function a target calls (one correct option among distractors). Prefers a real
LongBench-v2 CodeQA export when staged.
"""

from __future__ import annotations

import random
import string

from ..base import LongContextTask, TaskCase, register_task
from ._datasets import load_real_case


def _fn_name(rng: random.Random) -> str:
    return "fn_" + "".join(rng.choices(string.ascii_lowercase, k=6))


class LongBenchCodeQaTask(LongContextTask):
    name = "longbench_codeqa"
    complexity = "code"
    real_dataset = False

    def build(self, scale: int, *, seed: int = 0) -> TaskCase:
        real = load_real_case(self.name, index=seed)
        if real:
            return TaskCase(
                mode="real",
                grader_kind=real.get("grader_kind", "choice"),
                **{k: real[k] for k in ("context", "question", "answer")},
            )

        rng = random.Random(seed)  # nosec B311 — deterministic synthetic benchmark data, not crypto
        names = [_fn_name(rng) for _ in range(8)]
        target = names[0]
        callee = names[1]
        files: list[str] = []
        # The gold file: target calls callee.
        files.append(
            f"# file core.py\ndef {target}(x):\n    y = {callee}(x)\n    return y + 1\n\n"
        )
        files.append(f"# file util.py\ndef {callee}(x):\n    return x * 2\n\n")
        # Distractor functions across many files until ~scale chars.
        idx = 0
        while sum(len(f) for f in files) < scale:
            a, b = rng.choice(names), rng.choice(names)
            files.append(
                f"# file mod_{idx:05d}.py\ndef {a}(x):\n    return {b}(x) - 1\n\n"
            )
            idx += 1
        rng.shuffle(files)
        context = "".join(files)
        # Build 4 options; correct is the callee.
        distractors = [n for n in names if n not in (target, callee)]
        rng.shuffle(distractors)
        options = [callee] + distractors[:3]
        rng.shuffle(options)
        letters = ["A", "B", "C", "D"]
        correct_letter = letters[options.index(callee)]
        opt_text = "\n".join(f"{letters[i]}) {opt}" for i, opt in enumerate(options))
        return TaskCase(
            context=context,
            question=(
                f"Which function does {target} directly call?\n{opt_text}\n"
                f"Answer with only the option letter."
            ),
            answer=correct_letter,
            grader_kind="choice",
            mode="synthetic",
            meta={"target": target, "callee": callee, "n_files": len(files)},
        )


register_task(LongBenchCodeQaTask())
