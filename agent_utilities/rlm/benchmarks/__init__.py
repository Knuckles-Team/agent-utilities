"""RLM long-context benchmark harness.

CONCEPT:AHE-3.32 — Long-Context RLM Benchmark Harness and Paper-Comparison Scoreboard

A reproducible harness that runs agent-utilities' RLM and baseline scaffolds over the
long-context tasks from Zhang et al. (2025, "Recursive Language Models", arXiv:2512.24601):
S-NIAH, OOLONG, OOLONG-Pairs, BrowseComp-Plus, and LongBench-v2 CodeQA. It scores accuracy
**and** cost so we can demonstrate parity/superiority against the paper's published numbers
(``scoreboard.PAPER_RESULTS``) rather than merely claiming the mechanism exists.

Real datasets are loaded from the local research cache when present; otherwise each task falls
back to a paper-faithful synthetic generator (clearly labelled ``mode="synthetic"`` on every
result and scoreboard row — no silent substitution).
"""

from .base import BenchResult, LongContextTask, TaskCase, get_task, list_tasks
from .runner import run_benchmark
from .scoreboard import render_scoreboard

__all__ = [
    "BenchResult",
    "LongContextTask",
    "TaskCase",
    "get_task",
    "list_tasks",
    "run_benchmark",
    "render_scoreboard",
]
