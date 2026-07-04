"""Real-dataset loader for benchmark tasks (CONCEPT:AU-AHE.rlm.long-context-benchmark).

Each task prefers a real dataset dropped into ``<data_dir>/rlm_benchmarks/<name>.jsonl`` (one JSON
object per line with ``context``/``question``/``answer`` keys, optional ``grader_kind``); when the
file is absent the task uses its synthetic generator. No network access happens here — operators
stage the real corpora out-of-band, keeping the harness and its tests hermetic.
"""

from __future__ import annotations

import json
from pathlib import Path


def dataset_dir() -> Path:
    from agent_utilities.core.paths import data_dir

    return data_dir() / "rlm_benchmarks"


def load_real_case(name: str, index: int = 0) -> dict | None:
    """Return the ``index``-th record from ``<dataset_dir>/<name>.jsonl``, or ``None`` if absent."""
    path = dataset_dir() / f"{name}.jsonl"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    if not rows:
        return None
    return rows[index % len(rows)]
