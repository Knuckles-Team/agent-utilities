# CONCEPT:AU-KG.ingest.big-repo-structural-split - Big-repo structural-ingest split: partition a large repo's source files into K balanced, deterministic buckets keyed on a coarse path prefix so each bucket routes to its OWN per-repo-shard graph and the K buckets commit in parallel across the engine's K redb shard writers, instead of one giant repo pinning a single worker/shard for minutes.
"""Deterministic big-repo split planner (CONCEPT:AU-KG.ingest.big-repo-structural-split).

The tail problem this removes. A single huge repo (agent-utilities, epistemic-graph:
thousands of files) is ONE ``codebase`` :Task → ONE per-repo graph (``code:<repo>``,
KG-2.269) → ONE redb shard writer (EG-026). Its structural write therefore
*serialises* on one writer thread and pins one worker for many minutes (live tail:
codebase p50=36s but p95=650s / max=797s), while the other K-1 shard writers sit
idle. The K-way codebase admission floor (KG-2.279) only helps when there are K
*different* repos in flight; one big repo still hits one shard.

The fix. Partition the repo's source files into ``k`` balanced buckets and ingest
each as its own sub-task routed to a distinct graph (``code:<repo>__s<i>``), so the
K buckets hash to K *different* shard writers and commit concurrently — the same
"shard the bottleneck" move applied to a single repo's file axis.

Grain + correctness. Files are grouped by a COARSE path prefix (deepened only until
there are enough groups to balance ``k`` buckets, capped at
:data:`_MAX_SPLIT_DEPTH`), then bin-packed largest-first into the emptiest bucket.
Keeping each sub-package's files together within one bucket preserves *intra*-package
cross-file CALL/INHERIT resolution (the IndexRepository pass runs per sub-task over
its bucket); only calls that cross a bucket boundary are not resolved into edges —
a bounded, coarse-grained tradeoff. Every file is ingested exactly once across the
buckets, so the union of the K per-shard graphs is the complete repo (the read path
already fans queries across the active content-graph set, KG-2.269). The assignment
is a pure function of the file set, so a re-ingest reproduces the same buckets and
each bucket's per-file content-hash delta (KG-2.8) stays valid.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "plan_repo_split",
    "SPLIT_MIN_FILES",
    "split_graph_suffix",
]

# Only split a repo with strictly more than this many source files. Small/medium
# repos (the healthy p50) take the unchanged single-task path, so the median is
# never touched. Auto-sized constant (no knob): well above a normal package, below
# the thousands-of-files monorepos that pin a worker for minutes.
SPLIT_MIN_FILES: int = 1200

# How deep into the tree the grouping prefix may go when the top level is too
# coarse to balance the buckets (e.g. a repo where ~all files live under one
# top-level package). Capped so a deep tree can't explode into per-file groups.
_MAX_SPLIT_DEPTH: int = 3

# Aim for at least this many groups per bucket so bin-packing can balance well.
_GROUPS_PER_BUCKET: int = 3


def split_graph_suffix(index: int) -> str:
    """The per-bucket graph-name suffix (``__s<index>``) for routing (KG-2.287)."""
    return f"__s{index}"


def _group_key(rel: Path, depth: int) -> str:
    """The coarse grouping key for ``rel`` at ``depth`` path components."""
    parts = rel.parts
    if len(parts) <= 1:
        # A top-level file (no directory) groups under its own name.
        return parts[0] if parts else ""
    return "/".join(parts[: min(depth, len(parts) - 1)])


def _choose_depth(rels: list[Path], k: int) -> int:
    """Smallest depth whose grouping yields enough groups to balance ``k`` buckets."""
    target = max(k * _GROUPS_PER_BUCKET, k)
    for depth in range(1, _MAX_SPLIT_DEPTH + 1):
        groups = {_group_key(r, depth) for r in rels}
        if len(groups) >= target:
            return depth
    return _MAX_SPLIT_DEPTH


def plan_repo_split(
    repo_root: str | Path,
    files: list[Path],
    k: int,
) -> list[list[Path]]:
    """Partition ``files`` into ``k`` balanced, deterministic buckets (KG-2.287).

    Returns a list of up to ``k`` non-empty buckets (each a list of ``Path``), or a
    single bucket containing every file when a split would not help (``k <= 1``,
    too few files, or only one group). The union of the buckets is exactly the input
    file set with no duplicates, so the repo's code graph stays complete across the
    per-shard graphs. Bin-packs groups largest-first into the currently-emptiest
    bucket (a deterministic LPT schedule), so the buckets are size-balanced and the
    K shard writers do comparable work.
    """
    root = Path(repo_root)
    if k <= 1 or len(files) <= 1:
        return [list(files)] if files else []

    # Group by a coarse, deterministic path prefix.
    rels: list[tuple[Path, Path]] = []
    for f in files:
        try:
            rel = Path(f).relative_to(root)
        except ValueError:
            rel = Path(Path(f).name)
        rels.append((Path(f), rel))
    depth = _choose_depth([r for _, r in rels], k)

    groups: dict[str, list[Path]] = {}
    for absp, rel in rels:
        groups.setdefault(_group_key(rel, depth), []).append(absp)
    if len(groups) <= 1:
        # Nothing to spread across shards — keep it whole (no false fan-out).
        return [list(files)]

    k = min(k, len(groups))
    # LPT bin-packing: assign the largest groups first to the emptiest bucket.
    # Tie-break on group key for determinism.
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    buckets: list[list[Path]] = [[] for _ in range(k)]
    loads = [0] * k
    for _key, members in ordered:
        i = min(range(k), key=lambda j: (loads[j], j))
        buckets[i].extend(members)
        loads[i] += len(members)
    # Sort within each bucket for stable, reproducible task payloads.
    out = [sorted(b, key=str) for b in buckets if b]
    return out
