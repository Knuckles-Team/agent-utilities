"""CONCEPT:AHE-3.22 — SWE-bench instance corpus.

A thin, framework-agnostic loader for SWE-bench(-style) instances: repo + base commit + problem
statement + the gold ``test_patch`` and the FAIL_TO_PASS / PASS_TO_PASS test selectors that
define "resolved". Persists through the general :class:`~agent_utilities.harness.eval_corpus.EvalCorpus`
(graph-first, memory fallback) so instances live in the same place as every other eval case —
we do NOT reuse the finance-shaped ``BacktestHarness`` schema, which would be a forced fit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SweBenchInstance:
    """One SWE-bench(-style) task. Field names follow the SWE-bench dataset convention."""

    instance_id: str
    repo: str  # "owner/name" (informational)
    base_commit: str
    problem_statement: str
    fail_to_pass: list[str] = field(default_factory=list)  # must pass AFTER the fix
    pass_to_pass: list[str] = field(
        default_factory=list
    )  # must still pass AFTER the fix
    test_patch: str = ""  # gold diff that introduces/updates the tests
    repo_url: str = (
        ""  # clone source; empty => instance must be provisioned another way
    )
    setup_commands: list[str] = field(default_factory=list)  # e.g. ["pip install -e ."]
    image: str = ""  # optional per-instance base image (SWE-bench publishes these)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SweBenchInstance:
        def _split(v: Any) -> list[str]:
            if isinstance(v, str):
                # SWE-bench stores these as JSON-encoded lists
                try:
                    parsed = json.loads(v)
                    return [str(x) for x in parsed] if isinstance(parsed, list) else [v]
                except Exception:  # noqa: BLE001
                    return [x for x in v.split() if x]
            if isinstance(v, list):
                return [str(x) for x in v]
            return []

        return SweBenchInstance(
            instance_id=str(d.get("instance_id") or d.get("id") or ""),
            repo=str(d.get("repo", "")),
            base_commit=str(d.get("base_commit", "")),
            problem_statement=str(d.get("problem_statement", "")),
            fail_to_pass=_split(d.get("FAIL_TO_PASS") or d.get("fail_to_pass")),
            pass_to_pass=_split(d.get("PASS_TO_PASS") or d.get("pass_to_pass")),
            test_patch=str(d.get("test_patch", "")),
            repo_url=str(d.get("repo_url", "")),
            setup_commands=list(d.get("setup_commands", []) or []),
            image=str(d.get("image", "")),
        )


def load_instances(data: list[dict[str, Any]]) -> list[SweBenchInstance]:
    """Build instances from a list of dataset rows (SWE-bench Lite/Verified JSON)."""
    return [SweBenchInstance.from_dict(d) for d in data]


def persist_to_corpus(instances: list[SweBenchInstance], corpus: Any) -> list[str]:
    """Store instances in an :class:`EvalCorpus` (the problem statement is the query).

    Returns the corpus case ids. The full instance rides on ``metadata`` so a scorer can
    reconstruct it.
    """
    ids: list[str] = []
    for inst in instances:
        ids.append(
            corpus.add_case(
                query=inst.problem_statement,
                expected_output="resolved",
                tags=["swebench", inst.repo],
                reason=inst.instance_id,
                metadata=asdict(inst),
            )
        )
    return ids
