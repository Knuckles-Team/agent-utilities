#!/usr/bin/python
from __future__ import annotations

"""Autonomous code-synthesis stage for promoted proposals.

CONCEPT:AHE-3.22 — autonomous single-file code-synthesis stage that generates a real diff for an attributed promoted proposal so the deployed evolution loop emits code instead of only a prose plan

The genotypic-RSI *generator*: the deployed self-evolution loop can already
promote, govern, sandbox-validate and branch a ``kind="code"`` change — but
nothing on the live path ever *emits the diff*, so every real proposal falls
back to the ``kind="sdd_plan"`` prose skeleton (``change_synthesis.py``) and a
human still writes every line. This module fills that hole: for a promoted
proposal whose target file is resolvable, it reads that file and produces a
single-file ``{path, content}`` edit, which is handed to the **unchanged**
``synthesize_change_set → validate_in_sandbox → change_publisher`` pipeline as
``extra_files``.

Safety envelope (why default-on is acceptable):

* **Single, repo-relative, existing ``.py`` file only** — never multi-file, never
  a new path; an un-resolvable proposal yields ``None`` → prose fallback, exactly
  as before (zero behaviour change for un-attributed proposals).
* The generated file is sandbox-validated (syntax + import) by the existing
  ``change_synthesis`` gate — a broken diff is ``publishable == False`` and never
  branched.
* Publication still passes the OS-5.24 ActionPolicy ``merge_promotion`` gate,
  which ships defaulting to a human approval queue — generated code cannot
  auto-merge.

The LLM call lives here (not in ``change_synthesis``, which stays generation-free).
The completion fn self-degrades to an empty string when no model is configured,
so this module never *requires* an LLM to import or run.
"""

import logging
import re
from typing import Any, Protocol, runtime_checkable

from .change_synthesis import FileChange, _proposal_field, _safe_rel_path

logger = logging.getLogger(__name__)

#: Fields a proposal may carry that name the single file to edit, in priority order.
_TARGET_FIELDS = ("target_file", "file_path", "target_path", "target_component")

_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n(.*?)\n```\s*$", re.DOTALL)


@runtime_checkable
class CodeSynthesizer(Protocol):
    """Produce the full revised content of a single target file, or ``None``."""

    def generate(
        self, *, goal: str, target_path: str, current_source: str
    ) -> str | None: ...


def _repo_root() -> str | None:
    """The target repository root (where a resolved file must exist), if known."""
    from .change_publisher import default_target_repo

    root = default_target_repo()
    return str(root) if root else None


def resolve_target_file(proposal: Any, *, repo_root: str | None = None) -> str | None:
    """The single repo-relative ``.py`` file this proposal targets, if resolvable.

    Returns ``None`` (→ prose fallback) unless the proposal names a sanitized,
    repo-relative ``.py`` path that **exists** under ``repo_root``. We never
    invent a target: an un-attributed proposal is left for the prose skeleton.
    """
    import os

    root = repo_root or _repo_root()
    if root is None:
        return None
    for field in _TARGET_FIELDS:
        raw = _proposal_field(proposal, field)
        if not raw:
            continue
        rel = _safe_rel_path(raw)
        if rel is None or not rel.endswith(".py"):
            continue
        if os.path.isfile(os.path.join(root, *rel.split("/"))):
            return rel
    return None


def _strip_code_fence(text: str) -> str:
    m = _FENCE.match(text.strip())
    return (m.group(1) if m else text).strip("\n")


class LLMCodeSynthesizer:
    """Default generator: rewrite one file with the configured chat model.

    Backed by the shared ``make_lite_llm_fn`` completion fn (CONCEPT:KG-2.8), which
    returns ``""`` when no model is reachable — so ``generate`` yields ``None`` and
    the caller falls back to prose. No LLM is required to construct this object.
    """

    def __init__(self, llm_fn: Any = None) -> None:
        self._llm_fn = llm_fn

    def _fn(self) -> Any:
        if self._llm_fn is None:
            from ..enrichment.cards import make_lite_llm_fn

            self._llm_fn = make_lite_llm_fn()
        return self._llm_fn

    def generate(
        self, *, goal: str, target_path: str, current_source: str
    ) -> str | None:
        prompt = (
            "You are improving a single Python file in the agent-utilities codebase to "
            "address the goal below. Return ONLY the complete, revised contents of the "
            "file — no prose, no diff markers, no explanation. Make the smallest change "
            "that satisfies the goal; preserve all unrelated code, imports and style.\n\n"
            f"GOAL:\n{goal.strip()}\n\n"
            f"FILE: {target_path}\n"
            "CURRENT CONTENTS:\n"
            f"{current_source}\n"
        )
        try:
            out = self._fn()(prompt)
        except Exception as exc:  # noqa: BLE001 — a generator failure ⇒ prose fallback
            logger.warning(
                "[AHE-3.22] code synthesis LLM failed for %s: %s", target_path, exc
            )
            return None
        revised = _strip_code_fence(str(out or ""))
        if not revised or revised == current_source.strip("\n"):
            return None
        return revised


def get_code_synthesizer() -> CodeSynthesizer:
    """The default code synthesizer used on the live publish path."""
    return LLMCodeSynthesizer()


def synthesize_code(
    proposal: Any,
    *,
    synthesizer: CodeSynthesizer | None = None,
    repo_root: str | None = None,
) -> list[FileChange] | None:
    """Generate the single-file ``extra_files`` for an attributed proposal.

    Returns ``None`` when the proposal has no resolvable target, no synthesizer is
    available, or generation produced nothing — every such case leaves the proposal
    untouched so ``synthesize_change_set`` emits the prose SDD skeleton as before.
    """
    import os

    root = repo_root or _repo_root()
    if root is None:
        return None
    rel = resolve_target_file(proposal, repo_root=root)
    if rel is None:
        return None
    try:
        with open(os.path.join(root, *rel.split("/")), encoding="utf-8") as fh:
            current = fh.read()
    except OSError as exc:
        logger.warning("[AHE-3.22] could not read target %s: %s", rel, exc)
        return None

    gen = synthesizer or get_code_synthesizer()
    goal = str(
        _proposal_field(proposal, "goal", "problem", "summary", "description") or ""
    ).strip()
    revised = gen.generate(goal=goal, target_path=rel, current_source=current)
    # Never branch a no-op: an empty or unchanged generation falls back to prose.
    if not revised or revised.strip("\n") == current.strip("\n"):
        return None
    logger.info("[AHE-3.22] synthesized a single-file code edit for %s", rel)
    return [FileChange(path=rel, content=revised)]
