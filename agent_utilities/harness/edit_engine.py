#!/usr/bin/python
"""Robust multi-format code-edit application engine.

CONCEPT:ORCH-1.46 — apply LLM-proposed code edits to files with graduated fuzzy
matching, malformed-edit reflection, and an optional post-edit verification gate.

Motivation
----------
The simple :func:`agent_utilities.tools.developer_tools.replace_in_file` does a
single *exact* ``str.replace(old, new, 1)``. Any whitespace drift between what the
model emitted and what is on disk makes it fail, with no recovery. Production coding
harnesses (e.g. aider) instead parse one of a few well-known edit formats and apply
them with a ladder of increasingly-forgiving matchers, then *reflect* on failures by
re-prompting the model with did-you-mean hints. This module brings that capability
natively to agent-utilities so our own Claude and every spawned coding sub-agent get
materially higher edit-success rates.

Two formats are supported, auto-detected from the text:

* **search/replace blocks** — the canonical ``<<<<<<< SEARCH`` / ``=======`` /
  ``>>>>>>> REPLACE`` fenced block, with the target filename on the line above.
* **unified diff** — standard ``--- a/f`` / ``+++ b/f`` / ``@@`` hunk diffs.

The matching ladder (search/replace blocks) is: exact → leading-whitespace-flexible
→ drop-spurious-blank-line → ``...`` elision → ``SequenceMatcher`` closest-window.
The algorithms are our own implementation; the laddered strategy is inspired by
aider's ``editblock_coder``.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher, unified_diff
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "Edit",
    "EditOutcome",
    "EditResult",
    "parse_edits",
    "apply_edits",
    "apply_with_reflection",
    "render_failures_for_reflection",
]

# Search/replace block markers — tolerant of 5-9 marker chars, like git conflict
# markers, so models that emit slightly-off fences still parse.
_HEAD = re.compile(r"^<{5,9} SEARCH>?\s*$")
_DIVIDER = re.compile(r"^={5,9}\s*$")
_UPDATED = re.compile(r"^>{5,9} REPLACE\s*$")
_FENCE = re.compile(r"^\s*(```+|~~~+)")
_CLOSEST_MATCH_THRESHOLD = 0.8


@dataclass
class Edit:
    """A single parsed edit targeting one file."""

    path: str
    search: str  # empty => create/append
    replace: str
    fmt: str = "search-replace"  # or "unified-diff"


@dataclass
class EditOutcome:
    """The result of attempting to apply one :class:`Edit`."""

    path: str
    applied: bool
    strategy: str = ""  # which matcher succeeded
    reason: str = ""  # failure reason
    hint: str = ""  # did-you-mean nearest lines on failure
    diff: str = ""  # unified diff of the applied change


@dataclass
class EditResult:
    """Aggregate result of applying a batch of edits."""

    outcomes: list[EditOutcome] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.outcomes) and all(o.applied for o in self.outcomes)

    @property
    def failures(self) -> list[EditOutcome]:
        return [o for o in self.outcomes if not o.applied]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_edits(text: str, fmt: str = "auto") -> list[Edit]:
    """Parse model output into a list of :class:`Edit`.

    Args:
        text: Raw model output containing one or more edit blocks.
        fmt: ``"auto"`` (detect), ``"search-replace"``, or ``"unified-diff"``.

    Returns:
        The parsed edits (possibly empty).

    Raises:
        ValueError: If the text *looks like* an edit format but is malformed
            (e.g. an unterminated SEARCH block) — callers use this to drive the
            reflection loop.

    """
    if fmt == "auto":
        fmt = _detect_format(text)
    if fmt == "unified-diff":
        return _parse_unified_diff(text)
    return _parse_search_replace(text)


def _detect_format(text: str) -> str:
    has_sr = any(_HEAD.match(ln) for ln in text.splitlines())
    if has_sr:
        return "search-replace"
    if re.search(r"^@@ .* @@", text, re.MULTILINE) or re.search(
        r"^--- ", text, re.MULTILINE
    ):
        return "unified-diff"
    return "search-replace"


def _strip_filename(line: str) -> str | None:
    """Extract a filename from the line preceding a SEARCH fence."""
    fn = line.strip()
    if not fn or fn == "...":
        return None
    # Drop a leading fence if the model put the name on the fence line.
    fn = re.sub(r"^(```+|~~~+)", "", fn).strip()
    fn = fn.rstrip(":").lstrip("#").strip().strip("`").strip("*").strip()
    if not fn:
        return None
    # Require something filename-ish to avoid grabbing prose.
    if "." in fn or "/" in fn:
        return fn
    return fn or None


def _parse_search_replace(text: str) -> list[Edit]:
    lines = text.splitlines(keepends=True)
    edits: list[Edit] = []
    i = 0
    current_file: str | None = None
    n = len(lines)
    while i < n:
        line = lines[i]
        if _HEAD.match(line.rstrip("\n")):
            # The filename is the most recent non-blank, non-fence line above.
            fname = current_file
            j = i - 1
            while j >= 0:
                cand = lines[j].rstrip("\n")
                if not cand.strip() or _FENCE.match(cand):
                    j -= 1
                    continue
                got = _strip_filename(cand)
                if got:
                    fname = got
                break
            if not fname:
                raise ValueError(
                    "SEARCH block is missing a filename on the line above the "
                    "opening `<<<<<<< SEARCH` fence."
                )
            current_file = fname
            search_lines: list[str] = []
            i += 1
            while i < n and not _DIVIDER.match(lines[i].rstrip("\n")):
                if _HEAD.match(lines[i].rstrip("\n")) or _UPDATED.match(
                    lines[i].rstrip("\n")
                ):
                    raise ValueError(
                        f"Expected `=======` divider in SEARCH block for {fname}."
                    )
                search_lines.append(lines[i])
                i += 1
            if i >= n:
                raise ValueError(
                    f"Unterminated SEARCH block for {fname}: missing `=======`."
                )
            i += 1  # consume divider
            replace_lines: list[str] = []
            while i < n and not _UPDATED.match(lines[i].rstrip("\n")):
                if _DIVIDER.match(lines[i].rstrip("\n")) or _HEAD.match(
                    lines[i].rstrip("\n")
                ):
                    raise ValueError(
                        f"Expected `>>>>>>> REPLACE` to close block for {fname}."
                    )
                replace_lines.append(lines[i])
                i += 1
            if i >= n:
                raise ValueError(
                    f"Unterminated REPLACE block for {fname}: missing "
                    "`>>>>>>> REPLACE`."
                )
            i += 1  # consume REPLACE marker
            edits.append(
                Edit(
                    path=fname,
                    search="".join(search_lines),
                    replace="".join(replace_lines),
                )
            )
        else:
            i += 1
    return edits


def _parse_unified_diff(text: str) -> list[Edit]:
    """Parse unified-diff hunks into per-hunk search/replace edits."""
    edits: list[Edit] = []
    cur_path: str | None = None
    lines = text.splitlines(keepends=True)
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i].rstrip("\n")
        if raw.startswith("--- "):
            # filename comes from the +++ line (the "after" path).
            if i + 1 < n and lines[i + 1].startswith("+++ "):
                plus = lines[i + 1][4:].strip()
                cur_path = re.sub(r"^b/", "", plus.split("\t")[0]).strip()
                i += 2
                continue
        if raw.startswith("@@"):
            search_lines: list[str] = []
            replace_lines: list[str] = []
            i += 1
            while i < n:
                hl = lines[i]
                tag = hl[:1]
                if hl.rstrip("\n").startswith("@@") or hl.startswith("--- "):
                    break
                body = hl[1:]
                if tag == " ":
                    search_lines.append(body)
                    replace_lines.append(body)
                elif tag == "-":
                    search_lines.append(body)
                elif tag == "+":
                    replace_lines.append(body)
                elif hl.strip() == "":
                    search_lines.append("\n")
                    replace_lines.append("\n")
                else:
                    break
                i += 1
            if cur_path is None:
                raise ValueError("Unified diff hunk found before any `+++` filename.")
            edits.append(
                Edit(
                    path=cur_path,
                    search="".join(search_lines),
                    replace="".join(replace_lines),
                    fmt="unified-diff",
                )
            )
            continue
        i += 1
    return edits


# --------------------------------------------------------------------------- #
# Matching ladder
# --------------------------------------------------------------------------- #
def _prep(content: str) -> tuple[str, list[str]]:
    if content and not content.endswith("\n"):
        content += "\n"
    return content, content.splitlines(keepends=True)


def _perfect_replace(
    whole: list[str], part: list[str], rep: list[str]
) -> str | None:
    plen = len(part)
    ptup = tuple(part)
    for i in range(len(whole) - plen + 1):
        if tuple(whole[i : i + plen]) == ptup:
            return "".join(whole[:i] + rep + whole[i + plen :])
    return None


def _match_but_for_leading_ws(whole: list[str], part: list[str]) -> str | None:
    num = len(whole)
    if not all(whole[i].lstrip() == part[i].lstrip() for i in range(num)):
        return None
    add = {
        whole[i][: len(whole[i]) - len(part[i])]
        for i in range(num)
        if whole[i].strip()
    }
    if len(add) != 1:
        return None
    return add.pop()


def _replace_flexible_ws(
    whole: list[str], part: list[str], rep: list[str]
) -> str | None:
    leading = [len(p) - len(p.lstrip()) for p in part if p.strip()] + [
        len(p) - len(p.lstrip()) for p in rep if p.strip()
    ]
    if leading and min(leading):
        cut = min(leading)
        part = [p[cut:] if p.strip() else p for p in part]
        rep = [p[cut:] if p.strip() else p for p in rep]
    plen = len(part)
    for i in range(len(whole) - plen + 1):
        add = _match_but_for_leading_ws(whole[i : i + plen], part)
        if add is None:
            continue
        new_rep = [add + r if r.strip() else r for r in rep]
        return "".join(whole[:i] + new_rep + whole[i + plen :])
    return None


def _replace_closest(
    whole: list[str], part_text: str, part: list[str], rep: list[str]
) -> str | None:
    best = 0.0
    bi = bj = -1
    scale = 0.1
    lo = max(1, math.floor(len(part) * (1 - scale)))
    hi = math.ceil(len(part) * (1 + scale)) + 1
    for length in range(lo, hi):
        for i in range(len(whole) - length + 1):
            chunk = "".join(whole[i : i + length])
            sim = SequenceMatcher(None, chunk, part_text).ratio()
            if sim > best:
                best, bi, bj = sim, i, i + length
    if best < _CLOSEST_MATCH_THRESHOLD or bi < 0:
        return None
    return "".join(whole[:bi] + rep + whole[bj:])


def _apply_one(content: str, search: str, replace: str) -> tuple[str | None, str]:
    """Apply a single search→replace to ``content``.

    Returns ``(new_content, strategy)`` or ``(None, "")`` if no matcher hit.
    """
    if not search.strip():
        # Empty search = append (or whole-file when content is empty).
        base = content if content.endswith("\n") or not content else content + "\n"
        return base + replace, "append"

    _, whole = _prep(content)
    part_text, part = _prep(search)
    _, rep = _prep(replace)

    res = _perfect_replace(whole, part, rep)
    if res is not None:
        return res, "exact"
    res = _replace_flexible_ws(whole, part, rep)
    if res is not None:
        return res, "leading-whitespace"
    # Drop a spurious leading blank line the model sometimes adds.
    if len(part) > 2 and not part[0].strip():
        res = _perfect_replace(whole, part[1:], rep)
        if res is not None:
            return res, "exact-trim-blank"
        res = _replace_flexible_ws(whole, part[1:], rep)
        if res is not None:
            return res, "leading-whitespace-trim-blank"
    res = _replace_closest(whole, part_text, part, rep)
    if res is not None:
        return res, "closest-window"
    return None, ""


def _nearest_hint(content: str, search: str, n: int = 5) -> str:
    """Find the lines in ``content`` most similar to the failed search block."""
    search_lines = [s for s in search.splitlines() if s.strip()]
    if not search_lines:
        return ""
    content_lines = content.splitlines()
    target = "\n".join(search_lines)
    best = 0.0
    best_i = -1
    win = len(search_lines)
    for i in range(max(1, len(content_lines) - win + 1)):
        chunk = "\n".join(content_lines[i : i + win])
        sim = SequenceMatcher(None, chunk, target).ratio()
        if sim > best:
            best, best_i = sim, i
    if best_i < 0 or best < 0.4:
        return ""
    lo = max(0, best_i - 1)
    hi = min(len(content_lines), best_i + win + 1)
    snippet = "\n".join(content_lines[lo:hi])
    return f"closest existing lines (similarity {best:.0%}):\n{snippet}"


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
def apply_edits(
    edits: list[Edit], root: str | Path = ".", *, dry_run: bool = False
) -> EditResult:
    """Apply parsed edits to files under ``root``.

    Edits to the same file are applied sequentially so later hunks see earlier
    changes. Each edit is independent: one failure does not abort the rest.

    Args:
        edits: Parsed edits from :func:`parse_edits`.
        root: Base directory that edit paths are resolved against.
        dry_run: When True, compute outcomes/diffs but do not write files.

    Returns:
        An :class:`EditResult` with one :class:`EditOutcome` per edit.

    """
    root = Path(root)
    result = EditResult()
    # In-memory view so multiple edits to one file compose.
    pending: dict[str, str] = {}
    changed: set[str] = set()

    for edit in edits:
        fpath = (root / edit.path).resolve()
        if edit.path in pending:
            original = pending[edit.path]
        elif fpath.exists():
            original = fpath.read_text(encoding="utf-8")
        else:
            original = ""
        new_content, strategy = _apply_one(original, edit.search, edit.replace)
        if new_content is None:
            result.outcomes.append(
                EditOutcome(
                    path=edit.path,
                    applied=False,
                    reason="SEARCH block did not match the file content.",
                    hint=_nearest_hint(original, edit.search),
                )
            )
            continue
        diff = "".join(
            unified_diff(
                original.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{edit.path}",
                tofile=f"b/{edit.path}",
                n=2,
            )
        )
        pending[edit.path] = new_content
        changed.add(edit.path)
        result.outcomes.append(
            EditOutcome(path=edit.path, applied=True, strategy=strategy, diff=diff)
        )

    if not dry_run:
        for rel, content in pending.items():
            fpath = (root / rel).resolve()
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content, encoding="utf-8")
    result.files_changed = sorted(changed)
    return result


def render_failures_for_reflection(result: EditResult) -> str:
    """Render failed edits as a corrective prompt for the model to retry."""
    parts = ["Some edits could not be applied. Fix the SEARCH blocks and resend."]
    for o in result.failures:
        block = [f"\nFile: {o.path}\nReason: {o.reason}"]
        if o.hint:
            block.append(o.hint)
        parts.append("\n".join(block))
    parts.append(
        "\nResend ONLY the corrected edits as SEARCH/REPLACE blocks. The SEARCH "
        "text must match the file exactly (copy it verbatim from the hint above)."
    )
    return "\n".join(parts)


async def apply_with_reflection(
    initial_text: str,
    reprompt: Callable[[str], Awaitable[str]],
    *,
    root: str | Path = ".",
    fmt: str = "auto",
    max_reflections: int = 3,
    verify: Callable[[EditResult], Awaitable[str | None]] | None = None,
) -> EditResult:
    """Apply edits, reflecting on failures by re-prompting the model.

    Mirrors aider's ``max_reflections`` recovery: on a malformed parse or a
    non-matching SEARCH block, the model is re-prompted with did-you-mean hints
    (up to ``max_reflections`` times). An optional ``verify`` callback runs after a
    fully-applied batch (e.g. lint/test); returning a non-empty string feeds that
    failure back into the same reflection loop.

    Args:
        initial_text: First model response containing edits.
        reprompt: Async callback taking a correction message and returning the
            model's next response.
        root: Base directory for edit paths.
        fmt: Edit format (``"auto"`` by default).
        max_reflections: Max corrective rounds (default 3).
        verify: Optional async post-apply gate returning an error string or None.

    Returns:
        The final :class:`EditResult`.

    """
    text = initial_text
    last_result = EditResult()
    for attempt in range(max_reflections + 1):
        try:
            edits = parse_edits(text, fmt=fmt)
        except ValueError as exc:
            if attempt >= max_reflections:
                last_result.outcomes.append(
                    EditOutcome(path="", applied=False, reason=str(exc))
                )
                return last_result
            text = await reprompt(
                f"The edit block was malformed: {exc}\nResend valid "
                "SEARCH/REPLACE blocks."
            )
            continue

        if not edits:
            return last_result

        last_result = apply_edits(edits, root=root)

        if last_result.failures and attempt < max_reflections:
            text = await reprompt(render_failures_for_reflection(last_result))
            continue

        if last_result.ok and verify is not None:
            verdict = await verify(last_result)
            if verdict and attempt < max_reflections:
                text = await reprompt(
                    "The edits applied but verification failed:\n"
                    f"{verdict}\nSend follow-up SEARCH/REPLACE edits to fix it."
                )
                continue

        return last_result

    return last_result
