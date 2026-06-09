#!/usr/bin/python
from __future__ import annotations

"""Zero-LLM, pack-driven typed-edge extraction.

CONCEPT:KG-2.33 — Zero-LLM Pack-Driven Link Inference

Mirrors gbrain's ``link-inference.ts``: on every write, the active Schema Pack's
:class:`~agent_utilities.models.schema_pack.LinkInferenceRule` set runs over the
content with regex only — **no LLM call** — to materialise domain typed edges
(e.g. a ``research-state`` pack extracts ``supports`` / ``weakens`` / ``cites`` /
``uses_dataset`` edges). Deterministic, reproducible, and free.

Safety: user-supplied regular expressions are run **ReDoS-bounded**:
- input is capped at :data:`MAX_INPUT_CHARS`,
- each rule is run under a wall-clock ``timeout`` via the ``regex`` module
  (raises on catastrophic backtracking); when ``regex`` is unavailable, patterns
  with nested quantifiers are rejected at compile and stdlib ``re`` is used,
- matches per rule are capped at :data:`MAX_MATCHES_PER_RULE`.
"""


import logging
import re as _stdlib_re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_utilities.models.schema_pack import LinkInferenceRule

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 200_000
PER_PATTERN_TIMEOUT_S = 0.25
MAX_MATCHES_PER_RULE = 500

try:  # Prefer the `regex` module — it supports a hard execution timeout.
    import regex as _regex  # type: ignore

    _HAVE_REGEX = True
except ImportError:  # pragma: no cover - regex is a declared dependency
    _regex = None  # type: ignore
    _HAVE_REGEX = False

# Heuristic for catastrophic-backtracking shapes: a quantifier applied to a group
# that itself ends in a quantifier — e.g. ``(a+)+``, ``(a*)*``, ``(.+)+``, ``(\w+)*``.
_NESTED_QUANTIFIER = _stdlib_re.compile(r"\([^)]*[+*][^)]*\)[+*]")


def _compile(rule: LinkInferenceRule) -> Any | None:
    """Compile a rule's pattern, rejecting obviously unsafe stdlib patterns."""
    flags = _stdlib_re.IGNORECASE if rule.flags_ignorecase else 0
    if _HAVE_REGEX:
        try:
            rflags = _regex.IGNORECASE if rule.flags_ignorecase else 0
            return _regex.compile(rule.pattern, rflags)
        except Exception as e:
            logger.warning("link_inference: bad pattern %r: %s", rule.pattern, e)
            return None
    # stdlib fallback: refuse nested-quantifier patterns (no timeout available).
    if _NESTED_QUANTIFIER.search(rule.pattern):
        logger.warning(
            "link_inference: rejecting potentially catastrophic pattern %r "
            "(install the `regex` module for bounded execution)",
            rule.pattern,
        )
        return None
    try:
        return _stdlib_re.compile(rule.pattern, flags)
    except Exception as e:
        logger.warning("link_inference: bad pattern %r: %s", rule.pattern, e)
        return None


def _finditer_bounded(compiled: Any, text: str):
    """Yield matches under a per-rule wall-clock budget and match-count cap."""
    if _HAVE_REGEX:
        try:
            yield from compiled.finditer(text, timeout=PER_PATTERN_TIMEOUT_S)
        except TimeoutError:
            logger.warning("link_inference: pattern timed out; truncating matches")
        return
    # stdlib: no per-call timeout, but check elapsed between yields as a backstop.
    start = time.monotonic()
    for m in compiled.finditer(text):
        if time.monotonic() - start > PER_PATTERN_TIMEOUT_S:
            logger.warning("link_inference: time budget exceeded; truncating matches")
            return
        yield m


def _resolve_slot(slot: str, doc_id: str, match: Any) -> str | None:
    """Resolve a rule ``source``/``target`` slot to a concrete id/name."""
    if slot in ("doc", "self"):
        return doc_id
    if slot.startswith("group:"):
        try:
            grp = int(slot.split(":", 1)[1])
            value = match.group(grp)
            return value.strip() if value else None
        except (IndexError, ValueError):
            return None
    return None


def infer_links(
    content: str, source_id: str, rules: list[LinkInferenceRule]
) -> list[Any]:
    """Extract typed edges from ``content`` using the pack's link-inference rules.

    Returns a list of ``ExtractedRelationship`` objects (imported lazily to avoid a
    circular import with ``entity_claim_extractor``). Each carries the resolved
    ``source_name``/``target_name``, the rule's ``edge_type`` as ``relationship_type``,
    and the rule's confidence. Self-loops (source == target) are dropped.
    """
    if not rules or not content:
        return []
    from .entity_claim_extractor import ExtractedRelationship

    text = content[:MAX_INPUT_CHARS]
    out: list[Any] = []
    for rule in rules:
        compiled = _compile(rule)
        if compiled is None:
            continue
        count = 0
        for match in _finditer_bounded(compiled, text):
            src = _resolve_slot(rule.source, source_id, match)
            tgt = _resolve_slot(rule.target, source_id, match)
            if not src or not tgt or src == tgt:
                continue
            out.append(
                ExtractedRelationship(
                    source_name=src,
                    target_name=tgt,
                    relationship_type=rule.edge_type,
                    confidence=rule.confidence,
                )
            )
            count += 1
            if count >= MAX_MATCHES_PER_RULE:
                logger.debug(
                    "link_inference: rule %r hit match cap (%d)",
                    rule.edge_type,
                    MAX_MATCHES_PER_RULE,
                )
                break
    return out


__all__ = [
    "infer_links",
    "MAX_INPUT_CHARS",
    "PER_PATTERN_TIMEOUT_S",
    "MAX_MATCHES_PER_RULE",
]
