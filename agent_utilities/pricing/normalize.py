"""Model-name normalization & resolution for the pricing catalog.

CONCEPT:ECO-4.40 — Unified model pricing catalog.

Faithful Python port of agentsview ``internal/pricing/normalize.go``. The
resolution order is load-bearing for matching agent-reported model ids
(which arrive dotted, decorated, provider-qualified, or date-stamped) against
the dashed LiteLLM pricing keys. Arbitrary substring matching is deliberately
avoided so a shorter key (``gpt-5.5``) never silently misprices a distinct
longer model (``gpt-5.5-codex``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

T = TypeVar("T")

_RANKS = 3


def normalize_model_name(model: str) -> str:
    """Convert a model id's dots to dashes (``claude-opus-4.7`` -> ``...-4-7``).

    Use only as a fallback after an exact match.
    """
    return model.replace(".", "-")


def _canonicalize(s: str) -> str:
    """Strip any provider prefix (after the last ``/``), lowercase, and keep
    only ``[a-z0-9]``."""
    idx = s.rfind("/")
    if idx != -1:
        s = s[idx + 1 :]
    return "".join(ch for ch in s.lower() if ch.isascii() and ch.isalnum())


def _canonical_provider(s: str) -> str:
    """Canonicalized provider prefix (``openai/gpt-5.5`` -> ``openai``), or ``""``."""
    idx = s.rfind("/")
    if idx <= 0:
        return ""
    return "".join(ch for ch in s[:idx].lower() if ch.isascii() and ch.isalnum())


def _strip_trailing_group(s: str) -> str:
    """Remove one trailing ``(...)`` or ``[...]`` decoration.

    ``Gemini 3.5 Flash (Medium)`` -> ``Gemini 3.5 Flash``;
    ``claude-fable-5[1m]`` -> ``claude-fable-5``.
    """
    t = s.rstrip(" ")
    if t.endswith(")"):
        opener = "("
    elif t.endswith("]"):
        opener = "["
    else:
        return s
    i = t.rfind(opener)
    if i <= 0:
        return s
    return t[:i].rstrip(" ")


def _strip_trailing_date(s: str) -> str:
    """Remove a trailing ``-YYYYMMDD`` release-date suffix."""
    i = s.rfind("-")
    if i <= 0 or len(s) - i - 1 != 8:
        return s
    if not s[i + 1 :].isdigit():
        return s
    return s[:i]


def _canonical_candidates(model: str) -> list[str]:
    """Canonical forms of ``model`` to try, in decreasing specificity."""
    candidates: list[str] = []

    def add(value: str) -> None:
        c = _canonicalize(value)
        if c and c not in candidates:
            candidates.append(c)

    add(model)
    undecorated = _strip_trailing_group(model)
    add(undecorated)
    add(_strip_trailing_date(undecorated))
    return candidates


def _key_rank(model_provider: str, key_provider: str) -> int:
    """same-provider key (0), unqualified key (1), provider-qualified key for an
    unqualified model (2)."""
    if key_provider == "":
        return 1
    if key_provider == model_provider:
        return 0
    return 2


def _resolve_canonical(m: Mapping[str, T], model: str) -> tuple[T | None, bool]:
    candidates = _canonical_candidates(model)
    if not candidates:
        return None, False

    model_provider = _canonical_provider(model)
    counts = [[0] * _RANKS for _ in candidates]
    vals: list[list[T | None]] = [[None] * _RANKS for _ in candidates]

    for k, v in m.items():
        key_provider = _canonical_provider(k)
        if key_provider and model_provider and key_provider != model_provider:
            continue
        k_canon = _canonicalize(k)
        if not k_canon:
            continue
        rank = _key_rank(model_provider, key_provider)
        for i, c in enumerate(candidates):
            if k_canon == c:
                counts[i][rank] += 1
                vals[i][rank] = v
                break

    for i in range(len(candidates)):
        for r in range(_RANKS):
            if counts[i][r] == 1:
                return vals[i][r], True
            if counts[i][r] > 1:
                return None, False
    return None, False


def resolve(m: Mapping[str, T], model: str) -> tuple[T | None, bool]:
    """Look up ``model`` in ``m``: exact -> normalized -> case-insensitive ->
    canonical-with-decoration-stripped. Returns ``(value, found)``."""
    # 1. Exact match
    if model in m:
        return m[model], True
    # 2. Exact match on normalized (dotted -> dashed)
    norm = normalize_model_name(model)
    if norm != model and norm in m:
        return m[norm], True
    # 3. Case-insensitive exact match
    lower_model = model.lower()
    for k, v in m.items():
        if k.lower() == lower_model:
            return v, True
    lower_norm = norm.lower()
    for k, v in m.items():
        if k.lower() == lower_norm:
            return v, True
    # 4. Canonical match with curated decoration stripping
    return _resolve_canonical(m, model)
