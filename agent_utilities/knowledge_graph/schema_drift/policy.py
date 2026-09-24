"""The operator-declared contract evolution policy (EH-402).

Drift is contained by default: a drifted delta is quarantined and the source
checkpoint does not advance. The only way a drifted delta continues is a
DECLARED rule naming the source and the compatible classes it may absorb --
never a learned one, and never a breaking class.

Declaration (``SOURCE_CONTRACT_EVOLUTION_POLICY``, comma-separated entries)::

    container-manager-mcp=additive_nullable+additive_required,systems-manager=additive_nullable

An entry naming a breaking class (``rename_candidate``, ``type_widen``,
``removal``) or an unknown class makes the whole declaration invalid, and an
invalid declaration auto-continues nothing (every drift quarantines) while the
refusal is reported, so a typo can never widen what continues.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .classify import COMPATIBLE_CLASSES, DriftChange, DriftClass

#: The setting that carries the declaration.
POLICY_SETTING = "SOURCE_CONTRACT_EVOLUTION_POLICY"


class ContractPolicyError(ValueError):
    """The declared evolution policy is malformed or names a breaking class."""


@dataclass(frozen=True, slots=True)
class ContractEvolutionPolicy:
    """Source -> the compatible drift classes it may auto-continue."""

    auto_continue: Mapping[str, frozenset[DriftClass]] = field(default_factory=dict)
    #: Why the declaration was refused, when it was.
    refusal: str = ""

    def allows(self, source: str, changes: Iterable[DriftChange]) -> bool:
        """Whether every change is a class this source declared continuable."""
        allowed = self.auto_continue.get(source.strip().lower(), frozenset())
        return all(change.kind in allowed for change in changes)


def _parse_classes(source: str, spec: str) -> frozenset[DriftClass]:
    classes: set[DriftClass] = set()
    for name in filter(None, (part.strip().lower() for part in spec.split("+"))):
        try:
            kind = DriftClass(name)
        except ValueError as exc:
            raise ContractPolicyError(
                f"{source}: unknown drift class {name!r}"
            ) from exc
        if kind not in COMPATIBLE_CLASSES:
            raise ContractPolicyError(
                f"{source}: {kind.value} is breaking and can never auto-continue"
            )
        classes.add(kind)
    return frozenset(classes)


def parse_policy(text: str) -> ContractEvolutionPolicy:
    """Parse a declaration; raises :class:`ContractPolicyError` when invalid."""
    rules: dict[str, frozenset[DriftClass]] = {}
    for entry in filter(None, (part.strip() for part in text.split(","))):
        source, sep, spec = entry.partition("=")
        source = source.strip().lower()
        if not sep or not source:
            raise ContractPolicyError(
                f"entry {entry!r} is not <source>=<class>[+<class>]"
            )
        if source in rules:
            raise ContractPolicyError(f"{source}: declared twice")
        rules[source] = _parse_classes(source, spec)
    return ContractEvolutionPolicy(rules)


def declared_policy(text: str | None = None) -> ContractEvolutionPolicy:
    """The declared policy (from the setting unless ``text`` is given).

    An invalid declaration yields an empty policy that carries the refusal.
    """
    if text is None:
        from ...core.config import setting

        text = str(setting("SOURCE_CONTRACT_EVOLUTION_POLICY", default="") or "")
    try:
        return parse_policy(text)
    except ContractPolicyError as exc:
        return ContractEvolutionPolicy(refusal=str(exc))


__all__ = [
    "POLICY_SETTING",
    "ContractEvolutionPolicy",
    "ContractPolicyError",
    "declared_policy",
    "parse_policy",
]
