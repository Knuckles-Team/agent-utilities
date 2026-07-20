"""Ontology naming-style linter — flags interface/property names that violate
the platform's naming convention, plus a small set of common typos.

CONCEPT:AU-KG.ontology.style-lint — closes a gap analysed against Microsoft's
Ontology-Playground (open-source-libraries/Ontology-Playground): its
``scripts/style-validator.ts`` enforces PascalCase class labels / snake_case
property labels + common-typo detection as part of the CI gate that guards
community ontology contributions (``scripts/validate-rdf.ts``). This platform
already gates the *canonical* ontology library's validity/connectivity
(``scripts/check_ontology.py``: parses, no duplicate IRIs, OWL-RL closure,
SHACL loadability, import resolution, doc-index membership) and supply-chain
*integrity* (:mod:`agent_utilities.knowledge_graph.ontology.ontology_integrity`:
canonical hashing + release signing) — but nothing checks naming *style* or
flags typos in interface/property names and descriptions. This module is that
missing check, scoped to the live :class:`InterfaceRegistry` (the same
always-populated registry ``ontology_interface`` already serves).

Pure and read-only: never mutates the registry, has no engine/network
dependency, and returns structured issues rather than raising — callers (the
``ontology_interface`` MCP tool's ``lint`` action, a future pre-commit hook)
decide whether a warning/error blocks anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .interfaces import InterfaceRegistry

__all__ = ["StyleIssue", "lint_interfaces"]

_PASCAL_CASE_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")
_PROPERTY_NAME_RE = re.compile(r"^[a-z][a-zA-Z0-9_]*$")
_WORD_RE = re.compile(r"[A-Za-z']+")

# A small, curated set of common English typos — mirrors the style of
# Ontology-Playground's COMMON_TYPOS table (scripts/style-validator.ts). Grown
# deliberately (a real second false-positive earns an entry), not as a general
# spell-checker.
_COMMON_TYPOS: dict[str, str] = {
    "adress": "address",
    "acheive": "achieve",
    "accross": "across",
    "arguement": "argument",
    "calender": "calendar",
    "catagory": "category",
    "definately": "definitely",
    "enviroment": "environment",
    "existance": "existence",
    "heigth": "height",
    "independant": "independent",
    "lenght": "length",
    "neccessary": "necessary",
    "occured": "occurred",
    "occurence": "occurrence",
    "recieve": "receive",
    "refered": "referred",
    "seperate": "separate",
    "succesful": "successful",
    "teh": "the",
    "wich": "which",
    "widht": "width",
}


@dataclass(frozen=True)
class StyleIssue:
    """One naming-convention or typo finding."""

    interface: str
    label: str
    message: str
    severity: str  # "error" | "warning"

    def as_dict(self) -> dict[str, str]:
        return {
            "interface": self.interface,
            "label": self.label,
            "message": self.message,
            "severity": self.severity,
        }


def _split_words(name: str) -> list[str]:
    """Split a PascalCase/camelCase/snake_case identifier into words."""
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", name).replace("_", " ")
    return [w for w in spaced.split() if w]


def _typo_issues(text: str, *, interface: str, label: str) -> list[StyleIssue]:
    issues: list[StyleIssue] = []
    for word in _WORD_RE.findall(text):
        fix = _COMMON_TYPOS.get(word.lower())
        if fix:
            issues.append(
                StyleIssue(
                    interface=interface,
                    label=label,
                    message=f"possible typo: {word!r} -> {fix!r}",
                    severity="warning",
                )
            )
    return issues


def lint_interfaces(registry: InterfaceRegistry) -> list[StyleIssue]:
    """Lint every registered interface's name, description, and property names.

    Enforces:

    * Interface (class) names are PascalCase, e.g. ``HasProvenance``.
    * Property names are snake_case or camelCase — never PascalCase.
    * Interface/property names and descriptions are checked against a small
      curated typo table (warnings, never errors).
    """
    issues: list[StyleIssue] = []
    for iface in registry.list_interfaces():
        if not _PASCAL_CASE_RE.match(iface.name):
            issues.append(
                StyleIssue(
                    interface=iface.name,
                    label=iface.name,
                    message="interface names should be PascalCase (e.g. 'HasProvenance')",
                    severity="error",
                )
            )
        issues.extend(
            _typo_issues(" ".join(_split_words(iface.name)), interface=iface.name, label=iface.name)
        )
        if iface.description:
            issues.extend(
                _typo_issues(iface.description, interface=iface.name, label=f"{iface.name}.description")
            )

        for prop in iface.properties:
            label = f"{iface.name}.{prop.name}"
            if not _PROPERTY_NAME_RE.match(prop.name):
                issues.append(
                    StyleIssue(
                        interface=iface.name,
                        label=label,
                        message="property names should be snake_case or camelCase, not PascalCase",
                        severity="error",
                    )
                )
            issues.extend(
                _typo_issues(" ".join(_split_words(prop.name)), interface=iface.name, label=label)
            )
            if prop.description:
                issues.extend(
                    _typo_issues(prop.description, interface=iface.name, label=f"{label}.description")
                )
    return issues
