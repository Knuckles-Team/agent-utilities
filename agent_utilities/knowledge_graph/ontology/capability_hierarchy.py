#!/usr/bin/python
from __future__ import annotations

"""Ontology-native capability subsumption (CONCEPT:AU-P1-3 — X-4 ontology-driven routing).

AU-P1-3 gave capability-aware routing a filtered ANN + a durable contextual bandit,
but candidate matching was flat exact-string equality: a tool declaring
``providesCapability :DNSCapability`` was invisible to a request asking for the
broader ``:ServiceCapability`` even though the capability ontology
(``ontology_capability.ttl``) already models ``:DNSCapability rdfs:subClassOf
:ServiceCapability`` — an is-a relationship the reasoner should exploit.

This module reads that ``rdfs:subClassOf`` closure directly out of the bundled
Turtle ontology files with a small, dependency-free scan — NOT ``rdflib``. rdflib/
owlready2/pyshacl are the optional ``owl`` extra, deliberately excluded from the
serving plane (every profile, pi-tier included — see
``knowledge_graph/ontology/ontology_integrity.py``'s module docstring). Ontology
subsumption has to be available on *every* install for it to be a live routing
input rather than a dev-only nicety, so :class:`CapabilityHierarchy` never imports
rdflib: it strips anonymous ``owl:Restriction`` blocks (``[ ... ]``) and reads the
remaining named ``rdfs:subClassOf`` object list — exactly the axioms
``ontology_capability.ttl`` uses for its capability taxonomy (``EncryptedTransport``
⊑ ``TransportCapability`` ⊑ ``ServiceCapability``, etc.).

Subsumption direction: ``child rdfs:subClassOf parent`` means the child IS-A the
parent (narrower ⊑ broader). :meth:`is_subtype_of` answers "is ``candidate`` the
same as, or a narrower type than, ``required``?" — the question routing needs to
ask of a tool's *declared* capability against a request's *required* capability
type: a DNS tool (declared ``DNSCapability``) satisfies a request for the broader
``ServiceCapability``, but a tool declaring only the broad ``TransportCapability``
does NOT satisfy a request for the narrower, stronger ``EncryptedTransport``.
"""

import logging
import re
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CapabilityHierarchy",
    "get_default_hierarchy",
    "load_capability_hierarchy",
]

# Matches a top-level Turtle subject that declares an ``owl:Class`` — the anchor
# for one class definition block. Local (prefix-relative) names only: federated/
# fully-qualified subjects (``bfo:0000031`` etc.) are never capability classes in
# this ontology and are skipped.
_CLASS_DEF_RE = re.compile(r"(?m)^:(\w+)\s+a\s+(?:owl:Class\b|[^;.]*,\s*owl:Class\b)")
# Anonymous restriction blocks — ``[ a owl:Restriction ; ... ]`` — are never a
# *named* superclass, so they are stripped before the subClassOf object list is
# read. None of this ontology's restriction blocks nest a second ``[`` inside one,
# so a single non-greedy bracket match is exact, not an approximation.
_BRACKET_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)
_SUBCLASS_RE = re.compile(
    r"rdfs:subClassOf\s+([^.]*?)(?=\s*(?:;|\.\s|\.\n|\Z))", re.DOTALL
)
# ``:LocalName`` only — the negative lookbehind excludes a foreign-prefixed URI
# like ``bfo:0000031`` (where ``:`` is preceded by the ``bfo`` prefix letters, not
# whitespace/punctuation), so a BFO root parent is correctly ignored rather than
# misread as a local class named "0000031".
_LOCAL_NAME_RE = re.compile(r"(?<![\w:]):(\w+)")

_ONTOLOGY_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_ONTOLOGY_FILES: tuple[str, ...] = ("ontology_capability.ttl",)

_DEFAULT_HIERARCHY: CapabilityHierarchy | None = None


class CapabilityHierarchy:
    """In-memory ``rdfs:subClassOf`` closure over the capability ontology.

    Built once (bootstrap-only, like the CDC-maintained capability index caches
    elsewhere in AU-P1-3) from the bundled ``.ttl`` source(s); ancestor/descendant
    closures are memoized since the hierarchy is immutable after load. No network,
    no rdflib, no engine round-trip — a pure-Python scan of local files, safe to
    construct in the serving-plane hot path.
    """

    def __init__(self) -> None:
        self._parents: dict[str, set[str]] = {}
        self._children: dict[str, set[str]] = {}
        self._ancestor_cache: dict[str, frozenset[str]] = {}
        self._descendant_cache: dict[str, frozenset[str]] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_files(cls, paths: Any) -> CapabilityHierarchy:
        """Build a hierarchy from an iterable of ``.ttl`` paths.

        A missing or malformed file is skipped (logged at debug), never raised —
        a bad ontology file must never take routing down; it just means fewer
        subsumption edges are known (degrades to the pre-X-4 exact-match
        behaviour for the affected classes).
        """
        hierarchy = cls()
        for path in paths:
            p = Path(path)
            if not p.exists():
                continue
            try:
                hierarchy._parse(p.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001 — a malformed ttl never breaks routing
                logger.debug("CapabilityHierarchy: failed parsing %s: %s", p, e)
        return hierarchy

    def _parse(self, text: str) -> None:
        matches = list(_CLASS_DEF_RE.finditer(text))
        for i, match in enumerate(matches):
            name = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[start:end]
            block = _BRACKET_RE.sub(" ", block)
            self._parents.setdefault(name, set())
            for sc in _SUBCLASS_RE.finditer(block):
                for parent in _LOCAL_NAME_RE.findall(sc.group(1)):
                    if parent == name:
                        continue
                    self._parents[name].add(parent)
                    self._children.setdefault(parent, set()).add(name)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def parents_of(self, name: str) -> frozenset[str]:
        """Direct (one-level) named superclasses of ``name``."""
        return frozenset(self._parents.get(name, ()))

    def children_of(self, name: str) -> frozenset[str]:
        """Direct (one-level) named subclasses of ``name``."""
        return frozenset(self._children.get(name, ()))

    def ancestors(self, name: str) -> frozenset[str]:
        """Transitive closure of superclasses of ``name`` (excludes ``name`` itself)."""
        cached = self._ancestor_cache.get(name)
        if cached is not None:
            return cached
        seen: set[str] = set()
        frontier = list(self._parents.get(name, ()))
        while frontier:
            p = frontier.pop()
            if p in seen:
                continue
            seen.add(p)
            frontier.extend(self._parents.get(p, ()))
        result = frozenset(seen)
        self._ancestor_cache[name] = result
        return result

    def descendants(self, name: str) -> frozenset[str]:
        """Transitive closure of subclasses of ``name`` (excludes ``name`` itself).

        This is the set routing expands a *required* capability type into: any
        tool declaring one of these (narrower) types also satisfies a request for
        ``name`` itself.
        """
        cached = self._descendant_cache.get(name)
        if cached is not None:
            return cached
        seen: set[str] = set()
        frontier = list(self._children.get(name, ()))
        while frontier:
            c = frontier.pop()
            if c in seen:
                continue
            seen.add(c)
            frontier.extend(self._children.get(c, ()))
        result = frozenset(seen)
        self._descendant_cache[name] = result
        return result

    def is_subtype_of(self, candidate: str, required: str) -> bool:
        """True if ``candidate`` IS ``required``, or a (transitive) narrower subtype."""
        return candidate == required or required in self.ancestors(candidate)

    def subsumption_path(self, candidate: str, required: str) -> list[str] | None:
        """The shortest ``subClassOf`` chain from ``candidate`` up to ``required``.

        Returns ``[candidate, ..., required]`` inclusive, or ``None`` when
        ``candidate`` is not ``required`` and does not subsume-up to it. Used to
        explain WHY a candidate was eligible (X-4 explainability).
        """
        if candidate == required:
            return [candidate]
        if required not in self.ancestors(candidate):
            return None
        visited = {candidate}
        queue: deque[list[str]] = deque([[candidate]])
        while queue:
            path = queue.popleft()
            node = path[-1]
            for parent in sorted(self._parents.get(node, ())):
                if parent in visited:
                    continue
                new_path = path + [parent]
                if parent == required:
                    return new_path
                visited.add(parent)
                queue.append(new_path)
        return None  # pragma: no cover — unreachable given the ancestors() guard above

    def known_classes(self) -> frozenset[str]:
        """Every class name this hierarchy has any edge for (parent or child)."""
        names = set(self._parents) | set(self._children)
        return frozenset(names)


def load_capability_hierarchy(paths: Any = None) -> CapabilityHierarchy:
    """Build a fresh :class:`CapabilityHierarchy` from ``paths`` (default: the bundled files)."""
    if paths is None:
        paths = [_ONTOLOGY_DIR / name for name in _DEFAULT_ONTOLOGY_FILES]
    return CapabilityHierarchy.from_files(paths)


def get_default_hierarchy() -> CapabilityHierarchy:
    """The process-wide singleton hierarchy over the bundled capability ontology.

    Lazily built once per process and cached — callers that want a fresh scan
    (tests, a hot-reloaded ontology) should use :func:`load_capability_hierarchy`
    directly instead.
    """
    global _DEFAULT_HIERARCHY
    if _DEFAULT_HIERARCHY is None:
        _DEFAULT_HIERARCHY = load_capability_hierarchy()
    return _DEFAULT_HIERARCHY
