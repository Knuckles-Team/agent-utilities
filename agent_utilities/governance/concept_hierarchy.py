"""Concept-hierarchy grammar: flat ↔ 3-level ``NS-<pillar>.<concept>.<segment>``.

CONCEPT:OS-5.76 — concept-hierarchy standardization (B5).

This is the ONE canonical place that knows how to parse, classify, and
canonicalize a concept id. ``scripts/migrate_concepts_hierarchy.py``,
``scripts/build_concepts_yaml.py``, ``scripts/check_concepts.py`` and the
``concept`` CLI all import from here so the flat→dotted mapping can never drift.

Grammar
-------
Canonical id::

    NS-<pillar>.<concept>[.<segment>]

* ``NS``       — a namespace (project/pillar family, e.g. ``EG``/``KG``/``ECO``).
* ``<pillar>`` — the coarse grouping index inside the namespace (e.g. ``2`` of
  ``KG-2``). Legacy flat ids that carry no pillar are assigned the reserved
  **legacy pillar ``0``** (curate later via :data:`PILLAR_MAP`).
* ``<concept>``— the concept index inside the pillar.
* ``<segment>``— OPTIONAL finer subdivision minted going forward (``EG-3.31.20``).
  Absent ⇒ implicit ``.0``; never auto-assigned to legacy ids.

Two id schemes coexist and both stay valid forever:

* **Project/pillar namespaces** (:data:`PROJECT_NAMESPACES`, the cross-project
  pillars EG↔KG↔ECO↔OS↔AHE↔ORCH…) adopt the dotted grammar.
* **Package namespaces** (letters-only local registries — ``KEY``/``OKTA``/…)
  keep their ``PKG-NNN`` form untouched (recognized + passed through).

Alias strategy (NON-BREAKING)
-----------------------------
Canonicalizing never invalidates an existing marker. Every string ever used
(``EG-321``, ``KG-2.312``, ``ORCH-1.105``) remains a permanently-valid **alias**
that resolves to its canonical dotted id:

* ``EG-321``   → canonical ``EG-0.321``  (alias ``EG-321``)   — legacy pillar 0.
* ``KG-2.312`` → canonical ``KG-2.312``  (already compliant; alias == self).
* ``EG-3.31.20`` → canonical ``EG-3.31.20`` (already 3-level).

``--apply`` may later rewrite markers to the canonical form while keeping the
flat id as a recorded alias, but the resolver accepts BOTH forever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Canonical marker grammar (mirrors concept_allocator.MARKER_RE, extended to
# accept an unbounded number of dotted segments so 3-level ids parse).
# ---------------------------------------------------------------------------
#: ``CONCEPT:<NS>-<n>[.<seg>]*`` — 1, 2, or 3+ dotted segments.
HIERARCHY_MARKER_RE = re.compile(r"CONCEPT:(?P<id>[A-Z]+-\d+(?:\.[0-9A-Za-z]+)*)")

#: An id string on its own (no ``CONCEPT:`` prefix), fully anchored.
ID_RE = re.compile(r"^(?P<ns>[A-Z]+)-(?P<rest>\d+(?:\.[0-9A-Za-z]+)*)$")

#: Reserved pillar for legacy flat ids that carry no pillar of their own.
LEGACY_PILLAR = "0"

#: The cross-project pillar/framework namespaces that adopt the dotted grammar.
#: Everything else is treated as a package-local registry (passed through).
#: Curated + reviewable; a namespace is *also* treated as a project namespace at
#: runtime if it is observed carrying a 2+-segment id (see :func:`classify_namespace`).
PROJECT_NAMESPACES: frozenset[str] = frozenset(
    {
        "EG",  # epistemic-graph engine
        "KG",  # knowledge-graph pillar
        "ECO",  # ecosystem / agent-bus pillar
        "OS",  # operating-system / platform pillar
        "AHE",  # agentic-harness-engineering pillar
        "ORCH",  # orchestration pillar
        "EE",  # evaluation engine
        "ML",  # machine-learning
        "CE",  # cognitive engine
        "SAFE",  # safety
        "LGC",  # legacy/compat
        "CTX",  # context plane
        "UTIL",  # utilities pillar
    }
)

#: Curated overrides for legacy flat ids → an explicit pillar.
#: Key ``(NS, concept_index_str)`` → pillar string. Empty by default: every
#: legacy flat id lands in :data:`LEGACY_PILLAR` (``0``) and is flagged for
#: curation. Reviewers populate this before an ``--apply`` cutover to give EG
#: (and friends) real pillars.
PILLAR_MAP: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class ConceptId:
    """A parsed concept id and its canonical/alias projection."""

    raw: str
    namespace: str
    pillar: str
    concept: str
    segment: str | None
    is_project: bool
    #: Non-fatal reasons this mapping needs review (empty ⇒ clean).
    flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def canonical(self) -> str:
        """The canonical dotted id (segment shown only when present)."""
        if not self.is_project:
            # Package-scoped ids keep their existing form verbatim.
            return self.raw
        base = f"{self.namespace}-{self.pillar}.{self.concept}"
        if self.segment is not None:
            return f"{base}.{self.segment}"
        return base

    @property
    def aliases(self) -> tuple[str, ...]:
        """Every string that must keep resolving to this concept."""
        al = {self.raw, self.canonical}
        return tuple(sorted(al))

    @property
    def flat(self) -> str:
        """The historical flat id (``NS-<concept>``) for a project concept."""
        if not self.is_project:
            return self.raw
        return f"{self.namespace}-{self.concept}"

    @property
    def needs_curation(self) -> bool:
        return "legacy-pillar-0" in self.flags


def classify_namespace(
    namespace: str, *, observed_project_ns: frozenset[str] | None = None
) -> bool:
    """Return ``True`` if *namespace* is a project/pillar namespace.

    Curated :data:`PROJECT_NAMESPACES` ∪ any namespace observed carrying a
    2+-segment id in the corpus (passed in via *observed_project_ns*).
    """
    if namespace in PROJECT_NAMESPACES:
        return True
    if observed_project_ns and namespace in observed_project_ns:
        return True
    return False


def parse_concept_id(
    cid: str, *, observed_project_ns: frozenset[str] | None = None
) -> ConceptId:
    """Parse a raw concept id into its canonical hierarchy projection.

    Deterministic + reversible. Raises :class:`ValueError` for a string that is
    not a well-formed concept id at all (caller flags it as *unmappable*).
    """
    m = ID_RE.match(cid)
    if not m:
        raise ValueError(f"unparseable concept id: {cid!r}")
    ns = m.group("ns")
    segs = m.group("rest").split(".")
    is_project = classify_namespace(ns, observed_project_ns=observed_project_ns)
    flags: list[str] = []

    if not is_project:
        # Package-scoped registry (KEY-001, OKTA-1.2, …) — pass through as-is.
        pillar = segs[0]
        concept = ".".join(segs[1:]) if len(segs) > 1 else ""
        return ConceptId(
            raw=cid,
            namespace=ns,
            pillar=pillar,
            concept=concept,
            segment=None,
            is_project=False,
            flags=("package-scoped",),
        )

    if len(segs) == 1:
        # Legacy flat project id (EG-321): no pillar of its own.
        concept = segs[0]
        pillar = PILLAR_MAP.get((ns, concept))
        if pillar is None:
            pillar = LEGACY_PILLAR
            flags.append("legacy-pillar-0")
        segment = None
    elif len(segs) == 2:
        pillar, concept = segs[0], segs[1]
        segment = None
    elif len(segs) == 3:
        pillar, concept, segment = segs[0], segs[1], segs[2]
    else:
        # >3 segments — grammar only defines 3 levels.
        pillar, concept, segment = segs[0], segs[1], segs[2]
        flags.append("over-segmented")

    return ConceptId(
        raw=cid,
        namespace=ns,
        pillar=pillar,
        concept=concept,
        segment=segment,
        is_project=True,
        flags=tuple(flags),
    )


def canonicalize(cid: str, *, observed_project_ns: frozenset[str] | None = None) -> str:
    """Return the canonical dotted id for *cid* (self for already-compliant)."""
    return parse_concept_id(cid, observed_project_ns=observed_project_ns).canonical


def build_alias_index(
    ids: list[str], *, observed_project_ns: frozenset[str] | None = None
) -> dict[str, str]:
    """Map every alias (flat + canonical) → canonical id, for resolution."""
    index: dict[str, str] = {}
    for cid in ids:
        try:
            parsed = parse_concept_id(cid, observed_project_ns=observed_project_ns)
        except ValueError:
            continue
        for alias in parsed.aliases:
            index[alias] = parsed.canonical
    return index


def observed_project_namespaces(ids: list[str]) -> frozenset[str]:
    """Namespaces seen carrying a 2+-segment id ⇒ treated as project namespaces."""
    found: set[str] = set()
    for cid in ids:
        m = ID_RE.match(cid)
        if m and "." in m.group("rest"):
            found.add(m.group("ns"))
    return frozenset(found)


def derive_part_of_edges(
    parsed: list[ConceptId],
) -> list[tuple[str, str]]:
    """``(concept, pillar)`` and ``(pillar, namespace)`` partOf edges.

    Deterministic mereology: every project concept is partOf its pillar, and
    every pillar is partOf its namespace. Package concepts are partOf their
    namespace directly.
    """
    edges: set[tuple[str, str]] = set()
    for p in parsed:
        if p.is_project:
            pillar_id = f"{p.namespace}-{p.pillar}"
            edges.add((p.canonical, pillar_id))
            edges.add((pillar_id, p.namespace))
        else:
            edges.add((p.canonical, p.namespace))
    return sorted(edges)
