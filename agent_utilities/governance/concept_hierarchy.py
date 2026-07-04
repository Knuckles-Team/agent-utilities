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
from functools import lru_cache
from pathlib import Path

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


# ===========================================================================
# OKF-CIS — the unified cross-repo Concept-ID standard (CONCEPT:OS-5.77).
#
#   <SLUG>-<PILLAR>.<domain>.<concept>[.<facet>...]
#       AU-KG.ingest.entropy-dedup
#       EG-KG.storage.redb
#       DS-AHE.trainer.gpu-slot
#
# * SLUG    — 2-letter uppercase repo/project code (provenance + global uniqueness).
# * PILLAR  — the ONE closed global taxonomy of 6 (shared by every repo).
# * domain  — a curated, CLOSED sub-vocabulary per pillar (the anti-sprawl gate).
# * concept — a short semantic kebab slug (never a bare number).
# * facet   — optional deeper semantic segments.
#
# The id maps deterministically to BOTH an OKF bundle path (dots -> slashes) and
# a resolvable RDF IRI, so the file tree and the ontology share one transform.
#
# ADDITIVE: this coexists with the legacy grammar above during migration. The
# legacy matchers (``HIERARCHY_MARKER_RE``/``ID_RE`` here, plus the copies in
# ``concept_allocator``/``check_concept_governance``/``reserve_concepts_hook`` and
# the CI grep) are retired ATOMICALLY with the apply cutover — see
# ``scripts/apply_concept_migration.py``. Until then the read-only planner uses the
# legacy matchers to SCAN old markers and these helpers to EMIT new ids.
# ===========================================================================

#: The closed set of 6 ecosystem pillars — shared across ALL repos. A concept's
#: SLUG says which repo owns it; its PILLAR says which capability domain it is.
PILLARS: tuple[str, ...] = ("ORCH", "KG", "AHE", "ECO", "OS", "GBOT")

_PILLAR_ALT = "|".join(PILLARS)
#: One kebab segment: lowercase alnum words joined by hyphens (``entropy-dedup``).
_SEG = r"[a-z0-9]+(?:-[a-z0-9]+)*"

#: A bare OKF-CIS id (no ``CONCEPT:`` prefix), fully anchored.
OKF_ID_RE = re.compile(
    rf"^(?P<slug>[A-Z]{{2}})-(?P<pillar>{_PILLAR_ALT})(?P<segs>(?:\.{_SEG})+)$"
)
#: The ``CONCEPT:<id>`` marker form — the ONE canonical marker regex going
#: forward (replaces the six divergent legacy matchers at cutover).
OKF_MARKER_RE = re.compile(
    rf"CONCEPT:(?P<id>[A-Z]{{2}}-(?:{_PILLAR_ALT})(?:\.{_SEG})+)"
)

#: Ecosystem base IRIs (``http://knuckles.team/kg`` wins the two-IRI split; the
#: ``agent-utilities.dev/ontology#`` projection is reconciled onto this at cutover).
CONCEPT_IRI_BASE = "http://knuckles.team/kg/concept"
PILLAR_IRI_BASE = "http://knuckles.team/kg/pillar"
SCHEME_IRI_BASE = "http://knuckles.team/kg/scheme"


@dataclass(frozen=True)
class OkfConceptId:
    """A parsed OKF-CIS id with its OKF-path and RDF-IRI projections."""

    raw: str
    slug: str
    pillar: str
    domain: str
    concept: str
    facets: tuple[str, ...] = field(default_factory=tuple)

    @property
    def segments(self) -> tuple[str, ...]:
        """The dotted segments below the pillar: ``domain.concept[.facet...]``."""
        return (self.domain, self.concept, *self.facets)

    @property
    def canonical(self) -> str:
        """The canonical dotted id (== ``raw`` for a well-formed id)."""
        return f"{self.slug}-{self.pillar}." + ".".join(self.segments)

    @property
    def path(self) -> str:
        """OKF bundle path (dots -> slashes): ``AU/KG/ingest/entropy-dedup``."""
        return "/".join((self.slug, self.pillar, *self.segments))

    @property
    def iri(self) -> str:
        """Deterministic, resolvable concept IRI."""
        return f"{CONCEPT_IRI_BASE}/{self.path}"

    @property
    def domain_iri(self) -> str:
        return f"{CONCEPT_IRI_BASE}/{self.slug}/{self.pillar}/{self.domain}"

    @property
    def pillar_iri(self) -> str:
        """Shared pillar IRI — NO slug, so every repo's ``*-KG.*`` federates here."""
        return f"{PILLAR_IRI_BASE}/{self.pillar}"

    @property
    def scheme_iri(self) -> str:
        """The owning repo's ``skos:ConceptScheme`` IRI."""
        return f"{SCHEME_IRI_BASE}/{self.slug}"


def is_okf_id(cid: str) -> bool:
    """True iff *cid* is a well-formed OKF-CIS concept id (>=2 segments)."""
    try:
        parse_okf_id(cid)
        return True
    except ValueError:
        return False


def parse_okf_id(cid: str) -> OkfConceptId:
    """Parse an OKF-CIS id. Raises :class:`ValueError` if malformed.

    Enforces (beyond the regex) the ``>=2 segments`` rule: a 1-segment id
    (``AU-KG.ingest``) is a domain node, not a concept.
    """
    m = OKF_ID_RE.match(cid)
    if not m:
        raise ValueError(f"not a valid OKF-CIS concept id: {cid!r}")
    segs = m.group("segs").lstrip(".").split(".")
    if len(segs) < 2:
        raise ValueError(
            f"OKF-CIS id needs >=2 segments (domain.concept): {cid!r}"
        )
    domain, concept, *facets = segs
    return OkfConceptId(
        raw=cid,
        slug=m.group("slug"),
        pillar=m.group("pillar"),
        domain=domain,
        concept=concept,
        facets=tuple(facets),
    )


def concept_iri(cid: str | OkfConceptId) -> str:
    """The single IRI-minter: OKF-CIS id -> resolvable concept IRI.

    Used by both the OKF file-tree writer and the RDF generator so path and IRI
    can never drift.
    """
    parsed = cid if isinstance(cid, OkfConceptId) else parse_okf_id(cid)
    return parsed.iri


def okf_id_to_path(cid: str) -> str:
    """OKF-CIS id -> OKF bundle path (``AU-KG.ingest.x`` -> ``AU/KG/ingest/x``)."""
    return parse_okf_id(cid).path


def path_to_okf_id(path: str) -> str:
    """OKF bundle path -> OKF-CIS id (inverse of :func:`okf_id_to_path`).

    ``AU/KG/ingest/entropy-dedup`` -> ``AU-KG.ingest.entropy-dedup``. The ``.md``
    suffix, if present, is stripped first.
    """
    p = path.strip("/")
    if p.endswith(".md"):
        p = p[: -len(".md")]
    parts = p.split("/")
    if len(parts) < 4:
        raise ValueError(f"path is not a concept (needs >=4 parts): {path!r}")
    slug, pillar, *segs = parts
    cid = f"{slug}-{pillar}." + ".".join(segs)
    parse_okf_id(cid)  # validate slug/pillar/segments
    return cid


#: The closed domain vocabulary shipped alongside this module.
DOMAIN_VOCAB_PATH = Path(__file__).with_name("domain_vocab.yaml")


@lru_cache(maxsize=8)
def load_domain_vocab(path: str | None = None) -> dict[str, dict[str, list[str]]]:
    """Load the closed ``{PILLAR: {domain: [signals]}}`` vocabulary.

    Cached. Pass *path* to load a specific file (tests / the CI gate); default is
    the shipped :data:`DOMAIN_VOCAB_PATH`.
    """
    import yaml

    src = Path(path) if path else DOMAIN_VOCAB_PATH
    data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    pillars = data.get("pillars", {})
    # Normalize: every domain maps to a list of lowercase signal strings.
    out: dict[str, dict[str, list[str]]] = {}
    for pillar, domains in pillars.items():
        out[pillar] = {
            dom: [str(s).lower() for s in (signals or [])]
            for dom, signals in (domains or {}).items()
        }
    return out


def valid_domains(pillar: str, *, vocab_path: str | None = None) -> frozenset[str]:
    """The closed set of domains allowed under *pillar* (empty for unknown pillar)."""
    return frozenset(load_domain_vocab(vocab_path).get(pillar, {}))


def is_valid_domain(pillar: str, domain: str, *, vocab_path: str | None = None) -> bool:
    """True iff *domain* is in the closed vocabulary for *pillar*."""
    return domain in valid_domains(pillar, vocab_path=vocab_path)


#: The canonical repo->SLUG registry shipped alongside this module.
SLUG_REGISTRY_PATH = Path(__file__).with_name("slug_registry.yaml")


@lru_cache(maxsize=8)
def load_slug_registry(path: str | None = None) -> dict[str, str]:
    """Load the ``{repo-name: SLUG}`` registry. Cached.

    Validates that every SLUG is exactly two uppercase letters and globally
    unique — a malformed/duplicated registry is a hard error, not silent.
    """
    import yaml

    src = Path(path) if path else SLUG_REGISTRY_PATH
    data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    slugs: dict[str, str] = dict(data.get("slugs", {}))
    seen: dict[str, str] = {}
    for repo, slug in slugs.items():
        if not re.fullmatch(r"[A-Z]{2}", str(slug)):
            raise ValueError(f"SLUG for {repo!r} must be 2 uppercase letters: {slug!r}")
        if slug in seen:
            raise ValueError(f"duplicate SLUG {slug!r}: {seen[slug]!r} and {repo!r}")
        seen[slug] = repo
    return slugs


def slug_for_repo(repo_name: str, *, path: str | None = None) -> str | None:
    """SLUG for a repo directory name, or ``None`` if unregistered."""
    return load_slug_registry(path).get(repo_name)


def okf_part_of_edges(parsed: list[OkfConceptId]) -> list[tuple[str, str]]:
    """OKF-CIS mereology as IRI pairs: concept->domain->pillar->scheme.

    The RDF generator emits these as ``:partOf`` (and the matching ``skos:broader``)
    triples. Pillar IRIs are shared (no slug), so cross-repo concepts under the
    same pillar federate onto one node.
    """
    edges: set[tuple[str, str]] = set()
    for p in parsed:
        edges.add((p.iri, p.domain_iri))
        edges.add((p.domain_iri, p.pillar_iri))
        edges.add((p.pillar_iri, p.scheme_iri))
    return sorted(edges)
