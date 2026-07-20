"""Canonical OKF-CIS concept hierarchy.

CONCEPT:AU-OS.governance.concept-hierarchy-standardization — concept-hierarchy standardization (B5).

This is the one place that parses concept IDs and derives their OKF path and
RDF IRI. Registry generation, validation, allocation, and the CLI import this
grammar so authoring and governance cannot drift.

Grammar
-------
Canonical id::

    <SLUG>-<PILLAR>.<domain>.<concept>[.<facet>...]

* ``SLUG`` is the two-letter repository code.
* ``PILLAR`` is one member of the closed ecosystem taxonomy.
* ``domain`` is a member of the pillar's closed domain vocabulary.
* ``concept`` and optional facets are semantic kebab-case segments.

No alternate spelling, numeric ID, alias, or implicit hierarchy is accepted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

#: The closed set of 6 ecosystem pillars — shared across ALL repos. A concept's
#: SLUG says which repo owns it; its PILLAR says which capability domain it is.
PILLARS: tuple[str, ...] = ("ORCH", "KG", "AHE", "ECO", "OS", "GBOT")

_PILLAR_ALT = "|".join(PILLARS)
#: One kebab segment: lowercase alnum words joined by hyphens (``entropy-dedup``).
_SEG = r"[a-z0-9]+(?:-[a-z0-9]+)*"

#: A bare OKF-CIS id (no ``CONCEPT:`` prefix), fully anchored.
OKF_ID_RE = re.compile(
    rf"^(?P<slug>[A-Z]{{2}})-(?P<pillar>{_PILLAR_ALT})(?P<segs>(?:\.{_SEG}){{2,}})$"
)
#: The ``CONCEPT:<id>`` marker form — the one canonical marker regex.
OKF_MARKER_RE = re.compile(
    rf"CONCEPT:(?P<id>[A-Z]{{2}}-(?:{_PILLAR_ALT})(?:\.{_SEG}){{2,}})"
)

#: Canonical ecosystem IRI roots.
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
        raise ValueError(f"OKF-CIS id needs >=2 segments (domain.concept): {cid!r}")
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
