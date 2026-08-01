"""Domain pack discovery, validation, and (in-process) registry
(CONCEPT:AU-KG.ingest.domain-pack-framework).

A domain pack is **installable, versioned, and removable**, and several packs
coexist — exactly the same "one engine authority, many domain ontology
modules" shape the fleet already uses (``ontology_servicenow.ttl``,
``ontology_leanix.ttl``, ... all ``owl:imports``-ed into one canonical
``ontology.ttl``, one epistemic-graph engine authoritative over all of them).
This module is the domain-pack-specific front door onto that same pattern:
discover packs from a root directory, validate each **independently and fail
closed**, and keep an in-process registry so multiple packs' facts can be
produced side by side without coupling one pack's mapping to another's.

**Fail-closed load-time checks** (any failure -> :class:`DomainPackError`,
never a partial/best-effort load):

1. YAML parses and validates against :class:`DomainPackManifest` exactly
   (``extra="forbid"`` everywhere in the schema — an unknown key is refused,
   not ignored).
2. The pack's ``version`` is real semver and its directory name matches
   ``pack`` (mirrors ``ConnectorManifest``'s own identity-matches-directory
   check in ``connector_manifest_gate.bundled_provider_contract``).
3. ``provenance.integrity.hash`` matches a fresh recompute over the pack's own
   canonical document (:func:`pack_integrity_hash` — the domain-pack analogue
   of :func:`ontology_integrity.canonical_manifest_hash`; see that function's
   docstring for why a pack needs its own, self-excluding variant) — a
   hand-edited-after-signing pack is refused exactly like a hand-edited
   connector manifest. An optional ``signer``/``signature`` is verified with
   the same Ed25519 machinery (:mod:`..ontology.ontology_integrity`) when present.
4. Every mapping rule's ``node_type``/``edge_target_type`` resolves to either
   (a) one of the pack's own declared ``ontology.resources`` (fast path, no
   canonical lookup needed) or (b) an existing class in the canonical ontology
   library (:func:`canonical_ontology_class_names`) — never an invented class
   name. This is the concrete enforcement of "packs must extend the canonical
   ontology library."
5. Every bundled ``shacl_shapes`` file parses as valid Turtle/SHACL and its
   ``sh:targetClass`` values resolve under the same rule as (4).
6. Every declared ``evaluation_cases`` entry reproduces **exactly** (entities +
   relationships as sets) when run through :func:`.dsl.evaluate_mapping` — the
   pack's own self-check that its mappings are inspectable/predictable.

Any one of these failing raises before the pack is usable; there is no
"loaded with warnings" state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..ontology import ontology_integrity
from ..ontology.connector_manifest import OntologySpec
from ..ontology.manifest_compiler import compile_manifest
from .domain_pack import DomainPackManifest
from .dsl import evaluate_mapping
from .fragment_contract import Artifact, Fragment

__all__ = [
    "DomainPackError",
    "LoadedDomainPack",
    "canonical_ontology_class_names",
    "pack_integrity_hash",
    "load_pack",
    "DomainPackRegistry",
]


def pack_integrity_hash(manifest: DomainPackManifest) -> str:
    """Canonical SHA-256 of the WHOLE domain pack document.

    Deliberately its own function rather than a call to
    :func:`ontology_integrity.canonical_manifest_hash`: that connector-manifest
    helper only excludes ``provenance.signature`` before hashing, which is
    correct there because a connector's ``provenance.integrity.hash`` pins an
    *external* artifact (the compiled ontology TTL) — no self-reference. A
    domain pack instead uses ``provenance.integrity.hash`` to pin **the pack
    document itself**, so the hash MUST also exclude its own
    ``integrity.hash`` field, or every hash would cover its own value
    (unsatisfiable). ``signature`` is excluded for the same reason
    ``canonical_manifest_hash`` excludes it: a document can't sign a digest
    that includes the signature.
    """
    data = manifest.model_dump(mode="json")
    provenance = dict(data.get("provenance") or {})
    integrity = dict(provenance.get("integrity") or {})
    integrity["hash"] = None
    provenance["integrity"] = integrity
    provenance["signature"] = None
    data["provenance"] = provenance
    payload = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class DomainPackError(RuntimeError):
    """A domain pack is missing, malformed, or fails a fail-closed check.

    Every raise site names *which* check failed so an operator can fix the
    pack rather than guess."""


@dataclass(frozen=True)
class LoadedDomainPack:
    """A pack that has passed every load-time check — safe to run mappings from."""

    manifest: DomainPackManifest
    ontology_spec: OntologySpec
    path: Path

    @property
    def own_class_names(self) -> frozenset[str]:
        return frozenset(c.local for c in self.ontology_spec.classes)


def _knowledge_graph_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def canonical_ontology_class_names() -> frozenset[str]:
    """Every ``owl:Class`` local name declared anywhere in the bundled
    canonical ontology library (``ontology.ttl`` + every sibling
    ``ontology_*.ttl`` domain module physically present in the package).

    Offline, deterministic, no LLM — a straight rdflib parse over files that
    ship with the package regardless of which are currently ``owl:imports``-ed
    at runtime (mirrors ``manifest_compiler._canonical_owl_imports``'s own
    file-based approach). Used only to check that a domain pack's mappings
    reference a class that genuinely exists somewhere in the canonical
    library — never to decide whether that module is wired/active.
    """
    import rdflib

    kg_dir = _knowledge_graph_dir()
    names: set[str] = set()
    owl_class = rdflib.URIRef("http://www.w3.org/2002/07/owl#Class")
    for ttl_path in sorted(kg_dir.glob("ontology*.ttl")):
        graph = rdflib.Graph()
        try:
            graph.parse(str(ttl_path), format="turtle")
        except Exception:  # noqa: BLE001 - a malformed sibling module must not
            # hide a genuine pack-validation failure behind an unrelated parse
            # error; skip it (it will independently fail its OWN gate).
            continue
        for subject in graph.subjects(rdflib.RDF.type, owl_class):
            if isinstance(subject, rdflib.URIRef):
                names.add(str(subject).rsplit("#", 1)[-1])
    return frozenset(names)


def _resolve_class(
    name: str | None, *, own_classes: frozenset[str], canonical: frozenset[str]
) -> bool:
    if name is None:
        return True
    return name in own_classes or name in canonical


def _mapping_class_names(rule: Any) -> list[tuple[str, str | None]]:
    """[(field_name, class_name_or_None), ...] for one mapping rule."""
    pairs: list[tuple[str, str | None]] = []
    if hasattr(rule, "node_type"):
        pairs.append(("node_type", rule.node_type))
    if hasattr(rule, "row_node_type"):
        pairs.append(("row_node_type", rule.row_node_type))
    if hasattr(rule, "edge_target_type"):
        pairs.append(("edge_target_type", rule.edge_target_type))
    if hasattr(rule, "target_node_type"):
        pairs.append(("target_node_type", rule.target_node_type))
    if hasattr(rule, "columns"):
        for column in rule.columns.values():
            if column.edge_target_type:
                pairs.append(("columns.edge_target_type", column.edge_target_type))
    return pairs


def _check_ontology_classes(
    manifest: DomainPackManifest, ontology_spec: OntologySpec, *, label: str
) -> None:
    own_classes = frozenset(c.local for c in ontology_spec.classes)
    canonical = canonical_ontology_class_names()
    for rule in manifest.mappings:
        for field_name, class_name in _mapping_class_names(rule):
            if not _resolve_class(
                class_name, own_classes=own_classes, canonical=canonical
            ):
                raise DomainPackError(
                    f"{label}: mapping {rule.kind!r} references unknown ontology "
                    f"class {class_name!r} in {field_name!r} — it is neither one of "
                    f"this pack's own ontology.resources nor an existing class in "
                    f"the canonical ontology library. Add it to ontology.resources "
                    f"(with a schema_mappings.ontology_class crosswalk), or fix the "
                    f"typo."
                )


def _check_shacl_shapes(
    manifest: DomainPackManifest, pack_dir: Path, *, label: str
) -> None:
    if not manifest.shacl_shapes:
        return
    import rdflib

    own_classes = frozenset(r.name for r in manifest.ontology.resources)
    canonical = canonical_ontology_class_names()
    sh_target_class = rdflib.URIRef("http://www.w3.org/ns/shacl#targetClass")
    for rel_path in manifest.shacl_shapes:
        shape_path = pack_dir / rel_path
        if not shape_path.is_file():
            raise DomainPackError(
                f"{label}: shacl_shapes entry {rel_path!r} does not exist"
            )
        graph = rdflib.Graph()
        try:
            graph.parse(str(shape_path), format="turtle")
        except Exception as exc:  # noqa: BLE001 - fail-closed on invalid SHACL
            raise DomainPackError(
                f"{label}: shacl_shapes entry {rel_path!r} is not valid Turtle/SHACL"
            ) from exc
        for target in graph.objects(predicate=sh_target_class):
            class_name = str(target).rsplit("#", 1)[-1]
            if not _resolve_class(
                class_name, own_classes=own_classes, canonical=canonical
            ):
                raise DomainPackError(
                    f"{label}: {rel_path} sh:targetClass {class_name!r} is neither "
                    "a pack resource nor a canonical ontology class"
                )


def _as_set(dicts: list[dict[str, Any]]) -> set[str]:
    return {
        json.dumps(d, sort_keys=True, separators=(",", ":"), default=str) for d in dicts
    }


def _check_evaluation_cases(manifest: DomainPackManifest, *, label: str) -> None:
    for case in manifest.evaluation_cases:
        artifact = Artifact(**case.artifact)
        fragments = [Fragment(**f) for f in case.fragments]
        result = evaluate_mapping(manifest, artifact, fragments)
        actual_entities = _as_set(result.entity_dicts())
        actual_relationships = _as_set(result.relationship_dicts())
        expected_entities = _as_set(case.expect_entities)
        expected_relationships = _as_set(case.expect_relationships)
        if (
            actual_entities != expected_entities
            or actual_relationships != expected_relationships
        ):
            raise DomainPackError(
                f"{label}: evaluation_case {case.name!r} does not reproduce — "
                f"expected {len(expected_entities)} entit(y/ies)/"
                f"{len(expected_relationships)} relationship(s), got "
                f"{len(actual_entities)}/{len(actual_relationships)}. A pack whose "
                "own declared self-check doesn't pass is refused (fail closed) — "
                "fix the mappings or the evaluation_case."
            )


def load_pack(path: str | Path) -> LoadedDomainPack:
    """Load and fail-closed-validate one ``domain_pack.yml``.

    Raises :class:`DomainPackError` (or lets a schema ``ValidationError``
    surface) on ANY defect. There is no partial-success return — a pack either
    passes every check and is returned whole, or nothing is returned at all.
    """
    pack_path = Path(path)
    manifest_path = pack_path / "domain_pack.yml" if pack_path.is_dir() else pack_path
    pack_dir = manifest_path.parent
    label = f"{pack_dir.name}/domain_pack.yml"

    if not manifest_path.is_file():
        raise DomainPackError(f"{label}: no domain_pack.yml found")

    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        manifest = DomainPackManifest.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - fail-closed on any parse/schema defect
        raise DomainPackError(
            f"{label}: does not parse/validate as a DomainPackManifest "
            f"({type(exc).__name__})"
        ) from exc

    if manifest.pack != pack_dir.name:
        raise DomainPackError(
            f"{label}: pack name {manifest.pack!r} does not match its directory "
            f"{pack_dir.name!r}"
        )

    manifest_hash = pack_integrity_hash(manifest)
    if manifest_hash != manifest.provenance.integrity.hash:
        raise DomainPackError(
            f"{label}: recomputed integrity hash {manifest_hash} != "
            f"provenance.integrity.hash {manifest.provenance.integrity.hash} — "
            "the pack was hand-edited after its hash was pinned, or is stale."
        )
    if manifest.provenance.signer or manifest.provenance.signature:
        if not manifest.provenance.signer or not manifest.provenance.signature:
            raise DomainPackError(
                f"{label}: partial signature (signer/signature must both be set)"
            )
        if not ontology_integrity.verify_release_signature(
            manifest_hash,
            manifest.provenance.signature,
            signer_id=manifest.provenance.signer,
            algorithm=manifest.provenance.signature_algorithm,
            public_key=manifest.provenance.signing_public_key,
        ):
            raise DomainPackError(f"{label}: provenance.signature failed verification")

    try:
        ontology_spec = compile_manifest(manifest.ontology)
    except Exception as exc:  # noqa: BLE001 - fail-closed on an uncompilable ontology extension
        raise DomainPackError(
            f"{label}: ontology extension does not compile ({type(exc).__name__})"
        ) from exc

    _check_ontology_classes(manifest, ontology_spec, label=label)
    _check_shacl_shapes(manifest, pack_dir, label=label)
    _check_evaluation_cases(manifest, label=label)

    return LoadedDomainPack(
        manifest=manifest, ontology_spec=ontology_spec, path=pack_dir
    )


class DomainPackRegistry:
    """In-process registry so **multiple packs coexist** without coupling.

    Each pack is loaded and validated fully independently
    (:func:`load_pack`) — one pack's invalid mapping never blocks another
    already-installed pack, and removing a pack is just dropping it from the
    registry (no shared mutable ontology state; the epistemic-graph engine
    remains the one authority across every installed pack's facts, exactly as
    it already is across every existing domain ontology module).
    """

    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._packs: dict[str, LoadedDomainPack] = {}

    @property
    def root(self) -> Path:
        return self._root

    def install(self, pack_dir: str | Path) -> LoadedDomainPack:
        """Validate and register one pack. Raises (never silently degrades)
        on any :class:`DomainPackError`."""
        loaded = load_pack(pack_dir)
        self._packs[loaded.manifest.pack] = loaded
        return loaded

    def remove(self, pack: str) -> bool:
        return self._packs.pop(pack, None) is not None

    def get(self, pack: str) -> LoadedDomainPack | None:
        return self._packs.get(pack)

    def list(self) -> list[LoadedDomainPack]:
        return sorted(self._packs.values(), key=lambda p: p.manifest.pack)

    def discover_and_install_all(self) -> list[LoadedDomainPack]:
        """Install every ``<root>/<pack>/domain_pack.yml`` found. Fails closed:
        one bad pack raises immediately rather than silently skipping it —
        an operator debugging "why didn't my pack load" must never learn the
        answer is "the loader swallowed the error"."""
        loaded: list[LoadedDomainPack] = []
        if not self._root.is_dir():
            return loaded
        for child in sorted(self._root.iterdir()):
            if child.is_dir() and (child / "domain_pack.yml").is_file():
                loaded.append(self.install(child))
        return loaded
