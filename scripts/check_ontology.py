#!/usr/bin/env python3
"""Ontology library gate — every ontology must be VALID and CONNECTED.

CONCEPT:KG-2.112 — anti-sprawl / anti-drift gate for the bundled OWL/RDF ontology
library under ``agent_utilities/knowledge_graph/``. It enforces, in one place, the
invariants that keep the ontology library from rotting into the state we just fixed
(a divergent duplicate ``core/ontology.ttl`` the reasoner silently loaded instead of
the real one, and ~17 domain modules that no canonical file referenced):

  VALID
    1. Every ``*.ttl`` parses as Turtle (syntax).
    2. No two files declare the SAME ``owl:Ontology`` IRI (drift / duplicate guard).
    3. The merged ontology survives OWL-RL closure without error (no reasoning breakage).
    4. Every ``shapes/*.ttl`` is well-formed SHACL that pyshacl can load and run
       (catches a broken shape or an ontology change that breaks SHACL validation).

  CONNECTED (no unlinked, no dangling)
    5. Every domain module (``ontology_<name>.ttl``) declares exactly one
       ``owl:Ontology`` IRI AND is imported by the canonical ``ontology.ttl`` —
       an unreferenced module is a build failure, not a warning.
    6. Every ``owl:imports`` target in our own namespace
       (``http://knuckles.team/kg*`` / ``https://agent-utilities.dev/*``) resolves
       to a present local file — no broken/dangling import IRIs. External standard
       vocabularies (w3.org, purl.org, schema.org, edmcouncil, …) are allowed remote.

  DOCUMENTED
    7. Every ``*.ttl`` on disk is listed in ``docs/architecture/ontology_library.md``
       (catches a new ontology added without a library-index entry).

Usage:
  python3 scripts/check_ontology.py          # check (exit 1 on any violation)
  python3 scripts/check_ontology.py -v        # verbose (print per-check detail)

Exit 0 = all ontologies valid + connected + documented, 1 = violation(s) found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Ensure the in-repo ``agent_utilities`` is importable even when the package isn't
# pip-installed, so the KG-2.320 federation discoverer/registry can be reached.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
KG_DIR = ROOT / "agent_utilities" / "knowledge_graph"
SHAPES_DIR = KG_DIR / "shapes"
CANONICAL = KG_DIR / "ontology.ttl"
LIBRARY_DOC = ROOT / "docs" / "architecture" / "ontology_library.md"

# IRIs whose authority we own — an import of one of these MUST resolve to a local
# file (anything else, e.g. w3.org/purl.org/schema.org, is a legitimate remote vocab).
_OWN_PREFIXES = ("http://knuckles.team/kg", "https://agent-utilities.dev/")

OWL_IMPORTS = "http://www.w3.org/2002/07/owl#imports"
OWL_ONTOLOGY = "http://www.w3.org/2002/07/owl#Ontology"


def _fail(violations: list[str], msg: str) -> None:
    violations.append(msg)


def _rel(p: Path) -> Path | str:
    """``p`` relative to the repo root when possible, else the absolute path.

    Contributed (federated) ontology TTLs live inside another package's wheel /
    editable checkout — outside this repo — so ``relative_to(ROOT)`` would raise.
    """
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def _provider_ttls() -> list[Path]:
    """Contributed ontology TTLs from installed fleet packages (CONCEPT:KG-2.320).

    Reuses the federation discoverer so the gate sweeps package-contributed
    ontologies identically to bundled ones. Failure-isolated: if the discoverer
    (or its package) can't be imported, federation is simply an empty superset.
    """
    try:
        from agent_utilities.knowledge_graph.core.ontology_federation import (
            discover_provider_ontologies,
        )

        return [p for _provider, p in discover_provider_ontologies()]
    except Exception:  # noqa: BLE001 — federation is additive; base gate must not break
        return []


def _federated_iris() -> set[str]:
    """Known package-owned ontology IRIs (CONCEPT:KG-2.320).

    The canonical bundle may keep an ``owl:imports`` edge to one of these even when
    the owning package is not installed; such an import is a superset no-op, not a
    dangling reference. Failure-isolated (empty when the registry is unavailable).
    """
    try:
        from agent_utilities.knowledge_graph.core.ontology_federation import (
            registered_federated_iris,
        )

        return registered_federated_iris()
    except Exception:  # noqa: BLE001
        return set()


def _is_shape(p: Path) -> bool:
    return p.parent.name == "shapes"


def _domain_modules() -> list[Path]:
    """Domain modules — the set the loader/publisher glob over, plus federated ones.

    Bundled: ``ontology_*.ttl`` directly in ``knowledge_graph/`` (the same glob the
    owlready2 backend and ``collect_bundled_ontology_graph`` use), excluding the
    canonical ``ontology.ttl`` itself. Federated (CONCEPT:KG-2.320): every
    contributed non-shape ``*.ttl`` from installed ontology-provider packages, so a
    moved module (e.g. servicenow now living in the servicenow-api wheel) is
    connectivity/closure-checked exactly like a bundled one.
    """
    bundled = [p for p in KG_DIR.glob("ontology_*.ttl")]
    federated = [p for p in _provider_ttls() if not _is_shape(p)]
    return sorted(set(bundled + federated))


def _all_ttls() -> list[Path]:
    return sorted(set(list(KG_DIR.rglob("*.ttl")) + _provider_ttls()))


def _parse(path: Path):
    import rdflib

    g = rdflib.Graph()
    g.parse(str(path), format="turtle")
    return g


def _declared_ontology_iris(g) -> list[str]:
    import rdflib

    return [
        str(s)
        for s in g.subjects(
            predicate=rdflib.RDF.type, object=rdflib.URIRef(OWL_ONTOLOGY)
        )
    ]


def _imports(g) -> list[str]:
    import rdflib

    return [str(o) for o in g.objects(predicate=rdflib.URIRef(OWL_IMPORTS))]


def check(verbose: bool = False) -> int:
    violations: list[str] = []
    notes: list[str] = []

    try:
        import rdflib  # noqa: F401
    except ImportError:
        print("check_ontology: rdflib not installed — cannot validate; failing closed.")
        return 1

    if not CANONICAL.exists():
        print(f"check_ontology: canonical ontology missing: {CANONICAL}")
        return 1

    all_ttls = _all_ttls()
    parsed: dict[Path, object] = {}

    # ── 1. Syntax: every .ttl parses ────────────────────────────────────────
    for t in all_ttls:
        try:
            parsed[t] = _parse(t)
        except Exception as exc:  # noqa: BLE001
            _fail(violations, f"[syntax] {_rel(t)} does not parse: {exc}")
    notes.append(f"parsed {len(parsed)}/{len(all_ttls)} TTL files")

    # ── 2. No duplicate ontology IRIs (drift / duplicate guard) ─────────────
    iri_to_files: dict[str, list[Path]] = {}
    for t, g in parsed.items():
        for iri in _declared_ontology_iris(g):
            iri_to_files.setdefault(iri, []).append(t)
    for iri, files in iri_to_files.items():
        if len(files) > 1:
            rels = ", ".join(str(_rel(f)) for f in files)
            _fail(
                violations,
                f"[duplicate-iri] ontology IRI <{iri}> declared by multiple files: {rels}",
            )

    # ── 5. Connectivity: every domain module declares an IRI and is imported ─
    canonical_g = parsed.get(CANONICAL)
    canon_imports = set(_imports(canonical_g)) if canonical_g is not None else set()
    for mod in _domain_modules():
        g = parsed.get(mod)
        if g is None:
            continue  # syntax failure already reported
        iris = _declared_ontology_iris(g)
        if not iris:
            _fail(
                violations,
                f"[unlinked] {mod.name} declares no owl:Ontology IRI — it cannot be "
                f"imported/addressed. Add `<http://knuckles.team/kg/{mod.stem.removeprefix('ontology_')}> a owl:Ontology .`",
            )
            continue
        if len(iris) > 1:
            _fail(
                violations,
                f"[multi-iri] {mod.name} declares >1 owl:Ontology IRI: {iris}",
            )
        if not any(iri in canon_imports for iri in iris):
            _fail(
                violations,
                f"[unlinked] {mod.name} ({iris[0]}) is NOT imported by the canonical "
                f"ontology.ttl — add an owl:imports edge so the module is connected.",
            )

    # ── 6. No dangling imports in our own namespace ─────────────────────────
    # CONCEPT:KG-2.320 — a package-owned (federated) IRI is allowed to be imported
    # even when its provider package isn't installed here (a superset no-op), so the
    # canonical bundle can keep its ``owl:imports`` edge to a moved module without
    # the base install going red.
    declared = set(iri_to_files) | _federated_iris()
    for t, g in parsed.items():
        for imp in _imports(g):
            if imp.startswith(_OWN_PREFIXES) and imp not in declared:
                _fail(
                    violations,
                    f"[dangling-import] {_rel(t)} imports <{imp}> which "
                    f"resolves to no local ontology file.",
                )

    # ── 4. SHACL shapes well-formed + runnable ──────────────────────────────
    try:
        import pyshacl  # noqa: F401

        have_pyshacl = True
    except ImportError:
        have_pyshacl = False
        notes.append("pyshacl not installed — SHACL well-formedness check skipped")
    if have_pyshacl and SHAPES_DIR.exists():
        import rdflib

        for shape_file in sorted(SHAPES_DIR.glob("*.ttl")):
            sg = parsed.get(shape_file)
            if sg is None:
                continue
            try:
                # Validate a trivial data graph WITH these shapes — this forces
                # pyshacl to load/compile every shape; a malformed SHACL construct
                # raises ShapeLoadError/ConstraintLoadError here.
                pyshacl.validate(
                    data_graph=rdflib.Graph(),
                    shacl_graph=sg,
                    inference="none",
                    abort_on_first=False,
                )
            except Exception as exc:  # noqa: BLE001
                _fail(
                    violations,
                    f"[shacl] {shape_file.relative_to(ROOT)} is not well-formed SHACL: {exc}",
                )

    # ── 3. OWL-RL closure over the merged graph (no reasoning breakage) ──────
    try:
        import owlrl
        import rdflib

        merged = rdflib.Graph()
        for mod in [CANONICAL, *_domain_modules()]:
            g = parsed.get(mod)
            if g is not None:
                for triple in g:
                    merged.add(triple)
        owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(merged)
        notes.append(f"OWL-RL closure ok ({len(merged)} triples after expansion)")
    except ImportError:
        notes.append("owlrl not installed — OWL-RL closure check skipped")
    except Exception as exc:  # noqa: BLE001
        _fail(violations, f"[owl-rl] merged ontology breaks OWL-RL closure: {exc}")

    # ── 7. Documentation: every .ttl listed in the library index ────────────
    if not LIBRARY_DOC.exists():
        _fail(
            violations,
            f"[docs] ontology library index missing: {LIBRARY_DOC.relative_to(ROOT)}",
        )
    else:
        doc = LIBRARY_DOC.read_text()
        for t in all_ttls:
            if t.name not in doc:
                _fail(
                    violations,
                    f"[docs] {t.name} is not listed in docs/architecture/ontology_library.md",
                )

    # ── Report ──────────────────────────────────────────────────────────────
    if verbose:
        for n in notes:
            print(f"  · {n}")
    if violations:
        print(f"check_ontology: {len(violations)} violation(s):")
        for v in violations:
            print(f"  ✗ {v}")
        return 1
    print(
        f"check_ontology: OK — {len(parsed)} ontologies valid, connected, and documented."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="print per-check detail"
    )
    args = ap.parse_args()
    return check(verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
