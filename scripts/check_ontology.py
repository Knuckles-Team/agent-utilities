#!/usr/bin/env python3
"""Ontology library gate — every ontology must be VALID and CONNECTED.

CONCEPT:AU-KG.ontology.anti-sprawl-gate — anti-sprawl / anti-drift gate for the bundled OWL/RDF ontology
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
import os
import re
import stat
import sys
import tomllib
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

_PROVIDER_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$", re.ASCII)
_MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_MAX_PROVIDERS = 256
_MAX_DIRECTORY_ENTRIES = 4_096
_MAX_ASSETS = 2_048
_MAX_ASSET_BYTES = 4 * 1024 * 1024
_MAX_TOTAL_BYTES = 128 * 1024 * 1024
_MAX_PYPROJECT_BYTES = 1024 * 1024
_SOURCE_LABELS: dict[Path, Path] = {}


class FleetScanError(RuntimeError):
    """Privacy-safe provider source scan failure."""


def _fail(violations: list[str], msg: str) -> None:
    violations.append(msg)


def _rel(p: Path) -> Path | str:
    """Return a repository-relative or generic provider-owned display path.

    Contributed (federated) ontology TTLs live inside another package's wheel /
    editable checkout. Their machine-local installation paths must never cross this
    gate's diagnostic boundary.
    """
    if p in _SOURCE_LABELS:
        return _SOURCE_LABELS[p]
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return Path("provider-assets") / p.name


def _read_regular(path: Path, *, maximum: int, code: str) -> bytes:
    """Read one bounded regular file without following a link or retaining its path."""

    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise FleetScanError(code)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
                or opened.st_size > maximum
            ):
                raise FleetScanError(code)
            data = b""
            while len(data) <= maximum:
                chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(data)))
                if not chunk:
                    break
                data += chunk
            if len(data) > maximum:
                raise FleetScanError(code)
            return data
        finally:
            os.close(descriptor)
    except FleetScanError:
        raise
    except OSError as exc:
        raise FleetScanError(code) from exc


def _require_directory(path: Path, code: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FleetScanError(code) from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise FleetScanError(code)


def _bounded_children(path: Path, *, maximum: int, code: str) -> list[Path]:
    """List a directory deterministically without accepting an unbounded fan-out."""

    children: list[Path] = []
    try:
        for child in path.iterdir():
            if child.name.startswith("."):
                continue
            if len(children) >= maximum:
                raise FleetScanError(code)
            children.append(child)
    except FleetScanError:
        raise
    except OSError as exc:
        raise FleetScanError(code) from exc
    return sorted(children, key=lambda item: item.name.casefold())


def _source_provider_ttls(agents_root: Path) -> list[Path]:
    """Discover declared provider assets from a bounded, no-follow source fleet."""

    _require_directory(agents_root, "provider-root-type")
    children = _bounded_children(
        agents_root, maximum=_MAX_PROVIDERS, code="provider-count-bound"
    )
    if len(children) > _MAX_PROVIDERS:
        raise FleetScanError("provider-count-bound")

    assets: list[Path] = []
    total_bytes = 0
    for project in children:
        try:
            project_metadata = project.lstat()
        except OSError as exc:
            raise FleetScanError("provider-project-read") from exc
        if not stat.S_ISDIR(project_metadata.st_mode):
            # The fleet root may contain non-provider aliases or marker files.
            # Never follow them; authoritative membership is enforced by the
            # separate provider-fleet/workspace gate.
            continue
        pyproject = project / "pyproject.toml"
        if not pyproject.exists():
            continue
        try:
            document = tomllib.loads(
                _read_regular(
                    pyproject,
                    maximum=_MAX_PYPROJECT_BYTES,
                    code="provider-metadata-type",
                ).decode("utf-8")
            )
        except FleetScanError:
            raise
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise FleetScanError("provider-metadata-parse") from exc
        try:
            registrations = document["project"]["entry-points"][
                "agent_utilities.ontology_providers"
            ]
        except (KeyError, TypeError):
            continue
        if not isinstance(registrations, dict) or not registrations:
            raise FleetScanError("provider-registration")
        for provider, module in sorted(registrations.items()):
            if (
                not isinstance(provider, str)
                or _PROVIDER_ID.fullmatch(provider) is None
                or not isinstance(module, str)
                or _MODULE.fullmatch(module) is None
            ):
                raise FleetScanError("provider-registration")
            ontology_dir = project.joinpath(*module.split("."))
            current = project
            for component in module.split("."):
                current /= component
                _require_directory(current, "provider-ontology-root")
            candidates: list[tuple[Path, Path]] = []
            for entry in _bounded_children(
                ontology_dir,
                maximum=_MAX_DIRECTORY_ENTRIES,
                code="provider-ontology-entry-bound",
            ):
                if entry.suffix == ".ttl":
                    candidates.append((entry, Path(entry.name)))
            shapes = ontology_dir / "shapes"
            if shapes.exists():
                _require_directory(shapes, "provider-shapes-root")
                for entry in _bounded_children(
                    shapes,
                    maximum=_MAX_DIRECTORY_ENTRIES,
                    code="provider-shapes-entry-bound",
                ):
                    if entry.suffix == ".ttl":
                        candidates.append((entry, Path("shapes") / entry.name))
            for asset, relative in candidates:
                data = _read_regular(
                    asset, maximum=_MAX_ASSET_BYTES, code="provider-asset-type"
                )
                total_bytes += len(data)
                if len(assets) >= _MAX_ASSETS or total_bytes > _MAX_TOTAL_BYTES:
                    raise FleetScanError("provider-assets-bound")
                assets.append(asset)
                _SOURCE_LABELS[asset] = Path("provider-assets") / provider / relative
    return sorted(assets, key=lambda path: _SOURCE_LABELS[path].as_posix())


def _provider_ttls(provider_root: Path | None = None) -> list[Path]:
    """Contributed ontology TTLs from installed fleet packages (CONCEPT:AU-KG.ontology.package-owned-ontology).

    Reuses the federation read-path resolver (XDG-first, CONCEPT:AU-OS.deployment.unified-install-tree) so the gate
    sweeps package-contributed ontologies from the same place the runtime does — the
    materialized unified tree when populated, else live entry-point discovery.
    Failure-isolated: if the resolver (or its package) can't be imported, federation
    is simply an empty superset.
    """
    if provider_root is not None:
        return _source_provider_ttls(provider_root)
    try:
        from agent_utilities.knowledge_graph.core.ontology_federation import (
            resolve_provider_ontologies,
        )

        return [p for _provider, p in resolve_provider_ontologies()]
    except Exception:  # noqa: BLE001 — federation is additive; base gate must not break
        return []


def _federated_iris() -> set[str]:
    """Known package-owned ontology IRIs (CONCEPT:AU-KG.ontology.package-owned-ontology).

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


def _domain_modules(provider_ttls: list[Path] | None = None) -> list[Path]:
    """Domain modules — the set the loader/publisher glob over, plus federated ones.

    Bundled: ``ontology_*.ttl`` directly in ``knowledge_graph/`` (the same glob the
    owlready2 backend and ``collect_bundled_ontology_graph`` use), excluding the
    canonical ``ontology.ttl`` itself. Federated (CONCEPT:AU-KG.ontology.package-owned-ontology): every
    contributed non-shape ``*.ttl`` from installed ontology-provider packages, so a
    moved module (e.g. servicenow now living in the servicenow-api wheel) is
    connectivity/closure-checked exactly like a bundled one.
    """
    bundled = [p for p in KG_DIR.glob("ontology_*.ttl")]
    providers = _provider_ttls() if provider_ttls is None else provider_ttls
    federated = [p for p in providers if not _is_shape(p)]
    return sorted(set(bundled + federated))


def _all_ttls(provider_ttls: list[Path] | None = None) -> list[Path]:
    providers = _provider_ttls() if provider_ttls is None else provider_ttls
    return sorted(set(list(KG_DIR.rglob("*.ttl")) + providers))


def _parse(path: Path):
    import rdflib

    g = rdflib.Graph()
    raw = _read_regular(path, maximum=_MAX_ASSET_BYTES, code="ontology-asset-type")
    g.parse(data=raw.decode("utf-8"), format="turtle")
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


def _has_import_path(
    iri: str, import_graph: dict[str, set[str]], anchors: set[str]
) -> bool:
    """Return whether an ontology IRI reaches the canonical import component."""

    pending = [iri]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in anchors:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(import_graph.get(current, ()))
    return False


def check(verbose: bool = False, provider_root: Path | None = None) -> int:
    violations: list[str] = []
    notes: list[str] = []

    try:
        import owlrl
        import pyshacl
        import rdflib
    except Exception:  # noqa: BLE001 - an unusable validator must fail closed
        print(
            "check_ontology: required validation dependencies unavailable; "
            "failing closed."
        )
        return 1

    if not CANONICAL.exists():
        print(f"check_ontology: canonical ontology missing: {CANONICAL}")
        return 1

    _SOURCE_LABELS.clear()
    try:
        provider_ttls = _provider_ttls(provider_root)
        all_ttls = _all_ttls(provider_ttls)
    except FleetScanError as exc:
        print(f"check_ontology: provider fleet scan failed ({exc}).")
        return 1
    parsed: dict[Path, object] = {}

    # ── 1. Syntax: every .ttl parses ────────────────────────────────────────
    for t in all_ttls:
        try:
            parsed[t] = _parse(t)
        except Exception as exc:  # noqa: BLE001
            _fail(
                violations,
                f"[syntax] {_rel(t)} does not parse ({type(exc).__name__})",
            )
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
    missing_registered = sorted(_federated_iris() - canon_imports)
    for iri in missing_registered:
        _fail(
            violations,
            f"[unlinked-registry] canonical ontology.ttl does not import <{iri}>",
        )
    canonical_iris = (
        set(_declared_ontology_iris(canonical_g)) if canonical_g is not None else set()
    )
    import_graph: dict[str, set[str]] = {}
    for graph in parsed.values():
        imports = set(_imports(graph))
        for iri in _declared_ontology_iris(graph):
            import_graph.setdefault(iri, set()).update(imports)
    connectivity_anchors = canonical_iris | canon_imports

    for mod in _domain_modules(provider_ttls):
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
        if not any(
            _has_import_path(iri, import_graph, connectivity_anchors) for iri in iris
        ):
            _fail(
                violations,
                f"[unlinked] {_rel(mod)} ({iris[0]}) has no import path to the "
                "canonical ontology component.",
            )

    # ── 6. No dangling imports in our own namespace ─────────────────────────
    # CONCEPT:AU-KG.ontology.package-owned-ontology — a package-owned (federated) IRI is allowed to be imported
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
    if SHAPES_DIR.exists():
        for shape_file in sorted(t for t in all_ttls if _is_shape(t)):
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
                    f"[shacl] {_rel(shape_file)} is not well-formed SHACL "
                    f"({type(exc).__name__})",
                )

    # ── 3. OWL-RL closure over the merged graph (no reasoning breakage) ──────
    try:
        merged = rdflib.Graph()
        for mod in [CANONICAL, *_domain_modules(provider_ttls)]:
            g = parsed.get(mod)
            if g is not None:
                for triple in g:
                    merged.add(triple)
        owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(merged)
        notes.append(f"OWL-RL closure ok ({len(merged)} triples after expansion)")
    except Exception as exc:  # noqa: BLE001
        _fail(
            violations,
            f"[owl-rl] merged ontology breaks OWL-RL closure ({type(exc).__name__})",
        )

    # ── 7. Documentation: every .ttl listed in the library index ────────────
    if not LIBRARY_DOC.exists():
        _fail(
            violations,
            f"[docs] ontology library index missing: {LIBRARY_DOC.relative_to(ROOT)}",
        )
    else:
        doc = LIBRARY_DOC.read_text()
        for t in all_ttls:
            display = _rel(t)
            if t in _SOURCE_LABELS:
                provider = display.parts[1]
                relative = Path(*display.parts[2:]).as_posix()
                documented = any(
                    f"`{provider}`" in line and f"`{relative}`" in line
                    for line in doc.splitlines()
                )
            else:
                relative = t.relative_to(KG_DIR).as_posix()
                documented = any(
                    f"`{candidate}`" in doc for candidate in (t.name, relative)
                )
            if not documented:
                _fail(
                    violations,
                    f"[docs] {display} is not listed with its owner in the ontology library",
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
    ap.add_argument(
        "--provider-root",
        type=Path,
        help="explicit provider fleet agents root for source certification",
    )
    args = ap.parse_args()
    return check(verbose=args.verbose, provider_root=args.provider_root)


if __name__ == "__main__":
    sys.exit(main())
