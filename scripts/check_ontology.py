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
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
# Ensure the in-repo ``agent_utilities`` is importable even when the package isn't
# pip-installed, so the KG-2.320 federation discoverer/registry can be reached.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts._git_scan import tracked_or_walked  # noqa: E402

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


@dataclass(frozen=True)
class _ValidationDependencies:
    """Libraries required by the ontology validators."""

    owlrl: Any
    pyshacl: Any
    rdflib: Any


@dataclass(frozen=True)
class _OntologyBundle:
    """The one parsed snapshot shared by every ontology check."""

    all_ttls: list[Path]
    domain_modules: list[Path]
    parsed: dict[Path, Any]


@dataclass(frozen=True)
class _OntologyIndex:
    """Indexes derived once from an ontology snapshot."""

    iri_to_files: dict[str, list[Path]]
    canonical_imports: set[str]
    canonical_iris: set[str]
    import_graph: dict[str, set[str]]
    federated_iris: set[str]
    connectivity_anchors: set[str]
    declared_iris: set[str]


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


def _regular_file_metadata(path: Path, maximum: int, code: str) -> os.stat_result:
    """Read and validate the initial metadata for one bounded regular file."""
    try:
        metadata = path.lstat()
    except FleetScanError:
        raise
    except OSError as exc:
        raise FleetScanError(code) from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
        raise FleetScanError(code)
    return metadata


def _open_regular_file(
    path: Path, maximum: int, code: str
) -> tuple[int, os.stat_result]:
    """Open a bounded regular file and return its descriptor plus initial metadata."""
    before = _regular_file_metadata(path, maximum, code)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FleetScanError(code) from exc
    return descriptor, before


def _read_regular_descriptor(
    descriptor: int, *, before: os.stat_result, maximum: int, code: str
) -> bytes:
    """Read and revalidate a descriptor opened by :func:`_open_regular_file`."""
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


def _read_regular_unchecked(path: Path, *, maximum: int, code: str) -> bytes:
    """Read an already-opened regular file, leaving OS errors to the wrapper."""
    descriptor, before = _open_regular_file(path, maximum, code)
    try:
        return _read_regular_descriptor(
            descriptor, before=before, maximum=maximum, code=code
        )
    finally:
        os.close(descriptor)


def _read_regular(path: Path, *, maximum: int, code: str) -> bytes:
    """Read one bounded regular file without following a link or retaining its path."""
    try:
        return _read_regular_unchecked(path, maximum=maximum, code=code)
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
    try:
        children = list(
            islice(
                (child for child in path.iterdir() if not child.name.startswith(".")),
                maximum + 1,
            )
        )
    except OSError as exc:
        raise FleetScanError(code) from exc
    if len(children) > maximum:
        raise FleetScanError(code)
    return sorted(children, key=lambda item: item.name.casefold())


def _provider_metadata(pyproject: Path) -> dict[str, Any]:
    """Parse one provider's bounded project metadata."""
    try:
        return tomllib.loads(
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


def _provider_registration_entries(
    document: dict[str, Any],
) -> dict[Any, Any] | None:
    """Extract one provider's ontology entry-point table."""
    try:
        registrations = document["project"]["entry-points"][
            "agent_utilities.ontology_providers"
        ]
    except (KeyError, TypeError):
        return None
    if not isinstance(registrations, dict) or not registrations:
        raise FleetScanError("provider-registration")
    return registrations


def _validate_provider_registration(entry: tuple[Any, Any]) -> tuple[str, str]:
    """Validate one provider identifier and its ontology module path."""
    provider, module = entry
    if (
        not isinstance(provider, str)
        or _PROVIDER_ID.fullmatch(provider) is None
        or not isinstance(module, str)
        or _MODULE.fullmatch(module) is None
    ):
        raise FleetScanError("provider-registration")
    return provider, module


def _provider_registrations(project: Path) -> list[tuple[Any, Any]] | None:
    """Return sorted ontology registrations for one provider project."""
    try:
        project_metadata = project.lstat()
    except OSError as exc:
        raise FleetScanError("provider-project-read") from exc
    if not stat.S_ISDIR(project_metadata.st_mode):
        # The fleet root may contain non-provider aliases or marker files.
        # Never follow them; authoritative membership is enforced by the
        # separate provider-fleet/workspace gate.
        return None
    pyproject = project / "pyproject.toml"
    if not pyproject.exists():
        return None
    registrations = _provider_registration_entries(_provider_metadata(pyproject))
    if registrations is None:
        return None
    return sorted(registrations.items())


def _provider_asset_candidates(project: Path, module: str) -> list[tuple[Path, Path]]:
    """Find ontology and SHACL assets under one validated provider module."""
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
    return candidates


def _append_provider_assets(
    *,
    project: Path,
    provider: str,
    module: str,
    assets: list[Path],
    total_bytes: int,
) -> int:
    """Read, bound, and label the assets declared by one provider module."""
    for asset, relative in _provider_asset_candidates(project, module):
        data = _read_regular(
            asset, maximum=_MAX_ASSET_BYTES, code="provider-asset-type"
        )
        total_bytes += len(data)
        if len(assets) >= _MAX_ASSETS or total_bytes > _MAX_TOTAL_BYTES:
            raise FleetScanError("provider-assets-bound")
        assets.append(asset)
        _SOURCE_LABELS[asset] = Path("provider-assets") / provider / relative
    return total_bytes


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
        registrations = _provider_registrations(project)
        if registrations is None:
            continue
        for entry in registrations:
            provider, module = _validate_provider_registration(entry)
            total_bytes = _append_provider_assets(
                project=project,
                provider=provider,
                module=module,
                assets=assets,
                total_bytes=total_bytes,
            )
    return sorted(assets, key=lambda path: _SOURCE_LABELS[path].as_posix())


def _label_provider_assets(entries: Any) -> list[Path]:
    """Label ontology assets returned by the federation resolver."""
    assets: list[Path] = []
    for provider, path in entries:
        assets.append(path)
        _SOURCE_LABELS[path] = Path("provider-assets") / provider / path.name
    return assets


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

        return _label_provider_assets(resolve_provider_ontologies())
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


def _bundled_ttls(kg_dir: Path) -> list[Path]:
    """Every ``*.ttl`` under the bundled canonical ontology dir (BUG-043).

    Prefers the git-tracked set over a raw ``rglob`` — a filesystem walk also
    picks up gitignored, generated build output (a stale packaging-build
    copy, a local scratch ``.ttl``, ...), which could reintroduce a duplicate
    IRI / broken-import violation this gate already cleared in real source.
    Falls back to ``rglob`` only when ``kg_dir`` is not inside a git working
    tree (e.g. an installed, non-editable copy with no ``.git``).
    """
    return tracked_or_walked(kg_dir, "*.ttl", root=ROOT)


def _all_ttls(provider_ttls: list[Path] | None = None) -> list[Path]:
    providers = _provider_ttls() if provider_ttls is None else provider_ttls
    return sorted(set(_bundled_ttls(KG_DIR) + providers))


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


def _load_validation_dependencies() -> _ValidationDependencies | None:
    """Load optional validators as one typed dependency bundle."""
    try:
        import owlrl
        import pyshacl
        import rdflib
    except Exception:  # noqa: BLE001 - an unusable validator must fail closed
        return None
    return _ValidationDependencies(owlrl=owlrl, pyshacl=pyshacl, rdflib=rdflib)


def _parse_ontology_files(paths: list[Path]) -> tuple[dict[Path, Any], list[str]]:
    """Parse every candidate and return its syntax findings."""
    parsed: dict[Path, Any] = {}
    violations: list[str] = []
    for path in paths:
        try:
            parsed[path] = _parse(path)
        except Exception as exc:  # noqa: BLE001
            _fail(
                violations,
                f"[syntax] {_rel(path)} does not parse ({type(exc).__name__})",
            )
    return parsed, violations


def _build_ontology_bundle(
    provider_root: Path | None,
) -> tuple[_OntologyBundle, list[str]]:
    """Discover and parse one stable ontology snapshot."""
    provider_ttls = _provider_ttls(provider_root)
    all_ttls = _all_ttls(provider_ttls)
    parsed, violations = _parse_ontology_files(all_ttls)
    bundle = _OntologyBundle(
        all_ttls=all_ttls,
        domain_modules=_domain_modules(provider_ttls),
        parsed=parsed,
    )
    return bundle, violations


def _index_declared_iris(parsed: dict[Path, Any]) -> dict[str, list[Path]]:
    """Index every declared ontology IRI by its source files."""
    iri_to_files: dict[str, list[Path]] = {}
    for path, graph in parsed.items():
        for iri in _declared_ontology_iris(graph):
            iri_to_files.setdefault(iri, []).append(path)
    return iri_to_files


def _index_import_graph(parsed: dict[Path, Any]) -> dict[str, set[str]]:
    """Index each ontology declaration's imports for connectivity traversal."""
    import_graph: dict[str, set[str]] = {}
    for graph in parsed.values():
        imports = set(_imports(graph))
        for iri in _declared_ontology_iris(graph):
            import_graph.setdefault(iri, set()).update(imports)
    return import_graph


def _build_ontology_index(bundle: _OntologyBundle) -> _OntologyIndex:
    """Build declaration and import indexes shared by semantic checks."""
    iri_to_files = _index_declared_iris(bundle.parsed)
    canonical_graph = bundle.parsed.get(CANONICAL)
    canonical_imports = (
        set(_imports(canonical_graph)) if canonical_graph is not None else set()
    )
    canonical_iris = (
        set(_declared_ontology_iris(canonical_graph))
        if canonical_graph is not None
        else set()
    )
    import_graph = _index_import_graph(bundle.parsed)
    federated_iris = _federated_iris()
    return _OntologyIndex(
        iri_to_files=iri_to_files,
        canonical_imports=canonical_imports,
        canonical_iris=canonical_iris,
        import_graph=import_graph,
        federated_iris=federated_iris,
        connectivity_anchors=canonical_iris | canonical_imports,
        declared_iris=set(iri_to_files) | federated_iris,
    )


def _check_duplicate_iris(index: _OntologyIndex, violations: list[str]) -> None:
    """Report ontology IRIs declared by more than one file."""
    for iri, files in index.iri_to_files.items():
        if len(files) <= 1:
            continue
        rels = ", ".join(str(_rel(path)) for path in files)
        _fail(
            violations,
            f"[duplicate-iri] ontology IRI <{iri}> declared by multiple files: {rels}",
        )


def _check_registered_imports(index: _OntologyIndex, violations: list[str]) -> None:
    """Ensure every registered federated ontology is imported canonically."""
    for iri in sorted(index.federated_iris - index.canonical_imports):
        _fail(
            violations,
            f"[unlinked-registry] canonical ontology.ttl does not import <{iri}>",
        )


def _check_domain_module(
    *,
    module: Path,
    bundle: _OntologyBundle,
    index: _OntologyIndex,
    violations: list[str],
) -> None:
    """Check one bundled or federated domain module's ontology identity and path."""
    graph = bundle.parsed.get(module)
    if graph is None:
        return  # syntax failure already reported
    iris = _declared_ontology_iris(graph)
    if not iris:
        _fail(
            violations,
            f"[unlinked] {module.name} declares no owl:Ontology IRI — it cannot be "
            f"imported/addressed. Add `<http://knuckles.team/kg/{module.stem.removeprefix('ontology_')}> a owl:Ontology .`",
        )
        return
    if len(iris) > 1:
        _fail(
            violations,
            f"[multi-iri] {module.name} declares >1 owl:Ontology IRI: {iris}",
        )
    if not any(
        _has_import_path(iri, index.import_graph, index.connectivity_anchors)
        for iri in iris
    ):
        _fail(
            violations,
            f"[unlinked] {_rel(module)} ({iris[0]}) has no import path to the "
            "canonical ontology component.",
        )


def _check_domain_modules(
    bundle: _OntologyBundle, index: _OntologyIndex, violations: list[str]
) -> None:
    """Check connectivity for every discovered domain module."""
    for module in bundle.domain_modules:
        _check_domain_module(
            module=module, bundle=bundle, index=index, violations=violations
        )


def _check_connectivity(
    bundle: _OntologyBundle, index: _OntologyIndex, violations: list[str]
) -> None:
    """Run registry and domain-module connectivity checks in gate order."""
    _check_registered_imports(index, violations)
    _check_domain_modules(bundle, index, violations)


def _check_dangling_imports(
    bundle: _OntologyBundle, index: _OntologyIndex, violations: list[str]
) -> None:
    """Reject owned import IRIs that have no local declaration or registration."""
    for path, graph in bundle.parsed.items():
        for imported in _imports(graph):
            if (
                imported.startswith(_OWN_PREFIXES)
                and imported not in index.declared_iris
            ):
                _fail(
                    violations,
                    f"[dangling-import] {_rel(path)} imports <{imported}> which "
                    "resolves to no local ontology file.",
                )


def _check_shape(
    *,
    shape_file: Path,
    graph: Any,
    dependencies: _ValidationDependencies,
    violations: list[str],
) -> None:
    """Force pyshacl to load and compile one shape graph."""
    try:
        dependencies.pyshacl.validate(
            data_graph=dependencies.rdflib.Graph(),
            shacl_graph=graph,
            inference="none",
            abort_on_first=False,
        )
    except Exception as exc:  # noqa: BLE001
        _fail(
            violations,
            f"[shacl] {_rel(shape_file)} is not well-formed SHACL "
            f"({type(exc).__name__})",
        )


def _check_shapes(
    bundle: _OntologyBundle,
    dependencies: _ValidationDependencies,
    violations: list[str],
) -> None:
    """Validate every bundled SHACL shape when the local shape directory exists."""
    if not SHAPES_DIR.exists():
        return
    for shape_file in sorted(path for path in bundle.all_ttls if _is_shape(path)):
        graph = bundle.parsed.get(shape_file)
        if graph is not None:
            _check_shape(
                shape_file=shape_file,
                graph=graph,
                dependencies=dependencies,
                violations=violations,
            )


def _merged_ontology(bundle: _OntologyBundle, rdflib: Any) -> Any:
    """Combine the canonical ontology and domain modules for OWL-RL closure."""
    merged = rdflib.Graph()
    for module in [CANONICAL, *bundle.domain_modules]:
        graph = bundle.parsed.get(module)
        if graph is None:
            continue
        for triple in graph:
            merged.add(triple)
    return merged


def _check_owl_rl(
    *,
    bundle: _OntologyBundle,
    dependencies: _ValidationDependencies,
    notes: list[str],
    violations: list[str],
) -> None:
    """Run OWL-RL closure over the merged ontology and record its size."""
    try:
        merged = _merged_ontology(bundle, dependencies.rdflib)
        dependencies.owlrl.DeductiveClosure(dependencies.owlrl.OWLRL_Semantics).expand(
            merged
        )
        notes.append(f"OWL-RL closure ok ({len(merged)} triples after expansion)")
    except Exception as exc:  # noqa: BLE001
        _fail(
            violations,
            f"[owl-rl] merged ontology breaks OWL-RL closure ({type(exc).__name__})",
        )


def _is_documented(path: Path, lines: list[str]) -> bool:
    """Return whether one ontology asset appears in the library index."""
    display = _rel(path)
    if path in _SOURCE_LABELS:
        provider = display.parts[1]
        relative = Path(*display.parts[2:]).as_posix()
        return any(
            f"`{provider}`" in line and f"`{relative}`" in line for line in lines
        )
    relative = path.relative_to(KG_DIR).as_posix()
    return any(
        f"`{candidate}`" in line
        for line in lines
        for candidate in (path.name, relative)
    )


def _check_documentation(bundle: _OntologyBundle, violations: list[str]) -> None:
    """Ensure every ontology asset is named in the canonical library index."""
    if not LIBRARY_DOC.exists():
        _fail(
            violations,
            f"[docs] ontology library index missing: {LIBRARY_DOC.relative_to(ROOT)}",
        )
        return
    lines = LIBRARY_DOC.read_text().splitlines()
    for path in bundle.all_ttls:
        if not _is_documented(path, lines):
            _fail(
                violations,
                f"[docs] {_rel(path)} is not listed with its owner in the ontology library",
            )


def _report(
    *, parsed_count: int, notes: list[str], violations: list[str], verbose: bool
) -> int:
    """Render the gate result using the historical output contract."""
    if verbose:
        for n in notes:
            print(f"  · {n}")
    if violations:
        print(f"check_ontology: {len(violations)} violation(s):")
        for v in violations:
            print(f"  ✗ {v}")
        return 1
    print(
        f"check_ontology: OK — {parsed_count} ontologies valid, connected, and documented."
    )
    return 0


def check(verbose: bool = False, provider_root: Path | None = None) -> int:
    """Run all ontology validity, connectivity, and documentation checks."""
    dependencies = _load_validation_dependencies()
    if dependencies is None:
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
        bundle, violations = _build_ontology_bundle(provider_root)
    except FleetScanError as exc:
        print(f"check_ontology: provider fleet scan failed ({exc}).")
        return 1
    notes = [f"parsed {len(bundle.parsed)}/{len(bundle.all_ttls)} TTL files"]
    index = _build_ontology_index(bundle)
    _check_duplicate_iris(index, violations)
    _check_connectivity(bundle, index, violations)
    _check_dangling_imports(bundle, index, violations)
    _check_shapes(bundle, dependencies, violations)
    _check_owl_rl(
        bundle=bundle,
        dependencies=dependencies,
        notes=notes,
        violations=violations,
    )
    _check_documentation(bundle, violations)
    return _report(
        parsed_count=len(bundle.parsed),
        notes=notes,
        violations=violations,
        verbose=verbose,
    )


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
