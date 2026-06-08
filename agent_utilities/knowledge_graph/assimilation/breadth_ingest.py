#!/usr/bin/python
from __future__ import annotations

"""Breadth-ingest orchestration (CONCEPT:KG-2.7).

Brings the *whole* comparison corpus into the assimilation graph: the OSS library
categories, our ~62 ecosystem repos, and a documents batch. Classification +
discovery are pure filesystem reads (testable on a temp tree); the heavy codebase
ingest is injected (default = the real `IngestionEngine` codebase path) so the
orchestration is unit-testable without a live engine, and idempotent because the
underlying ingest is content-addressed (unchanged sources skip).

* :func:`discover_projects` — find project roots under a tree (by build-file marker).
* :func:`classify_project` — language + domain + target pillars (dir-name heuristics).
* :func:`organize_libraries` — write a ``manifest.json`` per project (non-destructive).
* :func:`run_breadth_ingest` — orchestrate codebase + document ingest into the graph.

Concept: breadth-ingest
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .ingest import ingest_documents

# build-file → language
_LANG_MARKERS: dict[str, str] = {
    "Cargo.toml": "rust",
    "pyproject.toml": "python",
    "setup.py": "python",
    "package.json": "node",
    "go.mod": "go",
    "pom.xml": "java",
    "build.gradle": "java",
}
# path-keyword → target pillar(s) (ORCH/KG/AHE/ECO/OS)
_DOMAIN_PILLARS: dict[str, tuple[str, ...]] = {
    "memory": ("KG",),
    "rag": ("KG",),
    "kg": ("KG",),
    "graph": ("KG",),
    "knowledge": ("KG",),
    "agent": ("ORCH",),
    "orchestr": ("ORCH",),
    "swarm": ("ORCH",),
    "council": ("ORCH",),
    "rlm": ("AHE",),
    "rl": ("AHE",),
    "evolu": ("AHE",),
    "prompt": ("AHE",),
    "design": ("AHE",),
    "quant": ("ECO",),
    "trad": ("ECO",),
    "crypto": ("ECO",),
    "poly": ("ECO",),
    "finance": ("ECO",),
    "eunomia": ("OS",),
    "infra": ("OS",),
    "deploy": ("OS",),
    "security": ("OS",),
}
_SKIP = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    "site-packages",
    "target",
    "dist",
    "build",
}


@dataclass
class ProjectManifest:
    name: str
    path: str
    language: str = "unknown"
    domain: str = ""
    pillars: list[str] = field(default_factory=list)


def classify_project(path: str | Path) -> ProjectManifest:
    """Classify a project dir: language (build marker) + pillars (path keywords)."""
    p = Path(path)
    language = "unknown"
    for marker, lang in _LANG_MARKERS.items():
        if (p / marker).exists():
            language = lang
            break
    haystack = "/".join(part.lower() for part in p.parts[-3:])
    pillars: list[str] = []
    domain = ""
    for kw, pil in _DOMAIN_PILLARS.items():
        if kw in haystack:
            for x in pil:
                if x not in pillars:
                    pillars.append(x)
            domain = domain or kw
    return ProjectManifest(
        name=p.name, path=str(p), language=language, domain=domain, pillars=pillars
    )


def discover_projects(root: str | Path, *, max_depth: int = 3) -> list[ProjectManifest]:
    """Find project roots (dirs containing a build-file marker) under ``root``."""
    root = Path(root)
    if not root.is_dir():
        return []
    found: dict[str, ProjectManifest] = {}
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in _walk(root):
        depth = len(Path(dirpath).parts) - root_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in _SKIP]
        if any(m in filenames for m in _LANG_MARKERS):
            key = str(Path(dirpath).resolve())
            if key not in found:
                found[key] = classify_project(dirpath)
            dirnames[:] = []  # don't descend into a project's own subdirs
    return list(found.values())


def _walk(root: Path):
    import os

    yield from os.walk(root)


def organize_libraries(
    root: str | Path, *, write: bool = True
) -> list[ProjectManifest]:
    """Classify every project under ``root`` and (optionally) write a manifest.json."""
    manifests = discover_projects(root)
    if write:
        for m in manifests:
            try:
                (Path(m.path) / "manifest.json").write_text(
                    json.dumps(asdict(m), indent=2), encoding="utf-8"
                )
            except OSError:
                pass  # read-only / vendored — classification still returned
    return manifests


@dataclass
class BreadthReport:
    projects: int = 0
    docs: int = 0
    codebases_ingested: int = 0
    docs_ingested: int = 0
    skipped: int = 0
    manifests: list[dict[str, Any]] = field(default_factory=list)


def run_breadth_ingest(
    engine: Any,
    *,
    library_roots: list[str] | None = None,
    repo_roots: list[str] | None = None,
    docs: list[dict[str, Any]] | None = None,
    codebase_ingest: Callable[[Any, ProjectManifest], bool] | None = None,
    doc_ingest: Callable[[Any, list[dict[str, Any]]], int] | None = None,
) -> BreadthReport:
    """Ingest libraries + repos (codebases) + docs into the assimilation graph.

    ``codebase_ingest(engine, manifest) -> bool`` (ingested vs skipped) defaults to
    the real ``IngestionEngine`` codebase path; ``doc_ingest`` defaults to
    :func:`assimilation.ingest.ingest_documents`. Both injectable for testing.
    """
    cb = codebase_ingest or _default_codebase_ingest
    di = doc_ingest or _default_doc_ingest

    report = BreadthReport()
    projects: list[ProjectManifest] = []
    for r in (library_roots or []) + (repo_roots or []):
        projects.extend(discover_projects(r))
    report.projects = len(projects)
    report.manifests = [asdict(m) for m in projects]
    for m in projects:
        try:
            if cb(engine, m):
                report.codebases_ingested += 1
            else:
                report.skipped += 1
        except Exception:  # pragma: no cover - per-project best-effort
            report.skipped += 1

    if docs:
        report.docs = len(docs)
        try:
            report.docs_ingested = int(di(engine, docs))
        except Exception:  # pragma: no cover
            pass
    return report


def _default_doc_ingest(engine: Any, docs: list[dict[str, Any]]) -> int:
    """Ingest docs as Requirement nodes; returns newly-ingested + updated count."""
    r = ingest_documents(engine, docs)
    return r.ingested + r.updated


def _default_codebase_ingest(engine: Any, manifest: ProjectManifest) -> bool:
    """Submit a codebase ingest via the live engine (content-addressed skip)."""
    submit = getattr(engine, "submit_task", None)
    if not callable(submit):
        return False
    submit(
        target_path=manifest.path,
        is_codebase=True,
        task_type="codebase",
        provenance={"language": manifest.language, "pillars": manifest.pillars},
    )
    return True


__all__ = [
    "ProjectManifest",
    "BreadthReport",
    "classify_project",
    "discover_projects",
    "organize_libraries",
    "run_breadth_ingest",
]
