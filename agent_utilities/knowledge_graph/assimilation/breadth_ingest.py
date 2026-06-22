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
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .ingest import ingest_concepts, ingest_documents

# ``CONCEPT:<ID>`` markers declared in source/docs (KG-2.7) — the fallback concept
# source for repos that ship no ``docs/concepts.yaml`` registry.
_CONCEPT_MARKER = re.compile(r"CONCEPT:([A-Z]{2,6}-\d+(?:\.\d+[a-z]?|-\d+)?)")
_CONCEPT_SCAN_EXT = (
    ".py",
    ".rs",
    ".md",
    ".js",
    ".ts",
    ".tsx",
    ".go",
    ".java",
    ".txt",
    ".yaml",
    ".yml",
)

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


def _scan_concept_markers(root: Path, *, max_files: int = 4000) -> set[str]:
    """Collect ``CONCEPT:<ID>`` ids declared in a repo's source/docs (bounded walk)."""
    ids: set[str] = set()
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP]
        for fn in filenames:
            if not fn.endswith(_CONCEPT_SCAN_EXT):
                continue
            seen += 1
            if seen > max_files:
                return ids
            try:
                txt = (Path(dirpath) / fn).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for m in _CONCEPT_MARKER.findall(txt):
                ids.add(m.strip().upper())
    return ids


def discover_concepts(roots: list[str], *, max_depth: int = 3) -> list[dict[str, Any]]:
    """Collect ecosystem capability concepts under ``roots`` for :func:`ingest_concepts`.

    Authoritative source is each repo's ``docs/concepts.yaml`` registry (id + name +
    pillar + status); repos that ship none fall back to a bounded ``CONCEPT:<ID>``
    marker scan (id only). Registry entries win on id collisions. (CONCEPT:KG-2.7)
    """
    import yaml

    found: dict[str, dict[str, Any]] = {}
    for root in roots:
        rp = Path(root)
        if not rp.is_dir():
            continue
        registry = rp / "docs" / "concepts.yaml"
        had_registry = False
        if registry.is_file():
            try:
                data = yaml.safe_load(registry.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                data = {}
            entries = data.get("concepts") if isinstance(data, dict) else data
            if isinstance(entries, list):
                for e in entries:
                    if isinstance(e, dict) and e.get("id"):
                        cid = str(e["id"]).strip().upper()
                        item = dict(e)
                        item["source"] = str(registry)
                        found[cid] = item  # registry is authoritative
                        had_registry = True
        if not had_registry:
            for cid in _scan_concept_markers(rp):
                found.setdefault(
                    cid, {"id": cid, "name": cid, "source": f"{rp}:marker"}
                )
    return list(found.values())


@dataclass
class BreadthReport:
    projects: int = 0
    docs: int = 0
    codebases_ingested: int = 0
    docs_ingested: int = 0
    skipped: int = 0
    concepts: int = 0
    concepts_ingested: int = 0
    manifests: list[dict[str, Any]] = field(default_factory=list)


def run_breadth_ingest(
    engine: Any,
    *,
    library_roots: list[str] | None = None,
    repo_roots: list[str] | None = None,
    docs: list[dict[str, Any]] | None = None,
    codebase_ingest: Callable[[Any, ProjectManifest], bool] | None = None,
    doc_ingest: Callable[[Any, list[dict[str, Any]]], int] | None = None,
    concept_ingest: Callable[[Any, list[dict[str, Any]]], int] | None = None,
) -> BreadthReport:
    """Ingest libraries + repos (codebases) + concepts + docs into the assimilation graph.

    ``codebase_ingest(engine, manifest) -> bool`` (ingested vs skipped) defaults to
    the real ``IngestionEngine`` codebase path; ``doc_ingest`` /  ``concept_ingest``
    default to :func:`assimilation.ingest.ingest_documents` / ``ingest_concepts``.
    All injectable for testing. The concept pass is what gives the golden-loop gap
    matcher its "already-built" comparison surface (CONCEPT:KG-2.7).
    """
    cb = codebase_ingest or _default_codebase_ingest
    di = doc_ingest or _default_doc_ingest
    ci = concept_ingest or _default_concept_ingest

    report = BreadthReport()
    roots_all = (library_roots or []) + (repo_roots or [])
    projects: list[ProjectManifest] = []
    for r in roots_all:
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

    # Ecosystem capability registry → built Concept nodes (the gap-matcher's
    # "already-built" side). Without this, assimilate has nothing to compare
    # research against and every paper is an open gap.
    concepts = discover_concepts(roots_all)
    report.concepts = len(concepts)
    if concepts:
        try:
            report.concepts_ingested = int(ci(engine, concepts))
        except Exception:  # pragma: no cover - best-effort
            pass

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


def _default_concept_ingest(engine: Any, concepts: list[dict[str, Any]]) -> int:
    """Ingest ecosystem concepts as Concept nodes; returns new + updated count."""
    r = ingest_concepts(engine, concepts)
    return r.ingested + r.updated


def _is_self_repo(path: str | Path) -> bool:
    """True if ``path`` is the agent-utilities checkout itself (the self-ingest).

    The self-repo is the one box that is routinely DIRTY (active development),
    so it is the case worth scoping to git-status-modified files. Detected by the
    presence of the ``agent_utilities`` package at the root. (CONCEPT:KG-2.150)
    """
    p = Path(path)
    return (p / "agent_utilities" / "__init__.py").is_file()


def _git_modified_source_files(path: str | Path) -> list[str]:
    """Git-status-modified (staged/unstaged/untracked) SOURCE files under ``path``.

    The dirty-tree analogue of the engine's clean-tree ``git diff`` delta: parses
    ``git status --porcelain`` and keeps only added/modified/untracked files whose
    extension is engine-supported and outside vendored/build dirs. Renames keep the
    new name; deletions are dropped. Returns absolute paths; ``[]`` if git can't
    answer or nothing source-relevant changed. (CONCEPT:KG-2.150)
    """
    import subprocess

    from ..enrichment.pipeline import SOURCE_EXTENSIONS

    root = Path(path)
    try:
        r = subprocess.run(  # nosec B607 B603
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if r.returncode != 0:
        return []
    out: list[str] = []
    for line in r.stdout.splitlines():
        if not line or len(line) < 4:
            continue
        status, rest = line[:2], line[3:]
        if "D" in status:  # a pure deletion — nothing to (re)parse
            continue
        rel = rest.split(" -> ", 1)[1] if " -> " in rest else rest  # rename → new
        rel = rel.strip().strip('"')
        fp = (root / rel).resolve()
        if fp.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        if any(part in _SKIP for part in fp.parts):
            continue
        if fp.is_file():
            out.append(str(fp))
    return out


def _default_codebase_ingest(engine: Any, manifest: ProjectManifest) -> bool:
    """Submit a codebase ingest via the live engine (content-addressed skip).

    Git-SHA pre-skip (CONCEPT:KG-2.8): an always-on breadth loop re-runs every few
    minutes, but a clean git work-tree still at the HEAD we last ingested has nothing
    new. Mirror the engine's ``codebase_git`` delta watermark *exactly* (same
    ``_git_head_sha`` / ``_git_worktree_clean`` helpers + manifest key) so we skip the
    task enqueue — and the whole-tree stat-walk the engine's seen-check would do —
    only when the engine would have skipped anyway. (CONCEPT:KG-2.150 extends this so
    the pre-skip is the authoritative gate: a clean tree at the ingested HEAD never
    enqueues a task at all — saving the parse/chunk, not just the write.)

    Dirty-tree self-ingest scoping (CONCEPT:KG-2.150): the agent-utilities checkout
    is routinely dirty, which disables the engine's clean-tree git-diff and forces a
    FULL re-walk every tick. For that one repo, scope the submitted task to just the
    git-status-modified source files (``only_files``) so the engine parses only those.

    Non-git, dirty (non-self), or first-ingest repos fall through to a whole-repo
    submit; the engine then runs its own git-diff + per-file content-hash delta.
    Returns True if a task was submitted (likely-changed), False if skipped.
    """
    submit = getattr(engine, "submit_task", None)
    if not callable(submit):
        return False
    extra_meta: dict[str, Any] | None = None
    try:
        from pathlib import Path

        from ..ingestion.engine import _git_head_sha, _git_worktree_clean
        from ..ingestion.manifest import DeltaManifest

        head = _git_head_sha(manifest.path)
        clean = _git_worktree_clean(manifest.path) if head else False
        if head and clean:
            gc = getattr(engine, "graph_compute", None)
            gname = getattr(gc, "graph_name", None) or "__commons__"
            repo_key = str(Path(manifest.path).resolve())
            dm = DeltaManifest(backend=getattr(engine, "backend", None))
            if dm.get(gname, "codebase_git", repo_key) == head:
                return False  # unchanged HEAD + clean tree → no task churn this tick
        elif head and not clean and _is_self_repo(manifest.path):
            # Dirty self-repo: scope to the git-status-modified source files. An
            # empty modified set means the dirty bits are all non-source (docs,
            # build artefacts) → nothing to re-parse, so skip the enqueue entirely.
            modified = _git_modified_source_files(manifest.path)
            if not modified:
                return False
            extra_meta = {"only_files": modified}
    except Exception:  # noqa: BLE001 — best-effort; fall through to submit
        pass
    submit(
        target_path=manifest.path,
        is_codebase=True,
        task_type="codebase",
        provenance={"language": manifest.language, "pillars": manifest.pillars},
        **({"extra_meta": extra_meta} if extra_meta else {}),
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
