#!/usr/bin/python
from __future__ import annotations

"""Breadth-ingest orchestration (CONCEPT:AU-KG.query.vendor-agnostic-traversal).

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
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .ingest import ingest_concepts, ingest_documents

# Canonical semantic ``CONCEPT:<ID>`` markers declared in source/docs — the
# fallback concept source for repos that ship no ``docs/concepts.yaml`` registry.
_CONCEPT_MARKER = re.compile(
    r"CONCEPT:([A-Z]{2}-(?:ORCH|KG|AHE|ECO|OS|GBOT)\."
    r"[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9_-])"
)
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


def _record_project_dir(
    dirpath: str,
    dirnames: list[str],
    filenames: list[str],
    *,
    root_depth: int,
    max_depth: int,
    found: dict[str, ProjectManifest],
) -> None:
    """Classify one walked directory in place, pruning ``dirnames`` as needed."""
    depth = len(Path(dirpath).parts) - root_depth
    if depth > max_depth:
        dirnames[:] = []
        return
    dirnames[:] = [d for d in dirnames if d not in _SKIP]
    if any(m in filenames for m in _LANG_MARKERS):
        key = str(Path(dirpath).resolve())
        if key not in found:
            found[key] = classify_project(dirpath)
        dirnames[:] = []  # don't descend into a project's own subdirs


def discover_projects(root: str | Path, *, max_depth: int = 3) -> list[ProjectManifest]:
    """Find project roots (dirs containing a build-file marker) under ``root``."""
    root = Path(root)
    if not root.is_dir():
        return []
    found: dict[str, ProjectManifest] = {}
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in _walk(root):
        _record_project_dir(
            dirpath,
            dirnames,
            filenames,
            root_depth=root_depth,
            max_depth=max_depth,
            found=found,
        )
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
            except OSError:  # noqa: BLE001 — read-only/vendor project remains classified
                pass  # read-only / vendored — classification still returned
    return manifests


def _scan_file_for_concepts(path: Path) -> set[str] | None:
    """Read one file and extract its ``CONCEPT:<ID>`` ids; None if unreadable."""
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    return {m.strip().upper() for m in _CONCEPT_MARKER.findall(txt)}


def _scan_directory_for_concepts(
    dirpath: str, filenames: list[str], *, seen: int, max_files: int
) -> tuple[set[str], int, bool]:
    """Scan one directory's candidate files; returns (ids, new_seen, hit_bound)."""
    ids: set[str] = set()
    for fn in filenames:
        if not fn.endswith(_CONCEPT_SCAN_EXT):
            continue
        seen += 1
        if seen > max_files:
            return ids, seen, True
        found = _scan_file_for_concepts(Path(dirpath) / fn)
        if found:
            ids |= found
    return ids, seen, False


def _scan_concept_markers(root: Path, *, max_files: int = 4000) -> set[str]:
    """Collect ``CONCEPT:<ID>`` ids declared in a repo's source/docs (bounded walk)."""
    ids: set[str] = set()
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP]
        found, seen, hit_bound = _scan_directory_for_concepts(
            dirpath, filenames, seen=seen, max_files=max_files
        )
        ids |= found
        if hit_bound:
            return ids
    return ids


def _load_concept_registry(registry: Path) -> dict[str, dict[str, Any]]:
    """Parse one repo's ``docs/concepts.yaml`` registry into id -> concept entry."""
    import yaml

    try:
        data = yaml.safe_load(registry.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        data = {}
    entries = data.get("concepts") if isinstance(data, dict) else data
    result: dict[str, dict[str, Any]] = {}
    if isinstance(entries, list):
        for e in entries:
            if isinstance(e, dict) and e.get("id"):
                cid = str(e["id"]).strip().upper()
                item = dict(e)
                item["source"] = str(registry)
                result[cid] = item
    return result


def _marker_fallback_concepts(rp: Path) -> dict[str, dict[str, Any]]:
    """Fallback ``CONCEPT:<ID>`` marker scan for a repo with no usable registry."""
    return {
        cid: {"id": cid, "name": cid, "source": f"{rp}:marker"}
        for cid in _scan_concept_markers(rp)
    }


def discover_concepts(roots: list[str], *, max_depth: int = 3) -> list[dict[str, Any]]:
    """Collect ecosystem capability concepts under ``roots`` for :func:`ingest_concepts`.

    Authoritative source is each repo's ``docs/concepts.yaml`` registry (id + name +
    pillar + status); repos that ship none fall back to a bounded ``CONCEPT:<ID>``
    marker scan (id only). Registry entries win on id collisions. (CONCEPT:AU-KG.query.vendor-agnostic-traversal)
    """
    found: dict[str, dict[str, Any]] = {}
    for root in roots:
        rp = Path(root)
        if not rp.is_dir():
            continue
        registry = rp / "docs" / "concepts.yaml"
        registry_entries = (
            _load_concept_registry(registry) if registry.is_file() else {}
        )
        if registry_entries:
            found.update(registry_entries)  # registry is authoritative
        else:
            for cid, item in _marker_fallback_concepts(rp).items():
                found.setdefault(cid, item)
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


def _ingest_project_codebases(
    engine: Any,
    projects: list[ProjectManifest],
    codebase_ingest: Callable[[Any, ProjectManifest], bool],
    report: BreadthReport,
) -> None:
    for m in projects:
        try:
            if codebase_ingest(engine, m):
                report.codebases_ingested += 1
            else:
                report.skipped += 1
        except Exception:  # noqa: BLE001 — one project must not abort fleet ingestion
            report.skipped += 1


def _ingest_ecosystem_concepts(
    engine: Any,
    roots_all: list[str],
    concept_ingest: Callable[[Any, list[dict[str, Any]]], int],
    report: BreadthReport,
) -> None:
    # Ecosystem capability registry → built Concept nodes (the gap-matcher's
    # "already-built" side). Without this, assimilate has nothing to compare
    # research against and every paper is an open gap.
    concepts = discover_concepts(roots_all)
    report.concepts = len(concepts)
    if concepts:
        try:
            report.concepts_ingested = int(concept_ingest(engine, concepts))
        except Exception:  # noqa: BLE001 — optional concept ingest is best-effort
            pass


def _ingest_docs_batch(
    engine: Any,
    docs: list[dict[str, Any]] | None,
    doc_ingest: Callable[[Any, list[dict[str, Any]]], int],
    report: BreadthReport,
) -> None:
    if not docs:
        return
    report.docs = len(docs)
    try:
        report.docs_ingested = int(doc_ingest(engine, docs))
    except Exception:  # noqa: BLE001 — optional document ingest is best-effort
        pass


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
    matcher its "already-built" comparison surface (CONCEPT:AU-KG.query.vendor-agnostic-traversal).
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

    _ingest_project_codebases(engine, projects, cb, report)
    _ingest_ecosystem_concepts(engine, roots_all, ci, report)
    _ingest_docs_batch(engine, docs, di, report)
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
    presence of the ``agent_utilities`` package at the root. (CONCEPT:AU-KG.ingest.agent-utilities-checkout)
    """
    p = Path(path)
    return (p / "agent_utilities" / "__init__.py").is_file()


def _run_git_status(root: Path) -> subprocess.CompletedProcess[str] | None:
    """Run ``git status --porcelain`` bounded, returning None on any failure."""
    try:
        return subprocess.run(  # nosec B607 B603
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _source_file_from_status_line(
    line: str, root: Path, source_extensions: frozenset[str]
) -> str | None:
    """Resolve one ``git status --porcelain`` line to an absolute source path."""
    if not line or len(line) < 4:
        return None
    status, rest = line[:2], line[3:]
    if "D" in status:  # a pure deletion — nothing to (re)parse
        return None
    rel = rest.split(" -> ", 1)[1] if " -> " in rest else rest  # rename → new
    rel = rel.strip().strip('"')
    fp = (root / rel).resolve()
    if fp.suffix.lower() not in source_extensions:
        return None
    if any(part in _SKIP for part in fp.parts):
        return None
    return str(fp) if fp.is_file() else None


def _git_modified_source_files(path: str | Path) -> list[str]:
    """Git-status-modified (staged/unstaged/untracked) SOURCE files under ``path``.

    The dirty-tree analogue of the engine's clean-tree ``git diff`` delta: parses
    ``git status --porcelain`` and keeps only added/modified/untracked files whose
    extension is engine-supported and outside vendored/build dirs. Renames keep the
    new name; deletions are dropped. Returns absolute paths; ``[]`` if git can't
    answer or nothing source-relevant changed. (CONCEPT:AU-KG.ingest.agent-utilities-checkout)
    """
    from ..enrichment.pipeline import SOURCE_EXTENSIONS

    root = Path(path)
    r = _run_git_status(root)
    if r is None or r.returncode != 0:
        return []
    out: list[str] = []
    for line in r.stdout.splitlines():
        found = _source_file_from_status_line(line, root, SOURCE_EXTENSIONS)
        if found is not None:
            out.append(found)
    return out


def _clean_tree_already_ingested(
    engine: Any, manifest: ProjectManifest, head: str
) -> bool:
    """Whether the delta watermark already recorded this exact clean HEAD."""
    from ..ingestion.manifest import DeltaManifest

    gc = getattr(engine, "graph_compute", None)
    gname = getattr(gc, "graph_name", None) or "__commons__"
    repo_key = str(Path(manifest.path).resolve())
    dm = DeltaManifest(backend=getattr(engine, "backend", None))
    return bool(dm.get(gname, "codebase_git", repo_key) == head)


def _pre_skip_and_extra_meta(
    engine: Any, manifest: ProjectManifest
) -> tuple[bool, dict[str, Any] | None]:
    """Return ``(should_skip, extra_meta)`` per the git-delta pre-skip rules.

    Best-effort: any failure resolving git/backend state falls through to a full
    submit (``should_skip=False, extra_meta=None``), matching the original bare
    ``except: pass``.
    """
    try:
        from ..ingestion.engine import _git_head_sha, _git_worktree_clean

        head = _git_head_sha(manifest.path)
        clean = _git_worktree_clean(manifest.path) if head else False
        if head and clean:
            if _clean_tree_already_ingested(engine, manifest, head):
                return (
                    True,
                    None,
                )  # unchanged HEAD + clean tree → no task churn this tick
            return False, None
        if head and not clean and _is_self_repo(manifest.path):
            # Dirty self-repo: scope to the git-status-modified source files. An
            # empty modified set means the dirty bits are all non-source (docs,
            # build artefacts) → nothing to re-parse, so skip the enqueue entirely.
            modified = _git_modified_source_files(manifest.path)
            if not modified:
                return True, None
            return False, {"only_files": modified}
    except Exception:  # noqa: BLE001 — best-effort; fall through to submit
        pass
    return False, None


def _default_codebase_ingest(engine: Any, manifest: ProjectManifest) -> bool:
    """Submit a codebase ingest via the live engine (content-addressed skip).

    Git-SHA pre-skip (CONCEPT:EG-KG.storage.nonblocking-checkpoint): an always-on breadth loop re-runs every few
    minutes, but a clean git work-tree still at the HEAD we last ingested has nothing
    new. Mirror the engine's ``codebase_git`` delta watermark *exactly* (same
    ``_git_head_sha`` / ``_git_worktree_clean`` helpers + manifest key) so we skip the
    task enqueue — and the whole-tree stat-walk the engine's seen-check would do —
    only when the engine would have skipped anyway. (CONCEPT:AU-KG.ingest.agent-utilities-checkout extends this so
    the pre-skip is the authoritative gate: a clean tree at the ingested HEAD never
    enqueues a task at all — saving the parse/chunk, not just the write.)

    Dirty-tree self-ingest scoping (CONCEPT:AU-KG.ingest.agent-utilities-checkout): the agent-utilities checkout
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
    should_skip, extra_meta = _pre_skip_and_extra_meta(engine, manifest)
    if should_skip:
        return False
    submit(
        target_path=manifest.path,
        is_codebase=True,
        task_type="codebase",
        priority=3,
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
