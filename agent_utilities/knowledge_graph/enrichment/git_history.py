"""Git commit-history → Knowledge Graph (CONCEPT:AU-KG.ingest.normal-codebase-ingest-also).

A repo's commit history *is* a graph: commits → authors → files, evolving over
time — which fits the KG natively. Tools like Gource / SourceTree only *render*
that evolution; we INGEST it as first-class graph data so codebase evolution
becomes a free native query: who-owns-what, change-coupling, churn hotspots,
per-file timelines, temporal "as-of" state, blast-radius.

Design — FAST + intelligent (the user stressed FAST):

* **One ``git log --numstat`` pass** with a machine-parseable ``--pretty``
  format (NOT a subprocess per commit, the Gource-slow way). Streamed + parsed
  in memory, batch-written through the engine's bulk path.
* **Auto-bounded** for huge histories (``max_count``, default 5000) with a
  ``--since`` option; capping is detected and logged.
* **Delta / idempotent:** commits already in the KG (by sha) are skipped, so a
  no-change re-ingest is a no-op. Authors/files upsert by stable id.

Graph model (linked to the same ``file:<path>`` ids the structural code ingest
and KG-2.104 change-coupling already use, so history and structure are ONE
graph):

* ``:Commit``  ``commit:<sha>``   — sha, message, timestamp, parents, author
* ``:Author``  ``author:<email>`` — name, email
* ``:File``    ``file:<path>``    — path + churn aggregates (hotspots)
* ``AUTHORED`` author → commit
* ``PARENT``   commit → commit     (the DAG)
* ``TOUCHED``  commit → file       (+insertions/deletions/renamed_from)
* ``FILE_CHANGES_WITH`` file ↔ file (change-coupling, reused from KG-2.104)
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .git_coupling import DEFAULT_MIN_SUPPORT, parse_change_coupling

logger = logging.getLogger(__name__)

# Auto-bound: histories larger than this are capped (newest-first) unless the
# caller raises it. 5000 commits keeps even a busy repo's ingest sub-second.
DEFAULT_MAX_COMMITS = 5000

# Field/record separators chosen to never appear in git metadata.
_RS = "\x1e"  # record separator — marks the start of each commit
_US = "\x1f"  # unit separator — between header fields
# Strict-ISO author date, unix author timestamp, subject (single line).
_PRETTY = _RS + _US.join(["%H", "%P", "%an", "%ae", "%aI", "%at", "%s"])


@dataclass
class FileChange:
    """A single file's delta within one commit (from ``--numstat``)."""

    path: str
    insertions: int = 0
    deletions: int = 0
    renamed_from: str | None = None


@dataclass
class CommitRecord:
    """One commit, fully parsed from the single ``git log`` pass."""

    sha: str
    parents: list[str]
    author_name: str
    author_email: str
    timestamp: int  # unix seconds
    iso_date: str
    subject: str
    files: list[FileChange] = field(default_factory=list)


@dataclass
class ExtractResult:
    """Outcome of the one-pass extraction."""

    commits: list[CommitRecord]
    capped: bool
    elapsed_s: float


def _clean_path(p: str) -> str:
    return p.replace("//", "/").strip()


def _normalize_rename(path: str) -> tuple[str, str | None]:
    """Resolve git ``--numstat`` rename notation to ``(new_path, old_path)``.

    Git emits renames as ``old => new``, or the compact brace form
    ``dir/{old => new}/file`` / ``{old => new}``. We return the post-rename path
    (so a file's identity follows its current location) plus the old path.
    """
    if "=>" not in path:
        return path, None
    m = re.search(r"\{(.*?) => (.*?)\}", path)
    if m:
        old = path[: m.start()] + m.group(1) + path[m.end() :]
        new = path[: m.start()] + m.group(2) + path[m.end() :]
        return _clean_path(new), _clean_path(old)
    parts = path.split(" => ")
    if len(parts) == 2:
        return _clean_path(parts[1]), _clean_path(parts[0])
    return _clean_path(path), None


def _parse_log(text: str) -> list[CommitRecord]:
    """Parse the raw ``git log --numstat`` stream into ``CommitRecord``s."""
    commits: list[CommitRecord] = []
    for block in text.split(_RS):
        block = block.strip("\n")
        if not block:
            continue
        lines = block.split("\n")
        fields = lines[0].split(_US)
        if len(fields) < 7:
            continue
        sha, parents_s, an, ae, iso, at, subject = fields[:7]
        try:
            ts = int(at) if at else 0
        except ValueError:
            ts = 0
        files: list[FileChange] = []
        for ln in lines[1:]:
            if not ln.strip():
                continue
            parts = ln.split("\t")
            if len(parts) < 3:
                continue
            add_s, del_s = parts[0], parts[1]
            raw_path = "\t".join(parts[2:])
            # Binary files report "-" for adds/dels.
            adds = 0 if add_s.strip() in ("-", "") else int(add_s)
            dels = 0 if del_s.strip() in ("-", "") else int(del_s)
            new_path, old_path = _normalize_rename(raw_path)
            files.append(FileChange(new_path, adds, dels, old_path))
        commits.append(
            CommitRecord(
                sha=sha,
                parents=parents_s.split() if parents_s else [],
                author_name=an,
                author_email=ae,
                timestamp=ts,
                iso_date=iso,
                subject=subject,
                files=files,
            )
        )
    return commits


def extract_commits(
    repo_path: str,
    *,
    max_count: int = DEFAULT_MAX_COMMITS,
    since: str | None = None,
) -> ExtractResult:
    """Extract commit history in ONE ``git log --numstat`` subprocess.

    Returns an empty result when ``repo_path`` is not a git work-tree or git is
    unavailable (callers degrade gracefully). When more than ``max_count``
    commits exist the newest ``max_count`` are kept and ``capped`` is True.
    """
    cmd = [
        "git",
        "-C",
        repo_path,
        "log",
        "--no-color",
        "--numstat",
        f"--pretty=format:{_PRETTY}",
    ]
    # Request one extra so we can *detect* (and report) capping cheaply.
    fetch = max_count + 1 if max_count and max_count > 0 else 0
    if fetch:
        cmd.append(f"--max-count={fetch}")
    if since:
        cmd.append(f"--since={since}")
    t0 = time.perf_counter()
    try:
        out = subprocess.run(  # nosec B607 B603
            cmd, capture_output=True, text=True, timeout=300, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ExtractResult([], False, 0.0)
    if out.returncode != 0:
        return ExtractResult([], False, time.perf_counter() - t0)
    commits = _parse_log(out.stdout)
    capped = bool(max_count and len(commits) > max_count)
    if capped:
        commits = commits[:max_count]
    return ExtractResult(commits, capped, time.perf_counter() - t0)


def aggregate_churn(commits: list[CommitRecord]) -> dict[str, dict[str, Any]]:
    """Per-file churn aggregates (the hotspot signal) over ``commits``.

    For each file: number of commits touching it, total insertions/deletions,
    the most-recent commit timestamp, and the count of distinct authors — the
    exact signals "which files are churn hotspots / bus-factor risks" needs.
    """
    churn: dict[str, dict[str, Any]] = {}
    for c in commits:
        author = c.author_email or c.author_name
        for f in c.files:
            d = churn.setdefault(
                f.path,
                {
                    "commits": 0,
                    "insertions": 0,
                    "deletions": 0,
                    "last_ts": 0,
                    "authors": set(),
                },
            )
            d["commits"] += 1
            d["insertions"] += f.insertions
            d["deletions"] += f.deletions
            d["last_ts"] = max(d["last_ts"], c.timestamp)
            d["authors"].add(author)
    return churn


def existing_commit_shas(backend: Any, repo_name: str | None = None) -> set[str]:
    """Shas of commits already in the KG (the delta watermark for re-ingest).

    Scans ``:Commit`` nodes via the engine label index; filters to ``repo_name``
    when given so per-repo re-ingest is correctly scoped. Best-effort — returns
    an empty set (full ingest) on any read failure.
    """
    out: set[str] = set()
    scan = getattr(backend, "nodes_by_label", None)
    if not callable(scan):
        return out
    try:
        rows = scan("Commit") or []
    except Exception:  # noqa: BLE001 — read best-effort
        return out
    for nid, props in rows:
        props = props or {}
        if repo_name and props.get("repo") not in (None, "", repo_name):
            continue
        sha = props.get("sha") or (nid.split(":", 1)[1] if ":" in str(nid) else nid)
        if sha:
            out.add(str(sha))
    return out


def ingest_commit_history(
    backend: Any,
    repo_path: str,
    *,
    repo_name: str | None = None,
    existing_shas: set[str] | None = None,
    min_support: int = DEFAULT_MIN_SUPPORT,
    max_count: int = DEFAULT_MAX_COMMITS,
    since: str | None = None,
) -> dict[str, Any]:
    """Ingest a repo's commit history into the KG as a graph (CONCEPT:AU-KG.ingest.normal-codebase-ingest-also).

    Fast: ONE ``git log`` pass, batch-written through the engine bulk path.
    Delta: commits already present (``existing_shas``) are skipped; a re-ingest
    with no new commits is a no-op. Churn + change-coupling aggregates are
    recomputed over the full window and upserted idempotently so incremental
    ingests keep them correct.

    Returns a counts dict (``commits``/``authors``/``files``/edges/``capped``…).
    """
    from .pipeline import _BatchedBackend  # local import avoids an import cycle

    repo_name = repo_name or Path(repo_path).name
    res = extract_commits(repo_path, max_count=max_count, since=since)
    if not res.commits:
        return {
            "commits": 0,
            "authors": 0,
            "files": 0,
            "touched_edges": 0,
            "parent_edges": 0,
            "coupling_edges": 0,
            "skipped": 0,
            "no_op": True,
            "capped": False,
            "commits_per_sec": 0.0,
        }

    existing = existing_shas or set()
    new_commits = [c for c in res.commits if c.sha not in existing]
    if not new_commits:
        # True no-op: nothing new since last ingest → no writes at all.
        return {
            "commits": 0,
            "authors": 0,
            "files": 0,
            "touched_edges": 0,
            "parent_edges": 0,
            "coupling_edges": 0,
            "skipped": len(res.commits),
            "no_op": True,
            "capped": res.capped,
            "commits_per_sec": (
                round(len(res.commits) / res.elapsed_s, 1) if res.elapsed_s else 0.0
            ),
        }

    batched = _BatchedBackend(backend, batch_size=1000)
    authors_written: set[str] = set()
    touched = 0
    parents = 0

    for c in new_commits:
        cid = f"commit:{c.sha}"
        batched.add_node(
            cid,
            type="Commit",
            sha=c.sha,
            message=c.subject,
            timestamp=c.timestamp,
            iso_date=c.iso_date,
            author_email=c.author_email,
            author_name=c.author_name,
            parents=",".join(c.parents),
            repo=repo_name,
            files_changed=len(c.files),
            insertions=sum(f.insertions for f in c.files),
            deletions=sum(f.deletions for f in c.files),
        )
        aid = f"author:{c.author_email or c.author_name}"
        if aid not in authors_written:
            batched.add_node(
                aid, type="Author", name=c.author_name, email=c.author_email
            )
            authors_written.add(aid)
        batched.add_edge(aid, cid, rel_type="AUTHORED")
        for p in c.parents:
            batched.add_edge(cid, f"commit:{p}", rel_type="PARENT")
            parents += 1
        for f in c.files:
            props: dict[str, Any] = {
                "insertions": f.insertions,
                "deletions": f.deletions,
            }
            if f.renamed_from:
                props["renamed_from"] = f.renamed_from
            batched.add_edge(cid, f"file:{f.path}", rel_type="TOUCHED", **props)
            touched += 1

    # File nodes carry churn aggregates (hotspots) computed over the FULL window
    # — written once with the full props (upsert), so incremental ingests refresh
    # them. ``_BatchedBackend.flush`` writes all nodes before any edges, so the
    # TOUCHED / FILE_CHANGES_WITH endpoints always exist.
    churn = aggregate_churn(res.commits)
    for path, d in churn.items():
        batched.add_node(
            f"file:{path}",
            type="File",
            path=path,
            repo=repo_name,
            commit_count=d["commits"],
            churn_insertions=d["insertions"],
            churn_deletions=d["deletions"],
            churn=d["insertions"] + d["deletions"],
            last_commit_ts=d["last_ts"],
            author_count=len(d["authors"]),
        )

    # Change-coupling (KG-2.104) — files that co-change get a symmetric
    # FILE_CHANGES_WITH edge. Recomputed over the full window for correctness.
    coupling = parse_change_coupling(
        [[f.path for f in c.files] for c in res.commits], min_support
    )
    for e in coupling:
        batched.add_edge(e.source, e.target, rel_type=e.rel_type, **e.props)

    batched.flush()

    cps = round(len(res.commits) / res.elapsed_s, 1) if res.elapsed_s else 0.0
    if res.capped:
        logger.info(
            "[KG-2.282] commit-history capped at %d commits for %s "
            "(older history skipped; raise commit_history_max to ingest more)",
            max_count,
            repo_name,
        )
    return {
        "commits": len(new_commits),
        "authors": len(authors_written),
        "files": len(churn),
        "touched_edges": touched,
        "parent_edges": parents,
        "coupling_edges": len(coupling),
        "skipped": len(res.commits) - len(new_commits),
        "no_op": False,
        "capped": res.capped,
        "commits_per_sec": cps,
    }


# ── Evolution query surface (CONCEPT:AU-KG.enrichment.query-ingested-commit-history) ───────────────────────────────


def _rows(backend: Any, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Read-only query helper, tolerant of the engine read API; never raises."""
    try:
        ex = getattr(backend, "execute", None)
        if callable(ex):
            return list(ex(cypher, params) or [])
    except Exception:  # pragma: no cover - read best-effort by design
        return []
    return []


def _as_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def query_evolution(
    backend: Any, mode: str, target: str = "", limit: int = 20
) -> dict[str, Any]:
    """Answer codebase-evolution questions over the ingested history graph.

    Modes (the queries Gource/SourceTree can't answer — they only render):

    * ``file``     — the timeline of commits that touched ``target`` (a file
      path), newest first, with author + churn (the "evolution of file X").
    * ``owners``   — who owns subsystem ``target`` (path substring): authors
      ranked by how many commits they made to files under it.
    * ``hotspots`` — the highest-churn files (refactor / bus-factor risks).
    * ``coupled``  — files that historically co-change with ``target`` (the
      hidden blast-radius the AST can't see — KG-2.104).
    """
    mode = (mode or "file").strip().lower()
    if mode == "hotspots":
        rows = []
        scan = getattr(backend, "nodes_by_label", None)
        if callable(scan):
            try:
                rows = scan("File") or []
            except Exception:  # noqa: BLE001
                rows = []
        files = [
            {
                "file": (p or {}).get("path") or nid,
                "churn": _as_int((p or {}).get("churn")),
                "commits": _as_int((p or {}).get("commit_count")),
                "authors": _as_int((p or {}).get("author_count")),
            }
            for nid, p in rows
        ]
        files.sort(key=lambda d: d["churn"], reverse=True)
        return {"mode": mode, "hotspots": files[:limit]}

    if mode == "owners":
        if not target:
            return {"mode": mode, "error": "owners needs a path substring in target"}
        rows = _rows(
            backend,
            "MATCH (c)-[r]->(f) WHERE type(r) IN ['TOUCHED','touched'] "
            "AND f.path CONTAINS $p "
            "RETURN c.sha AS sha, c.author_email AS email, c.author_name AS name",
            {"p": target},
        )
        by_author: dict[str, dict[str, Any]] = {}
        for r in rows:
            key = r.get("email") or r.get("name") or "?"
            d = by_author.setdefault(
                key,
                {
                    "author": r.get("name") or key,
                    "email": r.get("email") or "",
                    "commits": set(),
                },
            )
            if r.get("sha"):
                d["commits"].add(r["sha"])
        owners = [
            {"author": d["author"], "email": d["email"], "commits": len(d["commits"])}
            for d in by_author.values()
        ]
        owners.sort(key=lambda d: d["commits"], reverse=True)
        return {"mode": mode, "subsystem": target, "owners": owners[:limit]}

    if mode == "coupled":
        if not target:
            return {"mode": mode, "error": "coupled needs a file path in target"}
        rows = _rows(
            backend,
            "MATCH (a)-[r]-(b) WHERE type(r) IN ['FILE_CHANGES_WITH','file_changes_with'] "
            "AND (a.path = $fp OR a.id = $fid OR a.id CONTAINS $fp) "
            "RETURN DISTINCT b.path AS path, b.id AS id, r.support AS support",
            {"fp": target, "fid": f"file:{target}"},
        )
        coupled = [
            {"file": r.get("path") or r.get("id"), "support": _as_int(r.get("support"))}
            for r in rows
        ]
        coupled.sort(key=lambda d: d["support"], reverse=True)
        return {"mode": mode, "file": target, "coupled": coupled[:limit]}

    # default: file timeline
    if not target:
        return {"mode": "file", "error": "file mode needs a file path in target"}
    rows = _rows(
        backend,
        "MATCH (c)-[r]->(f) WHERE type(r) IN ['TOUCHED','touched'] "
        "AND (f.path = $fp OR f.id = $fid) "
        "RETURN c.sha AS sha, c.iso_date AS date, c.timestamp AS ts, "
        "c.author_name AS author, c.message AS message, "
        "r.insertions AS ins, r.deletions AS dels",
        {"fp": target, "fid": f"file:{target}"},
    )
    timeline = [
        {
            "sha": (r.get("sha") or "")[:12],
            "date": r.get("date"),
            "ts": _as_int(r.get("ts")),
            "author": r.get("author"),
            "message": r.get("message"),
            "insertions": _as_int(r.get("ins")),
            "deletions": _as_int(r.get("dels")),
        }
        for r in rows
    ]
    timeline.sort(key=lambda d: d["ts"], reverse=True)
    return {"mode": "file", "file": target, "timeline": timeline[:limit]}
