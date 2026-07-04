"""OKF conformance for skill-graphs (CONCEPT:AU-KG.research.okf-bundle-conformance).

Google's Open Knowledge Format (OKF) models a knowledge bundle as a tree of markdown
files with YAML frontmatter, an ``index.md`` per directory (progressive disclosure) and
a root ``log.md`` (history). Our skill-graph pipeline already emits a ``reference/``
markdown tree + machine ``index.json`` + ``sources.json``; this makes that output
OKF-conformant so a skill-graph is a valid OKF bundle:

* every ``reference/*.md`` gets YAML frontmatter with the REQUIRED ``type`` (+ title/
  description/timestamp/resource) if it lacks it — non-destructive, preserves the body;
* an ``index.md`` is written per reference directory (the OKF twin of ``index.json``);
* a root ``log.md`` records the build (the OKF twin of ``sources.json`` freshness).

For the *agent-utilities* concept skill-graph specifically, callers pass ``concept_id``
so each concept file is stamped with its OKF-CIS ``id:`` — making that skill-graph the
canonical OKF concept bundle. See ``docs/okf-cis.md``.
"""

from __future__ import annotations

from pathlib import Path

_FM_DELIM = "---"


def _has_frontmatter(text: str) -> bool:
    return text.lstrip().startswith(_FM_DELIM)


def _title_from(path: Path, body: str) -> str:
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
    return path.stem.replace("-", " ").replace("_", " ").title()


def _description_from(body: str) -> str:
    for line in body.splitlines():
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith(_FM_DELIM):
            return s[:200]
    return ""


def _yaml_escape(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def add_frontmatter(
    path: Path, *, ftype: str, timestamp: str, resource: str | None = None,
    concept_id: str | None = None,
) -> bool:
    """Add OKF frontmatter to *path* if it has none. Returns True if written."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if _has_frontmatter(text):
        return False
    fm = [_FM_DELIM, f"type: {ftype}",
          f"title: {_yaml_escape(_title_from(path, text))}"]
    desc = _description_from(text)
    if desc:
        fm.append(f"description: {_yaml_escape(desc)}")
    if concept_id:
        fm.append(f"id: {concept_id}")
    if resource:
        fm.append(f"resource: {_yaml_escape(resource)}")
    fm.append(f"timestamp: {timestamp}")
    fm.append(_FM_DELIM)
    fm.append("")
    path.write_text("\n".join(fm) + text, encoding="utf-8")
    return True


def write_dir_index(directory: Path) -> None:
    """Write an OKF ``index.md`` listing this directory's concepts + subdirs (§6)."""
    entries: list[str] = []
    for child in sorted(directory.iterdir()):
        if child.name in {"index.md", "log.md"} or child.name.startswith("."):
            continue
        if child.is_dir():
            entries.append(f"* [{child.name}/]({child.name}/) - subdirectory")
        elif child.suffix == ".md":
            title = _title_from(child, _safe_read(child))
            entries.append(f"* [{title}]({child.name}) - {_description_from(_safe_read(child))[:80]}")
    body = [f"# {directory.name or 'index'}", ""] + entries + [""]
    (directory / "index.md").write_text("\n".join(body), encoding="utf-8")


def _safe_read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def write_okf_conformance(
    skill_dir: Path, *, ftype: str = "Reference", timestamp: str = "",
    resource: str | None = None, concept_ids: dict[str, str] | None = None,
) -> dict[str, int]:
    """Make a built skill-graph an OKF bundle. Returns counts.

    *timestamp* — ISO-8601 build time (pass-through; the module never calls a clock).
    *concept_ids* — optional ``{reference-relpath: okf-id}`` to stamp concept files
    (used for the agent-utilities concept bundle).
    """
    ref = skill_dir / "reference"
    stamped = indexed = 0
    concept_ids = concept_ids or {}
    if ref.is_dir():
        for md in ref.rglob("*.md"):
            if md.name in {"index.md", "log.md"}:
                continue
            rel = md.relative_to(ref).as_posix()
            if add_frontmatter(md, ftype=ftype, timestamp=timestamp,
                               resource=resource, concept_id=concept_ids.get(rel)):
                stamped += 1
        for d in [ref, *[p for p in ref.rglob("*") if p.is_dir()]]:
            write_dir_index(d)
            indexed += 1
    # root log.md (OKF §7) — the human twin of sources.json
    log = skill_dir / "log.md"
    if not log.exists():
        log.write_text(
            f"# Update Log\n\n## {timestamp[:10] or 'build'}\n"
            f"* **Build**: skill-graph generated (OKF-conformant). "
            f"Machine provenance in `sources.json`.\n",
            encoding="utf-8",
        )
    return {"frontmatter_added": stamped, "index_md_written": indexed}
