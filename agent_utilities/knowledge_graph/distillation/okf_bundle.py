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

import hashlib
import json
from pathlib import Path
from typing import Any

_FM_DELIM = "---"

# OKF SPEC §4.1: ``type`` is the ONE strictly-required frontmatter key. Everything
# else (title/description/resource/timestamp/id) is recommended, not required —
# we align our REQUIRED set with the spec so our bundles are minimally-conformant
# and our *consumer* is permissive (SPEC §9): tolerate unknown types + keys.
REQUIRED_KEYS: tuple[str, ...] = ("type",)
RECOMMENDED_KEYS: tuple[str, ...] = ("title", "description", "resource")


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


def frontmatter_text(
    body: str,
    *,
    ftype: str,
    timestamp: str = "",
    resource: str | None = None,
    concept_id: str | None = None,
    title: str | None = None,
    extra: dict[str, str] | None = None,
) -> str:
    """Return *body* prefixed with OKF YAML frontmatter (in-memory twin of
    :func:`add_frontmatter`).

    Stamps the SPEC §4.1 REQUIRED ``type`` plus recommended title/description/
    resource/id/timestamp. Used by connectors that need to normalize a raw
    markdown body to an OKF concept *without* touching a file on disk
    (CONCEPT:AU-ECO.connector.openwiki-preset). If *body* already carries
    frontmatter it is returned unchanged (non-destructive, permissive).
    """
    if _has_frontmatter(body):
        return body
    fm = [
        _FM_DELIM,
        f"type: {ftype}",
        f"title: {_yaml_escape(title or _title_from(Path('doc.md'), body))}",
    ]
    desc = _description_from(body)
    if desc:
        fm.append(f"description: {_yaml_escape(desc)}")
    if concept_id:
        fm.append(f"id: {concept_id}")
    if resource:
        fm.append(f"resource: {_yaml_escape(resource)}")
    for key, value in (extra or {}).items():
        fm.append(f"{key}: {_yaml_escape(str(value))}")
    if timestamp:
        fm.append(f"timestamp: {timestamp}")
    fm.append(_FM_DELIM)
    fm.append("")
    return "\n".join(fm) + body


def add_frontmatter(
    path: Path,
    *,
    ftype: str,
    timestamp: str,
    resource: str | None = None,
    concept_id: str | None = None,
) -> bool:
    """Add OKF frontmatter to *path* if it has none. Returns True if written."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if _has_frontmatter(text):
        return False
    stamped = frontmatter_text(
        text,
        ftype=ftype,
        timestamp=timestamp,
        resource=resource,
        concept_id=concept_id,
        title=_title_from(path, text),
    )
    path.write_text(stamped, encoding="utf-8")
    return True


def read_frontmatter(source: str | Path) -> tuple[dict[str, Any], str]:
    """Permissive OKF frontmatter parser → ``(frontmatter_dict, body)``.

    CONCEPT:AU-KG.research.okf-overlay-mode — the *permissive consumer* half of the
    OKF standard (SPEC §9): tolerate documents with **unknown types and unknown
    keys**, never reject, and preserve every key when round-tripping. A document
    with no frontmatter yields ``({}, whole_text)``. Values are read as plain
    strings (lists as ``[a, b]`` → list); no schema is imposed. This is
    deliberately dependency-free (no yaml) so it can run in the ingest hot path.
    """
    text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    if not _has_frontmatter(text):
        return {}, text
    stripped = text.lstrip("\n")
    lead = len(text) - len(stripped)
    rest = stripped[len(_FM_DELIM) :]
    end = rest.find(f"\n{_FM_DELIM}")
    if end == -1:
        return {}, text  # unterminated block → treat as bodyless, tolerate
    block = rest[:end]
    body = rest[end + 1 + len(_FM_DELIM) :]
    if body.startswith("\n"):
        body = body[1:]
    fm: dict[str, Any] = {}
    for line in block.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, raw = line.partition(":")
        key = key.strip()
        val: Any = raw.strip()
        if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            val = [p.strip().strip("'\"") for p in inner.split(",")] if inner else []
        elif isinstance(val, str) and len(val) >= 2 and val[0] == val[-1] == '"':
            val = val[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        fm[key] = val
    return fm, (text[:lead] + body if lead else body)


def frontmatter_conforms(fm: dict[str, Any]) -> bool:
    """True iff *fm* carries the SPEC §4.1 REQUIRED keys (only ``type``).

    Permissive by design: unknown types and extra keys are fine — we only assert
    the one field a consumer needs to route a concept.
    """
    return all(fm.get(k) for k in REQUIRED_KEYS)


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
            entries.append(
                f"* [{title}]({child.name}) - {_description_from(_safe_read(child))[:80]}"
            )
    body = [f"# {directory.name or 'index'}", ""] + entries + [""]
    (directory / "index.md").write_text("\n".join(body), encoding="utf-8")


def _safe_read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def write_okf_conformance(
    skill_dir: Path,
    *,
    ftype: str = "Reference",
    timestamp: str = "",
    resource: str | None = None,
    concept_ids: dict[str, str] | None = None,
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
            if add_frontmatter(
                md,
                ftype=ftype,
                timestamp=timestamp,
                resource=resource,
                concept_id=concept_ids.get(rel),
            ):
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


# ═══════════════════════════════════════════════════════════════════════════
# Thin-overlay concept mode (CONCEPT:AU-KG.research.okf-overlay-mode)
# ═══════════════════════════════════════════════════════════════════════════
# openalgo's pattern: index an *existing* repo's markdown into the KG without
# forking its content. An overlay concept is a tiny stub — ``type`` + ``resource``
# (the source URI) + cross-links — whose body defers to the source ("edit the
# source, not this file"). This lets the 80+ fleet repos' markdown be catalogued
# as OKF concepts with ZERO content duplication: the KG holds the map, the repo
# holds the territory.


def render_overlay_concept(
    *,
    title: str,
    ftype: str,
    resource: str,
    timestamp: str = "",
    concept_id: str | None = None,
    links: list[tuple[str, str]] | None = None,
    summary: str = "",
) -> str:
    """Render a thin-overlay OKF concept markdown (index-only, body defers to source).

    CONCEPT:AU-KG.research.okf-overlay-mode. The concept carries the required
    ``type`` + a ``resource`` URI pointing at the real content, an ``overlay: true``
    marker, and outbound cross-links (``[label](target)``) — but NO copied body,
    so the source stays the single point of truth.
    """
    body_lines = [f"# {title}", ""]
    if summary:
        body_lines += [summary, ""]
    body_lines += [
        f"> **Overlay concept** — the content lives at [`{resource}`]({resource}). "
        "Edit the source, not this file; this node only indexes it into the graph.",
        "",
    ]
    if links:
        body_lines.append("## Related")
        body_lines += [f"* [{label}]({target})" for label, target in links]
        body_lines.append("")
    body = "\n".join(body_lines)
    return frontmatter_text(
        body,
        ftype=ftype,
        resource=resource,
        timestamp=timestamp,
        concept_id=concept_id,
        title=title,
        extra={"overlay": "true"},
    )


def write_overlay_concept(path: Path, **kwargs: Any) -> None:
    """Write a thin-overlay concept (see :func:`render_overlay_concept`) to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_overlay_concept(**kwargs), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Bidirectional round-trip sync (CONCEPT:AU-ECO.connector.okf-roundtrip-sync)
# ═══════════════════════════════════════════════════════════════════════════
# We were one-way (source→KG). This is the KG→OKF-bundle-on-disk push, modelled
# on the mdcode Catalog-Snapshot protocol: a checksum ``.catalog.state`` file
# separates tool state from user content; a push FAILS FAST if a target file was
# modified in the interim (conflict → require a pull to resolve); files present
# in state but absent from the new snapshot are treated as INTENT-TO-DELETE.

STATE_FILENAME = ".catalog.state"
_STATE_SCHEMA = "okf-catalog-state/1"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_catalog_state(catalog_dir: Path) -> dict[str, str]:
    """Map every content file's bundle-relative POSIX path → its sha256.

    Excludes the state file itself and dotfiles (tool state, not user content).
    """
    state: dict[str, str] = {}
    if not catalog_dir.is_dir():
        return state
    for p in sorted(catalog_dir.rglob("*")):
        if not p.is_file() or p.name == STATE_FILENAME:
            continue
        rel = p.relative_to(catalog_dir).as_posix()
        if any(part.startswith(".") for part in Path(rel).parts):
            continue
        state[rel] = _sha256_file(p)
    return state


def read_catalog_state(catalog_dir: Path) -> dict[str, str]:
    """Read the persisted ``.catalog.state`` checksums (``{}`` if none)."""
    sf = catalog_dir / STATE_FILENAME
    if not sf.exists():
        return {}
    try:
        data = json.loads(sf.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entries = data.get("entries") if isinstance(data, dict) else None
    return {str(k): str(v) for k, v in (entries or {}).items()}


def write_catalog_state(catalog_dir: Path, state: dict[str, str]) -> None:
    """Persist checksums to ``.catalog.state`` (separate from user content)."""
    payload = {"schema": _STATE_SCHEMA, "entries": dict(sorted(state.items()))}
    (catalog_dir / STATE_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


class OkfConflictError(RuntimeError):
    """A push target was modified in the interim — abort and pull to resolve.

    CONCEPT:AU-ECO.connector.okf-roundtrip-sync — the mdcode fail-fast rule
    (§3.3): never silently clobber an out-of-band edit.
    """

    def __init__(self, conflicts: list[str]) -> None:
        self.conflicts = conflicts
        super().__init__(
            f"{len(conflicts)} OKF catalog file(s) changed since last sync — "
            f"pull to resolve before pushing: {', '.join(conflicts[:5])}"
        )


class OkfRoundTripSync:
    """Serialize a KG-distilled skill-graph → OKF bundle on disk, safely.

    CONCEPT:AU-ECO.connector.okf-roundtrip-sync. A *skill-graph directory* (the
    canonical serialization of KG-distilled knowledge — ``reference/`` tree +
    ``index.md``/``log.md``/``sources.json``) is the source of truth; ``push``
    mirrors it into a *catalog directory* with a ``.catalog.state`` checksum file,
    fail-fast conflict detection, and intent-to-delete — exactly mdcode's
    Catalog-Snapshot contract, but over OUR skill-graph standard (no parallel
    format).
    """

    #: Skill-graph companion files carried into the OKF bundle alongside reference/.
    _CARRY = (
        "SKILL.md",
        "index.md",
        "log.md",
        "sources.json",
        "index.json",
        "OVERVIEW.md",
        "kg_manifest.json",
    )

    def __init__(self, skill_dir: str | Path, catalog_dir: str | Path) -> None:
        self.skill_dir = Path(skill_dir)
        self.catalog_dir = Path(catalog_dir)

    def _snapshot_sources(self) -> dict[str, str]:
        """``{bundle-relpath: content}`` the skill-graph would serialize into OKF."""
        out: dict[str, str] = {}
        ref = self.skill_dir / "reference"
        if ref.is_dir():
            for p in sorted(ref.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(self.skill_dir).as_posix()
                    out[rel] = p.read_text(encoding="utf-8", errors="replace")
        for name in self._CARRY:
            f = self.skill_dir / name
            if f.is_file():
                out[name] = f.read_text(encoding="utf-8", errors="replace")
        return out

    def plan(self, *, allow_delete: bool = False) -> dict[str, Any]:
        """Diff the skill-graph snapshot against the catalog + its ``.catalog.state``.

        Returns ``{creates, updates, conflicts, deletes, unchanged}`` (lists of
        bundle-relpaths). *conflicts* = catalog files whose on-disk checksum no
        longer matches the recorded state (modified in the interim). *deletes* =
        files in state but absent from the new snapshot (intent-to-delete), only
        surfaced when *allow_delete*.
        """
        sources = self._snapshot_sources()
        recorded = read_catalog_state(self.catalog_dir)
        on_disk = compute_catalog_state(self.catalog_dir)
        creates: list[str] = []
        updates: list[str] = []
        conflicts: list[str] = []
        unchanged: list[str] = []
        for rel, content in sources.items():
            new_sum = _sha256_text(content)
            if rel not in on_disk:
                creates.append(rel)
                continue
            # Fail-fast: the catalog copy diverged from what we last wrote.
            if rel in recorded and on_disk[rel] != recorded[rel]:
                conflicts.append(rel)
                continue
            (unchanged if on_disk[rel] == new_sum else updates).append(rel)
        deletes = (
            [
                rel
                for rel in recorded
                if rel not in sources and (self.catalog_dir / rel).exists()
            ]
            if allow_delete
            else []
        )
        return {
            "creates": sorted(creates),
            "updates": sorted(updates),
            "conflicts": sorted(conflicts),
            "deletes": sorted(deletes),
            "unchanged": sorted(unchanged),
        }

    def push(
        self,
        *,
        dry_run: bool = True,
        allow_delete: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Push the skill-graph → OKF catalog. Dry-run previews; live writes files.

        Aborts with :class:`OkfConflictError` when *conflicts* are detected unless
        *force*. On a live push it materializes creates/updates, applies deletes
        (when *allow_delete*), and rewrites ``.catalog.state``.
        """
        plan = self.plan(allow_delete=allow_delete)
        result = {
            "skill_dir": str(self.skill_dir),
            "catalog_dir": str(self.catalog_dir),
            "dry_run": dry_run,
            **plan,
        }
        if plan["conflicts"] and not force:
            if dry_run:
                result["status"] = "conflict"
                return result
            raise OkfConflictError(plan["conflicts"])
        if dry_run:
            result["status"] = "previewed"
            return result

        sources = self._snapshot_sources()
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        for rel in (
            plan["creates"] + plan["updates"] + (plan["conflicts"] if force else [])
        ):
            dest = self.catalog_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(sources[rel], encoding="utf-8")
        for rel in plan["deletes"]:
            (self.catalog_dir / rel).unlink(missing_ok=True)
        write_catalog_state(self.catalog_dir, compute_catalog_state(self.catalog_dir))
        result["status"] = "pushed"
        return result


# ═══════════════════════════════════════════════════════════════════════════
# External free-``type`` → governed domain mapping (CONCEPT:AU-KG.ingest.okf-type-mapping)
# ═══════════════════════════════════════════════════════════════════════════
# OKF ``type`` is an OPEN vocabulary (SPEC §4.1: "not registered centrally"). To
# normalize external bundles (openalgo, openwiki, third-party OKF) into OUR
# governed OKF-CIS ``domain`` axis, map each free type onto a closed domain. An
# unmapped type is NOT dropped — it is parked on a review queue (a producer's
# type is signal, not noise) and the concept still ingests under a permissive
# default domain, matching the permissive-consumer rule (SPEC §9).

#: Curated seed: common external OKF ``type`` values → (pillar, domain). Extend as
#: the fleet's corpora reveal new types (review-queue drives curation).
TYPE_DOMAIN_MAP: dict[str, tuple[str, str]] = {
    "reference": ("KG", "research"),
    "playbook": ("KG", "research"),
    "wiki": ("KG", "research"),
    "article": ("KG", "research"),
    "guide": ("KG", "research"),
    "concept": ("KG", "ontology"),
    "metric": ("KG", "compute"),
    "api endpoint": ("KG", "query"),
    "api": ("KG", "query"),
    "bigquery table": ("KG", "backend"),
    "bigquery dataset": ("KG", "backend"),
    "table": ("KG", "backend"),
    "dataset": ("KG", "backend"),
    "document": ("KG", "ingest"),
    "process": ("KG", "domains"),
}

#: Where a concept lands when its ``type`` maps to no governed domain (permissive).
DEFAULT_TYPE_DOMAIN: tuple[str, str] = ("KG", "research")


def _review_queue_path(queue_path: str | Path | None) -> Path:
    if queue_path is not None:
        return Path(queue_path)
    import platformdirs

    base = Path(platformdirs.user_state_dir("agent-utilities"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "okf_type_review_queue.json"


def map_external_type(
    ext_type: str,
    *,
    pillar: str = "KG",
) -> tuple[str, str] | None:
    """Map an external OKF ``type`` → ``(pillar, domain)`` in the governed vocab.

    CONCEPT:AU-KG.ingest.okf-type-mapping. Resolution order: exact seed match →
    signal match against the closed ``domain_vocab`` for *pillar* → ``None`` (the
    caller queues it for review and falls back to :data:`DEFAULT_TYPE_DOMAIN`).
    """
    from agent_utilities.governance.concept_hierarchy import load_domain_vocab

    key = (ext_type or "").strip().lower()
    if not key:
        return None
    if key in TYPE_DOMAIN_MAP:
        return TYPE_DOMAIN_MAP[key]
    domains = load_domain_vocab().get(pillar, {})
    for domain, signals in domains.items():
        if key == domain or key in signals:
            return (pillar, domain)
    # substring signal match (e.g. "reference guide" → research via "reference")
    for domain, signals in domains.items():
        if any(sig in key or key in sig for sig in [domain, *signals]):
            return (pillar, domain)
    return None


def resolve_type_domain(
    ext_type: str,
    *,
    pillar: str = "KG",
    queue_path: str | Path | None = None,
    provenance: str = "",
) -> tuple[str, str]:
    """Map *ext_type* to a governed domain, queueing unmapped types for review.

    Never fails: an unmapped type is appended to the review queue and the concept
    lands on :data:`DEFAULT_TYPE_DOMAIN` so ingestion is never blocked.
    """
    mapped = map_external_type(ext_type, pillar=pillar)
    if mapped is not None:
        return mapped
    queue_unmapped_type(ext_type, provenance=provenance, queue_path=queue_path)
    return DEFAULT_TYPE_DOMAIN


def queue_unmapped_type(
    ext_type: str,
    *,
    provenance: str = "",
    queue_path: str | Path | None = None,
) -> None:
    """Append an unmapped external ``type`` to the review queue (dedup by type)."""
    path = _review_queue_path(queue_path)
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except (OSError, ValueError):
        existing = []
    if not isinstance(existing, list):
        existing = []
    key = (ext_type or "").strip()
    for row in existing:
        if isinstance(row, dict) and row.get("type") == key:
            if provenance and provenance not in row.get("provenance", []):
                row.setdefault("provenance", []).append(provenance)
            path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            return
    existing.append({"type": key, "provenance": [provenance] if provenance else []})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def list_type_review_queue(
    queue_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return the queued unmapped types awaiting a domain-vocab curation decision."""
    path = _review_queue_path(queue_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []
