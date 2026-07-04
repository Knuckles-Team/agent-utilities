#!/usr/bin/env python3
"""Read-only OKF-CIS migration planner (CONCEPT:OS-5.77).

Scans one or more repos for legacy ``CONCEPT:<id>`` markers *per occurrence*,
derives the new ``<SLUG>-<PILLAR>.<domain>.<concept>`` id for each, disambiguates
collisions (the same flat id reused for several meanings), and emits a reviewable
``migration_plan.yaml`` + draft ``legacy_map.yaml``. **It writes nothing else and
rewrites no source** — the applier (``apply_concept_migration.py``) consumes the
frozen plan. Curate ``needs_curation`` rows before any cutover.

Pipeline (per the plan): SLUG from provenance (file's repo, via slug_registry) →
PILLAR from the 8-row reassignment table → domain scored against the closed
``domain_vocab.yaml`` → concept-slug kebabed from the concept name. Low-confidence
domains and un-splittable collisions are flagged, never guessed silently.

Usage:
    python scripts/plan_concept_migration.py [--repo NAME=PATH ...] \
        [--out DIR] [--legacy-map PATH]
Defaults to scanning this agent-utilities checkout as repo ``agent-utilities``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Import the ONE grammar/vocab source (never re-declare the regexes here).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_utilities.governance import concept_hierarchy as ch  # noqa: E402
from agent_utilities.governance.concept_allocator import MARKER_RE  # noqa: E402

# --------------------------------------------------------------------------- #
# The 8-row anti-leak reassignment table: legacy id-prefix -> global PILLAR.
# Native families map to themselves; leaked cross-repo families are re-homed.
# This is the ENTIRE taxonomy decision surface — reviewed once, by a human.
# --------------------------------------------------------------------------- #
PILLAR_REASSIGN: dict[str, str] = {
    "KG": "KG",
    "ORCH": "ORCH",
    "AHE": "AHE",
    "ECO": "ECO",
    "OS": "OS",
    "GBOT": "GBOT",
    # leaked / satellite families -> their true pillar:
    "EG": "KG",  # epistemic-graph engine IS the KG store
    "EE": "AHE",  # evaluation engine -> harness
    "ML": "AHE",  # ml/trainer -> harness
    "CE": "KG",  # cognitive engine -> knowledge
    "SAFE": "OS",  # safety -> platform/governance
    "CTX": "KG",  # context plane -> memory/retrieval
    "LGC": "OS",  # legacy/compat -> platform
    "UTIL": "OS",  # utilities -> platform
}

_SCAN_EXT = {".py", ".rs", ".md"}
_SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules", "target", "build", "dist"}
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "via",
    "engine", "layer", "system", "support", "based", "driven", "aware", "new",
}


def _iter_repo_files(root: Path):
    for p in root.rglob("*"):
        if p.suffix not in _SCAN_EXT:
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        yield p


def _clean_doc(text: str) -> str:
    """Trailing text after a marker -> a short human doc (mirrors build_concepts)."""
    s = text.strip()
    s = re.sub(r"^[\s—\-:\]\},.;]+", "", s)
    s = re.split(r"[()\[\]{}\"'`:%#]", s, maxsplit=1)[0]
    return s.strip()


def _legacy_prefix(old_id: str) -> str:
    """``KG-2.7`` -> ``KG``; ``EE-033`` -> ``EE``."""
    return old_id.split("-", 1)[0]


def _kebab(name: str, *, keep: int = 4) -> str:
    """A short kebab concept-slug from a human name (stopword-stripped).

    Drops bare-number tokens (we are escaping numeric ids) and version suffixes;
    an empty/degenerate result becomes ``concept`` (flagged for curation upstream).
    """
    words = re.findall(r"[A-Za-z0-9]+", name.lower())
    words = [w for w in words if not w.isdigit()]  # never carry numbers forward
    meaningful = [w for w in words if w not in _STOPWORDS] or words
    slug = "-".join(meaningful[:keep])
    return slug or "concept"


#: A definition marker: ``CONCEPT:<id>`` followed (maybe past a ``)``) by an
#: em-dash / colon / hyphen separator + description. References (parenthetical
#: mid-sentence, or no separator) do NOT define and are excluded from clustering.
_DEF_SEP_RE = re.compile(r"^\)?\s*[—:–-]\s+\S")


def _is_degenerate_slug(concept: str, pillar: str) -> bool:
    """A concept-slug too weak to keep (needs a human rename)."""
    if concept in {"concept", pillar.lower()}:
        return True
    toks = concept.split("-")
    # a single short token, or ends on a filler word, or is just the pillar family
    if len(toks) == 1 and len(concept) < 5:
        return True
    if toks[-1] in {"the", "that", "this", "when", "become", "becomes", "is", "a"}:
        return True
    return False


@dataclass
class Occurrence:
    old_id: str
    repo: str
    slug: str
    rel_file: str
    line: int
    doc: str
    is_def: bool = False

    @property
    def top_dir(self) -> str:
        parts = Path(self.rel_file).parts
        # second component is usually the meaningful package subdir
        return parts[1] if len(parts) > 1 else (parts[0] if parts else "")


@dataclass
class PlanEntry:
    old_id: str
    slug: str
    pillar: str
    domain: str | None
    concept: str
    new_id: str | None
    confidence: str  # high | medium | low
    needs_curation: bool
    files: list[str] = field(default_factory=list)
    doc: str = ""
    notes: list[str] = field(default_factory=list)


def scan_occurrences(repo: str, root: Path, slug: str) -> list[Occurrence]:
    occs: list[Occurrence] = []
    for path in _iter_repo_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "CONCEPT:" not in text:
            continue
        rel = str(path.relative_to(root))
        # A definition can only come from real (non-test) source — tests and .md
        # (concept_map/docs) reference concepts, they don't define their meaning.
        is_source = path.suffix in {".py", ".rs"} and "test" not in rel.lower()
        for ln, line in enumerate(text.splitlines(), 1):
            for m in MARKER_RE.finditer(line):
                old_id = m.group("id")
                if ch.is_okf_id(old_id):
                    continue  # already migrated — planner is idempotent
                tail = line[m.end():]
                is_def = is_source and bool(_DEF_SEP_RE.match(tail))
                occs.append(Occurrence(old_id, repo, slug, rel, ln, _clean_doc(tail), is_def))
    return occs


def _score_domain(pillar: str, files: list[str], doc: str) -> tuple[str | None, str]:
    """Return (domain, confidence) scored against the closed vocab for *pillar*."""
    vocab = ch.load_domain_vocab().get(pillar, {})
    if not vocab:
        return None, "low"
    hay_dirs = " ".join(Path(f).as_posix().lower() for f in files)
    hay_doc = doc.lower()
    scores: dict[str, int] = defaultdict(int)
    for domain, signals in vocab.items():
        for sig in [domain, *signals]:
            if sig in hay_dirs:
                scores[domain] += 2  # module-dir signal is high-value
            if re.search(rf"\b{re.escape(sig)}\b", hay_doc):
                scores[domain] += 1
    if not scores:
        return None, "low"
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    if best_score >= 2 and best_score > second_score:
        return best, "high"
    if best_score >= 1:
        return best, "medium"
    return None, "low"


def _norm_doc(doc: str) -> str:
    return re.sub(r"\s+", " ", doc.strip().lower())


def _sig_tokens(doc: str) -> set[str]:
    """Significant lowercase word-stems (len>=4, minus stopwords) for clustering."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", doc.lower())
    return {w for w in words if w not in _STOPWORDS}


def _cluster_meanings(group: list[Occurrence]) -> list[list[Occurrence]]:
    """Split a legacy id's occurrences into DISTINCT-MEANING clusters.

    A concept legitimately has ONE substantive description; near-duplicate
    comment lines are the same meaning. Only genuinely different substantive
    docs (keyed by a normalized prefix) count as separate meanings — this is
    what catches ``KG-2.7`` (7 real meanings) without treating every reference
    line as a collision. Reference-only occurrences (empty/terse doc) attach to
    the single meaning, or are held out when the id is a true collision.
    """
    defs = [o for o in group if o.is_def and len(_norm_doc(o.doc)) >= 12]
    rest = [o for o in group if o not in defs]
    if not defs:
        return [group]  # no definition site -> single concept (refs only)

    # Greedy token-set clustering: two definition docs are the SAME meaning when
    # their significant-word sets overlap (>=2 shared tokens or Jaccard>=0.34).
    # This merges wording variants ("Telemetry-Driven Optimization" vs "telemetry
    # driven optimization loop") while keeping KG-2.7's 7 distinct docs apart.
    clusters: list[tuple[set[str], list[Occurrence]]] = []
    for o in sorted(defs, key=lambda o: -len(o.doc)):
        toks = _sig_tokens(o.doc)
        placed = False
        for rep, members in clusters:
            shared = toks & rep
            union = toks | rep
            if shared and (len(shared) >= 2 or len(shared) / max(1, len(union)) >= 0.34):
                members.append(o)
                rep |= toks
                placed = True
                break
        if not placed:
            clusters.append((set(toks), [o]))

    out = [members for _, members in clusters]
    if len(out) == 1:
        out[0].extend(rest)  # not a collision: everything is one concept
    return out


def load_curated_names(repos: list[tuple[str, Path]]) -> dict[str, str]:
    """``old_id -> curated name`` from each repo's ``docs/concepts.yaml`` (if any).

    These names are already deduped/curated by build_concepts_yaml, so they make
    far better concept-slugs than raw trailing comment text — used for the clean
    (non-collision) path; collision splits fall back to their per-cluster doc.
    """
    names: dict[str, str] = {}
    for _name, root in repos:
        cy = root / "docs" / "concepts.yaml"
        if not cy.exists():
            continue
        data = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
        for c in data.get("concepts", []):
            cid, nm = c.get("id"), c.get("name")
            if cid and nm and not ch.is_okf_id(str(cid)):
                names[str(cid)] = str(nm)
    return names


def build_plan(
    occs: list[Occurrence], names: dict[str, str] | None = None
) -> list[PlanEntry]:
    names = names or {}
    by_id: dict[str, list[Occurrence]] = defaultdict(list)
    for o in occs:
        by_id[o.old_id].append(o)

    entries: list[PlanEntry] = []
    used_slugs: set[str] = set()  # (slug,pillar,domain,concept) uniqueness

    for old_id, group in sorted(by_id.items()):
        prefix = _legacy_prefix(old_id)
        pillar = PILLAR_REASSIGN.get(prefix)
        clusters = _cluster_meanings(group)
        is_collision = len(clusters) > 1

        for cocc in clusters:
            slug = cocc[0].slug  # provenance: all in one cluster share a repo
            files = sorted({o.rel_file for o in cocc})
            doc = max((o.doc for o in cocc), key=len, default="")
            notes: list[str] = []
            if is_collision:
                notes.append(f"collision-split of {old_id} ({len(clusters)} meanings)")
            if pillar is None:
                entries.append(PlanEntry(
                    old_id, slug, "?", None, _kebab(doc or old_id), None,
                    "low", True, files, doc,
                    notes + [f"unknown legacy prefix {prefix!r} — assign a pillar"],
                ))
                continue
            # Clean (non-collision) ids take the curated concepts.yaml name;
            # collision splits must use their own per-cluster doc to stay distinct.
            slug_source = doc
            if not is_collision and old_id in names:
                slug_source = names[old_id]
            domain, conf = _score_domain(pillar, files, f"{slug_source} {doc}")
            concept = _kebab(slug_source)
            # ensure (slug,pillar,domain,concept) uniqueness
            base_concept = concept
            n = 2
            while domain and (slug, pillar, domain, concept) in used_slugs:
                concept = f"{base_concept}-{n}"
                n += 1
            needs = domain is None or conf == "low" or is_collision
            if _is_degenerate_slug(concept, pillar):
                needs = True
                notes.append(f"weak concept-slug {concept!r} — rename during curation")
            new_id = f"{slug}-{pillar}.{domain}.{concept}" if domain else None
            if domain:
                used_slugs.add((slug, pillar, domain, concept))
                # final validity gate against the grammar + closed vocab
                if not (ch.is_okf_id(new_id) and ch.is_valid_domain(pillar, domain)):
                    notes.append("INVALID generated id — needs curation")
                    needs = True
            entries.append(PlanEntry(
                old_id, slug, pillar, domain, concept, new_id,
                conf, needs, files, doc, notes,
            ))
    return entries


def _entry_dict(e: PlanEntry) -> dict:
    d = {
        "old_id": e.old_id,
        "new_id": e.new_id,
        "slug": e.slug,
        "pillar": e.pillar,
        "domain": e.domain,
        "concept": e.concept,
        "confidence": e.confidence,
        "needs_curation": e.needs_curation,
        "doc": e.doc,
        "files": e.files,
    }
    if e.notes:
        d["notes"] = e.notes
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo", action="append", default=[],
        metavar="NAME=PATH", help="repo to scan (repeatable); default: this checkout",
    )
    ap.add_argument("--out", default="docs", help="output dir for the plan (default docs/)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    repos: list[tuple[str, Path]] = []
    if args.repo:
        for spec in args.repo:
            name, _, path = spec.partition("=")
            repos.append((name, Path(path)))
    else:
        repos.append(("agent-utilities", repo_root))

    all_occs: list[Occurrence] = []
    unslugged: set[str] = set()
    for name, root in repos:
        slug = ch.slug_for_repo(name)
        if slug is None:
            unslugged.add(name)
            continue
        all_occs.extend(scan_occurrences(name, root, slug))

    names = load_curated_names([(n, r) for n, r in repos if ch.slug_for_repo(n)])
    entries = build_plan(all_occs, names)

    # legacy_map (historical resolver): simple 1:1 + collisions keyed (old,file)
    simple: dict[str, str] = {}
    collisions: dict[str, list[dict]] = defaultdict(list)
    by_old: dict[str, list[PlanEntry]] = defaultdict(list)
    for e in entries:
        by_old[e.old_id].append(e)
    for old_id, es in by_old.items():
        if len(es) == 1 and es[0].new_id:
            simple[old_id] = es[0].new_id
        else:
            for e in es:
                for f in e.files:
                    collisions[old_id].append({"file": f, "new": e.new_id})

    unmapped = sorted({e.old_id for e in entries if not e.new_id})
    needs = [e for e in entries if e.needs_curation]
    new_domains = sorted({f"{e.pillar}.{e.domain}" for e in entries if e.domain})

    summary = {
        "total_legacy_ids": len({e.old_id for e in entries}),
        "total_new_concepts": len([e for e in entries if e.new_id]),
        "collisions_detected": len([o for o, es in by_old.items() if len(es) > 1]),
        "needs_curation": len(needs),
        "unmapped": len(unmapped),
        "distinct_domains_used": len(new_domains),
        "repos_scanned": [n for n, _ in repos],
        "unslugged_repos": sorted(unslugged),
    }

    out_dir = repo_root / args.out
    out_dir.mkdir(exist_ok=True)
    plan = {
        "generated_by": "scripts/plan_concept_migration.py",
        "concept": "OS-5.77",
        "summary": summary,
        "domains_used": new_domains,
        "entries": [_entry_dict(e) for e in sorted(entries, key=lambda e: e.old_id)],
    }
    (out_dir / "migration_plan.yaml").write_text(
        yaml.safe_dump(plan, sort_keys=False, width=100), encoding="utf-8"
    )
    legacy = {
        "version": 1,
        "generated_by": "scripts/plan_concept_migration.py",
        "simple": dict(sorted(simple.items())),
        "collisions": {k: v for k, v in sorted(collisions.items())},
        "unmapped": unmapped,
    }
    (out_dir / "legacy_map.yaml").write_text(
        yaml.safe_dump(legacy, sort_keys=False, width=100), encoding="utf-8"
    )

    print("OKF-CIS migration plan written to", out_dir)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
