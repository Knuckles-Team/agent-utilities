#!/usr/bin/env python3
"""Auto-curate a repo's REMAINING legacy concept ids into OKF-CIS (OS-5.77).

For the EG drift delta + the ~60 tail repos: deterministically map every legacy
CONCEPT:<PREFIX>-<n> still present in a repo to a valid <SLUG>-<PILLAR>.<domain>.
<concept> id (slug from repo provenance, pillar from the reassignment table,
domain scored against the closed vocab with a safe per-pillar fallback, concept
kebab'd from the marker's doc). Emits a supplement plan; the applier consumes it.

Usage: autocurate_repo.py <repo-name> <repo-path> <out-supplement.yaml>
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

CANON = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, CANON)
from agent_utilities.governance import concept_hierarchy as ch  # noqa: E402

PILLAR_REASSIGN = {
    "KG": "KG", "ORCH": "ORCH", "AHE": "AHE", "ECO": "ECO", "OS": "OS", "GBOT": "GBOT",
    "EG": "KG", "EE": "AHE", "ML": "AHE", "CE": "KG", "SAFE": "OS", "CTX": "KG",
    "LGC": "OS", "UTIL": "OS",
}
# safe default domain per pillar when scoring finds nothing
DEFAULT_DOMAIN = {"KG": "compute", "ORCH": "execution", "AHE": "harness",
                  "ECO": "mcp", "OS": "config", "GBOT": "cockpit"}
_LEGACY_RE = re.compile(r"CONCEPT:([A-Z]+-\d+(?:\.[0-9A-Za-z]+)*)")
_DEF_SEP = re.compile(r"^\)?\s*[—:–-]\s+\S")
_EXT = {".py", ".rs", ".md"}
_SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules", "target", "build", "dist"}
_SKIP_NAMES = {"CHANGELOG.md", "concepts.yaml", "concept_reservations.yaml", "concept_map.md"}
_STOP = {"the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "via",
         "engine", "layer", "system", "support", "based", "driven", "new"}


def _kebab(s: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9]+", s.lower()) if not w.isdigit()]
    words = [w for w in words if w not in _STOP] or words
    return "-".join(words[:4]) or "concept"


def _score_domain(pillar: str, files: list[str], doc: str) -> str:
    vocab = ch.load_domain_vocab().get(pillar, {})
    hay = (" ".join(files) + " " + doc).lower()
    best, best_n = None, 0
    for dom, sigs in vocab.items():
        n = sum(1 for s in [dom, *sigs] if s in hay)
        if n > best_n:
            best, best_n = dom, n
    return best or DEFAULT_DOMAIN.get(pillar, "config")


def main() -> int:
    name, root_s, out = sys.argv[1], sys.argv[2], sys.argv[3]
    root = Path(root_s)
    slug = ch.slug_for_repo(name)
    if not slug:
        print(f"no slug for {name}")
        return 2
    docs: dict[str, str] = {}
    files: dict[str, set[str]] = defaultdict(set)
    for p in root.rglob("*"):
        if p.suffix not in _EXT or p.name in _SKIP_NAMES:
            continue
        if any(s in p.parts for s in _SKIP_DIRS):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "CONCEPT:" not in text:
            continue
        rel = str(p.relative_to(root))
        for line in text.splitlines():
            for m in _LEGACY_RE.finditer(line):
                cid = m.group(1)
                if ch.is_okf_id(cid):
                    continue
                files[cid].add(rel)
                tail = line[m.end():]
                if _DEF_SEP.match(tail):
                    d = re.sub(r"^[\s—:\-–)]+", "", tail).strip()[:80]
                    if len(d) > len(docs.get(cid, "")):
                        docs[cid] = d
    entries, used = [], set()
    for cid in sorted(files):
        prefix = cid.split("-", 1)[0]
        pillar = PILLAR_REASSIGN.get(prefix, "OS")
        flist = sorted(files[cid])
        doc = docs.get(cid, "")
        domain = _score_domain(pillar, flist, doc)
        concept = _kebab(doc or cid)
        base, n = concept, 2
        while (slug, pillar, domain, concept) in used:
            concept = f"{base}-{n}"
            n += 1
        used.add((slug, pillar, domain, concept))
        new_id = f"{slug}-{pillar}.{domain}.{concept}"
        assert ch.is_okf_id(new_id) and ch.is_valid_domain(pillar, domain), new_id
        entries.append({"old_id": cid, "new_id": new_id, "slug": slug,
                        "pillar": pillar, "domain": domain, "files": []})
    yaml.safe_dump({"entries": entries}, open(out, "w"), sort_keys=False, width=100)
    print(f"{name}: auto-curated {len(entries)} remaining legacy ids -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
