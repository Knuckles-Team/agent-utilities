#!/usr/bin/env python3
"""Triage a whole ``PILLAR.domain`` group of undocumented concepts, with evidence.

CONCEPT:AU-OS.governance.concept-domain-triage — mechanised domain triage.

The problem this solves
-----------------------
``check_concept_governance.py --audit-merged`` reports **1,039 live concepts
with no design document**, clustered into 80 ``PILLAR.domain`` groups. Read as
"write 1,039 documents", that is a year of work and most of it would be filler.
Read correctly it is far smaller: the ``AU-KG.compute`` pilot (79 live concepts)
resolved to **21 documents, 47 pointers and 11 retirements** — the unit of
documentation is the *decision*, not the marker.

But measuring that took reading 79 markers, their modules and their introducing
commits. Doing it 79 more times by hand — or with 79 agents — is the thing to
avoid. This tool does the mechanical majority **with no model tokens at all**:
deterministic signals (module locality, git archaeology, id shape) decide the
pointers and the retirements; only a small, explicitly-marked ``review`` residue
plus the proposed cluster heads need judgement. It never applies a retirement or
a parent link on its own suggestion — a wrong parent silently buries a real
decision, so a human/agent confirms and the tool does the typing.

Token cost, measured on ``AU-KG.compute``
-----------------------------------------
``propose`` is ~12 s of CPU and **zero** model tokens. ``packet`` then emits the
undecided residue only — ~6.5k tokens for the whole 79-concept domain, versus
roughly 60-80k to read every marker and module by hand. Confirming a cluster is
ONE judgement covering N pointers, not N judgements.

Four dispositions
-----------------
``document``
    A genuine trade-off with a rejected alternative. Earns its own design doc.
``parent``
    A marker on code realising an already-documented decision. Points at that
    decision's doc via ``agent_utilities/governance/concept_lineage.yaml``.
``retire``
    A marker that never named a decision — most often an id auto-derived from a
    prose fragment (``…compute.when-exposes``, minted from the sentence "when
    the engine exposes the native...") or a bare legacy pillar citation
    (``…compute.kg-2``). The marker is deleted and the retirement recorded, so
    re-introducing the id fails the gate.
``keep``
    A deliberate non-decision: leave it in the accepted baseline for now, with a
    stated reason. Never suggested, only chosen.

The tool's own fifth *suggestion*, ``review``, is not a disposition — it is the
tool saying "the cheap signals ran out here". Its size is what a domain costs.

Evidence collected per concept
------------------------------
* **Marker sites** — every ``file:line`` with its text, split into source /
  test / doc. A concept living only in a test file is rarely a decision.
* **Module locality** — the non-test source modules carrying the marker.
* **Git archaeology** — the commit that first *added* the marker and the other
  concepts added by that same commit. ONE history walk per domain
  (``git log -G<prefix> -p``), not one per concept: ~9 s for 79 concepts rather
  than ~4.5 minutes.
* **Id quality** — whether the id reads as a slugified prose fragment, a bare
  legacy pillar citation, or a single generic noun; and whether the marker text
  in source is *truncated* by the grammar (the author wrote
  ``…compute.vector/214/215``; only ``…compute.vector`` is a legal id, so the
  recorded id is an artifact of the regex rather than a name anyone chose).

Clustering
----------
Connected components over a **thresholded similarity** graph: Jaccard overlap of
source-file footprints, plus a link for any *narrow* introducing commit. Raw
"shares a file" does not work — see :data:`CLUSTER_MIN_JACCARD`.

Workflow (resumable, idempotent, per domain)
--------------------------------------------
    concept_domain_triage.py domains                  # what is left, biggest first
    concept_domain_triage.py propose AU-KG.compute    # free: evidence + suggestions
    concept_domain_triage.py packet  AU-KG.compute    # the undecided residue only
    #   ... a human/agent edits `decision:` (+ parent/rationale/reason) per concept
    concept_domain_triage.py apply AU-KG.compute      # dry run: shows every edit
    concept_domain_triage.py apply AU-KG.compute --write
    concept_domain_triage.py status AU-KG.compute

``propose`` re-run on an already-triaged domain preserves every human decision
and refreshes only the evidence and suggestions, so it is safe to re-run after
the tree moves. ``apply`` is idempotent: an already-written lineage entry or an
already-removed marker is a no-op, not an error.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_utilities.governance.concept_hierarchy import (  # noqa: E402
    iter_okf_markers,
    parse_okf_id,
)
from agent_utilities.governance.concept_lineage import (  # noqa: E402
    LINEAGE_PATH,
    LineageError,
    parse_lineage,
)
from scripts.check_concept_governance import (  # noqa: E402
    all_registered_concepts,
    has_design_doc,
    read_baseline,
)

#: Proposals live under ``.specify/`` (the spec-driven-development state
#: tree), NOT ``reports/`` — the repo gitignores ``reports/`` wholesale, and a
#: triage proposal is durable, resumable, reviewable state that has to survive
#: a worktree prune. It is deliberately outside ``.specify/design/``, which the
#: gate greps: a proposal must never be able to satisfy the rule it feeds.
PROPOSAL_DIR = ROOT / ".specify" / "triage"

DECISIONS = ("PENDING", "document", "parent", "retire", "keep")

#: Words that make an id read as a sentence fragment rather than a name. An id
#: whose FIRST or LAST segment word is one of these was almost certainly
#: slugified out of prose ("when the engine exposes...", "data is private, its
#: owner...") rather than chosen.
#: Deliberately narrow: only closed-class function words (articles, copulas,
#: auxiliaries, conjunctions, pronouns, prepositions). An earlier, wider list
#: that included quantifiers and ``per``/``via``/``across`` produced false
#: positives on perfectly good names — ``per-channel-embedding-backfill`` is a
#: name; ``when-exposes`` is a sentence.
_PROSE_LEADERS = frozenset(
    """a an the and or but if when while is are was were be been being has have had
    does did do this that these those it its their his her our your my as at by for
    from in into of on to with so then than there here we you they he she not
    such""".split()
)

#: A slug that is a single generic noun carries no decision either — it names a
#: subject area, not a choice. (``AU-KG.compute.vector`` is not a decision; "use
#: the engine's native ANN instead of a Python vector path" is.)
_MIN_SLUG_WORDS = 2

#: ``kg-2`` / ``orch-1`` — the concept segment is a bare reference to the OLD
#: ``KG-2.NN`` pillar numbering that OKF-CIS replaced. Migration glued the
#: citation on as if it were a name.
_PILLAR_NUMERIC_RE = re.compile(r"(?:orch|kg|ahe|eco|os|gbot)-\d+", re.I)


#: Emitted evidence quotes real source lines, which contain real ``CONCEPT:``
#: markers. The audit scans ``.md``/``.yaml`` under ``reports/`` too, so quoting
#: a marker verbatim MINTS A NEW LIVE CONCEPT out of the triage output itself —
#: this tool's first run invented ``AU-KG.compute.homelab-rss-reader`` that way,
#: by quoting a marker that its own 100-char excerpt had cut short. Every quote
#: is therefore emitted with the prefix's colon replaced by a middle dot: still
#: obvious to a reader, invisible to the scanner. (The gate's own meta-test
#: dodges the same trap by string concatenation — see commit 38a9b918.)
_MARKER_PREFIX = "CONCEPT:"
_MASKED_PREFIX = "CONCEPT·"


def mask_markers(text: str) -> str:
    """Neutralise ``CONCEPT:`` prefixes so emitted evidence is not itself a marker."""
    return text.replace(_MARKER_PREFIX, _MASKED_PREFIX)


def _run(*args: str, cwd: Path = ROOT) -> str:
    proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 and not proc.stdout:
        raise RuntimeError(f"{' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


@dataclass
class Site:
    path: str
    line: int
    text: str

    @property
    def kind(self) -> str:
        if self.path.startswith(("tests/", "test/")):
            return "test"
        if self.path.endswith((".md", ".txt")) or self.path.startswith(
            (".specify/", "docs/")
        ):
            return "doc"
        return "source"


@dataclass
class Evidence:
    concept: str
    sites: list[Site] = field(default_factory=list)
    introduced_commit: str = ""
    introduced_date: str = ""
    introduced_subject: str = ""
    cohort: list[str] = field(default_factory=list)
    has_doc: bool = False
    truncated_marker: str = ""

    @property
    def source_files(self) -> list[str]:
        return sorted({s.path for s in self.sites if s.kind == "source"})

    @property
    def test_files(self) -> list[str]:
        return sorted({s.path for s in self.sites if s.kind == "test"})

    @property
    def doc_files(self) -> list[str]:
        return sorted({s.path for s in self.sites if s.kind == "doc"})

    @property
    def slug_words(self) -> list[str]:
        return parse_okf_id(self.concept).concept.split("-")

    @property
    def prose_id(self) -> bool:
        """The id reads as a slugified sentence fragment, not a chosen name."""
        words = self.slug_words
        if not words:
            return True
        if words[0] in _PROSE_LEADERS:
            return True
        return words[-1] in _PROSE_LEADERS

    @property
    def artifact_id(self) -> str:
        """Why this id names nothing, or ``""`` if it looks like a real name.

        Three independent tells, all found in AU-KG.compute: a slugified prose
        fragment (``when-exposes``, from "when the engine exposes..."), a bare
        legacy pillar reference (``kg-2``, from the old ``KG-2.16`` numbering —
        that is a *citation*, not a name), and a single generic noun
        (``vector``, ``resolve``, ``automates`` — a subject area, not a choice).
        """
        words = self.slug_words
        if self.prose_id:
            return f"reads as a slugified prose fragment ({'-'.join(words)})"
        if _PILLAR_NUMERIC_RE.fullmatch("-".join(words)):
            return (
                f"is a bare legacy pillar reference ({'-'.join(words)}) — a citation "
                "of the old KG-N.NN numbering, not a name anyone chose"
            )
        if len(words) < _MIN_SLUG_WORDS:
            return "is a single generic noun — it names a subject area, not a choice"
        return ""


def collect_sites(domain_prefix: str, concepts: set[str]) -> dict[str, list[Site]]:
    """Every marker site for *concepts*, from one ``git grep`` over the tree."""
    out: dict[str, list[Site]] = defaultdict(list)
    raw = _run("git", "grep", "-n", "-E", f"CONCEPT:{re.escape(domain_prefix)}\\.")
    for row in raw.splitlines():
        parts = row.split(":", 2)
        if len(parts) < 3:
            continue
        path, lineno, text = parts[0], parts[1], parts[2]
        if not lineno.isdigit():
            continue
        for marker in iter_okf_markers(text):
            cid = marker.id
            if cid not in concepts:
                continue
            out[cid].append(Site(path=path, line=int(lineno), text=text.strip()))
    return out


def detect_truncation(sites: list[Site], concept: str) -> str:
    """Return an example line where the author wrote MORE than the id captured.

    A marker written ``CONCEPT·AU-KG.compute.vector/214/215`` (masked here so
    this docstring does not itself mint the concept) records the id
    ``...compute.vector`` only because the marker grammar stops at ``/``. The
    recorded id is then an artifact of the regex, not a name — combined with a
    weak id, the strongest available retire signal.
    """
    for site in sites:
        for marker in iter_okf_markers(site.text):
            if marker.id == concept and marker.tail:
                return site.text
    return ""


def git_archaeology(domain_prefix: str, concepts: set[str]) -> dict[str, Evidence]:
    """One history walk for the whole domain: first-adding commit + cohort.

    ``git log -G<prefix> -p`` walks history once and prints every patch that
    changed a line matching the domain prefix. Walking per-concept instead costs
    ~3.5s each (~4.5 minutes for a 77-concept domain); this is ~9s total.
    """
    marker = "@@TRIAGE@@"
    raw = _run(
        "git",
        "log",
        f"--format={marker} %H %ad %s",
        "--date=short",
        f"-G CONCEPT:{re.escape(domain_prefix)}\\.",
        "-p",
        "--unified=0",
    )
    # git log is newest-first; the LAST commit that adds an id is the first one
    # chronologically, so simply overwrite as we walk.
    first: dict[str, tuple[str, str, str]] = {}
    per_commit: dict[str, set[str]] = defaultdict(set)
    commit = date = subject = ""
    for line in raw.splitlines():
        if line.startswith(marker):
            _, commit, date, subject = (line.split(" ", 3) + ["", "", ""])[:4]
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for mk in iter_okf_markers(line):
            cid = mk.id
            if cid not in concepts:
                continue
            first[cid] = (commit, date, subject)
            per_commit[commit].add(cid)

    out: dict[str, Evidence] = {}
    for cid in concepts:
        commit, date, subject = first.get(cid, ("", "", ""))
        cohort = sorted(per_commit.get(commit, set()) - {cid}) if commit else []
        out[cid] = Evidence(
            concept=cid,
            introduced_commit=commit,
            introduced_date=date,
            introduced_subject=subject,
            cohort=cohort,
        )
    return out


#: Two concepts are proposed as siblings when their *source-file footprints*
#: overlap at least this much (Jaccard). A raw "shares any file" relation does
#: not work: hub modules (``source_sync.py``, ``graph_compute.py``,
#: ``core/config.py``) each carry a dozen unrelated markers, and transitive
#: closure over "shares a file" fused 70 of AU-KG.compute's 79 concepts into one
#: meaningless blob on the first run. Jaccard asks the right question — "are
#: these two markers on the SAME body of code?" — so nine connector markers
#: living in exactly {source_sync, mcp_tool, owl_bridge} cluster tightly (0.67+)
#: while two strangers that merely both touch ``graph_compute.py`` do not.
CLUSTER_MIN_JACCARD = 0.4

#: A shared introducing commit is strong evidence of one decision — but only
#: when the commit was about these concepts. A mass rename (the 2026-07-03
#: OKF-CIS cutover) "introduced" hundreds of ids at once and says nothing about
#: any of them, so commits above this width contribute no link.
MAX_COHESIVE_COMMIT_CONCEPTS = 6


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_clusters(evidence: dict[str, Evidence]) -> dict[str, list[str]]:
    """Connected components over a *thresholded similarity* graph.

    Edge (a, b) exists when the two markers sit on substantially the same source
    files, or when one narrow commit introduced both. See
    :data:`CLUSTER_MIN_JACCARD` for why raw file-sharing is not usable.
    """
    ids = sorted(evidence)
    footprint = {cid: set(evidence[cid].source_files) for cid in ids}

    by_commit: dict[str, list[str]] = defaultdict(list)
    for cid in ids:
        if evidence[cid].introduced_commit:
            by_commit[evidence[cid].introduced_commit].append(cid)
    cohesive_commits = {
        sha: members
        for sha, members in by_commit.items()
        if 1 < len(members) <= MAX_COHESIVE_COMMIT_CONCEPTS
    }

    parent: dict[str, str] = {cid: cid for cid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if _jaccard(footprint[a], footprint[b]) >= CLUSTER_MIN_JACCARD:
                union(a, b)
    for members in cohesive_commits.values():
        for other in members[1:]:
            union(members[0], other)

    clusters: dict[str, list[str]] = defaultdict(list)
    for cid in ids:
        clusters[find(cid)].append(cid)
    return {root: sorted(members) for root, members in clusters.items()}


def choose_parent(members: list[str], evidence: dict[str, Evidence]) -> str | None:
    """The cluster member most likely to be the decision the others realise.

    Preference order: a member that already has a design document (the decision
    is written down, point at it); else the member with the widest source-module
    spread, since the marker on the most code is the one naming the thing rather
    than a use of it. Ids that name nothing (:attr:`Evidence.artifact_id`) are
    never eligible — pointing children at a junk id buries them.
    """
    documented = [cid for cid in members if evidence[cid].has_doc]
    if documented:
        return sorted(documented, key=lambda c: (-len(evidence[c].source_files), c))[0]
    eligible = [cid for cid in members if not evidence[cid].artifact_id]
    if not eligible:
        return None
    return sorted(eligible, key=lambda c: (-len(evidence[c].source_files), c))[0]


def suggest(
    cid: str, evidence: dict[str, Evidence], clusters: dict[str, list[str]]
) -> tuple[str, str, str]:
    """Propose ``(decision, parent_or_empty, why)`` for one concept.

    ``review`` is a first-class outcome, not a failure: it is how the tool says
    "the cheap signals disagree or run out here". Everything else is decided
    deterministically for free, so ``review`` is exactly the set worth spending
    model tokens on — and its size is what the per-domain cost scales with.
    """
    ev = evidence[cid]
    artifact = ev.artifact_id
    if artifact:
        why = f"the id {artifact}"
        if ev.truncated_marker:
            why += (
                "; and the marker in source is truncated "
                f"({mask_markers(ev.truncated_marker.strip()[:90])!r}) — the recorded id is what "
                "the grammar could hold, not what the author wrote"
            )
        return "retire", "", why

    if ev.truncated_marker:
        # The captured id reads like a real name, but the marker text carries
        # trailing legacy numbering the grammar dropped. That is a marker-text
        # defect, NOT evidence the concept is junk — deciding between "clean the
        # marker and keep it" and "retire it" needs a human.
        return (
            "review",
            "",
            "the marker text is truncated by the grammar "
            f"({mask_markers(ev.truncated_marker.strip()[:90])!r}) — the id itself reads like a "
            "real name, so the marker text needs cleaning either way; decide "
            "whether the concept survives that cleanup",
        )

    if not ev.source_files:
        where = "test files" if ev.test_files else "prose/doc files"
        return (
            "review",
            "",
            f"the marker exists only in {where} — nothing in the shipped tree "
            "realises it, which is usually a retirement but occasionally a real "
            "decision recorded only in prose",
        )

    for members in clusters.values():
        if cid not in members:
            continue
        if len(members) == 1:
            break
        candidate = choose_parent(members, evidence)
        if candidate and candidate != cid:
            shared = sorted(
                set(evidence[cid].source_files) & set(evidence[candidate].source_files)
            )
            why = f"clusters with {len(members) - 1} sibling(s) around {candidate}"
            if shared:
                why += f"; shares {', '.join(shared[:3])}"
            if (
                ev.introduced_commit
                and ev.introduced_commit == evidence[candidate].introduced_commit
            ):
                why += f"; both introduced by {ev.introduced_commit[:8]} ({ev.introduced_subject[:60]})"
            if evidence[candidate].has_doc:
                why += "; the candidate parent already has a design document"
            return "parent", candidate, why
        if candidate == cid:
            return (
                "document",
                "",
                f"the head of a {len(members)}-concept cluster — the siblings are "
                "proposed as pointers at this one, so this is the decision that "
                "has to be written down",
            )
        break

    return (
        "document",
        "",
        "a singleton: no sibling shares its source footprint or introducing commit "
        f"({len(ev.source_files)} source file(s), {len(ev.sites)} marker site(s))",
    )


def gather(domain_prefix: str) -> tuple[dict[str, Evidence], dict[str, list[str]]]:
    """All evidence + clusters for one ``PILLAR.domain`` (e.g. ``AU-KG.compute``)."""
    live = set(all_registered_concepts())
    concepts = {cid for cid in live if cid.startswith(f"{domain_prefix}.")}
    if not concepts:
        raise SystemExit(f"no live concepts under {domain_prefix!r}")

    evidence = git_archaeology(domain_prefix, concepts)
    sites = collect_sites(domain_prefix, concepts)
    for cid, ev in evidence.items():
        ev.sites = sorted(sites.get(cid, []), key=lambda s: (s.path, s.line))
        ev.has_doc = has_design_doc(cid)
        ev.truncated_marker = detect_truncation(ev.sites, cid)
    return evidence, build_clusters(evidence)


def proposal_path(domain_prefix: str) -> Path:
    return PROPOSAL_DIR / f"{domain_prefix}.yaml"


def _load_yaml(path: Path) -> dict:
    import yaml

    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, data: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


def cmd_propose(domain_prefix: str, *, refresh_suggestions: bool = True) -> int:
    evidence, clusters = gather(domain_prefix)
    path = proposal_path(domain_prefix)
    existing = _load_yaml(path).get("concepts") or {}

    baseline = read_baseline()
    concepts_out: dict[str, dict] = {}
    for cid in sorted(evidence):
        ev = evidence[cid]
        prior = existing.get(cid) or {}
        decision, parent, why = suggest(cid, evidence, clusters)
        entry = {
            "decision": prior.get("decision", "PENDING"),
            "parent": prior.get("parent", ""),
            "rationale": prior.get("rationale", ""),
            "reason": prior.get("reason", ""),
        }
        if refresh_suggestions or "suggestion" not in prior:
            entry["suggestion"] = decision
            entry["suggested_parent"] = parent
            entry["suggestion_why"] = why
        else:
            entry["suggestion"] = prior.get("suggestion", decision)
            entry["suggested_parent"] = prior.get("suggested_parent", parent)
            entry["suggestion_why"] = prior.get("suggestion_why", why)
        entry["evidence"] = {
            "has_design_doc": ev.has_doc,
            "in_baseline": cid in baseline,
            "source_files": ev.source_files,
            "test_files": ev.test_files,
            "doc_files": ev.doc_files,
            "introduced": (
                f"{ev.introduced_commit[:8]} {ev.introduced_date} {ev.introduced_subject}"
                if ev.introduced_commit
                else ""
            ),
            "cohort": ev.cohort,
            "artifact_id": ev.artifact_id,
            "truncated_marker": mask_markers(ev.truncated_marker),
            "sites": [f"{s.path}:{s.line}" for s in ev.sites],
        }
        concepts_out[cid] = entry

    dropped = sorted(set(existing) - set(concepts_out))
    for cid in dropped:
        if (existing[cid] or {}).get("decision", "PENDING") not in {
            "PENDING",
            "retire",
        }:
            print(
                f"WARNING: {cid} carried decision "
                f"{existing[cid]['decision']!r} but is no longer a live concept — "
                "keeping the entry so the decision is not silently lost.",
                file=sys.stderr,
            )
            concepts_out[cid] = existing[cid]

    _dump_yaml(
        path,
        {
            "domain": domain_prefix,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "generator": "scripts/concept_domain_triage.py",
            "how_to_use": (
                "Set `decision` on each concept to one of: document | parent | retire | "
                "keep. `parent` needs `parent:` + `rationale:`; `retire` needs `reason:`; "
                "`keep` needs `reason:`. Then run `apply` (dry-run) and `apply --write`."
            ),
            "clusters": {
                sorted(members)[0]: sorted(members)
                for members in clusters.values()
                if len(members) > 1
            },
            "concepts": concepts_out,
        },
    )
    counts: dict[str, int] = defaultdict(int)
    for entry in concepts_out.values():
        counts[entry["suggestion"]] += 1
    print(f"{path.relative_to(ROOT)}: {len(concepts_out)} concept(s)")
    print("  suggested: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


#: Leftovers that prove the tidy-up did not understand the line: an empty bold
#: span, an empty or comma-only parenthetical, a dangling separator. Rather than
#: guess further, the site is reported ``MANUAL`` and left untouched — a mangled
#: comment is worse than an un-retired marker, because nobody re-reads it.
_BROKEN_LEFTOVERS = (
    re.compile(r"\*\*\s*\*\*"),
    re.compile(r"\(\s*[,;]?\s*\)"),
    re.compile(r"[,;]\s*\)"),
    re.compile(r"·\s*·"),
    re.compile(r"``\s*``"),
    # `(/214/215)` — the marker was TRUNCATED by the grammar, so removing the
    # legal-id part leaves the citation tail the author actually wrote. Only a
    # human knows whether that tail should survive.
    re.compile(r"\(\s*[/.]"),
    # The rest are all one shape: the marker was part of a construct that SPANS
    # LINES, so a line-local edit cannot see the other half. `(engine )`,
    # `( → KG-2.16)`, `retriever — ),` and a line that now begins with `.` or
    # `)` are each the residue of a parenthetical or sentence opened elsewhere.
    # Rewriting them needs the surrounding lines; the tool only sees one.
    re.compile(r"\(\s*[→—–]"),
    re.compile(r"[—–]\s*\)"),
    re.compile(r"[ \t]+\)"),
    re.compile(r"^[ \t]*[.,;)]"),
)


def _strip_marker(text: str, concept: str) -> tuple[str, bool]:
    """Remove every ``CONCEPT:<id>`` token from *text*, tidying the leftovers.

    Returns ``(new_text, needs_human)``. ``needs_human`` is True when the line
    would be left as a husk (a comment that was ONLY the marker) or with visible
    debris the tidy-up could not resolve. Those sites are reported, never
    rewritten: this function is allowed to be conservative and useless, it is not
    allowed to quietly damage prose.
    """
    token = f"CONCEPT:{concept}"
    esc = re.escape(token)
    # Leading whitespace is INDENTATION and is never touched. The first version
    # of this ran a `\s{2,} -> " "` collapse over the whole line and silently
    # dedented a Python docstring into an IndentationError; the damage is
    # invisible in a one-line diff preview, which is exactly why it survived a
    # dry-run review.
    indent = text[: len(text) - len(text.lstrip())]
    out = text[len(indent) :]

    # Markdown bold/inline-code wrappers around the marker go with it.
    out = re.sub(rf"\*\*{esc}\*\*", "", out)
    out = re.sub(rf"``{esc}``", "", out)
    # A parenthetical that was ONLY the marker, or a trailing ", MARKER)" tail.
    out = re.sub(rf"\s*\({esc}\)", "", out)
    out = re.sub(rf"[,;]\s*{esc}(?=\s*[)\]])", "", out)
    # A leading "MARKER — " label.
    out = re.sub(rf"{esc}\s*[—–-]\s*", "", out)
    out = out.replace(token, "")
    out = re.sub(r"\(\s*,\s*", "(", out)
    out = re.sub(r"[ \t]{2,}", " ", out).rstrip()

    if any(pattern.search(out) for pattern in _BROKEN_LEFTOVERS):
        return text, True
    husk = out.strip().strip("#>*·").strip("─-—= \t\"'")
    return indent + out, not husk


def _apply_retirement(concept: str, sites: list[Site], *, write: bool) -> list[str]:
    """Delete the marker from every site. Returns human-readable edit lines."""
    edits: list[str] = []
    by_file: dict[str, list[Site]] = defaultdict(list)
    for site in sites:
        by_file[site.path].append(site)
    for path, file_sites in sorted(by_file.items()):
        target = ROOT / path
        if not target.exists():
            continue
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        changed = False
        for site in file_sites:
            idx = site.line - 1
            if idx >= len(lines) or f"CONCEPT:{concept}" not in lines[idx]:
                continue  # already applied, or the tree moved — idempotent no-op
            ending = "\n" if lines[idx].endswith("\n") else ""
            new_text, degenerate = _strip_marker(lines[idx].rstrip("\n"), concept)
            if degenerate:
                edits.append(
                    f"  MANUAL {path}:{site.line} — line is only the marker: {site.text[:80]}"
                )
                continue
            edits.append(
                f"  edit   {path}:{site.line}\n    -{site.text[:100]}\n    +{new_text.strip()[:100]}"
            )
            lines[idx] = new_text + ending
            changed = True
        if changed and write:
            target.write_text("".join(lines), encoding="utf-8")
    return edits


def cmd_apply(domain_prefix: str, *, write: bool) -> int:
    path = proposal_path(domain_prefix)
    data = _load_yaml(path)
    if not data:
        raise SystemExit(f"no proposal at {path} — run `propose {domain_prefix}` first")
    concepts = data.get("concepts") or {}

    evidence, _ = gather(domain_prefix)
    lineage_raw = _load_yaml(LINEAGE_PATH)
    parents = dict(lineage_raw.get("parents") or {})
    retired = dict(lineage_raw.get("retired") or {})

    problems: list[str] = []
    owed: list[str] = []
    deferred: list[str] = []
    to_retire: list[str] = []
    edits: list[str] = []
    counts: dict[str, int] = defaultdict(int)

    for cid, entry in sorted(concepts.items()):
        decision = (entry or {}).get("decision", "PENDING")
        counts[decision] += 1
        if decision in {"PENDING", "keep"}:
            if decision == "keep" and not (entry.get("reason") or "").strip():
                problems.append(f"{cid}: decision 'keep' requires a reason")
            continue
        if decision not in DECISIONS:
            problems.append(
                f"{cid}: unknown decision {decision!r} (expected one of {DECISIONS})"
            )
            continue
        if decision == "document":
            # NOT a blocker: "this earns its own document" is a classification,
            # and classifying a domain is useful before every document is
            # written. The gate is the enforcer here — an undocumented concept
            # simply stays in the accepted-debt baseline until someone writes it.
            if not has_design_doc(cid):
                owed.append(cid)
            continue
        if decision == "parent":
            parent = (entry.get("parent") or "").strip()
            if not parent:
                problems.append(f"{cid}: decision 'parent' requires a `parent:` id")
                continue
            if not has_design_doc(parent):
                # Deferred, not blocked. Refusing to write a link into an
                # undocumented parent is the safety property; refusing every
                # OTHER link because one parent's document is unwritten is just
                # an unresumable tool. The decision stays recorded in the
                # proposal and lands on a later run.
                deferred.append(
                    f"{cid} -> {parent} (waiting on the parent's design document)"
                )
                continue
            parents[cid] = {
                "parent": parent,
                "rationale": (entry.get("rationale") or "").strip(),
            }
            edits.append(f"  lineage parent  {cid} -> {parent}")
            continue
        if decision == "retire":
            reason = (entry.get("reason") or "").strip()
            if not reason:
                problems.append(f"{cid}: decision 'retire' requires a `reason:`")
                continue
            retired[cid] = {"reason": reason}
            edits.append(f"  lineage retire  {cid}")
            # A retired concept that is no longer live has already had its
            # markers removed (by an earlier `apply --write`, or by hand for a
            # site the tool refused to touch). Recording the retirement is still
            # required — that is the ratchet — but there is nothing left to edit.
            if cid in evidence:
                to_retire.append(cid)

    # Validate the candidate registry BEFORE writing it: the loader's rules (one
    # hop, no self-parent, no restatement rationale) must hold, and finding out
    # after the write means a broken registry is already on disk.
    candidate = {"parents": parents, "retired": retired}
    try:
        parse_lineage(candidate)
    except LineageError as exc:
        raise SystemExit(
            f"the resulting lineage registry would be invalid: {exc}"
        ) from exc

    # Only NOW touch the tree. `apply` is all-or-nothing on purpose: a run that
    # is going to refuse to write the registry must not have already deleted
    # markers from source, or the tree ends up with retired markers gone and no
    # record that they were retired — a state the ratchet cannot detect.
    for cid in to_retire:
        edits.extend(
            _apply_retirement(cid, evidence[cid].sites, write=write and not problems)
        )

    print(
        f"{domain_prefix}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    if owed:
        print(
            f"\n{len(owed)} concept(s) are classified 'document' but nobody has "
            "written it yet — they stay in the accepted-debt baseline until someone does:"
        )
        for cid in owed:
            print(f"  - {cid}")
    if deferred:
        print(
            f"\n{len(deferred)} parent link(s) DEFERRED — the decision is recorded, "
            "the link lands once the parent's design document exists:"
        )
        for msg in deferred:
            print(f"  - {msg}")
    if problems:
        print("\nBLOCKED — fix these in the proposal first:")
        for msg in problems:
            print(f"  - {msg}")
    if edits:
        print(
            f"\n{'APPLYING' if write else 'DRY RUN — would apply'} {len(edits)} edit(s):"
        )
        for msg in edits:
            print(msg)
    if problems:
        return 1
    if write:
        _write_lineage(parents, retired)
        print(f"\nWrote {LINEAGE_PATH.relative_to(ROOT)}.")
        print(
            "Next: `python3 scripts/check_concept_governance.py --audit-merged` to "
            "confirm the gate honours the links, then `--update-baseline` to shrink "
            "the accepted-debt list."
        )
    elif edits:
        print("\nRe-run with --write to apply.")
    return 0


def _write_lineage(parents: dict, retired: dict) -> None:
    """Rewrite the registry, preserving its explanatory header."""
    import yaml

    header_lines: list[str] = []
    if LINEAGE_PATH.exists():
        for line in LINEAGE_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                header_lines.append(line)
            else:
                break
    body = yaml.safe_dump(
        {"parents": parents or {}, "retired": retired or {}},
        sort_keys=True,
        allow_unicode=True,
        width=100,
    )
    LINEAGE_PATH.write_text(
        "\n".join(header_lines).rstrip() + "\n\n" + body, encoding="utf-8"
    )


def _best_marker_lines(ev: Evidence, limit: int = 2) -> list[str]:
    """The most informative marker lines: prefer source, then the longest prose.

    A marker's own sentence is the single densest evidence of what decision (if
    any) it names — far denser than the file list. Two lines is empirically
    enough to adjudicate and keeps the packet small.
    """
    ranked = sorted(
        ev.sites,
        key=lambda s: ({"source": 0, "doc": 1, "test": 2}[s.kind], -len(s.text)),
    )
    seen: set[str] = set()
    out: list[str] = []
    for site in ranked:
        text = re.sub(r"\s+", " ", site.text).strip()
        if text in seen:
            continue
        seen.add(text)
        out.append(mask_markers(f"{site.path}:{site.line} | {text[:180]}"))
        if len(out) >= limit:
            break
    return out


def cmd_packet(domain_prefix: str) -> int:
    """Emit a compact adjudication packet — ONLY what the cheap pass could not decide.

    This is the token-cost lever. The deterministic pass classifies most of a
    domain for free; re-reading all of it costs tokens for no new information.
    The packet carries the ``review`` residue, the proposed cluster heads (which
    are the "is this really a decision?" calls), and one line per cluster instead
    of one per member — so confirming N pointers costs one judgement, not N.
    """
    data = _load_yaml(proposal_path(domain_prefix))
    if not data:
        raise SystemExit(f"no proposal for {domain_prefix} — run `propose` first")
    evidence, _ = gather(domain_prefix)
    concepts = data.get("concepts") or {}
    clusters = data.get("clusters") or {}

    prefix = f"{domain_prefix}."
    short = lambda cid: cid[len(prefix) :] if cid.startswith(prefix) else cid  # noqa: E731

    by_suggestion: dict[str, list[str]] = defaultdict(list)
    for cid, entry in concepts.items():
        by_suggestion[(entry or {}).get("suggestion", "review")].append(cid)

    lines: list[str] = [
        f"# Adjudication packet — {domain_prefix}",
        "",
        f"{len(concepts)} live concepts. The deterministic pass already decided "
        f"{len(by_suggestion['parent'])} pointer(s) and "
        f"{len(by_suggestion['retire'])} retirement(s) from module locality, git "
        "archaeology and id shape alone. Confirm or correct the items below, then "
        f"write the decisions into .specify/triage/{domain_prefix}.yaml.",
        "",
        "## Clusters — confirm ONE parent each; the members inherit it",
        "",
    ]
    for head, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        proposed = next(
            (
                (concepts[m] or {}).get("suggested_parent")
                for m in members
                if (concepts.get(m) or {}).get("suggested_parent")
                and (concepts[m] or {}).get("suggested_parent") in evidence
            ),
            head,
        )
        # An already-applied domain has retired ids still named in the stale
        # cluster map; they are simply gone from the tree now.
        if proposed not in evidence:
            continue
        lines.append(f"### {short(proposed)}  ({len(members)} concepts)")
        for line in _best_marker_lines(evidence[proposed], 2):
            lines.append(f"    {line}")
        lines.append("    members: " + ", ".join(short(m) for m in members))
        lines.append("")

    for bucket, title in (
        ("retire", "Proposed RETIRE — the id names nothing (confirm or rescue)"),
        ("review", "UNDECIDED — the cheap signals ran out here"),
        ("document", "Proposed OWN DOCUMENT — is this really a decision?"),
    ):
        members = sorted(by_suggestion.get(bucket, []))
        if not members:
            continue
        lines += [f"## {title} ({len(members)})", ""]
        for cid in members:
            if cid not in evidence:
                continue  # retired since this proposal was written
            entry = concepts[cid]
            lines.append(f"### {short(cid)}")
            lines.append(f"    why: {entry.get('suggestion_why', '')}")
            for line in _best_marker_lines(evidence[cid], 2):
                lines.append(f"    {line}")
            lines.append("")

    out = "\n".join(lines).rstrip() + "\n"
    path = PROPOSAL_DIR / f"{domain_prefix}.packet.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out, encoding="utf-8")
    words = len(out.split())
    print(
        f"{path.relative_to(ROOT)}: {len(out)} chars, ~{words} words "
        f"(~{len(out) // 4} tokens) covering "
        f"{len(by_suggestion['review']) + len(by_suggestion['document'])} undecided "
        f"concept(s) + {len(clusters)} cluster head(s) out of {len(concepts)}."
    )
    return 0


def cmd_domains(limit: int) -> int:
    """Every ``PILLAR.domain`` with undocumented concepts, biggest group first."""
    baseline = read_baseline()
    live = set(all_registered_concepts())
    groups: dict[str, list[str]] = defaultdict(list)
    for cid in sorted(baseline & live):
        parsed = parse_okf_id(cid)
        groups[f"{parsed.slug}-{parsed.pillar}.{parsed.domain}"].append(cid)
    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    print(
        f"{len(ranked)} domain group(s), {sum(len(v) for v in groups.values())} concepts:"
    )
    for name, members in ranked[:limit]:
        triaged = proposal_path(name).exists()
        print(f"  {len(members):4d}  {name}{'  [proposal exists]' if triaged else ''}")
    return 0


def cmd_status(domain_prefix: str) -> int:
    data = _load_yaml(proposal_path(domain_prefix))
    if not data:
        raise SystemExit(f"no proposal for {domain_prefix}")
    counts: dict[str, int] = defaultdict(int)
    for entry in (data.get("concepts") or {}).values():
        counts[(entry or {}).get("decision", "PENDING")] += 1
    print(
        json.dumps(
            {"domain": domain_prefix, "decisions": dict(sorted(counts.items()))},
            indent=2,
        )
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_domains = sub.add_parser(
        "domains", help="list undocumented-concept domain groups"
    )
    p_domains.add_argument("--limit", type=int, default=100)

    p_propose = sub.add_parser(
        "propose", help="write/refresh a domain's triage proposal"
    )
    p_propose.add_argument("domain", help="e.g. AU-KG.compute")
    p_propose.add_argument(
        "--keep-suggestions",
        action="store_true",
        help="do not recompute suggestions for concepts already in the proposal",
    )

    p_apply = sub.add_parser(
        "apply", help="apply the confirmed decisions (dry run by default)"
    )
    p_apply.add_argument("domain")
    p_apply.add_argument(
        "--write", action="store_true", help="actually write the edits"
    )

    p_packet = sub.add_parser(
        "packet", help="emit a compact adjudication packet (only the undecided residue)"
    )
    p_packet.add_argument("domain")

    p_status = sub.add_parser("status", help="decision counts for a domain")
    p_status.add_argument("domain")

    args = ap.parse_args()
    if args.cmd == "domains":
        return cmd_domains(args.limit)
    if args.cmd == "packet":
        return cmd_packet(args.domain)
    if args.cmd == "propose":
        return cmd_propose(args.domain, refresh_suggestions=not args.keep_suggestions)
    if args.cmd == "apply":
        return cmd_apply(args.domain, write=args.write)
    if args.cmd == "status":
        return cmd_status(args.domain)
    raise AssertionError(f"unhandled command {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
