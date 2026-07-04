#!/usr/bin/python
from __future__ import annotations

"""Autonomous overnight knowledge swarm over a local-first markdown vault.

CONCEPT:AU-KG.research.run-one-autonomous-night

Operationalizes the "second brain night-shift" — the scheduled overnight swarm
of the *"How To Build a Second Brain That Runs Itself"* article — directly over a
LOCAL-FIRST MARKDOWN VAULT. While the day belongs to the human, at night five
named roles refine raw intake into a linked, contradiction-aware knowledge base
and leave a morning briefing behind:

    Scout       gather sources           → ``sources/`` (verbatim originals)
    Cataloger   source → atomic notes     → ``2-atoms/`` (one idea per file)
    Cartographer link each atom to >= 2    → ``links:`` frontmatter
    Critic      surface [FRICTION]         → never silently overwrite a belief
    Editor      weave threads + briefing   → ``3-threads/`` + ``briefings/``

The Critic role is delegated to the existing explicit node-vs-node
:class:`~..adaptation.contradiction_detector.ContradictionDetector` (CONCEPT:AU-KG.research.explicit-node-node-contradiction):
when a *new* atom contradicts an existing one the swarm attaches a **[FRICTION]**
note pointing at the conflicting belief — it proposes, it never arbitrates. Like
:mod:`..research.loop_controller`, the whole shift is **propose-only**: it adds
links and friction notes and writes briefings, but it never resolves a
contradiction and never deletes an atom.

House rules (the vault's constitution), enforced here:

* **Prime Directive — every atom traces to a real source.** ``0-raw/`` and
  ``sources/`` are READ-ONLY after first write; an atom is only ever created from
  a recorded source and always carries that ``source_id``. No source, no note.
* **Never delete — retire instead.** Retiring an atom marks it ``[RETIRED]`` and
  moves it to ``2-atoms/archive/``; the file is never removed.
* **Every atom links to >= ``min_links`` others** (default 2), so no idea is an
  island.

Everything is zero-infra and deterministic: atom extraction defaults to a
paragraph/sentence splitter and linking uses the same dependency-free lexical
similarity as the Critic, so the swarm runs with NO LLM and NO network. A caller
may inject a stronger ``extract_fn`` (e.g. an LLM atomizer) without changing any
other behavior. Paths are passed in, never read from the environment.

Concept: second-brain-night-shift
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..adaptation.contradiction_detector import (
    Claim,
    ContradictionDetector,
    lexical_similarity,
)

__all__ = [
    "ExtractFn",
    "AtomNote",
    "ShiftReport",
    "NightShiftSwarm",
    "default_extract",
]

# (source_text) -> list of atomic idea strings. The default is a deterministic
# paragraph/sentence splitter (no LLM, no network).
ExtractFn = Callable[[str], list[str]]

# Read-only stage names: the swarm must never mutate these after first write.
_READONLY_STAGES = ("0-raw", "sources")
# Writable refinement stages.
_ATOMS_STAGE = "2-atoms"
_THREADS_STAGE = "3-threads"
_BRIEFINGS_STAGE = "briefings"
_ARCHIVE_STAGE = "2-atoms/archive"

# Sentence boundary for the fallback splitter (a paragraph with no blank-line
# breaks is split into one atom per sentence so atoms stay "one idea per file").
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
# Frontmatter key/value lines in an atom .md file.
_FM_LINE = re.compile(r"^([a-z_]+):\s*(.*)$")
# Characters not safe in a flat filename — collapsed to ``_``.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def default_extract(source_text: str) -> list[str]:
    """Deterministic atomizer: one atom per paragraph, else per sentence.

    The zero-infra Cataloger: split the source on blank lines into paragraphs;
    a paragraph that still bundles several sentences is further split so each
    atom carries a single idea. Whitespace-only fragments are dropped. No model,
    no network — fully reproducible.
    """
    text = (source_text or "").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    atoms: list[str] = []
    for para in paragraphs:
        flat = " ".join(para.split())
        sentences = [s.strip() for s in _SENTENCE.split(flat) if s.strip()]
        atoms.extend(sentences or [flat])
    return atoms


@dataclass
class AtomNote:
    """One atomic idea: a single claim that always traces back to a source.

    ``links`` holds the ids of related atoms the Cartographer wired up (>= the
    swarm's ``min_links`` target); ``frictions`` holds human-readable
    ``[FRICTION]`` notes the Critic raised against existing beliefs. Retiring an
    atom flips ``retired`` and moves its file to the archive — it is never
    deleted.
    """

    atom_id: str
    claim: str
    source_id: str
    links: list[str] = field(default_factory=list)
    frictions: list[str] = field(default_factory=list)
    retired: bool = False


@dataclass
class ShiftReport:
    """Outcome of one overnight shift — what came in, grew, and needs judgment."""

    sources_ingested: int
    atoms_created: int
    links_added: int
    frictions: list[str]
    briefing_path: str


class NightShiftSwarm:
    """Autonomous overnight knowledge swarm over a local markdown vault (CONCEPT:AU-KG.research.run-one-autonomous-night)."""

    def __init__(
        self,
        vault_root: str | Path,
        *,
        extract_fn: ExtractFn | None = None,
        min_links: int = 2,
        similarity_min: float = 0.3,
    ) -> None:
        """Open (and lay out) a vault for the night shift.

        Args:
            vault_root: The local markdown vault directory. Created if missing,
                along with every refinement-stage subdirectory.
            extract_fn: Source-text → atomic-idea splitter. Defaults to the
                deterministic :func:`default_extract` (no LLM/network); inject a
                stronger atomizer to change *only* how sources become atoms.
            min_links: How many related atoms the Cartographer wires each atom to
                (the "every atom links to >= 2 others" house rule).
            similarity_min: Lexical-similarity floor below which two atoms are not
                considered related when wiring links.
        """
        self.vault_root = Path(vault_root)
        self.extract_fn: ExtractFn = extract_fn or default_extract
        self.min_links = int(min_links)
        self.similarity_min = float(similarity_min)
        self._critic = ContradictionDetector(min_similarity=0.15)
        self._ensure_layout()

    # -- vault layout ------------------------------------------------------ #
    def _ensure_layout(self) -> None:
        """Create the refinement-stage subdirs if missing (idempotent)."""
        for stage in (
            *_READONLY_STAGES,
            _ATOMS_STAGE,
            _THREADS_STAGE,
            _BRIEFINGS_STAGE,
            _ARCHIVE_STAGE,
        ):
            (self.vault_root / stage).mkdir(parents=True, exist_ok=True)

    @property
    def _atoms_dir(self) -> Path:
        return self.vault_root / _ATOMS_STAGE

    @property
    def _archive_dir(self) -> Path:
        return self.vault_root / _ARCHIVE_STAGE

    @staticmethod
    def _safe(name: str) -> str:
        """Collapse a free-form id into a safe flat filename stem."""
        cleaned = _UNSAFE.sub("_", (name or "").strip()).strip("_")
        return cleaned or "unnamed"

    # -- Scout ------------------------------------------------------------- #
    def scout(self, items: list[tuple[str, str]]) -> int:
        """Gather sources: write each verbatim to ``sources/`` + a ``0-raw/`` pointer.

        ``items`` are ``(source_id, text)`` pairs. Each source is written once,
        verbatim, to ``sources/<source_id>.md`` (READ-ONLY thereafter) with a
        breadcrumb pointer in ``0-raw/``. An already-recorded source is never
        overwritten (the Prime Directive's immutable provenance). Returns the
        number of NEW sources written.
        """
        sources_dir = self.vault_root / "sources"
        raw_dir = self.vault_root / "0-raw"
        written = 0
        for source_id, text in items:
            stem = self._safe(source_id)
            src_path = sources_dir / f"{stem}.md"
            if src_path.exists():
                # Immutable: a recorded source is never rewritten.
                continue
            src_path.write_text(text or "", encoding="utf-8")
            (raw_dir / f"{stem}.intake.md").write_text(
                f"intake: {source_id}\nsource: sources/{stem}.md\n",
                encoding="utf-8",
            )
            written += 1
        return written

    def _recorded_sources(self) -> dict[str, str]:
        """Map ``source_id`` (stem) → source text for every recorded source."""
        sources_dir = self.vault_root / "sources"
        out: dict[str, str] = {}
        for path in sorted(sources_dir.glob("*.md")):
            out[path.stem] = path.read_text(encoding="utf-8")
        return out

    # -- Cataloger --------------------------------------------------------- #
    def _existing_atoms(self) -> list[AtomNote]:
        """Load every atom currently filed in ``2-atoms/`` (archive excluded)."""
        atoms: list[AtomNote] = []
        for path in sorted(self._atoms_dir.glob("*.md")):
            atoms.append(self._read_atom(path))
        return atoms

    def _read_atom(self, path: Path) -> AtomNote:
        """Parse an atom .md file (frontmatter + body) back into an AtomNote."""
        text = path.read_text(encoding="utf-8")
        fm: dict[str, str] = {}
        body_lines: list[str] = []
        in_fm = False
        seen_fm = False
        for line in text.splitlines():
            if line.strip() == "---":
                in_fm = not in_fm
                seen_fm = True
                continue
            if in_fm:
                m = _FM_LINE.match(line)
                if m:
                    fm[m.group(1)] = m.group(2).strip()
            elif seen_fm:
                body_lines.append(line)
        links = [x for x in fm.get("links", "").split(",") if x.strip()]
        frictions_raw = fm.get("frictions", "")
        frictions = [f for f in frictions_raw.split("||") if f.strip()]
        return AtomNote(
            atom_id=path.stem,
            claim="\n".join(body_lines).strip(),
            source_id=fm.get("source", ""),
            links=[x.strip() for x in links],
            frictions=[f.strip() for f in frictions],
            retired=fm.get("retired", "false").strip().lower() == "true",
        )

    def _write_atom(self, atom: AtomNote, *, directory: Path | None = None) -> Path:
        """Serialize an AtomNote to a markdown file (frontmatter + claim body)."""
        directory = directory or self._atoms_dir
        path = directory / f"{atom.atom_id}.md"
        fm = [
            "---",
            f"id: {atom.atom_id}",
            f"source: {atom.source_id}",
            f"links: {', '.join(atom.links)}",
            f"frictions: {' || '.join(atom.frictions)}",
            f"retired: {'true' if atom.retired else 'false'}",
            "---",
            "",
            atom.claim,
            "",
        ]
        path.write_text("\n".join(fm), encoding="utf-8")
        return path

    def catalog(self) -> list[AtomNote]:
        """Catalog NEW sources into atomic notes, one ``.md`` per idea.

        For every recorded source that has no atoms yet, split its text via
        ``extract_fn`` and write one atom file per idea into ``2-atoms/``, each
        carrying its ``source_id`` in frontmatter. The Prime Directive is
        structural: an atom is only ever produced from a recorded source, so it
        can never exist without one. Returns the atoms created this call.
        """
        existing = self._existing_atoms()
        catalogued_sources = {a.source_id for a in existing}
        existing_ids = {a.atom_id for a in existing}
        new_atoms: list[AtomNote] = []
        for source_id, text in self._recorded_sources().items():
            if source_id in catalogued_sources:
                continue  # already catalogued — sources are append-only intake
            for idx, claim in enumerate(self.extract_fn(text)):
                claim = claim.strip()
                if not claim:
                    continue
                atom_id = self._unique_atom_id(source_id, idx, existing_ids)
                existing_ids.add(atom_id)
                atom = AtomNote(atom_id=atom_id, claim=claim, source_id=source_id)
                self._write_atom(atom)
                new_atoms.append(atom)
        return new_atoms

    @staticmethod
    def _unique_atom_id(source_id: str, idx: int, taken: set[str]) -> str:
        """Deterministic, collision-free atom id derived from its source."""
        base = f"{source_id}-a{idx:03d}"
        candidate = base
        bump = 0
        while candidate in taken:
            bump += 1
            candidate = f"{base}-{bump}"
        return candidate

    # -- Cartographer ------------------------------------------------------ #
    def cartograph(self, atoms: list[AtomNote]) -> int:
        """Link each given atom to its most-similar peers (>= ``min_links``).

        The Cartographer wires every atom in ``atoms`` to the ``min_links`` other
        atoms (across the whole vault) most lexically similar to it above
        ``similarity_min`` — falling back to the next-best peers so the "every
        atom links to >= 2 others" rule still holds in a sparse vault. Links are
        persisted to each atom's frontmatter. Returns the number of links added.
        """
        all_atoms = {a.atom_id: a for a in self._existing_atoms()}
        # Fold in the freshly-created atoms (their on-disk copies are current).
        for a in atoms:
            all_atoms[a.atom_id] = a
        pool = [a for a in all_atoms.values() if not a.retired]
        added = 0
        for atom in atoms:
            if atom.retired:
                continue
            scored: list[tuple[float, str]] = []
            for other in pool:
                if other.atom_id == atom.atom_id:
                    continue
                sim = lexical_similarity(atom.claim, other.claim)
                scored.append((sim, other.atom_id))
            # Strongest links first; id as a deterministic tiebreaker.
            scored.sort(key=lambda s: (-s[0], s[1]))
            chosen: list[str] = []
            for sim, oid in scored:
                if oid in atom.links or oid in chosen:
                    continue
                if sim >= self.similarity_min or len(chosen) < self.min_links:
                    chosen.append(oid)
                if len(chosen) >= self.min_links and sim < self.similarity_min:
                    break
            for oid in chosen:
                atom.links.append(oid)
                added += 1
            if chosen:
                self._write_atom(atom)
        return added

    # -- Critic ------------------------------------------------------------ #
    def critique(self, atoms: list[AtomNote]) -> list[str]:
        """Surface [FRICTION] where a new atom contradicts an existing belief.

        Runs the explicit :class:`ContradictionDetector` (CONCEPT:AU-KG.research.explicit-node-node-contradiction) over
        the new atoms' claims against the existing belief set. Each contradiction
        is recorded as a ``[FRICTION]`` note on the *new* atom pointing at the
        conflicting one and persisted to frontmatter. Propose-only: the swarm
        never resolves the tension or overwrites either belief. Returns the
        friction strings raised this call.
        """
        existing = {a.atom_id: a for a in self._existing_atoms()}
        new_ids = {a.atom_id for a in atoms}
        # The existing belief set the new atoms are judged against = everything
        # not part of this batch and not retired.
        belief_claims = [
            Claim(id=a.atom_id, text=a.claim)
            for a in existing.values()
            if a.atom_id not in new_ids and not a.retired
        ]
        frictions: list[str] = []
        for atom in atoms:
            if atom.retired:
                continue
            findings = self._critic.check(
                Claim(atom.atom_id, atom.claim), belief_claims
            )
            for f in findings:
                note = (
                    f"[FRICTION] atom '{atom.atom_id}' contradicts '{f.conflict_id}' "
                    f"(severity {f.severity}): {f.reason}"
                )
                if note not in atom.frictions:
                    atom.frictions.append(note)
                    frictions.append(note)
            if findings:
                # Re-read from disk to keep links written by an earlier stage,
                # then re-persist with the friction notes attached.
                disk = existing.get(atom.atom_id)
                if disk is not None:
                    atom.links = disk.links or atom.links
                self._write_atom(atom)
        return frictions

    # -- Editor ------------------------------------------------------------ #
    def edit(self, atoms: list[AtomNote]) -> str:
        """Weave related atoms into threads + write the morning briefing.

        Clusters linked atoms into thread documents under ``3-threads/`` and
        writes a morning briefing to ``briefings/<n>.md`` with three sections:
        *What Came In* (the night's new atoms), *Contradictions To Resolve* (the
        Critic's frictions, for human judgment), and *Threads That Grew* (the
        clusters). Returns the briefing path.
        """
        live = [a for a in self._existing_atoms() if not a.retired]
        threads = self._cluster(live)
        self._write_threads(threads)

        frictions: list[str] = []
        for a in live:
            frictions.extend(a.frictions)

        briefing_dir = self.vault_root / _BRIEFINGS_STAGE
        n = len(list(briefing_dir.glob("*.md"))) + 1
        briefing_path = briefing_dir / f"{n:04d}.md"
        lines: list[str] = [
            f"# Morning Briefing {n}",
            "",
            "## What Came In",
        ]
        if atoms:
            for a in atoms:
                lines.append(f"- `{a.atom_id}` — {a.claim} (source: {a.source_id})")
        else:
            lines.append("- (no new atoms)")
        lines += ["", "## Contradictions To Resolve"]
        if frictions:
            lines.extend(f"- {fr}" for fr in frictions)
        else:
            lines.append("- (none)")
        lines += ["", "## Threads That Grew"]
        if threads:
            for i, cluster in enumerate(threads, start=1):
                members = ", ".join(sorted(cluster))
                lines.append(f"- Thread {i}: {members}")
        else:
            lines.append("- (none)")
        lines.append("")
        briefing_path.write_text("\n".join(lines), encoding="utf-8")
        return str(briefing_path)

    def _cluster(self, live: list[AtomNote]) -> list[set[str]]:
        """Union-find over the link graph → connected clusters (deterministic)."""
        parent: dict[str, str] = {a.atom_id: a.atom_id for a in live}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[min(ra, rb)] = max(ra, rb)

        for atom in live:
            for oid in atom.links:
                if oid in parent:
                    union(atom.atom_id, oid)
        clusters: dict[str, set[str]] = {}
        for aid in parent:
            clusters.setdefault(find(aid), set()).add(aid)
        # Only clusters of >= 2 atoms are "threads"; sort for determinism.
        return [c for _, c in sorted(clusters.items()) if len(c) >= 2]

    def _write_threads(self, threads: list[set[str]]) -> None:
        """Write one thread doc per cluster into ``3-threads/`` (overwrites own)."""
        threads_dir = self.vault_root / _THREADS_STAGE
        by_id = {a.atom_id: a for a in self._existing_atoms()}
        for i, cluster in enumerate(threads, start=1):
            path = threads_dir / f"thread-{i:03d}.md"
            members = sorted(cluster)
            lines = [f"# Thread {i}", ""]
            for aid in members:
                atom = by_id.get(aid)
                claim = atom.claim if atom else ""
                lines.append(f"- `{aid}`: {claim}")
            lines.append("")
            path.write_text("\n".join(lines), encoding="utf-8")

    # -- retire ------------------------------------------------------------ #
    def retire(self, atom_id: str) -> bool:
        """Mark an atom ``[RETIRED]`` and move it to the archive — never delete.

        The atom file is rewritten with ``retired: true`` (and a ``[RETIRED]``
        marker on its claim) and relocated to ``2-atoms/archive/``. The file
        always continues to exist. Returns ``True`` if an atom was retired,
        ``False`` if no such live atom existed.
        """
        path = self._atoms_dir / f"{atom_id}.md"
        if not path.exists():
            return False
        atom = self._read_atom(path)
        atom.retired = True
        if not atom.claim.startswith("[RETIRED]"):
            atom.claim = f"[RETIRED] {atom.claim}"
        self._write_atom(atom, directory=self._archive_dir)
        path.unlink()  # moved, not destroyed — the archived copy persists
        return True

    # -- full pipeline ----------------------------------------------------- #
    def run_shift(self, items: list[tuple[str, str]] | None = None) -> ShiftReport:
        """Run the full night shift scout→catalog→cartograph→critique→edit.

        Honors immutability throughout: it writes ONLY to ``2-atoms/``,
        ``3-threads/`` and ``briefings/`` and never edits ``0-raw/`` or
        ``sources/`` after a source's first write. Returns a populated
        :class:`ShiftReport`.
        """
        sources_ingested = self.scout(items or [])
        new_atoms = self.catalog()
        links_added = self.cartograph(new_atoms)
        # Re-load so the critique stage sees the links the cartographer persisted.
        refreshed = {a.atom_id for a in new_atoms}
        new_atoms = [a for a in self._existing_atoms() if a.atom_id in refreshed]
        frictions = self.critique(new_atoms)
        new_atoms = [a for a in self._existing_atoms() if a.atom_id in refreshed]
        briefing_path = self.edit(new_atoms)
        return ShiftReport(
            sources_ingested=sources_ingested,
            atoms_created=len(new_atoms),
            links_added=links_added,
            frictions=frictions,
            briefing_path=briefing_path,
        )
