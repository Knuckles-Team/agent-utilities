#!/usr/bin/python
"""Unit tests for the night-shift second-brain swarm (CONCEPT:KG-2.84).

Deterministic and offline: uses ``tmp_path`` as the vault root and the default
(or an injected) paragraph splitter — no LLM, no network.
"""

from __future__ import annotations

from pathlib import Path

from agent_utilities.knowledge_graph.research.night_shift import (
    AtomNote,
    NightShiftSwarm,
    ShiftReport,
)


def _swarm(tmp_path: Path, **kw) -> NightShiftSwarm:
    return NightShiftSwarm(tmp_path, **kw)


def test_layout_created(tmp_path: Path) -> None:
    _swarm(tmp_path)
    for stage in (
        "0-raw",
        "sources",
        "2-atoms",
        "3-threads",
        "briefings",
        "2-atoms/archive",
    ):
        assert (tmp_path / stage).is_dir()


def test_scout_writes_sources_and_does_not_overwrite(tmp_path: Path) -> None:
    swarm = _swarm(tmp_path)
    n = swarm.scout([("paper-1", "original text"), ("paper-2", "second text")])
    assert n == 2
    src = tmp_path / "sources" / "paper-1.md"
    assert src.read_text(encoding="utf-8") == "original text"
    # A 0-raw pointer exists.
    assert (tmp_path / "0-raw" / "paper-1.intake.md").exists()

    # Re-scout the same id with different text: must NOT overwrite (immutability).
    again = swarm.scout([("paper-1", "TAMPERED")])
    assert again == 0
    assert src.read_text(encoding="utf-8") == "original text"


def test_catalog_one_atom_per_idea_with_source(tmp_path: Path) -> None:
    # Injected deterministic extractor → exactly two atoms.
    swarm = _swarm(
        tmp_path, extract_fn=lambda t: [s.strip() for s in t.split("|") if s.strip()]
    )
    swarm.scout([("src-a", "idea one | idea two")])
    atoms = swarm.catalog()
    assert len(atoms) == 2
    # Prime Directive: every atom traces to a source.
    for atom in atoms:
        assert atom.source_id == "src-a"
        assert atom.source_id  # non-empty
    # No atom file exists without a source_id in frontmatter.
    for path in (tmp_path / "2-atoms").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert "source: src-a" in text

    # Re-cataloging the same (already-catalogued) source produces nothing new.
    assert swarm.catalog() == []


def test_cartograph_links_min_two(tmp_path: Path) -> None:
    swarm = _swarm(
        tmp_path,
        extract_fn=lambda t: [s for s in t.split("|") if s.strip()],
        min_links=2,
    )
    swarm.scout(
        [("s", "alpha topic one|beta topic two|gamma topic three|delta topic four")]
    )
    atoms = swarm.catalog()
    swarm.cartograph(atoms)
    # Re-load each atom from disk and assert >= 2 links.
    reloaded = {a.atom_id: a for a in swarm._existing_atoms()}
    for atom in atoms:
        assert len(reloaded[atom.atom_id].links) >= 2


def test_critique_surfaces_friction_for_contradiction(tmp_path: Path) -> None:
    swarm = _swarm(tmp_path)
    # Existing belief: lithium cost is the binding constraint on EV adoption.
    swarm.scout(
        [("battery-econ-2024", "lithium cost is the binding constraint on EV adoption")]
    )
    swarm.catalog()
    swarm.cartograph(swarm._existing_atoms())

    # New source whose atom contradicts it (sodium-ion undercuts lithium).
    swarm.scout(
        [
            (
                "sodium-ion-2026",
                "sodium-ion undercuts lithium so lithium is not the constraint on EV adoption",
            )
        ]
    )
    new_atoms = swarm.catalog()
    frictions = swarm.critique(new_atoms)
    assert frictions, "expected a [FRICTION] for the contradicting pair"
    assert any("[FRICTION]" in f for f in frictions)
    # The friction points back at the conflicting belief.
    assert any("battery-econ-2024" in f for f in frictions)


def test_edit_writes_briefing_mentioning_contradiction(tmp_path: Path) -> None:
    swarm = _swarm(tmp_path)
    report = swarm.run_shift(
        [
            (
                "battery-econ-2024",
                "lithium cost is the binding constraint on EV adoption",
            ),
        ]
    )
    # Second shift introduces the contradiction.
    report2 = swarm.run_shift(
        [
            (
                "sodium-ion-2026",
                "sodium-ion undercuts lithium so lithium is not the constraint on EV adoption",
            ),
        ]
    )
    briefing = Path(report2.briefing_path)
    assert briefing.exists()
    text = briefing.read_text(encoding="utf-8")
    assert "Morning Briefing" in text
    assert "Contradictions To Resolve" in text
    assert "[FRICTION]" in text
    # First briefing also exists and is a distinct file.
    assert Path(report.briefing_path).exists()
    assert report.briefing_path != report2.briefing_path


def test_retire_moves_to_archive_never_deletes(tmp_path: Path) -> None:
    swarm = _swarm(tmp_path, extract_fn=lambda t: [t])
    swarm.scout([("doc", "an old idea worth retiring")])
    atoms = swarm.catalog()
    atom_id = atoms[0].atom_id

    live_path = tmp_path / "2-atoms" / f"{atom_id}.md"
    assert live_path.exists()

    assert swarm.retire(atom_id) is True
    # Gone from the live dir...
    assert not live_path.exists()
    # ...but still exists in the archive (never deleted).
    archived = tmp_path / "2-atoms" / "archive" / f"{atom_id}.md"
    assert archived.exists()
    text = archived.read_text(encoding="utf-8")
    assert "[RETIRED]" in text
    assert "retired: true" in text

    # Retiring a non-existent atom is a no-op False.
    assert swarm.retire("does-not-exist") is False


def test_run_shift_returns_populated_report(tmp_path: Path) -> None:
    swarm = _swarm(tmp_path)
    report = swarm.run_shift(
        [
            (
                "a",
                "Solar power is getting cheaper every year. Wind power also keeps falling in cost.",
            ),
            ("b", "Battery storage costs are declining rapidly alongside renewables."),
        ]
    )
    assert isinstance(report, ShiftReport)
    assert report.sources_ingested == 2
    assert report.atoms_created >= 3
    assert report.links_added >= 1
    assert Path(report.briefing_path).exists()


def test_sources_immutable_across_two_shifts(tmp_path: Path) -> None:
    swarm = _swarm(tmp_path)
    swarm.run_shift([("immut", "this source text must never change")])
    src = tmp_path / "sources" / "immut.md"
    before = src.read_bytes()

    # A second shift over the same vault (with a new source) must not touch it.
    swarm.run_shift([("other", "a different, second source")])
    after = src.read_bytes()
    assert before == after, "sources/ files must be byte-identical across shifts"


def test_atom_dataclass_defaults() -> None:
    atom = AtomNote(atom_id="x", claim="c", source_id="s")
    assert atom.links == []
    assert atom.frictions == []
    assert atom.retired is False
