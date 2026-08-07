"""Meta-tests for the concept-governance gate (``scripts/check_concept_governance.py``).

Two things are proven here:

1. ``has_design_doc``/``all_registered_concepts`` behave correctly in
   isolation (a gate that can't fail is not a gate).
2. The NEW ``--audit-merged`` mode (D-RG2-3,
   CONCEPT:AU-OS.governance.merged-concept-visibility-audit) actually sees
   debt that the pre-existing diff-based mode is structurally blind to once a
   lane has merged — proving the fix for the exact hole reconciliation gate 2
   found: "the gate stops asking once a lane is merged". This is exercised
   entirely against a throwaway tmp tree/design-corpus/baseline (never the
   real repo's), so it proves the *mechanism*, not a snapshot of today's
   concept count.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_concept_governance.py"

sys.path.insert(0, str(ROOT))
from scripts.check_concept_governance import (  # noqa: E402
    all_registered_concepts,
    audit_merged,
    has_design_doc,
    read_baseline,
    write_baseline,
)


def _write_markers(root: Path, *, path: str, ids: list[str]) -> None:
    """Write a source file at ``root/path`` carrying one CONCEPT: marker per id."""
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(f"# CONCEPT:{cid}" for cid in ids), encoding="utf-8")


def test_has_design_doc_reads_the_design_corpus(tmp_path):
    design_dir = tmp_path / "design"
    design_dir.mkdir()
    (design_dir / "feature").mkdir(parents=True)
    # Built via concatenation, never as a literal "CONCEPT:AU-..." substring
    # in THIS file's own source — otherwise this test file's fake demo id
    # would itself be swept up as a real marker by the very scanner it tests.
    (design_dir / "feature" / "design.md").write_text(
        "covers " + "CONCEPT:" + "AU-KG.demo.covered", encoding="utf-8"
    )
    assert has_design_doc("AU-KG.demo.covered", design_dir=design_dir)
    assert not has_design_doc("AU-KG.demo.not-covered", design_dir=design_dir)


def test_all_registered_concepts_scans_the_whole_tree_not_just_one_package(tmp_path):
    """Regression for the exact gap this mode had to avoid: a marker written
    in a script/test/doc file (not under a package source root) must still be
    discovered — 5 of the 39 concepts D-RG2-2 found lived ONLY in such files
    (scripts/, mcp_v2_gateway/, tests/, AGENTS.md) and would have been
    silently exempted by a package-scoped scan."""
    _write_markers(tmp_path, path="agent_utilities/core/thing.py", ids=["AU-KG.demo.a"])
    _write_markers(tmp_path, path="scripts/some_tool.py", ids=["AU-KG.demo.b"])
    _write_markers(tmp_path, path="AGENTS.md", ids=["AU-KG.demo.c"])
    assert all_registered_concepts(tmp_path) == [
        "AU-KG.demo.a",
        "AU-KG.demo.b",
        "AU-KG.demo.c",
    ]


def test_all_registered_concepts_skips_vcs_and_cache_dirs(tmp_path):
    _write_markers(tmp_path, path=".git/objects/pack/whatever.py", ids=["AU-KG.demo.ghost"])
    _write_markers(tmp_path, path="src/real.py", ids=["AU-KG.demo.real"])
    assert all_registered_concepts(tmp_path) == ["AU-KG.demo.real"]


def test_audit_merged_sees_debt_the_diff_based_gate_cannot(tmp_path):
    """The core regression this whole lane exists to fix.

    Simulate the exact D-RG2-2/D-RG2-3 scenario: a concept with no design doc
    that has ALREADY LANDED (there is no diff, no base — it is simply part of
    the tree). The old diff-based ``new_concepts(base)`` path has nothing to
    compare against once it's merged and would silently report "no new
    concepts". ``--audit-merged`` has no such blind spot: it scans the live
    tree directly, so it fails on the undocumented, unbaselined concept
    regardless of merge history.
    """
    scan_root = tmp_path / "repo"
    design_dir = tmp_path / "design"
    design_dir.mkdir()
    baseline = tmp_path / "baseline.txt"

    _write_markers(
        scan_root,
        path="agent_utilities/feature.py",
        ids=["AU-KG.demo.already-documented", "AU-KG.demo.merged-with-no-doc"],
    )
    (design_dir / "feature.md").write_text(
        "CONCEPT:" + "AU-KG.demo.already-documented", encoding="utf-8"
    )

    # No baseline yet -> the undocumented, merged concept is visible and fails.
    rc = audit_merged(
        update=False,
        scan_root=scan_root,
        design_dir=design_dir,
        baseline_path=baseline,
    )
    assert rc == 1


def test_audit_merged_does_not_relitigate_baselined_debt(tmp_path):
    """A baselined (already-known/accepted) gap does not fail the gate — only
    NEW gaps do. This is what makes the ratchet adoptable without instantly
    redlining every pre-existing undocumented concept in the repo."""
    scan_root = tmp_path / "repo"
    design_dir = tmp_path / "design"
    design_dir.mkdir()
    baseline = tmp_path / "baseline.txt"

    _write_markers(scan_root, path="agent_utilities/x.py", ids=["AU-KG.demo.known-debt"])
    write_baseline({"AU-KG.demo.known-debt"}, baseline)

    rc = audit_merged(
        update=False,
        scan_root=scan_root,
        design_dir=design_dir,
        baseline_path=baseline,
    )
    assert rc == 0


def test_audit_merged_distinguishes_retired_from_resolved_baseline_entries(tmp_path, capsys):
    """A baselined id that was RETIRED (marker deleted entirely) must be
    reported only as stale/no-longer-exists, never ALSO as "now documented" —
    those are different, mutually exclusive outcomes. Regression: `resolved`
    was computed as `baseline - undocumented`, which a retired id also
    satisfies (it's vacuously "not undocumented" because it doesn't exist),
    so a removed marker double-reported under both sections."""
    scan_root = tmp_path / "repo"
    design_dir = tmp_path / "design"
    design_dir.mkdir()
    baseline = tmp_path / "baseline.txt"

    # Baseline remembers two ids; only "still-here" is still a live marker —
    # "retired" has been removed from the tree entirely (no file references it).
    write_baseline({"AU-KG.demo.still-here", "AU-KG.demo.retired"}, baseline)
    _write_markers(scan_root, path="agent_utilities/x.py", ids=["AU-KG.demo.still-here"])
    (design_dir / "feature.md").write_text(
        "CONCEPT:" + "AU-KG.demo.still-here", encoding="utf-8"
    )

    rc = audit_merged(
        update=False, scan_root=scan_root, design_dir=design_dir, baseline_path=baseline
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "AU-KG.demo.still-here" in out  # resolved: still exists, now documented
    assert "AU-KG.demo.retired" not in out.split("no longer exist")[0]  # never "resolved"
    assert "AU-KG.demo.retired" in out  # reported once, under stale only


def test_audit_merged_still_fails_on_a_NEW_gap_alongside_baselined_debt(tmp_path):
    """The precise regression-recurrence case: one concept is already-accepted
    debt (baselined), a SECOND concept lands merged with no doc. The second one
    must fail even though the first is silently tolerated -- proving the
    ratchet can never be satisfied by baselining a violation away instead of
    fixing/retiring it, and that new debt cannot hide behind old debt."""
    scan_root = tmp_path / "repo"
    design_dir = tmp_path / "design"
    design_dir.mkdir()
    baseline = tmp_path / "baseline.txt"

    _write_markers(
        scan_root,
        path="agent_utilities/x.py",
        ids=["AU-KG.demo.known-debt", "AU-KG.demo.newly-merged-gap"],
    )
    write_baseline({"AU-KG.demo.known-debt"}, baseline)

    rc = audit_merged(
        update=False,
        scan_root=scan_root,
        design_dir=design_dir,
        baseline_path=baseline,
    )
    assert rc == 1


def test_update_baseline_freezes_the_current_undocumented_set(tmp_path):
    scan_root = tmp_path / "repo"
    design_dir = tmp_path / "design"
    design_dir.mkdir()
    baseline = tmp_path / "baseline.txt"
    _write_markers(scan_root, path="agent_utilities/x.py", ids=["AU-KG.demo.a", "AU-KG.demo.b"])

    rc = audit_merged(
        update=True, scan_root=scan_root, design_dir=design_dir, baseline_path=baseline
    )
    assert rc == 0
    assert read_baseline(baseline) == {"AU-KG.demo.a", "AU-KG.demo.b"}

    # A subsequent plain run is now green against that frozen baseline.
    rc = audit_merged(
        update=False, scan_root=scan_root, design_dir=design_dir, baseline_path=baseline
    )
    assert rc == 0


def test_cli_audit_merged_flag_is_wired():
    """Wiring proof: the CLI flag actually reaches the base-less code path,
    not just the importable function. Runs against the REAL repo tree/
    baseline here (no override flags on the CLI), so we only assert it
    executed the audit-merged code path (distinct banner text), not a
    specific pass/fail outcome."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--audit-merged"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert "Merged-concept audit:" in result.stdout, result.stdout


def test_update_baseline_without_audit_merged_is_rejected():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--update-baseline"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "only applies with --audit-merged" in result.stderr


# ──────────────────────────────────────────────────────────────────────────────
# Parent-satisfied documentation (AU-OS.governance.concept-lineage-parent-doc)
#
# The mechanism's whole risk is that a pointer can HIDE a real decision, so the
# tests below are weighted toward the ways it could lie: a parent with no doc, a
# parent that does not exist, a chain, and a rationale that only restates the id.
# ──────────────────────────────────────────────────────────────────────────────

import pytest  # noqa: E402

from agent_utilities.governance.concept_lineage import (  # noqa: E402
    LineageError,
    parse_lineage,
)
from scripts.check_concept_governance import (  # noqa: E402
    reintroduced_retirements,
    resolve_documentation,
)

_GOOD_RATIONALE = (
    "one of nine per-connector entity-type declarations realising the single "
    "declarative source mapping decision written up under the parent"
)


def _lineage(parents=None, retired=None):
    return parse_lineage({"parents": parents or {}, "retired": retired or {}})


def _design(tmp_path, **docs):
    design_dir = tmp_path / "design"
    design_dir.mkdir(exist_ok=True)
    for name, ids in docs.items():
        (design_dir / f"{name}.md").write_text(
            "\n".join("CONCEPT:" + cid for cid in ids), encoding="utf-8"
        )
    return design_dir


def test_a_declared_parent_makes_a_child_documented(tmp_path):
    design_dir = _design(tmp_path, decision=["AU-KG.demo.the-decision"])
    lineage = _lineage(
        {"AU-KG.demo.a-marker": {"parent": "AU-KG.demo.the-decision", "rationale": _GOOD_RATIONALE}}
    )
    live = frozenset({"AU-KG.demo.a-marker", "AU-KG.demo.the-decision"})

    documented, broken = resolve_documentation(
        "AU-KG.demo.a-marker", lineage=lineage, design_dir=design_dir, live_ids=live
    )
    assert documented and broken is None


def test_a_parent_without_its_own_doc_is_a_violation_not_a_pass(tmp_path):
    """The failure mode that would make this feature harmful: pointing a child
    at a parent nobody ever documented reads as coverage while covering
    nothing. It must be reported BY NAME, never silently accepted."""
    design_dir = _design(tmp_path)
    lineage = _lineage(
        {"AU-KG.demo.a-marker": {"parent": "AU-KG.demo.undocumented", "rationale": _GOOD_RATIONALE}}
    )
    live = frozenset({"AU-KG.demo.a-marker", "AU-KG.demo.undocumented"})

    documented, broken = resolve_documentation(
        "AU-KG.demo.a-marker", lineage=lineage, design_dir=design_dir, live_ids=live
    )
    assert not documented
    assert broken and "NO design document" in broken


def test_a_parent_that_is_not_a_live_concept_is_a_violation(tmp_path):
    design_dir = _design(tmp_path, decision=["AU-KG.demo.ghost"])
    lineage = _lineage(
        {"AU-KG.demo.a-marker": {"parent": "AU-KG.demo.ghost", "rationale": _GOOD_RATIONALE}}
    )
    documented, broken = resolve_documentation(
        "AU-KG.demo.a-marker",
        lineage=lineage,
        design_dir=design_dir,
        live_ids=frozenset({"AU-KG.demo.a-marker"}),
    )
    assert not documented
    assert broken and "not a live concept" in broken


def test_parent_chains_are_rejected_at_load_time():
    """A -> B -> C reads as documented at every step while only C is checked.
    One hop keeps 'is this documented?' answerable without a traversal."""
    with pytest.raises(LineageError, match="chains are not allowed"):
        _lineage(
            {
                "AU-KG.demo.a": {"parent": "AU-KG.demo.b", "rationale": _GOOD_RATIONALE},
                "AU-KG.demo.b": {"parent": "AU-KG.demo.c", "rationale": _GOOD_RATIONALE},
            }
        )


def test_a_rationale_that_restates_the_id_is_rejected():
    """The anti-filler rule, applied to pointers: a doc that restates its id is
    already forbidden, and a pointer that restates its id is the same lie in one
    line instead of one page."""
    with pytest.raises(LineageError, match="restates the concept id"):
        _lineage(
            {
                "AU-KG.demo.entropy-dedup": {
                    "parent": "AU-KG.demo.the-decision",
                    "rationale": "entropy dedup — the demo decision",
                }
            }
        )


def test_self_parent_is_rejected():
    with pytest.raises(LineageError, match="cannot be its own parent"):
        _lineage({"AU-KG.demo.a": {"parent": "AU-KG.demo.a", "rationale": _GOOD_RATIONALE}})


def test_a_concept_cannot_be_both_retired_and_linked():
    with pytest.raises(LineageError, match="both retired and used in a parent link"):
        _lineage(
            parents={"AU-KG.demo.a": {"parent": "AU-KG.demo.b", "rationale": _GOOD_RATIONALE}},
            retired={"AU-KG.demo.b": {"reason": "never named a decision"}},
        )


def test_a_retirement_needs_a_reason():
    with pytest.raises(LineageError, match="requires a reason"):
        _lineage(retired={"AU-KG.demo.a": {}})


def test_reintroducing_a_retired_id_is_detected():
    """Retirement is a ratchet, not a deletion: the id coming back must fail."""
    lineage = _lineage(retired={"AU-KG.demo.gone": {"reason": "a slugified prose fragment"}})
    assert reintroduced_retirements(frozenset({"AU-KG.demo.gone"}), lineage) == [
        ("AU-KG.demo.gone", "a slugified prose fragment")
    ]
    assert reintroduced_retirements(frozenset({"AU-KG.demo.other"}), lineage) == []


def test_audit_merged_counts_a_parent_linked_concept_as_documented(tmp_path):
    """End-to-end through the gate itself, not just the resolver: a marker with
    no doc of its own and no baseline entry FAILS, and passes once — and only
    once — a parent link to a documented decision is declared."""
    scan_root = tmp_path / "repo"
    baseline = tmp_path / "baseline.txt"
    design_dir = _design(tmp_path, decision=["AU-KG.demo.the-decision"])
    _write_markers(
        scan_root,
        path="agent_utilities/x.py",
        ids=["AU-KG.demo.the-decision", "AU-KG.demo.a-marker"],
    )
    # Two distinct files, not one file rewritten: `load_lineage` is lru_cached
    # by path (matching `load_domain_vocab`/`load_slug_registry`), and the gate
    # is a one-shot process, so re-reading a mutated path is a scenario that
    # never occurs in production and would only be testing the cache.
    empty = tmp_path / "lineage-empty.yaml"
    empty.write_text("parents: {}\nretired: {}\n", encoding="utf-8")
    linked = tmp_path / "lineage-linked.yaml"
    linked.write_text(
        "parents:\n"
        "  AU-KG.demo.a-marker:\n"
        "    parent: AU-KG.demo.the-decision\n"
        f"    rationale: {_GOOD_RATIONALE}\n"
        "retired: {}\n",
        encoding="utf-8",
    )

    assert (
        audit_merged(
            update=False,
            scan_root=scan_root,
            design_dir=design_dir,
            baseline_path=baseline,
            lineage_path=str(empty),
        )
        == 1
    )
    assert (
        audit_merged(
            update=False,
            scan_root=scan_root,
            design_dir=design_dir,
            baseline_path=baseline,
            lineage_path=str(linked),
        )
        == 0
    )


# ──────────────────────────────────────────────────────────────────────────────
# Rename (D-CC-1) — the fourth disposition: a marker names a real decision but
# needs a different id (most often because its domain is not in the closed
# vocab, the live AU-KG.trace -> AU-KG.identity precedent this mechanism
# formalises). Tests weighted the same way as retirement's: the risk is a
# silently-revived old name, a chain nobody can resolve in one hop, or a
# collision with an id that already means something else.
# ──────────────────────────────────────────────────────────────────────────────

from scripts.check_concept_governance import reintroduced_renames  # noqa: E402


def test_a_rename_needs_a_reason():
    with pytest.raises(LineageError, match="requires a reason"):
        parse_lineage({"renamed": {"AU-KG.demo.a": {"to": "AU-KG.demo.b"}}})


def test_self_rename_is_rejected():
    with pytest.raises(LineageError, match="cannot be renamed to itself"):
        parse_lineage(
            {"renamed": {"AU-KG.demo.a": {"to": "AU-KG.demo.a", "reason": "x"}}}
        )


def test_rename_chains_are_rejected_at_load_time():
    """A -> B -> C: the same one-hop reasoning as parent chains. Whoever renames
    B again must flatten A's entry to point at C directly, not append a hop."""
    with pytest.raises(LineageError, match="rename chains are not allowed"):
        parse_lineage(
            {
                "renamed": {
                    "AU-KG.demo.a": {"to": "AU-KG.demo.b", "reason": "domain fix"},
                    "AU-KG.demo.b": {"to": "AU-KG.demo.c", "reason": "domain fix 2"},
                }
            }
        )


def test_a_concept_cannot_be_both_renamed_and_retired():
    with pytest.raises(LineageError, match="renamed .* and retired"):
        parse_lineage(
            {
                "renamed": {"AU-KG.demo.a": {"to": "AU-KG.demo.b", "reason": "x"}},
                "retired": {"AU-KG.demo.a": {"reason": "y"}},
            }
        )


def test_a_concept_cannot_be_both_renamed_and_used_in_a_parent_link():
    with pytest.raises(LineageError, match="renamed .* and retired"):
        parse_lineage(
            {
                "renamed": {"AU-KG.demo.a": {"to": "AU-KG.demo.b", "reason": "x"}},
                "parents": {
                    "AU-KG.demo.a": {
                        "parent": "AU-KG.demo.z",
                        "rationale": _GOOD_RATIONALE,
                    }
                },
            }
        )


def test_resolve_follows_a_rename_and_is_a_passthrough_otherwise():
    lineage = parse_lineage(
        {
            "renamed": {
                "AU-KG.trace.canonical-id-non-idempotence": {
                    "to": "AU-KG.identity.canonical-id-non-idempotence",
                    "reason": "trace was never a registered KG domain",
                }
            }
        }
    )
    assert (
        lineage.resolve("AU-KG.trace.canonical-id-non-idempotence")
        == "AU-KG.identity.canonical-id-non-idempotence"
    )
    # Passthrough: an id that was never renamed resolves to itself.
    assert lineage.resolve("AU-KG.demo.untouched") == "AU-KG.demo.untouched"


def test_reintroducing_a_renamed_old_id_is_detected():
    """Rename is a ratchet too: the OLD id coming back must fail, mirroring
    `reintroduced_retirements` — the decision moved, it did not vanish, so
    reviving the old name is still a regression, just a different one."""
    lineage = parse_lineage(
        {
            "renamed": {
                "AU-KG.demo.old-name": {
                    "to": "AU-KG.demo.new-name",
                    "reason": "domain word retired from the closed vocab",
                }
            }
        }
    )
    assert reintroduced_renames(frozenset({"AU-KG.demo.old-name"}), lineage) == [
        ("AU-KG.demo.old-name", "AU-KG.demo.new-name", "domain word retired from the closed vocab")
    ]
    assert reintroduced_renames(frozenset({"AU-KG.demo.new-name"}), lineage) == []


def test_audit_merged_fails_when_a_renamed_old_id_has_a_live_marker_again(tmp_path):
    """End-to-end through the gate: a live marker under an id that was
    deliberately renamed away must fail the audit, same as a revived
    retirement — never a silent pass."""
    scan_root = tmp_path / "repo"
    design_dir = tmp_path / "design"
    design_dir.mkdir()
    baseline = tmp_path / "baseline.txt"
    lineage_path = tmp_path / "lineage.yaml"
    lineage_path.write_text(
        "parents: {}\nretired: {}\n"
        "renamed:\n"
        "  AU-KG.demo.old-name:\n"
        "    to: AU-KG.demo.new-name\n"
        "    reason: domain word retired from the closed vocab\n",
        encoding="utf-8",
    )
    # The old marker is back in the tree, which the rename ratchet forbids.
    _write_markers(
        scan_root, path="agent_utilities/x.py", ids=["AU-KG.demo.old-name"]
    )

    rc = audit_merged(
        update=False,
        scan_root=scan_root,
        design_dir=design_dir,
        baseline_path=baseline,
        lineage_path=str(lineage_path),
    )
    assert rc == 1
