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
