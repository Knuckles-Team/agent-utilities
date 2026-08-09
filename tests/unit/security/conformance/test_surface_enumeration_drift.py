"""Drift test for the GOC-62 conformance suite's enumeration mechanism (D2/D3(b)).

CONCEPT:AU-OS.identity.stack-wide-auth-conformance

This is the test that makes "a new surface is covered by default" real rather
than aspirational: it re-derives the live set of `graph_query` dialects by
parsing the ACTUAL source (never a hand-maintained list) and asserts every one
has a reviewed disposition in `surface_manifest.py`. A dialect added later with
no manifest entry makes THIS test fail — the surface is never silently
uncovered.
"""

from __future__ import annotations

from agent_utilities.security.conformance import surface_inventory, surface_manifest


def test_every_live_dialect_has_a_manifest_disposition() -> None:
    """The core drift assertion. If this fails, someone added a new
    `graph_query` dialect (a new `if scope == "..."` branch) without adding a
    reviewed `SurfaceEntry` for it — exactly the "new surface silently
    omitted" failure mode this suite exists to prevent."""

    live_surfaces = surface_inventory.enumerate_query_dialect_surfaces()
    live_names = {s.name for s in live_surfaces}
    manifest_names = {
        entry.surface_id.removeprefix("query_dialect:")
        for entry in surface_manifest.QUERY_DIALECT_MANIFEST
    }

    missing = live_names - manifest_names
    assert not missing, (
        f"graph_query dialect(s) {sorted(missing)} exist in "
        "mcp/tools/query_tools.py but have NO entry in "
        "surface_manifest.QUERY_DIALECT_MANIFEST — a new surface must never "
        "ship uncovered. Add a reviewed SurfaceEntry (disposition + citation) "
        "for each before merging."
    )


def test_manifest_has_no_stale_entries_for_dialects_that_no_longer_exist() -> None:
    """The reverse check — informational, not a hard failure mode this suite
    treats as urgent (legitimately removed code should not block CI), but
    still asserted so the manifest is kept honest rather than accreting dead
    rows silently."""

    live_names = {
        s.name for s in surface_inventory.enumerate_query_dialect_surfaces()
    }
    manifest_names = {
        entry.surface_id.removeprefix("query_dialect:")
        for entry in surface_manifest.QUERY_DIALECT_MANIFEST
    }
    stale = manifest_names - live_names
    assert not stale, (
        f"surface_manifest.QUERY_DIALECT_MANIFEST names dialect(s) {sorted(stale)} "
        "that no longer exist in query_tools.py — remove the stale entry (or "
        "the introspector needs updating if the dialect moved rather than "
        "disappeared)."
    )


def test_enumeration_finds_the_four_known_dialects() -> None:
    """A concrete regression pin: today's real dialect set is exactly
    {local, sql, sparql, federated}. This is not the drift mechanism itself
    (the two tests above are) — it exists so a change to query_tools.py that
    silently drops a dialect (rather than adding one) is also caught."""

    live_names = {
        s.name for s in surface_inventory.enumerate_query_dialect_surfaces()
    }
    assert live_names == {"local", "sql", "sparql", "federated"}


def test_enumerator_fails_loudly_on_a_missing_module() -> None:
    """Known-bad-input proof: point the introspector at a nonexistent path and
    confirm it raises rather than silently reporting zero surfaces (which
    would make the drift tests above vacuously pass — the exact "gate that
    reports more coverage than it has" failure mode this design avoids)."""

    import pytest

    from pathlib import Path

    with pytest.raises(FileNotFoundError):
        surface_inventory.enumerate_query_dialect_surfaces(
            module_path=Path("/nonexistent/does-not-exist/query_tools.py")
        )


def test_enumerator_fails_loudly_on_a_module_with_no_dialect_branches() -> None:
    """Known-bad-input proof: an empty/unrelated module must raise, not report
    zero surfaces silently."""

    import pytest
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        empty_module = Path(tmp) / "empty_query_tools.py"
        empty_module.write_text("def _run_graph_query():\n    pass\n")
        with pytest.raises(RuntimeError, match="zero"):
            surface_inventory.enumerate_query_dialect_surfaces(
                module_path=empty_module
            )
