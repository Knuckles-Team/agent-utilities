"""The known fail-opens, wired into the GOC-62 conformance suite (D3(b)).

CONCEPT:AU-OS.identity.stack-wide-auth-conformance

This file does NOT re-implement the behavioral proofs — duplicating them here
would be exactly the "second, independently-maintained copy of the same
decision" pattern this standard's §5 (nav-vs-backend drift) warns against.
Instead it asserts the conformance manifest and the real, executed proofs
agree:

1. **BUG-036 (AU federation reader) — FIXED.** Proof:
   `tests/unit/knowledge_graph/test_federation_carrier_authority.py`. Was RED
   (`DID NOT RAISE SessionRequiredError`) until `engine_federation.py` gained
   an explicit `resolve_session(required_scope="kg:read")` guard (matching the
   cypher/sql/sparql dialects) AND `ActorContextMiddleware` (the structural
   root cause -- an unconditional no-op with no token) gained a
   `require_verified_session` mode wired on for graph-os specifically
   (`mcp/middlewares.py`, `mcp/server_factory.py`). The manifest disposition
   is now `AUTHENTICATED_REQUIRED`, not `KNOWN_FAIL_OPEN` -- this test asserts
   that (not the reverse) so a REGRESSION (someone reverting the fix without
   reverting the manifest) is caught the same way a stale-manifest drift
   would have been caught before: a disposition that no longer matches the
   real, executed proof's behavior.

2. **BUG-037 (EG observability ingest)** — proof is new in this pass:
   `epistemic-graph/src/server/obs/mod.rs`'s
   `tests::bug_037_obs_ingest_post_bypasses_the_deny_gate`, committed on the
   `goc/goc-62-keycloak-auth-standard` branch in the `epistemic-graph`
   worktree (a Rust test — this Python suite cannot execute it directly, but
   asserts its existence in source so an accidental deletion is caught). Still
   open (owned by `OWNER-EG-OBS`, a different repository); unaffected by the
   BUG-036 fix above.
"""

from __future__ import annotations

from pathlib import Path

from agent_utilities.security.conformance.surface_manifest import (
    Disposition,
    lookup_query_dialect,
)

# Resolved relative to this file: tests/unit/security/conformance/ -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EG_SIBLING_WORKTREE = (
    Path.home()
    / ".local"
    / "state"
    / "repository-worktrees"
    / "epistemic-graph"
    / "goc-62-auth"
)


def test_federation_reader_is_disposed_as_authenticated_required_post_fix() -> None:
    """BUG-036 is fixed: the manifest must say so, not still claim fail-open.

    A stale `KNOWN_FAIL_OPEN` disposition after the code is fixed is exactly
    the drift D2 warns about ("a green run on a known_fail_open surface is
    itself a signal something is wrong ... the surface was silently fixed and
    the manifest is stale"). This asserts the manifest was updated alongside
    the fix, not left behind.
    """
    entry = lookup_query_dialect("query_dialect:federated")
    assert entry is not None
    assert entry.disposition is Disposition.AUTHENTICATED_REQUIRED
    assert entry.owning_bug == "BUG-036"
    assert entry.proof is not None


def test_federation_reader_proof_file_exists_and_names_the_right_assertion() -> None:
    """The manifest's citation for BUG-036 must point at a REAL file+test that
    actually exists — a citation to a deleted/renamed file would be worse than
    no citation (a false sense of coverage). This regression test must keep
    proving SessionRequiredError is raised even now that the bug is fixed."""

    entry = lookup_query_dialect("query_dialect:federated")
    assert entry is not None and entry.proof is not None
    proof_path_str, _, test_name = entry.proof.partition("::")
    proof_path = _REPO_ROOT / proof_path_str
    assert proof_path.is_file(), f"BUG-036 proof file missing: {proof_path}"
    source = proof_path.read_text(encoding="utf-8")
    assert test_name in source, (
        f"BUG-036 proof file {proof_path} no longer contains "
        f"{test_name!r} -- the manifest citation is stale"
    )
    assert "SessionRequiredError" in source, (
        "the BUG-036 proof must assert SessionRequiredError is raised -- "
        "the exact fail-open this manifest entry claimed is now closed, and "
        "must stay closed"
    )


def test_eg_obs_ingest_fail_open_proof_exists_in_source() -> None:
    """BUG-037's proof lives in the epistemic-graph repo (Rust), committed on
    this same GOC-62 branch in its own worktree. This Python suite cannot
    execute a Rust test, so it asserts the proof exists in source -- an
    accidental deletion of the test (in either repo's worktree) is caught
    here rather than discovered only when someone goes looking."""

    candidates = [
        _EG_SIBLING_WORKTREE / "src" / "server" / "obs" / "mod.rs",
        _REPO_ROOT.parent / "epistemic-graph" / "src" / "server" / "obs" / "mod.rs",
    ]
    obs_mod = next((c for c in candidates if c.is_file()), None)
    if obs_mod is None:
        import pytest

        pytest.skip(
            "epistemic-graph worktree not present at any known sibling path in "
            "this environment -- the Rust proof's existence cannot be checked "
            "from here; see epistemic-graph src/server/obs/mod.rs::tests::"
            "bug_037_obs_ingest_post_bypasses_the_deny_gate directly"
        )

    source = obs_mod.read_text(encoding="utf-8")
    assert "bug_037_obs_ingest_post_bypasses_the_deny_gate" in source
    assert "is_observability_read_carrier" in source
    # The proof must exercise BOTH a read (contrast case, correctly denied)
    # and an ingest POST (the actual bug) -- a test asserting only the ingest
    # side could be vacuously green if the whole harness were broken.
    assert '"403 Forbidden"' in source
