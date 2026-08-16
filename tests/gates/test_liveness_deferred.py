"""Meta-test: the liveness ratchet's owner + review-by expiry
(``scripts/liveness_deferred.py``, enforced by ``scripts/check_liveness.py``).

GOC-68's named failure mode is "a bare ratchet lets deferral become
permanent" — this is the THIRD baseline found carrying that exact defect
(after ``epistemic-graph/tests/protocol_unbound_baseline.txt``, fixed
2026-08-13). This file proves BOTH directions on synthetic, in-memory text
(no file I/O, no real wall-clock wait — the same self-contained-proof shape
already used by ``epistemic-graph/tests/test_protocol_parity.py``'s
``test_review_date_gate_catches_stale_and_honors_fresh``):

1. A malformed entry (missing owner, or neither review-by nor a justified
   PERMANENT) fails ``test_baseline_entries_well_formed``-equivalent parsing.
2. A past-due ``review-by`` entry fails the gate; a fresh one does not; a
   ``PERMANENT`` entry never does, regardless of how old its line is.
3. The REAL ``scripts/liveness_deferred.tsv`` shipped in this change parses
   cleanly, is fully well-formed, and has no already-stale entry (so the
   proof above isn't purely synthetic — the shipped file is held to the same
   bar).
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "liveness_deferred.py"


def _load_liveness_deferred():
    spec = importlib.util.spec_from_file_location(
        "_liveness_deferred_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ld = _load_liveness_deferred()


_FIXTURE_TEXT = "\n".join(
    [
        "# a comment line is ignored",
        "orphan_modules\tagent_utilities/overdue.py\t# owner=@proof review-by=2026-08-01 note=one-day-overdue-at-as-of",
        "orphan_modules\tagent_utilities/fresh.py\t# owner=@proof review-by=2026-08-03 note=one-day-left-at-as-of",
        "orphan_modules\tagent_utilities/due_today.py\t# owner=@proof review-by=2026-08-02 note=due-today-not-yet-overdue",
        "dead_definitions\tagent_utilities/forever.py\t# owner=@proof PERMANENT reason=engine-internal-forever",
    ]
)
_AS_OF = date(2026, 8, 2)


def test_parser_finds_the_fixture_entries():
    entries = ld.parse_entries(_FIXTURE_TEXT)
    assert len(entries) == 4, "parser found the wrong number of entries"


def test_review_date_gate_catches_stale_and_honors_fresh():
    entries = ld.parse_entries(_FIXTURE_TEXT)
    stale = ld.stale_entries(entries, _AS_OF)
    stale_patterns = {e.pattern for e in stale}
    assert stale_patterns == {"agent_utilities/overdue.py"}, (
        f"expected exactly the overdue entry to be stale at {_AS_OF}, got {stale_patterns}"
    )


def test_well_formed_accepts_review_by_and_permanent_with_reason():
    entries = ld.parse_entries(_FIXTURE_TEXT)
    malformed = [e.pattern for e in entries if not ld.is_well_formed(e)]
    assert not malformed, f"fixture entries should all be well-formed: {malformed}"


def test_malformed_entries_are_rejected():
    text = "\n".join(
        [
            "orphan_modules\tagent_utilities/missing_owner.py\t# review-by=2099-01-01",
            "dead_definitions\tagent_utilities/permanent_no_reason.py\t# owner=@x PERMANENT",
            "dead_definitions\tagent_utilities/neither.py\t# owner=@x",
        ]
    )
    entries = ld.parse_entries(text)
    assert len(entries) == 3
    malformed = {e.pattern for e in entries if not ld.is_well_formed(e)}
    assert malformed == {
        "agent_utilities/missing_owner.py",
        "agent_utilities/permanent_no_reason.py",
        "agent_utilities/neither.py",
    }


def test_permanent_entry_never_goes_stale():
    entries = ld.parse_entries(_FIXTURE_TEXT)
    far_future = date(2099, 1, 1)
    stale = ld.stale_entries(entries, far_future)
    assert not any(e.category == "dead_definitions" and e.permanent for e in stale)


def test_unparsable_line_raises():
    import pytest

    with pytest.raises(ValueError):
        ld.parse_entries("this line has no tabs at all")


# ---------------------------------------------------------------------------
# The REAL shipped baseline is held to the same bar.
# ---------------------------------------------------------------------------


def test_real_liveness_deferred_tsv_is_well_formed_and_not_stale():
    entries = ld.load_entries()
    assert entries, "expected at least one real deferred entry"
    malformed = [e.pattern for e in entries if not ld.is_well_formed(e)]
    assert not malformed, (
        f"real liveness_deferred.tsv has malformed entries: {malformed}"
    )
    stale = ld.stale_entries(entries, date.today())
    assert not stale, f"real liveness_deferred.tsv has past-due entries: {stale}"
