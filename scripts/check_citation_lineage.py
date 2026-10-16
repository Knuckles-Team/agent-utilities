#!/usr/bin/env python3
"""Citation-lineage gate (universal-ingestion program, tracks 8/9).

Proves one of this program's non-negotiable contracts with a tiny, fully
synthetic, checked-in fixture (no network, no live KG — mirrors
a hermetic-fixture style):

1. **Mandatory evidence citation** (CONCEPT:AU-KG.retrieval.mandatory-evidence-citation):
   every retrieval chunk built from the golden document resolves to a REAL,
   addressable evidence-spine Fragment — not just a non-empty id, but one that
   is actually present in the document's own fragment set and whose address
   matches the golden expectation below.
   (The embedding-version refusal moved to EG with the capability index:
   EG-DECISION-ENGINE-R172.6.)

The golden document + its expected fragment addresses are the "golden set":
small, real markdown, checked directly into this file so it never rots
silently out of sync with a fixture nobody runs.

Usage::

    python3 scripts/check_citation_lineage.py [--degrade]

``--degrade`` breaks each check on purpose (used by the meta-test proving the
gate has teeth): it corrupts a chunk's char span so it can no longer overlap
any real fragment.

Exit 0 = both checks pass. 1 = a regression. 2 = build error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# The golden document — small, real, checked in. Has a heading, two
# paragraphs, and a table, so fragment_markdown produces every FRAGMENT_KIND
# this gate needs to prove citation against.
GOLDEN_DOC = (
    "# Refund Policy\n\n"
    "Customers may request a refund within 30 days of purchase.\n\n"
    "## Exceptions\n\n"
    "Digital downloads are non-refundable once accessed.\n\n"
    "| Product Type | Refund Window |\n"
    "|---------------|---------------|\n"
    "| Physical | 30 days |\n"
    "| Digital  | 7 days  |\n"
)

# The golden expectation: every one of these addresses must exist in the
# document's own fragment set (i.e. fragment_markdown must actually produce
# them). This is what makes the fixture "real" rather than an assumption.
GOLDEN_FRAGMENT_ADDRESSES = {
    "heading:refund-policy",
    "heading:refund-policy/paragraph:0",
    "heading:refund-policy/heading:exceptions",
    "heading:refund-policy/heading:exceptions/paragraph:0",
    "heading:refund-policy/heading:exceptions/table:product-type-refund-window",
}


def _check_citation_resolution(*, degrade: bool) -> tuple[bool, str]:
    from agent_utilities.knowledge_graph.ingestion.evidence_spine import (
        artifact_id_for,
        fragment_markdown,
    )
    from agent_utilities.knowledge_graph.ontology.document_processing import (
        ChunkingConfig,
        _fragment_ids_for_span,
        chunk_text,
    )

    artifact_id = artifact_id_for("check", "golden", "refund-policy")
    fragments = fragment_markdown(GOLDEN_DOC, artifact_id=artifact_id)
    fragment_addresses = {f.address for f in fragments}
    missing_golden = GOLDEN_FRAGMENT_ADDRESSES - fragment_addresses
    if missing_golden:
        return False, (
            f"golden fragment address(es) not produced by fragment_markdown: "
            f"{sorted(missing_golden)} (fixture is stale or the fragmenter regressed)"
        )

    spans = chunk_text(GOLDEN_DOC, ChunkingConfig(chunk_size=90, overlap=15))
    if not spans:
        return False, "chunk_text produced no chunks for the golden document"

    real_fragment_ids = {f.fragment_id for f in fragments}
    uncited = []
    for sp in spans:
        char_start, char_end = sp.char_start, sp.char_end
        if degrade:
            # Shift the span past the end of the document so it cannot
            # overlap ANY real fragment — proves the gate actually checks
            # citation, rather than trivially passing on any non-empty list.
            char_start = char_end = len(GOLDEN_DOC) + 1_000
        cited = _fragment_ids_for_span(char_start, char_end, fragments)
        if not cited:
            uncited.append(sp.index)
            continue
        unresolvable = [fid for fid in cited if fid not in real_fragment_ids]
        if unresolvable:
            return (
                False,
                f"chunk {sp.index} cites non-existent fragment(s) {unresolvable}",
            )
    if uncited:
        return False, f"chunk(s) {uncited} have no fragment citation — not citable"
    return True, f"{len(spans)} chunk(s), all cite >=1 resolvable fragment"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrade", action="store_true")
    args = ap.parse_args()

    try:
        citation_ok, citation_msg = _check_citation_resolution(degrade=args.degrade)
    except Exception as exc:  # noqa: BLE001 — reported as a build error, not a silent pass
        print(f"ERROR: citation-lineage gate build failed: {exc}", file=sys.stderr)
        return 2

    print(f"[citation-resolution] {'OK' if citation_ok else 'FAIL'}: {citation_msg}")

    if not citation_ok:
        print("FAIL: citation-lineage gate regression.", file=sys.stderr)
        return 1
    print("OK: citation-lineage gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
