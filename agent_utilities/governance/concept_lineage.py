"""Concept lineage — a concept may declare that a *parent* concept documents it.

CONCEPT:AU-OS.governance.concept-lineage-parent-doc — parent-satisfied documentation.

Why this exists
---------------
``check_concept_governance.py --audit-merged`` asks one question of every live
``CONCEPT:<id>`` marker: does a design document under ``.specify/design/**``
mention it? On 2026-07-31 that audit found **1,039 live concepts with no design
doc**, clustered into just 80 ``pillar.domain`` groups. Read naively, that is a
demand for 1,039 design documents.

It is not. Auditing one whole group (``AU-KG.compute``, 79 live concepts — see
``.specify/triage/AU-KG.compute.yaml``) resolved it to **21 design documents,
47 pointers and 11 retirements**. The pointers are markers *on code that
realises an already-decided thing*: sixteen ``source_sync``/``owl_bridge``/
``mcp_tool`` per-connector markers all realising ONE decision about how a
connector's entity types reach the graph; four ``numeric/`` markers all
realising ONE decision to drop numpy/scipy for a compiled kernel. Writing
seventeen documents for the first cluster would mean writing sixteen
restatements — which is worse than none, because a restatement satisfies the
gate, looks like documentation, and teaches nothing.

So the unit of documentation is the **decision**, not the marker. A cohesive
cluster needs ONE document plus N pointers. This module is the pointer: a
machine-checked, hand-authored declaration that ``child`` is covered by
``parent``'s design document.

The registry also records **retirements**: markers that were removed because
they never named a decision at all (an id auto-derived from a prose fragment,
say ``AU-KG.compute.when-exposes`` minted from the sentence "when the engine
exposes the native..."). Recording them is what makes retirement a ratchet
rather than a deletion — re-introducing a deliberately retired id is a gate
failure, not a silent regression.

Rules the loader enforces
-------------------------
Every rule below exists because breaking it would let the mechanism *hide* a
real decision, which is the only way this feature can do harm:

* **One hop only.** A ``parent`` may not itself be a ``child``. Chains make
  "is this documented?" a graph traversal whose terminus nobody checks; a
  two-hop chain ending at an undocumented root reads as documented at every
  intermediate step.
* **No self-parent, no cycles.** Implied by the one-hop rule, checked anyway
  so a malformed edit fails loudly instead of resolving to nothing.
* **A rationale, and not a restatement of the id.** The pointer must say
  *which* decision covers the marker and *why*, in words that are not simply
  the concept's own slug re-spelled. A pointer that restates its id is the
  same filler the design-doc rule already forbids.
* **Disjoint dispositions.** A concept is retired *or* linked, never both.

The gate (``check_concept_governance.py``) consumes :func:`load_lineage` and
treats ``has_design_doc(child) or has_design_doc(parent_of(child))`` as
documented — but only after verifying the parent genuinely has a document. A
parent link pointing at an undocumented parent is a *violation*, reported by
name, not a silent pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from agent_utilities.governance.concept_hierarchy import parse_okf_id

#: The hand-authored lineage registry shipped alongside this module.
LINEAGE_PATH = Path(__file__).with_name("concept_lineage.yaml")

#: A rationale must carry this many words *beyond* the words already present in
#: the child and parent ids. Eight was chosen by reading the shortest honest
#: pointer written during the AU-KG.compute pilot ("one of twelve per-source
#: entity-type declarations under the single decision that a dedicated tracker
#: reaches its upstream only through its fleet MCP server") and leaving no
#: headroom below it.
MIN_RATIONALE_NOVEL_WORDS = 8


class LineageError(ValueError):
    """The lineage registry is malformed. Always raised with the offending id."""


@dataclass(frozen=True)
class ParentLink:
    """``child`` is documented by ``parent``'s design document."""

    child: str
    parent: str
    rationale: str


@dataclass(frozen=True)
class Retirement:
    """``concept``'s marker was removed; re-introducing the id is a failure."""

    concept: str
    reason: str


@dataclass(frozen=True)
class Lineage:
    """The validated registry: parent links keyed by child, plus retirements."""

    parents: dict[str, ParentLink]
    retired: dict[str, Retirement]

    def parent_of(self, concept: str) -> str | None:
        """The concept whose design document covers *concept*, if declared."""
        link = self.parents.get(concept)
        return link.parent if link else None

    @property
    def parent_concepts(self) -> frozenset[str]:
        """Every concept that is cited as a parent by at least one child."""
        return frozenset(link.parent for link in self.parents.values())

    def children_of(self, concept: str) -> tuple[str, ...]:
        """Every child pointing at *concept*, sorted."""
        return tuple(
            sorted(c for c, link in self.parents.items() if link.parent == concept)
        )


def _id_words(*concept_ids: str) -> set[str]:
    """Every lowercase word occurring in the given concept ids."""
    words: set[str] = set()
    for cid in concept_ids:
        for chunk in cid.replace("-", ".").replace("_", ".").split("."):
            if chunk:
                words.add(chunk.lower())
    return words


def _check_id(cid: str, *, field: str) -> None:
    try:
        parse_okf_id(cid)
    except ValueError as exc:
        raise LineageError(
            f"{field} {cid!r} is not a valid OKF-CIS concept id"
        ) from exc


def _check_rationale(link_child: str, link_parent: str, rationale: str) -> None:
    """A pointer must explain the coverage in words the ids do not already supply."""
    text = (rationale or "").strip()
    if not text:
        raise LineageError(f"{link_child!r}: a parent link requires a rationale")
    known = _id_words(link_child, link_parent)
    punctuation = ".,;:()[]{}'\"`—–…!?"
    novel = {
        word
        for word in (token.strip(punctuation).lower() for token in text.split())
        if word and word not in known
    }
    if len(novel) < MIN_RATIONALE_NOVEL_WORDS:
        raise LineageError(
            f"{link_child!r}: rationale restates the concept id — it adds only "
            f"{len(novel)} word(s) beyond the ids themselves, need at least "
            f"{MIN_RATIONALE_NOVEL_WORDS}. Say WHICH decision covers this marker "
            f"and why, not what the id already spells."
        )


def parse_lineage(data: dict) -> Lineage:
    """Validate a raw registry mapping into a :class:`Lineage`.

    Split from :func:`load_lineage` so the rules are testable without a file and
    so the triage tool can validate a candidate registry before writing it.
    """
    raw_parents = data.get("parents") or {}
    raw_retired = data.get("retired") or {}
    if not isinstance(raw_parents, dict):
        raise LineageError(
            "'parents' must be a mapping of child-id -> {parent, rationale}"
        )
    if not isinstance(raw_retired, dict):
        raise LineageError("'retired' must be a mapping of concept-id -> {reason}")

    parents: dict[str, ParentLink] = {}
    for child, entry in raw_parents.items():
        _check_id(child, field="parent-link child")
        if not isinstance(entry, dict):
            raise LineageError(
                f"{child!r}: entry must be a mapping with 'parent' and 'rationale'"
            )
        parent = str(entry.get("parent") or "").strip()
        _check_id(parent, field=f"parent of {child!r}")
        if parent == child:
            raise LineageError(f"{child!r}: a concept cannot be its own parent")
        _check_rationale(child, parent, str(entry.get("rationale") or ""))
        parents[child] = ParentLink(
            child=child, parent=parent, rationale=str(entry["rationale"]).strip()
        )

    # One hop only. Checked after the whole table is read so the error names the
    # actual chain rather than depending on mapping order.
    for child, link in parents.items():
        if link.parent in parents:
            raise LineageError(
                f"{child!r} -> {link.parent!r} -> {parents[link.parent].parent!r}: "
                "parent chains are not allowed. A parent must own a design document "
                "itself, so 'is this documented?' is answerable in one hop."
            )

    retired: dict[str, Retirement] = {}
    for concept, entry in raw_retired.items():
        _check_id(concept, field="retired concept")
        if not isinstance(entry, dict):
            raise LineageError(
                f"{concept!r}: retirement entry must be a mapping with 'reason'"
            )
        reason = str(entry.get("reason") or "").strip()
        if not reason:
            raise LineageError(f"{concept!r}: a retirement requires a reason")
        retired[concept] = Retirement(concept=concept, reason=reason)

    overlap = sorted(
        set(retired) & (set(parents) | {lk.parent for lk in parents.values()})
    )
    if overlap:
        raise LineageError(
            "concept(s) are both retired and used in a parent link: "
            + ", ".join(overlap)
            + " — a retired marker no longer exists, so it can neither be covered "
            "nor cover anything."
        )

    return Lineage(parents=parents, retired=retired)


@lru_cache(maxsize=8)
def load_lineage(path: str | None = None) -> Lineage:
    """Load and validate the lineage registry. Cached.

    Raises :class:`LineageError` on any malformed entry — the registry is a
    governance record, so a silent partial load would be worse than a crash.
    """
    import yaml

    src = Path(path) if path else LINEAGE_PATH
    if not src.exists():
        return Lineage(parents={}, retired={})
    try:
        data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise LineageError(f"{src} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise LineageError(f"{src} must contain a mapping at the top level")
    return parse_lineage(data)
