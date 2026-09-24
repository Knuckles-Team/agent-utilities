"""EH-033: source schema mapping (source labels -> ontology classes) through EG ``Decide``.

Order, per DECIDE-LAYER-DESIGN §10: the deterministic crosswalk
(``map_labels_to_ontology``) maps first; every label it leaves unmapped is a
``schema_mapping`` question (policy safety: never explored) over the target
classes, scored on the class name against the source label (candidate-local
BM25). The model-backed semantic suggestion, when one was made, is the
fallback; otherwise the label stays unmapped for human approval. Either way
the result is only a PROPOSED profile -- operator approval (and the target
pack's SHACL shapes at import) remain the acceptance.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence

from agent_utilities import decide
from agent_utilities.decide.options import Option, text_param

UNMAPPED = "unmapped"
#: EG decides over at most 64 options; one is reserved for ``unmapped``.
MAX_TARGETS = 63


def _local_name(iri: str) -> str:
    return iri.rstrip("/#").rsplit("/", 1)[-1].rsplit("#", 1)[-1].rsplit(":", 1)[-1]


def map_label(
    label: str, targets: Sequence[str], suggestion: str | None
) -> decide.Choice:
    """One label's proposed target class (the semantic suggestion when EG does not decide)."""
    options = [Option(t, texts={"label": _local_name(t)}) for t in targets]
    options.append(Option(UNMAPPED, texts={"label": ""}))
    return decide.choose(
        "au.schema.mapping",
        options,
        lambda: suggestion or UNMAPPED,
        params=[text_param("column", label)],
    )


def apply_decided_mappings(
    labels: Sequence[str],
    targets: Sequence[str],
    suggestions: Mapping[str, str],
    type_map: MutableMapping[str, str],
    methods: MutableMapping[str, tuple[str, float]],
) -> None:
    """Fill labels the deterministic crosswalk left unmapped; record how each was mapped."""
    ordered = sorted(set(targets))
    if len(ordered) > MAX_TARGETS:
        return
    for label in labels:
        if label in type_map:
            continue
        choice = map_label(label, ordered, suggestions.get(label))
        if choice.option_id in (None, UNMAPPED) or choice.option_id not in ordered:
            continue
        type_map[label] = str(choice.option_id)
        methods[label] = (
            ("eg-decision", 0.8) if choice.decided else ("semantic-proposal", 0.6)
        )


__all__ = ["MAX_TARGETS", "UNMAPPED", "apply_decided_mappings", "map_label"]
