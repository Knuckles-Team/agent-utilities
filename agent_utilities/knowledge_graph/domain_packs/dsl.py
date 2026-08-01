"""The declarative mapping DSL evaluator (CONCEPT:AU-KG.ingest.mapping-dsl).

Turns a :class:`~.domain_pack.DomainPackManifest`'s ``mappings`` + a stream of
:class:`~.fragment_contract.Fragment` objects into **proposed facts** —
plain, inert entity/relationship dicts in the exact shape
``ingestion.envelope_ingest.ingest_graph_slice`` already accepts. Nothing here
touches the graph: :func:`evaluate_mapping` is a pure function (fragments in,
facts out), and :mod:`.envelope_bridge` is the ONLY place a caller can choose
to actually submit the result through the governed ingest path.

**Inspectability is the point.** Every :class:`ProposedEntity`/
:class:`ProposedRelationship` carries the rule name(s) that produced it and
the fragment id(s) it came from — kept on the dataclass, stripped only by
``to_entity_dict``/``to_relationship_dict`` right before the plain dict form
reaches ``ingest_graph_slice`` (see :mod:`.envelope_bridge`) — so a dry run
(:func:`evaluate_mapping` itself, called without ever reaching the ingest
boundary) shows an operator exactly which rule produced which fact from which
fragment, before anything is written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .domain_pack import (
    DomainPackManifest,
    FrontmatterMapping,
    HeadingMapping,
    JSONFieldMapping,
    LinkMapping,
    TableMapping,
)
from .fragment_contract import ArtifactLike, FragmentLike

__all__ = [
    "MappingError",
    "ProposedEntity",
    "ProposedRelationship",
    "MappingResult",
    "evaluate_mapping",
]


class MappingError(RuntimeError):
    """A mapping rule could not be applied to a fragment (e.g. a template
    referenced a field the fragment doesn't carry). Raised, never swallowed —
    a template that silently resolves to a garbage id is worse than a loud
    failure at dry-run time."""


@dataclass(frozen=True)
class ProposedEntity:
    """One proposed node upsert — the entity-dict shape ``ingest_graph_slice`` accepts.

    Multiple mapping rules may all touch the same node id (e.g. one
    frontmatter rule sets a property, another attaches an edge FROM the same
    node) — :meth:`MappingResult.add_entity` merges these into ONE entity so
    ``ingest_graph_slice`` never sees a duplicate id (it requires auxiliary
    node ids to be unique). ``rules``/``fragment_ids`` accumulate every
    contributor so the merged fact stays fully traceable.
    """

    id: str
    node_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    rules: tuple[str, ...] = ()
    fragment_ids: tuple[str, ...] = ()

    def to_entity_dict(self) -> dict[str, Any]:
        return {"id": self.id, "node_type": self.node_type, **self.properties}


@dataclass(frozen=True)
class ProposedRelationship:
    """One proposed edge upsert — the relationship-dict shape ``ingest_graph_slice`` accepts."""

    source: str
    target: str
    relationship: str
    rule: str = ""
    fragment_id: str = ""

    def to_relationship_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship,
        }


@dataclass
class MappingResult:
    """The dry-run-safe output of evaluating a pack's mappings over fragments."""

    entities: list[ProposedEntity] = field(default_factory=list)
    relationships: list[ProposedRelationship] = field(default_factory=list)
    unmatched_fragment_ids: list[str] = field(default_factory=list)
    _index_by_id: dict[str, int] = field(
        default_factory=dict, repr=False, compare=False
    )

    def add_entity(
        self,
        *,
        id: str,  # noqa: A002 - matches the entity-dict key it becomes
        node_type: str,
        properties: dict[str, Any] | None = None,
        rule: str = "",
        fragment_id: str = "",
    ) -> None:
        """Upsert one proposed entity, merging by id (see :class:`ProposedEntity`)."""
        properties = dict(properties or {})
        existing_index = self._index_by_id.get(id)
        if existing_index is None:
            self._index_by_id[id] = len(self.entities)
            self.entities.append(
                ProposedEntity(
                    id=id,
                    node_type=node_type,
                    properties=properties,
                    rules=(rule,) if rule else (),
                    fragment_ids=(fragment_id,) if fragment_id else (),
                )
            )
            return
        existing = self.entities[existing_index]
        if existing.node_type != node_type:
            raise MappingError(
                f"conflicting node_type for id {id!r}: {existing.node_type!r} "
                f"(from rule(s) {existing.rules}) vs {node_type!r} (from rule {rule!r})"
            )
        self.entities[existing_index] = ProposedEntity(
            id=id,
            node_type=node_type,
            properties={**existing.properties, **properties},
            rules=existing.rules + ((rule,) if rule else ()),
            fragment_ids=existing.fragment_ids
            + ((fragment_id,) if fragment_id else ()),
        )

    def entity_dicts(self) -> list[dict[str, Any]]:
        return [e.to_entity_dict() for e in self.entities]

    def relationship_dicts(self) -> list[dict[str, Any]]:
        return [r.to_relationship_dict() for r in self.relationships]


def _render(template: str, context: dict[str, Any], *, rule: str) -> str:
    try:
        return template.format(**context)
    except (KeyError, IndexError) as exc:
        raise MappingError(
            f"rule {rule!r}: id/edge template {template!r} could not be "
            f"rendered from context keys {sorted(context)}"
        ) from exc


def _apply_frontmatter(
    rule: FrontmatterMapping,
    artifact: ArtifactLike,
    fragment: FragmentLike,
    result: MappingResult,
) -> bool:
    if fragment.fragment_type != "frontmatter_key":
        return False
    if fragment.metadata.get("key") != rule.key:
        return False
    context = {
        "artifact_id": artifact.artifact_id,
        "key": rule.key,
        "value": fragment.value,
    }
    node_id = _render(rule.id_template, context, rule=f"frontmatter:{rule.key}")
    if rule.produce == "property":
        result.add_entity(
            id=node_id,
            node_type=rule.node_type,
            properties={rule.property: fragment.value},
            rule=f"frontmatter:{rule.key}",
            fragment_id=fragment.fragment_id,
        )
    else:
        target_context = {**context}
        target_id = _render(
            rule.edge_target_id_template or "{value}",
            target_context,
            rule=f"frontmatter:{rule.key}",
        )
        result.add_entity(
            id=node_id, node_type=rule.node_type, rule=f"frontmatter:{rule.key}"
        )
        if rule.edge_target_type:
            result.add_entity(
                id=target_id,
                node_type=rule.edge_target_type,
                rule=f"frontmatter:{rule.key}",
                fragment_id=fragment.fragment_id,
            )
        result.relationships.append(
            ProposedRelationship(
                source=node_id,
                target=target_id,
                relationship=rule.relation,
                rule=f"frontmatter:{rule.key}",
                fragment_id=fragment.fragment_id,
            )
        )
    return True


def _apply_table(
    rule: TableMapping,
    artifact: ArtifactLike,
    fragment: FragmentLike,
    result: MappingResult,
) -> bool:
    if fragment.fragment_type != "table_row":
        return False
    if (
        rule.heading_path is not None
        and fragment.metadata.get("heading_path") != rule.heading_path
    ):
        return False
    row_index = fragment.metadata.get("row_index", 0)
    row = fragment.value if isinstance(fragment.value, dict) else {}
    context = {
        "artifact_id": artifact.artifact_id,
        "row_index": row_index,
    }
    row_id = _render(rule.row_id_template, context, rule="table")
    properties: dict[str, Any] = {}
    for column, mapping in rule.columns.items():
        if column not in row:
            continue
        cell_value = row[column]
        if mapping.produce == "property":
            properties[mapping.property] = cell_value
        else:
            target_context = {**context, "value": cell_value}
            target_id = _render(
                mapping.edge_target_id_template or "{value}",
                target_context,
                rule=f"table:{column}",
            )
            if mapping.edge_target_type:
                result.add_entity(
                    id=target_id,
                    node_type=mapping.edge_target_type,
                    rule=f"table:{column}",
                    fragment_id=fragment.fragment_id,
                )
            result.relationships.append(
                ProposedRelationship(
                    source=row_id,
                    target=target_id,
                    relationship=mapping.relation,
                    rule=f"table:{column}",
                    fragment_id=fragment.fragment_id,
                )
            )
    result.add_entity(
        id=row_id,
        node_type=rule.row_node_type,
        properties=properties,
        rule="table",
        fragment_id=fragment.fragment_id,
    )
    return True


def _apply_heading(
    rule: HeadingMapping,
    artifact: ArtifactLike,
    fragment: FragmentLike,
    result: MappingResult,
) -> bool:
    if fragment.fragment_type != "heading_section":
        return False
    heading_text = fragment.metadata.get("heading", "")
    if not re.search(rule.heading_pattern, heading_text):
        return False
    heading_path = fragment.metadata.get("heading_path", heading_text)
    context = {
        "artifact_id": artifact.artifact_id,
        "heading": heading_text,
        "heading_path": heading_path,
    }
    node_id = _render(rule.id_template, context, rule="heading")
    result.add_entity(
        id=node_id,
        node_type=rule.node_type,
        properties={"title": heading_text},
        rule="heading",
        fragment_id=fragment.fragment_id,
    )
    if rule.parent_relation:
        parent_id = _render(rule.parent_id_template, context, rule="heading")
        result.relationships.append(
            ProposedRelationship(
                source=parent_id,
                target=node_id,
                relationship=rule.parent_relation,
                rule="heading",
                fragment_id=fragment.fragment_id,
            )
        )
    return True


def _apply_link(
    rule: LinkMapping,
    artifact: ArtifactLike,
    fragment: FragmentLike,
    result: MappingResult,
) -> bool:
    if fragment.fragment_type != "link":
        return False
    href = fragment.metadata.get("href", "")
    if rule.href_pattern is not None and not re.search(rule.href_pattern, href):
        return False
    context = {"artifact_id": artifact.artifact_id, "href": href}
    source_id = _render(rule.source_id_template, context, rule="link")
    target_id = _render(rule.target_id_template, context, rule="link")
    if rule.target_node_type:
        result.add_entity(
            id=target_id,
            node_type=rule.target_node_type,
            rule="link",
            fragment_id=fragment.fragment_id,
        )
    result.relationships.append(
        ProposedRelationship(
            source=source_id,
            target=target_id,
            relationship=rule.relation,
            rule="link",
            fragment_id=fragment.fragment_id,
        )
    )
    return True


def _apply_json_field(
    rule: JSONFieldMapping,
    artifact: ArtifactLike,
    fragment: FragmentLike,
    result: MappingResult,
) -> bool:
    if fragment.fragment_type != "json_field":
        return False
    if fragment.locator != rule.path:
        return False
    context = {
        "artifact_id": artifact.artifact_id,
        "path": rule.path,
        "value": fragment.value,
    }
    node_id = _render(rule.id_template, context, rule=f"json_field:{rule.path}")
    if rule.produce == "property":
        result.add_entity(
            id=node_id,
            node_type=rule.node_type,
            properties={rule.property: fragment.value},
            rule=f"json_field:{rule.path}",
            fragment_id=fragment.fragment_id,
        )
    else:
        target_id = _render(
            rule.edge_target_id_template or "{value}",
            context,
            rule=f"json_field:{rule.path}",
        )
        result.add_entity(
            id=node_id, node_type=rule.node_type, rule=f"json_field:{rule.path}"
        )
        if rule.edge_target_type:
            result.add_entity(
                id=target_id,
                node_type=rule.edge_target_type,
                rule=f"json_field:{rule.path}",
                fragment_id=fragment.fragment_id,
            )
        result.relationships.append(
            ProposedRelationship(
                source=node_id,
                target=target_id,
                relationship=rule.relation,
                rule=f"json_field:{rule.path}",
                fragment_id=fragment.fragment_id,
            )
        )
    return True


_APPLY_BY_KIND = {
    "frontmatter": _apply_frontmatter,
    "table": _apply_table,
    "heading": _apply_heading,
    "link": _apply_link,
    "json_field": _apply_json_field,
}


def evaluate_mapping(
    pack: DomainPackManifest,
    artifact: ArtifactLike,
    fragments: list[FragmentLike],
) -> MappingResult:
    """Pure evaluation: fragments -> proposed facts. Never touches the graph.

    This IS the dry-run: calling this function and inspecting its
    ``MappingResult`` shows exactly the facts a pack would produce, with no
    write ever happening — writing is a distinct, explicit step
    (:mod:`.envelope_bridge`).
    """
    result = MappingResult()
    for fragment in fragments:
        matched = False
        for rule in pack.mappings:
            applier = _APPLY_BY_KIND[rule.kind]
            if applier(rule, artifact, fragment, result):
                matched = True
        if not matched:
            result.unmatched_fragment_ids.append(fragment.fragment_id)
    return result
