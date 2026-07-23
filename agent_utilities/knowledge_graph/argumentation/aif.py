"""Argument Interchange Format (AIF) — argument-map objects + graph bridge.

CONCEPT:AU-KG.epistemic.aif.

AIF (Rahwan & Reed, "The Argument Interchange Format"; AIFdb/arg-tech.org) is
the community interchange standard for representing an argument as a graph of
two node kinds:

* **I-nodes** — Information nodes: the propositional content (a claim,
  premise, datum) asserted somewhere in an argument.
* **S-nodes** — Scheme application nodes: the application of a named scheme
  linking I-nodes together. Three core kinds, plus two AIF+ dialogue
  extensions:

  * **RA** (Rule of inference / inference-scheme Application) — one or more
    premise I-nodes support a single conclusion I-node.
  * **CA** (Conflict-scheme Application) — one or more premise I-nodes
    conflict with a single conclusion I-node.
  * **PA** (Preference-scheme Application) — two or more compared
    alternatives, with the single conclusion marking the preferred one.
  * **TA** (Transition-scheme Application, AIF+) — a dialogue-state
    transition.
  * **YA** (Illocutionary-scheme Application, AIF+) — links a locution to the
    illocutionary act/proposition it performs.

Every edge in an AIF graph connects an I-node/S-node to an S-node (a
*premise* edge) or an S-node to a node (its *conclusion* edge) — see
``ontology_argumentation.ttl`` for the OWL encoding (``:aifHasPremise`` /
``:aifHasConclusion`` / ``:aifFulfills``).

**This module is the typed interchange vocabulary layered over the engine's
existing argumentation machinery — it does NOT duplicate it.** An
``:AIFInformationNode`` (I-node) IS a ``:Belief`` (the SAME Claim/Evidence/
BeliefState confidence machinery every other belief already uses).
:func:`import_argument_map` writes both the AIF-typed structure (for
lossless interchange fidelity) AND the underlying ``SUPPORTS``/``ATTACKS``
edges ``eg-epistemic``'s belief propagation and Dung argumentation
(``Method::ResolveConflict``) already read — see :func:`to_dung`, the pure
structural projection a caller hands to the engine's REAL grounded/preferred/
stable solver (no solver lives in this module; see the ``graph_argument``
MCP tool's ``evaluate`` action / ``mcp/tools/argument_tools.py``).

Reuses the ONE connector write/read path (``memory/native_ingest.py`` +
``ingestion/envelope_ingest.ingest_graph_slice``) rather than inventing a
parallel store.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AIF_NODE_TYPES",
    "SCHEME_NODE_TYPES",
    "NODE_TYPE_CLASS",
    "SCHEME_CLASS_BY_KIND",
    "AIFNode",
    "AIFEdge",
    "ArgumentMap",
    "DungProjection",
    "validate_argument_map",
    "from_aifdb_json",
    "to_aifdb_json",
    "to_dung",
    "import_argument_map",
    "export_argument_map",
    "add_scheme",
]

#: AIF Upper/Forms Ontology node-type codes -> the OWL class each becomes on
#: import (``ontology_argumentation.ttl``).
NODE_TYPE_CLASS: dict[str, str] = {
    "I": "AIFInformationNode",
    "RA": "AIFRuleApplicationNode",
    "CA": "AIFConflictApplicationNode",
    "PA": "AIFPreferenceApplicationNode",
    "TA": "AIFTransitionApplicationNode",
    "YA": "AIFIllocutionaryApplicationNode",
}

AIF_NODE_TYPES: frozenset[str] = frozenset(NODE_TYPE_CLASS)
#: The S-node (scheme application) subset of :data:`AIF_NODE_TYPES`.
SCHEME_NODE_TYPES: frozenset[str] = frozenset({"RA", "CA", "PA", "TA", "YA"})

#: ``add_scheme`` kind -> the AIF Forms-Ontology OWL class it mints.
SCHEME_CLASS_BY_KIND: dict[str, str] = {
    "inference": "AIFInferenceScheme",
    "conflict": "AIFConflictScheme",
    "preference": "AIFPreferenceScheme",
}

#: Node property keys minted on import that are never end-user metadata (used
#: to split "known" from "extra" fields both ways across the JSON bridge).
_KNOWN_NODE_KEYS = frozenset(
    {
        "nodeID",
        "id",
        "node_id",
        "type",
        "node_type",
        "text",
        "content",
        "scheme",
        "schemeName",
    }
)
_KNOWN_EDGE_KEYS = frozenset(
    {"edgeID", "id", "fromID", "source", "from", "toID", "target", "to"}
)


@dataclass(frozen=True)
class AIFNode:
    """One AIF node — an I-node (a claim) or an S-node (a scheme application).

    Attributes:
        node_id: The node's own (map-local, unqualified) id — mirrors AIFdb's
            ``nodeID``.
        node_type: One of :data:`AIF_NODE_TYPES` (``"I"``/``"RA"``/``"CA"``/
            ``"PA"``/``"TA"``/``"YA"``).
        text: Propositional text (I-node) or scheme display name (S-node) —
            mirrors AIFdb's node ``"text"`` field verbatim.
        scheme_name: The named scheme this S-node fulfils (e.g. ``"Default
            Inference"``, ``"Expert Opinion"``). Empty for an I-node.
        metadata: Any other JSON-safe fields on the source node dict, kept for
            lossless round-tripping through :func:`to_aifdb_json`.
    """

    node_id: str
    node_type: str
    text: str = ""
    scheme_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_type not in AIF_NODE_TYPES:
            raise ValueError(
                f"unknown AIF node type {self.node_type!r}; expected one of "
                f"{sorted(AIF_NODE_TYPES)}"
            )

    @property
    def is_scheme_node(self) -> bool:
        """Whether this is an S-node (RA/CA/PA/TA/YA) rather than an I-node."""
        return self.node_type in SCHEME_NODE_TYPES

    @property
    def owl_class(self) -> str:
        """The ``ontology_argumentation.ttl`` OWL class this node becomes."""
        return NODE_TYPE_CLASS[self.node_type]


@dataclass(frozen=True)
class AIFEdge:
    """One generic AIF wire edge (mirrors AIFdb's ``{edgeID, fromID, toID}``).

    Its ROLE (premise vs. conclusion) is not stored explicitly — per the AIF
    Upper Ontology it is structurally implied by which endpoint is the
    S-node: an edge INTO an S-node is one of its premises; an edge OUT OF an
    S-node is its conclusion (see ``ontology_argumentation.ttl``).
    """

    edge_id: str
    from_id: str
    to_id: str


@dataclass
class ArgumentMap:
    """One AIF argument map: a set of nodes + the edges linking them.

    ``map_id`` scopes every node this map writes into the KG
    (``aif:<map_id>:<node_id>``) so several maps can coexist and
    :func:`export_argument_map` can pull one back out by tag.
    """

    map_id: str = field(default_factory=lambda: f"map-{uuid.uuid4().hex}")
    nodes: list[AIFNode] = field(default_factory=list)
    edges: list[AIFEdge] = field(default_factory=list)

    def node(self, node_id: str) -> AIFNode | None:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def i_nodes(self) -> list[AIFNode]:
        """Every I-node (claim) in this map."""
        return [n for n in self.nodes if n.node_type == "I"]

    def scheme_nodes(self, kind: str | None = None) -> list[AIFNode]:
        """Every S-node, optionally filtered to one ``kind`` (e.g. ``"CA"``)."""
        if kind is not None:
            return [n for n in self.nodes if n.node_type == kind]
        return [n for n in self.nodes if n.is_scheme_node]

    def premises_of(self, scheme_node_id: str) -> list[AIFNode]:
        """Nodes with an edge INTO ``scheme_node_id`` (its premises/inputs)."""
        ids = [e.from_id for e in self.edges if e.to_id == scheme_node_id]
        return [n for n in (self.node(i) for i in ids) if n is not None]

    def conclusions_of(self, scheme_node_id: str) -> list[AIFNode]:
        """Nodes with an edge OUT OF ``scheme_node_id`` (should be exactly one)."""
        ids = [e.to_id for e in self.edges if e.from_id == scheme_node_id]
        return [n for n in (self.node(i) for i in ids) if n is not None]


def validate_argument_map(argument_map: ArgumentMap) -> list[str]:
    """Structural arity checks — the SAME rules ``shapes/argumentation.shapes.ttl``
    enforces at engine-admission time, run locally first so a bad map fails
    fast with a readable reason instead of a SHACL rejection deep in the
    ingest path. Returns an empty list when the map is well-formed.
    """
    violations: list[str] = []

    seen: set[str] = set()
    for n in argument_map.nodes:
        if n.node_id in seen:
            violations.append(f"duplicate node id {n.node_id!r}")
        seen.add(n.node_id)

    node_ids = {n.node_id for n in argument_map.nodes}
    for e in argument_map.edges:
        if e.from_id not in node_ids:
            violations.append(
                f"edge {e.edge_id!r} references unknown fromID {e.from_id!r}"
            )
        if e.to_id not in node_ids:
            violations.append(f"edge {e.edge_id!r} references unknown toID {e.to_id!r}")

    for n in argument_map.nodes:
        if n.node_type == "I" and not n.text.strip():
            violations.append(f"I-node {n.node_id!r} has no text")
            continue
        if not n.is_scheme_node:
            continue
        premises = argument_map.premises_of(n.node_id)
        conclusions = argument_map.conclusions_of(n.node_id)
        min_premises = 2 if n.node_type == "PA" else 1
        if len(premises) < min_premises:
            violations.append(
                f"{n.node_type}-node {n.node_id!r} needs >= {min_premises} "
                f"premise(s), has {len(premises)}"
            )
        if len(conclusions) != 1:
            violations.append(
                f"{n.node_type}-node {n.node_id!r} needs exactly 1 conclusion, "
                f"has {len(conclusions)}"
            )

    return violations


# ── JSON bridge (AIFdb-shaped) ──────────────────────────────────────────────


def from_aifdb_json(data: dict[str, Any], *, map_id: str | None = None) -> ArgumentMap:
    """Parse an AIFdb-shaped JSON argument map: ``{"nodes": [{"nodeID",
    "text", "type", ...}, ...], "edges": [{"edgeID", "fromID", "toID"}, ...]}``.

    Tolerant of common key aliases (``id``/``node_id`` for ``nodeID``,
    ``content`` for ``text``, ``source``/``from`` for ``fromID``, etc.) since
    "AIF-db-shaped" data in the wild is not byte-identical across exporters.
    Raises :class:`ValueError` on a node/edge missing its required id(s).
    """
    raw_nodes = data.get("nodes") or []
    raw_edges = data.get("edges") or []

    nodes: list[AIFNode] = []
    for raw in raw_nodes:
        node_id = str(
            raw.get("nodeID") or raw.get("id") or raw.get("node_id") or ""
        ).strip()
        if not node_id:
            raise ValueError(f"AIF node missing nodeID/id: {raw!r}")
        node_type = str(raw.get("type") or raw.get("node_type") or "").strip().upper()
        text = str(raw.get("text") or raw.get("content") or "")
        scheme_name = str(
            raw.get("scheme")
            or raw.get("schemeName")
            or (text if node_type in SCHEME_NODE_TYPES else "")
        )
        metadata = {k: v for k, v in raw.items() if k not in _KNOWN_NODE_KEYS}
        nodes.append(
            AIFNode(
                node_id=node_id,
                node_type=node_type,
                text=text,
                scheme_name=scheme_name,
                metadata=metadata,
            )
        )

    edges: list[AIFEdge] = []
    for idx, raw in enumerate(raw_edges):
        edge_id = str(raw.get("edgeID") or raw.get("id") or f"e{idx}")
        from_id = str(
            raw.get("fromID") or raw.get("source") or raw.get("from") or ""
        ).strip()
        to_id = str(raw.get("toID") or raw.get("target") or raw.get("to") or "").strip()
        if not from_id or not to_id:
            raise ValueError(f"AIF edge missing fromID/toID: {raw!r}")
        edges.append(AIFEdge(edge_id=edge_id, from_id=from_id, to_id=to_id))

    resolved_map_id = (
        map_id
        or str(data.get("map_id") or data.get("id") or "")
        or f"map-{uuid.uuid4().hex}"
    )
    return ArgumentMap(map_id=resolved_map_id, nodes=nodes, edges=edges)


def to_aifdb_json(argument_map: ArgumentMap) -> dict[str, Any]:
    """Render an :class:`ArgumentMap` back to AIFdb-shaped JSON — the exact
    inverse of :func:`from_aifdb_json` for a map that round-tripped through
    it (metadata excluded from the known-key set is preserved verbatim).
    """
    nodes = []
    for n in argument_map.nodes:
        row: dict[str, Any] = {"nodeID": n.node_id, "type": n.node_type, "text": n.text}
        if n.is_scheme_node and n.scheme_name:
            row["scheme"] = n.scheme_name
        row.update(n.metadata)
        nodes.append(row)
    edges = [
        {"edgeID": e.edge_id, "fromID": e.from_id, "toID": e.to_id}
        for e in argument_map.edges
    ]
    return {"map_id": argument_map.map_id, "nodes": nodes, "edges": edges}


# ── Dung projection bridge ──────────────────────────────────────────────────


@dataclass(frozen=True)
class DungProjection:
    """Pure structural projection of an AIF map onto a Dung abstract
    argumentation framework (AF = arguments + an attack relation).

    * ``arguments`` — every I-node id (the AF's argument set).
    * ``attacks`` — one ``(attacker_id, attacked_id)`` pair per CA-node
      premise->conclusion pair, AFTER preference-based filtering (below).
    * ``supports`` — one ``(supporter_id, supported_id)`` pair per RA-node
      premise->conclusion pair. Provenance only: classical Dung semantics are
      attack-only (the exact scope ``eg-epistemic``'s ``Method::ResolveConflict``
      computes over), so ``supports`` is not fed into acceptability directly.
    * ``preferences`` — one ``(preferred_id, dispreferred_id)`` pair per
      PA-node conclusion/premise pair.
    * ``dropped_attacks`` — attacks removed because a PA-node declared the
      attacked side preferred over the attacker on a MUTUAL (symmetric)
      conflict — preference-based argumentation (Amgoud & Cayrol 2002): a
      dispreferred attacker's attack on a preferred target does not survive.
      Only symmetric attacks are ever touched; a one-directional attack with
      nothing to resolve is left exactly as CA-nodes declared it.

    This is a DATA projection only — no grounded/preferred/stable solver
    lives here. Hand ``arguments`` to the engine's real
    ``client.query.resolve_conflict`` (via the ``graph_argument`` MCP tool's
    ``evaluate`` action, or ``graph_epistemic``'s ``resolve_conflict``
    action) to actually compute acceptability; the underlying attack/support
    edges :func:`import_argument_map` writes make that computation see the
    SAME topology this projection describes.
    """

    arguments: tuple[str, ...]
    attacks: tuple[tuple[str, str], ...]
    supports: tuple[tuple[str, str], ...]
    preferences: tuple[tuple[str, str], ...]
    dropped_attacks: tuple[tuple[str, str], ...]


def to_dung(argument_map: ArgumentMap) -> DungProjection:
    """Project ``argument_map`` onto a :class:`DungProjection`. See the class
    docstring for the exact CA -> attack / RA -> support / PA -> preference
    rules.
    """
    arguments = tuple(n.node_id for n in argument_map.i_nodes())

    raw_attacks: list[tuple[str, str]] = []
    for ca in argument_map.scheme_nodes("CA"):
        conclusions = argument_map.conclusions_of(ca.node_id)
        if len(conclusions) != 1:
            continue
        attacked = conclusions[0].node_id
        for premise in argument_map.premises_of(ca.node_id):
            raw_attacks.append((premise.node_id, attacked))

    supports: list[tuple[str, str]] = []
    for ra in argument_map.scheme_nodes("RA"):
        conclusions = argument_map.conclusions_of(ra.node_id)
        if len(conclusions) != 1:
            continue
        supported = conclusions[0].node_id
        for premise in argument_map.premises_of(ra.node_id):
            supports.append((premise.node_id, supported))

    preferences: list[tuple[str, str]] = []
    for pa in argument_map.scheme_nodes("PA"):
        conclusions = argument_map.conclusions_of(pa.node_id)
        if len(conclusions) != 1:
            continue
        preferred = conclusions[0].node_id
        for alt in argument_map.premises_of(pa.node_id):
            if alt.node_id != preferred:
                preferences.append((preferred, alt.node_id))

    preferred_over = set(preferences)
    attack_pairs = set(raw_attacks)
    kept: list[tuple[str, str]] = []
    dropped: list[tuple[str, str]] = []
    for attacker, attacked in raw_attacks:
        # Only a MUTUAL attack is a conflict a preference can resolve; a
        # one-directional attack stands exactly as the CA-node declared it.
        mutual = (attacked, attacker) in attack_pairs
        attacked_preferred = (attacked, attacker) in preferred_over
        if mutual and attacked_preferred:
            dropped.append((attacker, attacked))
        else:
            kept.append((attacker, attacked))

    return DungProjection(
        arguments=arguments,
        attacks=tuple(dict.fromkeys(kept)),
        supports=tuple(dict.fromkeys(supports)),
        preferences=tuple(dict.fromkeys(preferences)),
        dropped_attacks=tuple(dict.fromkeys(dropped)),
    )


# ── graph bridge (ChangeEnvelope/ingest path — the one connector write path) ──


def _qualified(map_id: str, local_id: str) -> str:
    return f"aif:{map_id}:{local_id}"


def import_argument_map(
    argument_map: ArgumentMap, *, engine: Any = None
) -> dict[str, Any]:
    """Write an AIF argument map into the KG via the shared ChangeEnvelope/
    ingest path (CONCEPT:AU-KG.ingest.change-envelope) — the SAME
    ``ingest_graph_slice`` primitive every native connector uses
    (``memory/native_ingest.py``). No parallel store: every I-node becomes a
    ``:Belief``-typed claim written alongside every other belief; every
    RA/CA-node ALSO writes the underlying ``SUPPORTS``/``ATTACKS`` edge
    ``eg-epistemic``'s belief propagation and Dung argumentation already
    read (see :func:`to_dung`), so acceptability is computed by the real
    engine solver, never re-implemented here.

    ``engine`` overrides the resolved write authority (test/dependency
    injection); ``None`` resolves the process-owned authority via
    :func:`~agent_utilities.knowledge_graph.memory.native_ingest.native_authority`.

    Returns ``{"status": "rejected", "violations": [...]}`` for a
    structurally invalid map (see :func:`validate_argument_map`) without
    writing anything. A downstream engine failure raises (writes fail loud,
    they are never silently dropped) — the SAME contract every other native
    connector write follows.
    """
    violations = validate_argument_map(argument_map)
    if violations:
        return {
            "status": "rejected",
            "map_id": argument_map.map_id,
            "violations": violations,
        }

    from ..ingestion.envelope_ingest import ingest_graph_slice
    from ..memory.native_ingest import native_authority

    entities: list[dict[str, Any]] = []
    for n in argument_map.nodes:
        row: dict[str, Any] = {
            "id": _qualified(argument_map.map_id, n.node_id),
            "node_type": n.owl_class,
            "aif_node_type": n.node_type,
            "aif_node_text": n.text,
            "aif_map_id": argument_map.map_id,
            "aif_raw_node_id": n.node_id,
        }
        if n.is_scheme_node and n.scheme_name:
            row["aif_scheme_name"] = n.scheme_name
        for key, value in n.metadata.items():
            row.setdefault(f"aif_meta_{key}", value)
        entities.append(row)

    relationships: list[dict[str, Any]] = []
    for e in argument_map.edges:
        from_node = argument_map.node(e.from_id)
        to_node = argument_map.node(e.to_id)
        if from_node is None or to_node is None:
            continue
        # AIF Upper Ontology: an edge INTO an S-node is a premise; an edge
        # OUT OF an S-node is its conclusion (ontology_argumentation.ttl).
        relationship = "aifHasPremise" if to_node.is_scheme_node else "aifHasConclusion"
        relationships.append(
            {
                "source": _qualified(argument_map.map_id, e.from_id),
                "target": _qualified(argument_map.map_id, e.to_id),
                "relationship": relationship,
                "aif_edge_id": e.edge_id,
                "aif_map_id": argument_map.map_id,
            }
        )

    # Project RA/CA structure onto the engine's OWN attack/support vocabulary
    # (eg_epistemic::model::classify_relationship) so belief propagation and
    # Dung argumentation compute over it natively — no duplicated solver.
    projection = to_dung(argument_map)
    for supporter, supported in projection.supports:
        relationships.append(
            {
                "source": _qualified(argument_map.map_id, supporter),
                "target": _qualified(argument_map.map_id, supported),
                "relationship": "SUPPORTS",
                "aif_map_id": argument_map.map_id,
                "aif_derived": True,
            }
        )
    for attacker, attacked in projection.attacks:
        relationships.append(
            {
                "source": _qualified(argument_map.map_id, attacker),
                "target": _qualified(argument_map.map_id, attacked),
                "relationship": "ATTACKS",
                "aif_map_id": argument_map.map_id,
                "aif_derived": True,
            }
        )

    authority = engine if engine is not None else native_authority()
    result = ingest_graph_slice(
        authority,
        "aif",
        entities,
        relationships,
        source_instance=argument_map.map_id,
    )
    return {
        "status": result.get("status", "failed"),
        "map_id": argument_map.map_id,
        "nodes_written": len(entities),
        "edges_written": len(relationships),
        "dung_arguments": len(projection.arguments),
        "dung_attacks": len(projection.attacks),
        "engine_result": result,
    }


def _decode_node_properties(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _decode_edge_properties(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray, list)):
        try:
            import msgpack

            decoded = msgpack.unpackb(bytes(raw), raw=False)
        except Exception:  # noqa: BLE001 — best-effort decode, never raise
            return {}
        return decoded if isinstance(decoded, dict) else {}
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def export_argument_map(map_id: str, *, engine: Any = None) -> ArgumentMap:
    """Best-effort reconstruction of a previously-imported AIF map straight
    off the live KG (graph -> AIF JSON), by its ``aif_map_id`` tag — hand the
    result to :func:`to_aifdb_json` for the wire format.

    Mirrors the established ``envelope_ingest._snapshot_rows``/
    ``ops_causal_graph.load_ops_causal_neighborhood`` idiom: a full node/edge
    scan filtered by tag, "degraded, not broken" — this NEVER raises; an
    unreachable engine or empty match returns an empty
    ``ArgumentMap(map_id=map_id)``. Only the AIF-native ``aifHasPremise``/
    ``aifHasConclusion`` edges are reconstructed; the derived ``SUPPORTS``/
    ``ATTACKS`` projection :func:`import_argument_map` also writes is skipped
    here (it is recomputable from the map via :func:`to_dung`).

    Expensive on a large shared graph (a full node+edge scan) — acceptable
    for an occasional export, not a hot path.
    """
    from ..memory.native_ingest import NativeIngestError, native_authority

    try:
        authority = engine if engine is not None else native_authority()
    except NativeIngestError:
        return ArgumentMap(map_id=map_id)

    client = getattr(authority, "client", authority)
    nodes_ns = getattr(client, "nodes", None)
    edges_ns = getattr(client, "edges", None)
    if nodes_ns is None or edges_ns is None:
        return ArgumentMap(map_id=map_id)

    try:
        raw_nodes = nodes_ns.list()
    except Exception:  # noqa: BLE001 — degrade, don't raise
        return ArgumentMap(map_id=map_id)

    nodes: list[AIFNode] = []
    qualified_to_raw: dict[str, str] = {}
    for raw_id, raw_properties in raw_nodes or []:
        props = _decode_node_properties(raw_properties)
        if str(props.get("aif_map_id") or "") != map_id:
            continue
        raw_node_id = str(props.get("aif_raw_node_id") or "")
        node_type = str(props.get("aif_node_type") or "")
        if not raw_node_id or node_type not in AIF_NODE_TYPES:
            continue
        metadata = {
            key[len("aif_meta_") :]: value
            for key, value in props.items()
            if key.startswith("aif_meta_")
        }
        nodes.append(
            AIFNode(
                node_id=raw_node_id,
                node_type=node_type,
                text=str(props.get("aif_node_text") or ""),
                scheme_name=str(props.get("aif_scheme_name") or ""),
                metadata=metadata,
            )
        )
        qualified_to_raw[str(raw_id)] = raw_node_id

    edges: list[AIFEdge] = []
    if qualified_to_raw:
        try:
            raw_edges = edges_ns.list()
        except Exception:  # noqa: BLE001 — degrade, don't raise
            raw_edges = []
        for source_id, target_id, raw_properties in raw_edges or []:
            props = _decode_edge_properties(raw_properties)
            if str(props.get("aif_map_id") or "") != map_id:
                continue
            relationship = str(props.get("relationship") or "")
            # AIF-native structural edges only — the derived SUPPORTS/ATTACKS
            # projection this same import wrote is recomputable via to_dung().
            if relationship not in {"aifHasPremise", "aifHasConclusion"}:
                continue
            source = qualified_to_raw.get(str(source_id))
            target = qualified_to_raw.get(str(target_id))
            if source is None or target is None:
                continue
            edge_id = str(props.get("aif_edge_id") or f"e{len(edges)}")
            edges.append(AIFEdge(edge_id=edge_id, from_id=source, to_id=target))

    return ArgumentMap(map_id=map_id, nodes=nodes, edges=edges)


def add_scheme(
    scheme_name: str,
    kind: str,
    *,
    description: str = "",
    scheme_id: str | None = None,
    engine: Any = None,
) -> dict[str, Any]:
    """Register a new named AIF Scheme template (the Forms-Ontology side —
    e.g. a Waltonian scheme such as ``"Argument from Expert Opinion"``, or a
    custom domain scheme) so a future RA/CA/PA-node can ``:aifFulfills`` it.

    ``kind`` is one of :data:`SCHEME_CLASS_BY_KIND` (``"inference"`` /
    ``"conflict"`` / ``"preference"``). Uses the SAME ChangeEnvelope/ingest
    write path as :func:`import_argument_map`; no parallel store.
    """
    kind_key = (kind or "").strip().lower()
    owl_class = SCHEME_CLASS_BY_KIND.get(kind_key)
    if owl_class is None:
        return {
            "status": "rejected",
            "error": f"unknown scheme kind {kind!r}; expected one of "
            f"{sorted(SCHEME_CLASS_BY_KIND)}",
        }
    name = (scheme_name or "").strip()
    if not name:
        return {"status": "rejected", "error": "scheme_name is required"}

    from ..ingestion.envelope_ingest import ingest_graph_slice
    from ..memory.native_ingest import native_authority

    local_id = (scheme_id or name).strip().lower().replace(" ", "-")
    node_id = f"aif:scheme:{kind_key}:{local_id}"
    entity: dict[str, Any] = {
        "id": node_id,
        "node_type": owl_class,
        "aif_scheme_name": name,
        "aif_scheme_kind": kind_key,
    }
    if description:
        entity["description"] = description

    authority = engine if engine is not None else native_authority()
    result = ingest_graph_slice(
        authority, "aif", [entity], [], source_instance=f"scheme:{kind_key}"
    )
    return {
        "status": result.get("status", "failed"),
        "scheme_id": node_id,
        "engine_result": result,
    }
