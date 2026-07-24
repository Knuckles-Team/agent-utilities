#!/usr/bin/python
from __future__ import annotations

"""Per-graph ownership/provenance pass over the epistemic-graph catalog (W2.8, S6, HG-4).

CONCEPT:AU-KG.audit.graph-ownership-disposition — deterministic ownership disposition
over the epistemic-graph engine's multi-graph catalog.

BACKGROUND
----------
The engine enforces "RLS default-deny: rows require explicit public visibility or an
owner grant" (its own startup banner, confirmed live in
``reports/HANDOFF-2026-07-22.md`` §8). Identity/RBAC was introduced onto a catalog that
already held data: the 2026-07-22 recovery found **85 graphs** in the durable catalog,
of which only 3 (``__commons__`` and the two ``tenant__homelab__*`` graphs) carry a
grant a provisioned identity can use — the remaining **82 legacy ``code_<repo>``
graphs** (one per code-KG-ingested repository, predating identity/RBAC) are
catalog-present but ownerless: no grant, no public flag. That is exactly the "S6"
gap this module closes the tooling for. **Do not guess grants** (repeated verbatim
in both ``reports/epistemic-graph-master-analysis-2026-07-23.md`` and the handoff) —
a wrong grant is a durable authorization change over the whole knowledge base, hence
the UNAMBIGUOUS/AMBIGUOUS split below and the human decision gate (HG-4).

WHAT THIS MODULE DOES (read-only; no mutating engine call anywhere in this file)
----------------------------------------------------------------------------------
For every graph in the catalog, collect independent provenance **signals** and reduce
them to one **disposition**:

* ``UNAMBIGUOUS(owner=X)`` — a single, consistent owning identity, either from an
  authoritative naming convention (:func:`infer_owner_from_name`, e.g. the confirmed
  ``code_<repo>`` / ``tenant__<slug>__<base>`` conventions) corroborated by (or at
  least not contradicted by) sampled node properties, or a decisive (>=90%) sampled
  node-property majority on its own.
* ``AMBIGUOUS(reasons=[...])`` — signals conflict, are absent, or are too weak to act
  on without a human. Per the "do not guess" rule this is the SAFE default whenever
  in doubt.

Grant *application* (the actual ``rbac.add_grant``/``add_role`` mutating calls) lives
in the sibling module :mod:`graph_ownership_apply` — this module never mutates
anything, so importing/running it (the report path) is inherently safe against any
live store.

SIGNALS AND THEIR HONEST LIMITS
--------------------------------
* **Graph-name convention** (:func:`infer_owner_from_name`) — every graph-creation
  call site in this codebase names graphs deterministically from an owning identity
  (:func:`~agent_utilities.knowledge_graph.core.shard_topology.tenant_graph_name`,
  the ``agent:``/``team:`` ``GraphType`` conventions, the confirmed ``code_<repo>``
  legacy-ingestion convention). This is the one signal available for EVERY graph at
  zero extra round-trips (it comes free with ``ListGraphs``).
* **Sampled node properties** — a bounded sample of each graph's nodes is read for
  ``_owner_id`` (:data:`...core.tenant_sharing.OWNER_KEY`), ``source_system``
  (:func:`...enrichment.provenance.stamp_source`'s stamped key) and ``tenant_id``.
  A >=90% majority on one value is treated as decisive.
* **Existing RBAC grants** (``rbac.list()``) — grants whose ``resource`` selector
  covers the graph (``"All"`` or an exact ``{"Graph": name}``; see
  :func:`grants_covering_graph`). A graph that already has one needs no new grant.
* **Engine ledger** (``GetLedger``) — verified against
  ``crates/eg-core/src/graph.rs::get_ledger``/``apply_ledger``: the wire format is an
  anonymous mutation log (``"ADD_NODE|<id>"``, ``"ADD_EDGE|<src>|<tgt>"``, ...) with
  **no actor/writer identity field** — confirmed by reading the parser
  (``apply_ledger`` splits on ``|`` and the first token is the op verb, never an
  agent id) and the audit chain's ``audit_line`` (same op-shape strings, chained for
  tamper-evidence, still no actor). So "last N writers from the ledger" is, with the
  CURRENT engine build, only an **activity** signal (op count / tail sample) — never
  an identity signal. This module surfaces it honestly as such rather than fabricate
  writer names; genuine per-write actor provenance is W2.6 (provenance anchoring,
  not yet shipped). Do not treat ``ledger_op_count``/``ledger_sample`` as ownership
  evidence anywhere downstream of this module.
* **``GraphType`` (Commons/Global/Agent/Team)** — returned by ``ListGraphs`` as
  ``"type"``. Note: reading ``crates/eg-core/src/isolation.rs::check_access``, the
  mandatory ``security`` feature build (what ``full``/the default build ships)
  evaluates RBAC *only* (``let _ = (graph_type, graph_owner);`` — both are explicitly
  ignored) — the ``graph_type``/``owner``-based branch only exists in a non-``security``
  build. So on the deployed engine, a ``Commons``/``Global`` type does **not** by
  itself enforce open access. This module still treats it as a genuine "explicit
  public flag" signal (:func:`is_structurally_public`) that satisfies the invariant
  ("grant OR explicit public flag") on its own — no grant is auto-recommended for
  such a graph. Given the ``security``-feature RBAC-only enforcement finding above,
  a human may still choose to add a defense-in-depth RBAC grant for one; this module
  surfaces the caveat per-graph in the rendered report rather than silently assuming
  the type alone is sufficient, but does not auto-recommend the extra grant (the
  invariant does not require it, and the program's own "do not guess grants" rule
  argues against volunteering an un-asked-for mutation).

DISPOSITION IS SAFETY-BIASED
-----------------------------
Any conflict between signals, any graph with zero readable nodes and no naming
convention, and any medium-confidence name hint with no corroborating sample all
resolve to AMBIGUOUS. Only a high-confidence naming convention or a decisive sampled
majority (with no contradicting signal) resolves UNAMBIGUOUS. This mirrors the
program decision this module implements: grants are auto-apply-UNAMBIGUOUS /
hold-ambiguous (see :mod:`graph_ownership_apply`).
"""

import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_LEDGER_SAMPLE_LIMIT",
    "DEFAULT_NODE_SAMPLE_LIMIT",
    "EngineCatalogClient",
    "EngineUnreachableError",
    "FixtureCatalogClient",
    "GrantRecord",
    "GraphOwnershipInvariantViolation",
    "GraphOwnershipSignals",
    "GraphRecord",
    "InvariantViolation",
    "LiveEngineCatalogClient",
    "NameHint",
    "OwnershipDisposition",
    "OwnershipReport",
    "RecommendedGrant",
    "build_ownership_report",
    "check_invariant",
    "collect_signals",
    "compute_disposition",
    "find_invariant_violations",
    "grants_covering_graph",
    "infer_owner_from_name",
    "invariant_enforced",
    "is_structurally_public",
    "render_markdown",
    "resolve_catalog_client",
]

#: Bounded per-graph node-property sample size (kept small: this is a one-time
#: provenance pass over ~85 graphs, not a bulk scan; a 90%+ majority is legible from
#: a small sample and a huge LIMIT would multiply the round-trip cost by graph count
#: for no disposition benefit).
DEFAULT_NODE_SAMPLE_LIMIT = 50

#: Bounded tail of the raw (anonymous) engine ledger kept per graph for the
#: activity-diagnostic column — NOT identity evidence, see the module docstring.
DEFAULT_LEDGER_SAMPLE_LIMIT = 20

#: A sampled property value must clear this share of the (non-null) sample to be
#: treated as decisive. Below this, multiple real values are competing and the
#: disposition must be AMBIGUOUS ("do not guess").
_MAJORITY_THRESHOLD = 0.9

# Property keys sampled from each graph's nodes — reuses the EXISTING stamping
# conventions rather than inventing new ones (never redefine, extend):
#   tenant_sharing.OWNER_KEY == "_owner_id", tenant_sharing.TENANT_KEY == "tenant_id"
#   enrichment.provenance.stamp_source's "source_system"
_SAMPLE_OWNER_KEY = "_owner_id"
_SAMPLE_TENANT_KEY = "tenant_id"
_SAMPLE_SOURCE_KEY = "source_system"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphRecord:
    """One ``ListGraphs`` entry — the wire shape confirmed against
    ``src/server/dispatch.rs::Method::ListGraphs`` (``{"name", "type",
    "materialization", "source_snapshot_version", "completeness_cursor", "valid",
    "index_manifests"}``; no ``owner`` field is returned over the wire)."""

    name: str
    graph_type: str | None = None
    materialization: str | None = None
    valid: bool | None = None

    @classmethod
    def from_wire(cls, row: dict[str, Any]) -> GraphRecord:
        name = str(row.get("name") or "").strip()
        if not name:
            raise ValueError("ListGraphs row is missing a non-empty 'name'")
        gtype = row.get("type")
        valid = row.get("valid")
        return cls(
            name=name,
            graph_type=str(gtype) if gtype is not None else None,
            materialization=(
                str(row.get("materialization"))
                if row.get("materialization") is not None
                else None
            ),
            valid=bool(valid) if isinstance(valid, bool) else None,
        )


@dataclass(frozen=True)
class GrantRecord:
    """One RBAC grant from ``rbac.list()``'s ``{"roles": [...], "grants": [...]}``.

    ``resource`` is a :class:`ResourceSelector` per ``epistemic_graph.client.RbacClient``:
    the literal string ``"All"``, or a one-key dict ``{"Graph": name}`` /
    ``{"Pattern": glob}`` / ``{"Label": label}``. ``action`` in
    ``{"Read","Write","Admin"}``; ``effect`` in ``{"Allow","Deny"}``.
    """

    role: str
    resource: Any
    action: str
    effect: str

    @classmethod
    def from_wire(cls, row: dict[str, Any]) -> GrantRecord | None:
        role = row.get("role")
        action = row.get("action")
        effect = row.get("effect")
        if not isinstance(role, str) or not role:
            return None
        if not isinstance(action, str) or not action:
            return None
        if not isinstance(effect, str) or not effect:
            return None
        return cls(
            role=role, resource=row.get("resource"), action=action, effect=effect
        )

    def resource_kind_and_value(self) -> tuple[str, str | None]:
        """Parse ``resource`` into ``(kind, value)``: ``("All", None)``,
        ``("Graph"|"Pattern"|"Label", value)``, or ``("unknown", None)`` for a shape
        this module does not recognize (fails soft — never crashes the report)."""
        if self.resource == "All":
            return "All", None
        if isinstance(self.resource, dict) and len(self.resource) == 1:
            ((kind, value),) = self.resource.items()
            if kind in ("Graph", "Pattern", "Label") and isinstance(value, str):
                return kind, value
        return "unknown", None

    def covers_graph(self, graph_name: str) -> bool:
        """Whether this grant's resource selector matches ``graph_name`` — a
        DELIBERATELY CONSERVATIVE match (exact ``Graph``, ``All``, or a ``fnmatch``
        ``Pattern``); ``Label`` selectors are node-label-scoped, not graph-level, so
        they never count toward "this graph has an owner grant". This module does
        NOT replicate the engine's most-specific-wins/Deny-override precedence — a
        human resolving an AMBIGUOUS graph with a matching Deny grant should read
        ``rbac.list()`` directly. Only ``Allow`` Read/Write grants cover a graph."""
        if self.effect != "Allow" or self.action not in ("Read", "Write"):
            return False
        kind, value = self.resource_kind_and_value()
        if kind == "All":
            return True
        if kind == "Graph":
            return value == graph_name
        if kind == "Pattern" and value:
            import fnmatch

            return fnmatch.fnmatchcase(graph_name, value)
        return False


def grants_covering_graph(
    grants: Iterable[GrantRecord], graph_name: str
) -> list[GrantRecord]:
    """Every grant in ``grants`` whose resource selector covers ``graph_name``."""
    return [g for g in grants if g.covers_graph(graph_name)]


@dataclass(frozen=True)
class NameHint:
    """A candidate owner inferred purely from a graph's NAME (zero round-trips)."""

    owner_hint: str | None
    basis: str
    confidence: Literal["high", "medium", "low", "none"]


@dataclass(frozen=True)
class RecommendedGrant:
    """One RBAC mutation :mod:`graph_ownership_apply` would issue for an
    UNAMBIGUOUS, not-yet-covered graph — mirrors ``RbacClient.add_grant`` exactly
    (``role``, ``resource``, ``action``, ``effect``)."""

    role: str
    resource: dict[str, str]
    action: str
    effect: str = "Allow"


@dataclass
class GraphOwnershipSignals:
    """Every independently-collected provenance signal for one graph, pre-fusion."""

    record: GraphRecord
    name_hint: NameHint
    sampled_node_count: int = 0
    owner_id_votes: Counter[str] = field(default_factory=Counter)
    source_system_votes: Counter[str] = field(default_factory=Counter)
    tenant_id_votes: Counter[str] = field(default_factory=Counter)
    sample_error: str | None = None
    ledger_op_count: int | None = None
    ledger_sample: tuple[str, ...] = ()
    ledger_error: str | None = None
    grants: tuple[GrantRecord, ...] = ()


@dataclass(frozen=True)
class OwnershipDisposition:
    """The fused, human-reviewable per-graph decision-table row."""

    graph: str
    status: Literal["UNAMBIGUOUS", "AMBIGUOUS"]
    owner: str | None
    reasons: tuple[str, ...]
    explicit_public: bool
    existing_grants: tuple[GrantRecord, ...]
    recommended_grants: tuple[RecommendedGrant, ...]
    sampled_node_count: int
    ledger_op_count: int | None
    graph_type: str | None = None

    @property
    def already_covered(self) -> bool:
        return self.explicit_public or bool(self.existing_grants)


@dataclass
class OwnershipReport:
    """The complete pass — one row per catalog graph, plus run metadata."""

    generated_at: str
    mode: Literal["live", "template"]
    source_note: str
    node_sample_limit: int
    dispositions: list[OwnershipDisposition]

    @property
    def counts(self) -> dict[str, int]:
        c = {
            "total": len(self.dispositions),
            "unambiguous": 0,
            "ambiguous": 0,
            "already_covered": 0,
            "explicit_public": 0,
            "needs_grant": 0,
        }
        for d in self.dispositions:
            c["unambiguous" if d.status == "UNAMBIGUOUS" else "ambiguous"] += 1
            if d.already_covered:
                c["already_covered"] += 1
            if d.explicit_public:
                c["explicit_public"] += 1
            if d.status == "UNAMBIGUOUS" and not d.already_covered:
                c["needs_grant"] += 1
        return c


# ---------------------------------------------------------------------------
# Structural "explicit public" + name-convention inference
# ---------------------------------------------------------------------------


def _default_graph_name(config: Any = None) -> str:
    try:
        from ..core.shard_topology import default_graph_name

        return default_graph_name(config)
    except Exception as exc:  # noqa: BLE001 — a naming hint must never crash the pass
        logger.debug("default_graph_name unavailable, using literal fallback: %s", exc)
        return "__commons__"


#: GraphType values that are a deliberate, explicit sharing choice at CreateGraph
#: time (see the module docstring's caveat re: the `security`-feature RBAC-only
#: enforcement path — this is still surfaced as a public-flag SIGNAL, not treated
#: as a substitute for an RBAC grant on the live engine).
_PUBLIC_GRAPH_TYPES = frozenset({"Commons", "Global"})


def is_structurally_public(
    name: str, *, record: GraphRecord | None = None, config: Any = None
) -> bool:
    """True when ``name``/``record`` is public by explicit architectural convention:
    the configured default/commons graph (``tenant_sharing.commons_graph_name`` —
    "readable across orgs" by design), or a ``GraphType`` of ``Commons``/``Global``.
    """
    if name == _default_graph_name(config) or name == "__commons__":
        return True
    if record is not None and record.graph_type in _PUBLIC_GRAPH_TYPES:
        return True
    return False


# ``tenant_graph_name`` builds ``tenant__<slug>__<base>``; slug/base may themselves
# contain underscores (see the module docstring / HANDOFF example
# ``tenant__homelab____commons__``), so this parses by stripping the confirmed
# ``tenant__`` prefix and a caller's ``__<base>`` suffix rather than a fragile
# double-underscore split.
_TENANT_PREFIX = "tenant__"
#: Known literal ``base`` values ``tenant_graph_name`` is called with in this
#: codebase: the configured ``default_graph_name`` (usually ``__commons__``) plus
#: the plain ``"default"`` literal confirmed in the real corpus — e.g.
#: ``tenant__homelab__default`` alongside ``tenant__homelab____commons__`` for the
#: SAME tenant (``reports/HANDOFF-2026-07-22.md`` §8). Not exhaustive: a genuinely
#: custom base this module cannot guess degrades SAFELY (no tenant match for that
#: one graph -> AMBIGUOUS via the normal fusion rule, never a false-positive owner).
_KNOWN_TENANT_BASES: tuple[str, ...] = ("default",)
_AGENT_RE = re.compile(r"^agent:(?P<who>.+)$")
_TEAM_RE = re.compile(r"^team:(?P<who>.+)$")
# Confirmed legacy convention (HANDOFF-2026-07-22 §8): 82/85 graphs recovered into
# the catalog before identity/RBAC existed are named ``code_<repo>`` (underscore),
# distinct from the node-property `source_system=code:<repo>` (colon) stamp.
_CODE_RE = re.compile(r"^code[_:](?P<repo>.+)$")


def _parse_tenant_graph(name: str, base: str) -> str | None:
    if not name.startswith(_TENANT_PREFIX):
        return None
    for candidate_base in dict.fromkeys((base, *_KNOWN_TENANT_BASES)):
        suffix = f"__{candidate_base}"
        if name.endswith(suffix) and len(name) > len(_TENANT_PREFIX) + len(suffix):
            slug = name[len(_TENANT_PREFIX) : -len(suffix)]
            if slug:
                return slug
    return None


def infer_owner_from_name(name: str, config: Any = None) -> NameHint:
    """Best-effort candidate owner from ``name`` alone — the one signal available
    for every catalog graph at zero extra round-trips. See the module docstring for
    the confidence rationale of each branch.

    The configured default/commons graph is treated as just another naming
    convention (owner ``"commons"``) rather than a special case bolted onto
    :func:`compute_disposition` — so it flows through the SAME fusion logic as
    every other graph (and, notably, can never mask an unrelated conflict the way
    a blanket "public wins" override would)."""
    base = _default_graph_name(config)
    if name in (base, "__commons__"):
        return NameHint(
            owner_hint="commons",
            basis=(
                "configured default/commons graph "
                "(tenant_sharing.commons_graph_name — readable across orgs by design)"
            ),
            confidence="high",
        )

    slug = _parse_tenant_graph(name, base)
    if slug:
        return NameHint(
            owner_hint=f"tenant:{slug}",
            basis="tenant_graph_name convention (KG-2.58 org routing)",
            confidence="high",
        )

    m = _CODE_RE.match(name)
    if m:
        repo = m.group("repo")
        return NameHint(
            owner_hint="system:code-ingestion",
            basis=(
                f"legacy code-KG ingestion graph for repo '{repo}' "
                "(HANDOFF-2026-07-22 confirmed convention: 82/85 legacy graphs)"
            ),
            confidence="high",
        )

    m = _AGENT_RE.match(name)
    if m:
        who = m.group("who")
        return NameHint(
            owner_hint=f"agent:{who}",
            basis="GraphType::Agent naming convention (owner set at CreateGraph, not wire-readable after)",
            confidence="medium",
        )

    m = _TEAM_RE.match(name)
    if m:
        who = m.group("who")
        return NameHint(
            owner_hint=f"team:{who}",
            basis="GraphType::Team naming convention",
            confidence="medium",
        )

    return NameHint(
        owner_hint=None, basis="no recognized naming convention", confidence="none"
    )


# ---------------------------------------------------------------------------
# Engine catalog client Protocol + two implementations
# ---------------------------------------------------------------------------


class EngineUnreachableError(RuntimeError):
    """The live engine (or one specific call against it) could not be used.

    Raised only by :class:`LiveEngineCatalogClient`; callers building a report
    catch this and fall back to :class:`FixtureCatalogClient` in TEMPLATE mode —
    never a reason to fabricate data.
    """


@runtime_checkable
class EngineCatalogClient(Protocol):
    """The minimal read-only surface this module needs from an engine — small and
    Protocol-typed so :class:`FixtureCatalogClient` (tests, TEMPLATE mode) and
    :class:`LiveEngineCatalogClient` (the real engine) are interchangeable."""

    def list_graphs(self) -> list[dict[str, Any]]:
        """``ListGraphs`` rows: ``[{"name", "type", ...}, ...]``."""
        ...

    def rbac_policy(self) -> dict[str, Any]:
        """``RbacAdmin{op:"List"}`` → ``{"roles": [...], "grants": [...]}``."""
        ...

    def sample_nodes(self, graph_name: str, limit: int) -> list[dict[str, Any]]:
        """Up to ``limit`` rows of ``{"owner_id", "source_system", "tenant_id"}``
        (aliased columns) sampled from ``graph_name``."""
        ...

    def graph_ledger(self, graph_name: str) -> list[str]:
        """The raw (anonymous) ``GetLedger`` op strings for ``graph_name``."""
        ...


class FixtureCatalogClient:
    """In-memory :class:`EngineCatalogClient` — the fixture/TEMPLATE-mode backend
    and the seam every unit test in ``tests/unit/knowledge_graph/test_graph_ownership.py``
    drives. Never touches a socket."""

    def __init__(
        self,
        graphs: Sequence[dict[str, Any]],
        *,
        grants: Sequence[dict[str, Any]] | None = None,
        roles: Sequence[dict[str, Any]] | None = None,
        node_samples: dict[str, list[dict[str, Any]]] | None = None,
        ledgers: dict[str, list[str]] | None = None,
    ) -> None:
        self._graphs = [dict(g) for g in graphs]
        self._policy: dict[str, Any] = {
            "roles": [dict(r) for r in (roles or [])],
            "grants": [dict(g) for g in (grants or [])],
        }
        self._node_samples = {k: list(v) for k, v in (node_samples or {}).items()}
        self._ledgers = {k: list(v) for k, v in (ledgers or {}).items()}

    def list_graphs(self) -> list[dict[str, Any]]:
        return [dict(g) for g in self._graphs]

    def rbac_policy(self) -> dict[str, Any]:
        return {
            "roles": [dict(r) for r in self._policy["roles"]],
            "grants": [dict(g) for g in self._policy["grants"]],
        }

    def sample_nodes(self, graph_name: str, limit: int) -> list[dict[str, Any]]:
        return [dict(r) for r in self._node_samples.get(graph_name, [])][
            : max(0, limit)
        ]

    def graph_ledger(self, graph_name: str) -> list[str]:
        return list(self._ledgers.get(graph_name, []))


class LiveEngineCatalogClient:
    """Read-only adapter over the real epistemic-graph engine, via the SAME
    process-authority path ``agent_utilities/mcp/tools/engine_tools.py`` uses
    (:class:`~agent_utilities.knowledge_graph.core.graph_compute.GraphComputeEngine`
    — one shared transport, cheap per-graph views). Every method normalizes any
    failure (unreachable engine, auth, timeout, malformed response) to
    :class:`EngineUnreachableError` so a caller can fall back to fixture/TEMPLATE
    mode rather than ever fabricate or half-report a graph's data.

    Only READ methods are called here (``tenants.list``, ``rbac.list``,
    ``query.cypher_read``, ``ledger.get``) — this class cannot mutate anything; grant
    *application* lives only in :mod:`graph_ownership_apply`.
    """

    def __init__(self, *, config: Any = None) -> None:
        self._config = config

    def _default_view(self) -> Any:
        from ..core.graph_compute import GraphComputeEngine

        return GraphComputeEngine.get_or_create().client

    def _graph_view(self, graph_name: str) -> Any:
        from ..core.graph_compute import GraphComputeEngine

        return GraphComputeEngine.get_or_create(graph_name=graph_name).client

    def list_graphs(self) -> list[dict[str, Any]]:
        try:
            rows = self._default_view().tenants.list()
        except Exception as exc:
            raise EngineUnreachableError("engine tenants.list() failed") from exc
        return [dict(r) for r in (rows or []) if isinstance(r, dict)]

    def rbac_policy(self) -> dict[str, Any]:
        try:
            policy = self._default_view().rbac.list()
        except Exception as exc:
            raise EngineUnreachableError("engine rbac.list() failed") from exc
        if not isinstance(policy, dict):
            raise EngineUnreachableError(
                "engine rbac.list() returned a non-dict policy"
            )
        return policy

    def sample_nodes(self, graph_name: str, limit: int) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 500))
        query = (
            "MATCH (n) RETURN "
            f"n.{_SAMPLE_OWNER_KEY} AS owner_id, "
            f"n.{_SAMPLE_SOURCE_KEY} AS source_system, "
            f"n.{_SAMPLE_TENANT_KEY} AS tenant_id "
            f"LIMIT {bounded}"
        )
        try:
            rows = self._graph_view(graph_name).query.cypher_read(query)
        except Exception as exc:
            raise EngineUnreachableError(
                f"engine node sample failed for graph {graph_name!r}"
            ) from exc
        return [dict(r) for r in (rows or []) if isinstance(r, dict)]

    def graph_ledger(self, graph_name: str) -> list[str]:
        try:
            ops = self._graph_view(graph_name).ledger.get()
        except Exception as exc:
            raise EngineUnreachableError(
                f"engine ledger.get() failed for graph {graph_name!r}"
            ) from exc
        return [str(op) for op in (ops or [])]


def resolve_catalog_client(config: Any = None) -> EngineCatalogClient:
    """Return a :class:`LiveEngineCatalogClient` bound to ``config``.

    Construction never raises/connects by itself (the adapter is lazy per-call) —
    callers should still wrap their FIRST call (e.g. :meth:`list_graphs`) in a
    try/except :class:`EngineUnreachableError` and fall back to
    :class:`FixtureCatalogClient` when it fails, per the module contract.
    """
    return LiveEngineCatalogClient(config=config)


# ---------------------------------------------------------------------------
# Signal collection
# ---------------------------------------------------------------------------


def collect_signals(
    client: EngineCatalogClient,
    record: GraphRecord,
    all_grants: Sequence[GrantRecord],
    *,
    node_sample_limit: int = DEFAULT_NODE_SAMPLE_LIMIT,
    ledger_sample_limit: int = DEFAULT_LEDGER_SAMPLE_LIMIT,
    config: Any = None,
) -> GraphOwnershipSignals:
    """Collect every independent signal for one graph. Best-effort per sub-signal:
    a failed node sample or ledger read degrades that ONE field to an error note,
    never aborts the whole graph's row (one unreadable graph must not blank the
    other 84)."""
    signals = GraphOwnershipSignals(
        record=record,
        name_hint=infer_owner_from_name(record.name, config),
        grants=tuple(grants_covering_graph(all_grants, record.name)),
    )

    try:
        rows = client.sample_nodes(record.name, node_sample_limit)
    except Exception as exc:  # noqa: BLE001 — one graph's sample must not abort the pass
        signals.sample_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "graph_ownership: node sample failed for %s (%s: %s)",
            record.name,
            type(exc).__name__,
            exc,
        )
        rows = []

    signals.sampled_node_count = len(rows)
    for row in rows:
        owner = row.get("owner_id")
        if isinstance(owner, str) and owner.strip():
            signals.owner_id_votes[owner.strip()] += 1
        source = row.get("source_system")
        if isinstance(source, str) and source.strip():
            signals.source_system_votes[source.strip()] += 1
        tenant = row.get("tenant_id")
        if isinstance(tenant, str) and tenant.strip():
            signals.tenant_id_votes[tenant.strip()] += 1

    try:
        ops = client.graph_ledger(record.name)
    except Exception as exc:  # noqa: BLE001 — ledger is a diagnostic, never fatal
        signals.ledger_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "graph_ownership: ledger read failed for %s (%s: %s)",
            record.name,
            type(exc).__name__,
            exc,
        )
        ops = []

    signals.ledger_op_count = len(ops) if ops or signals.ledger_error is None else None
    signals.ledger_sample = tuple(ops[-ledger_sample_limit:])

    return signals


# ---------------------------------------------------------------------------
# Disposition fusion (pure; the safety-biased decision rule)
# ---------------------------------------------------------------------------


def _majority(votes: Counter[str]) -> tuple[str, float] | None:
    """The top-voted value and its share of non-null votes, or ``None`` if empty."""
    total = sum(votes.values())
    if total == 0:
        return None
    value, count = votes.most_common(1)[0]
    return value, count / total


#: Coarse owner-hint values that are, BY DESIGN, decoupled from the per-graph
#: sampled value they should still corroborate — the ``code_*`` legacy convention
#: deliberately maps every repo's graph to the SAME reusable ``"system:code-
#: ingestion"`` owner (:func:`infer_owner_from_name`, one role reused across ~82
#: graphs) while a sampled node's ``source_system`` is necessarily per-repo
#: (``"code:<repo>"``) — plain substring containment can never relate the two.
#: Maps a coarse owner-hint value to the sampled-value PREFIX it corroborates.
_COARSE_OWNER_FAMILIES: tuple[tuple[str, str], ...] = (
    ("system:code-ingestion", "code:"),
)


def _related(a: str, b: str) -> bool:
    """Whether two owner-ish strings plausibly name the same thing.

    Two checks, both deliberately permissive (a FALSE negative here only
    degrades a graph from UNAMBIGUOUS to AMBIGUOUS — safe, a human looks at one
    more row; this is used only to CONFIRM agreement between two independently-
    derived signals, never to invent an owner neither one named on its own, so a
    false positive is the actual risk to guard against):

    1. Loose substring containment (e.g. name-hint ``"tenant:homelab"`` vs a
       sampled ``tenant_id`` value ``"homelab"``).
    2. The explicit :data:`_COARSE_OWNER_FAMILIES` mapping, for the one case
       substring containment structurally cannot cover (see its docstring).
    """
    a_norm, b_norm = a.lower(), b.lower()
    if a_norm in b_norm or b_norm in a_norm:
        return True
    for owner_value, sample_prefix in _COARSE_OWNER_FAMILIES:
        if (a_norm == owner_value and b_norm.startswith(sample_prefix)) or (
            b_norm == owner_value and a_norm.startswith(sample_prefix)
        ):
            return True
    return False


def compute_disposition(
    signals: GraphOwnershipSignals, *, config: Any = None
) -> OwnershipDisposition:
    """Fuse :class:`GraphOwnershipSignals` into one :class:`OwnershipDisposition`.

    Safety-biased by construction (see the module docstring): a conflict, an absent
    signal, or a merely-plausible-but-uncorroborated medium-confidence name hint all
    resolve AMBIGUOUS. Only a high-confidence naming convention, or a >=90% sampled
    property majority, resolves UNAMBIGUOUS — and only when nothing else
    contradicts it.
    """
    name = signals.record.name
    reasons: list[str] = []

    explicit_public = is_structurally_public(name, record=signals.record, config=config)
    if explicit_public:
        basis = (
            "configured default/commons graph"
            if name in (_default_graph_name(config), "__commons__")
            else f"GraphType={signals.record.graph_type} (explicit share choice at creation)"
        )
        reasons.append(f"structurally public: {basis}")

    existing = signals.grants
    if existing:
        role_actions = sorted({f"{g.role}/{g.action}" for g in existing})
        reasons.append("already has a covering grant: " + ", ".join(role_actions))

    hint = signals.name_hint
    owner_sample = _majority(signals.owner_id_votes)
    source_sample = _majority(signals.source_system_votes)
    tenant_sample = _majority(signals.tenant_id_votes)

    if signals.sample_error:
        reasons.append(f"node sample unavailable ({signals.sample_error})")

    # Prefer the more direct _owner_id signal over source_system when both are
    # decisive (tenant_sharing.OWNER_KEY names the actual writing actor; source_system
    # names an upstream connector, one level further from "who owns this graph").
    decisive_sample: tuple[str, str, float] | None = None  # (field, value, share)
    if owner_sample and owner_sample[1] >= _MAJORITY_THRESHOLD:
        decisive_sample = ("_owner_id", owner_sample[0], owner_sample[1])
    elif source_sample and source_sample[1] >= _MAJORITY_THRESHOLD:
        decisive_sample = ("source_system", source_sample[0], source_sample[1])

    conflicting_sample = False
    for label, votes in (
        ("owner_id", signals.owner_id_votes),
        ("source_system", signals.source_system_votes),
    ):
        total = sum(votes.values())
        if total >= 2 and len(votes) > 1:
            top_share = votes.most_common(1)[0][1] / total
            if top_share < _MAJORITY_THRESHOLD:
                conflicting_sample = True
                top3 = ", ".join(f"{v!r}:{c}" for v, c in votes.most_common(3))
                reasons.append(f"conflicting sampled {label} values: {top3}")

    status: Literal["UNAMBIGUOUS", "AMBIGUOUS"]
    owner: str | None

    if conflicting_sample:
        status, owner = "AMBIGUOUS", None
    elif decisive_sample is not None:
        field_name, value, share = decisive_sample
        sample_owner = f"{'actor' if field_name == '_owner_id' else 'source'}:{value}"
        reasons.append(
            f"sampled {field_name} majority ({share:.0%} of {signals.sampled_node_count}): {value!r}"
        )
        if hint.owner_hint and hint.confidence in ("high", "medium"):
            if _related(hint.owner_hint, value):
                status, owner = "UNAMBIGUOUS", hint.owner_hint
                reasons.append(f"corroborated by name convention: {hint.basis}")
            else:
                status, owner = "AMBIGUOUS", None
                reasons.append(
                    f"name convention suggests {hint.owner_hint!r} but sampled "
                    f"nodes show {value!r} — unresolved conflict"
                )
        else:
            status, owner = "UNAMBIGUOUS", sample_owner
    elif hint.confidence == "high":
        status, owner = "UNAMBIGUOUS", hint.owner_hint
        reasons.append(f"name convention ({hint.basis}); no contradicting sample")
    elif (
        hint.confidence == "medium"
        and signals.sampled_node_count > 0
        and not conflicting_sample
    ):
        # Medium-confidence name hint corroborated by a non-conflicting (even if not
        # individually decisive) sample — a real signal exists and does not disagree.
        status, owner = "UNAMBIGUOUS", hint.owner_hint
        reasons.append(
            f"name convention ({hint.basis}); sample present and non-conflicting"
        )
    else:
        status, owner = "AMBIGUOUS", None
        if hint.confidence in ("medium", "low", "none"):
            reasons.append(
                f"name convention ({hint.basis}) not independently corroborated"
                if hint.owner_hint
                else "no recognized naming convention and no decisive node sample"
            )

    # NOTE: `explicit_public` deliberately does NOT force `status` to UNAMBIGUOUS
    # here. `already_covered` (= explicit_public OR existing grants) already
    # satisfies the architecture invariant independent of `status` — forcing the
    # status would MASK a genuine name-vs-sample identity conflict on a graph that
    # also happens to carry a public GraphType (e.g. a code_* graph mismarked
    # Global). The one graph that both needs and deserves an automatic
    # "commons"-flavoured status is the actual default/commons graph — handled by
    # :func:`infer_owner_from_name` as a normal high-confidence name convention,
    # not a special case here.

    recommended: tuple[RecommendedGrant, ...] = ()
    if status == "UNAMBIGUOUS" and not existing and not explicit_public and owner:
        role = _role_for_owner(owner)
        resource = {"Graph": name}
        recommended = (
            RecommendedGrant(
                role=role, resource=resource, action="Read", effect="Allow"
            ),
            RecommendedGrant(
                role=role, resource=resource, action="Write", effect="Allow"
            ),
        )

    if tenant_sample and not decisive_sample:
        reasons.append(
            f"sampled tenant_id (corroborating only): {tenant_sample[0]!r} "
            f"({tenant_sample[1]:.0%})"
        )

    return OwnershipDisposition(
        graph=name,
        status=status,
        owner=owner,
        reasons=tuple(reasons) or ("no signals available",),
        explicit_public=explicit_public,
        existing_grants=existing,
        recommended_grants=recommended,
        sampled_node_count=signals.sampled_node_count,
        ledger_op_count=signals.ledger_op_count,
        graph_type=signals.record.graph_type,
    )


def _role_for_owner(owner: str) -> str:
    """Slugify a disposition ``owner`` string into a durable RBAC role name.

    ``owner:<slug>`` — deliberately coarse for the ``system:code-ingestion`` bucket
    (:func:`infer_owner_from_name` intentionally drops the per-repo suffix from the
    OWNER string for this exact reason: one reusable grantee role, N per-graph
    ``{"Graph": name}`` grants — never N near-duplicate roles for one system actor).
    """
    slug = re.sub(r"[^a-z0-9]+", "-", owner.lower()).strip("-") or "unknown"
    return f"owner:{slug}"


# ---------------------------------------------------------------------------
# Report assembly + rendering
# ---------------------------------------------------------------------------


def build_ownership_report(
    client: EngineCatalogClient,
    *,
    mode: Literal["live", "template"],
    source_note: str,
    node_sample_limit: int = DEFAULT_NODE_SAMPLE_LIMIT,
    ledger_sample_limit: int = DEFAULT_LEDGER_SAMPLE_LIMIT,
    config: Any = None,
) -> OwnershipReport:
    """Run the full pass: ``list_graphs`` + ``rbac_policy`` once, then per-graph
    :func:`collect_signals` + :func:`compute_disposition`. ``client`` may be a
    :class:`FixtureCatalogClient` (TEMPLATE mode / tests) or a
    :class:`LiveEngineCatalogClient` (live mode) — this function does not care
    which; the caller decides ``mode``/``source_note`` (see ``scripts/graph_ownership_report.py``
    for the live-with-fixture-fallback wiring)."""
    graph_rows = client.list_graphs()
    records = [GraphRecord.from_wire(row) for row in graph_rows]

    policy = client.rbac_policy()
    raw_grants = policy.get("grants") if isinstance(policy, dict) else None
    all_grants = [
        g
        for g in (
            GrantRecord.from_wire(row)
            for row in (raw_grants or [])
            if isinstance(row, dict)
        )
        if g is not None
    ]

    dispositions: list[OwnershipDisposition] = []
    for record in sorted(records, key=lambda r: r.name):
        signals = collect_signals(
            client,
            record,
            all_grants,
            node_sample_limit=node_sample_limit,
            ledger_sample_limit=ledger_sample_limit,
            config=config,
        )
        dispositions.append(compute_disposition(signals, config=config))

    return OwnershipReport(
        generated_at=datetime.now(UTC).isoformat(),
        mode=mode,
        source_note=source_note,
        node_sample_limit=node_sample_limit,
        dispositions=dispositions,
    )


def _fmt_grant(g: GrantRecord) -> str:
    kind, value = g.resource_kind_and_value()
    resource = kind if kind in ("All", "unknown") else f"{kind}:{value}"
    return f"{g.role} {g.action}/{g.effect} on {resource}"


def _fmt_recommended(rg: RecommendedGrant) -> str:
    return f"{rg.role} {rg.action}/{rg.effect} on Graph:{rg.resource.get('Graph', '?')}"


def render_markdown(report: OwnershipReport) -> str:
    """Render the HG-4 decision-table artifact."""
    counts = report.counts
    lines: list[str] = []
    lines.append("# Graph Ownership Report — W2.8 (HG-4 decision table)")
    lines.append("")
    if report.mode == "template":
        lines.append(
            "> **MODE: TEMPLATE.** The live engine was not reachable read-only from "
            "the tooling environment this run executed in — this report is generated "
            "against a fixture catalog, not the real ~82/85-graph legacy corpus. The "
            "tooling itself is the deliverable; re-run "
            "`scripts/graph_ownership_report.py` from an environment with a live, "
            "authenticated graph-os/engine connection to regenerate this report "
            "against the real catalog before HG-4 acts on it."
        )
        lines.append("")
    else:
        lines.append("> **MODE: LIVE.** Generated against the real engine catalog.")
        lines.append("")
    lines.append(f"- Generated: `{report.generated_at}`")
    lines.append(f"- Source: {report.source_note}")
    lines.append(f"- Node sample limit per graph: {report.node_sample_limit}")
    lines.append(f"- Total catalog graphs: {counts['total']}")
    lines.append(
        f"- Disposition: **{counts['unambiguous']} UNAMBIGUOUS**, "
        f"**{counts['ambiguous']} AMBIGUOUS**"
    )
    lines.append(
        f"- Coverage: {counts['already_covered']} already covered "
        f"(grant or public), {counts['explicit_public']} explicit-public, "
        f"**{counts['needs_grant']} need a new grant applied**"
    )
    lines.append("")
    lines.append(
        "**Program decision this table feeds (HG-4):** grants are "
        "auto-apply-UNAMBIGUOUS / hold-ambiguous. `scripts/apply_graph_ownership_grants.py "
        "--apply-unambiguous` applies ONLY the UNAMBIGUOUS rows below that are not "
        "already covered; every AMBIGUOUS row is held for a human decision — see its "
        "`reasons` column. This run never applied anything (dry-run only; see the "
        "dry-run sample output in the task's final report)."
    )
    lines.append("")
    lines.append(
        "**Engine ledger caveat:** `ledger_ops` is an activity count only — the "
        "engine's `GetLedger` records anonymous mutation-shape strings "
        "(`ADD_NODE|<id>`, ...) with no actor/writer identity in the current build "
        "(verified against `eg-core::graph::get_ledger`/`apply_ledger`); it is never "
        "used as ownership evidence in the disposition column."
    )
    lines.append("")
    lines.append(
        "| Graph | Type | Disposition | Owner | Already covered | Recommended grant(s) | Sampled nodes | Ledger ops | Reasons |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for d in report.dispositions:
        covered = "yes" if d.already_covered else "no"
        existing = "; ".join(_fmt_grant(g) for g in d.existing_grants)
        recommended = "; ".join(_fmt_recommended(g) for g in d.recommended_grants)
        grant_cell = existing or recommended or "—"
        ledger = "n/a" if d.ledger_op_count is None else str(d.ledger_op_count)
        reasons = "<br>".join(d.reasons)
        lines.append(
            f"| `{d.graph}` | {d.graph_type or '—'} | **{d.status}** | "
            f"{d.owner or '—'} | {covered} | {grant_cell} | {d.sampled_node_count} | "
            f"{ledger} | {reasons} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Architecture invariant — "every catalog graph has >=1 grant OR an explicit
# public flag" (report-only pre-HG-4, hard-fail post-HG-4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvariantViolation:
    graph: str
    reason: str


class GraphOwnershipInvariantViolation(AssertionError):
    """Raised by :func:`check_invariant` only when enforcement is active AND
    violations exist. Post-HG-4 (see :func:`invariant_enforced`) this is the
    permanent architecture invariant: every catalog graph must resolve
    ``already_covered`` (a grant or the explicit-public flag)."""

    def __init__(self, violations: Sequence[InvariantViolation]) -> None:
        self.violations = tuple(violations)
        head = "; ".join(f"{v.graph}: {v.reason}" for v in violations[:10])
        tail = "" if len(violations) <= 10 else f" (+{len(violations) - 10} more)"
        super().__init__(
            f"{len(violations)} catalog graph(s) lack a covering grant or explicit "
            f"public flag: {head}{tail}"
        )


def invariant_enforced() -> bool:
    """Whether the invariant HARD-FAILS (post-HG-4) or is report-only (pre-HG-4).

    ``KG_GRAPH_OWNERSHIP_ENFORCED`` (default ``false``) — a one-time migration
    cutover switch (Configuration discipline: not a per-deployment tunable; it
    exists solely to gate this program's HG-4 human decision, flipped exactly once
    after HG-4 applies the UNAMBIGUOUS grants and a human resolves every AMBIGUOUS
    graph — see ``docs/architecture/configuration.md``). Read live via
    ``config.setting`` (never a bare ``os.environ`` read) so a deploy can flip it
    without a code change.
    """
    from agent_utilities.core.config import setting

    return bool(setting("KG_GRAPH_OWNERSHIP_ENFORCED", False))


def find_invariant_violations(
    dispositions: Iterable[OwnershipDisposition],
) -> list[InvariantViolation]:
    """Every graph that is NEITHER explicitly public NOR already grant-covered —
    independent of UNAMBIGUOUS/AMBIGUOUS status (a graph can be AMBIGUOUS about WHO
    owns it yet still already covered by a pre-existing grant; that is not an
    invariant violation, just a disposition a human may still want to review)."""
    violations = []
    for d in dispositions:
        if d.already_covered:
            continue
        violations.append(
            InvariantViolation(d.graph, "no covering grant and not explicitly public")
        )
    return violations


def check_invariant(
    dispositions: Iterable[OwnershipDisposition], *, enforced: bool | None = None
) -> list[InvariantViolation]:
    """Report-only by default; raises :class:`GraphOwnershipInvariantViolation`
    when ``enforced`` (or, if ``None``, :func:`invariant_enforced`) is true and any
    violation exists. Always returns the violation list either way, so a
    report-only caller can still log/surface them."""
    dispositions = list(dispositions)
    violations = find_invariant_violations(dispositions)
    active = invariant_enforced() if enforced is None else enforced
    if active and violations:
        raise GraphOwnershipInvariantViolation(violations)
    return violations
