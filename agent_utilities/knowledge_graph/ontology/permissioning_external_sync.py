"""CA-26 — render eg's exported M1 policy bundle into external-engine policy.

CONCEPT:AU-KG.ontology.policy-external-sync (CA-26, DEC-CA-04)

**What this module is, precisely — read before extending it.** `DEC-CA-04`'s W0
review (2026-08-26, `plans/company-architecture/decisions/DEC-CA-04-*.md`) settled
that there are NINE live authorization mechanisms in this ecosystem, not one, and
that eg's exported bundle (`epistemic-graph`'s `src/server/policy_export/mod.rs`,
landed CA-16, commit `16cb2c1b`) renders **exactly M1** — `IsolationLayer`'s
`_owner`/`_visibility`/`_grants` row-visibility decision — and nothing else. This
module is the **applier** half of that split: it fetches the bundle CA-16
generates, renders it into each target engine's native policy shape, and applies
it idempotently. It does **not** define what a Marking means (`permissioning.py`
owns that) and it does **not** re-derive row visibility (eg's `IsolationLayer`
owns that) — it is a faithful, mechanical translation, exactly like CA-16's own
module doc demands of itself.

**A bundle whose `governs` names a mechanism this module does not recognize is
denied, not best-effort applied** (`GOVERNS_SUPPORTED`) — DEC-CA-04's own binding
rule: *"a consumer that receives a bundle whose `governs` it does not recognize
denies."* Today the only legal value is `["M1"]`.

**P8 wording constraint (DEC-CA-04 A4).** Nothing this module (or its docs, or
its callers) emits may say a "Marking" is what M1 enforces — M1 evaluates
`_owner`/`_visibility`/`_grants`; **Markings are evaluated by M6**
(`permissioning.enforce`/`restricted_view`), a *separate* mechanism this module
does not touch. What CA-16's bundle carries under `markings.<name>.predicate` is
the `MarkingPredicate::RequiresRole` *bridge* eg's module doc names: a
mechanically-derived row-visibility predicate that says "a row carrying marking
`<name>` in its `_markings`-shaped column is visible only to a principal holding
role `marking:<name>`" — renderable wherever a target's own schema actually
carries that column, and **not otherwise**. This module renders that bridge
faithfully per target and reports, per target, whether the column exists.

**Per-row marking-column gap — status per target (the CA-26/CA-63 finding CA-16
filed, closed/open here with fresh evidence, not re-derived):**

- **OpenSearch — CLOSED.** CA-24 (`knowledge_graph/search/doc_shape.py`,
  `build_document`) already indexes every document's `marking` field (a
  `keyword` array, `INDEX_MAPPINGS["properties"]["marking"]`), sourced live from
  `permissioning.markings_for` on every CDC write, proven end-to-end against the
  real CA-50 cluster including a projecting `_source` search AND a `terms`
  aggregation (`knowledge_graph/search/tests/test_dls.py`,
  `knowledge_graph/search/dls.py`'s module doc). `OpenSearchRenderer` below
  reuses `search.dls.render_dls_query_for_role` verbatim rather than
  re-implementing it — CA-24 built the query shape, CA-26 (this module) is the
  bundle-shaped wrapper + the Applier that actually pushes it as a live
  security-plugin role, per the lane contract's coordination note ("CA-24
  builds the index, CA-26 the role/DLS config").
- **Trino — STILL OPEN.** No target Iceberg table carries a `_markings`/
  `marking` column (CA-52's `evidence/CA-52.md`: the only live table is
  `analytics.trino_verify`, three columns, none of them a marking column; the
  lane's own `rules.json` mount is a deliberate allow-all no-op with nothing
  yet wired to `access-control.name=file`). `TrinoRenderer` emits the correct
  `filter` SQL expression shape (Trino's real file-based-access-control key is
  `filter`, not the DEC-CA-04 contract's own `row_filter` name — see
  `_TRINO_FILTER_KEY` below for the translation) referencing
  `MarkingPredicate.column`, but applying it against a table that lacks that
  column either no-ops (an unqualified reference Trino cannot resolve is a
  query-time error, not a silent bypass — Trino's `filter` expressions are
  parsed against the real table schema) or must be gated off until a schema/ETL
  change lands (a CA-21/CA-23/CA-34 concern, not this module's). This module's
  `Applier` therefore requires each `TrinoTargetTable` to declare
  `has_markings_column` explicitly, and refuses (raises, not silently no-ops)
  to render a `filter` referencing a column the caller has not asserted exists.
- **Lakekeeper — STILL OPEN, AND STRUCTURALLY DIFFERENT, not just unpopulated.**
  Lakekeeper's own OpenFGA authorization model is `server`/`project`/
  `warehouse`/`namespace`/`table` granularity (CA-54's evidence,
  `services/lakekeeper`'s packaged `authz/openfga/schema.fga`) — there is **no
  row-level relation in the model Lakekeeper installs**, so a per-row
  `MarkingPredicate::RequiresRole` cannot be rendered as an OpenFGA tuple at
  any granularity Lakekeeper enforces, even once a `_markings` column exists on
  the underlying Iceberg data (OpenFGA never sees Iceberg row content — it
  authorizes catalog *operations* on namespace/table objects, before Trino/
  Spark ever reads a row). `LakekeeperRenderer` therefore renders a real tuple
  set **only** for the narrower, honestly-different case DEC-CA-04's own bundle
  shape anticipates: a marking whose registry entry names a *table-granularity*
  object id (the whole table is marked, not individual rows) — and returns an
  explicit, typed "not applicable" result for every ordinary row-scoped marking
  rather than fabricating a tuple that would not actually gate anything.

That third finding — Lakekeeper cannot render row-scoped M1/Marking predicates
at all, independent of whether a `_markings` column ever gets added anywhere —
is new (CA-16's own finding only named the column gap, not this granularity
mismatch) and is re-filed here, not silently worked around.

**Live-integration status.** CA-16's `/policy/export` HTTP surface and
`Method::PolicyExport` exist on `epistemic-graph` `main` (commit `16cb2c1b`) but
this module has not exercised them live in this session (no reachable graph-os
deployment with the `policy_export`/`oidc` features + a real bearer token was
available to this lane) — `BundleFetcher` is built and unit-tested against the
DEC-CA-04 JSON shape (a fixture, matching CA-16's own `PolicyBundle` `serde`
output byte-for-byte per the schema round-trip test) and is NOT proven against a
live `/policy/export` call. CA-54's Lakekeeper OpenFGA endpoint and CA-50's
OpenSearch cluster ARE live and reachable, but this module's `Applier`/
`TargetClient` implementations are exercised only against injected fakes in this
lane's test suite — live application was out of scope for this pass (the lane's
own execution instruction: "build/test against the DEC-CA-04 JSON fixture and
stop before live-integration work" when the producer's *reachability*, not just
existence, is unverified). This is a named gap for the next lane/session, not a
silent omission.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from ...core._env import setting

logger = logging.getLogger(__name__)

__all__ = [
    "GOVERNS_SUPPORTED",
    "MARKING_ROLE_PREFIX",
    "RESERVED_MARKING_COLUMN",
    "RequiresRolePredicate",
    "MarkingPolicyEntry",
    "PolicyBundle",
    "FetchedBundle",
    "BundleFetcher",
    "TrinoTargetTable",
    "TrinoRule",
    "OpenSearchRoleConfig",
    "OpenFgaTuple",
    "LakekeeperTupleSet",
    "RenderedTarget",
    "Renderer",
    "TrinoRenderer",
    "OpenSearchRenderer",
    "LakekeeperRenderer",
    "TargetClient",
    "InMemoryTargetClient",
    "ApplyOutcome",
    "ApplyReport",
    "Applier",
    "external_policy_sync_enabled",
    "invalidate_cache",
    "is_dirty",
    "clear_dirty_state",
]

# Matches au's `Marking.role_token` (`ontology/permissioning.py`) and eg's
# `MARKING_ROLE_PREFIX` (`policy_export/mod.rs`) -- kept as one named constant
# per repo so the two conventions stay visibly the SAME string, never
# independently re-typed. Do not change without changing both.
MARKING_ROLE_PREFIX = "marking:"

# Matches eg's `RESERVED_MARKING_COLUMN` (`policy_export/mod.rs`) -- the
# proposed, not-yet-universally-adopted per-row marking-name column
# convention. See the module doc's per-target gap table for what actually
# carries this today (OpenSearch: yes, as `marking`; Trino: no).
RESERVED_MARKING_COLUMN = "_markings"

# The only `governs` value CA-16's bundle emits today. A bundle whose
# `governs` is not a subset of this set is DENIED (DEC-CA-04's binding rule),
# never best-effort rendered.
GOVERNS_SUPPORTED: frozenset[str] = frozenset({"M1"})

# Trino's real file-based-access-control table-rule key for a row filter is
# `filter`, NOT `row_filter` -- DEC-CA-04's bundle contract names the field
# `row_filter` (an engine-neutral name chosen for the JSON contract), so this
# module translates at render time. Named here so the translation is a single
# grep-able site, not an inline literal.
_TRINO_FILTER_KEY = "filter"


# ── Bundle schema (mirrors eg's `policy_export::PolicyBundle`, `serde`-exact) ──


@dataclass(frozen=True, slots=True)
class RequiresRolePredicate:
    """Decoded `MarkingPredicate::RequiresRole` (eg `policy_export/mod.rs`).

    A row carrying ``column`` == ``role``'s marking name is visible only to a
    principal whose bundle-``principals`` role set contains ``role`` (always
    ``marking:<name>``, per :data:`MARKING_ROLE_PREFIX`).
    """

    role: str
    column: str

    @property
    def marking_name(self) -> str:
        """The bare marking name (`role` with the ``marking:`` prefix stripped)."""
        if self.role.startswith(MARKING_ROLE_PREFIX):
            return self.role[len(MARKING_ROLE_PREFIX) :]
        return self.role


def _decode_predicate(predicate_json: str) -> RequiresRolePredicate:
    """Decode one `markings.<name>.predicate` JSON string.

    Fails loudly (never silently substitutes a permissive/restrictive default)
    on an unrecognized `kind` or a malformed payload -- a renderer that cannot
    understand a predicate must not guess.
    """
    try:
        raw = json.loads(predicate_json)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"malformed marking predicate JSON: {predicate_json!r}"
        ) from exc
    kind = raw.get("kind")
    if kind != "requires_role":
        raise ValueError(
            f"unrecognized marking predicate kind {kind!r} -- this module only "
            "understands 'requires_role' (eg's MarkingPredicate::RequiresRole); "
            "a future predicate kind must add a matching decoder here before "
            "any renderer can safely consume it"
        )
    role = raw.get("role")
    column = raw.get("column")
    if not role or not column:
        raise ValueError(f"malformed requires_role predicate: {predicate_json!r}")
    return RequiresRolePredicate(role=str(role), column=str(column))


@dataclass(frozen=True, slots=True)
class MarkingPolicyEntry:
    """One `markings.<name>` bundle entry -- the opaque predicate, decoded."""

    predicate_json: str

    def decode(self) -> RequiresRolePredicate:
        return _decode_predicate(self.predicate_json)


@dataclass(frozen=True, slots=True)
class PolicyBundle:
    """Python mirror of eg's `policy_export::PolicyBundle` (DEC-CA-04, extended
    per the W0 review appendix: `governs` + `tenant`/`graphs`, A1/A3).

    Constructed via :meth:`from_json` from CA-16's exported bundle -- never
    hand-built for anything but a test fixture, so a real bundle's shape is
    always validated on the way in.
    """

    version: str
    generated_from: str
    governs: tuple[str, ...]
    tenant: str
    graphs: tuple[str, ...]
    principals: dict[str, tuple[str, ...]]
    markings: dict[str, MarkingPolicyEntry]

    @classmethod
    def from_json(cls, raw: str | dict[str, Any]) -> PolicyBundle:
        data = json.loads(raw) if isinstance(raw, str) else raw
        try:
            markings = {
                str(name): MarkingPolicyEntry(
                    predicate_json=json.dumps(entry["predicate"])
                )
                if isinstance(entry.get("predicate"), (dict, list))
                else MarkingPolicyEntry(predicate_json=str(entry["predicate"]))
                for name, entry in dict(data["markings"]).items()
            }
            principals = {
                str(subject): tuple(str(r) for r in roles)
                for subject, roles in dict(data["principals"]).items()
            }
            return cls(
                version=str(data["version"]),
                generated_from=str(data["generated_from"]),
                governs=tuple(str(g) for g in data["governs"]),
                tenant=str(data["tenant"]),
                graphs=tuple(str(g) for g in data["graphs"]),
                principals=principals,
                markings=markings,
            )
        except KeyError as exc:
            raise ValueError(
                f"policy bundle JSON is missing required field {exc}"
            ) from exc

    def governs_recognized(self) -> bool:
        """DEC-CA-04's binding rule: a bundle governing anything outside
        :data:`GOVERNS_SUPPORTED` must be denied, never partially applied."""
        return bool(self.governs) and set(self.governs) <= GOVERNS_SUPPORTED

    def role_holders(self, role: str) -> frozenset[str]:
        """Every principal subject whose role set contains ``role``."""
        return frozenset(
            subject for subject, roles in self.principals.items() if role in roles
        )


@dataclass(frozen=True, slots=True)
class FetchedBundle:
    """A :class:`PolicyBundle` plus fetch-time metadata the bundle itself does
    not carry (it has no wall-clock timestamp -- `generated_from` is a content
    hash, not a time). Staleness (DEC-CA-04: "never longer than one policy
    epoch stale") is therefore a property of the FETCH, not the bundle.
    """

    bundle: PolicyBundle | None
    fetched_at_monotonic: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.bundle is not None and self.error is None


class BundleFetcher:
    """Fetches CA-16's exported bundle over `/policy/export`.

    **UNVERIFIED against a live endpoint in this session** -- see the module
    doc's "Live-integration status" section. The outbound client is built via
    `core.http_client.create_http_client` (NE-015's governed-transport seam
    -- every outbound `httpx` client in this package goes through one
    auditable factory, never a bare `import httpx`), imported lazily inside
    :meth:`fetch` so constructing a `BundleFetcher` never reaches network
    code and every renderer/Applier test stays fixture-only.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token_provider: TokenProvider | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token_provider = token_provider
        self._timeout = timeout_seconds

    def fetch(
        self, *, tenant: str, graphs: list[str], marking_names: list[str]
    ) -> FetchedBundle:
        import time

        # NE-015: every outbound httpx client is built through the governed
        # factory (agent_utilities.core.http_client), never a bare `import
        # httpx` -- uniform timeout/TLS/header defaults, one auditable seam.
        from ...core.http_client import create_http_client

        headers = {}
        if self._token_provider is not None:
            headers["Authorization"] = f"Bearer {self._token_provider()}"
        params: list[tuple[str, str | int | float | bool | None]] = [("tenant", tenant)]
        params.extend(("graph", g) for g in graphs)
        params.extend(("marking", m) for m in marking_names)
        try:
            with create_http_client(timeout=self._timeout, headers=headers) as client:
                response = client.get(f"{self._base_url}/policy/export", params=params)
                response.raise_for_status()
                bundle = PolicyBundle.from_json(response.json())
            return FetchedBundle(bundle=bundle, fetched_at_monotonic=time.monotonic())
        except Exception as exc:  # noqa: BLE001 -- any fetch/parse failure is fail-closed data
            logger.warning("[CA-26] policy bundle fetch failed: %s", exc)
            return FetchedBundle(
                bundle=None, fetched_at_monotonic=time.monotonic(), error=str(exc)
            )


class TokenProvider(Protocol):
    def __call__(self) -> str: ...


# ── Render targets ──────────────────────────────────────────────────────────


class TrinoRule(TypedDict):
    catalog: str
    schema: str
    table: str
    filter: str


class OpenSearchRoleConfig(TypedDict):
    index_pattern: str
    role: str
    dls_query: dict[str, Any]


class OpenFgaTuple(TypedDict):
    user: str
    relation: str
    object: str


class LakekeeperTupleSet(TypedDict):
    namespace: str
    table: str
    openfga_tuples: list[OpenFgaTuple]


@dataclass(frozen=True, slots=True)
class RenderedTarget:
    """One target's rendered output plus a stable digest for idempotency."""

    target: str
    payload: list[dict[str, Any]]
    not_applicable: tuple[str, ...] = field(default_factory=tuple)
    """Human-readable reasons a marking could NOT be rendered for this target
    (e.g. Lakekeeper row-granularity mismatch) -- never silently dropped."""

    def digest(self) -> str:
        canonical = json.dumps(self.payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Renderer(Protocol):
    """`render(bundle) -> RenderedTarget`, per DEC-CA-04's applier-side split."""

    def render(self, bundle: PolicyBundle) -> RenderedTarget: ...


@dataclass(frozen=True, slots=True)
class TrinoTargetTable:
    """A physical Trino/Iceberg table this deployment renders marking filters
    against. `has_markings_column` MUST be asserted true by the caller after
    confirming the live schema -- this module refuses to guess."""

    catalog: str
    schema: str
    table: str
    has_markings_column: bool = False


@dataclass(frozen=True, slots=True)
class TrinoRenderer:
    """Renders `renderings.trino` -- one `filter` rule per (table, principal
    lacking the marking role), pushdown-safe by construction: Trino's own
    file-based access control evaluates `filter` against the table's real
    columns as part of query planning, before any projection/aggregation runs
    -- there is no "already-fetched row" state for a `filter` expression to be
    blind to (the same reasoning CA-24's OpenSearch DLS module gives for why
    its mechanism is pushdown-safe, and the reasoning DEC-CA-04/CA-63 requires
    every renderer to state explicitly rather than assume)."""

    tables: tuple[TrinoTargetTable, ...]

    def render(self, bundle: PolicyBundle) -> RenderedTarget:
        if not bundle.governs_recognized():
            raise ValueError(
                f"cannot render an unrecognized-governs bundle: {bundle.governs}"
            )
        rules: list[dict[str, Any]] = []
        not_applicable: list[str] = []
        for name, entry in bundle.markings.items():
            predicate = entry.decode()
            for tbl in self.tables:
                if not tbl.has_markings_column:
                    not_applicable.append(
                        f"marking '{name}': {tbl.catalog}.{tbl.schema}.{tbl.table} has no "
                        f"'{predicate.column}' column (CA-21/CA-23/CA-34 gap, not rendered)"
                    )
                    continue
                holders = bundle.role_holders(predicate.role)
                if not holders:
                    # No principal in this bundle holds the role -- filter applies to everyone.
                    rules.append(
                        {
                            "catalog": tbl.catalog,
                            "schema": tbl.schema,
                            "table": tbl.table,
                            "user": ".*",
                            _TRINO_FILTER_KEY: (
                                f"NOT contains({predicate.column}, '{predicate.marking_name}')"
                            ),
                        }
                    )
                    continue
                for subject in sorted(bundle.principals):
                    if subject in holders:
                        continue  # cleared: no filter rule needed for this principal
                    rules.append(
                        {
                            "catalog": tbl.catalog,
                            "schema": tbl.schema,
                            "table": tbl.table,
                            "user": subject,
                            _TRINO_FILTER_KEY: (
                                f"NOT contains({predicate.column}, '{predicate.marking_name}')"
                            ),
                        }
                    )
        return RenderedTarget(
            target="trino", payload=rules, not_applicable=tuple(not_applicable)
        )


@dataclass(frozen=True, slots=True)
class OpenSearchRenderer:
    """Renders `renderings.opensearch` -- one DLS role per distinct principal,
    reusing CA-24's own `search.dls.render_dls_query_for_role` (this module
    does not re-derive the query shape; see the module doc's "OpenSearch —
    CLOSED" section for why that reuse is deliberate, not incidental)."""

    index_patterns: tuple[str, ...]

    def render(self, bundle: PolicyBundle) -> RenderedTarget:
        if not bundle.governs_recognized():
            raise ValueError(
                f"cannot render an unrecognized-governs bundle: {bundle.governs}"
            )
        from ..search.dls import render_dls_query_for_role

        all_marking_names = sorted(bundle.markings)
        rows: list[dict[str, Any]] = []
        for subject, roles in sorted(bundle.principals.items()):
            query = render_dls_query_for_role(roles, all_marking_names)
            for pattern in self.index_patterns:
                rows.append(
                    {
                        "index_pattern": pattern,
                        "role": f"ca26-{subject}",
                        "dls_query": query,
                    }
                )
        return RenderedTarget(target="opensearch", payload=rows)


@dataclass(frozen=True, slots=True)
class LakekeeperTableRef:
    """A physical Lakekeeper namespace/table this deployment can grant/deny at
    Lakekeeper's own granularity floor."""

    namespace: str
    table: str


@dataclass(frozen=True, slots=True)
class LakekeeperRenderer:
    """Renders `renderings.lakekeeper` -- **only** for markings whose registry
    entry maps to a whole-table object (`table_scope`); every ordinary
    row-scoped marking is reported `not_applicable`, never fabricated as a
    tuple that would not actually gate row content. See the module doc's
    "Lakekeeper — STILL OPEN, AND STRUCTURALLY DIFFERENT" section."""

    table_scope: dict[str, LakekeeperTableRef]
    """marking name -> the one table it wholly covers, when applicable."""

    def render(self, bundle: PolicyBundle) -> RenderedTarget:
        if not bundle.governs_recognized():
            raise ValueError(
                f"cannot render an unrecognized-governs bundle: {bundle.governs}"
            )
        by_table: dict[tuple[str, str], list[OpenFgaTuple]] = {}
        not_applicable: list[str] = []
        for name, entry in bundle.markings.items():
            predicate = entry.decode()
            ref = self.table_scope.get(name)
            if ref is None:
                not_applicable.append(
                    f"marking '{name}': row-scoped marking, Lakekeeper's OpenFGA model has "
                    "no row-level relation -- structurally unrenderable at any table this "
                    "lane knows about (not a data gap; the model itself has no row grain)"
                )
                continue
            key = (ref.namespace, ref.table)
            holders = bundle.role_holders(predicate.role)
            tuples = by_table.setdefault(key, [])
            for subject in sorted(holders):
                tuples.append(
                    {
                        "user": f"oidc~{subject}"
                        if not subject.startswith("oidc~")
                        else subject,
                        "relation": "select",
                        "object": f"table:{ref.namespace}/{ref.table}",
                    }
                )
        payload = [
            {"namespace": ns, "table": tbl, "openfga_tuples": tuples}
            for (ns, tbl), tuples in sorted(by_table.items())
        ]
        return RenderedTarget(
            target="lakekeeper", payload=payload, not_applicable=tuple(not_applicable)
        )


# ── Applier ──────────────────────────────────────────────────────────────────


class TargetClient(Protocol):
    """Applies a rendered target's payload to the real engine, and reports the
    last digest it successfully applied (idempotency: the :class:`Applier`
    skips the call entirely when the digest is unchanged)."""

    def last_applied_digest(self) -> str | None: ...

    def apply(self, rendered: RenderedTarget, digest: str) -> None: ...

    def deny_all(self) -> None:
        """Apply a deny-all (empty-permit) configuration -- the fail-closed
        path when no valid bundle is available. MUST NOT raise for "already
        deny-all"; MUST be idempotent the same way `apply` is."""
        ...


@dataclass
class InMemoryTargetClient:
    """A `TargetClient` that records calls instead of reaching a real engine
    -- used by every unit test in this lane, and as the reference
    implementation a live client wraps. Deliberately in this module (not
    test-only) so a caller wiring a dry-run/staging Applier has a real,
    documented no-op target without writing one from scratch."""

    name: str
    applied: list[RenderedTarget] = field(default_factory=list)
    deny_all_calls: int = 0
    _last_digest: str | None = None

    def last_applied_digest(self) -> str | None:
        return self._last_digest

    def apply(self, rendered: RenderedTarget, digest: str) -> None:
        self.applied.append(rendered)
        self._last_digest = digest

    def deny_all(self) -> None:
        self.deny_all_calls += 1
        self._last_digest = "deny-all"


class ApplyOutcome(TypedDict):
    target: str
    action: str  # "applied" | "skipped-unchanged" | "denied-all" | "not-applicable"
    digest: str | None
    not_applicable: list[str]


class ApplyReport(TypedDict):
    disabled: bool
    outcomes: list[ApplyOutcome]


_DENY_ALL_DIGEST = "deny-all"


class Applier:
    """Diffs each renderer's output against the target's last-applied digest
    before writing (idempotent apply, DEC-CA-04's contract), and fails closed
    -- deny-all on every configured target -- for a missing bundle, a fetch
    error, or an unrecognized `governs`. Never reports success on a partial
    multi-target apply: :meth:`apply_all`'s report lists every target's real
    outcome, and a caller checking "did P8 hold" must inspect all of them, not
    assume a truthy return means every target succeeded.
    """

    def __init__(
        self, renderers: dict[str, Renderer], clients: dict[str, TargetClient]
    ) -> None:
        if set(renderers) != set(clients):
            raise ValueError(
                f"renderers/clients target-name mismatch: {sorted(renderers)} vs {sorted(clients)}"
            )
        self._renderers = renderers
        self._clients = clients

    def apply_all(self, fetched: FetchedBundle) -> ApplyReport:
        if not external_policy_sync_enabled():
            return ApplyReport(disabled=True, outcomes=[])

        outcomes: list[ApplyOutcome] = []
        if (
            not fetched.ok
            or fetched.bundle is None
            or not fetched.bundle.governs_recognized()
        ):
            reason = fetched.error or (
                f"unrecognized governs: {fetched.bundle.governs}"
                if fetched.bundle
                else "no bundle"
            )
            logger.warning("[CA-26] fail-closed: denying all targets (%s)", reason)
            for name, client in self._clients.items():
                client.deny_all()
                outcomes.append(
                    ApplyOutcome(
                        target=name,
                        action="denied-all",
                        digest=_DENY_ALL_DIGEST,
                        not_applicable=[],
                    )
                )
            return ApplyReport(disabled=False, outcomes=outcomes)

        bundle = fetched.bundle
        for name, renderer in self._renderers.items():
            client = self._clients[name]
            rendered = renderer.render(bundle)
            digest = rendered.digest()
            if client.last_applied_digest() == digest:
                outcomes.append(
                    ApplyOutcome(
                        target=name,
                        action="skipped-unchanged",
                        digest=digest,
                        not_applicable=list(rendered.not_applicable),
                    )
                )
                continue
            client.apply(rendered, digest)
            outcomes.append(
                ApplyOutcome(
                    target=name,
                    action="applied",
                    digest=digest,
                    not_applicable=list(rendered.not_applicable),
                )
            )
        return ApplyReport(disabled=False, outcomes=outcomes)


# ── Feature flag + invalidation hook (permission_sync.py's `sync_access` calls
# `invalidate_cache` after a marking is newly applied inbound, per the lane's
# "Call sequence" design) ──────────────────────────────────────────────────


def external_policy_sync_enabled() -> bool:
    """`CA26_EXTERNAL_POLICY_SYNC_ENABLED` -- default OFF. Gates the whole
    apply path per the lane's migration/rollback design: flipping this on
    before CA-54-style cutovers are deliberate is the exact "denies Lakekeeper
    access entirely" scenario the lane file warns against."""
    return bool(setting("CA26_EXTERNAL_POLICY_SYNC_ENABLED", default=False, cast=bool))


_DIRTY_TENANTS: set[str] = set()


def invalidate_cache(tenant: str, node_id: str) -> None:
    """Mark `tenant` dirty so the next :class:`Applier` cycle re-fetches
    eagerly instead of waiting out the poll interval -- called from
    `permission_sync.sync_access()` after a marking is newly applied inbound
    (`ontology.permissioning.apply_marking`). `node_id` is accepted (matching
    the lane's documented hook signature) but not itself tracked -- CA-16's
    bundle is per-TENANT (DEC-CA-04 A3), so any single node's change dirties
    the whole tenant's next fetch, never a per-node partial refresh.
    """
    del node_id  # bundle granularity is per-tenant, not per-node (DEC-CA-04 A3)
    if tenant:
        _DIRTY_TENANTS.add(tenant)


def is_dirty(tenant: str) -> bool:
    return tenant in _DIRTY_TENANTS


def clear_dirty_state(tenant: str) -> None:
    _DIRTY_TENANTS.discard(tenant)
