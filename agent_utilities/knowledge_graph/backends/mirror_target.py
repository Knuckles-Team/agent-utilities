#!/usr/bin/python
# CONCEPT:AU-KG.backend.mirror-target-graph - One uniform mirror *target* concept resolved once in the fan-out layer and mapped by each external backend onto its native isolation unit (Neo4j database, Stardog database/named graph, Apache AGE graph, FalkorDB graph key), so a mirror can write either the instance default or a graph dedicated to mirroring.
# CONCEPT:AU-KG.backend.mirror-nonempty-default-guard - Pre-flight refusal: a mirror whose resolved target is the instance default and whose instance default already holds data is refused before a single write, naming the dedicated-graph alternative and the explicit override.
"""Configurable mirror target graph for the external mirror backends.

CONCEPT:AU-KG.backend.mirror-target-graph — Mirror Target Graph.
CONCEPT:AU-KG.backend.mirror-nonempty-default-guard — Non-Empty Default Guard.

Every external mirror (CONCEPT:AU-KG.backend.mirror-health-repair) writes into
*some* isolation unit of its store. Each store spells that unit differently —

===========  ====================================================
Backend      Native isolation unit
===========  ====================================================
Neo4j        a **database** (``session(database=...)``)
Stardog      a **database** *and* a SPARQL **named graph** (context)
AGE          an **AGE graph** (``ag_catalog.create_graph``)
FalkorDB     a **graph key** (``client.select_graph``)
===========  ====================================================

— and before this module each backend resolved it its own way, so "where does
mirroring land?" had four different answers and no single place to make it safe.

This module is that single place. A :class:`MirrorTarget` is resolved **once**
(in ``create_backend``, the one construction seam the fan-out mirror set goes
through) and handed to the backend, which maps it onto its native unit. The
resolution never happens inside a backend, so the four cannot drift.

The two modes an operator asks for
----------------------------------
``mirror_target`` is declared per connection in ``kg_connections`` — no new
environment variable (*Configuration discipline*):

``"mirror_target": "dedicated"``
    Mirror into a graph **dedicated to mirroring**, created if absent. The name
    defaults to :data:`DEDICATED_MIRROR_NAME`; ``{"mode": "dedicated", "name":
    "..."}`` picks another. Nothing that already lives in the instance is touched.

``"mirror_target": "default"``
    Mirror into the **instance default** (Neo4j's home database, Stardog's
    default database + default graph, the AGE/FalkorDB code-default graph).
    Guarded — see below.

``"mirror_target": "default-overwrite"``
    The instance default, **guard waived**. The explicit acknowledgement that
    mirrored data may commingle with, and overwrite, what is already there.

A connection that already names its native selector (``database`` /
``db_name`` / ``graph_name``) resolves to :data:`MODE_NAMED` and is left exactly
as it was — an existing deployment keeps working unchanged.

The safety property
-------------------
:func:`preflight_mirror_target` runs before a mirror is attached to the fan-out.
When the resolved target *is* the instance default (mode :data:`MODE_DEFAULT`,
whether declared or reached by omission) and that default **already contains
data**, the mirror is refused: nothing is written, the refusal is logged loudly
with the two ways to fix it, and the mirror is reported unhealthy. The guard
fails **closed** — a probe that cannot prove the target empty refuses too, since
the whole point is to never write into someone else's data on a guess.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ── Modes ────────────────────────────────────────────────────────────────────
#: The connection explicitly names its native selector (``database`` /
#: ``db_name`` / ``graph_name``). Derived, never declared: it is what every
#: pre-existing deployment resolves to, and it is never guarded or redirected.
MODE_NAMED = "named"
#: The instance default, guarded by :func:`preflight_mirror_target`.
MODE_DEFAULT = "default"
#: A graph dedicated to mirroring, created if absent.
MODE_DEDICATED = "dedicated"
#: The instance default with the guard explicitly waived.
MODE_DEFAULT_OVERWRITE = "default-overwrite"

#: Modes an operator may write in ``kg_connections`` (``named`` is derived).
DECLARABLE_MODES = frozenset({MODE_DEFAULT, MODE_DEDICATED, MODE_DEFAULT_OVERWRITE})

# ── Levels (only a two-level store such as Stardog uses this) ────────────────
#: A named graph / context *inside* a database.
LEVEL_GRAPH = "graph"
#: A whole database.
LEVEL_DATABASE = "database"

#: The graph a ``"dedicated"`` target uses when the operator names none. One
#: value that works everywhere, so the common case needs no name at all.
DEDICATED_MIRROR_NAME = "agent_kg_mirror"

#: The connection-spec key carrying the declaration.
MIRROR_TARGET_FIELD = "mirror_target"

#: IRI prefix for a dedicated SPARQL named graph. ``urn:`` (not http) for the
#: same reason ``source_partition.SOURCE_GRAPH_PREFIX`` is: a named-graph IRI
#: must never look dereferenceable.
MIRROR_GRAPH_PREFIX = "urn:mirror:"

# Conservative across every store's identifier rules at once: AGE's
# ``[A-Za-z_][A-Za-z0-9_]{0,62}`` is the tightest, Neo4j allows ``.-``, FalkorDB
# allows nearly anything. A backend's own validator still applies on top.
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,62}")


class MirrorTargetError(ValueError):
    """A ``mirror_target`` declaration that cannot be honored as written."""


class MirrorTargetRefused(RuntimeError):
    """The pre-flight guard refused a mirror rather than write into live data.

    CONCEPT:AU-KG.backend.mirror-nonempty-default-guard. Raised by
    :func:`preflight_mirror_target` *before* any mirror write, so the refusal
    costs the instance nothing.
    """


@dataclass(frozen=True)
class MirrorTarget:
    """Where one mirror writes, expressed independently of any store's dialect.

    Attributes:
        mode: one of :data:`MODE_NAMED`, :data:`MODE_DEFAULT`,
            :data:`MODE_DEDICATED`, :data:`MODE_DEFAULT_OVERWRITE`.
        name: the resolved unit name — the named selector for
            :data:`MODE_NAMED`, the dedicated graph for :data:`MODE_DEDICATED`,
            and the store's own default name (or ``None`` where the store has a
            genuine implicit default, e.g. Neo4j's home database) otherwise.
        level: which isolation unit a two-level store should use
            (:data:`LEVEL_GRAPH` / :data:`LEVEL_DATABASE`); ``None`` for a
            single-level store.
    """

    mode: str
    name: str | None = None
    level: str | None = None

    @property
    def is_dedicated(self) -> bool:
        """True when this target is a graph reserved for mirroring."""
        return self.mode == MODE_DEDICATED

    @property
    def is_instance_default(self) -> bool:
        """True when this target is the store's own default unit."""
        return self.mode in (MODE_DEFAULT, MODE_DEFAULT_OVERWRITE)

    @property
    def guarded(self) -> bool:
        """True when :func:`preflight_mirror_target` must prove the target empty.

        Exactly :data:`MODE_DEFAULT`. :data:`MODE_NAMED` is a unit the operator
        chose (so it keeps working unchanged), :data:`MODE_DEDICATED` belongs to
        the mirror, and :data:`MODE_DEFAULT_OVERWRITE` is the explicit waiver.
        """
        return self.mode == MODE_DEFAULT

    def describe(self) -> str:
        """A short human phrase for logs and refusal messages."""
        unit = self.level or "graph"
        if self.mode == MODE_NAMED:
            return f"the configured {unit} {self.name!r}"
        if self.is_dedicated:
            return f"the dedicated mirror {unit} {self.name!r}"
        named = f" {self.name!r}" if self.name else ""
        suffix = " (guard waived)" if self.mode == MODE_DEFAULT_OVERWRITE else ""
        return f"the instance default {unit}{named}{suffix}"


def parse_mirror_target(raw: Any) -> MirrorTarget | None:
    """Parse a ``mirror_target`` declaration; ``None`` when nothing is declared.

    Tolerant of the two shapes a declaration arrives in (the same tolerance
    ``_parse_mirror_targets`` gives the mirror *name* list):

    * a bare mode string — ``"dedicated"`` / ``"default"`` / ``"default-overwrite"``;
    * a mapping — ``{"mode": "dedicated", "name": "kg_mirror", "level": "graph"}``.

    Raises :class:`MirrorTargetError` on anything else, so a typo can never be
    read as "no declaration" and silently fall back to the instance default.
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, MirrorTarget):
        return raw
    mode: Any
    name: Any = None
    level: Any = None
    if isinstance(raw, str):
        mode = raw.strip().lower()
    elif isinstance(raw, dict):
        mode = str(raw.get("mode") or "").strip().lower()
        name = raw.get("name")
        level = raw.get("level")
        unsupported = set(raw).difference({"mode", "name", "level"})
        if unsupported:
            raise MirrorTargetError(
                f"{MIRROR_TARGET_FIELD} accepts only 'mode', 'name' and 'level'; "
                f"got unsupported {sorted(unsupported)!r}"
            )
    else:
        raise MirrorTargetError(
            f"{MIRROR_TARGET_FIELD} must be a mode string or an object, "
            f"not {type(raw).__name__}"
        )

    if mode not in DECLARABLE_MODES:
        raise MirrorTargetError(
            f"{MIRROR_TARGET_FIELD} mode {mode!r} is not one of "
            f"{sorted(DECLARABLE_MODES)!r}"
        )
    if name is not None:
        name = str(name).strip()
        if mode != MODE_DEDICATED:
            raise MirrorTargetError(
                f"{MIRROR_TARGET_FIELD} 'name' only applies to mode "
                f"{MODE_DEDICATED!r}; mode {mode!r} writes the instance default"
            )
        if not name:
            name = None
    if level is not None:
        level = str(level).strip().lower()
        if level not in (LEVEL_GRAPH, LEVEL_DATABASE):
            raise MirrorTargetError(
                f"{MIRROR_TARGET_FIELD} level {level!r} is not one of "
                f"{[LEVEL_GRAPH, LEVEL_DATABASE]!r}"
            )
    return MirrorTarget(mode=mode, name=name, level=level)


def validate_target_name(name: str) -> str:
    """Validate a dedicated target name against the strictest store's rules."""
    rendered = str(name or "").strip()
    if not _NAME_RE.fullmatch(rendered):
        raise MirrorTargetError(
            f"{MIRROR_TARGET_FIELD} name {rendered!r} is not a portable graph "
            "name ([A-Za-z_][A-Za-z0-9_.-]{0,62})"
        )
    return rendered


def resolve_mirror_target(
    raw: Any,
    *,
    backend_type: str,
    named_selector: str | None,
    default_name: str | None = None,
    supported_levels: tuple[str, ...] = (),
) -> MirrorTarget:
    """Resolve one backend's mirror target — the single resolution point.

    Args:
        raw: the connection spec's ``mirror_target`` value (or ``None``).
        backend_type: the backend token, for error messages.
        named_selector: the native selector the connection explicitly names
            (``database`` / ``db_name`` / ``graph_name``), or ``None`` when the
            connection names none.
        default_name: the store's own default unit name, when it has one
            (``"agent_graph"`` for AGE/FalkorDB, ``"agent_kg"`` for Stardog).
            ``None`` for a store with a genuine implicit default (Neo4j).
        supported_levels: the isolation levels this store has. A single-level
            store passes ``()``; Stardog passes both.

    Returns:
        The resolved :class:`MirrorTarget`.

    Raises:
        MirrorTargetError: on a declaration this backend cannot honor —
            including a connection that names its selector *and* declares a
            conflicting target, which is never silently resolved one way.
    """
    declared = parse_mirror_target(raw)
    named = str(named_selector or "").strip() or None

    if declared is None:
        # Nothing declared. A connection that names its selector keeps that unit
        # verbatim (an existing deployment is untouched); a connection that names
        # nothing is on the instance default and therefore guarded.
        if named:
            return MirrorTarget(mode=MODE_NAMED, name=named)
        return MirrorTarget(mode=MODE_DEFAULT, name=default_name)

    level = declared.level
    if level is not None and level not in supported_levels:
        raise MirrorTargetError(
            f"{backend_type} has no {level!r} isolation level; "
            f"supported: {list(supported_levels) or ['(single level)']!r}"
        )
    if level is None and supported_levels:
        level = supported_levels[0]

    if declared.is_dedicated:
        # "Dedicate a graph" is the operator's explicit, later decision, so it
        # SUPERSEDES a named selector on a single-level store rather than
        # conflicting with it — otherwise a store whose connection profile
        # requires a graph name (AGE, FalkorDB) could never dedicate at all. It
        # is announced, never silent. On a two-level store a named database plus
        # a dedicated named graph are both honored, so nothing is superseded.
        name = validate_target_name(declared.name or DEDICATED_MIRROR_NAME)
        if named and level != LEVEL_GRAPH and named != name:
            logger.info(
                "%s: %s=dedicated supersedes the connection's own target %r; "
                "mirroring writes %r instead",
                backend_type,
                MIRROR_TARGET_FIELD,
                named,
                name,
            )
        return MirrorTarget(mode=MODE_DEDICATED, name=name, level=level)

    # ``default`` / ``default-overwrite`` next to a named selector is a genuine
    # contradiction — "use the instance default" and "use <name>" cannot both
    # hold. Refuse rather than pick a winner.
    if named:
        raise MirrorTargetError(
            f"{backend_type} connection names its target ({named!r}) and also "
            f"declares {MIRROR_TARGET_FIELD}={declared.mode!r}; these disagree. "
            f"Drop one: remove the explicit selector to write the instance "
            f"default, or drop {MIRROR_TARGET_FIELD} to keep writing {named!r}."
        )
    return MirrorTarget(mode=declared.mode, name=default_name, level=level)


# ── Two-level (database + named graph) helpers ───────────────────────────────


def database_for_target(target: MirrorTarget, configured: str) -> str:
    """The database a two-level store opens for ``target``.

    Shared by BOTH Stardog backends (the SPARQL data backend and the OWL
    reasoning backend) so their database selection cannot drift apart.
    """
    if target.is_dedicated and target.level == LEVEL_DATABASE and target.name:
        return target.name
    if target.mode == MODE_NAMED and target.name:
        return target.name
    return configured


def graph_iri_for_target(target: MirrorTarget) -> str | None:
    """The dedicated named-graph IRI for ``target``, or ``None``.

    Only a dedicated target at :data:`LEVEL_GRAPH` has one; every other mode
    leaves the store's existing graph routing alone.
    """
    if target.is_dedicated and target.level == LEVEL_GRAPH and target.name:
        return f"{MIRROR_GRAPH_PREFIX}{target.name}"
    return None


# ── The guard ────────────────────────────────────────────────────────────────


def _locator(backend: Any, target: MirrorTarget) -> str:
    """The backend's own phrase for its target, falling back to the generic one."""
    describe = getattr(backend, "mirror_target_locator", None)
    if callable(describe):
        try:
            return str(describe())
        except Exception as exc:  # noqa: BLE001 — a description must never mask the refusal it is describing; the real cause is reported by the caller
            logger.warning(
                "mirror target locator for %s failed (%s: %s); using the generic "
                "description",
                type(backend).__name__,
                type(exc).__name__,
                exc,
            )
    return target.describe()


def _refusal_message(mirror: str, locator: str, reason: str) -> str:
    """The one actionable refusal text (CONCEPT:AU-KG.backend.mirror-nonempty-default-guard)."""
    return (
        f"mirror {mirror!r} REFUSED: its resolved target is {locator}, and "
        f"{reason} Mirroring would write the knowledge graph into data this "
        f"instance already holds, so nothing has been written and the mirror is "
        f"not attached. Choose one in this connection's kg_connections entry:\n"
        f'  "{MIRROR_TARGET_FIELD}": "{MODE_DEDICATED}"'
        f"  -> mirror into a graph dedicated to mirroring "
        f"(named {DEDICATED_MIRROR_NAME!r}, created if absent)\n"
        f'  "{MIRROR_TARGET_FIELD}": '
        f'{{"mode": "{MODE_DEDICATED}", "name": "<graph>"}}'
        f"  -> the same, under a name you choose\n"
        f'  "{MIRROR_TARGET_FIELD}": "{MODE_DEFAULT_OVERWRITE}"'
        f"  -> explicitly accept writing into the non-empty instance default"
    )


def preflight_mirror_target(mirror: str, backend: Any) -> None:
    """Prepare and police one mirror's target before it is attached to the fan-out.

    CONCEPT:AU-KG.backend.mirror-nonempty-default-guard. Two steps, in order:

    1. **Idempotent creation.** A dedicated target is created if absent
       (``ensure_mirror_target``) — reusing each store's existing create path
       (``ag_catalog.create_graph`` / ``_ensure_database`` / ``CREATE DATABASE
       IF NOT EXISTS``); a store whose graphs spring into being on first write
       (FalkorDB, a SPARQL named graph) does nothing.
    2. **The non-empty-default refusal.** When the target is the instance
       default and is not provably empty, raise :class:`MirrorTargetRefused`.

    A backend that does not expose a ``mirror_target`` is not part of this
    concept (LadybugDB, Fuseki) and passes through untouched.

    Raises:
        MirrorTargetRefused: the target is a non-empty instance default, or its
            emptiness could not be proven (the guard fails closed).
    """
    target = getattr(backend, "mirror_target", None)
    if not isinstance(target, MirrorTarget):
        return

    ensure = getattr(backend, "ensure_mirror_target", None)
    if target.is_dedicated and callable(ensure):
        ensure()

    if not target.guarded:
        logger.info("mirror %r targets %s", mirror, _locator(backend, target))
        return

    probe = getattr(backend, "mirror_target_has_data", None)
    locator = _locator(backend, target)
    if not callable(probe):
        raise MirrorTargetRefused(
            _refusal_message(
                mirror,
                locator,
                f"{type(backend).__name__} cannot prove that target empty (it "
                "exposes no mirror_target_has_data probe).",
            )
        )
    try:
        occupied = bool(probe())
    except Exception as exc:  # noqa: BLE001 — fail CLOSED: an unprovable target is treated as occupied, with the cause chained
        raise MirrorTargetRefused(
            _refusal_message(
                mirror,
                locator,
                f"whether it already holds data could not be determined "
                f"({type(exc).__name__}: {exc}).",
            )
        ) from exc
    if occupied:
        raise MirrorTargetRefused(
            _refusal_message(mirror, locator, "it ALREADY CONTAINS DATA.")
        )
    logger.info("mirror %r targets %s, verified empty", mirror, locator)


def cypher_target_has_data(backend: Any) -> bool:
    """Shared non-empty probe for the Cypher stores (Neo4j / AGE / FalkorDB).

    One read of the mirror's *own* target — the backend has already bound its
    session/graph selector, so this runs against exactly the unit the guard is
    asking about. Kept here so the three cannot drift apart.
    """
    rows = backend.execute("MATCH (n) RETURN count(n) AS mirror_target_nodes")
    for row in rows or []:
        for value in row.values():
            try:
                if int(value) > 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False
