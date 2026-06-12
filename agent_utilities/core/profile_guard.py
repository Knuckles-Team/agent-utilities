#!/usr/bin/python
"""Production profile guard.

Provides a runtime validator that refuses to let "toy"/single-node defaults
through when the deployment declares itself a production profile.

A deployment is treated as production when the ``APP_PROFILE`` environment
variable (case-insensitive) is one of :data:`PROD_PROFILE_VALUES` (``prod`` /
``production``). In that mode several settings that are perfectly fine for local
development become unsafe because they pin the system to a single process or a
single machine and therefore cannot scale or survive a restart:

* ``graph_persistence_type`` of ``file`` / ``sqlite`` -> single-host, non-shardable.
* ``GRAPH_BACKEND`` of ``memory``/``file``/``ladybug`` (or ``tiered`` with an
  embedded LadybugDB L2) -> single-host; prod needs a Postgres L2 / DSN.
* ``a2a_broker`` of ``in-memory`` -> messages lost on restart, no cross-node fan-out.
* ``a2a_storage`` of ``in-memory`` -> agent cards/state lost on restart.

:func:`assert_production_safe` raises :class:`ProductionProfileError` listing every
offending setting so an operator can fix them all at once, instead of discovering
them one redeploy at a time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_utilities.core.config import setting

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent_utilities.core.config import AgentConfig

#: Environment-variable name that selects the deployment profile.
PROFILE_ENV_VAR = "APP_PROFILE"

#: Values of ``APP_PROFILE`` that mean "this is production".
PROD_PROFILE_VALUES = frozenset({"prod", "production"})

#: ``graph_persistence_type`` values that are single-host / non-production.
UNSAFE_GRAPH_PERSISTENCE = frozenset({"file", "sqlite"})

#: ``GRAPH_BACKEND`` values that are single-host / embedded and therefore not
#: production-durable on their own. ``tiered`` is evaluated separately because
#: its durability depends on the resolved L2 tier.
UNSAFE_GRAPH_BACKEND = frozenset(
    {"memory", "file", "epistemic_graph", "ladybug", "falkordb"}
)

#: ``a2a_broker`` values that are single-process / non-production.
UNSAFE_A2A_BROKER = frozenset({"in-memory", "in_memory", "inmemory", "memory"})

#: ``a2a_storage`` values that are single-process / non-production.
UNSAFE_A2A_STORAGE = frozenset({"in-memory", "in_memory", "inmemory", "memory"})


class ProductionProfileError(RuntimeError):
    """Raised when a production profile is configured with unsafe toy settings.

    The :attr:`offending` attribute holds the list of human-readable problem
    descriptions, one per offending setting.
    """

    def __init__(self, offending: list[str], profile: str) -> None:
        self.offending = list(offending)
        self.profile = profile
        joined = "\n".join(f"  - {item}" for item in self.offending)
        super().__init__(
            f"Configuration is not production-safe under {PROFILE_ENV_VAR}={profile!r}.\n"
            f"The following settings use single-host/in-memory defaults that cannot "
            f"scale or survive a restart:\n{joined}\n"
            f"Set production-grade backends (e.g. graph_persistence_type='postgresql', "
            f"a2a_broker='kafka'/'nats', a2a_storage='postgresql'/'redis')."
        )


def is_production_profile(profile: str | None = None) -> bool:
    """Return True when the active deployment profile is production.

    :param profile: Explicit profile value. When ``None`` the ``APP_PROFILE``
        environment variable is consulted.
    """
    if profile is None:
        profile = setting(PROFILE_ENV_VAR)
    if not profile:
        return False
    return profile.strip().lower() in PROD_PROFILE_VALUES


def collect_production_violations(config: AgentConfig) -> list[str]:
    """Return a list of production-safety violations for ``config``.

    The list is empty when the configuration is production-safe. This function
    does *not* consult ``APP_PROFILE``; it always evaluates the rules, so callers
    can report problems regardless of the active profile.
    """
    offending: list[str] = []

    gpt = (getattr(config, "graph_persistence_type", "") or "").strip().lower()
    if gpt in UNSAFE_GRAPH_PERSISTENCE:
        offending.append(
            f"graph_persistence_type={gpt!r} is single-host and non-shardable; "
            f"use a distributed backend (e.g. 'postgresql')."
        )

    # Graph backend selection (mirrors ``create_backend`` resolution). The
    # out-of-box default is the zero-infra ``tiered`` backend whose L2 is an
    # embedded LadybugDB — fine for dev, but single-host for prod. Require a
    # durable L2 (a Postgres DSN or graph_backend_l2=postgresql) in production.
    graph_backend = (
        (getattr(config, "graph_backend", "tiered") or "tiered").strip().lower()
    )
    has_pg_dsn = bool(
        (getattr(config, "graph_db_uri", None) or "").strip()
        or (getattr(config, "pggraph_dsn", None) or "").strip()
    )
    if graph_backend == "tiered":
        l2 = (getattr(config, "graph_backend_l2", None) or "").strip().lower() or (
            "postgresql" if has_pg_dsn else "ladybug"
        )
        if l2 not in ("postgres", "postgresql", "pggraph"):
            offending.append(
                f"graph_backend=tiered resolves to a single-host L2 ({l2!r}); set "
                f"graph_db_uri to a Postgres/pggraph DSN (or "
                f"graph_backend_l2='postgresql') for a durable, shardable prod tier."
            )
    elif graph_backend in UNSAFE_GRAPH_BACKEND:
        offending.append(
            f"graph_backend={graph_backend!r} is single-host/embedded and cannot "
            f"survive a restart across nodes; use 'postgresql' or "
            f"'tiered' with a Postgres L2 (graph_db_uri) for prod."
        )

    broker = (getattr(config, "a2a_broker", "") or "").strip().lower()
    if broker in UNSAFE_A2A_BROKER:
        offending.append(
            f"a2a_broker={broker!r} loses messages on restart and cannot fan out "
            f"across nodes; use a real broker (e.g. 'kafka' or 'nats')."
        )

    storage = (getattr(config, "a2a_storage", "") or "").strip().lower()
    if storage in UNSAFE_A2A_STORAGE:
        offending.append(
            f"a2a_storage={storage!r} loses agent state on restart; use durable "
            f"storage (e.g. 'postgresql' or 'redis')."
        )

    # Plan 08 Synergy 2: the single reactive ledger (telemetry, KG write-back,
    # UI fan-out, replay) requires a distributed Kafka/Redpanda broker. Without
    # kafka_bootstrap_servers the event backend silently falls back to the
    # single-process, non-durable in-memory EventBus.
    kafka = (getattr(config, "kafka_bootstrap_servers", "") or "").strip()
    if not kafka:
        offending.append(
            "kafka_bootstrap_servers is unset; the reactive event ledger falls "
            "back to the single-process in-memory EventBus (no durability, no "
            "cross-node fan-out). Set it to a Redpanda/Kafka cluster for prod."
        )

    return offending


def assert_production_safe(config: AgentConfig, *, profile: str | None = None) -> None:
    """Validate ``config`` against production-safety rules.

    When the active profile is *not* production this is a no-op (dev/default
    profiles are unaffected). When it *is* production and any offending setting
    is found, raise :class:`ProductionProfileError` listing every offender.

    :param config: The :class:`AgentConfig` instance to validate.
    :param profile: Optional explicit profile override (otherwise ``APP_PROFILE``).
    :raises ProductionProfileError: If running as production with unsafe settings.
    """
    active_profile = profile if profile is not None else setting(PROFILE_ENV_VAR, "")
    if not is_production_profile(active_profile):
        return

    offending = collect_production_violations(config)
    if offending:
        raise ProductionProfileError(offending, active_profile)
