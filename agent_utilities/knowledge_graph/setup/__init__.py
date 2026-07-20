"""Database environment provisioning — wire Stardog + pg-age from credentials.

A thin orchestration layer that *composes* the existing graph/ontology machinery
(KG-2.6 ontology distribution, KG-2.7 tiered durable backfill, KG-2.63 named
connections, KG-2.74 fanout mirroring) into one credentials-driven flow so an
operator can stand up a prod (Stardog) or dev (local SPARQL) environment and
durably backfill the graph into Apache AGE. No new graph logic lives here.
"""

from .database_environment import (
    backfill_to_age,
    configure_backend,
    publish_ontology,
    register_stardog_mirror,
    setup_environment,
    verify_postgres,
    verify_sparql,
)

__all__ = [
    "backfill_to_age",
    "configure_backend",
    "publish_ontology",
    "register_stardog_mirror",
    "setup_environment",
    "verify_postgres",
    "verify_sparql",
]
