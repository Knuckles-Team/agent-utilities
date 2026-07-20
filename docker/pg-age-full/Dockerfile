# pg-age-full — ONE Postgres image with all three durable-tier extensions:
#   • Apache AGE   — native openCypher property graph (backend: "age")
#   • pgvector     — embedding / semantic search
#   • pg_search    — ParadeDB BM25 full-text (the hardest to install)
#
# This is the image services/pg-age/compose.yml already references
# (a deployment-defined image reference) but which had no Dockerfile in-repo. We base on the
# ParadeDB image (pg_search + pgvector ship here) and compile Apache AGE for the
# same Postgres major. The lightweight alternative — AGE + pgvector only, no
# BM25 — is docker/pg-age/Dockerfile.
#
# ── BUILD-TIME RISK (verify before pinning) ────────────────────────────────
# AGE and ParadeDB must agree on the Postgres MAJOR. Set PG_MAJOR to the major
# the chosen `paradedb/paradedb` digest ships, and AGE_REV to a reviewed commit
# Apache AGE release branch (release tags look like `PG16/v1.5.0-rc0`). If a
# compatible pair does not exist for your ParadeDB tag, fall back to TWO Postgres
# instances (one AGE+pgvector via docker/pg-age, one ParadeDB) — the setup
# provisioner supports that via separate DSNs.
#
# Keep the reviewed manifest digest aligned with PG_MAJOR and AGE_REV.
FROM paradedb/paradedb@sha256:87fcc5658642670712e6831c6afc578927ac1ff6ef4fa77c41c8a491cc13ccb9

USER root

# Pin to the Postgres major your ParadeDB base ships and an immutable AGE revision.
ARG PG_MAJOR=17
ARG AGE_REV=e1467f12e0b1d15dd35d3ab93f057a7112d425b8

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential git ca-certificates flex bison \
        "postgresql-server-dev-${PG_MAJOR}"; \
    git init /tmp/age; \
    git -C /tmp/age remote add origin https://github.com/apache/age.git; \
    git -C /tmp/age fetch --depth 1 origin "${AGE_REV}"; \
    git -C /tmp/age checkout --detach FETCH_HEAD; \
    make -C /tmp/age PG_CONFIG="/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config" install; \
    rm -rf /tmp/age; \
    apt-get purge -y --auto-remove build-essential git flex bison "postgresql-server-dev-${PG_MAJOR}"; \
    rm -rf /var/lib/apt/lists/*

USER postgres
