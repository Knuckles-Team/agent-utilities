# pg-age-full — ONE Postgres image with all three durable-tier extensions:
#   • Apache AGE   — native openCypher property graph (backend: "age")
#   • pgvector     — embedding / semantic search
#   • pg_search    — ParadeDB BM25 full-text (the hardest to install)
#
# This is the image services/pg-age/compose.yml already references
# (registry.arpa/pg-age) but which had no Dockerfile in-repo. We base on the
# ParadeDB image (pg_search + pgvector ship here) and compile Apache AGE for the
# same Postgres major. The lightweight alternative — AGE + pgvector only, no
# BM25 — is docker/pg-age/Dockerfile.
#
# ── BUILD-TIME RISK (verify before pinning) ────────────────────────────────
# AGE and ParadeDB must agree on the Postgres MAJOR. Set PG_MAJOR to the major
# the chosen `paradedb/paradedb` tag ships, and AGE_BRANCH to the matching
# Apache AGE release branch (release tags look like `PG16/v1.5.0-rc0`). If a
# compatible pair does not exist for your ParadeDB tag, fall back to TWO Postgres
# instances (one AGE+pgvector via docker/pg-age, one ParadeDB) — the setup
# provisioner supports that via separate DSNs.
#
# hadolint ignore=DL3007  # tag is deliberately operator-chosen (see PG_MAJOR note above), not pinned in-repo
FROM paradedb/paradedb:latest

USER root

# Pin to the Postgres major your ParadeDB base ships and the matching AGE branch.
ARG PG_MAJOR=17
ARG AGE_BRANCH=PG17/v1.5.0-rc0

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential git ca-certificates flex bison \
        "postgresql-server-dev-${PG_MAJOR}"; \
    git clone --depth 1 --branch "${AGE_BRANCH##*/}" https://github.com/apache/age.git /tmp/age \
        || git clone --depth 1 https://github.com/apache/age.git /tmp/age; \
    make -C /tmp/age PG_CONFIG="/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config" install; \
    rm -rf /tmp/age; \
    apt-get purge -y --auto-remove build-essential git flex bison "postgresql-server-dev-${PG_MAJOR}"; \
    rm -rf /var/lib/apt/lists/*

USER postgres
