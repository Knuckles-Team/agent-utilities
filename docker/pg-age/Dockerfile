# pg-age (Apache AGE + pgvector) — real openCypher-on-Postgres for agent-utilities.
#
# CONCEPT:KG-2.7 — the PostgreSQL graph tier runs Apache AGE (Cypher executed
# natively via the `cypher()` function, retiring the bounded regex transpiler) and
# keeps pgvector for embeddings. apache/age ships Postgres 18 (Debian trixie, PGDG)
# so pgvector installs from apt — no compile.
FROM apache/age:latest

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-18-pgvector \
    && rm -rf /var/lib/apt/lists/*
USER postgres
