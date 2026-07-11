-- CONCEPT:KG-2.61 — Postgres Row-Level Security for tenant isolation (L3).
--
-- Defense-in-depth BENEATH the KG-2.58 named-graph partition: even a Cypher/SQL
-- statement that forgets the tenant predicate cannot read another org's rows,
-- because Postgres itself filters them. The application sets a per-session GUC
-- ``app.tenant_id`` (see PostgreSQLBackend.set_request_tenant):
--
--   * set to an org id  → that org's rows + commons (tenant_id '' / NULL)
--   * unset or empty     → unrestricted (platform-admin / system / legacy path)
--
-- FORCE ROW LEVEL SECURITY is required because the app's DB role usually owns
-- the tables, and owners otherwise bypass RLS.
--
-- This script is idempotent. Run it once after the schema exists, as the table
-- owner. It templates over every base table in the current schema (node tables
-- + kg_edges); re-run after introducing new node types, or call
-- PostgreSQLBackend.enable_row_level_security() which regenerates the same DDL.
--
-- AU-P0-5: this loop is genuinely generic — it walks EVERY table in
-- current_schema(), not just the KG's own node/edge tables. So when the
-- unified state-store pool (agent_utilities.core.state_store, STATE_DB_URI)
-- shares this schema/database, the SAME script also RLS-secures its tables
-- (sessions/turns/dispatch_workers, the usage-analytics tables, durable-exec
-- checkpoints, the Postgres task queue) — no separate migration needed, they
-- just need to exist by the time this runs. Both connection pools now set the
-- GUC on every checkout before the caller's SQL runs:
--   * PostgreSQLBackend._conn() -> PostgreSQLBackend._scope_tenant()
--   * agent_utilities.core.state_store.open_state_connection() (Postgres path)
--     -> state_store.set_state_tenant()
-- If the state-store lives in a DIFFERENT database/schema than the KG
-- backend, run this script there too (it needs no KG-specific objects).

DO $$
DECLARE
    t  text;
    q  text;
    cond text;
BEGIN
    FOR t IN
        SELECT tablename FROM pg_tables
        WHERE schemaname = current_schema()
          AND tablename NOT LIKE 'pg_%'
          AND tablename NOT LIKE 'sql_%'
    LOOP
        q := format('%I', t);
        cond := format(
            '(tenant_id = current_setting(%L, true) '
            'OR tenant_id IS NULL OR tenant_id = %L '
            'OR current_setting(%L, true) IS NULL '
            'OR current_setting(%L, true) = %L)',
            'app.tenant_id', '', 'app.tenant_id', 'app.tenant_id', ''
        );

        EXECUTE format('ALTER TABLE %s ADD COLUMN IF NOT EXISTS tenant_id TEXT', q);
        EXECUTE format('ALTER TABLE %s ENABLE ROW LEVEL SECURITY', q);
        EXECUTE format('ALTER TABLE %s FORCE ROW LEVEL SECURITY', q);
        EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %s', q);
        EXECUTE format(
            'CREATE POLICY tenant_isolation ON %s USING %s WITH CHECK %s',
            q, cond, cond
        );
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON %s(tenant_id)',
            'idx_' || t || '_tenant', q
        );
    END LOOP;
END $$;

-- Verify (run manually): the platform/admin path sees all, a scoped session
-- sees only its tenant + commons:
--   SET app.tenant_id = '';        -- unrestricted
--   SET app.tenant_id = 'acme';    -- acme rows + commons only
