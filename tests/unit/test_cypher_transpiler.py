"""Unit tests for the Cypher-to-SQL transpiler."""

from typing import Any

from agent_utilities.knowledge_graph.backends.cypher_transpiler import (
    QueryType,
    transpile,
)

KNOWN_TABLES = {"Agent", "Tool", "Memory", "Code", "Skill", "Article", "Concept"}


class TestRelationshipTraversal:
    """Single-hop traversal (s:L1)-[:R]->(t:L2) → JOIN over kg_edges."""

    def test_count_traversal(self):
        cypher = "MATCH (s:Memory)-[r:MENTIONS]->(t:Concept) RETURN count(r) as c"
        tq = transpile(cypher, {}, KNOWN_TABLES)
        assert tq.query_type == QueryType.COUNT
        assert '"Memory" s' in tq.sql and '"Concept" t' in tq.sql
        assert "kg_edges e ON e.source_id = s.id" in tq.sql
        assert "e.rel_type = %s" in tq.sql and tq.params == ["MENTIONS"]
        assert tq.return_columns == ["c"]

    def test_count_distinct_target(self):
        cypher = (
            "MATCH (s:Memory)-[r:MENTIONS]->(t:Concept) RETURN count(DISTINCT t) as c"
        )
        tq = transpile(cypher, {}, KNOWN_TABLES)
        assert tq.query_type == QueryType.COUNT
        assert "count(DISTINCT t.id)" in tq.sql

    def test_property_projection_with_limit(self):
        cypher = (
            "MATCH (s:Memory)-[r:MENTIONS]->(t:Concept) "
            "RETURN s.id as m, t.name as concept LIMIT 5"
        )
        tq = transpile(cypher, {}, KNOWN_TABLES)
        assert tq.query_type == QueryType.SELECT
        assert 's."id" AS m' in tq.sql and 't."name" AS concept' in tq.sql
        assert "LIMIT 5" in tq.sql
        assert tq.return_columns == ["m", "concept"]


class TestCreateNode:
    def test_basic_create(self):
        cypher = "CREATE (n:Agent {id: $id, name: $name, type: $type})"
        params = {"id": "agent-1", "name": "Test", "type": "agent"}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type == QueryType.INSERT
        assert tq.target_table == "Agent"
        assert '"Agent"' in tq.sql
        assert "INSERT INTO" in tq.sql
        assert "ON CONFLICT" in tq.sql
        assert tq.params == ["agent-1", "Test", "agent"]

    def test_create_strips_nul_and_serializes_dict(self):
        """CONCEPT:KG-2.9 — node-property values are sanitized for Postgres binding:
        NUL (0x00) is stripped from strings (TEXT rejects it) and dict/list values are
        JSON-encoded (psycopg cannot adapt a bare dict to %s). Was dropping Article
        content (NUL) + Document rows (dict) on ingest."""
        import json

        cypher = "CREATE (n:Document {id: $id, content: $content, record: $record})"
        params = {
            "id": "doc-1",
            "content": "hello\x00world",
            "record": {"k": "v", "n": 1},
        }
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.params[0] == "doc-1"
        assert tq.params[1] == "helloworld"  # NUL stripped
        assert tq.params[2] == json.dumps({"k": "v", "n": 1})  # dict → JSON string


class TestMatchById:
    def test_match_with_set(self):
        cypher = "MATCH (n:Tool) WHERE n.id = $id SET n.name = $name RETURN n.id"
        params = {"id": "tool-1", "name": "Updated"}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type == QueryType.UPDATE
        assert tq.target_table == "Tool"
        assert "UPDATE" in tq.sql
        assert "Updated" in tq.params
        assert "tool-1" in tq.params

    def test_match_return_only(self):
        cypher = "MATCH (n:Memory) WHERE n.id = $id RETURN n"
        params = {"id": "mem-1"}
        tq = transpile(cypher, params, KNOWN_TABLES)
        # Should be a SELECT since there's no SET
        assert tq.query_type == QueryType.SELECT


class TestLabelLookup:
    def test_label_function(self):
        cypher = "MATCH (n) WHERE n.id = $id RETURN label(n) as lbl"
        params = {"id": "some-id"}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type == QueryType.LABEL_LOOKUP
        assert "UNION ALL" in tq.sql
        assert "lbl" in tq.sql


class TestContainsSearch:
    def test_tolower_contains(self):
        cypher = (
            "MATCH (n) WHERE (toLower(n.name) CONTAINS toLower($k0)) "
            "AND coalesce(n.status, '') <> 'ARCHIVED' RETURN n"
        )
        params = {"k0": "search_term"}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type == QueryType.SELECT
        assert "LIKE" in tq.sql


class TestMergeRelationship:
    def test_merge_edge(self):
        cypher = (
            "MATCH (s:Agent {id: $sid}), (t:Tool {id: $tid}) "
            "MERGE (s)-[r:PROVIDES]->(t)"
        )
        params = {"sid": "agent-1", "tid": "tool-1"}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type == QueryType.UPSERT_EDGE
        assert "kg_edges" in tq.sql
        assert "ON CONFLICT" in tq.sql


class TestMergeNode:
    def test_merge_node_upsert(self):
        # The graph-writer daemon + sync phase persist nodes via this pattern;
        # it must become an INSERT ... ON CONFLICT (id) DO UPDATE upsert.
        cypher = (
            "MERGE (n:Code {id: $id}) "
            "SET n.file_path = $props_file_path, n.type = $props_type"
        )
        params = {
            "id": "code-1",
            "props_file_path": "/a/b.py",
            "props_type": "symbol",
        }
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type == QueryType.INSERT
        assert tq.target_table == "Code"
        assert 'INSERT INTO "Code"' in tq.sql
        assert "ON CONFLICT (id) DO UPDATE SET" in tq.sql
        assert '"file_path" = EXCLUDED."file_path"' in tq.sql
        assert tq.params == ["code-1", "/a/b.py", "symbol"]

    def test_merge_node_no_set_is_do_nothing(self):
        tq = transpile("MERGE (n:Task {id: $id})", {"id": "job-1"}, KNOWN_TABLES)
        assert tq.query_type == QueryType.INSERT
        assert "ON CONFLICT (id) DO NOTHING" in tq.sql
        assert tq.params == ["job-1"]


class TestCountQuery:
    def test_count_pattern(self):
        cypher = "MATCH (t:Task {status: 'pending'}) RETURN count(t) as c"
        params: dict[str, Any] = {}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type in (QueryType.COUNT, QueryType.SELECT)


class TestDeleteQuery:
    def test_detach_delete(self):
        cypher = "MATCH (n:Memory) WHERE n.id = $id DETACH DELETE n"
        params = {"id": "mem-1"}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type == QueryType.DELETE
        assert "DELETE" in tq.sql


class TestLimitAndOrder:
    def test_limit_with_order(self):
        cypher = "MATCH (n:Agent) RETURN n ORDER BY n.name LIMIT 10"
        params: dict[str, Any] = {}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type == QueryType.SELECT
        assert "ORDER BY" in tq.sql
        assert "LIMIT" in tq.sql


class TestUnlabeledIdLookupPlaceholderParity:
    """An unlabeled ``WHERE n.id = $id`` lookup fans a UNION ALL across every
    known table; each branch repeats the ``%s`` placeholder, so the bound params
    must be repeated to match — the regression behind the live error
    "the query has N placeholders but 1 parameters were passed". (CONCEPT:KG-2.9h)
    """

    def test_placeholder_count_equals_param_count(self):
        tq = transpile(
            "MATCH (n) WHERE n.id = $id RETURN n", {"id": "abc"}, KNOWN_TABLES
        )
        assert tq.query_type == QueryType.SELECT
        assert tq.sql.count("%s") == len(tq.params)
        # one bind per table, all the same id
        assert tq.params == ["abc"] * len(KNOWN_TABLES)
        # behaviour preserved: a UNION ALL fan-out across the known tables
        assert tq.sql.count("UNION ALL") == len(KNOWN_TABLES) - 1

    def test_parity_holds_with_trailing_limit(self):
        tq = transpile(
            "MATCH (n) WHERE n.id = $id RETURN n LIMIT $lim",
            {"id": "abc", "lim": 7},
            KNOWN_TABLES,
        )
        assert tq.sql.count("%s") == len(tq.params)
        # trailing LIMIT param sits after the repeated id binds
        assert tq.params == ["abc"] * len(KNOWN_TABLES) + [7]

    def test_union_branches_have_uniform_column_count(self):
        # Regression (CONCEPT:KG-2.7): a label-less ``RETURN n`` previously
        # projected ``SELECT *`` per branch. Heterogeneous node tables have
        # different widths (per-label schema drift / pgvector ``embedding`` /
        # ``tenant_id``), so the UNION branches mismatched and Postgres rejected
        # the whole query ("each UNION query must have the same number of
        # columns"), poisoning the connection's transaction. Every branch must
        # now project the SAME fixed base column set.
        tq = transpile(
            "MATCH (n) WHERE n.id = $id RETURN n", {"id": "abc"}, KNOWN_TABLES
        )
        # the unsafe star projection must be gone
        assert "SELECT *," not in tq.sql
        branches = tq.sql.split("UNION ALL")
        assert len(branches) == len(KNOWN_TABLES)
        # every branch must select the same number of columns (commas before FROM)
        col_counts = {b.upper().split("FROM")[0].count(",") for b in branches}
        assert len(col_counts) == 1, f"non-uniform UNION arity: {col_counts}"


class TestUnknownPattern:
    def test_unsupported_cypher(self):
        cypher = "CALL db.schema.visualization()"
        tq = transpile(cypher, {}, KNOWN_TABLES)
        assert tq.query_type == QueryType.UNKNOWN


class TestEdgeCases:
    def test_empty_params(self):
        cypher = "MATCH (n:Agent) RETURN n"
        tq = transpile(cypher, {}, KNOWN_TABLES)
        assert tq.query_type == QueryType.SELECT

    def test_none_params(self):
        cypher = "MATCH (n:Agent) RETURN n"
        tq = transpile(cypher, None, KNOWN_TABLES)
        assert tq.query_type == QueryType.SELECT

    def test_internal_params_stripped(self):
        cypher = "MATCH (n:Agent) WHERE n.id = $id RETURN n"
        params = {"id": "a1", "_clearance_level": 999}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert "_clearance_level" not in str(tq.params)
