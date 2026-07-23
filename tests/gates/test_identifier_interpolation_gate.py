"""Meta-test: the identifier-interpolation gate trips on a broken fixture and
passes clean on a guarded one. A gate that can't fail is not a gate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_identifier_interpolation.py"


def _run(target: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), target],
        capture_output=True,
        text=True,
    )


def test_gate_trips_on_the_original_ladybug_prune_shape(tmp_path):
    """Reproduces the exact bug this task fixed: an unvalidated criteria value
    is spliced into a Cypher label fragment, which is then embedded in a
    MATCH ... DETACH DELETE. The colon lives inside the interpolated
    fragment's own value (not literal text this f-string's own tokens show),
    which is what makes this shape easy to miss with a naive per-token scan.
    """
    (tmp_path / "bad_backend.py").write_text(
        "class FakeBackend:\n"
        "    def prune(self, criteria):\n"
        '        node_type = criteria.get("node_type", "")\n'
        '        label = f":{node_type}" if node_type else ""\n'
        '        query = f"MATCH (n{label}) WHERE 1=1 DETACH DELETE n"\n'
        "        return self.execute(query, {})\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout
    assert "bad_backend.py" in result.stdout
    assert "label" in result.stdout


def test_gate_passes_when_the_fragment_is_validated_first(tmp_path):
    (tmp_path / "good_backend.py").write_text(
        "from agent_utilities.security.identifiers import validate_identifier\n\n\n"
        "class FakeBackend:\n"
        "    def prune(self, criteria):\n"
        '        node_type = criteria.get("node_type", "")\n'
        "        safe = validate_identifier(node_type, kind='label')\n"
        '        label = f":{safe}" if node_type else ""\n'
        '        query = f"MATCH (n{label}) WHERE 1=1 DETACH DELETE n"\n'
        "        return self.execute(query, {})\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout


def test_gate_trips_on_unguarded_create_table(tmp_path):
    (tmp_path / "bad_table.py").write_text(
        "class Store:\n"
        "    def ensure(self, cur, table_name):\n"
        '        cur.execute(f\'CREATE TABLE IF NOT EXISTS "{table_name}" (id TEXT)\')\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout
    assert "table_name" in result.stdout


def test_gate_passes_on_validated_create_table(tmp_path):
    (tmp_path / "good_table.py").write_text(
        "from agent_utilities.security.identifiers import validate_sql_identifier\n\n\n"
        "class Store:\n"
        "    def ensure(self, cur, table_name):\n"
        "        table_name = validate_sql_identifier(table_name, kind='table')\n"
        '        cur.execute(f\'CREATE TABLE IF NOT EXISTS "{table_name}" (id TEXT)\')\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout


def test_gate_ignores_ordinary_value_interpolation(tmp_path):
    """A f-string with a query keyword but no identifier-shaped slot (bound
    values / predicates only) must not be flagged — the gate's whole point is
    to distinguish identifier positions from value positions."""
    (tmp_path / "fine.py").write_text(
        "def query_lineage(limit):\n"
        '    where = ["n.kind = $kind"]\n'
        '    return f"MATCH (n) WHERE {\' AND \'.join(where)} LIMIT {int(limit)}"\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout


def test_gate_recognizes_local_helper_indirection(tmp_path):
    """A file-local sanitizer that itself calls a recognized guard counts as
    guarding its own callers (the ``sync.py`` / ``_safe_graph_identifier``
    shape) — one hop of indirection, no need to call the primitive at every
    single call site."""
    (tmp_path / "indirect.py").write_text(
        "import re\n"
        "from agent_utilities.security.identifiers import CYPHER_IDENTIFIER_RE\n\n\n"
        "def _safe(value, default=''):\n"
        "    rendered = re.sub(r'\\W+', '_', str(value or default)).strip('_')\n"
        "    return rendered if CYPHER_IDENTIFIER_RE.fullmatch(rendered) else default\n\n\n"
        "def sync_one(db, raw_label):\n"
        "    label = _safe(raw_label)\n"
        '    db.execute_batch(f"MERGE (n:{label} {{id: $id}})", [])\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout


def test_gate_recognizes_module_level_validated_constant(tmp_path):
    """A module constant assigned from a recognized guard is pre-validated
    everywhere it's later interpolated (the ``manifest.py`` / ``_LABEL``
    shape) — no repeated per-call-site validation needed."""
    (tmp_path / "constant.py").write_text(
        "from agent_utilities.security.identifiers import validate_identifier\n\n"
        "_LABEL = validate_identifier('IngestManifest', kind='label')\n\n\n"
        "def get(backend, node_id):\n"
        '    return backend.execute(f"MATCH (m:{_LABEL} {{id: $id}}) RETURN m", {})\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout


def test_gate_passes_on_the_repo_default_scope():
    """The gate's own default scope (mirror backends + the centralized
    ETL/extraction/ingestion/pipeline modules) must stay green in this repo —
    this is the regression lock for the actual fix this task shipped."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
