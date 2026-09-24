"""``scripts/build_concepts_yaml.py`` keeps a marker's adjacent tail out of docs.

The OKF-CIS grammar itself (and its own tests) moved to
``repository_manager.governance.concept_hierarchy`` (OQ-3); this is the
agent-utilities registry generator's contract on top of it.
"""

from __future__ import annotations

import pytest

from scripts import build_concepts_yaml


@pytest.mark.parametrize("tail", ["/2.15/2.34", "_legacy_suffix", "UppercaseSuffix"])
def test_concept_generator_excludes_marker_tail_from_docs(
    tmp_path, monkeypatch: pytest.MonkeyPatch, tail: str
) -> None:
    source_dir = tmp_path / "agent_utilities"
    source_dir.mkdir()
    source = source_dir / "marker.py"
    source.write_text(
        f'"""CONCEPT:AU-KG.ingest.entropy-dedup{tail}) — Durable description."""\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(build_concepts_yaml, "ROOT", tmp_path)
    monkeypatch.setattr(build_concepts_yaml, "SRC_DIR", source_dir)

    concepts = build_concepts_yaml.collect()

    entry = concepts["AU-KG.ingest.entropy-dedup"]
    assert entry["name"] == "Durable description"
    assert entry["doc"] == "Durable description"
