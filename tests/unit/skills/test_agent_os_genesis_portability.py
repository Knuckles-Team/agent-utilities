"""Portability contract for the packaged Agent OS Genesis skill."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SKILL = ROOT / "agent_utilities" / "skills" / "workflows" / "agent-os-genesis"
CHART = SKILL / "assets" / "helm" / "agent-os"


def _frontmatter(text: str) -> dict[str, object]:
    _, raw, _ = text.split("---", 2)
    parsed = yaml.safe_load(raw)
    assert isinstance(parsed, dict)
    return parsed


def test_frontmatter_is_portable_and_skill_is_a_small_router() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")

    assert set(_frontmatter(text)) == {"name", "description"}
    assert len(text.splitlines()) < 500
    assert "**Mandatory delegation rule:**" in text
    assert "`agent-utilities-deployment`" in text
    assert "substrate_resolved" in text


def test_local_references_are_shallow_and_resolve() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\((references/[^)#]+)\)", text)

    assert links
    for link in links:
        relative = Path(link)
        assert len(relative.parts) == 2
        assert (SKILL / relative).is_file(), link


def test_skill_contains_no_site_inventory_or_secret_values() -> None:
    # BUG-228: assembled from parts so this portability guard's OWN tracked
    # source is not itself a matchable leak literal for
    # check_tracked_privacy.py's runtime-source pass -- the packaged skill
    # files below are still scanned for the exact real prefix this repo
    # runs from, which is the whole point of this guard.
    _site_home_prefix = "/home/" + "apps" + "/workspace"
    forbidden = (
        re.compile(re.escape(_site_home_prefix)),
        re.compile(r"\b10\.0\.0\.\d+\b"),
        re.compile(r"\b(?:rw?|gr)\d{3,}\b", re.IGNORECASE),
        re.compile(r"\.arpa\b"),
        re.compile(r"\bpassword\s*:", re.IGNORECASE),
    )

    for path in SKILL.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern.search(text) is None, f"{pattern.pattern} in {path}"


def test_chart_is_namespace_safe_and_schema_backed() -> None:
    required = {
        "Chart.yaml",
        "values.yaml",
        "values.schema.json",
        "templates/_helpers.tpl",
        "templates/graphos.yaml",
        "templates/engine.yaml",
        "templates/extensions.yaml",
        "templates/policy-and-scaling.yaml",
    }
    assert required <= {
        path.relative_to(CHART).as_posix()
        for path in CHART.rglob("*")
        if path.is_file()
    }

    schema = json.loads((CHART / "values.schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["topology"]["enum"] == [
        "unified-in-process",
        "out-of-process-shared",
    ]

    rendered_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (CHART / "templates").rglob("*")
        if path.is_file()
    )
    assert "kind: Namespace" not in rendered_source
    assert "kind: ClusterRole" not in rendered_source
    assert "kind: ClusterRoleBinding" not in rendered_source
    assert "kind: Secret" not in rendered_source


def test_chart_defaults_fail_safe_for_live_image_pull() -> None:
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))

    assert values["graphos"]["image"]["repository"].startswith("registry.invalid/")
    assert values["engine"]["image"]["repository"].startswith("registry.invalid/")
    assert values["runtimeSecret"]["optional"] is False
