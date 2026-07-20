#!/usr/bin/env python3
"""Scaffold the ontology + in-repo source-preset legs for an agent-package.

CONCEPT:AU-KG.ontology.federation-provider-leg / OS-5.83. ``retrofit_fleet_contribution.py``
brings a package up to the skill/prompt contribution standard but does NOT touch the
ontology leg or the in-repo connector preset. This scaffolder emits both, idempotently:

  1. ``<module>/ontology/__init__.py`` (data-only docstring) + ``<module>/ontology/<domain>.ttl``
     — the two-file OWL/RDF module template (shared ``:<http://knuckles.team/kg#>`` namespace,
     a per-package ``owl:Ontology`` IRI importing the hub base). Hand-expand the stub classes.
  2. ``<module>/connectors/mcp_source_presets.json`` — a Tier-1 ``mcp_tool`` source-preset stub
     (declarative: server + tool + field map) so the package syncs into the KG with no
     agent-utilities code (AU-KG.ingest.mcp-tool-connector). Hand-fill the tool/field map.
  3. ``pyproject.toml`` entry-points (``agent_utilities.ontology_providers`` /
     ``agent_utilities.source_connector_providers``) + ``ontology/**``,``connectors/**`` package-data.

Run per-repo in a worktree; safe to re-run. Does NOT register the new IRI in the hub's
``REGISTERED_FEDERATED_IRIS`` — that is the serial Phase-2 integrator step.

Usage::

    python scripts/scaffold_ontology_leg.py /path/to/agents/<pkg> [--domain <slug>] [--no-preset] [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrofit_fleet_contribution import _display, _pkg_module, _write  # noqa: E402

BASE_IRI = "http://knuckles.team/kg"
ENTERPRISE_IRI = "http://knuckles.team/kg/enterprise"
# Packages whose domain extends the enterprise layer rather than the base.
_ENTERPRISE_DOMAINS = {
    "egeria",
    "leanix",
    "aris",
    "archimate",
    "camunda",
    "erpnext",
    "salesforce",
    "twenty",
    "onetrust",
    "ciso",
    "clarity",
    "sql",
    "database",
    "servicenow",
    "plane",
    "legal",
}


def _ontology_init(display: str, domain: str) -> str:
    return (
        f'"""{display} ontology contribution (CONCEPT:AU-KG.ontology.federation-provider-leg).\n\n'
        f"Data-only subpackage: it carries ``{domain}.ttl`` (the ``owl:Ontology``\n"
        f"``{BASE_IRI}/{domain}`` module) which the agent-utilities hub federates in via\n"
        "the ``agent_utilities.ontology_providers`` entry-point. It holds no business logic\n"
        "and no heavy imports so the hub can resolve it cheaply.\n"
        '"""\n'
    )


def _ttl_stub(display: str, domain: str) -> str:
    imports = ENTERPRISE_IRI if domain in _ENTERPRISE_DOMAINS else BASE_IRI
    cap = "".join(w.capitalize() for w in domain.replace("-", " ").split())
    return f"""@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix dc: <http://purl.org/dc/terms/> .
@prefix : <{BASE_IRI}#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# TODO: replace this stub with the real domain model. All classes/properties live in
# the SHARED ``:`` (kg#) namespace; only the owl:Ontology node below gets a per-package IRI.
# Classes -> owl:Class; links -> owl:ObjectProperty (+rdfs:range); fields -> owl:DatatypeProperty.

:{cap}Resource a owl:Class ;
    rdfs:label "{display} Resource" ;
    rdfs:comment "A primary object managed by the {display} system (STUB — rename/expand)." .

:managedBy a owl:ObjectProperty ;
    rdfs:label "managed by" ;
    rdfs:comment "Links a {display} resource to the Person/Agent responsible (STUB)." ;
    rdfs:domain :{cap}Resource ;
    rdfs:range :Person .

:{domain.replace("-", "")}Id a owl:DatatypeProperty ;
    rdfs:label "{domain} id" ;
    rdfs:comment "External identifier of the {display} resource (STUB)." ;
    rdfs:domain :{cap}Resource ;
    rdfs:range xsd:string .

<{BASE_IRI}/{domain}> a owl:Ontology ;
    rdfs:label "{display} Ontology" ;
    rdfs:comment \"\"\"{display} domain extensions federated into the agent-utilities
    knowledge-graph hub. STUB — replace with the real class/link/field model.\"\"\" ;
    owl:imports <{imports}> .
"""


def _preset_stub(repo_name: str, domain: str) -> str:
    server = (
        repo_name
        if repo_name.endswith(("-mcp", "-agent", "-api"))
        else f"{repo_name}-mcp"
    )
    data = {
        "_comment": (
            "Tier-1 mcp_tool source preset(s) for this package (AU-KG.ingest.mcp-tool-connector). "
            "Hand-fill: tool, action, records_path, id/title/text/updated fields, pagination. "
            "Contributed presets win over the central MCP_TOOL_PRESETS."
        ),
        f"{domain}-records": {
            "server": server,
            "tool": f"TODO_{domain}_list_tool",
            "action": "list",
            "records_path": "data",
            "id_field": "id",
            "title_field": "name",
            "text_field": "description",
            "updated_field": "updated_at",
            "doc_type": domain,
        },
    }
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _patch_pyproject(
    text: str, repo_name: str, module: str, preset: bool
) -> tuple[str, list[str]]:
    changes: list[str] = []
    out = text

    if "agent_utilities.ontology_providers" not in out:
        block = (
            f'[project.entry-points."agent_utilities.ontology_providers"]\n'
            f'{repo_name} = "{module}.ontology"\n\n'
        )
        if preset and "agent_utilities.source_connector_providers" not in out:
            block += (
                f'[project.entry-points."agent_utilities.source_connector_providers"]\n'
                f'{repo_name} = "{module}.connectors"\n\n'
            )
        for anchor in (
            '[project.entry-points."agent_utilities.prompt_providers"]',
            "[tool.setuptools]",
            "[tool.setuptools.packages.find]",
            "[build-system]",
        ):
            if anchor in out:
                out = out.replace(anchor, block + anchor, 1)
                changes.append("added ontology/source entry-points")
                break
        else:
            return text, ["ERROR: no anchor section for entry-points"]

    # package-data globs
    import re

    if "[tool.setuptools.package-data]" in out:

        def _augment(m: re.Match) -> str:
            line = m.group(0)
            for glob in ('"ontology/**"', '"connectors/**"'):
                if glob == '"connectors/**"' and not preset:
                    continue
                if glob not in line:
                    line = (
                        line.rstrip("]\n").rstrip().rstrip("]").rstrip() + f", {glob},]"
                    )
                    line = line.replace(",,", ",")
            return line

        new = re.sub(rf"{module} = \[[^\]]*\]", _augment, out, count=1)
        if new != out:
            out = new
            changes.append("augmented package-data")
    return out, changes


def scaffold(repo: Path, domain: str | None, preset: bool, check: bool) -> int:
    module_dir = _pkg_module(repo)
    if module_dir is None:
        print(f"SKIP {repo.name}: could not find package module dir", file=sys.stderr)
        return 2
    module = module_dir.name
    repo_name = module.replace("_", "-")
    display = _display(repo_name)
    dom = domain or (repo_name.rsplit("-", 1)[0] if "-" in repo_name else repo_name)
    changes: list[str] = []

    onto = module_dir / "ontology"
    if not any(onto.glob("*.ttl")) if onto.exists() else True:
        _write(onto / "__init__.py", _ontology_init(display, dom), changes, check)
        _write(onto / f"{dom}.ttl", _ttl_stub(display, dom), changes, check)
    else:
        print(f"OK {repo_name}: ontology/*.ttl already present — not overwriting")

    if preset:
        conn = module_dir / "connectors"
        _write(conn / "__init__.py", "", changes, check)
        if not (conn / "mcp_source_presets.json").exists():
            _write(
                conn / "mcp_source_presets.json",
                _preset_stub(repo_name, dom),
                changes,
                check,
            )

    pp = repo / "pyproject.toml"
    if pp.exists():
        new_text, pp_changes = _patch_pyproject(
            pp.read_text(encoding="utf-8"), repo_name, module, preset
        )
        if any(c.startswith("ERROR") for c in pp_changes):
            print(f"FAIL {repo_name}: {pp_changes}", file=sys.stderr)
            return 1
        if pp_changes:
            changes.extend(pp_changes)
            if not check:
                pp.write_text(new_text, encoding="utf-8")

    verb = "WOULD SCAFFOLD" if check else "SCAFFOLD"
    if changes:
        print(f"{verb} {repo_name} ({module}) domain={dom}:")
        for c in changes:
            print(f"  - {c}")
    else:
        print(f"OK {repo_name}: ontology leg already conformant")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Path to the agent-package repo.")
    parser.add_argument(
        "--domain", help="Ontology domain slug (default: package short slug)."
    )
    parser.add_argument(
        "--no-preset", action="store_true", help="Skip the connector-preset leg."
    )
    parser.add_argument("--check", action="store_true", help="Report only; no writes.")
    args = parser.parse_args(argv)
    return scaffold(
        Path(args.repo).resolve(), args.domain, not args.no_preset, args.check
    )


if __name__ == "__main__":
    raise SystemExit(main())
