#!/usr/bin/env python3
"""Deterministic Connector Ontology Manifest generator (CONCEPT:AU-KG.ontology.connector-manifest-generator).

**ZERO LLM calls, no network.** For a given connector package (an ``agents/<pkg>``
checkout), projects the artifacts the connector ALREADY ships into a single
declarative ``connector_manifest.yml``:

  * ``<module>/ontology/*.ttl``            -> ``resources`` (owl:Class) +
                                               ``schema_mappings`` (owl:DatatypeProperty
                                               field/XSD) + resource ``relations``
                                               (owl:ObjectProperty, only where the ttl
                                               declares an ``rdfs:domain``).
  * ``<module>/connectors/mcp_source_presets.json`` -> ``sync`` (:data:`MCP_TOOL_PRESETS`
                                               shape) + ``identity`` + a synthetic
                                               ``events`` watermark per preset.
  * ``a2a.json``                           -> ``actions`` (from ``capabilities``).

Every field that cannot be derived losslessly from those artifacts (the ontology-class
crosswalk, PII/RLS policy) is filled with a documented heuristic default and flagged in
``review_todos`` — never silently guessed, never invented by an LLM.

Same input, timestamp, and explicit release key -> byte-identical output. Pass ``--now``
to pin the provenance timestamp for reproducible/test runs. Applying a manifest requires
``ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF``; no development fallback or raw
configuration value can produce release evidence.

Usage:
  python3 scripts/generate_connector_manifests.py --connector-root <path> [--output PATH]
  python3 scripts/generate_connector_manifests.py --all --agents-root <path> [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.ontology import ontology_integrity  # noqa: E402
from agent_utilities.knowledge_graph.ontology.connector_manifest import (  # noqa: E402
    DEFAULT_ARCHIMATE_CROSSWALK,
    PII_HEURISTIC_FIELD_NAMES,
    ActionSpec,
    ConnectorManifest,
    EventSpec,
    IdentitySpec,
    IntegrityInfo,
    PermissionsSpec,
    PolicySpec,
    ProvenanceSpec,
    ResourceRelation,
    ResourceSpec,
    SchemaMapping,
    SyncSpec,
    nearest_hub_class,
)
from agent_utilities.knowledge_graph.ontology.manifest_compiler import (  # noqa: E402
    compile_manifest,
    export_manifest_ttl,
)
from agent_utilities.orchestration.fleet_reconciler import (  # noqa: E402
    registry_server_alias,
)

_XSD_NS = "http://www.w3.org/2001/XMLSchema#"
_OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
_OWL_OBJECT_PROPERTY = "http://www.w3.org/2002/07/owl#ObjectProperty"
_OWL_DATATYPE_PROPERTY = "http://www.w3.org/2002/07/owl#DatatypeProperty"
_RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
_RDFS_DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
_RDFS_RANGE = "http://www.w3.org/2000/01/rdf-schema#range"
_RDFS_SUBCLASSOF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
ONTOLOGY_LOCK = ROOT / "agent_utilities" / "knowledge_graph" / "ontology.lock"
_CONNECTOR_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def connector_project_name(connector_root: Path) -> str:
    """Resolve connector identity from project metadata, not checkout directory."""

    pyproject = connector_root / "pyproject.toml"
    if not pyproject.is_file():
        return connector_root.name
    try:
        document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        name = document["project"]["name"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError("connector project metadata is invalid") from exc
    if not isinstance(name, str) or not _CONNECTOR_NAME.fullmatch(name):
        raise RuntimeError("connector project name is invalid")
    return name


#: Fleet-wide default org for connectors whose repository URL cannot be derived
#: from a git remote or ``[project.urls]`` — every currently-resolvable connector
#: in the fleet belongs to this org, so it is a documented heuristic default
#: (flagged via ``review_todos``), never a per-connector guess.
_DEFAULT_A2A_ORG = "Knuckles-Team"

#: The SAME capability content as the ``Skill(id="epistemic-answer", ...)``
#: appended to every live AgentCard in ``agent_utilities/server/app.py``
#: (CONCEPT:AU-KB-CURRENCY) — kept in lockstep by hand (both are short and
#: reviewed together on change) rather than sharing a runtime import, so
#: generation stays a pure, dependency-free projection.
#:
#: This used to live in the now-deleted ``scripts/enrich_fleet_a2a_epistemic.py``
#: (a one-off backfill for providers whose ``a2a.json`` predated fleet-wide
#: generation, D-A2A-3): once every provider's ``a2a.json`` is generator-owned
#: (68/68, 2026-07-31 fleet re-certification), that standalone enrichment path
#: was a strict subset of what full regeneration already does, so it was
#: deleted rather than kept as a redundant parallel path (No-Legacy).
EPISTEMIC_CAPABILITY: dict[str, Any] = {
    "id": "epistemic-answer",
    "name": "Epistemic Answer",
    "description": (
        "Answers epistemic_status/why/what_changed queries over the shared "
        "knowledge graph: calibrated confidence, evidence/source citations, "
        "belief justification trees, bitemporal valid/tx history, and "
        "policy-redaction-aware provenance."
    ),
    "tags": ["epistemic", "provenance", "confidence", "kg"],
}

#: The ``a2a.json`` capability every agent-utilities-built connector ships
#: (CONCEPT:AU-KG.ontology.a2a-card-generation): graph-flow execution, universal
#: to the framework, plus the epistemic-answer capability every live AgentCard
#: already advertises.
DEFAULT_A2A_CAPABILITIES: tuple[dict[str, Any], ...] = (
    {
        "id": "run_graph_flow",
        "name": "Graph Flow Execution",
        "description": (
            "Execute a workflow through the agent's graph orchestration engine"
        ),
    },
    dict(EPISTEMIC_CAPABILITY),
)

#: The ``a2a.json`` tool every agent-utilities-built connector ships.
DEFAULT_A2A_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "id": "graph-flow",
        "type": "flow",
        "description": "Run complex multi-step workflows via Pydantic-Graph",
    },
)


def _a2a_pyproject_document(connector_root: Path) -> dict[str, Any]:
    """Load the full ``pyproject.toml`` document (not just ``[project]``)."""

    pyproject = connector_root / "pyproject.toml"
    try:
        return tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError("connector project metadata is invalid") from exc


def _resolve_a2a_version(connector_root: Path, document: dict[str, Any]) -> str:
    """Resolve the connector version: static ``[project.version]`` first, else
    the ``[tool.setuptools.dynamic] version.attr`` module attribute — never a
    hand-typed a2a.json value, which is exactly what drifts (CONCEPT:AU-KG.ontology.a2a-card-generation).
    """

    project = document.get("project", {})
    version = project.get("version")
    if isinstance(version, str) and version:
        return version
    dynamic_attr = (
        document.get("tool", {})
        .get("setuptools", {})
        .get("dynamic", {})
        .get("version", {})
        .get("attr")
    )
    if isinstance(dynamic_attr, str) and dynamic_attr:
        module_path, _, attr_name = dynamic_attr.rpartition(".")
        module_file = connector_root.joinpath(*module_path.split(".")).with_suffix(
            ".py"
        )
        if module_file.is_file():
            match = re.search(
                rf'{re.escape(attr_name)}\s*=\s*["\']([^"\']+)["\']',
                module_file.read_text(encoding="utf-8"),
            )
            if match:
                return match.group(1)
    raise RuntimeError("connector project version could not be resolved")


def _resolve_a2a_license(document: dict[str, Any]) -> str:
    """Resolve the SPDX-ish license label from ``[project.license]`` or a
    ``License :: OSI Approved :: ...`` classifier; ``MIT`` is the documented
    fleet-wide fallback (every connector observed to date uses it)."""

    project = document.get("project", {})
    license_field = project.get("license")
    if isinstance(license_field, dict):
        text = license_field.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    elif isinstance(license_field, str) and license_field.strip():
        return license_field.strip()
    for classifier in project.get("classifiers", []) or []:
        match = re.match(r"^License :: OSI Approved :: (.+) License$", str(classifier))
        if match:
            words = match.group(1).split()
            if words:
                return words[0]
    return "MIT"


def _git_remote_origin(connector_root: Path) -> str | None:
    """Best-effort, bounded ``git remote get-url origin`` — the connector repo's
    own checkout state, not the CWD, so output is stable across working
    directories."""

    try:
        result = subprocess.run(  # noqa: S603
            ["git", "-C", str(connector_root), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    remote = result.stdout.strip()
    return remote or None


def _normalize_repo_url(url: str) -> str:
    trimmed = url.strip()
    if trimmed.endswith(".git"):
        trimmed = trimmed[: -len(".git")]
    if "/tree/" in trimmed or "/blob/" in trimmed:
        return trimmed
    return trimmed.rstrip("/") + "/tree/main"


def _resolve_a2a_url(
    connector_root: Path,
    name: str,
    document: dict[str, Any],
    todos: list[str],
) -> str:
    """Resolve the connector's canonical repository URL: ``[project.urls]``
    first (repo-internal, works without a live git checkout), then the git
    ``origin`` remote of the connector's own repo, then a documented
    heuristic default (see ``_DEFAULT_A2A_ORG``)."""

    urls = document.get("project", {}).get("urls")
    if isinstance(urls, dict):
        for key in ("Homepage", "Repository", "Source", "homepage", "repository"):
            candidate = urls.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return _normalize_repo_url(candidate.strip())
    remote = _git_remote_origin(connector_root)
    if remote:
        return _normalize_repo_url(remote)
    todos.append(
        "a2a.json url could not be derived from [project.urls] or a git remote; "
        f"defaulting to https://github.com/{_DEFAULT_A2A_ORG}/{name}/tree/main — "
        "add a git remote or a [project.urls] Homepage to remove this heuristic."
    )
    return f"https://github.com/{_DEFAULT_A2A_ORG}/{name}/tree/main"


def _a2a_extra_entries(document: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Read the small, optional, per-connector ``[tool.a2a]`` residue table —
    the ONLY hand-maintained a2a.json content left after generation, reserved
    for capabilities/tools/description that are genuinely per-connector and not
    derivable from any other artifact (CONCEPT:AU-KG.ontology.a2a-card-generation)."""

    entries = document.get("tool", {}).get("a2a", {}).get(key, [])
    if not isinstance(entries, list):
        return []
    return [dict(entry) for entry in entries if isinstance(entry, dict)]


def build_a2a_card(
    connector_root: Path, *, todos: list[str] | None = None
) -> dict[str, Any]:
    """Build the ``a2a.json`` agent card for one connector — pure, deterministic,
    offline, no LLM (CONCEPT:AU-KG.ontology.a2a-card-generation). Every field is
    either read verbatim from ``pyproject.toml`` (the connector's own package
    metadata) or a documented heuristic default flagged in ``todos``; the only
    hand-maintained input is the small optional ``[tool.a2a]`` residue table for
    content that genuinely cannot be derived elsewhere (e.g. a bespoke
    capability description).
    """

    todos = todos if todos is not None else []
    document = _a2a_pyproject_document(connector_root)
    project = document.get("project", {})
    name = connector_project_name(connector_root)

    description_override = document.get("tool", {}).get("a2a", {}).get("description")
    if isinstance(description_override, str) and description_override.strip():
        description = description_override.strip()
    else:
        project_description = project.get("description")
        if isinstance(project_description, str) and project_description.strip():
            description = project_description.strip()
        else:
            description = f"Agent package for {name}"
            todos.append(
                "a2a.json description fell back to a generic placeholder — add "
                "[project] description (or a [tool.a2a] description override) "
                "to pyproject.toml."
            )

    capabilities = [dict(c) for c in DEFAULT_A2A_CAPABILITIES]
    capabilities.extend(_a2a_extra_entries(document, "capabilities"))
    tools = [dict(t) for t in DEFAULT_A2A_TOOLS]
    tools.extend(_a2a_extra_entries(document, "tools"))

    return {
        "name": f"{name}-agent",
        "type": "agent",
        "version": _resolve_a2a_version(connector_root, document),
        "description": description,
        "url": _resolve_a2a_url(connector_root, name, document, todos),
        "license": _resolve_a2a_license(document),
        "capabilities": capabilities,
        "tools": tools,
    }


def _canonical_a2a_bytes(card: dict[str, Any]) -> bytes:
    """Canonical, deterministic serialization: fixed field order (as constructed
    by :func:`build_a2a_card` — no dict-hash-randomization dependency, stable
    across processes/interpreters), 2-space indent, no trailing whitespace
    ambiguity, ASCII-safe, trailing newline, no timestamps, no absolute paths."""

    return (
        json.dumps(card, indent=2, sort_keys=False, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def write_a2a_card(
    connector_root: Path,
    *,
    dry_run: bool = False,
    check: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Regenerate ``a2a.json`` in place from the connector's own package
    metadata. Returns ``(card, changed)``.

    ``check=True`` is the fail-closed drift gate: it never writes, and raises
    if the on-disk file would change — the same mechanism
    ``scripts/check_connector_manifests.py`` uses for ``connector_manifest.yml``.
    """

    card = build_a2a_card(connector_root)
    payload = _canonical_a2a_bytes(card)
    target = connector_root / "a2a.json"
    existing = target.read_bytes() if target.is_file() else None
    changed = existing != payload
    if check:
        if changed:
            raise RuntimeError(
                f"a2a.json drift detected for connector {connector_root.name!r}: "
                "regenerate with generate_connector_manifests.py (no hand edits)."
            )
        return card, False
    if dry_run:
        print(f"# --- {connector_root.name}/a2a.json ---")
        print(payload.decode("utf-8"), end="")
    else:
        target.write_bytes(payload)
    return card, changed


def _local(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[1]
    return uri.rsplit("/", 1)[-1]


def _humanize(camel: str) -> str:
    import re

    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", camel)
    return s[:1].upper() + s[1:] if s else s


def _xsd_curie(uri: str) -> str:
    if uri.startswith(_XSD_NS):
        return f"xsd:{uri[len(_XSD_NS) :]}"
    return "xsd:string"


def _find_module_dir(connector_root: Path) -> Path | None:
    """The one subdirectory shipping ``ontology/*.ttl`` (the connector's python package)."""
    candidates = sorted(
        d for d in connector_root.iterdir() if d.is_dir() and (d / "ontology").is_dir()
    )
    return candidates[0] if candidates else None


def _detect_ontology_source(graph: Any) -> str | None:
    """The ttl's own declared ``owl:Ontology`` IRI local slug (e.g. "servicenow"), if any.

    A connector's python package name (``servicenow-api``) commonly differs from the
    ontology domain slug its ttl already declares and the canonical ``ontology.ttl``
    already ``owl:imports`` (``servicenow``) — detecting it here (rather than assuming
    package name == IRI slug) is what makes the anti-sprawl "already wired" check land
    on the real, existing federated module instead of a false new-source guess.
    """
    import rdflib

    iris = sorted(
        str(s)
        for s in graph.subjects(
            predicate=rdflib.RDF.type,
            object=rdflib.URIRef("http://www.w3.org/2002/07/owl#Ontology"),
        )
        if isinstance(s, rdflib.URIRef)
    )
    for iri in iris:
        if iri.startswith("http://knuckles.team/kg/"):
            return _local(iri)
    return None


def _read_ontology(
    module_dir: Path,
) -> tuple[list[ResourceSpec], dict[str, SchemaMapping], list[str], str | None]:
    """Read every ``*.ttl`` in ``module_dir/ontology`` into resources + schema_mappings.

    Returns ``(resources, schema_mappings, review_todos, ontology_source)``, sorted
    deterministically by resource/local name.
    """
    import rdflib

    graph = rdflib.Graph()
    for ttl in sorted((module_dir / "ontology").glob("*.ttl")):
        graph.parse(str(ttl), format="turtle")

    ontology_source = _detect_ontology_source(graph)
    todos: list[str] = []

    class_uris = sorted(
        {
            str(s)
            for s in graph.subjects(
                predicate=rdflib.RDF.type, object=rdflib.URIRef(_OWL_CLASS)
            )
            if isinstance(s, rdflib.URIRef)
        }
    )
    class_locals = {_local(u) for u in class_uris}

    def _label(uri: str) -> str:
        for lbl in graph.objects(
            subject=rdflib.URIRef(uri), predicate=rdflib.URIRef(_RDFS_LABEL)
        ):
            return str(lbl)
        return _humanize(_local(uri))

    # datatype properties: global (domain-free) field vocabulary, matching the
    # existing fleet convention (ontology_leanix.ttl, servicenow.ttl, gitlab.ttl all
    # keep DatatypeProperty domain-free) — shared across every resource in this ttl set.
    fields: dict[str, str] = {}
    for uri in sorted(
        {
            str(s)
            for s in graph.subjects(
                predicate=rdflib.RDF.type, object=rdflib.URIRef(_OWL_DATATYPE_PROPERTY)
            )
            if isinstance(s, rdflib.URIRef)
        }
    ):
        rng = next(
            graph.objects(
                subject=rdflib.URIRef(uri), predicate=rdflib.URIRef(_RDFS_RANGE)
            ),
            None,
        )
        fields[_local(uri)] = _xsd_curie(str(rng)) if rng is not None else "xsd:string"

    # object properties: only attach to a resource when rdfs:domain is explicitly
    # declared AND resolves to a known class — never guess a domain (Wire-First: no
    # LLM/heuristic invention of structure the source ttl doesn't state).
    relations_by_domain: dict[str, list[ResourceRelation]] = {}
    for uri in sorted(
        {
            str(s)
            for s in graph.subjects(
                predicate=rdflib.RDF.type, object=rdflib.URIRef(_OWL_OBJECT_PROPERTY)
            )
            if isinstance(s, rdflib.URIRef)
        }
    ):
        domain = next(
            graph.objects(
                subject=rdflib.URIRef(uri), predicate=rdflib.URIRef(_RDFS_DOMAIN)
            ),
            None,
        )
        rng = next(
            graph.objects(
                subject=rdflib.URIRef(uri), predicate=rdflib.URIRef(_RDFS_RANGE)
            ),
            None,
        )
        target = _local(str(rng)) if isinstance(rng, rdflib.URIRef) else "owl:Thing"
        local = _local(uri)
        if isinstance(domain, rdflib.URIRef) and _local(str(domain)) in class_locals:
            relations_by_domain.setdefault(_local(str(domain)), []).append(
                ResourceRelation(name=local, label=_label(uri), target=target)
            )
        else:
            todos.append(
                f"relation '{local}' has no declared rdfs:domain resolving to a known "
                f"resource in this connector's ontology — not attached to any resource; "
                f"verify its true domain manually."
            )

    resources: list[ResourceSpec] = []
    schema_mappings: dict[str, SchemaMapping] = {}
    for uri in class_uris:
        name = _local(uri)
        parent_ref = next(
            graph.objects(
                subject=rdflib.URIRef(uri), predicate=rdflib.URIRef(_RDFS_SUBCLASSOF)
            ),
            None,
        )
        # D16 residue: three-tier crosswalk, tried in decreasing order of confidence —
        # (1) the source ttl's OWN declared rdfs:subClassOf (not a heuristic at all),
        # (2) DEFAULT_ARCHIMATE_CROSSWALK (LeanIX/ArchiMate fact-sheet lookup by name),
        # (3) HUB_NAME_HEURISTIC_CROSSWALK (nearest hub-ontology class by name — a
        #     best-effort DRAFT, human sign-off required, never auto-enforced).
        # Never invented beyond this conservative table: no hit anywhere -> left None.
        subclass_crosswalk = (
            _local(str(parent_ref)) if isinstance(parent_ref, rdflib.URIRef) else None
        )
        archimate_crosswalk = (
            DEFAULT_ARCHIMATE_CROSSWALK.get(name)
            if subclass_crosswalk is None
            else None
        )
        hub_name_crosswalk = (
            nearest_hub_class(name)
            if subclass_crosswalk is None and archimate_crosswalk is None
            else None
        )
        crosswalk = subclass_crosswalk or archimate_crosswalk or hub_name_crosswalk
        crosswalk_kind = (
            "source ttl rdfs:subClassOf"
            if subclass_crosswalk
            else (
                "DEFAULT_ARCHIMATE_CROSSWALK (LeanIX/ArchiMate lookup by resource name)"
                if archimate_crosswalk
                else (
                    "DRAFT — nearest hub-canonical-class-by-name heuristic "
                    "(D16 residue; human sign-off required before use)"
                    if hub_name_crosswalk
                    else "UNRESOLVED — no crosswalk found by any heuristic"
                )
            )
        )
        resources.append(
            ResourceSpec(
                name=name,
                label=_label(uri),
                id_prefix=name.lower(),
                relations=sorted(
                    relations_by_domain.get(name, []), key=lambda r: r.name
                ),
            )
        )
        schema_mappings[name] = SchemaMapping(
            ontology_class=crosswalk, fields=dict(sorted(fields.items()))
        )
        todos.append(
            f"schema_mappings.{name}.ontology_class = {crosswalk!r} [{crosswalk_kind}] "
            "— verify manually before relying on this crosswalk for reasoning/joins."
        )

    return resources, schema_mappings, todos, ontology_source


def _read_sync(
    module_dir: Path,
    *,
    server_alias: str,
) -> tuple[list[SyncSpec], IdentitySpec, list[EventSpec], PermissionsSpec]:
    """Project ``mcp_source_presets.json`` onto ``sync``, with a DERIVED server.

    CONCEPT:AU-KG.ontology.registry-derived-server-alias — the MCP server a preset
    routes to is taken from ``deploy/mcp-fleet.registry.yml``, never from the
    preset's own ``server`` string. A preset that restates a *different* alias is a
    hard failure, not a warning: on 2026-07-28 that restatement put a server name
    the fleet does not run into 27 signed manifests. Deriving the value means a
    wrong alias cannot be signed; failing closed on disagreement means a wrong
    alias cannot even sit in the source tree unnoticed.
    """

    presets_path = module_dir / "connectors" / "mcp_source_presets.json"
    sync: list[SyncSpec] = []
    identity = IdentitySpec()
    events: list[EventSpec] = []
    permissions = PermissionsSpec()
    if not presets_path.exists():
        return sync, identity, events, permissions

    data = json.loads(presets_path.read_text(encoding="utf-8"))
    fingerprint_path = module_dir / "connectors" / "tool_schema_fingerprints.json"
    fingerprints: dict[str, str] = {}
    if fingerprint_path.exists():
        fingerprint_data = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        raw_fingerprints = (
            fingerprint_data.get("tools", fingerprint_data)
            if isinstance(fingerprint_data, dict)
            else {}
        )
        if isinstance(raw_fingerprints, dict):
            fingerprints = {
                str(name): str(digest).strip().lower()
                for name, digest in raw_fingerprints.items()
                if not str(name).startswith("_")
            }
    for key in sorted(k for k in data if not k.startswith("_")):
        preset = data[key]
        if not isinstance(preset, dict):
            continue
        declared_server = str(preset.get("server") or "")
        if declared_server and declared_server != server_alias:
            raise RuntimeError(
                "connector source preset declares an MCP server alias that the "
                "fleet registry does not register for this provider"
            )
        preset = {**preset, "server": server_alias}
        sync.append(
            SyncSpec(
                preset=key,
                server=server_alias,
                tool=str(preset.get("tool", "")),
                action=preset.get("action"),
                records_path=preset.get("records_path"),
                id_field=preset.get("id_field"),
                title_field=preset.get("title_field"),
                text_field=preset.get("text_field"),
                updated_field=preset.get("updated_field"),
                pagination=preset.get("pagination"),
                doc_type=preset.get("doc_type"),
                tool_schema_sha256=fingerprints.get(str(preset.get("tool", ""))),
                raw=preset,
            )
        )
        doc_type = str(preset.get("doc_type") or key)
        for attr, bucket in (
            ("id_field", identity.id_field),
            ("title_field", identity.title_field),
            ("text_field", identity.text_field),
            ("updated_field", identity.updated_field),
        ):
            val = preset.get(attr)
            if val:
                bucket[doc_type] = str(val)
        if preset.get("updated_field"):
            events.append(
                EventSpec(
                    name=f"{key}.updated",
                    resource=doc_type,
                    description=(
                        f"Watermark event for '{key}' — advances on "
                        f"{preset['updated_field']} (mcp_source_presets.json)."
                    ),
                )
            )
        for field_name in sorted(preset):
            if field_name.startswith("acl_"):
                permissions.acl_fields.append(field_name)

    return sync, identity, events, permissions


def _read_actions(connector_root: Path) -> list[ActionSpec]:
    a2a_path = connector_root / "a2a.json"
    if not a2a_path.exists():
        return []
    data = json.loads(a2a_path.read_text(encoding="utf-8"))
    caps = data.get("capabilities") or []
    return [
        ActionSpec(
            id=str(c.get("id", "")),
            name=str(c.get("name", "")),
            description=str(c.get("description", "")),
        )
        for c in caps
        if isinstance(c, dict) and c.get("id")
    ]


def _pii_policy(schema_mappings: dict[str, SchemaMapping]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for resource, mapping in sorted(schema_mappings.items()):
        hits = sorted(
            f for f in mapping.fields if f.lower() in PII_HEURISTIC_FIELD_NAMES
        )
        if hits:
            out[resource] = hits
    return out


def build_manifest(
    connector_root: Path,
    *,
    now: datetime | None = None,
    release_signer: ontology_integrity.ReleaseSigner | None = None,
    registry_path: Path | None = None,
) -> ConnectorManifest:
    """Build a :class:`ConnectorManifest` for one connector — pure, deterministic, offline."""
    connector = connector_project_name(connector_root)
    # Fail closed BEFORE any artifact is projected: an unregistered provider has no
    # derivable server alias, so nothing about its sync contract can be signed.
    server_alias = registry_server_alias(connector, registry_path)
    module_dir = _find_module_dir(connector_root)
    resources: list[ResourceSpec] = []
    schema_mappings: dict[str, SchemaMapping] = {}
    todos: list[str] = []
    sync: list[SyncSpec] = []
    identity = IdentitySpec()
    events: list[EventSpec] = []
    permissions = PermissionsSpec()
    source_artifacts: list[str] = []
    ontology_source: str | None = None

    if module_dir is not None:
        resources, schema_mappings, todos, ontology_source = _read_ontology(module_dir)
        source_artifacts.extend(
            str(p.relative_to(connector_root))
            for p in sorted((module_dir / "ontology").glob("*.ttl"))
        )
        sync, identity, events, permissions = _read_sync(
            module_dir, server_alias=server_alias
        )
        if (module_dir / "connectors" / "mcp_source_presets.json").exists():
            source_artifacts.append(
                str(
                    (module_dir / "connectors" / "mcp_source_presets.json").relative_to(
                        connector_root
                    )
                )
            )
        if (module_dir / "connectors" / "tool_schema_fingerprints.json").exists():
            source_artifacts.append(
                str(
                    (
                        module_dir / "connectors" / "tool_schema_fingerprints.json"
                    ).relative_to(connector_root)
                )
            )
    else:
        todos.append(
            "no <module>/ontology/*.ttl found for this connector — resources/"
            "schema_mappings could not be derived; ship an ontology module first."
        )

    actions = _read_actions(connector_root)
    if (connector_root / "a2a.json").exists():
        source_artifacts.append("a2a.json")

    policy = PolicySpec(pii_fields=_pii_policy(schema_mappings))
    if policy.pii_fields:
        todos.append(
            "policy.pii_fields was populated by a field-NAME heuristic "
            f"({sorted(PII_HEURISTIC_FIELD_NAMES)}) — verify against the actual data "
            "before relying on it for redaction/RLS."
        )
    todos.append(
        "policy.rls / policy.tenant_boundary are unset — no row-level-security "
        "predicate has been reviewed for this connector."
    )

    source_slug = ontology_source or connector
    placeholder = ConnectorManifest(
        connector=connector,
        ontology_source=(ontology_source or ""),
        resources=resources,
        actions=sorted(actions, key=lambda a: a.id),
        events=sorted(events, key=lambda e: e.name),
        identity=identity,
        permissions=permissions,
        schema_mappings=schema_mappings,
        sync=sync,
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="0" * 64)),
        policy=policy,
        review_todos=sorted(set(todos)),
    )

    spec = compile_manifest(placeholder)
    ttl = export_manifest_ttl(spec, source=source_slug)
    import rdflib

    g = rdflib.Graph()
    g.parse(data=ttl, format="turtle")
    digest, triple_count = ontology_integrity.canonical_hash(g)
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")

    signer = release_signer or ontology_integrity.release_signer_for_publication(
        lock_path=ONTOLOGY_LOCK
    )
    unsigned_provenance = ProvenanceSpec(
        generated_at=stamp,
        source_artifacts=sorted(source_artifacts),
        integrity=IntegrityInfo(hash=digest, triple_count=triple_count),
        signer=signer.signer_id,
        signature_algorithm=signer.algorithm,
        signing_public_key=signer.public_key,
        signature=None,
        # GOC-84/GOC-16: binds the frozen dependency-lock state into what gets
        # signed, so a lock drift after generation is provable, not assumed.
        dependency_lock_digest=ontology_integrity.dependency_lock_digest(),
    )
    unsigned = placeholder.model_copy(update={"provenance": unsigned_provenance})
    manifest_hash = ontology_integrity.canonical_manifest_hash(unsigned)
    provenance = unsigned_provenance.model_copy(
        update={"signature": signer.sign(manifest_hash)}
    )
    return unsigned.model_copy(update={"provenance": provenance})


def _to_yaml(manifest: ConnectorManifest) -> str:
    import yaml

    data = manifest.model_dump(mode="json", exclude_none=False)
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, width=100)


def write_manifest(
    connector_root: Path,
    output: Path,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
    generate_a2a: bool = True,
    registry_path: Path | None = None,
) -> ConnectorManifest:
    # a2a.json is generated FIRST (CONCEPT:AU-KG.ontology.a2a-card-generation): the
    # manifest's ``actions`` are read back from a2a.json's ``capabilities``
    # (``_read_actions``), so the connector-owned card must already be fresh,
    # deterministic, generated content before the manifest is built from it —
    # never a parallel/manual step.
    if generate_a2a and (connector_root / "pyproject.toml").is_file():
        write_a2a_card(connector_root, dry_run=dry_run)
    manifest = build_manifest(connector_root, now=now, registry_path=registry_path)
    text = _to_yaml(manifest)
    output_label = f"{output.parent.name}/{output.name}"
    if dry_run:
        print(f"# --- {output_label} ---")
        print(text)
    else:
        output.write_text(text, encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--connector-root", type=Path, help="path to one connector repo (agents/<pkg>)"
    )
    ap.add_argument(
        "--connector", help="connector name, resolved as <agents-root>/<name>"
    )
    ap.add_argument(
        "--all", action="store_true", help="process every connector under --agents-root"
    )
    ap.add_argument("--agents-root", type=Path, help="the agents/ fleet root")
    ap.add_argument(
        "--output",
        type=Path,
        help="output path (single-connector mode; default <root>/connector_manifest.yml)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (--all mode); default: write into each connector root",
    )
    ap.add_argument(
        "--now", help="ISO-8601 UTC timestamp override, for reproducible runs"
    )
    ap.add_argument("--dry-run", action="store_true", help="print instead of write")
    ap.add_argument(
        "--skip-a2a",
        action="store_true",
        help="do not (re)generate a2a.json — connector_manifest.yml only",
    )
    ap.add_argument(
        "--a2a-check",
        action="store_true",
        help=(
            "fail-closed drift gate: regenerate a2a.json in memory and error if it "
            "would differ from the committed file, without writing anything "
            "(connector_manifest.yml is not touched in this mode)"
        ),
    )
    ap.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="MCP fleet registry that owns the server aliases (default: the shipped one)",
    )
    args = ap.parse_args()

    now = (
        datetime.strptime(args.now, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        if args.now
        else None
    )

    roots: list[Path]
    if args.all:
        if not args.agents_root:
            ap.error("--all requires --agents-root")
        roots = sorted(
            d
            for d in args.agents_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    elif args.connector_root:
        roots = [args.connector_root]
    elif args.connector:
        if not args.agents_root:
            ap.error("--connector requires --agents-root")
        roots = [args.agents_root / args.connector]
    else:
        ap.error("one of --connector-root, --connector, or --all is required")
        return 2

    if args.a2a_check:
        failures = 0
        for root in roots:
            if not root.is_dir() or not (root / "pyproject.toml").is_file():
                continue
            try:
                write_a2a_card(root, check=True)
                print(f"OK    {root.name}/a2a.json")
            except RuntimeError as exc:
                failures += 1
                print(f"DRIFT {root.name}/a2a.json: {exc}", file=sys.stderr)
        return 1 if failures else 0

    for root in roots:
        if not root.is_dir():
            print(f"skip: connector {root.name!r} is not a directory", file=sys.stderr)
            continue
        out = (
            args.output
            if (args.output and len(roots) == 1)
            else (
                (args.output_dir / f"{root.name}.connector_manifest.yml")
                if args.output_dir
                else (root / "connector_manifest.yml")
            )
        )
        manifest = write_manifest(
            root,
            out,
            now=now,
            dry_run=args.dry_run,
            generate_a2a=not args.skip_a2a,
            registry_path=args.registry,
        )
        print(
            f"generated {out.parent.name}/{out.name}: {len(manifest.resources)} "
            f"resources, {len(manifest.sync)} sync presets"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
