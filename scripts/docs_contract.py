#!/usr/bin/env python3
"""Generate and validate the published documentation contract.

The contract has four deterministic parts:

* every publishable Markdown page is either in ``mkdocs.yml`` navigation or in
  the generated documentation catalog;
* every local Markdown link resolves in a standalone checkout;
* the generated runtime-configuration catalog mirrors ``AgentConfig`` plus
  call-site ``config.setting`` keys;
* the generated capability/action inventory remains discoverable from MkDocs.

Run ``python scripts/docs_contract.py --write`` after adding documentation or
configuration. CI and pre-commit use the default, side-effect-free ``--check``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
MKDOCS = ROOT / "mkdocs.yml"
DOC_CATALOG = DOCS / "reference" / "documentation-catalog.md"
CONFIG_CATALOG = DOCS / "reference" / "runtime-configuration.md"
CAPABILITY_CATALOG = DOCS / "capabilities-power.md"
CAPABILITY_DATA = DOCS / "capabilities-power.json"
GRAPHOS_SURFACE_DOC = DOCS / "pillars" / "4_ecosystem_peripherals.md"
SKILL_CERTIFICATION_DOC = DOCS / "release" / "skill-validation-certification.md"

SITE_EXCLUDES: frozenset[str] = frozenset()
REQUIRED_NAV = frozenset(
    {
        "architecture/graph-authority-convergence.md",
        "architecture/mandatory-context-compiler.md",
        "architecture/observability.md",
        "architecture/privacy-safe-external-ingestion.md",
        "architecture/self-evolution-flywheel.md",
        "architecture/universal-external-graph-connectors.md",
        "capabilities-power.md",
        "ecosystem-capability-fleet.md",
        "guides/kg-skill-suite.md",
        "reference/documentation-catalog.md",
        "reference/runtime-configuration.md",
        "release/compatibility-and-certification.md",
        "release/connector-live-certification.md",
    }
)

_INLINE_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\n]+)\)")
_REFERENCE_LINK_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.MULTILINE)
_HTML_LINK_RE = re.compile(r"\bhref=[\"']([^\"']+)[\"']", re.IGNORECASE)
_INTERNAL_DEFAULT_RE = re.compile(
    r"(?i)(?:[A-Za-z0-9-]+\.)+(?:arpa|internal)\b|"
    r"(?:[A-Za-z0-9-]+\.)*svc\.cluster\.local\b|"
    r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)
_PATH_KEY_PARTS = frozenset({"PATH", "ROOT", "DIR", "DIRECTORY", "FILE", "SOCKET"})
_HOST_IDENTITY_RE = re.compile(
    r"(?i)(?:^|[\s`|])(?:/(?:root|home|users|mnt)/|"
    r"[A-Z]:[\\/](?:Users|Documents and Settings)[\\/])"
)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_EXPLICIT_ID_RE = re.compile(r"\{\s*#([A-Za-z][\w:.-]*)[^}]*\}")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"`+[^`]*`+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MARKDOWN_DECORATION_RE = re.compile(r"[*_~]")


def _is_host_local_default(env_key: str, value: object) -> bool:
    parts = frozenset(str(env_key).upper().split("_"))
    if not parts.intersection(_PATH_KEY_PARTS):
        return False
    if isinstance(value, str):
        return bool(re.match(r"^(?:/|[A-Za-z]:[\\/])", value.strip()))
    if isinstance(value, list | tuple):
        return any(_is_host_local_default(env_key, item) for item in value)
    if isinstance(value, dict):
        return any(_is_host_local_default(env_key, item) for item in value.values())
    return False


class _MkDocsSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that treats MkDocs callable references as plain names."""


def _python_name_as_text(
    loader: _MkDocsSafeLoader, suffix: str, node: yaml.Node
) -> str:
    # The tag is a reference consumed later by MkDocs. The documentation
    # contract only needs navigation strings, so never import or construct it.
    if not isinstance(node, yaml.ScalarNode) or node.value:
        raise yaml.constructor.ConstructorError(
            None, None, "invalid Python-name reference", node.start_mark
        )
    return suffix


_MkDocsSafeLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:", _python_name_as_text
)


def _load_mkdocs() -> dict:
    """Load MkDocs config without constructing its Python-specific YAML tag."""
    value = yaml.load(MKDOCS.read_text(encoding="utf-8"), Loader=_MkDocsSafeLoader)
    if not isinstance(value, dict):
        raise ValueError("MkDocs configuration must be a mapping")
    return value


def _nav_paths() -> list[str]:
    paths: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, str):
            paths.append(value)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(_load_mkdocs().get("nav", []))
    return paths


def _site_pages() -> list[Path]:
    return sorted(
        path
        for path in DOCS.rglob("*.md")
        if path.relative_to(DOCS).as_posix() not in SITE_EXCLUDES
    )


def _relative(path: Path) -> str:
    return path.relative_to(DOCS).as_posix()


def _title(path: Path) -> str:
    if path == DOC_CATALOG:
        return "Documentation Catalog"
    if path == CONFIG_CATALOG:
        return "Runtime Configuration"
    in_fence = False
    marker = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fence = _FENCE_RE.match(line)
        if fence:
            token = fence.group(1)
            if not in_fence:
                in_fence = True
                marker = token[0]
            elif token[0] == marker:
                in_fence = False
            continue
        if in_fence:
            continue
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            title = _EXPLICIT_ID_RE.sub("", match.group(1)).strip()
            return _MARKDOWN_DECORATION_RE.sub("", title).replace("`", "")
    return path.stem.replace("_", " ").replace("-", " ").title()


def _escape_cell(value: object) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text.replace("`", "\\`")


def _capability_metrics() -> tuple[int, int]:
    data = json.loads(CAPABILITY_DATA.read_text(encoding="utf-8"))
    capabilities = data.get("capabilities", [])
    return len(capabilities), sum(len(item.get("does", [])) for item in capabilities)


def check_graphos_surface_contract(
    content: str | None = None,
    *,
    capability_count: int | None = None,
) -> list[str]:
    """Keep the public GraphOS startup surface aligned with current-only code."""
    if content is None:
        content = GRAPHOS_SURFACE_DOC.read_text(encoding="utf-8")
    if capability_count is None:
        capability_count = _capability_metrics()[0]
    normalized = " ".join(content.split())
    required = (
        "exactly **11 visible tools**",
        "**six intent verbs and five control tools**",
        "`ask`, `find`, `write`, `act`, `manage`, `why`",
        "`find_tools`, `list_catalog`, `load_tools`, `unload_tools`, `multiplexer_status`",
        f"**{capability_count} public capabilities**",
        "progressive disclosure, not a compatibility alias",
    )
    errors = [
        "GraphOS surface documentation is stale"
        for token in required
        if token not in normalized
    ]
    if "exposes **25 tools**" in normalized:
        errors.append(
            "GraphOS surface documentation restored the retired 25-tool claim"
        )
    return sorted(set(errors))


def check_skill_certification_docs_contract(content: str | None = None) -> list[str]:
    """Bind published skill certification to installed-release attestation."""
    if content is None:
        content = SKILL_CERTIFICATION_DOC.read_text(encoding="utf-8")
    required = (
        '"agentUtilitiesSha256": "sha256:<digest>"',
        '"agentUtilitiesFileCount": 10',
        '"distributionClosureSha256": "sha256:<digest>"',
        '"releasePythonSha256": "sha256:<digest>"',
        "`processGate.installedReleaseAttested` is exactly `true`",
        "they are not operator-supplied CLI values",
    )
    return (
        []
        if all(token in content for token in required)
        else ["skill certification documentation is stale"]
    )


def _setting_calls() -> dict[str, set[str]]:
    calls: dict[str, set[str]] = defaultdict(set)
    for path in sorted((ROOT / "agent_utilities").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            first = node.args[0]
            if (
                name == "setting"
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
            ):
                calls[first.value].add(path.relative_to(ROOT).as_posix())
    return calls


def _config_reference() -> tuple[list[dict], dict[str, set[str]]]:
    # Never let a developer's XDG config or deployment .env influence a generated
    # artifact. The schema defaults are the source, not the current host.
    os.environ["AGENT_UTILITIES_TESTING"] = "true"
    sys.path.insert(0, str(ROOT))
    from agent_utilities.deployment.config_generator import config_reference

    sections = config_reference()
    typed = {field["env"] for section in sections for field in section["fields"]}
    dynamic = {
        key: paths for key, paths in _setting_calls().items() if key not in typed
    }
    return sections, dynamic


def render_config_catalog() -> str:
    sections, dynamic = _config_reference()
    typed_count = sum(len(section["fields"]) for section in sections)
    lines = [
        "# Runtime Configuration Catalog",
        "",
        "> **GENERATED — do not edit by hand.** Run "
        "`python scripts/docs_contract.py --write`. Defaults come from the "
        "`AgentConfig` schema; secret values are never rendered.",
        "",
        f"{typed_count} typed fields · {len(dynamic)} runtime-only call-site inputs.",
        "",
        "```mermaid",
        "flowchart LR",
        "    Schema[AgentConfig schema] --> Generator[docs contract generator]",
        "    Live[config.setting call sites] --> Generator",
        "    Generator --> Catalog[versioned configuration catalog]",
        "    Catalog --> Gate{drift gate}",
        "    Gate -->|pass| XDG[XDG config or secret references at runtime]",
        "    XDG --> Normalize[normalize persisted provenance]",
        "    Normalize --> Neutral[repo:// · skill:// · connector://]",
        "    Gate -->|stale or unsafe| Block[block commit and docs build]",
        "```",
        "",
        "Use the XDG configuration file created by `setup-config generate`; "
        "deployment secrets must be references resolved by the configured secret store.",
        "",
        "Persisted provenance uses neutral identifiers such as "
        "`repo://<package>/<path>`, `skill://<provider>/<skill>`, or "
        "`connector://<provider>/<record>`. Fields including `workspace_path`, "
        "`source_path`, `skill_path`, `source_file`, and endpoint identities are "
        "runtime-only inputs: normalize or omit them before writing tracked files, "
        "logs, traces, caches, or graph metadata.",
        "",
        "Both GraphOS process-identity fields remain unset only for "
        "`graph-os --transport stdio` with `DEPLOYMENT_PROFILE=tiny`, no "
        "`GRAPH_SERVICE_ENDPOINTS`, and the packaged local engine. That exact "
        "boundary signs and validates a short-lived JWT with an in-memory key as "
        "a one-time proof, uses neutral service claims, destroys the proof key and "
        "token, and returns a process-lifetime session; it persists no personal, "
        "host, endpoint, filesystem, credential, or proof data. Every network "
        "transport, non-tiny profile, explicit "
        "engine endpoint, and other entry point requires exactly one of "
        "`KG_AUTH_TOKEN_REF` or `KG_IDENTITY_OAUTH2`; acquisition or validation "
        "failure never falls back locally. External stdio authority uses a "
        "renewable shared in-memory expiry-only lease. Renewal must preserve "
        "subject, actor type, capabilities, tenant, authentication state, and "
        "groups; drift is rejected, failure never extends the lease, and graph "
        "work fails closed at expiry while renewal retries.",
        "",
    ]
    for section in sections:
        lines.extend(
            [
                f"## {_escape_cell(section['section'])}",
                "",
                "| Environment key | Type | Default |",
                "|---|---|---|",
            ]
        )
        for field in section["fields"]:
            default = field["default"]
            if field["secret"]:
                rendered_default = "explicit runtime process value only"
            elif _is_host_local_default(field["env"], default):
                rendered_default = "resolved at runtime"
            elif isinstance(default, str) and _INTERNAL_DEFAULT_RE.search(default):
                rendered_default = "configured at runtime"
            elif default.__class__.__name__ == "PydanticUndefinedType":
                rendered_default = "unset"
            elif default is None:
                rendered_default = "unset"
            elif isinstance(default, (list, dict)):
                rendered_default = json.dumps(default, sort_keys=True)
            else:
                rendered_default = str(default)
            lines.append(
                f"| `{_escape_cell(field['env'])}` | `{_escape_cell(field['type'])}` "
                f"| `{_escape_cell(rendered_default)}` |"
            )
        lines.append("")

    lines.extend(
        [
            "## Runtime-only call-site inputs",
            "",
            "These keys are discovered from literal `config.setting(...)` calls but are "
            "not durable AgentConfig fields. This includes secret values materialized only "
            "inside a child process for an upstream SDK. Persisted configuration must use "
            "the corresponding `*_REF` field; an ordinary durable setting belongs in "
            "AgentConfig and will then move into the typed tables above.",
            "",
            "| Environment key | Read sites |",
            "|---|---:|",
        ]
    )
    for key, paths in sorted(dynamic.items()):
        lines.append(f"| `{_escape_cell(key)}` | {len(paths)} |")
    lines.append("")
    return "\n".join(lines)


def render_document_catalog() -> str:
    nav = set(_nav_paths())
    pages = [path for path in _site_pages() if path != DOC_CATALOG]
    capabilities, actions = _capability_metrics()
    config_sections, dynamic = _config_reference()
    typed_count = sum(len(section["fields"]) for section in config_sections)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in pages:
        rel = _relative(path)
        group = rel.split("/", 1)[0] if "/" in rel else "Top level"
        grouped[group].append(path)

    lines = [
        "# Documentation Catalog",
        "",
        "> **GENERATED — do not edit by hand.** Run "
        "`python scripts/docs_contract.py --write`. Every publishable Markdown page "
        "must be reachable from MkDocs navigation or this catalog.",
        "",
        f"{len(pages) + 1} publishable pages · {len(nav & {_relative(p) for p in pages})} "
        f"direct nav targets · {capabilities} public capabilities · {actions} action rows · "
        f"{typed_count} typed configuration fields · {len(dynamic)} runtime-only call-site inputs.",
        "",
        "The detailed public capability/action contract is the "
        "[generated Capability Power catalog](../capabilities-power.md). The complete "
        "configuration contract is the "
        "[generated Runtime Configuration catalog](runtime-configuration.md).",
        "",
    ]
    order = [
        "Top level",
        "guides",
        "recipes",
        "examples",
        "architecture",
        "pillars",
        "reference",
        "scaling",
    ]
    for group in order + sorted(set(grouped) - set(order)):
        if group not in grouped:
            continue
        lines.extend([f"## {group.replace('_', ' ').title()}", ""])
        for path in grouped[group]:
            rel = _relative(path)
            target = f"../{rel}"
            route = "direct nav" if rel in nav else "catalog"
            lines.append(f"- [{_title(path)}]({target}) — {route}")
        lines.append("")
    return "\n".join(lines)


def _prose_lines(path: Path) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    in_fence = False
    marker = ""
    for number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        fence = _FENCE_RE.match(line)
        if fence:
            token = fence.group(1)
            if not in_fence:
                in_fence = True
                marker = token[0]
            elif token[0] == marker:
                in_fence = False
            continue
        if not in_fence:
            lines.append((number, line))
    return lines


def _slugify(value: str) -> str:
    value = _EXPLICIT_ID_RE.sub("", value)
    value = re.sub(r"!?\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = _HTML_TAG_RE.sub("", value)
    value = _MARKDOWN_DECORATION_RE.sub("", value).replace("`", "")
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-")


def _anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = defaultdict(int)
    for _, line in _prose_lines(path):
        anchors.update(_EXPLICIT_ID_RE.findall(line))
        heading = _HEADING_RE.match(line)
        if not heading:
            continue
        base = _slugify(heading.group(1))
        if not base:
            continue
        suffix = counts[base]
        anchors.add(base if suffix == 0 else f"{base}_{suffix}")
        counts[base] += 1
    return anchors


def _extract_target(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("<") and ">" in raw:
        return raw[1 : raw.index(">")]
    return raw.split(maxsplit=1)[0].strip("<>")


def _resolve_target(source: Path, target: str) -> tuple[Path | None, str]:
    if not target or target.startswith(("//", "{{", "${")):
        return None, ""
    parsed = urlsplit(target)
    if parsed.scheme:
        return None, ""
    fragment = unquote(parsed.fragment)
    raw_path = unquote(parsed.path)
    if not raw_path:
        return source, fragment
    candidate = (
        (DOCS / raw_path.lstrip("/"))
        if raw_path.startswith("/")
        else (source.parent / raw_path)
    )
    candidate = candidate.resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError:
        return candidate, fragment
    if not candidate.exists() and not candidate.suffix:
        if candidate.with_suffix(".md").exists():
            candidate = candidate.with_suffix(".md")
        elif (candidate / "index.md").exists():
            candidate = candidate / "index.md"
        elif (candidate / "README.md").exists():
            candidate = candidate / "README.md"
    return candidate, fragment


def check_links() -> list[str]:
    errors: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}
    for source in _site_pages():
        text = "\n".join(
            _INLINE_CODE_RE.sub("", line) for _, line in _prose_lines(source)
        )
        raw_links = _INLINE_LINK_RE.findall(text)
        raw_links += _REFERENCE_LINK_RE.findall(text)
        raw_links += _HTML_LINK_RE.findall(text)
        for raw in raw_links:
            target = _extract_target(raw)
            display_source = source.relative_to(ROOT).as_posix()
            if urlsplit(target).scheme.lower() == "file":
                errors.append(f"{display_source}: non-portable file URI")
                continue
            candidate, fragment = _resolve_target(source, target)
            if candidate is None:
                continue
            if not candidate.exists():
                errors.append(f"{display_source}: unresolved local link `{target}`")
                continue
            try:
                candidate.relative_to(ROOT)
            except ValueError:
                errors.append(f"{display_source}: non-portable checkout-external link")
                continue
            if fragment and candidate.suffix.lower() == ".md":
                anchors = anchor_cache.setdefault(candidate, _anchors(candidate))
                if fragment not in anchors:
                    errors.append(
                        f"{display_source}: missing anchor `#{fragment}` in "
                        f"{candidate.relative_to(ROOT).as_posix()}"
                    )
    return sorted(set(errors))


def _catalog_links() -> set[str]:
    if not DOC_CATALOG.exists():
        return set()
    links: set[str] = set()
    for raw in _INLINE_LINK_RE.findall(DOC_CATALOG.read_text(encoding="utf-8")):
        candidate, _ = _resolve_target(DOC_CATALOG, _extract_target(raw))
        if candidate and candidate.is_file() and candidate.suffix == ".md":
            try:
                links.add(candidate.relative_to(DOCS).as_posix())
            except ValueError:
                pass
    return links


def metrics() -> dict[str, int]:
    docs_total = len(list(DOCS.rglob("*.md")))
    pages = {_relative(path) for path in _site_pages()}
    nav = set(_nav_paths())
    catalog = _catalog_links() | (
        {_relative(DOC_CATALOG)} if DOC_CATALOG.exists() else set()
    )
    return {
        "docs_total": docs_total,
        "publishable_pages": len(pages),
        "nav_unique": len(nav),
        "nav_coverage": len(pages & nav),
        "catalog_coverage": len(pages & catalog),
        "orphans": len(pages - nav - catalog),
        "missing_nav_targets": len(nav - pages),
    }


def check() -> list[str]:
    errors: list[str] = []
    expected = {
        DOC_CATALOG: render_document_catalog(),
        CONFIG_CATALOG: render_config_catalog(),
    }
    if _HOST_IDENTITY_RE.search(expected[CONFIG_CATALOG]):
        errors.append(
            "generated runtime configuration contains a host-local identity/path"
        )
    for path, content in expected.items():
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            errors.append(
                f"{path.relative_to(ROOT).as_posix()} is stale; run "
                "`python scripts/docs_contract.py --write`"
            )
    pages = {_relative(path) for path in _site_pages()}
    nav = set(_nav_paths())
    for missing in sorted(nav - pages):
        errors.append(f"mkdocs.yml points to missing page `{missing}`")
    for missing in sorted(REQUIRED_NAV - nav):
        errors.append(f"mkdocs.yml must expose `{missing}`")
    orphaned = pages - nav - _catalog_links() - {_relative(DOC_CATALOG)}
    for orphan in sorted(orphaned):
        errors.append(f"orphaned publishable page `{orphan}`")
    errors.extend(check_links())
    errors.extend(check_graphos_surface_contract())
    errors.extend(check_skill_certification_docs_contract())
    return errors


def write(*, runtime_config_only: bool = False) -> None:
    CONFIG_CATALOG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_CATALOG.write_text(render_config_catalog(), encoding="utf-8")
    if not runtime_config_only:
        DOC_CATALOG.write_text(render_document_catalog(), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate catalogs.")
    parser.add_argument(
        "--runtime-config-only",
        action="store_true",
        help="With --write, regenerate only the AgentConfig reference.",
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate without writing."
    )
    parser.add_argument(
        "--metrics", action="store_true", help="Print coverage metrics."
    )
    args = parser.parse_args()
    if args.runtime_config_only and not args.write:
        parser.error("--runtime-config-only requires --write")
    if args.write:
        write(runtime_config_only=args.runtime_config_only)
    if args.metrics:
        for key, value in metrics().items():
            print(f"{key}={value}")
    if args.write and not args.check:
        return 0
    errors = check()
    if errors:
        print("Documentation contract FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    current = metrics()
    print(
        "Documentation contract PASSED "
        f"({current['publishable_pages']} pages, {current['orphans']} orphans)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
