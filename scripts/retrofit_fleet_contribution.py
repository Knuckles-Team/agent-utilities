#!/usr/bin/env python3
"""Retrofit an existing agent-package as a skill/prompt fleet contributor.

CONCEPT:AU-OS.deployment.agent-factory-autoload / ORCH-1.80. Idempotently brings one ``agents/<pkg>`` repo up to
the modular-contribution standard the scaffolder now emits for new packages:

  1. canonical ``<pkg_module>/prompts/main_agent.json`` (+ ``prompts/__init__.py``),
     migrated from any existing ``<pkg_module>/main_agent.json`` (which is also
     canonicalised in place so the two stay identical);
  2. a starter skill ``<pkg_module>/skills/<short>-starter/SKILL.md``
     (+ ``skills/__init__.py``) when the package ships no skill yet;
  3. ``pyproject.toml`` entry-points (``agent_utilities.skill_providers`` /
     ``prompt_providers``) + ``prompts/**``,``skills/**`` package-data;
  4. widened ``MANIFEST.in`` for sdist parity.

Run per-repo (in a worktree). Safe to re-run. Prints a per-step report and exits
non-zero if it cannot safely patch ``pyproject.toml`` (hand-fix those outliers).

Usage::

    python scripts/retrofit_fleet_contribution.py /path/to/agents/<pkg> [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from migrate_prompts import migrate_data  # noqa: E402


def _pkg_module(repo: Path) -> Path | None:
    """Return the importable module dir (e.g. servicenow_api) inside the repo."""
    name = repo.name.replace("-", "_")
    candidate = repo / name
    if candidate.is_dir():
        return candidate
    # Fall back: a single dir containing mcp_server.py / agent_server.py.
    for d in sorted(p for p in repo.iterdir() if p.is_dir()):
        if (d / "agent_server.py").exists() or (d / "mcp_server.py").exists():
            return d
    return None


def _canonical_main_agent(display: str, description: str, source: str) -> dict:
    return {
        "schema_version": "1.0",
        "task": "main-agent",
        "type": "prompt",
        "source": source,
        "description": description,
        "extends": "agent-utilities:base",
        "compose": "append",
        "identity": {"role": f"{display} Agent", "goal": description},
        "instructions": {
            "core_directive": (
                f"# {display} Agent\n\nYou are the {display} Agent. {description}\n\n"
                "Use the `mcp-client` universal skill and the reference docs to "
                "discover the exact tags and tools available for your "
                "capabilities.\n\n### Core Principles\n"
                "* Be concise and efficient.\n"
                "* Use the knowledge graph to discover tools and experts.\n"
                "* Verify your work before concluding."
            )
        },
        "tools": ["workspace-manager", "agent-workflows"],
    }


def _starter_skill(display: str, slug: str, description: str) -> str:
    return f"""---
name: {slug}
description: >-
  {description} Use when working with {display} via this package's MCP
  server/agent — discover its tools, run an operation, or check its reference
  docs. Replace this starter with real, trigger-oriented capabilities.
license: MIT
tags: [{slug}, starter, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---

# {display} Starter Skill

Starter skill for the **{display}** package. Use the `mcp-client` universal skill
to connect to this package's MCP server and invoke its tools.

> Replace this with concrete, trigger-oriented capabilities (one per skill).
"""


def _display(repo_name: str) -> str:
    base = repo_name.replace("-api", "").replace("-mcp", "").replace("-agent", "")
    return " ".join(w.capitalize() for w in base.replace("_", "-").split("-"))


def _write(path: Path, content: str, changes: list[str], check: bool) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    changes.append(("would write" if check else "wrote") + f" {path.name} ({path})")
    if not check:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _patch_pyproject(text: str, repo_name: str, module: str) -> tuple[str, list[str]]:
    import re

    changes: list[str] = []
    out = text

    if "agent_utilities.skill_providers" not in out:
        block = (
            f'[project.entry-points."agent_utilities.skill_providers"]\n'
            f'{repo_name} = "{module}.skills"\n\n'
            f'[project.entry-points."agent_utilities.prompt_providers"]\n'
            f'{repo_name} = "{module}.prompts"\n\n'
        )
        for anchor in (
            "[tool.setuptools]",
            "[tool.setuptools.packages.find]",
            "[tool.ruff]",
            "[build-system]",
        ):
            if anchor in out:
                out = out.replace(anchor, block + anchor, 1)
                changes.append("added entry-points")
                break
        else:
            return text, ["ERROR: no anchor section for entry-points"]
    else:
        # Corrective: ensure the provider NAME is the package name (an earlier
        # pass mistakenly used the worktree dir name). The value is unchanged.
        for grp, sub in (
            ("skill_providers", "skills"),
            ("prompt_providers", "prompts"),
        ):
            pat = (
                rf'(\[project\.entry-points\."agent_utilities\.{grp}"\]\n)'
                rf'\S+ = "{module}\.{sub}"'
            )
            new = re.sub(pat, rf'\1{repo_name} = "{module}.{sub}"', out)
            if new != out:
                out = new
                changes.append(f"corrected {grp} name -> {repo_name}")

    # package-data globs
    if "[tool.setuptools.package-data]" in out:
        import re

        def _augment(m: re.Match) -> str:
            line = m.group(0)
            for glob in ('"prompts/**"', '"skills/**"'):
                if glob not in line:
                    line = line.rstrip("]\n").rstrip() + f", {glob},]"
                    line = line.replace(",,", ",")
            return line

        new = re.sub(rf"{module} = \[[^\]]*\]", _augment, out, count=1)
        if new != out:
            out = new
            changes.append("augmented package-data")
    else:
        block = (
            "[tool.setuptools.package-data]\n"
            f'{module} = [ "mcp_config.json", "prompts/**", "skills/**",]\n\n'
        )
        if "[tool.setuptools.packages.find]" in out:
            out = out.replace(
                "[tool.setuptools.packages.find]",
                block + "[tool.setuptools.packages.find]",
                1,
            )
            changes.append("added package-data")
    return out, changes


def retrofit(repo: Path, check: bool) -> int:
    module_dir = _pkg_module(repo)
    if module_dir is None:
        print(f"SKIP {repo.name}: could not find package module dir", file=sys.stderr)
        return 2
    module = module_dir.name
    # Derive the package name from the MODULE dir, never from the path passed in
    # (a worktree dir like '.../modular-contrib' would otherwise poison the
    # provider name / prompt source for every package).
    repo_name = module.replace("_", "-")
    display = _display(repo_name)
    description = f"{display} API + MCP Server + A2A Server"
    changes: list[str] = []

    # Detect contamination from the earlier worktree-name bug (generic
    # "modular-contrib" / "Modular Contrib" baked into prompt + starter).
    contaminated = False
    for cand in (
        module_dir / "prompts" / "main_agent.json",
        module_dir / "main_agent.json",
    ):
        if cand.exists() and (
            "modular-contrib" in (blob := cand.read_text(encoding="utf-8"))
            or "Modular Contrib" in blob
        ):
            contaminated = True
            break

    # 1. canonical prompt — preserve a real existing prompt; regenerate fresh
    # when missing or contaminated. Always force-correct the source provenance.
    root_main = module_dir / "main_agent.json"
    if root_main.exists() and not contaminated:
        try:
            data = json.loads(root_main.read_text(encoding="utf-8"))
            data, _ = migrate_data(data, source=repo_name)
            data["source"] = repo_name
        except (OSError, json.JSONDecodeError):
            data = _canonical_main_agent(display, description, repo_name)
    else:
        data = _canonical_main_agent(display, description, repo_name)
    blob = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    _write(root_main, blob, changes, check)
    _write(module_dir / "prompts" / "__init__.py", "", changes, check)
    _write(module_dir / "prompts" / "main_agent.json", blob, changes, check)

    # 2. starter skill (only if no real skill exists)
    skills_dir = module_dir / "skills"
    short = repo_name.rsplit("-", 1)[0] if "-" in repo_name else repo_name
    # Remove the contaminated generic starter from the buggy first pass.
    bad_starter = skills_dir / "modular-starter"
    if bad_starter.exists():
        changes.append("removed modular-starter")
        if not check:
            import shutil

            shutil.rmtree(bad_starter)
    existing = [
        p
        for p in (skills_dir.rglob("SKILL.md") if skills_dir.exists() else [])
        if "modular-starter" not in p.parts
    ]
    _write(skills_dir / "__init__.py", "", changes, check)
    if not existing:
        _write(
            skills_dir / f"{short}-starter" / "SKILL.md",
            _starter_skill(display, f"{short}-starter", description),
            changes,
            check,
        )

    # 3. pyproject
    pp = repo / "pyproject.toml"
    if pp.exists():
        new_text, pp_changes = _patch_pyproject(
            pp.read_text(encoding="utf-8"), repo_name, module
        )
        if any(c.startswith("ERROR") for c in pp_changes):
            print(f"FAIL {repo_name}: {pp_changes}", file=sys.stderr)
            return 1
        if pp_changes:
            changes.extend(pp_changes)
            if not check:
                pp.write_text(new_text, encoding="utf-8")

    # 4. MANIFEST.in
    man = repo / "MANIFEST.in"
    if man.exists():
        mtext = man.read_text(encoding="utf-8")
        if f"recursive-include {module}" in mtext and "*.md" not in mtext:
            import re

            mtext2 = re.sub(
                rf"(recursive-include {module} )(.*)",
                r"\1*.py *.json *.md *.yaml *.yml",
                mtext,
                count=1,
            )
            if mtext2 != mtext and not check:
                man.write_text(mtext2, encoding="utf-8")
            if mtext2 != mtext:
                changes.append("widened MANIFEST.in")

    verb = "WOULD RETROFIT" if check else "RETROFIT"
    if changes:
        print(f"{verb} {repo_name} ({module}):")
        for c in changes:
            print(f"  - {c}")
    else:
        print(f"OK {repo_name}: already conformant")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Path to the agent-package repo to retrofit.")
    parser.add_argument("--check", action="store_true", help="Report only; no writes.")
    args = parser.parse_args(argv)
    return retrofit(Path(args.repo).resolve(), args.check)


if __name__ == "__main__":
    raise SystemExit(main())
