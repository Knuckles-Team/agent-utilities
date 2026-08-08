# Design Document: A per-repo idempotent retrofit script brings an EXISTING `agents/<pkg>` package up to the current modular-contribution standard, instead of requiring a hand-migration or leaving old packages permanently non-conformant

CONCEPT:AU-OS.deployment.agent-factory-autoload

> `scripts/retrofit_fleet_contribution.py:1-40` (module docstring);
> `agent_utilities/cli/__init__.py:382` (`agent-utilities install`, the loader
> that consumes what this retrofit produces).

## Decision — `retrofit_fleet_contribution.py` mechanically migrates one existing agent package to the standard the scaffolder now emits for NEW packages — canonical `<pkg_module>/prompts/main_agent.json` (migrated in place from any legacy `main_agent.json`), a starter skill when the package ships none, `pyproject.toml` entry-points for `skill_providers`/`prompt_providers`, and widened `MANIFEST.in` for sdist parity — safe to re-run, and non-zero-exiting rather than guessing when it cannot safely patch `pyproject.toml`

The fleet's skill/prompt auto-loading (`agent-utilities install`,
`cli/__init__.py:382`) depends on every `agents/<pkg>` repo declaring itself as a
contributor through a consistent, entry-point-discoverable shape. New packages get
that shape from the scaffolder automatically. Packages that predate the current
standard do not, and the fleet has ~62 agent packages plus a growing skill library —
manually bringing each up to date is exactly the kind of repetitive, error-prone,
per-repo migration the platform's own "delegate the mechanical majority" philosophy
argues against doing 62 times by hand (the same reasoning `scripts/concept_domain_triage.py`
applies to concept-governance backlog: a deterministic tool does the mechanical
part once, safely, and is re-runnable). The script is explicitly idempotent ("Safe
to re-run") and fails loud rather than silently leaving a package half-migrated
when it cannot patch `pyproject.toml` safely ("hand-fix those outliers" —
`retrofit_fleet_contribution.py:17`).

## Rejected alternative — hand-migrate each existing package once, or require every package to be re-scaffolded from scratch

Two alternatives were available and rejected. First, a one-time manual migration
per package: this was rejected because it does not scale to ~62+ packages and,
crucially, is not re-runnable — if the scaffolder's target shape changes again
later (as it has before), a manual migration has to be redone by hand again, while
a script re-applies cleanly. Second, discarding an existing package's history and
re-running the NEW-package scaffolder against it (effectively re-creating it): that
was rejected because it would clobber real, already-committed package-specific
content (existing skills, prompts, code) that a from-scratch scaffold does not know
how to preserve — the retrofit script instead reads what already exists (e.g. "any
existing `<pkg_module>/main_agent.json`") and canonicalizes it in place rather than
overwriting it.

## Risk Assessment

- **Blast Radius**: `scripts/retrofit_fleet_contribution.py`; the target repo's
  `pyproject.toml`, `MANIFEST.in`, and `<pkg_module>/{prompts,skills}/`.
- **Backward Compatible**: Yes — idempotent, additive migration; a package already
  at the current standard is a no-op re-run.
- **Known weak point**: `pyproject.toml` patching is pattern-based, not a full
  TOML-semantic rewrite — a package whose `pyproject.toml` deviates enough from
  the expected shape makes the script refuse (non-zero exit) rather than risk a
  bad edit, which means those outliers still need the hand-fix the tool exists to
  avoid for the common case.
