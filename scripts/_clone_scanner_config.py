"""Shared, checked-in configuration for the repository clone scanners.

The scanners are native binaries rather than Python dependencies.  Keeping the
versions, thresholds, formats, and exclusion policy in ``pyproject.toml`` gives
the wrappers and their hooks one reviewable source of truth while still letting
an operator install the binaries through the platform package manager of their
choice.  This module deliberately has no third-party dependencies so a missing
scanner can be reported honestly instead of being hidden behind an installer.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any


class CloneScannerConfigError(ValueError):
    """Raised when the checked-in clone-scanner contract is invalid."""


_CONFIG_KEYS = frozenset(
    {
        "dupehound_version",
        "dupehound_threshold",
        "dupehound_min_tokens",
        "jscpd_version",
        "jscpd_min_tokens",
        "jscpd_min_lines",
        "jscpd_mode",
        "jscpd_diff_formats",
        "jscpd_diff_extensions",
        "exclusions",
        "prune_directories",
        "jscpd_diff_format_extensions",
        "jscpd_diff_format_names",
    }
)


@dataclass(frozen=True)
class CloneScannerConfig:
    """Validated values consumed by both clone-scanner wrappers."""

    dupehound_version: str
    dupehound_threshold: float
    dupehound_min_tokens: int
    jscpd_version: str
    jscpd_min_tokens: int
    jscpd_min_lines: int
    jscpd_mode: str
    jscpd_diff_formats: tuple[str, ...]
    jscpd_diff_extensions: tuple[str, ...]
    jscpd_diff_format_extensions: tuple[tuple[str, tuple[str, ...]], ...]
    jscpd_diff_format_names: tuple[tuple[str, tuple[str, ...]], ...]
    exclusions: tuple[str, ...]
    prune_directories: frozenset[str]

    @property
    def dupehound_version_output(self) -> str:
        """The exact version line emitted by dupehound's clap CLI."""

        return f"dupehound {self.dupehound_version}"

    @property
    def jscpd_version_output(self) -> str:
        """The exact version line emitted by the jscpd v5 binary."""

        # jscpd v5 is published as ``cpd`` even though its executable is
        # ``jscpd``.  The wrapper checks this spelling to reject drift.
        return f"cpd {self.jscpd_version}"

    @property
    def jscpd_diff_filenames(self) -> tuple[str, ...]:
        """Extensionless filenames mapped to a jscpd format."""

        return tuple(
            filename
            for _format, filenames in self.jscpd_diff_format_names
            for filename in filenames
        )

    @property
    def jscpd_format_names_arg(self) -> str:
        """The native CLI value for ``--formats-names``."""

        return ";".join(
            f"{format_name}:{','.join(filenames)}"
            for format_name, filenames in self.jscpd_diff_format_names
        )

    @property
    def jscpd_format_exts_arg(self) -> str:
        """The native CLI value for ``--formats-exts``."""

        return ";".join(
            f"{format_name}:{','.join(extensions)}"
            for format_name, extensions in self.jscpd_diff_format_extensions
        )


def _require(mapping: Mapping[str, Any], name: str) -> Any:
    try:
        return mapping[name]
    except KeyError as exc:
        raise CloneScannerConfigError(
            f"[tool.agent_utilities.clone_scanners] is missing {name!r}"
        ) from exc


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CloneScannerConfigError(f"{name} must be a non-empty string")
    return value.strip()


def _version(value: Any, name: str) -> str:
    version = _string(value, name)
    parts = version.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise CloneScannerConfigError(
            f"{name} must be an exact three-part version, got {version!r}"
        )
    return version


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CloneScannerConfigError(f"{name} must be a positive integer")
    return value


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CloneScannerConfigError(f"{name} must be a number between 0 and 1")
    fraction = float(value)
    if not 0.0 <= fraction <= 1.0:
        raise CloneScannerConfigError(f"{name} must be between 0 and 1")
    return fraction


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CloneScannerConfigError(f"{name} must be a non-empty array")
    result = tuple(
        _string(item, f"{name}[{index}]") for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise CloneScannerConfigError(f"{name} must not contain duplicate entries")
    return result


def _format_names(
    value: Any, name: str, allowed_formats: tuple[str, ...]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return _format_mapping(
        value, name, allowed_formats, _filenames, description="filename"
    )


def _filenames(value: Any, name: str) -> tuple[str, ...]:
    filenames = _strings(value, name)
    if any("/" in filename or "\\" in filename for filename in filenames):
        raise CloneScannerConfigError(f"{name} must contain filenames, not paths")
    return filenames


def _record_unique_values(
    seen: set[str],
    values: tuple[str, ...],
    table_name: str,
    *,
    description: str,
) -> None:
    duplicates = seen.intersection(values)
    if duplicates:
        formatted = ", ".join(sorted(duplicates))
        raise CloneScannerConfigError(
            f"{table_name} maps a {description} more than once: {formatted}"
        )
    seen.update(values)


def _bare_extensions(value: Any, name: str) -> tuple[str, ...]:
    extensions = _strings(value, name)
    invalid = any(
        not extension
        or extension.startswith(".")
        or "/" in extension
        or "\\" in extension
        for extension in extensions
    )
    if invalid:
        raise CloneScannerConfigError(f"{name} must contain bare extensions")
    return extensions


def _file_extensions(value: Any, name: str) -> tuple[str, ...]:
    extensions = _strings(value, name)
    invalid = any(
        not extension.startswith(".")
        or extension == "."
        or "/" in extension
        or "\\" in extension
        for extension in extensions
    )
    if invalid:
        raise CloneScannerConfigError(
            f"{name} must contain lowercase dotted file extensions"
        )
    if any(extension != extension.lower() for extension in extensions):
        raise CloneScannerConfigError(f"{name} must contain lowercase extensions")
    return extensions


def _format_extensions(
    value: Any, name: str, allowed_formats: tuple[str, ...]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return _format_mapping(
        value, name, allowed_formats, _bare_extensions, description="extension"
    )


def _format_mapping(
    value: Any,
    name: str,
    allowed_formats: tuple[str, ...],
    value_reader: Callable[[Any, str], tuple[str, ...]],
    *,
    description: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(value, Mapping) or not value:
        raise CloneScannerConfigError(f"{name} must be a non-empty TOML table")
    allowed = set(allowed_formats)
    entries: list[tuple[str, tuple[str, ...]]] = []
    seen_values: set[str] = set()
    for raw_format, raw_values in value.items():
        format_name = _string(raw_format, f"{name} key")
        if format_name not in allowed:
            raise CloneScannerConfigError(
                f"{name} maps unknown/non-differential format {format_name!r}"
            )
        values = value_reader(raw_values, f"{name}.{format_name}")
        _record_unique_values(seen_values, values, name, description=description)
        entries.append((format_name, values))
    return tuple(sorted(entries))


def load_clone_scanner_config(path: Path) -> CloneScannerConfig:
    """Load and validate the clone-scanner table from ``path``.

    Missing or malformed configuration is an environment/configuration error;
    callers should map :class:`CloneScannerConfigError` to their fail-closed
    exit status rather than substituting local defaults.
    """

    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CloneScannerConfigError(f"cannot read {path}: {exc}") from exc

    try:
        table = document["tool"]["agent_utilities"]["clone_scanners"]
    except (KeyError, TypeError) as exc:
        raise CloneScannerConfigError(
            "pyproject.toml has no [tool.agent_utilities.clone_scanners] table"
        ) from exc
    if not isinstance(table, Mapping):
        raise CloneScannerConfigError(
            "[tool.agent_utilities.clone_scanners] must be a TOML table"
        )
    _reject_unknown_keys(table)

    jscpd_diff_formats = _strings(
        _require(table, "jscpd_diff_formats"), "jscpd_diff_formats"
    )
    jscpd_diff_format_names = _format_names(
        _require(table, "jscpd_diff_format_names"),
        "jscpd_diff_format_names",
        jscpd_diff_formats,
    )
    jscpd_diff_format_extensions = _format_extensions(
        _require(table, "jscpd_diff_format_extensions"),
        "jscpd_diff_format_extensions",
        jscpd_diff_formats,
    )

    return CloneScannerConfig(
        dupehound_version=_version(
            _require(table, "dupehound_version"), "dupehound_version"
        ),
        dupehound_threshold=_fraction(
            _require(table, "dupehound_threshold"), "dupehound_threshold"
        ),
        dupehound_min_tokens=_positive_int(
            _require(table, "dupehound_min_tokens"), "dupehound_min_tokens"
        ),
        jscpd_version=_version(_require(table, "jscpd_version"), "jscpd_version"),
        jscpd_min_tokens=_positive_int(
            _require(table, "jscpd_min_tokens"), "jscpd_min_tokens"
        ),
        jscpd_min_lines=_positive_int(
            _require(table, "jscpd_min_lines"), "jscpd_min_lines"
        ),
        jscpd_mode=_jscpd_mode(_require(table, "jscpd_mode")),
        jscpd_diff_formats=jscpd_diff_formats,
        jscpd_diff_extensions=_file_extensions(
            _require(table, "jscpd_diff_extensions"), "jscpd_diff_extensions"
        ),
        jscpd_diff_format_extensions=jscpd_diff_format_extensions,
        jscpd_diff_format_names=jscpd_diff_format_names,
        exclusions=_strings(_require(table, "exclusions"), "exclusions"),
        prune_directories=frozenset(
            _strings(_require(table, "prune_directories"), "prune_directories")
        ),
    )


def _reject_unknown_keys(table: Mapping[str, Any]) -> None:
    unknown = set(table).difference(_CONFIG_KEYS)
    if unknown:
        formatted = ", ".join(sorted(unknown))
        raise CloneScannerConfigError(
            f"unknown clone-scanner configuration key(s): {formatted}"
        )


def _jscpd_mode(value: Any) -> str:
    mode = _string(value, "jscpd_mode")
    if mode not in {"mild", "weak", "strict"}:
        raise CloneScannerConfigError(
            "jscpd_mode must be one of 'mild', 'weak', or 'strict'"
        )
    return mode


def is_excluded_path(path: str | Path, patterns: tuple[str, ...]) -> bool:
    """Return whether a repo-relative path is covered by an exclusion glob.

    ``fnmatch`` treats ``*`` as crossing path separators, while gitignore-like
    ``dir/**`` patterns are also common in the checked-in policy.  The explicit
    prefix check makes that latter intent unambiguous and keeps selection tests
    independent of the native binary's own glob implementation.
    """

    rel = PurePosixPath(str(path).replace("\\", "/")).as_posix()
    rel = rel.lstrip("/")
    while rel.startswith("./"):
        rel = rel[2:]
    for pattern in patterns:
        normalized = pattern.replace("\\", "/").lstrip("./")
        if normalized.endswith("/**"):
            prefix = normalized[:-3].rstrip("/")
            if rel == prefix or rel.startswith(f"{prefix}/"):
                return True
        if fnmatch(rel, normalized) or fnmatch(f"/{rel}", normalized):
            return True
    return False


def is_jscpd_diff_path(path: str | Path, config: CloneScannerConfig) -> bool:
    """Return whether ``path`` belongs to jscpd's reviewed diff scope."""

    candidate = PurePosixPath(str(path).replace("\\", "/"))
    return (
        candidate.name in config.jscpd_diff_filenames
        or candidate.suffix.lower() in config.jscpd_diff_extensions
    )
