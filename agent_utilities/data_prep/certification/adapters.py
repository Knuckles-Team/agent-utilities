"""Lazy optional Great Expectations and Pandera adapter boundaries.

Provider APIs intentionally remain outside the core contract.  An operator
supplies a trusted runner that knows how to construct a provider suite/schema
and how to publish a bounded Data Docs artifact through the deployment's
access-controlled artifact store.  The adapter imports its provider only when
the operator job is actually executed, never when GraphOS or data preparation
is imported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol

from .models import (
    AdapterObservation,
    AuthorizedArrowSample,
    CertificationBackend,
)

__all__ = [
    "CertificationAdapter",
    "CertificationDependencyUnavailable",
    "GreatExpectationsAdapter",
    "PanderaAdapter",
]


class CertificationDependencyUnavailable(RuntimeError):
    """Raised without provider detail when an optional package is absent."""


class CertificationAdapter(Protocol):
    """Trusted operator adapter invoked only by the certification job."""

    backend: CertificationBackend

    def run(
        self,
        table: Any,
        *,
        authorization: AuthorizedArrowSample,
    ) -> AdapterObservation: ...


Runner = Callable[[Any, AuthorizedArrowSample], AdapterObservation | Mapping[str, Any]]


def _provider_version(distribution: str) -> str:
    try:
        value = version(distribution)
    except PackageNotFoundError as exc:
        raise CertificationDependencyUnavailable from exc
    # AdapterObservation applies the same opaque-reference grammar.  Returning
    # the package version here avoids accepting a caller-supplied version claim.
    return value


def _observation(
    value: AdapterObservation | Mapping[str, Any],
    *,
    provider_version: str,
) -> AdapterObservation:
    if isinstance(value, AdapterObservation):
        # Keep nested DataClassification enum instances for strict protocol
        # re-validation; JSON mode would turn them into strings.
        payload = value.model_dump(mode="python")
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError("provider runner must return a bounded observation")
    payload["schema_version"] = "data-quality-observation.v1"
    payload["provider_version"] = provider_version
    return AdapterObservation.model_validate(payload)


@dataclass(frozen=True, slots=True)
class GreatExpectationsAdapter:
    """Composition adapter for a trusted Great Expectations runner.

    The runner owns suite lookup and Data Docs publication.  It may return only
    the fields accepted by :class:`AdapterObservation`; report bytes and suite
    configuration never cross this package boundary.
    """

    runner: Runner
    backend: CertificationBackend = "great_expectations"

    def run(
        self,
        table: Any,
        *,
        authorization: AuthorizedArrowSample,
    ) -> AdapterObservation:
        try:
            import great_expectations  # noqa: F401  # lazy optional boundary
        except ImportError as exc:
            raise CertificationDependencyUnavailable from exc
        return _observation(
            self.runner(table, authorization),
            provider_version=_provider_version("great_expectations"),
        )


@dataclass(frozen=True, slots=True)
class PanderaAdapter:
    """Composition adapter for a trusted Pandera runner."""

    runner: Runner
    backend: CertificationBackend = "pandera"

    def run(
        self,
        table: Any,
        *,
        authorization: AuthorizedArrowSample,
    ) -> AdapterObservation:
        try:
            import pandera  # noqa: F401  # lazy optional boundary
        except ImportError as exc:
            raise CertificationDependencyUnavailable from exc
        return _observation(
            self.runner(table, authorization),
            provider_version=_provider_version("pandera"),
        )
