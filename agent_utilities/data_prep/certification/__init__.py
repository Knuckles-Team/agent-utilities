"""Optional operator-only GE/Pandera certification boundary.

Importing this package does not import Great Expectations, Pandera, pandas,
NumPy, or any provider runtime.  The package is not part of the agent hot path;
operators install one named extra and invoke :func:`run_certification_job` from
an isolated job.
"""

from .adapters import (
    CertificationAdapter,
    CertificationDependencyUnavailable,
    GreatExpectationsAdapter,
    PanderaAdapter,
)
from .job import CertificationSigner, mark_not_requested, run_certification_job
from .models import (
    AdapterObservation,
    ArtifactPolicy,
    AuthorizedArrowSample,
    CertificationArtifact,
    CertificationBackend,
    CertificationJobSpec,
    CertificationResult,
    CertificationState,
    FailureCode,
    SignedResultSummary,
)

__all__ = [
    "AdapterObservation",
    "ArtifactPolicy",
    "AuthorizedArrowSample",
    "CertificationAdapter",
    "CertificationArtifact",
    "CertificationBackend",
    "CertificationDependencyUnavailable",
    "CertificationJobSpec",
    "CertificationResult",
    "CertificationSigner",
    "CertificationState",
    "FailureCode",
    "GreatExpectationsAdapter",
    "PanderaAdapter",
    "SignedResultSummary",
    "mark_not_requested",
    "run_certification_job",
]
