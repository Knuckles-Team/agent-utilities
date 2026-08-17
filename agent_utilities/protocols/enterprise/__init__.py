"""Normalized enterprise read models — one vocabulary the UI/agents consume
regardless of source vendor.

CONCEPT:AU-ECO.ui.normalized-enterprise-read-models

See ``read_models.py`` for the envelope, ``ChangeRequest``/``Build`` models,
and the GitHub/GitLab adapters that close BUG-012 (GitHub/GitLab contract
mismatch, GOC-27-W01/W02/W03).
"""

from .read_models import (
    Build,
    ChangeRequest,
    SchemaMismatchError,
    build_from_github_workflow_run,
    build_from_gitlab_pipeline,
    change_request_from_github_pull_request,
    change_request_from_gitlab_merge_request,
)

__all__ = [
    "Build",
    "ChangeRequest",
    "SchemaMismatchError",
    "build_from_github_workflow_run",
    "build_from_gitlab_pipeline",
    "change_request_from_github_pull_request",
    "change_request_from_gitlab_merge_request",
]
