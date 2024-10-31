# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Required, Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["CommitCreateParams", "Commit"]


class CommitCreateParams(TypedDict, total=False):
    commit: Required[Commit]
    """The details of a commit (project version)."""

    storage_uri: Required[Annotated[str, PropertyInfo(alias="storageUri")]]
    """The storage URI where the commit bundle is stored."""

    archived: Optional[bool]
    """Whether the commit is archived."""

    deployment_status: Annotated[str, PropertyInfo(alias="deploymentStatus")]
    """The deployment status associated with the commit's model."""


class Commit(TypedDict, total=False):
    message: Required[str]
    """The commit message."""
