# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import List, Optional
from typing_extensions import Literal, Required, Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["InferencePipelineCreateParams", "Project", "Workspace"]


class InferencePipelineCreateParams(TypedDict, total=False):
    description: Required[Optional[str]]
    """The inference pipeline description."""

    name: Required[str]
    """The inference pipeline name."""

    project: Optional[Project]

    workspace: Optional[Workspace]


class Project(TypedDict, total=False):
    name: Required[str]
    """The project name."""

    task_type: Required[
        Annotated[
            Literal["llm-base", "tabular-classification", "tabular-regression", "text-classification"],
            PropertyInfo(alias="taskType"),
        ]
    ]
    """The task type of the project."""

    description: Optional[str]
    """The project description."""


class Workspace(TypedDict, total=False):
    name: Required[str]
    """The workspace name."""

    slug: Required[str]
    """The workspace slug."""

    invite_code: Annotated[str, PropertyInfo(alias="inviteCode")]
    """The workspace invite code."""

    saml_only_access: Annotated[bool, PropertyInfo(alias="samlOnlyAccess")]
    """Whether the workspace only allows SAML authentication."""

    wildcard_domains: Annotated[List[str], PropertyInfo(alias="wildcardDomains")]
