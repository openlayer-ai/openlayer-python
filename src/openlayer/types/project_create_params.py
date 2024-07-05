# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Literal, Required, Annotated, TypedDict

from .._utils import PropertyInfo

__all__ = ["ProjectCreateParams", "GitRepo"]


class ProjectCreateParams(TypedDict, total=False):
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

    git_repo: Annotated[Optional[GitRepo], PropertyInfo(alias="gitRepo")]


class GitRepo(TypedDict, total=False):
    git_account_id: Required[Annotated[str, PropertyInfo(alias="gitAccountId")]]

    git_id: Required[Annotated[int, PropertyInfo(alias="gitId")]]

    branch: str

    root_dir: Annotated[str, PropertyInfo(alias="rootDir")]
