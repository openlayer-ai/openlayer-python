# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Annotated, TypedDict

from .._utils import PropertyInfo

__all__ = ["ProjectListParams"]


class ProjectListParams(TypedDict, total=False):
    name: str
    """Filter list of items by project name."""

    page: int
    """The page to return in a paginated query."""

    per_page: Annotated[int, PropertyInfo(alias="perPage")]
    """Maximum number of items to return per page."""

    task_type: Annotated[
        Literal["llm-base", "tabular-classification", "tabular-regression", "text-classification"],
        PropertyInfo(alias="taskType"),
    ]
    """Filter list of items by task type."""
