# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import TypedDict, Annotated, Literal

from .._utils import PropertyInfo

from typing import List, Union, Dict, Optional
from typing_extensions import Literal, TypedDict, Required, Annotated
from .._types import FileTypes
from .._utils import PropertyInfo
from ..types import shared_params

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