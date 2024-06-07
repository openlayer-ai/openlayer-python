# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import TypedDict, Annotated, Literal

from ..._utils import PropertyInfo

from typing import List, Union, Dict, Optional
from typing_extensions import Literal, TypedDict, Required, Annotated
from ..._types import FileTypes
from ..._utils import PropertyInfo
from ...types import shared_params

__all__ = ["TestResultListParams"]


class TestResultListParams(TypedDict, total=False):
    include_archived: Annotated[bool, PropertyInfo(alias="includeArchived")]
    """Include archived goals."""

    page: int
    """The page to return in a paginated query."""

    per_page: Annotated[int, PropertyInfo(alias="perPage")]
    """Maximum number of items to return per page."""

    status: Literal["running", "passing", "failing", "skipped", "error"]
    """Filter list of test results by status.

    Available statuses are `running`, `passing`, `failing`, `skipped`, and `error`.
    """

    type: Literal["integrity", "consistency", "performance", "fairness", "robustness"]
    """Filter objects by test type.

    Available types are `integrity`, `consistency`, `performance`, `fairness`, and
    `robustness`.
    """
