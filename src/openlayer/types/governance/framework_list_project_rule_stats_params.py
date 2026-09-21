# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["FrameworkListProjectRuleStatsParams"]


class FrameworkListProjectRuleStatsParams(TypedDict, total=False):
    asc: bool
    """Whether to sort in ascending order."""

    page: int
    """The page to return in a paginated query."""

    per_page: Annotated[int, PropertyInfo(alias="perPage")]
    """Maximum number of items to return per page."""

    sort_column: Annotated[
        Literal[
            "projectName",
            "total",
            "overallCompletion",
            "totalPassing",
            "totalFailing",
            "totalSkipped",
            "totalRunning",
            "totalError",
            "totalPending",
            "totalDueSoon",
        ],
        PropertyInfo(alias="sortColumn"),
    ]
    """The column to sort on."""
