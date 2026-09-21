# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Annotated, TypedDict

from ..._types import SequenceNotStr
from ..._utils import PropertyInfo

__all__ = ["FrameworkListParams"]


class FrameworkListParams(TypedDict, total=False):
    asc: bool
    """Whether to sort in ascending order."""

    completion_operator: Annotated[Literal["is", ">", ">=", "<", "<=", "!="], PropertyInfo(alias="completionOperator")]
    """How to compare each framework's completion percentage with `completionValue`.

    Must be sent together with `completionValue`.
    """

    completion_value: Annotated[int, PropertyInfo(alias="completionValue")]
    """The completion percentage to compare against, from 0 to 100."""

    enabled: bool
    """Only include frameworks that are enabled (or disabled)."""

    include_rule_stats: Annotated[bool, PropertyInfo(alias="includeRuleStats")]
    """
    Whether to include a `ruleStats` object on each framework, with its rule result
    status counts and its per-project completion buckets. Computed over the returned
    page only.
    """

    page: int
    """The page to return in a paginated query."""

    per_page: Annotated[int, PropertyInfo(alias="perPage")]
    """Maximum number of items to return per page."""

    project_id: Annotated[str, PropertyInfo(alias="projectId")]
    """Only include items that apply to this project."""

    search_query: Annotated[str, PropertyInfo(alias="searchQuery")]
    """Filter by a free-text search over names and descriptions."""

    sort_column: Annotated[
        Literal["name", "enabled", "dateCreated", "dateUpdated", "overallCompletion", "projectCompletionBuckets"],
        PropertyInfo(alias="sortColumn"),
    ]
    """The column to sort on."""

    tags: SequenceNotStr[str]
    """Only include frameworks carrying all of these tags."""
