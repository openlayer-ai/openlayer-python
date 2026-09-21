# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["RuleResultListParams"]


class RuleResultListParams(TypedDict, total=False):
    enabled_framework_only: Annotated[bool, PropertyInfo(alias="enabledFrameworkOnly")]
    """Only include items belonging to at least one enabled framework."""

    framework_id: Annotated[str, PropertyInfo(alias="frameworkId")]
    """Only include items belonging to this framework."""

    include_unframed: Annotated[bool, PropertyInfo(alias="includeUnframed")]
    """Whether to include rules that are not part of any framework."""

    page: int
    """The page to return in a paginated query."""

    per_page: Annotated[int, PropertyInfo(alias="perPage")]
    """Maximum number of items to return per page."""

    project_id: Annotated[str, PropertyInfo(alias="projectId")]
    """Only include items that apply to this project."""

    rule_id: Annotated[str, PropertyInfo(alias="ruleId")]
    """Only include results of this rule."""

    scope: Literal["project", "workspace"]
    """Only include rules with this scope."""

    search_query: Annotated[str, PropertyInfo(alias="searchQuery")]
    """Filter by a free-text search over names and descriptions."""

    status: Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"]
    """Only include items whose rule result has this compliance status."""

    type: Literal["platform", "evidence"]
    """Only include rules of this type."""
