# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Annotated, TypedDict

from ..._types import SequenceNotStr
from ..._utils import PropertyInfo

__all__ = ["RuleListParams"]


class RuleListParams(TypedDict, total=False):
    asc: bool
    """Whether to sort in ascending order."""

    assignee_id: Annotated[str, PropertyInfo(alias="assigneeId")]
    """Only include rules assigned to this user."""

    deactivated: bool
    """Only include rules that are deactivated (or active)."""

    enabled_framework_only: Annotated[bool, PropertyInfo(alias="enabledFrameworkOnly")]
    """Only include items belonging to at least one enabled framework."""

    framework_id: Annotated[str, PropertyInfo(alias="frameworkId")]
    """Only include items belonging to this framework."""

    group: Literal["open", "excluded", "done"]
    """Only include rules in one bucket of the compliance workflow.

    `open` covers rules that still need attention, `done` covers rules that are
    fully satisfied, and `excluded` covers rules that have been deactivated.
    """

    include_results: Annotated[bool, PropertyInfo(alias="includeResults")]
    """Whether to include each rule's results inline, in a `results` array."""

    include_unframed: Annotated[bool, PropertyInfo(alias="includeUnframed")]
    """Whether to include rules that are not part of any framework."""

    page: int
    """The page to return in a paginated query."""

    per_page: Annotated[int, PropertyInfo(alias="perPage")]
    """Maximum number of items to return per page."""

    project_id: Annotated[str, PropertyInfo(alias="projectId")]
    """Only include items that apply to this project."""

    scope: Literal["project", "workspace"]
    """Only include rules with this scope."""

    search_query: Annotated[str, PropertyInfo(alias="searchQuery")]
    """Filter by a free-text search over names and descriptions."""

    sort_by: Annotated[Literal["name", "status", "frameworks", "scope", "dateCreated"], PropertyInfo(alias="sortBy")]
    """The field to sort on."""

    status: Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"]
    """Only include items whose rule result has this compliance status."""

    tags: SequenceNotStr[str]
    """Only include rules carrying all of these rule tags.

    Pass tag ids, which you can look up with
    [List rule tags](/api-reference/rest/governance/list-rule-tags).
    """

    type: Literal["platform", "evidence"]
    """Only include rules of this type."""
