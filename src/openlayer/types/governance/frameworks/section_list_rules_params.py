# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, Annotated, TypedDict

from ...._utils import PropertyInfo

__all__ = ["SectionListRulesParams"]


class SectionListRulesParams(TypedDict, total=False):
    framework_id: Required[Annotated[str, PropertyInfo(alias="frameworkId")]]

    include_results: Annotated[bool, PropertyInfo(alias="includeResults")]
    """Whether to include each rule's results inline, in a `results` array."""

    include_subsection_rules: Annotated[bool, PropertyInfo(alias="includeSubsectionRules")]
    """Whether to also include the rules mapped to the section's subsections."""

    page: int
    """The page to return in a paginated query."""

    per_page: Annotated[int, PropertyInfo(alias="perPage")]
    """Maximum number of items to return per page."""

    project_id: Annotated[str, PropertyInfo(alias="projectId")]
    """Only include items that apply to this project."""

    status: Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"]
    """Only include items whose rule result has this compliance status."""
