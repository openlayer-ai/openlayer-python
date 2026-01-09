# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Annotated, TypedDict

from .._types import SequenceNotStr
from .._utils import PropertyInfo

__all__ = ["TestListResultsParams"]


class TestListResultsParams(TypedDict, total=False):
    end_timestamp: Annotated[float, PropertyInfo(alias="endTimestamp")]
    """Filter for results that use data starting before the end timestamp."""

    include_insights: Annotated[bool, PropertyInfo(alias="includeInsights")]
    """Include the insights linked to each test result"""

    inference_pipeline_id: Annotated[Optional[str], PropertyInfo(alias="inferencePipelineId")]
    """Retrive test results for a specific inference pipeline."""

    page: int
    """The page to return in a paginated query."""

    per_page: Annotated[int, PropertyInfo(alias="perPage")]
    """Maximum number of items to return per page."""

    project_version_id: Annotated[Optional[str], PropertyInfo(alias="projectVersionId")]
    """Retrive test results for a specific project version."""

    start_timestamp: Annotated[float, PropertyInfo(alias="startTimestamp")]
    """Filter for results that use data ending after the start timestamp."""

    status: SequenceNotStr[str]
    """Filter by status(es)."""
