# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Literal, Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["TestListParams"]


class TestListParams(TypedDict, total=False):
    include_archived: Annotated[bool, PropertyInfo(alias="includeArchived")]
    """Filter for archived tests."""

    origin_version_id: Annotated[Optional[str], PropertyInfo(alias="originVersionId")]
    """Retrive tests created by a specific project version."""

    page: int
    """The page to return in a paginated query."""

    per_page: Annotated[int, PropertyInfo(alias="perPage")]
    """Maximum number of items to return per page."""

    suggested: bool
    """Filter for suggested tests."""

    type: Literal["integrity", "consistency", "performance", "fairness", "robustness"]
    """Filter objects by test type.

    Available types are `integrity`, `consistency`, `performance`, `fairness`, and
    `robustness`.
    """

    uses_production_data: Annotated[Optional[bool], PropertyInfo(alias="usesProductionData")]
    """Retrive tests with usesProductionData (monitoring)."""
