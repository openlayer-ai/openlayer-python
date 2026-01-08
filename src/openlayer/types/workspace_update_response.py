# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from datetime import date, datetime
from typing_extensions import Literal

from pydantic import Field as FieldInfo

from .._models import BaseModel

__all__ = ["WorkspaceUpdateResponse", "MonthlyUsage"]


class MonthlyUsage(BaseModel):
    execution_time_ms: Optional[int] = FieldInfo(alias="executionTimeMs", default=None)

    month_year: Optional[date] = FieldInfo(alias="monthYear", default=None)

    prediction_count: Optional[int] = FieldInfo(alias="predictionCount", default=None)


class WorkspaceUpdateResponse(BaseModel):
    id: str
    """The workspace id."""

    creator_id: Optional[str] = FieldInfo(alias="creatorId", default=None)
    """The workspace creator id."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The workspace creation date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The workspace last updated date."""

    invite_count: int = FieldInfo(alias="inviteCount")
    """The number of invites in the workspace."""

    member_count: int = FieldInfo(alias="memberCount")
    """The number of members in the workspace."""

    name: str
    """The workspace name."""

    period_end_date: Optional[datetime] = FieldInfo(alias="periodEndDate", default=None)
    """The end date of the current billing period."""

    period_start_date: Optional[datetime] = FieldInfo(alias="periodStartDate", default=None)
    """The start date of the current billing period."""

    project_count: int = FieldInfo(alias="projectCount")
    """The number of projects in the workspace."""

    slug: str
    """The workspace slug."""

    status: Literal[
        "active", "past_due", "unpaid", "canceled", "incomplete", "incomplete_expired", "trialing", "paused"
    ]

    monthly_usage: Optional[List[MonthlyUsage]] = FieldInfo(alias="monthlyUsage", default=None)

    saml_only_access: Optional[bool] = FieldInfo(alias="samlOnlyAccess", default=None)
    """Whether the workspace only allows SAML authentication."""

    wildcard_domains: Optional[List[str]] = FieldInfo(alias="wildcardDomains", default=None)
