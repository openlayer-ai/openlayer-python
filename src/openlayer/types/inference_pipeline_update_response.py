# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from datetime import date, datetime
from typing_extensions import Literal

from pydantic import Field as FieldInfo

from .._models import BaseModel

__all__ = [
    "InferencePipelineUpdateResponse",
    "Links",
    "Project",
    "ProjectLinks",
    "ProjectGitRepo",
    "Workspace",
    "WorkspaceMonthlyUsage",
]


class Links(BaseModel):
    app: str


class ProjectLinks(BaseModel):
    app: str


class ProjectGitRepo(BaseModel):
    id: str

    date_connected: datetime = FieldInfo(alias="dateConnected")

    date_updated: datetime = FieldInfo(alias="dateUpdated")

    git_account_id: str = FieldInfo(alias="gitAccountId")

    git_id: int = FieldInfo(alias="gitId")

    name: str

    private: bool

    project_id: str = FieldInfo(alias="projectId")

    slug: str

    url: str

    branch: Optional[str] = None

    root_dir: Optional[str] = FieldInfo(alias="rootDir", default=None)


class Project(BaseModel):
    id: str
    """The project id."""

    creator_id: Optional[str] = FieldInfo(alias="creatorId", default=None)
    """The project creator id."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The project creation date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The project last updated date."""

    development_goal_count: int = FieldInfo(alias="developmentGoalCount")
    """The number of tests in the development mode of the project."""

    goal_count: int = FieldInfo(alias="goalCount")
    """The total number of tests in the project."""

    inference_pipeline_count: int = FieldInfo(alias="inferencePipelineCount")
    """The number of inference pipelines in the project."""

    links: ProjectLinks
    """Links to the project."""

    monitoring_goal_count: int = FieldInfo(alias="monitoringGoalCount")
    """The number of tests in the monitoring mode of the project."""

    name: str
    """The project name."""

    source: Optional[Literal["web", "api", "null"]] = None
    """The source of the project."""

    task_type: Literal["llm-base", "tabular-classification", "tabular-regression", "text-classification"] = FieldInfo(
        alias="taskType"
    )
    """The task type of the project."""

    version_count: int = FieldInfo(alias="versionCount")
    """The number of versions (commits) in the project."""

    workspace_id: Optional[str] = FieldInfo(alias="workspaceId", default=None)
    """The workspace id."""

    description: Optional[str] = None
    """The project description."""

    git_repo: Optional[ProjectGitRepo] = FieldInfo(alias="gitRepo", default=None)


class WorkspaceMonthlyUsage(BaseModel):
    execution_time_ms: Optional[int] = FieldInfo(alias="executionTimeMs", default=None)

    month_year: Optional[date] = FieldInfo(alias="monthYear", default=None)

    prediction_count: Optional[int] = FieldInfo(alias="predictionCount", default=None)


class Workspace(BaseModel):
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

    monthly_usage: Optional[List[WorkspaceMonthlyUsage]] = FieldInfo(alias="monthlyUsage", default=None)

    saml_only_access: Optional[bool] = FieldInfo(alias="samlOnlyAccess", default=None)
    """Whether the workspace only allows SAML authentication."""

    wildcard_domains: Optional[List[str]] = FieldInfo(alias="wildcardDomains", default=None)


class InferencePipelineUpdateResponse(BaseModel):
    id: str
    """The inference pipeline id."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The creation date."""

    date_last_evaluated: Optional[datetime] = FieldInfo(alias="dateLastEvaluated", default=None)
    """The last test evaluation date."""

    date_last_sample_received: Optional[datetime] = FieldInfo(alias="dateLastSampleReceived", default=None)
    """The last data sample received date."""

    date_of_next_evaluation: Optional[datetime] = FieldInfo(alias="dateOfNextEvaluation", default=None)
    """The next test evaluation date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The last updated date."""

    description: Optional[str] = None
    """The inference pipeline description."""

    failing_goal_count: int = FieldInfo(alias="failingGoalCount")
    """The number of tests failing."""

    links: Links

    name: str
    """The inference pipeline name."""

    passing_goal_count: int = FieldInfo(alias="passingGoalCount")
    """The number of tests passing."""

    project_id: str = FieldInfo(alias="projectId")
    """The project id."""

    status: Literal["queued", "running", "paused", "failed", "completed", "unknown"]
    """The status of test evaluation for the inference pipeline."""

    status_message: Optional[str] = FieldInfo(alias="statusMessage", default=None)
    """The status message of test evaluation for the inference pipeline."""

    total_goal_count: int = FieldInfo(alias="totalGoalCount")
    """The total number of tests."""

    project: Optional[Project] = None

    workspace: Optional[Workspace] = None

    workspace_id: Optional[str] = FieldInfo(alias="workspaceId", default=None)
    """The workspace id."""
