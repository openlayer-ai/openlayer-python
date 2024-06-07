# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from datetime import datetime
from typing_extensions import Literal

from pydantic import Field as FieldInfo

from .._models import BaseModel

__all__ = ["ProjectListResponse", "_Meta", "Item", "ItemLinks", "ItemGitRepo"]


class _Meta(BaseModel):
    page: int
    """The current page."""

    per_page: int = FieldInfo(alias="perPage")
    """The number of items per page."""

    total_items: int = FieldInfo(alias="totalItems")
    """The total number of items."""

    total_pages: int = FieldInfo(alias="totalPages")
    """The total number of pages."""


class ItemLinks(BaseModel):
    app: str


class ItemGitRepo(BaseModel):
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


class Item(BaseModel):
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

    links: ItemLinks
    """Links to the project."""

    monitoring_goal_count: int = FieldInfo(alias="monitoringGoalCount")
    """The number of tests in the monitoring mode of the project."""

    name: str
    """The project name."""

    sample: bool
    """Whether the project is a sample project or a user-created project."""

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

    git_repo: Optional[ItemGitRepo] = FieldInfo(alias="gitRepo", default=None)

    slack_channel_id: Optional[str] = FieldInfo(alias="slackChannelId", default=None)
    """The slack channel id connected to the project."""

    slack_channel_name: Optional[str] = FieldInfo(alias="slackChannelName", default=None)
    """The slack channel connected to the project."""

    slack_channel_notifications_enabled: Optional[bool] = FieldInfo(
        alias="slackChannelNotificationsEnabled", default=None
    )
    """Whether slack channel notifications are enabled for the project."""

    unread_notification_count: Optional[int] = FieldInfo(alias="unreadNotificationCount", default=None)
    """The number of unread notifications in the project."""


class ProjectListResponse(BaseModel):
    api_meta: _Meta = FieldInfo(alias="_meta")

    items: List[Item]
