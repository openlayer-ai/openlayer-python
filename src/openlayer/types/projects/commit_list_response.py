# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from datetime import datetime
from typing_extensions import Literal

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = ["CommitListResponse", "_Meta", "Item", "ItemCommit", "ItemLinks"]


class _Meta(BaseModel):
    page: int
    """The current page."""

    per_page: int = FieldInfo(alias="perPage")
    """The number of items per page."""

    total_items: int = FieldInfo(alias="totalItems")
    """The total number of items."""

    total_pages: int = FieldInfo(alias="totalPages")
    """The total number of pages."""


class ItemCommit(BaseModel):
    id: str
    """The commit id."""

    author_id: str = FieldInfo(alias="authorId")
    """The author id of the commit."""

    file_size: Optional[int] = FieldInfo(alias="fileSize", default=None)
    """The size of the commit bundle in bytes."""

    message: str
    """The commit message."""

    ml_model_id: Optional[str] = FieldInfo(alias="mlModelId", default=None)
    """The model id."""

    storage_uri: str = FieldInfo(alias="storageUri")
    """The storage URI where the commit bundle is stored."""

    training_dataset_id: Optional[str] = FieldInfo(alias="trainingDatasetId", default=None)
    """The training dataset id."""

    validation_dataset_id: Optional[str] = FieldInfo(alias="validationDatasetId", default=None)
    """The validation dataset id."""

    date_created: Optional[datetime] = FieldInfo(alias="dateCreated", default=None)
    """The commit creation date."""

    git_commit_ref: Optional[str] = FieldInfo(alias="gitCommitRef", default=None)
    """The ref of the corresponding git commit."""

    git_commit_sha: Optional[int] = FieldInfo(alias="gitCommitSha", default=None)
    """The SHA of the corresponding git commit."""

    git_commit_url: Optional[str] = FieldInfo(alias="gitCommitUrl", default=None)
    """The URL of the corresponding git commit."""


class ItemLinks(BaseModel):
    app: str


class Item(BaseModel):
    id: str
    """The project version (commit) id."""

    commit: ItemCommit
    """The details of a commit (project version)."""

    date_archived: Optional[datetime] = FieldInfo(alias="dateArchived", default=None)
    """The commit archive date."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The project version (commit) creation date."""

    failing_goal_count: int = FieldInfo(alias="failingGoalCount")
    """The number of tests that are failing for the commit."""

    ml_model_id: Optional[str] = FieldInfo(alias="mlModelId", default=None)
    """The model id."""

    passing_goal_count: int = FieldInfo(alias="passingGoalCount")
    """The number of tests that are passing for the commit."""

    project_id: str = FieldInfo(alias="projectId")
    """The project id."""

    status: Literal["queued", "running", "paused", "failed", "completed", "unknown"]
    """The commit status.

    Initially, the commit is `queued`, then, it switches to `running`. Finally, it
    can be `paused`, `failed`, or `completed`.
    """

    status_message: Optional[str] = FieldInfo(alias="statusMessage", default=None)
    """The commit status message."""

    total_goal_count: int = FieldInfo(alias="totalGoalCount")
    """The total number of tests for the commit."""

    training_dataset_id: Optional[str] = FieldInfo(alias="trainingDatasetId", default=None)
    """The training dataset id."""

    validation_dataset_id: Optional[str] = FieldInfo(alias="validationDatasetId", default=None)
    """The validation dataset id."""

    archived: Optional[bool] = None
    """Whether the commit is archived."""

    deployment_status: Optional[str] = FieldInfo(alias="deploymentStatus", default=None)
    """The deployment status associated with the commit's model."""

    links: Optional[ItemLinks] = None


class CommitListResponse(BaseModel):
    api_meta: _Meta = FieldInfo(alias="_meta")

    items: List[Item]
