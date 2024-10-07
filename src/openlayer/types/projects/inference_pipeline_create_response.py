# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from datetime import datetime
from typing_extensions import Literal

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = ["InferencePipelineCreateResponse", "Links"]


class Links(BaseModel):
    app: str


class InferencePipelineCreateResponse(BaseModel):
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
