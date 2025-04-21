# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Union, Optional
from datetime import datetime
from typing_extensions import Literal

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = ["TestCreateResponse", "Threshold", "ThresholdInsightParameter"]


class ThresholdInsightParameter(BaseModel):
    name: str
    """The name of the insight filter."""

    value: object


class Threshold(BaseModel):
    insight_name: Optional[str] = FieldInfo(alias="insightName", default=None)
    """The insight name to be evaluated."""

    insight_parameters: Optional[List[ThresholdInsightParameter]] = FieldInfo(alias="insightParameters", default=None)
    """The insight parameters. Required only for some test subtypes."""

    measurement: Optional[str] = None
    """The measurement to be evaluated."""

    operator: Optional[Literal["is", ">", ">=", "<", "<=", "!="]] = None
    """The operator to be used for the evaluation."""

    threshold_mode: Optional[Literal["automatic", "manual"]] = FieldInfo(alias="thresholdMode", default=None)
    """Whether to use automatic anomaly detection or manual thresholds"""

    value: Union[float, bool, str, List[str], None] = None
    """The value to be compared."""


class TestCreateResponse(BaseModel):
    __test__ = False
    id: str
    """The test id."""

    comment_count: int = FieldInfo(alias="commentCount")
    """The number of comments on the test."""

    creator_id: Optional[str] = FieldInfo(alias="creatorId", default=None)
    """The test creator id."""

    date_archived: Optional[datetime] = FieldInfo(alias="dateArchived", default=None)
    """The date the test was archived."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The creation date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The last updated date."""

    description: Optional[object] = None
    """The test description."""

    name: str
    """The test name."""

    number: int
    """The test number."""

    origin_project_version_id: Optional[str] = FieldInfo(alias="originProjectVersionId", default=None)
    """The project version (commit) id where the test was created."""

    subtype: str
    """The test subtype."""

    suggested: bool
    """Whether the test is suggested or user-created."""

    thresholds: List[Threshold]

    type: str
    """The test type."""

    archived: Optional[bool] = None
    """Whether the test is archived."""

    delay_window: Optional[float] = FieldInfo(alias="delayWindow", default=None)
    """The delay window in seconds. Only applies to tests that use production data."""

    evaluation_window: Optional[float] = FieldInfo(alias="evaluationWindow", default=None)
    """The evaluation window in seconds.

    Only applies to tests that use production data.
    """

    uses_ml_model: Optional[bool] = FieldInfo(alias="usesMlModel", default=None)
    """Whether the test uses an ML model."""

    uses_production_data: Optional[bool] = FieldInfo(alias="usesProductionData", default=None)
    """Whether the test uses production data (monitoring mode only)."""

    uses_reference_dataset: Optional[bool] = FieldInfo(alias="usesReferenceDataset", default=None)
    """Whether the test uses a reference dataset (monitoring mode only)."""

    uses_training_dataset: Optional[bool] = FieldInfo(alias="usesTrainingDataset", default=None)
    """Whether the test uses a training dataset."""

    uses_validation_dataset: Optional[bool] = FieldInfo(alias="usesValidationDataset", default=None)
    """Whether the test uses a validation dataset."""
