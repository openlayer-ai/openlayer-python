# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import List, Union, Iterable, Optional
from typing_extensions import Literal, Required, Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["TestCreateParams", "Threshold", "ThresholdInsightParameter"]


class TestCreateParams(TypedDict, total=False):
    description: Required[Optional[object]]
    """The test description."""

    name: Required[str]
    """The test name."""

    subtype: Required[str]
    """The test subtype."""

    thresholds: Required[Iterable[Threshold]]

    type: Required[str]
    """The test type."""

    archived: bool
    """Whether the test is archived."""

    delay_window: Annotated[Optional[float], PropertyInfo(alias="delayWindow")]
    """The delay window in seconds. Only applies to tests that use production data."""

    evaluation_window: Annotated[Optional[float], PropertyInfo(alias="evaluationWindow")]
    """The evaluation window in seconds.

    Only applies to tests that use production data.
    """

    uses_ml_model: Annotated[bool, PropertyInfo(alias="usesMlModel")]
    """Whether the test uses an ML model."""

    uses_production_data: Annotated[bool, PropertyInfo(alias="usesProductionData")]
    """Whether the test uses production data (monitoring mode only)."""

    uses_reference_dataset: Annotated[bool, PropertyInfo(alias="usesReferenceDataset")]
    """Whether the test uses a reference dataset (monitoring mode only)."""

    uses_training_dataset: Annotated[bool, PropertyInfo(alias="usesTrainingDataset")]
    """Whether the test uses a training dataset."""

    uses_validation_dataset: Annotated[bool, PropertyInfo(alias="usesValidationDataset")]
    """Whether the test uses a validation dataset."""


class ThresholdInsightParameter(TypedDict, total=False):
    name: Required[str]
    """The name of the insight filter."""

    value: Required[object]


class Threshold(TypedDict, total=False):
    insight_name: Annotated[str, PropertyInfo(alias="insightName")]
    """The insight name to be evaluated."""

    insight_parameters: Annotated[
        Optional[Iterable[ThresholdInsightParameter]], PropertyInfo(alias="insightParameters")
    ]
    """The insight parameters. Required only for some test subtypes."""

    measurement: str
    """The measurement to be evaluated."""

    operator: Literal["is", ">", ">=", "<", "<=", "!="]
    """The operator to be used for the evaluation."""

    threshold_mode: Annotated[Literal["automatic", "manual"], PropertyInfo(alias="thresholdMode")]
    """Whether to use automatic anomaly detection or manual thresholds"""

    value: Union[float, bool, str, List[str]]
    """The value to be compared."""
