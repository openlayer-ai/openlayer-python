# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import List, Union, Iterable, Optional
from typing_extensions import Literal, Required, Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["TestUpdateParams", "Payload", "PayloadThreshold", "PayloadThresholdInsightParameter"]


class TestUpdateParams(TypedDict, total=False):
    payloads: Required[Iterable[Payload]]


class PayloadThresholdInsightParameter(TypedDict, total=False):
    name: Required[str]
    """The name of the insight filter."""

    value: Required[object]


class PayloadThreshold(TypedDict, total=False):
    insight_name: Annotated[
        Literal[
            "characterLength",
            "classImbalance",
            "expectColumnAToBeInColumnB",
            "columnAverage",
            "columnDrift",
            "columnValuesMatch",
            "confidenceDistribution",
            "conflictingLabelRowCount",
            "containsPii",
            "containsValidUrl",
            "correlatedFeatures",
            "customMetric",
            "duplicateRowCount",
            "emptyFeatures",
            "featureDrift",
            "featureProfile",
            "greatExpectations",
            "groupByColumnStatsCheck",
            "illFormedRowCount",
            "isCode",
            "isJson",
            "llmRubricV2",
            "labelDrift",
            "metrics",
            "newCategories",
            "newLabels",
            "nullRowCount",
            "ppScore",
            "quasiConstantFeatures",
            "sentenceLength",
            "sizeRatio",
            "specialCharacters",
            "stringValidation",
            "trainValLeakageRowCount",
        ],
        PropertyInfo(alias="insightName"),
    ]
    """The insight name to be evaluated."""

    insight_parameters: Annotated[
        Optional[Iterable[PayloadThresholdInsightParameter]], PropertyInfo(alias="insightParameters")
    ]
    """The insight parameters.

    Required only for some test subtypes. For example, for tests that require a
    column name, the insight parameters will be [{'name': 'column_name', 'value':
    'Age'}]
    """

    measurement: str
    """The measurement to be evaluated."""

    operator: Literal["is", ">", ">=", "<", "<=", "!="]
    """The operator to be used for the evaluation."""

    threshold_mode: Annotated[Literal["automatic", "manual"], PropertyInfo(alias="thresholdMode")]
    """Whether to use automatic anomaly detection or manual thresholds"""

    value: Union[float, bool, str, List[str]]
    """The value to be compared."""


class Payload(TypedDict, total=False):
    id: Required[str]

    archived: bool
    """Whether the test is archived."""

    description: Optional[object]
    """The test description."""

    name: str
    """The test name."""

    suggested: Literal[False]

    thresholds: Iterable[PayloadThreshold]
