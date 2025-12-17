# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Required, Annotated, TypedDict

from .._utils import PropertyInfo

__all__ = ["TestEvaluateParams"]


class TestEvaluateParams(TypedDict, total=False):
    end_timestamp: Required[Annotated[int, PropertyInfo(alias="endTimestamp")]]
    """End timestamp in seconds (Unix epoch)"""

    start_timestamp: Required[Annotated[int, PropertyInfo(alias="startTimestamp")]]
    """Start timestamp in seconds (Unix epoch)"""

    inference_pipeline_id: Annotated[str, PropertyInfo(alias="inferencePipelineId")]
    """ID of the inference pipeline to evaluate.

    If not provided, all inference pipelines the test applies to will be evaluated.
    """

    overwrite_results: Annotated[bool, PropertyInfo(alias="overwriteResults")]
    """Whether to overwrite existing test results"""
