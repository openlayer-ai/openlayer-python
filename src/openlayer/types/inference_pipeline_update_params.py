# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Annotated, TypedDict

from .._utils import PropertyInfo

__all__ = ["InferencePipelineUpdateParams"]


class InferencePipelineUpdateParams(TypedDict, total=False):
    description: Optional[str]
    """The inference pipeline description."""

    name: str
    """The inference pipeline name."""

    reference_dataset_uri: Annotated[Optional[str], PropertyInfo(alias="referenceDatasetUri")]
    """The storage uri of your reference dataset.

    We recommend using the Python SDK or the UI to handle your reference dataset
    updates.
    """
