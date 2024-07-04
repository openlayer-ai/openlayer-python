# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Literal, Required, Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["InferencePipelineCreateParams"]


class InferencePipelineCreateParams(TypedDict, total=False):
    description: Required[Optional[str]]
    """The inference pipeline description."""

    name: Required[str]
    """The inference pipeline name."""

    reference_dataset_uri: Annotated[Optional[str], PropertyInfo(alias="referenceDatasetUri")]
    """The reference dataset URI."""

    storage_type: Annotated[Literal["local", "s3", "gcs", "azure"], PropertyInfo(alias="storageType")]
    """The storage type."""
