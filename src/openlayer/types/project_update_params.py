# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Annotated, TypedDict

from .._types import SequenceNotStr
from .._utils import PropertyInfo

__all__ = ["ProjectUpdateParams"]


class ProjectUpdateParams(TypedDict, total=False):
    data_retention_days: Annotated[Optional[int], PropertyInfo(alias="dataRetentionDays")]
    """Number of days to retain monitoring data for this project.

    Null means data is retained indefinitely.
    """

    description: Optional[str]
    """The project description."""

    model_developer: Annotated[Optional[str], PropertyInfo(alias="modelDeveloper")]
    """Who developed the model used in this project."""

    model_types: Annotated[Optional[SequenceNotStr[str]], PropertyInfo(alias="modelTypes")]
    """The kinds of model used in this project."""

    name: str
    """The project name."""

    purpose: Optional[str]
    """What the system in this project is intended to do."""
