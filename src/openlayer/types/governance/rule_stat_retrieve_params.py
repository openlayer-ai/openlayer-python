# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["RuleStatRetrieveParams"]


class RuleStatRetrieveParams(TypedDict, total=False):
    framework_id: Annotated[str, PropertyInfo(alias="frameworkId")]
    """Only include items belonging to this framework."""

    project_id: Annotated[str, PropertyInfo(alias="projectId")]
    """Only include items that apply to this project."""
