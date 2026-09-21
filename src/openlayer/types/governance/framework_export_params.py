# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Annotated, TypedDict

from ..._utils import PropertyInfo

__all__ = ["FrameworkExportParams"]


class FrameworkExportParams(TypedDict, total=False):
    project_id: Annotated[Optional[str], PropertyInfo(alias="projectId")]
    """Scope the export to this project. It must belong to the framework."""
