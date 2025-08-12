# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional

from ..._models import BaseModel

__all__ = ["ExportDataResponse", "Outputs"]


class Outputs(BaseModel):
    storage_uri: Optional[str] = None
    """URI of the exported data in storage."""


class ExportDataResponse(BaseModel):
    task_result_url: str
    """URL to poll for task completion status."""

    complete: Optional[bool] = None
    """Whether the export task has completed."""

    outputs: Optional[Outputs] = None
    """Output information, available when task is complete."""
