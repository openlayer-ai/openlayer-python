# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional

from ..._models import BaseModel

__all__ = ["TaskStatusResponse", "Outputs"]


class Outputs(BaseModel):
    storage_uri: Optional[str] = None
    """URI of the exported data in storage."""


class TaskStatusResponse(BaseModel):
    complete: bool
    """Whether the task has completed."""

    outputs: Optional[Outputs] = None
    """Output information, available when task is complete."""

    error: Optional[str] = None
    """Error message if the task failed."""
