# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from datetime import datetime

from pydantic import Field as FieldInfo

from .._models import BaseModel

__all__ = ["BackgroundTaskRetrieveResponse"]


class BackgroundTaskRetrieveResponse(BaseModel):
    id: str
    """The background task id."""

    complete: bool
    """Whether the task has finished. Check this before reading `outputs`."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """When the task was queued."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """When the task last reported progress."""

    name: str
    """The task's internal name, including the arguments it was queued with."""

    progress: float
    """How far along the task is, from 0 to 100."""

    error: Optional[str] = None
    """Why the task failed, or `null` if it has not failed."""

    outputs: Optional[object] = None
    """Whatever the task produced, keyed by name.

    `null` until the task completes. A framework export returns `storageUri` -- pass
    it to `GET /storage/presigned-url` to download the archive -- along with
    `filename`, `controlCount`, `evidenceCount` and `missingEvidenceCount`.
    """
