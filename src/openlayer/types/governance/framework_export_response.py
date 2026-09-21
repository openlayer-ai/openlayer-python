# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = ["FrameworkExportResponse"]


class FrameworkExportResponse(BaseModel):
    task_result_id: str = FieldInfo(alias="taskResultId")
    """The background task id, for `GET /background-tasks/{taskId}`."""

    task_result_url: str = FieldInfo(alias="taskResultUrl")
    """The path to poll for this export's status and result.

    Already `/v1`-prefixed, so it is relative to the API host rather than to the
    `/v1` base url -- or just pass `taskResultId` to
    `GET /background-tasks/{taskId}`.
    """
