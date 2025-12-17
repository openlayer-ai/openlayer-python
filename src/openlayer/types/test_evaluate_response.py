# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List

from pydantic import Field as FieldInfo

from .._models import BaseModel

__all__ = ["TestEvaluateResponse", "Task"]


class Task(BaseModel):
    pipeline_id: str = FieldInfo(alias="pipelineId")
    """ID of the inference pipeline this task is for"""

    task_result_id: str = FieldInfo(alias="taskResultId")
    """ID of the background task"""

    task_result_url: str = FieldInfo(alias="taskResultUrl")
    """URL to check the status of this background task"""


class TestEvaluateResponse(BaseModel):
    __test__ = False
    message: str

    pipeline_count: int = FieldInfo(alias="pipelineCount")
    """Number of inference pipelines the test was queued for evaluation on"""

    requested_end_timestamp: int = FieldInfo(alias="requestedEndTimestamp")
    """The end timestamp you requested (in seconds)"""

    requested_start_timestamp: int = FieldInfo(alias="requestedStartTimestamp")
    """The start timestamp you requested (in seconds)"""

    tasks: List[Task]
    """Array of background task information for each pipeline evaluation"""
