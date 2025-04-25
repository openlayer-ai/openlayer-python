# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = ["TestUpdateResponse"]


class TestUpdateResponse(BaseModel):
    __test__ = False
    task_result_id: Optional[str] = FieldInfo(alias="taskResultId", default=None)

    task_result_url: Optional[str] = FieldInfo(alias="taskResultUrl", default=None)
