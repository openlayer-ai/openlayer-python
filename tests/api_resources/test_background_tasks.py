# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types import BackgroundTaskRetrieveResponse

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestBackgroundTasks:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_retrieve(self, client: Openlayer) -> None:
        background_task = client.background_tasks.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(BackgroundTaskRetrieveResponse, background_task, path=["response"])

    @parametrize
    def test_raw_response_retrieve(self, client: Openlayer) -> None:
        response = client.background_tasks.with_raw_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        background_task = response.parse()
        assert_matches_type(BackgroundTaskRetrieveResponse, background_task, path=["response"])

    @parametrize
    def test_streaming_response_retrieve(self, client: Openlayer) -> None:
        with client.background_tasks.with_streaming_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            background_task = response.parse()
            assert_matches_type(BackgroundTaskRetrieveResponse, background_task, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_retrieve(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `task_id` but received ''"):
            client.background_tasks.with_raw_response.retrieve(
                "",
            )


class TestAsyncBackgroundTasks:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @parametrize
    async def test_method_retrieve(self, async_client: AsyncOpenlayer) -> None:
        background_task = await async_client.background_tasks.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(BackgroundTaskRetrieveResponse, background_task, path=["response"])

    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.background_tasks.with_raw_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        background_task = await response.parse()
        assert_matches_type(BackgroundTaskRetrieveResponse, background_task, path=["response"])

    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.background_tasks.with_streaming_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            background_task = await response.parse()
            assert_matches_type(BackgroundTaskRetrieveResponse, background_task, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `task_id` but received ''"):
            await async_client.background_tasks.with_raw_response.retrieve(
                "",
            )
