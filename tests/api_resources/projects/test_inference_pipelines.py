# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from tests.utils import assert_matches_type
from openlayer_test import Openlayer, AsyncOpenlayer
from openlayer_test.types.projects import InferencePipelineListResponse

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestInferencePipelines:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_list(self, client: Openlayer) -> None:
        inference_pipeline = client.projects.inference_pipelines.list(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

    @parametrize
    def test_method_list_with_all_params(self, client: Openlayer) -> None:
        inference_pipeline = client.projects.inference_pipelines.list(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="string",
            page=1,
            per_page=1,
        )
        assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

    @parametrize
    def test_raw_response_list(self, client: Openlayer) -> None:
        response = client.projects.inference_pipelines.with_raw_response.list(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        inference_pipeline = response.parse()
        assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

    @parametrize
    def test_streaming_response_list(self, client: Openlayer) -> None:
        with client.projects.inference_pipelines.with_streaming_response.list(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            inference_pipeline = response.parse()
            assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_list(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `id` but received ''"):
            client.projects.inference_pipelines.with_raw_response.list(
                "",
            )


class TestAsyncInferencePipelines:
    parametrize = pytest.mark.parametrize("async_client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    async def test_method_list(self, async_client: AsyncOpenlayer) -> None:
        inference_pipeline = await async_client.projects.inference_pipelines.list(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        inference_pipeline = await async_client.projects.inference_pipelines.list(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="string",
            page=1,
            per_page=1,
        )
        assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

    @parametrize
    async def test_raw_response_list(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.projects.inference_pipelines.with_raw_response.list(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        inference_pipeline = await response.parse()
        assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.projects.inference_pipelines.with_streaming_response.list(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            inference_pipeline = await response.parse()
            assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_list(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `id` but received ''"):
            await async_client.projects.inference_pipelines.with_raw_response.list(
                "",
            )
