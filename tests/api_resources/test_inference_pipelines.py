# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types import (
    InferencePipelineUpdateResponse,
    InferencePipelineRetrieveResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestInferencePipelines:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_retrieve(self, client: Openlayer) -> None:
        inference_pipeline = client.inference_pipelines.retrieve(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(InferencePipelineRetrieveResponse, inference_pipeline, path=["response"])

    @parametrize
    def test_method_retrieve_with_all_params(self, client: Openlayer) -> None:
        inference_pipeline = client.inference_pipelines.retrieve(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            expand=["project"],
        )
        assert_matches_type(InferencePipelineRetrieveResponse, inference_pipeline, path=["response"])

    @parametrize
    def test_raw_response_retrieve(self, client: Openlayer) -> None:
        response = client.inference_pipelines.with_raw_response.retrieve(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        inference_pipeline = response.parse()
        assert_matches_type(InferencePipelineRetrieveResponse, inference_pipeline, path=["response"])

    @parametrize
    def test_streaming_response_retrieve(self, client: Openlayer) -> None:
        with client.inference_pipelines.with_streaming_response.retrieve(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            inference_pipeline = response.parse()
            assert_matches_type(InferencePipelineRetrieveResponse, inference_pipeline, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_retrieve(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            client.inference_pipelines.with_raw_response.retrieve(
                inference_pipeline_id="",
            )

    @parametrize
    def test_method_update(self, client: Openlayer) -> None:
        inference_pipeline = client.inference_pipelines.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(InferencePipelineUpdateResponse, inference_pipeline, path=["response"])

    @parametrize
    def test_method_update_with_all_params(self, client: Openlayer) -> None:
        inference_pipeline = client.inference_pipelines.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This pipeline is used for production.",
            name="production",
            reference_dataset_uri="referenceDatasetUri",
        )
        assert_matches_type(InferencePipelineUpdateResponse, inference_pipeline, path=["response"])

    @parametrize
    def test_raw_response_update(self, client: Openlayer) -> None:
        response = client.inference_pipelines.with_raw_response.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        inference_pipeline = response.parse()
        assert_matches_type(InferencePipelineUpdateResponse, inference_pipeline, path=["response"])

    @parametrize
    def test_streaming_response_update(self, client: Openlayer) -> None:
        with client.inference_pipelines.with_streaming_response.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            inference_pipeline = response.parse()
            assert_matches_type(InferencePipelineUpdateResponse, inference_pipeline, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_update(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            client.inference_pipelines.with_raw_response.update(
                inference_pipeline_id="",
            )

    @parametrize
    def test_method_delete(self, client: Openlayer) -> None:
        inference_pipeline = client.inference_pipelines.delete(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert inference_pipeline is None

    @parametrize
    def test_raw_response_delete(self, client: Openlayer) -> None:
        response = client.inference_pipelines.with_raw_response.delete(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        inference_pipeline = response.parse()
        assert inference_pipeline is None

    @parametrize
    def test_streaming_response_delete(self, client: Openlayer) -> None:
        with client.inference_pipelines.with_streaming_response.delete(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            inference_pipeline = response.parse()
            assert inference_pipeline is None

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_delete(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            client.inference_pipelines.with_raw_response.delete(
                "",
            )


class TestAsyncInferencePipelines:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @parametrize
    async def test_method_retrieve(self, async_client: AsyncOpenlayer) -> None:
        inference_pipeline = await async_client.inference_pipelines.retrieve(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(InferencePipelineRetrieveResponse, inference_pipeline, path=["response"])

    @parametrize
    async def test_method_retrieve_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        inference_pipeline = await async_client.inference_pipelines.retrieve(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            expand=["project"],
        )
        assert_matches_type(InferencePipelineRetrieveResponse, inference_pipeline, path=["response"])

    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.inference_pipelines.with_raw_response.retrieve(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        inference_pipeline = await response.parse()
        assert_matches_type(InferencePipelineRetrieveResponse, inference_pipeline, path=["response"])

    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.inference_pipelines.with_streaming_response.retrieve(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            inference_pipeline = await response.parse()
            assert_matches_type(InferencePipelineRetrieveResponse, inference_pipeline, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            await async_client.inference_pipelines.with_raw_response.retrieve(
                inference_pipeline_id="",
            )

    @parametrize
    async def test_method_update(self, async_client: AsyncOpenlayer) -> None:
        inference_pipeline = await async_client.inference_pipelines.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(InferencePipelineUpdateResponse, inference_pipeline, path=["response"])

    @parametrize
    async def test_method_update_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        inference_pipeline = await async_client.inference_pipelines.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This pipeline is used for production.",
            name="production",
            reference_dataset_uri="referenceDatasetUri",
        )
        assert_matches_type(InferencePipelineUpdateResponse, inference_pipeline, path=["response"])

    @parametrize
    async def test_raw_response_update(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.inference_pipelines.with_raw_response.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        inference_pipeline = await response.parse()
        assert_matches_type(InferencePipelineUpdateResponse, inference_pipeline, path=["response"])

    @parametrize
    async def test_streaming_response_update(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.inference_pipelines.with_streaming_response.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            inference_pipeline = await response.parse()
            assert_matches_type(InferencePipelineUpdateResponse, inference_pipeline, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_update(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            await async_client.inference_pipelines.with_raw_response.update(
                inference_pipeline_id="",
            )

    @parametrize
    async def test_method_delete(self, async_client: AsyncOpenlayer) -> None:
        inference_pipeline = await async_client.inference_pipelines.delete(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert inference_pipeline is None

    @parametrize
    async def test_raw_response_delete(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.inference_pipelines.with_raw_response.delete(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        inference_pipeline = await response.parse()
        assert inference_pipeline is None

    @parametrize
    async def test_streaming_response_delete(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.inference_pipelines.with_streaming_response.delete(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            inference_pipeline = await response.parse()
            assert inference_pipeline is None

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_delete(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            await async_client.inference_pipelines.with_raw_response.delete(
                "",
            )
