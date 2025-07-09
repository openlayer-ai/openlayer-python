# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types.projects import (
    InferencePipelineListResponse,
    InferencePipelineCreateResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestInferencePipelines:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_create(self, client: Openlayer) -> None:
        inference_pipeline = client.projects.inference_pipelines.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This pipeline is used for production.",
            name="production",
        )
        assert_matches_type(InferencePipelineCreateResponse, inference_pipeline, path=["response"])

    @parametrize
    def test_method_create_with_all_params(self, client: Openlayer) -> None:
        inference_pipeline = client.projects.inference_pipelines.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This pipeline is used for production.",
            name="production",
            project={
                "name": "My Project",
                "task_type": "llm-base",
                "description": "My project description.",
            },
            workspace={
                "name": "Openlayer",
                "slug": "openlayer",
                "invite_code": "inviteCode",
                "saml_only_access": True,
                "wildcard_domains": ["string"],
            },
        )
        assert_matches_type(InferencePipelineCreateResponse, inference_pipeline, path=["response"])

    @parametrize
    def test_raw_response_create(self, client: Openlayer) -> None:
        response = client.projects.inference_pipelines.with_raw_response.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This pipeline is used for production.",
            name="production",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        inference_pipeline = response.parse()
        assert_matches_type(InferencePipelineCreateResponse, inference_pipeline, path=["response"])

    @parametrize
    def test_streaming_response_create(self, client: Openlayer) -> None:
        with client.projects.inference_pipelines.with_streaming_response.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This pipeline is used for production.",
            name="production",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            inference_pipeline = response.parse()
            assert_matches_type(InferencePipelineCreateResponse, inference_pipeline, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_create(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `project_id` but received ''"):
            client.projects.inference_pipelines.with_raw_response.create(
                project_id="",
                description="This pipeline is used for production.",
                name="production",
            )

    @parametrize
    def test_method_list(self, client: Openlayer) -> None:
        inference_pipeline = client.projects.inference_pipelines.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

    @parametrize
    def test_method_list_with_all_params(self, client: Openlayer) -> None:
        inference_pipeline = client.projects.inference_pipelines.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="name",
            page=1,
            per_page=1,
        )
        assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

    @parametrize
    def test_raw_response_list(self, client: Openlayer) -> None:
        response = client.projects.inference_pipelines.with_raw_response.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        inference_pipeline = response.parse()
        assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

    @parametrize
    def test_streaming_response_list(self, client: Openlayer) -> None:
        with client.projects.inference_pipelines.with_streaming_response.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            inference_pipeline = response.parse()
            assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_list(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `project_id` but received ''"):
            client.projects.inference_pipelines.with_raw_response.list(
                project_id="",
            )


class TestAsyncInferencePipelines:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @parametrize
    async def test_method_create(self, async_client: AsyncOpenlayer) -> None:
        inference_pipeline = await async_client.projects.inference_pipelines.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This pipeline is used for production.",
            name="production",
        )
        assert_matches_type(InferencePipelineCreateResponse, inference_pipeline, path=["response"])

    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        inference_pipeline = await async_client.projects.inference_pipelines.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This pipeline is used for production.",
            name="production",
            project={
                "name": "My Project",
                "task_type": "llm-base",
                "description": "My project description.",
            },
            workspace={
                "name": "Openlayer",
                "slug": "openlayer",
                "invite_code": "inviteCode",
                "saml_only_access": True,
                "wildcard_domains": ["string"],
            },
        )
        assert_matches_type(InferencePipelineCreateResponse, inference_pipeline, path=["response"])

    @parametrize
    async def test_raw_response_create(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.projects.inference_pipelines.with_raw_response.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This pipeline is used for production.",
            name="production",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        inference_pipeline = await response.parse()
        assert_matches_type(InferencePipelineCreateResponse, inference_pipeline, path=["response"])

    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.projects.inference_pipelines.with_streaming_response.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This pipeline is used for production.",
            name="production",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            inference_pipeline = await response.parse()
            assert_matches_type(InferencePipelineCreateResponse, inference_pipeline, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_create(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `project_id` but received ''"):
            await async_client.projects.inference_pipelines.with_raw_response.create(
                project_id="",
                description="This pipeline is used for production.",
                name="production",
            )

    @parametrize
    async def test_method_list(self, async_client: AsyncOpenlayer) -> None:
        inference_pipeline = await async_client.projects.inference_pipelines.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        inference_pipeline = await async_client.projects.inference_pipelines.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="name",
            page=1,
            per_page=1,
        )
        assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

    @parametrize
    async def test_raw_response_list(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.projects.inference_pipelines.with_raw_response.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        inference_pipeline = await response.parse()
        assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.projects.inference_pipelines.with_streaming_response.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            inference_pipeline = await response.parse()
            assert_matches_type(InferencePipelineListResponse, inference_pipeline, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_list(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `project_id` but received ''"):
            await async_client.projects.inference_pipelines.with_raw_response.list(
                project_id="",
            )
