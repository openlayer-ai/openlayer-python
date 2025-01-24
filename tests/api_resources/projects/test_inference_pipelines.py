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
                "creator_id": "589ece63-49a2-41b4-98e1-10547761d4b0",
                "date_created": "2024-03-22T11:31:01.185Z",
                "date_updated": "2024-03-22T11:31:01.185Z",
                "development_goal_count": 5,
                "goal_count": 10,
                "inference_pipeline_count": 1,
                "monitoring_goal_count": 5,
                "name": "My Project",
                "task_type": "llm-base",
                "version_count": 2,
                "workspace_id": "055fddb1-261f-4654-8598-f6347ee46a09",
                "description": "My project description.",
                "git_repo": {
                    "id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "date_connected": "2019-12-27T18:11:19.117Z",
                    "date_updated": "2019-12-27T18:11:19.117Z",
                    "git_account_id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "git_id": 0,
                    "name": "name",
                    "private": True,
                    "project_id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "slug": "slug",
                    "url": "url",
                    "branch": "branch",
                    "root_dir": "rootDir",
                },
            },
            workspace={
                "creator_id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                "date_created": "2019-12-27T18:11:19.117Z",
                "date_updated": "2019-12-27T18:11:19.117Z",
                "invite_count": 0,
                "member_count": 0,
                "name": "Openlayer",
                "period_end_date": "2019-12-27T18:11:19.117Z",
                "period_start_date": "2019-12-27T18:11:19.117Z",
                "project_count": 0,
                "slug": "openlayer",
                "invite_code": "inviteCode",
                "monthly_usage": [
                    {
                        "execution_time_ms": 0,
                        "month_year": "2019-12-27",
                        "prediction_count": 0,
                    }
                ],
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
    parametrize = pytest.mark.parametrize("async_client", [False, True], indirect=True, ids=["loose", "strict"])

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
                "creator_id": "589ece63-49a2-41b4-98e1-10547761d4b0",
                "date_created": "2024-03-22T11:31:01.185Z",
                "date_updated": "2024-03-22T11:31:01.185Z",
                "development_goal_count": 5,
                "goal_count": 10,
                "inference_pipeline_count": 1,
                "monitoring_goal_count": 5,
                "name": "My Project",
                "task_type": "llm-base",
                "version_count": 2,
                "workspace_id": "055fddb1-261f-4654-8598-f6347ee46a09",
                "description": "My project description.",
                "git_repo": {
                    "id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "date_connected": "2019-12-27T18:11:19.117Z",
                    "date_updated": "2019-12-27T18:11:19.117Z",
                    "git_account_id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "git_id": 0,
                    "name": "name",
                    "private": True,
                    "project_id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "slug": "slug",
                    "url": "url",
                    "branch": "branch",
                    "root_dir": "rootDir",
                },
            },
            workspace={
                "creator_id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                "date_created": "2019-12-27T18:11:19.117Z",
                "date_updated": "2019-12-27T18:11:19.117Z",
                "invite_count": 0,
                "member_count": 0,
                "name": "Openlayer",
                "period_end_date": "2019-12-27T18:11:19.117Z",
                "period_start_date": "2019-12-27T18:11:19.117Z",
                "project_count": 0,
                "slug": "openlayer",
                "invite_code": "inviteCode",
                "monthly_usage": [
                    {
                        "execution_time_ms": 0,
                        "month_year": "2019-12-27",
                        "prediction_count": 0,
                    }
                ],
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
