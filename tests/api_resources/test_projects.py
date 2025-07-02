# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types import ProjectListResponse, ProjectCreateResponse

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestProjects:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_create(self, client: Openlayer) -> None:
        project = client.projects.create(
            name="My Project",
            task_type="llm-base",
        )
        assert_matches_type(ProjectCreateResponse, project, path=["response"])

    @parametrize
    def test_method_create_with_all_params(self, client: Openlayer) -> None:
        project = client.projects.create(
            name="My Project",
            task_type="llm-base",
            description="My project description.",
        )
        assert_matches_type(ProjectCreateResponse, project, path=["response"])

    @parametrize
    def test_raw_response_create(self, client: Openlayer) -> None:
        response = client.projects.with_raw_response.create(
            name="My Project",
            task_type="llm-base",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        project = response.parse()
        assert_matches_type(ProjectCreateResponse, project, path=["response"])

    @parametrize
    def test_streaming_response_create(self, client: Openlayer) -> None:
        with client.projects.with_streaming_response.create(
            name="My Project",
            task_type="llm-base",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            project = response.parse()
            assert_matches_type(ProjectCreateResponse, project, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_method_list(self, client: Openlayer) -> None:
        project = client.projects.list()
        assert_matches_type(ProjectListResponse, project, path=["response"])

    @parametrize
    def test_method_list_with_all_params(self, client: Openlayer) -> None:
        project = client.projects.list(
            name="name",
            page=1,
            per_page=1,
            task_type="llm-base",
        )
        assert_matches_type(ProjectListResponse, project, path=["response"])

    @parametrize
    def test_raw_response_list(self, client: Openlayer) -> None:
        response = client.projects.with_raw_response.list()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        project = response.parse()
        assert_matches_type(ProjectListResponse, project, path=["response"])

    @parametrize
    def test_streaming_response_list(self, client: Openlayer) -> None:
        with client.projects.with_streaming_response.list() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            project = response.parse()
            assert_matches_type(ProjectListResponse, project, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncProjects:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @parametrize
    async def test_method_create(self, async_client: AsyncOpenlayer) -> None:
        project = await async_client.projects.create(
            name="My Project",
            task_type="llm-base",
        )
        assert_matches_type(ProjectCreateResponse, project, path=["response"])

    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        project = await async_client.projects.create(
            name="My Project",
            task_type="llm-base",
            description="My project description.",
        )
        assert_matches_type(ProjectCreateResponse, project, path=["response"])

    @parametrize
    async def test_raw_response_create(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.projects.with_raw_response.create(
            name="My Project",
            task_type="llm-base",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        project = await response.parse()
        assert_matches_type(ProjectCreateResponse, project, path=["response"])

    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.projects.with_streaming_response.create(
            name="My Project",
            task_type="llm-base",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            project = await response.parse()
            assert_matches_type(ProjectCreateResponse, project, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_method_list(self, async_client: AsyncOpenlayer) -> None:
        project = await async_client.projects.list()
        assert_matches_type(ProjectListResponse, project, path=["response"])

    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        project = await async_client.projects.list(
            name="name",
            page=1,
            per_page=1,
            task_type="llm-base",
        )
        assert_matches_type(ProjectListResponse, project, path=["response"])

    @parametrize
    async def test_raw_response_list(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.projects.with_raw_response.list()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        project = await response.parse()
        assert_matches_type(ProjectListResponse, project, path=["response"])

    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.projects.with_streaming_response.list() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            project = await response.parse()
            assert_matches_type(ProjectListResponse, project, path=["response"])

        assert cast(Any, response.is_closed) is True
