# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types import CommitCreateResponse

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestCommits:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_create(self, client: Openlayer) -> None:
        commit = client.commits.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            commit={"message": "Updated the prompt."},
            storage_uri="s3://...",
        )
        assert_matches_type(CommitCreateResponse, commit, path=["response"])

    @parametrize
    def test_method_create_with_all_params(self, client: Openlayer) -> None:
        commit = client.commits.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            commit={"message": "Updated the prompt."},
            storage_uri="s3://...",
            archived=False,
            deployment_status="Deployed",
        )
        assert_matches_type(CommitCreateResponse, commit, path=["response"])

    @parametrize
    def test_raw_response_create(self, client: Openlayer) -> None:
        response = client.commits.with_raw_response.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            commit={"message": "Updated the prompt."},
            storage_uri="s3://...",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        commit = response.parse()
        assert_matches_type(CommitCreateResponse, commit, path=["response"])

    @parametrize
    def test_streaming_response_create(self, client: Openlayer) -> None:
        with client.commits.with_streaming_response.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            commit={"message": "Updated the prompt."},
            storage_uri="s3://...",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            commit = response.parse()
            assert_matches_type(CommitCreateResponse, commit, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_create(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `project_id` but received ''"):
            client.commits.with_raw_response.create(
                project_id="",
                commit={"message": "Updated the prompt."},
                storage_uri="s3://...",
            )


class TestAsyncCommits:
    parametrize = pytest.mark.parametrize("async_client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    async def test_method_create(self, async_client: AsyncOpenlayer) -> None:
        commit = await async_client.commits.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            commit={"message": "Updated the prompt."},
            storage_uri="s3://...",
        )
        assert_matches_type(CommitCreateResponse, commit, path=["response"])

    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        commit = await async_client.commits.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            commit={"message": "Updated the prompt."},
            storage_uri="s3://...",
            archived=False,
            deployment_status="Deployed",
        )
        assert_matches_type(CommitCreateResponse, commit, path=["response"])

    @parametrize
    async def test_raw_response_create(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.commits.with_raw_response.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            commit={"message": "Updated the prompt."},
            storage_uri="s3://...",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        commit = await response.parse()
        assert_matches_type(CommitCreateResponse, commit, path=["response"])

    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.commits.with_streaming_response.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            commit={"message": "Updated the prompt."},
            storage_uri="s3://...",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            commit = await response.parse()
            assert_matches_type(CommitCreateResponse, commit, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_create(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `project_id` but received ''"):
            await async_client.commits.with_raw_response.create(
                project_id="",
                commit={"message": "Updated the prompt."},
                storage_uri="s3://...",
            )
