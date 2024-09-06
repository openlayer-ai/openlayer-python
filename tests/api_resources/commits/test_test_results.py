# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types.commits import TestResultListResponse

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestTestResults:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_list(self, client: Openlayer) -> None:
        test_result = client.commits.test_results.list(
            project_version_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(TestResultListResponse, test_result, path=["response"])

    @parametrize
    def test_method_list_with_all_params(self, client: Openlayer) -> None:
        test_result = client.commits.test_results.list(
            project_version_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            include_archived=True,
            page=1,
            per_page=1,
            status="running",
            type="integrity",
        )
        assert_matches_type(TestResultListResponse, test_result, path=["response"])

    @parametrize
    def test_raw_response_list(self, client: Openlayer) -> None:
        response = client.commits.test_results.with_raw_response.list(
            project_version_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        test_result = response.parse()
        assert_matches_type(TestResultListResponse, test_result, path=["response"])

    @parametrize
    def test_streaming_response_list(self, client: Openlayer) -> None:
        with client.commits.test_results.with_streaming_response.list(
            project_version_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            test_result = response.parse()
            assert_matches_type(TestResultListResponse, test_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_list(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `project_version_id` but received ''"):
            client.commits.test_results.with_raw_response.list(
                project_version_id="",
            )


class TestAsyncTestResults:
    parametrize = pytest.mark.parametrize("async_client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    async def test_method_list(self, async_client: AsyncOpenlayer) -> None:
        test_result = await async_client.commits.test_results.list(
            project_version_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(TestResultListResponse, test_result, path=["response"])

    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        test_result = await async_client.commits.test_results.list(
            project_version_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            include_archived=True,
            page=1,
            per_page=1,
            status="running",
            type="integrity",
        )
        assert_matches_type(TestResultListResponse, test_result, path=["response"])

    @parametrize
    async def test_raw_response_list(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.commits.test_results.with_raw_response.list(
            project_version_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        test_result = await response.parse()
        assert_matches_type(TestResultListResponse, test_result, path=["response"])

    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.commits.test_results.with_streaming_response.list(
            project_version_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            test_result = await response.parse()
            assert_matches_type(TestResultListResponse, test_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_list(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `project_version_id` but received ''"):
            await async_client.commits.test_results.with_raw_response.list(
                project_version_id="",
            )
