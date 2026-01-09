# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types import (
    TestEvaluateResponse,
    TestListResultsResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestTests:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_evaluate(self, client: Openlayer) -> None:
        test = client.tests.evaluate(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            end_timestamp=1700006400,
            start_timestamp=1699920000,
        )
        assert_matches_type(TestEvaluateResponse, test, path=["response"])

    @parametrize
    def test_method_evaluate_with_all_params(self, client: Openlayer) -> None:
        test = client.tests.evaluate(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            end_timestamp=1700006400,
            start_timestamp=1699920000,
            inference_pipeline_id="123e4567-e89b-12d3-a456-426614174000",
            overwrite_results=False,
        )
        assert_matches_type(TestEvaluateResponse, test, path=["response"])

    @parametrize
    def test_raw_response_evaluate(self, client: Openlayer) -> None:
        response = client.tests.with_raw_response.evaluate(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            end_timestamp=1700006400,
            start_timestamp=1699920000,
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        test = response.parse()
        assert_matches_type(TestEvaluateResponse, test, path=["response"])

    @parametrize
    def test_streaming_response_evaluate(self, client: Openlayer) -> None:
        with client.tests.with_streaming_response.evaluate(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            end_timestamp=1700006400,
            start_timestamp=1699920000,
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            test = response.parse()
            assert_matches_type(TestEvaluateResponse, test, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_evaluate(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `test_id` but received ''"):
            client.tests.with_raw_response.evaluate(
                test_id="",
                end_timestamp=1700006400,
                start_timestamp=1699920000,
            )

    @parametrize
    def test_method_list_results(self, client: Openlayer) -> None:
        test = client.tests.list_results(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(TestListResultsResponse, test, path=["response"])

    @parametrize
    def test_method_list_results_with_all_params(self, client: Openlayer) -> None:
        test = client.tests.list_results(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            end_timestamp=0,
            include_insights=True,
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            page=1,
            per_page=1,
            project_version_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            start_timestamp=0,
            status=["string"],
        )
        assert_matches_type(TestListResultsResponse, test, path=["response"])

    @parametrize
    def test_raw_response_list_results(self, client: Openlayer) -> None:
        response = client.tests.with_raw_response.list_results(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        test = response.parse()
        assert_matches_type(TestListResultsResponse, test, path=["response"])

    @parametrize
    def test_streaming_response_list_results(self, client: Openlayer) -> None:
        with client.tests.with_streaming_response.list_results(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            test = response.parse()
            assert_matches_type(TestListResultsResponse, test, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_list_results(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `test_id` but received ''"):
            client.tests.with_raw_response.list_results(
                test_id="",
            )


class TestAsyncTests:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @parametrize
    async def test_method_evaluate(self, async_client: AsyncOpenlayer) -> None:
        test = await async_client.tests.evaluate(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            end_timestamp=1700006400,
            start_timestamp=1699920000,
        )
        assert_matches_type(TestEvaluateResponse, test, path=["response"])

    @parametrize
    async def test_method_evaluate_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        test = await async_client.tests.evaluate(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            end_timestamp=1700006400,
            start_timestamp=1699920000,
            inference_pipeline_id="123e4567-e89b-12d3-a456-426614174000",
            overwrite_results=False,
        )
        assert_matches_type(TestEvaluateResponse, test, path=["response"])

    @parametrize
    async def test_raw_response_evaluate(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.tests.with_raw_response.evaluate(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            end_timestamp=1700006400,
            start_timestamp=1699920000,
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        test = await response.parse()
        assert_matches_type(TestEvaluateResponse, test, path=["response"])

    @parametrize
    async def test_streaming_response_evaluate(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.tests.with_streaming_response.evaluate(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            end_timestamp=1700006400,
            start_timestamp=1699920000,
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            test = await response.parse()
            assert_matches_type(TestEvaluateResponse, test, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_evaluate(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `test_id` but received ''"):
            await async_client.tests.with_raw_response.evaluate(
                test_id="",
                end_timestamp=1700006400,
                start_timestamp=1699920000,
            )

    @parametrize
    async def test_method_list_results(self, async_client: AsyncOpenlayer) -> None:
        test = await async_client.tests.list_results(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(TestListResultsResponse, test, path=["response"])

    @parametrize
    async def test_method_list_results_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        test = await async_client.tests.list_results(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            end_timestamp=0,
            include_insights=True,
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            page=1,
            per_page=1,
            project_version_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            start_timestamp=0,
            status=["string"],
        )
        assert_matches_type(TestListResultsResponse, test, path=["response"])

    @parametrize
    async def test_raw_response_list_results(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.tests.with_raw_response.list_results(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        test = await response.parse()
        assert_matches_type(TestListResultsResponse, test, path=["response"])

    @parametrize
    async def test_streaming_response_list_results(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.tests.with_streaming_response.list_results(
            test_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            test = await response.parse()
            assert_matches_type(TestListResultsResponse, test, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_list_results(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `test_id` but received ''"):
            await async_client.tests.with_raw_response.list_results(
                test_id="",
            )
