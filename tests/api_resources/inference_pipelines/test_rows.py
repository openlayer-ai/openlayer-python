# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types.inference_pipelines import (
    RowListResponse,
    RowUpdateResponse,
    RowRetrieveResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestRows:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_retrieve(self, client: Openlayer) -> None:
        row = client.inference_pipelines.rows.retrieve(
            inference_id="inferenceId",
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RowRetrieveResponse, row, path=["response"])

    @parametrize
    def test_raw_response_retrieve(self, client: Openlayer) -> None:
        response = client.inference_pipelines.rows.with_raw_response.retrieve(
            inference_id="inferenceId",
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        row = response.parse()
        assert_matches_type(RowRetrieveResponse, row, path=["response"])

    @parametrize
    def test_streaming_response_retrieve(self, client: Openlayer) -> None:
        with client.inference_pipelines.rows.with_streaming_response.retrieve(
            inference_id="inferenceId",
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            row = response.parse()
            assert_matches_type(RowRetrieveResponse, row, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_retrieve(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            client.inference_pipelines.rows.with_raw_response.retrieve(
                inference_id="inferenceId",
                inference_pipeline_id="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_id` but received ''"):
            client.inference_pipelines.rows.with_raw_response.retrieve(
                inference_id="",
                inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            )

    @parametrize
    def test_method_update(self, client: Openlayer) -> None:
        row = client.inference_pipelines.rows.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            inference_id="inferenceId",
            row={},
        )
        assert_matches_type(RowUpdateResponse, row, path=["response"])

    @parametrize
    def test_method_update_with_all_params(self, client: Openlayer) -> None:
        row = client.inference_pipelines.rows.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            inference_id="inferenceId",
            row={},
            config={
                "ground_truth_column_name": "ground_truth",
                "human_feedback_column_name": "human_feedback",
                "inference_id_column_name": "id",
                "latency_column_name": "latency",
                "timestamp_column_name": "timestamp",
            },
        )
        assert_matches_type(RowUpdateResponse, row, path=["response"])

    @parametrize
    def test_raw_response_update(self, client: Openlayer) -> None:
        response = client.inference_pipelines.rows.with_raw_response.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            inference_id="inferenceId",
            row={},
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        row = response.parse()
        assert_matches_type(RowUpdateResponse, row, path=["response"])

    @parametrize
    def test_streaming_response_update(self, client: Openlayer) -> None:
        with client.inference_pipelines.rows.with_streaming_response.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            inference_id="inferenceId",
            row={},
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            row = response.parse()
            assert_matches_type(RowUpdateResponse, row, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_update(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            client.inference_pipelines.rows.with_raw_response.update(
                inference_pipeline_id="",
                inference_id="inferenceId",
                row={},
            )

    @parametrize
    def test_method_list(self, client: Openlayer) -> None:
        row = client.inference_pipelines.rows.list(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RowListResponse, row, path=["response"])

    @parametrize
    def test_method_list_with_all_params(self, client: Openlayer) -> None:
        row = client.inference_pipelines.rows.list(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            asc=True,
            page=1,
            per_page=1,
            sort_column="sortColumn",
            column_filters=[
                {
                    "measurement": "openlayer_token_set",
                    "operator": "contains_none",
                    "value": ["cat"],
                }
            ],
            exclude_row_id_list=[0],
            not_search_query_and=["string"],
            not_search_query_or=["string"],
            row_id_list=[0],
            search_query_and=["string"],
            search_query_or=["string"],
        )
        assert_matches_type(RowListResponse, row, path=["response"])

    @parametrize
    def test_raw_response_list(self, client: Openlayer) -> None:
        response = client.inference_pipelines.rows.with_raw_response.list(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        row = response.parse()
        assert_matches_type(RowListResponse, row, path=["response"])

    @parametrize
    def test_streaming_response_list(self, client: Openlayer) -> None:
        with client.inference_pipelines.rows.with_streaming_response.list(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            row = response.parse()
            assert_matches_type(RowListResponse, row, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_list(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            client.inference_pipelines.rows.with_raw_response.list(
                inference_pipeline_id="",
            )

    @parametrize
    def test_method_delete(self, client: Openlayer) -> None:
        row = client.inference_pipelines.rows.delete(
            inference_id="inferenceId",
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert row is None

    @parametrize
    def test_raw_response_delete(self, client: Openlayer) -> None:
        response = client.inference_pipelines.rows.with_raw_response.delete(
            inference_id="inferenceId",
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        row = response.parse()
        assert row is None

    @parametrize
    def test_streaming_response_delete(self, client: Openlayer) -> None:
        with client.inference_pipelines.rows.with_streaming_response.delete(
            inference_id="inferenceId",
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            row = response.parse()
            assert row is None

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_delete(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            client.inference_pipelines.rows.with_raw_response.delete(
                inference_id="inferenceId",
                inference_pipeline_id="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_id` but received ''"):
            client.inference_pipelines.rows.with_raw_response.delete(
                inference_id="",
                inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            )


class TestAsyncRows:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @parametrize
    async def test_method_retrieve(self, async_client: AsyncOpenlayer) -> None:
        row = await async_client.inference_pipelines.rows.retrieve(
            inference_id="inferenceId",
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RowRetrieveResponse, row, path=["response"])

    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.inference_pipelines.rows.with_raw_response.retrieve(
            inference_id="inferenceId",
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        row = await response.parse()
        assert_matches_type(RowRetrieveResponse, row, path=["response"])

    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.inference_pipelines.rows.with_streaming_response.retrieve(
            inference_id="inferenceId",
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            row = await response.parse()
            assert_matches_type(RowRetrieveResponse, row, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            await async_client.inference_pipelines.rows.with_raw_response.retrieve(
                inference_id="inferenceId",
                inference_pipeline_id="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_id` but received ''"):
            await async_client.inference_pipelines.rows.with_raw_response.retrieve(
                inference_id="",
                inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            )

    @parametrize
    async def test_method_update(self, async_client: AsyncOpenlayer) -> None:
        row = await async_client.inference_pipelines.rows.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            inference_id="inferenceId",
            row={},
        )
        assert_matches_type(RowUpdateResponse, row, path=["response"])

    @parametrize
    async def test_method_update_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        row = await async_client.inference_pipelines.rows.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            inference_id="inferenceId",
            row={},
            config={
                "ground_truth_column_name": "ground_truth",
                "human_feedback_column_name": "human_feedback",
                "inference_id_column_name": "id",
                "latency_column_name": "latency",
                "timestamp_column_name": "timestamp",
            },
        )
        assert_matches_type(RowUpdateResponse, row, path=["response"])

    @parametrize
    async def test_raw_response_update(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.inference_pipelines.rows.with_raw_response.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            inference_id="inferenceId",
            row={},
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        row = await response.parse()
        assert_matches_type(RowUpdateResponse, row, path=["response"])

    @parametrize
    async def test_streaming_response_update(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.inference_pipelines.rows.with_streaming_response.update(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            inference_id="inferenceId",
            row={},
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            row = await response.parse()
            assert_matches_type(RowUpdateResponse, row, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_update(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            await async_client.inference_pipelines.rows.with_raw_response.update(
                inference_pipeline_id="",
                inference_id="inferenceId",
                row={},
            )

    @parametrize
    async def test_method_list(self, async_client: AsyncOpenlayer) -> None:
        row = await async_client.inference_pipelines.rows.list(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RowListResponse, row, path=["response"])

    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        row = await async_client.inference_pipelines.rows.list(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            asc=True,
            page=1,
            per_page=1,
            sort_column="sortColumn",
            column_filters=[
                {
                    "measurement": "openlayer_token_set",
                    "operator": "contains_none",
                    "value": ["cat"],
                }
            ],
            exclude_row_id_list=[0],
            not_search_query_and=["string"],
            not_search_query_or=["string"],
            row_id_list=[0],
            search_query_and=["string"],
            search_query_or=["string"],
        )
        assert_matches_type(RowListResponse, row, path=["response"])

    @parametrize
    async def test_raw_response_list(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.inference_pipelines.rows.with_raw_response.list(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        row = await response.parse()
        assert_matches_type(RowListResponse, row, path=["response"])

    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.inference_pipelines.rows.with_streaming_response.list(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            row = await response.parse()
            assert_matches_type(RowListResponse, row, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_list(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            await async_client.inference_pipelines.rows.with_raw_response.list(
                inference_pipeline_id="",
            )

    @parametrize
    async def test_method_delete(self, async_client: AsyncOpenlayer) -> None:
        row = await async_client.inference_pipelines.rows.delete(
            inference_id="inferenceId",
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert row is None

    @parametrize
    async def test_raw_response_delete(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.inference_pipelines.rows.with_raw_response.delete(
            inference_id="inferenceId",
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        row = await response.parse()
        assert row is None

    @parametrize
    async def test_streaming_response_delete(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.inference_pipelines.rows.with_streaming_response.delete(
            inference_id="inferenceId",
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            row = await response.parse()
            assert row is None

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_delete(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            await async_client.inference_pipelines.rows.with_raw_response.delete(
                inference_id="inferenceId",
                inference_pipeline_id="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_id` but received ''"):
            await async_client.inference_pipelines.rows.with_raw_response.delete(
                inference_id="",
                inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            )
