# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types.inference_pipelines import DataStreamResponse

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestData:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_stream(self, client: Openlayer) -> None:
        data = client.inference_pipelines.data.stream(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            config={"output_column_name": "output"},
            rows=[
                {
                    "user_query": "bar",
                    "output": "bar",
                    "tokens": "bar",
                    "cost": "bar",
                    "timestamp": "bar",
                }
            ],
        )
        assert_matches_type(DataStreamResponse, data, path=["response"])

    @parametrize
    def test_method_stream_with_all_params(self, client: Openlayer) -> None:
        data = client.inference_pipelines.data.stream(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            config={
                "output_column_name": "output",
                "context_column_name": "context",
                "cost_column_name": "cost",
                "ground_truth_column_name": "ground_truth",
                "inference_id_column_name": "id",
                "input_variable_names": ["user_query"],
                "latency_column_name": "latency",
                "metadata": {},
                "num_of_token_column_name": "tokens",
                "prompt": [
                    {
                        "content": "{{ user_query }}",
                        "role": "user",
                    }
                ],
                "question_column_name": "question",
                "timestamp_column_name": "timestamp",
            },
            rows=[
                {
                    "user_query": "bar",
                    "output": "bar",
                    "tokens": "bar",
                    "cost": "bar",
                    "timestamp": "bar",
                }
            ],
        )
        assert_matches_type(DataStreamResponse, data, path=["response"])

    @parametrize
    def test_raw_response_stream(self, client: Openlayer) -> None:
        response = client.inference_pipelines.data.with_raw_response.stream(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            config={"output_column_name": "output"},
            rows=[
                {
                    "user_query": "bar",
                    "output": "bar",
                    "tokens": "bar",
                    "cost": "bar",
                    "timestamp": "bar",
                }
            ],
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        data = response.parse()
        assert_matches_type(DataStreamResponse, data, path=["response"])

    @parametrize
    def test_streaming_response_stream(self, client: Openlayer) -> None:
        with client.inference_pipelines.data.with_streaming_response.stream(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            config={"output_column_name": "output"},
            rows=[
                {
                    "user_query": "bar",
                    "output": "bar",
                    "tokens": "bar",
                    "cost": "bar",
                    "timestamp": "bar",
                }
            ],
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            data = response.parse()
            assert_matches_type(DataStreamResponse, data, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_stream(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            client.inference_pipelines.data.with_raw_response.stream(
                inference_pipeline_id="",
                config={"output_column_name": "output"},
                rows=[
                    {
                        "user_query": "bar",
                        "output": "bar",
                        "tokens": "bar",
                        "cost": "bar",
                        "timestamp": "bar",
                    }
                ],
            )


class TestAsyncData:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @parametrize
    async def test_method_stream(self, async_client: AsyncOpenlayer) -> None:
        data = await async_client.inference_pipelines.data.stream(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            config={"output_column_name": "output"},
            rows=[
                {
                    "user_query": "bar",
                    "output": "bar",
                    "tokens": "bar",
                    "cost": "bar",
                    "timestamp": "bar",
                }
            ],
        )
        assert_matches_type(DataStreamResponse, data, path=["response"])

    @parametrize
    async def test_method_stream_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        data = await async_client.inference_pipelines.data.stream(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            config={
                "output_column_name": "output",
                "context_column_name": "context",
                "cost_column_name": "cost",
                "ground_truth_column_name": "ground_truth",
                "inference_id_column_name": "id",
                "input_variable_names": ["user_query"],
                "latency_column_name": "latency",
                "metadata": {},
                "num_of_token_column_name": "tokens",
                "prompt": [
                    {
                        "content": "{{ user_query }}",
                        "role": "user",
                    }
                ],
                "question_column_name": "question",
                "timestamp_column_name": "timestamp",
            },
            rows=[
                {
                    "user_query": "bar",
                    "output": "bar",
                    "tokens": "bar",
                    "cost": "bar",
                    "timestamp": "bar",
                }
            ],
        )
        assert_matches_type(DataStreamResponse, data, path=["response"])

    @parametrize
    async def test_raw_response_stream(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.inference_pipelines.data.with_raw_response.stream(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            config={"output_column_name": "output"},
            rows=[
                {
                    "user_query": "bar",
                    "output": "bar",
                    "tokens": "bar",
                    "cost": "bar",
                    "timestamp": "bar",
                }
            ],
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        data = await response.parse()
        assert_matches_type(DataStreamResponse, data, path=["response"])

    @parametrize
    async def test_streaming_response_stream(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.inference_pipelines.data.with_streaming_response.stream(
            inference_pipeline_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            config={"output_column_name": "output"},
            rows=[
                {
                    "user_query": "bar",
                    "output": "bar",
                    "tokens": "bar",
                    "cost": "bar",
                    "timestamp": "bar",
                }
            ],
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            data = await response.parse()
            assert_matches_type(DataStreamResponse, data, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_stream(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `inference_pipeline_id` but received ''"):
            await async_client.inference_pipelines.data.with_raw_response.stream(
                inference_pipeline_id="",
                config={"output_column_name": "output"},
                rows=[
                    {
                        "user_query": "bar",
                        "output": "bar",
                        "tokens": "bar",
                        "cost": "bar",
                        "timestamp": "bar",
                    }
                ],
            )
