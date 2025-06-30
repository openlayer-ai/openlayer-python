# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types.storage import PresignedURLCreateResponse

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestPresignedURL:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_create(self, client: Openlayer) -> None:
        presigned_url = client.storage.presigned_url.create(
            object_name="objectName",
        )
        assert_matches_type(PresignedURLCreateResponse, presigned_url, path=["response"])

    @parametrize
    def test_raw_response_create(self, client: Openlayer) -> None:
        response = client.storage.presigned_url.with_raw_response.create(
            object_name="objectName",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        presigned_url = response.parse()
        assert_matches_type(PresignedURLCreateResponse, presigned_url, path=["response"])

    @parametrize
    def test_streaming_response_create(self, client: Openlayer) -> None:
        with client.storage.presigned_url.with_streaming_response.create(
            object_name="objectName",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            presigned_url = response.parse()
            assert_matches_type(PresignedURLCreateResponse, presigned_url, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncPresignedURL:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @parametrize
    async def test_method_create(self, async_client: AsyncOpenlayer) -> None:
        presigned_url = await async_client.storage.presigned_url.create(
            object_name="objectName",
        )
        assert_matches_type(PresignedURLCreateResponse, presigned_url, path=["response"])

    @parametrize
    async def test_raw_response_create(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.storage.presigned_url.with_raw_response.create(
            object_name="objectName",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        presigned_url = await response.parse()
        assert_matches_type(PresignedURLCreateResponse, presigned_url, path=["response"])

    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.storage.presigned_url.with_streaming_response.create(
            object_name="objectName",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            presigned_url = await response.parse()
            assert_matches_type(PresignedURLCreateResponse, presigned_url, path=["response"])

        assert cast(Any, response.is_closed) is True
