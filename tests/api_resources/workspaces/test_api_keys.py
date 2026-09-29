# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer._utils import parse_datetime
from openlayer.types.workspaces import (
    APIKeyListResponse,
    APIKeyCreateResponse,
    APIKeyRotateResponse,
    APIKeyUpdateResponse,
    APIKeyRetrieveResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestAPIKeys:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_create(self, client: Openlayer) -> None:
        api_key = client.workspaces.api_keys.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(APIKeyCreateResponse, api_key, path=["response"])

    @parametrize
    def test_method_create_with_all_params(self, client: Openlayer) -> None:
        api_key = client.workspaces.api_keys.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            expires_at=parse_datetime("2027-01-01T00:00:00Z"),
            name="Secret Key",
        )
        assert_matches_type(APIKeyCreateResponse, api_key, path=["response"])

    @parametrize
    def test_raw_response_create(self, client: Openlayer) -> None:
        response = client.workspaces.api_keys.with_raw_response.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        api_key = response.parse()
        assert_matches_type(APIKeyCreateResponse, api_key, path=["response"])

    @parametrize
    def test_streaming_response_create(self, client: Openlayer) -> None:
        with client.workspaces.api_keys.with_streaming_response.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            api_key = response.parse()
            assert_matches_type(APIKeyCreateResponse, api_key, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_create(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            client.workspaces.api_keys.with_raw_response.create(
                workspace_id="",
            )

    @parametrize
    def test_method_retrieve(self, client: Openlayer) -> None:
        api_key = client.workspaces.api_keys.retrieve(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(APIKeyRetrieveResponse, api_key, path=["response"])

    @parametrize
    def test_raw_response_retrieve(self, client: Openlayer) -> None:
        response = client.workspaces.api_keys.with_raw_response.retrieve(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        api_key = response.parse()
        assert_matches_type(APIKeyRetrieveResponse, api_key, path=["response"])

    @parametrize
    def test_streaming_response_retrieve(self, client: Openlayer) -> None:
        with client.workspaces.api_keys.with_streaming_response.retrieve(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            api_key = response.parse()
            assert_matches_type(APIKeyRetrieveResponse, api_key, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_retrieve(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            client.workspaces.api_keys.with_raw_response.retrieve(
                api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                workspace_id="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `api_key_id` but received ''"):
            client.workspaces.api_keys.with_raw_response.retrieve(
                api_key_id="",
                workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            )

    @parametrize
    def test_method_update(self, client: Openlayer) -> None:
        api_key = client.workspaces.api_keys.update(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(APIKeyUpdateResponse, api_key, path=["response"])

    @parametrize
    def test_method_update_with_all_params(self, client: Openlayer) -> None:
        api_key = client.workspaces.api_keys.update(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="Secret Key",
        )
        assert_matches_type(APIKeyUpdateResponse, api_key, path=["response"])

    @parametrize
    def test_raw_response_update(self, client: Openlayer) -> None:
        response = client.workspaces.api_keys.with_raw_response.update(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        api_key = response.parse()
        assert_matches_type(APIKeyUpdateResponse, api_key, path=["response"])

    @parametrize
    def test_streaming_response_update(self, client: Openlayer) -> None:
        with client.workspaces.api_keys.with_streaming_response.update(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            api_key = response.parse()
            assert_matches_type(APIKeyUpdateResponse, api_key, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_update(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            client.workspaces.api_keys.with_raw_response.update(
                api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                workspace_id="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `api_key_id` but received ''"):
            client.workspaces.api_keys.with_raw_response.update(
                api_key_id="",
                workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            )

    @parametrize
    def test_method_list(self, client: Openlayer) -> None:
        api_key = client.workspaces.api_keys.list(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(APIKeyListResponse, api_key, path=["response"])

    @parametrize
    def test_raw_response_list(self, client: Openlayer) -> None:
        response = client.workspaces.api_keys.with_raw_response.list(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        api_key = response.parse()
        assert_matches_type(APIKeyListResponse, api_key, path=["response"])

    @parametrize
    def test_streaming_response_list(self, client: Openlayer) -> None:
        with client.workspaces.api_keys.with_streaming_response.list(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            api_key = response.parse()
            assert_matches_type(APIKeyListResponse, api_key, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_list(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            client.workspaces.api_keys.with_raw_response.list(
                "",
            )

    @parametrize
    def test_method_delete(self, client: Openlayer) -> None:
        api_key = client.workspaces.api_keys.delete(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert api_key is None

    @parametrize
    def test_raw_response_delete(self, client: Openlayer) -> None:
        response = client.workspaces.api_keys.with_raw_response.delete(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        api_key = response.parse()
        assert api_key is None

    @parametrize
    def test_streaming_response_delete(self, client: Openlayer) -> None:
        with client.workspaces.api_keys.with_streaming_response.delete(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            api_key = response.parse()
            assert api_key is None

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_delete(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            client.workspaces.api_keys.with_raw_response.delete(
                api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                workspace_id="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `api_key_id` but received ''"):
            client.workspaces.api_keys.with_raw_response.delete(
                api_key_id="",
                workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            )

    @parametrize
    def test_method_rotate(self, client: Openlayer) -> None:
        api_key = client.workspaces.api_keys.rotate(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(APIKeyRotateResponse, api_key, path=["response"])

    @parametrize
    def test_method_rotate_with_all_params(self, client: Openlayer) -> None:
        api_key = client.workspaces.api_keys.rotate(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            expires_at=parse_datetime("2027-01-01T00:00:00Z"),
            grace_period_hours=24,
        )
        assert_matches_type(APIKeyRotateResponse, api_key, path=["response"])

    @parametrize
    def test_raw_response_rotate(self, client: Openlayer) -> None:
        response = client.workspaces.api_keys.with_raw_response.rotate(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        api_key = response.parse()
        assert_matches_type(APIKeyRotateResponse, api_key, path=["response"])

    @parametrize
    def test_streaming_response_rotate(self, client: Openlayer) -> None:
        with client.workspaces.api_keys.with_streaming_response.rotate(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            api_key = response.parse()
            assert_matches_type(APIKeyRotateResponse, api_key, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_rotate(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            client.workspaces.api_keys.with_raw_response.rotate(
                api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                workspace_id="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `api_key_id` but received ''"):
            client.workspaces.api_keys.with_raw_response.rotate(
                api_key_id="",
                workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            )


class TestAsyncAPIKeys:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @parametrize
    async def test_method_create(self, async_client: AsyncOpenlayer) -> None:
        api_key = await async_client.workspaces.api_keys.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(APIKeyCreateResponse, api_key, path=["response"])

    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        api_key = await async_client.workspaces.api_keys.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            expires_at=parse_datetime("2027-01-01T00:00:00Z"),
            name="Secret Key",
        )
        assert_matches_type(APIKeyCreateResponse, api_key, path=["response"])

    @parametrize
    async def test_raw_response_create(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.workspaces.api_keys.with_raw_response.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        api_key = await response.parse()
        assert_matches_type(APIKeyCreateResponse, api_key, path=["response"])

    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.workspaces.api_keys.with_streaming_response.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            api_key = await response.parse()
            assert_matches_type(APIKeyCreateResponse, api_key, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_create(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            await async_client.workspaces.api_keys.with_raw_response.create(
                workspace_id="",
            )

    @parametrize
    async def test_method_retrieve(self, async_client: AsyncOpenlayer) -> None:
        api_key = await async_client.workspaces.api_keys.retrieve(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(APIKeyRetrieveResponse, api_key, path=["response"])

    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.workspaces.api_keys.with_raw_response.retrieve(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        api_key = await response.parse()
        assert_matches_type(APIKeyRetrieveResponse, api_key, path=["response"])

    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.workspaces.api_keys.with_streaming_response.retrieve(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            api_key = await response.parse()
            assert_matches_type(APIKeyRetrieveResponse, api_key, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            await async_client.workspaces.api_keys.with_raw_response.retrieve(
                api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                workspace_id="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `api_key_id` but received ''"):
            await async_client.workspaces.api_keys.with_raw_response.retrieve(
                api_key_id="",
                workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            )

    @parametrize
    async def test_method_update(self, async_client: AsyncOpenlayer) -> None:
        api_key = await async_client.workspaces.api_keys.update(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(APIKeyUpdateResponse, api_key, path=["response"])

    @parametrize
    async def test_method_update_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        api_key = await async_client.workspaces.api_keys.update(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="Secret Key",
        )
        assert_matches_type(APIKeyUpdateResponse, api_key, path=["response"])

    @parametrize
    async def test_raw_response_update(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.workspaces.api_keys.with_raw_response.update(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        api_key = await response.parse()
        assert_matches_type(APIKeyUpdateResponse, api_key, path=["response"])

    @parametrize
    async def test_streaming_response_update(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.workspaces.api_keys.with_streaming_response.update(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            api_key = await response.parse()
            assert_matches_type(APIKeyUpdateResponse, api_key, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_update(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            await async_client.workspaces.api_keys.with_raw_response.update(
                api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                workspace_id="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `api_key_id` but received ''"):
            await async_client.workspaces.api_keys.with_raw_response.update(
                api_key_id="",
                workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            )

    @parametrize
    async def test_method_list(self, async_client: AsyncOpenlayer) -> None:
        api_key = await async_client.workspaces.api_keys.list(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(APIKeyListResponse, api_key, path=["response"])

    @parametrize
    async def test_raw_response_list(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.workspaces.api_keys.with_raw_response.list(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        api_key = await response.parse()
        assert_matches_type(APIKeyListResponse, api_key, path=["response"])

    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.workspaces.api_keys.with_streaming_response.list(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            api_key = await response.parse()
            assert_matches_type(APIKeyListResponse, api_key, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_list(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            await async_client.workspaces.api_keys.with_raw_response.list(
                "",
            )

    @parametrize
    async def test_method_delete(self, async_client: AsyncOpenlayer) -> None:
        api_key = await async_client.workspaces.api_keys.delete(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert api_key is None

    @parametrize
    async def test_raw_response_delete(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.workspaces.api_keys.with_raw_response.delete(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        api_key = await response.parse()
        assert api_key is None

    @parametrize
    async def test_streaming_response_delete(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.workspaces.api_keys.with_streaming_response.delete(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            api_key = await response.parse()
            assert api_key is None

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_delete(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            await async_client.workspaces.api_keys.with_raw_response.delete(
                api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                workspace_id="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `api_key_id` but received ''"):
            await async_client.workspaces.api_keys.with_raw_response.delete(
                api_key_id="",
                workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            )

    @parametrize
    async def test_method_rotate(self, async_client: AsyncOpenlayer) -> None:
        api_key = await async_client.workspaces.api_keys.rotate(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(APIKeyRotateResponse, api_key, path=["response"])

    @parametrize
    async def test_method_rotate_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        api_key = await async_client.workspaces.api_keys.rotate(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            expires_at=parse_datetime("2027-01-01T00:00:00Z"),
            grace_period_hours=24,
        )
        assert_matches_type(APIKeyRotateResponse, api_key, path=["response"])

    @parametrize
    async def test_raw_response_rotate(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.workspaces.api_keys.with_raw_response.rotate(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        api_key = await response.parse()
        assert_matches_type(APIKeyRotateResponse, api_key, path=["response"])

    @parametrize
    async def test_streaming_response_rotate(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.workspaces.api_keys.with_streaming_response.rotate(
            api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            api_key = await response.parse()
            assert_matches_type(APIKeyRotateResponse, api_key, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_rotate(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            await async_client.workspaces.api_keys.with_raw_response.rotate(
                api_key_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                workspace_id="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `api_key_id` but received ''"):
            await async_client.workspaces.api_keys.with_raw_response.rotate(
                api_key_id="",
                workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            )
