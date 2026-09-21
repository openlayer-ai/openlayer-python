# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types.governance import RuleStatRetrieveResponse

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestRuleStats:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_retrieve(self, client: Openlayer) -> None:
        rule_stat = client.governance.rule_stats.retrieve(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleStatRetrieveResponse, rule_stat, path=["response"])

    @parametrize
    def test_method_retrieve_with_all_params(self, client: Openlayer) -> None:
        rule_stat = client.governance.rule_stats.retrieve(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleStatRetrieveResponse, rule_stat, path=["response"])

    @parametrize
    def test_raw_response_retrieve(self, client: Openlayer) -> None:
        response = client.governance.rule_stats.with_raw_response.retrieve(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule_stat = response.parse()
        assert_matches_type(RuleStatRetrieveResponse, rule_stat, path=["response"])

    @parametrize
    def test_streaming_response_retrieve(self, client: Openlayer) -> None:
        with client.governance.rule_stats.with_streaming_response.retrieve(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule_stat = response.parse()
            assert_matches_type(RuleStatRetrieveResponse, rule_stat, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_retrieve(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            client.governance.rule_stats.with_raw_response.retrieve(
                workspace_id="",
            )


class TestAsyncRuleStats:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @parametrize
    async def test_method_retrieve(self, async_client: AsyncOpenlayer) -> None:
        rule_stat = await async_client.governance.rule_stats.retrieve(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleStatRetrieveResponse, rule_stat, path=["response"])

    @parametrize
    async def test_method_retrieve_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        rule_stat = await async_client.governance.rule_stats.retrieve(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleStatRetrieveResponse, rule_stat, path=["response"])

    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.rule_stats.with_raw_response.retrieve(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule_stat = await response.parse()
        assert_matches_type(RuleStatRetrieveResponse, rule_stat, path=["response"])

    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.rule_stats.with_streaming_response.retrieve(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule_stat = await response.parse()
            assert_matches_type(RuleStatRetrieveResponse, rule_stat, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            await async_client.governance.rule_stats.with_raw_response.retrieve(
                workspace_id="",
            )
