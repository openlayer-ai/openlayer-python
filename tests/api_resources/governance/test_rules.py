# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types.governance import (
    RuleListResponse,
    RuleCreateResponse,
    RuleUpdateResponse,
    RuleRetrieveResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestRules:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_create(self, client: Openlayer) -> None:
        rule = client.governance.rules.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="Monitoring enabled",
            scope="project",
            type="platform",
        )
        assert_matches_type(RuleCreateResponse, rule, path=["response"])

    @parametrize
    def test_method_create_with_all_params(self, client: Openlayer) -> None:
        rule = client.governance.rules.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="Monitoring enabled",
            scope="project",
            type="platform",
            assignee_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            automation_params={"foo": "bar"},
            automation_type="monitoring_mode_enabled",
            deactivated=True,
            description="Each project must have Openlayer monitoring mode enabled.",
            evidence_type="document",
            renewal_cadence_days=90,
            tag_ids=["182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e"],
        )
        assert_matches_type(RuleCreateResponse, rule, path=["response"])

    @parametrize
    def test_raw_response_create(self, client: Openlayer) -> None:
        response = client.governance.rules.with_raw_response.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="Monitoring enabled",
            scope="project",
            type="platform",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule = response.parse()
        assert_matches_type(RuleCreateResponse, rule, path=["response"])

    @parametrize
    def test_streaming_response_create(self, client: Openlayer) -> None:
        with client.governance.rules.with_streaming_response.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="Monitoring enabled",
            scope="project",
            type="platform",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule = response.parse()
            assert_matches_type(RuleCreateResponse, rule, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_create(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            client.governance.rules.with_raw_response.create(
                workspace_id="",
                name="Monitoring enabled",
                scope="project",
                type="platform",
            )

    @parametrize
    def test_method_retrieve(self, client: Openlayer) -> None:
        rule = client.governance.rules.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleRetrieveResponse, rule, path=["response"])

    @parametrize
    def test_raw_response_retrieve(self, client: Openlayer) -> None:
        response = client.governance.rules.with_raw_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule = response.parse()
        assert_matches_type(RuleRetrieveResponse, rule, path=["response"])

    @parametrize
    def test_streaming_response_retrieve(self, client: Openlayer) -> None:
        with client.governance.rules.with_streaming_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule = response.parse()
            assert_matches_type(RuleRetrieveResponse, rule, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_retrieve(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `rule_id` but received ''"):
            client.governance.rules.with_raw_response.retrieve(
                "",
            )

    @parametrize
    def test_method_update(self, client: Openlayer) -> None:
        rule = client.governance.rules.update(
            rule_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleUpdateResponse, rule, path=["response"])

    @parametrize
    def test_method_update_with_all_params(self, client: Openlayer) -> None:
        rule = client.governance.rules.update(
            rule_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            assignee_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            deactivated=True,
            description="Each project must have Openlayer monitoring mode enabled.",
            name="Monitoring enabled",
            renewal_cadence_days=90,
            tag_ids=["182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e"],
        )
        assert_matches_type(RuleUpdateResponse, rule, path=["response"])

    @parametrize
    def test_raw_response_update(self, client: Openlayer) -> None:
        response = client.governance.rules.with_raw_response.update(
            rule_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule = response.parse()
        assert_matches_type(RuleUpdateResponse, rule, path=["response"])

    @parametrize
    def test_streaming_response_update(self, client: Openlayer) -> None:
        with client.governance.rules.with_streaming_response.update(
            rule_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule = response.parse()
            assert_matches_type(RuleUpdateResponse, rule, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_update(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `rule_id` but received ''"):
            client.governance.rules.with_raw_response.update(
                rule_id="",
            )

    @parametrize
    def test_method_list(self, client: Openlayer) -> None:
        rule = client.governance.rules.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleListResponse, rule, path=["response"])

    @parametrize
    def test_method_list_with_all_params(self, client: Openlayer) -> None:
        rule = client.governance.rules.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            asc=True,
            assignee_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            deactivated=True,
            enabled_framework_only=True,
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            group="open",
            include_results=True,
            include_unframed=True,
            page=1,
            per_page=1,
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            scope="project",
            search_query="searchQuery",
            sort_by="name",
            status="passing",
            tags=["182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e"],
            type="platform",
        )
        assert_matches_type(RuleListResponse, rule, path=["response"])

    @parametrize
    def test_raw_response_list(self, client: Openlayer) -> None:
        response = client.governance.rules.with_raw_response.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule = response.parse()
        assert_matches_type(RuleListResponse, rule, path=["response"])

    @parametrize
    def test_streaming_response_list(self, client: Openlayer) -> None:
        with client.governance.rules.with_streaming_response.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule = response.parse()
            assert_matches_type(RuleListResponse, rule, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_list(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            client.governance.rules.with_raw_response.list(
                workspace_id="",
            )

    @parametrize
    def test_method_delete(self, client: Openlayer) -> None:
        rule = client.governance.rules.delete(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert rule is None

    @parametrize
    def test_raw_response_delete(self, client: Openlayer) -> None:
        response = client.governance.rules.with_raw_response.delete(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule = response.parse()
        assert rule is None

    @parametrize
    def test_streaming_response_delete(self, client: Openlayer) -> None:
        with client.governance.rules.with_streaming_response.delete(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule = response.parse()
            assert rule is None

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_delete(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `rule_id` but received ''"):
            client.governance.rules.with_raw_response.delete(
                "",
            )


class TestAsyncRules:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @parametrize
    async def test_method_create(self, async_client: AsyncOpenlayer) -> None:
        rule = await async_client.governance.rules.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="Monitoring enabled",
            scope="project",
            type="platform",
        )
        assert_matches_type(RuleCreateResponse, rule, path=["response"])

    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        rule = await async_client.governance.rules.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="Monitoring enabled",
            scope="project",
            type="platform",
            assignee_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            automation_params={"foo": "bar"},
            automation_type="monitoring_mode_enabled",
            deactivated=True,
            description="Each project must have Openlayer monitoring mode enabled.",
            evidence_type="document",
            renewal_cadence_days=90,
            tag_ids=["182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e"],
        )
        assert_matches_type(RuleCreateResponse, rule, path=["response"])

    @parametrize
    async def test_raw_response_create(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.rules.with_raw_response.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="Monitoring enabled",
            scope="project",
            type="platform",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule = await response.parse()
        assert_matches_type(RuleCreateResponse, rule, path=["response"])

    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.rules.with_streaming_response.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="Monitoring enabled",
            scope="project",
            type="platform",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule = await response.parse()
            assert_matches_type(RuleCreateResponse, rule, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_create(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            await async_client.governance.rules.with_raw_response.create(
                workspace_id="",
                name="Monitoring enabled",
                scope="project",
                type="platform",
            )

    @parametrize
    async def test_method_retrieve(self, async_client: AsyncOpenlayer) -> None:
        rule = await async_client.governance.rules.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleRetrieveResponse, rule, path=["response"])

    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.rules.with_raw_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule = await response.parse()
        assert_matches_type(RuleRetrieveResponse, rule, path=["response"])

    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.rules.with_streaming_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule = await response.parse()
            assert_matches_type(RuleRetrieveResponse, rule, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `rule_id` but received ''"):
            await async_client.governance.rules.with_raw_response.retrieve(
                "",
            )

    @parametrize
    async def test_method_update(self, async_client: AsyncOpenlayer) -> None:
        rule = await async_client.governance.rules.update(
            rule_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleUpdateResponse, rule, path=["response"])

    @parametrize
    async def test_method_update_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        rule = await async_client.governance.rules.update(
            rule_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            assignee_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            deactivated=True,
            description="Each project must have Openlayer monitoring mode enabled.",
            name="Monitoring enabled",
            renewal_cadence_days=90,
            tag_ids=["182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e"],
        )
        assert_matches_type(RuleUpdateResponse, rule, path=["response"])

    @parametrize
    async def test_raw_response_update(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.rules.with_raw_response.update(
            rule_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule = await response.parse()
        assert_matches_type(RuleUpdateResponse, rule, path=["response"])

    @parametrize
    async def test_streaming_response_update(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.rules.with_streaming_response.update(
            rule_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule = await response.parse()
            assert_matches_type(RuleUpdateResponse, rule, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_update(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `rule_id` but received ''"):
            await async_client.governance.rules.with_raw_response.update(
                rule_id="",
            )

    @parametrize
    async def test_method_list(self, async_client: AsyncOpenlayer) -> None:
        rule = await async_client.governance.rules.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleListResponse, rule, path=["response"])

    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        rule = await async_client.governance.rules.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            asc=True,
            assignee_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            deactivated=True,
            enabled_framework_only=True,
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            group="open",
            include_results=True,
            include_unframed=True,
            page=1,
            per_page=1,
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            scope="project",
            search_query="searchQuery",
            sort_by="name",
            status="passing",
            tags=["182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e"],
            type="platform",
        )
        assert_matches_type(RuleListResponse, rule, path=["response"])

    @parametrize
    async def test_raw_response_list(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.rules.with_raw_response.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule = await response.parse()
        assert_matches_type(RuleListResponse, rule, path=["response"])

    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.rules.with_streaming_response.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule = await response.parse()
            assert_matches_type(RuleListResponse, rule, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_list(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            await async_client.governance.rules.with_raw_response.list(
                workspace_id="",
            )

    @parametrize
    async def test_method_delete(self, async_client: AsyncOpenlayer) -> None:
        rule = await async_client.governance.rules.delete(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert rule is None

    @parametrize
    async def test_raw_response_delete(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.rules.with_raw_response.delete(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule = await response.parse()
        assert rule is None

    @parametrize
    async def test_streaming_response_delete(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.rules.with_streaming_response.delete(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule = await response.parse()
            assert rule is None

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_delete(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `rule_id` but received ''"):
            await async_client.governance.rules.with_raw_response.delete(
                "",
            )
