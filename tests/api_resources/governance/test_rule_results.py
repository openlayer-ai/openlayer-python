# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types.governance import (
    RuleResultListResponse,
    RuleResultUpdateResponse,
    RuleResultRetrieveResponse,
    RuleResultListEvidenceResponse,
    RuleResultCreateEvidenceResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestRuleResults:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_retrieve(self, client: Openlayer) -> None:
        rule_result = client.governance.rule_results.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleResultRetrieveResponse, rule_result, path=["response"])

    @parametrize
    def test_raw_response_retrieve(self, client: Openlayer) -> None:
        response = client.governance.rule_results.with_raw_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule_result = response.parse()
        assert_matches_type(RuleResultRetrieveResponse, rule_result, path=["response"])

    @parametrize
    def test_streaming_response_retrieve(self, client: Openlayer) -> None:
        with client.governance.rule_results.with_streaming_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule_result = response.parse()
            assert_matches_type(RuleResultRetrieveResponse, rule_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_retrieve(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `rule_result_id` but received ''"):
            client.governance.rule_results.with_raw_response.retrieve(
                "",
            )

    @parametrize
    def test_method_update(self, client: Openlayer) -> None:
        rule_result = client.governance.rule_results.update(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleResultUpdateResponse, rule_result, path=["response"])

    @parametrize
    def test_method_update_with_all_params(self, client: Openlayer) -> None:
        rule_result = client.governance.rule_results.update(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            assignee_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            blocked_by=[
                {
                    "id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "status": "passing",
                }
            ],
            blocking=[
                {
                    "id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "status": "passing",
                }
            ],
            deactivated=True,
            deactivated_reason="deactivatedReason",
        )
        assert_matches_type(RuleResultUpdateResponse, rule_result, path=["response"])

    @parametrize
    def test_raw_response_update(self, client: Openlayer) -> None:
        response = client.governance.rule_results.with_raw_response.update(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule_result = response.parse()
        assert_matches_type(RuleResultUpdateResponse, rule_result, path=["response"])

    @parametrize
    def test_streaming_response_update(self, client: Openlayer) -> None:
        with client.governance.rule_results.with_streaming_response.update(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule_result = response.parse()
            assert_matches_type(RuleResultUpdateResponse, rule_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_update(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `rule_result_id` but received ''"):
            client.governance.rule_results.with_raw_response.update(
                rule_result_id="",
            )

    @parametrize
    def test_method_list(self, client: Openlayer) -> None:
        rule_result = client.governance.rule_results.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleResultListResponse, rule_result, path=["response"])

    @parametrize
    def test_method_list_with_all_params(self, client: Openlayer) -> None:
        rule_result = client.governance.rule_results.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            enabled_framework_only=True,
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            include_unframed=True,
            page=1,
            per_page=1,
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            rule_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            scope="project",
            search_query="searchQuery",
            status="passing",
            type="platform",
        )
        assert_matches_type(RuleResultListResponse, rule_result, path=["response"])

    @parametrize
    def test_raw_response_list(self, client: Openlayer) -> None:
        response = client.governance.rule_results.with_raw_response.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule_result = response.parse()
        assert_matches_type(RuleResultListResponse, rule_result, path=["response"])

    @parametrize
    def test_streaming_response_list(self, client: Openlayer) -> None:
        with client.governance.rule_results.with_streaming_response.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule_result = response.parse()
            assert_matches_type(RuleResultListResponse, rule_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_list(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            client.governance.rule_results.with_raw_response.list(
                workspace_id="",
            )

    @parametrize
    def test_method_create_evidence(self, client: Openlayer) -> None:
        rule_result = client.governance.rule_results.create_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleResultCreateEvidenceResponse, rule_result, path=["response"])

    @parametrize
    def test_method_create_evidence_with_all_params(self, client: Openlayer) -> None:
        rule_result = client.governance.rule_results.create_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="description",
            name="Model risk assessment 2026",
            storage_uri="s3://openlayer-evidence/evidence.pdf",
            text="text",
            url="https://openlayer.com/evidence",
        )
        assert_matches_type(RuleResultCreateEvidenceResponse, rule_result, path=["response"])

    @parametrize
    def test_raw_response_create_evidence(self, client: Openlayer) -> None:
        response = client.governance.rule_results.with_raw_response.create_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule_result = response.parse()
        assert_matches_type(RuleResultCreateEvidenceResponse, rule_result, path=["response"])

    @parametrize
    def test_streaming_response_create_evidence(self, client: Openlayer) -> None:
        with client.governance.rule_results.with_streaming_response.create_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule_result = response.parse()
            assert_matches_type(RuleResultCreateEvidenceResponse, rule_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_create_evidence(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `rule_result_id` but received ''"):
            client.governance.rule_results.with_raw_response.create_evidence(
                rule_result_id="",
            )

    @parametrize
    def test_method_list_evidence(self, client: Openlayer) -> None:
        rule_result = client.governance.rule_results.list_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleResultListEvidenceResponse, rule_result, path=["response"])

    @parametrize
    def test_method_list_evidence_with_all_params(self, client: Openlayer) -> None:
        rule_result = client.governance.rule_results.list_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            page=1,
            per_page=1,
        )
        assert_matches_type(RuleResultListEvidenceResponse, rule_result, path=["response"])

    @parametrize
    def test_raw_response_list_evidence(self, client: Openlayer) -> None:
        response = client.governance.rule_results.with_raw_response.list_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule_result = response.parse()
        assert_matches_type(RuleResultListEvidenceResponse, rule_result, path=["response"])

    @parametrize
    def test_streaming_response_list_evidence(self, client: Openlayer) -> None:
        with client.governance.rule_results.with_streaming_response.list_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule_result = response.parse()
            assert_matches_type(RuleResultListEvidenceResponse, rule_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_list_evidence(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `rule_result_id` but received ''"):
            client.governance.rule_results.with_raw_response.list_evidence(
                rule_result_id="",
            )


class TestAsyncRuleResults:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @parametrize
    async def test_method_retrieve(self, async_client: AsyncOpenlayer) -> None:
        rule_result = await async_client.governance.rule_results.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleResultRetrieveResponse, rule_result, path=["response"])

    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.rule_results.with_raw_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule_result = await response.parse()
        assert_matches_type(RuleResultRetrieveResponse, rule_result, path=["response"])

    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.rule_results.with_streaming_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule_result = await response.parse()
            assert_matches_type(RuleResultRetrieveResponse, rule_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `rule_result_id` but received ''"):
            await async_client.governance.rule_results.with_raw_response.retrieve(
                "",
            )

    @parametrize
    async def test_method_update(self, async_client: AsyncOpenlayer) -> None:
        rule_result = await async_client.governance.rule_results.update(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleResultUpdateResponse, rule_result, path=["response"])

    @parametrize
    async def test_method_update_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        rule_result = await async_client.governance.rule_results.update(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            assignee_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            blocked_by=[
                {
                    "id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "status": "passing",
                }
            ],
            blocking=[
                {
                    "id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "status": "passing",
                }
            ],
            deactivated=True,
            deactivated_reason="deactivatedReason",
        )
        assert_matches_type(RuleResultUpdateResponse, rule_result, path=["response"])

    @parametrize
    async def test_raw_response_update(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.rule_results.with_raw_response.update(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule_result = await response.parse()
        assert_matches_type(RuleResultUpdateResponse, rule_result, path=["response"])

    @parametrize
    async def test_streaming_response_update(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.rule_results.with_streaming_response.update(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule_result = await response.parse()
            assert_matches_type(RuleResultUpdateResponse, rule_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_update(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `rule_result_id` but received ''"):
            await async_client.governance.rule_results.with_raw_response.update(
                rule_result_id="",
            )

    @parametrize
    async def test_method_list(self, async_client: AsyncOpenlayer) -> None:
        rule_result = await async_client.governance.rule_results.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleResultListResponse, rule_result, path=["response"])

    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        rule_result = await async_client.governance.rule_results.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            enabled_framework_only=True,
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            include_unframed=True,
            page=1,
            per_page=1,
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            rule_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            scope="project",
            search_query="searchQuery",
            status="passing",
            type="platform",
        )
        assert_matches_type(RuleResultListResponse, rule_result, path=["response"])

    @parametrize
    async def test_raw_response_list(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.rule_results.with_raw_response.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule_result = await response.parse()
        assert_matches_type(RuleResultListResponse, rule_result, path=["response"])

    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.rule_results.with_streaming_response.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule_result = await response.parse()
            assert_matches_type(RuleResultListResponse, rule_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_list(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            await async_client.governance.rule_results.with_raw_response.list(
                workspace_id="",
            )

    @parametrize
    async def test_method_create_evidence(self, async_client: AsyncOpenlayer) -> None:
        rule_result = await async_client.governance.rule_results.create_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleResultCreateEvidenceResponse, rule_result, path=["response"])

    @parametrize
    async def test_method_create_evidence_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        rule_result = await async_client.governance.rule_results.create_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="description",
            name="Model risk assessment 2026",
            storage_uri="s3://openlayer-evidence/evidence.pdf",
            text="text",
            url="https://openlayer.com/evidence",
        )
        assert_matches_type(RuleResultCreateEvidenceResponse, rule_result, path=["response"])

    @parametrize
    async def test_raw_response_create_evidence(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.rule_results.with_raw_response.create_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule_result = await response.parse()
        assert_matches_type(RuleResultCreateEvidenceResponse, rule_result, path=["response"])

    @parametrize
    async def test_streaming_response_create_evidence(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.rule_results.with_streaming_response.create_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule_result = await response.parse()
            assert_matches_type(RuleResultCreateEvidenceResponse, rule_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_create_evidence(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `rule_result_id` but received ''"):
            await async_client.governance.rule_results.with_raw_response.create_evidence(
                rule_result_id="",
            )

    @parametrize
    async def test_method_list_evidence(self, async_client: AsyncOpenlayer) -> None:
        rule_result = await async_client.governance.rule_results.list_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(RuleResultListEvidenceResponse, rule_result, path=["response"])

    @parametrize
    async def test_method_list_evidence_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        rule_result = await async_client.governance.rule_results.list_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            page=1,
            per_page=1,
        )
        assert_matches_type(RuleResultListEvidenceResponse, rule_result, path=["response"])

    @parametrize
    async def test_raw_response_list_evidence(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.rule_results.with_raw_response.list_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        rule_result = await response.parse()
        assert_matches_type(RuleResultListEvidenceResponse, rule_result, path=["response"])

    @parametrize
    async def test_streaming_response_list_evidence(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.rule_results.with_streaming_response.list_evidence(
            rule_result_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            rule_result = await response.parse()
            assert_matches_type(RuleResultListEvidenceResponse, rule_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_list_evidence(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `rule_result_id` but received ''"):
            await async_client.governance.rule_results.with_raw_response.list_evidence(
                rule_result_id="",
            )
