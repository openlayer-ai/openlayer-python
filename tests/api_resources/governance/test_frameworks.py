# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types.governance import (
    FrameworkListResponse,
    FrameworkCreateResponse,
    FrameworkExportResponse,
    FrameworkUpdateResponse,
    FrameworkRetrieveResponse,
    FrameworkListRulesResponse,
    FrameworkListProjectsResponse,
    FrameworkListProjectRuleStatsResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestFrameworks:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_create(self, client: Openlayer) -> None:
        framework = client.governance.frameworks.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="EU AI Act",
        )
        assert_matches_type(FrameworkCreateResponse, framework, path=["response"])

    @parametrize
    def test_method_create_with_all_params(self, client: Openlayer) -> None:
        framework = client.governance.frameworks.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="EU AI Act",
            description="Requirements for high-risk AI systems under the EU AI Act.",
            enabled=True,
            project_selector={
                "match": [
                    {
                        "property": "riskLevel",
                        "value": ["high", "critical"],
                        "operator": "operator",
                    }
                ]
            },
            tags=["regulation", "eu"],
        )
        assert_matches_type(FrameworkCreateResponse, framework, path=["response"])

    @parametrize
    def test_raw_response_create(self, client: Openlayer) -> None:
        response = client.governance.frameworks.with_raw_response.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="EU AI Act",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = response.parse()
        assert_matches_type(FrameworkCreateResponse, framework, path=["response"])

    @parametrize
    def test_streaming_response_create(self, client: Openlayer) -> None:
        with client.governance.frameworks.with_streaming_response.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="EU AI Act",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = response.parse()
            assert_matches_type(FrameworkCreateResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_create(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            client.governance.frameworks.with_raw_response.create(
                workspace_id="",
                name="EU AI Act",
            )

    @parametrize
    def test_method_retrieve(self, client: Openlayer) -> None:
        framework = client.governance.frameworks.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(FrameworkRetrieveResponse, framework, path=["response"])

    @parametrize
    def test_raw_response_retrieve(self, client: Openlayer) -> None:
        response = client.governance.frameworks.with_raw_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = response.parse()
        assert_matches_type(FrameworkRetrieveResponse, framework, path=["response"])

    @parametrize
    def test_streaming_response_retrieve(self, client: Openlayer) -> None:
        with client.governance.frameworks.with_streaming_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = response.parse()
            assert_matches_type(FrameworkRetrieveResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_retrieve(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `framework_id` but received ''"):
            client.governance.frameworks.with_raw_response.retrieve(
                "",
            )

    @parametrize
    def test_method_update(self, client: Openlayer) -> None:
        framework = client.governance.frameworks.update(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(FrameworkUpdateResponse, framework, path=["response"])

    @parametrize
    def test_method_update_with_all_params(self, client: Openlayer) -> None:
        framework = client.governance.frameworks.update(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            avatar={
                "type": "emoji",
                "value": "🧭",
            },
            description="Requirements for high-risk AI systems under the EU AI Act.",
            enabled=True,
            extended_description={"foo": "bar"},
            href="href",
            name="EU AI Act",
            project_selector={
                "match": [
                    {
                        "property": "riskLevel",
                        "value": ["high", "critical"],
                        "operator": "operator",
                    }
                ]
            },
            tags=["regulation", "eu"],
        )
        assert_matches_type(FrameworkUpdateResponse, framework, path=["response"])

    @parametrize
    def test_raw_response_update(self, client: Openlayer) -> None:
        response = client.governance.frameworks.with_raw_response.update(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = response.parse()
        assert_matches_type(FrameworkUpdateResponse, framework, path=["response"])

    @parametrize
    def test_streaming_response_update(self, client: Openlayer) -> None:
        with client.governance.frameworks.with_streaming_response.update(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = response.parse()
            assert_matches_type(FrameworkUpdateResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_update(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `framework_id` but received ''"):
            client.governance.frameworks.with_raw_response.update(
                framework_id="",
            )

    @parametrize
    def test_method_list(self, client: Openlayer) -> None:
        framework = client.governance.frameworks.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(FrameworkListResponse, framework, path=["response"])

    @parametrize
    def test_method_list_with_all_params(self, client: Openlayer) -> None:
        framework = client.governance.frameworks.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            asc=True,
            completion_operator="is",
            completion_value=0,
            enabled=True,
            include_rule_stats=True,
            page=1,
            per_page=1,
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            search_query="searchQuery",
            sort_column="name",
            tags=["string"],
        )
        assert_matches_type(FrameworkListResponse, framework, path=["response"])

    @parametrize
    def test_raw_response_list(self, client: Openlayer) -> None:
        response = client.governance.frameworks.with_raw_response.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = response.parse()
        assert_matches_type(FrameworkListResponse, framework, path=["response"])

    @parametrize
    def test_streaming_response_list(self, client: Openlayer) -> None:
        with client.governance.frameworks.with_streaming_response.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = response.parse()
            assert_matches_type(FrameworkListResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_list(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            client.governance.frameworks.with_raw_response.list(
                workspace_id="",
            )

    @parametrize
    def test_method_export(self, client: Openlayer) -> None:
        framework = client.governance.frameworks.export(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(FrameworkExportResponse, framework, path=["response"])

    @parametrize
    def test_method_export_with_all_params(self, client: Openlayer) -> None:
        framework = client.governance.frameworks.export(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            project_id="3fa85f64-5717-4562-b3fc-2c963f66afa6",
        )
        assert_matches_type(FrameworkExportResponse, framework, path=["response"])

    @parametrize
    def test_raw_response_export(self, client: Openlayer) -> None:
        response = client.governance.frameworks.with_raw_response.export(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = response.parse()
        assert_matches_type(FrameworkExportResponse, framework, path=["response"])

    @parametrize
    def test_streaming_response_export(self, client: Openlayer) -> None:
        with client.governance.frameworks.with_streaming_response.export(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = response.parse()
            assert_matches_type(FrameworkExportResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_export(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `framework_id` but received ''"):
            client.governance.frameworks.with_raw_response.export(
                framework_id="",
            )

    @parametrize
    def test_method_list_project_rule_stats(self, client: Openlayer) -> None:
        framework = client.governance.frameworks.list_project_rule_stats(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(FrameworkListProjectRuleStatsResponse, framework, path=["response"])

    @parametrize
    def test_method_list_project_rule_stats_with_all_params(self, client: Openlayer) -> None:
        framework = client.governance.frameworks.list_project_rule_stats(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            asc=True,
            page=1,
            per_page=1,
            sort_column="projectName",
        )
        assert_matches_type(FrameworkListProjectRuleStatsResponse, framework, path=["response"])

    @parametrize
    def test_raw_response_list_project_rule_stats(self, client: Openlayer) -> None:
        response = client.governance.frameworks.with_raw_response.list_project_rule_stats(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = response.parse()
        assert_matches_type(FrameworkListProjectRuleStatsResponse, framework, path=["response"])

    @parametrize
    def test_streaming_response_list_project_rule_stats(self, client: Openlayer) -> None:
        with client.governance.frameworks.with_streaming_response.list_project_rule_stats(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = response.parse()
            assert_matches_type(FrameworkListProjectRuleStatsResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_list_project_rule_stats(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `framework_id` but received ''"):
            client.governance.frameworks.with_raw_response.list_project_rule_stats(
                framework_id="",
            )

    @parametrize
    def test_method_list_projects(self, client: Openlayer) -> None:
        framework = client.governance.frameworks.list_projects(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(FrameworkListProjectsResponse, framework, path=["response"])

    @parametrize
    def test_method_list_projects_with_all_params(self, client: Openlayer) -> None:
        framework = client.governance.frameworks.list_projects(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            page=1,
            per_page=1,
        )
        assert_matches_type(FrameworkListProjectsResponse, framework, path=["response"])

    @parametrize
    def test_raw_response_list_projects(self, client: Openlayer) -> None:
        response = client.governance.frameworks.with_raw_response.list_projects(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = response.parse()
        assert_matches_type(FrameworkListProjectsResponse, framework, path=["response"])

    @parametrize
    def test_streaming_response_list_projects(self, client: Openlayer) -> None:
        with client.governance.frameworks.with_streaming_response.list_projects(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = response.parse()
            assert_matches_type(FrameworkListProjectsResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_list_projects(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `framework_id` but received ''"):
            client.governance.frameworks.with_raw_response.list_projects(
                framework_id="",
            )

    @parametrize
    def test_method_list_rules(self, client: Openlayer) -> None:
        framework = client.governance.frameworks.list_rules(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(FrameworkListRulesResponse, framework, path=["response"])

    @parametrize
    def test_method_list_rules_with_all_params(self, client: Openlayer) -> None:
        framework = client.governance.frameworks.list_rules(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            page=1,
            per_page=1,
        )
        assert_matches_type(FrameworkListRulesResponse, framework, path=["response"])

    @parametrize
    def test_raw_response_list_rules(self, client: Openlayer) -> None:
        response = client.governance.frameworks.with_raw_response.list_rules(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = response.parse()
        assert_matches_type(FrameworkListRulesResponse, framework, path=["response"])

    @parametrize
    def test_streaming_response_list_rules(self, client: Openlayer) -> None:
        with client.governance.frameworks.with_streaming_response.list_rules(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = response.parse()
            assert_matches_type(FrameworkListRulesResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_list_rules(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `framework_id` but received ''"):
            client.governance.frameworks.with_raw_response.list_rules(
                framework_id="",
            )


class TestAsyncFrameworks:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @parametrize
    async def test_method_create(self, async_client: AsyncOpenlayer) -> None:
        framework = await async_client.governance.frameworks.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="EU AI Act",
        )
        assert_matches_type(FrameworkCreateResponse, framework, path=["response"])

    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        framework = await async_client.governance.frameworks.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="EU AI Act",
            description="Requirements for high-risk AI systems under the EU AI Act.",
            enabled=True,
            project_selector={
                "match": [
                    {
                        "property": "riskLevel",
                        "value": ["high", "critical"],
                        "operator": "operator",
                    }
                ]
            },
            tags=["regulation", "eu"],
        )
        assert_matches_type(FrameworkCreateResponse, framework, path=["response"])

    @parametrize
    async def test_raw_response_create(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.frameworks.with_raw_response.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="EU AI Act",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = await response.parse()
        assert_matches_type(FrameworkCreateResponse, framework, path=["response"])

    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.frameworks.with_streaming_response.create(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            name="EU AI Act",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = await response.parse()
            assert_matches_type(FrameworkCreateResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_create(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            await async_client.governance.frameworks.with_raw_response.create(
                workspace_id="",
                name="EU AI Act",
            )

    @parametrize
    async def test_method_retrieve(self, async_client: AsyncOpenlayer) -> None:
        framework = await async_client.governance.frameworks.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(FrameworkRetrieveResponse, framework, path=["response"])

    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.frameworks.with_raw_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = await response.parse()
        assert_matches_type(FrameworkRetrieveResponse, framework, path=["response"])

    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.frameworks.with_streaming_response.retrieve(
            "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = await response.parse()
            assert_matches_type(FrameworkRetrieveResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `framework_id` but received ''"):
            await async_client.governance.frameworks.with_raw_response.retrieve(
                "",
            )

    @parametrize
    async def test_method_update(self, async_client: AsyncOpenlayer) -> None:
        framework = await async_client.governance.frameworks.update(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(FrameworkUpdateResponse, framework, path=["response"])

    @parametrize
    async def test_method_update_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        framework = await async_client.governance.frameworks.update(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            avatar={
                "type": "emoji",
                "value": "🧭",
            },
            description="Requirements for high-risk AI systems under the EU AI Act.",
            enabled=True,
            extended_description={"foo": "bar"},
            href="href",
            name="EU AI Act",
            project_selector={
                "match": [
                    {
                        "property": "riskLevel",
                        "value": ["high", "critical"],
                        "operator": "operator",
                    }
                ]
            },
            tags=["regulation", "eu"],
        )
        assert_matches_type(FrameworkUpdateResponse, framework, path=["response"])

    @parametrize
    async def test_raw_response_update(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.frameworks.with_raw_response.update(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = await response.parse()
        assert_matches_type(FrameworkUpdateResponse, framework, path=["response"])

    @parametrize
    async def test_streaming_response_update(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.frameworks.with_streaming_response.update(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = await response.parse()
            assert_matches_type(FrameworkUpdateResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_update(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `framework_id` but received ''"):
            await async_client.governance.frameworks.with_raw_response.update(
                framework_id="",
            )

    @parametrize
    async def test_method_list(self, async_client: AsyncOpenlayer) -> None:
        framework = await async_client.governance.frameworks.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(FrameworkListResponse, framework, path=["response"])

    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        framework = await async_client.governance.frameworks.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            asc=True,
            completion_operator="is",
            completion_value=0,
            enabled=True,
            include_rule_stats=True,
            page=1,
            per_page=1,
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            search_query="searchQuery",
            sort_column="name",
            tags=["string"],
        )
        assert_matches_type(FrameworkListResponse, framework, path=["response"])

    @parametrize
    async def test_raw_response_list(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.frameworks.with_raw_response.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = await response.parse()
        assert_matches_type(FrameworkListResponse, framework, path=["response"])

    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.frameworks.with_streaming_response.list(
            workspace_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = await response.parse()
            assert_matches_type(FrameworkListResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_list(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace_id` but received ''"):
            await async_client.governance.frameworks.with_raw_response.list(
                workspace_id="",
            )

    @parametrize
    async def test_method_export(self, async_client: AsyncOpenlayer) -> None:
        framework = await async_client.governance.frameworks.export(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(FrameworkExportResponse, framework, path=["response"])

    @parametrize
    async def test_method_export_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        framework = await async_client.governance.frameworks.export(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            project_id="3fa85f64-5717-4562-b3fc-2c963f66afa6",
        )
        assert_matches_type(FrameworkExportResponse, framework, path=["response"])

    @parametrize
    async def test_raw_response_export(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.frameworks.with_raw_response.export(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = await response.parse()
        assert_matches_type(FrameworkExportResponse, framework, path=["response"])

    @parametrize
    async def test_streaming_response_export(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.frameworks.with_streaming_response.export(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = await response.parse()
            assert_matches_type(FrameworkExportResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_export(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `framework_id` but received ''"):
            await async_client.governance.frameworks.with_raw_response.export(
                framework_id="",
            )

    @parametrize
    async def test_method_list_project_rule_stats(self, async_client: AsyncOpenlayer) -> None:
        framework = await async_client.governance.frameworks.list_project_rule_stats(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(FrameworkListProjectRuleStatsResponse, framework, path=["response"])

    @parametrize
    async def test_method_list_project_rule_stats_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        framework = await async_client.governance.frameworks.list_project_rule_stats(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            asc=True,
            page=1,
            per_page=1,
            sort_column="projectName",
        )
        assert_matches_type(FrameworkListProjectRuleStatsResponse, framework, path=["response"])

    @parametrize
    async def test_raw_response_list_project_rule_stats(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.frameworks.with_raw_response.list_project_rule_stats(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = await response.parse()
        assert_matches_type(FrameworkListProjectRuleStatsResponse, framework, path=["response"])

    @parametrize
    async def test_streaming_response_list_project_rule_stats(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.frameworks.with_streaming_response.list_project_rule_stats(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = await response.parse()
            assert_matches_type(FrameworkListProjectRuleStatsResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_list_project_rule_stats(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `framework_id` but received ''"):
            await async_client.governance.frameworks.with_raw_response.list_project_rule_stats(
                framework_id="",
            )

    @parametrize
    async def test_method_list_projects(self, async_client: AsyncOpenlayer) -> None:
        framework = await async_client.governance.frameworks.list_projects(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(FrameworkListProjectsResponse, framework, path=["response"])

    @parametrize
    async def test_method_list_projects_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        framework = await async_client.governance.frameworks.list_projects(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            page=1,
            per_page=1,
        )
        assert_matches_type(FrameworkListProjectsResponse, framework, path=["response"])

    @parametrize
    async def test_raw_response_list_projects(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.frameworks.with_raw_response.list_projects(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = await response.parse()
        assert_matches_type(FrameworkListProjectsResponse, framework, path=["response"])

    @parametrize
    async def test_streaming_response_list_projects(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.frameworks.with_streaming_response.list_projects(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = await response.parse()
            assert_matches_type(FrameworkListProjectsResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_list_projects(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `framework_id` but received ''"):
            await async_client.governance.frameworks.with_raw_response.list_projects(
                framework_id="",
            )

    @parametrize
    async def test_method_list_rules(self, async_client: AsyncOpenlayer) -> None:
        framework = await async_client.governance.frameworks.list_rules(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(FrameworkListRulesResponse, framework, path=["response"])

    @parametrize
    async def test_method_list_rules_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        framework = await async_client.governance.frameworks.list_rules(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            page=1,
            per_page=1,
        )
        assert_matches_type(FrameworkListRulesResponse, framework, path=["response"])

    @parametrize
    async def test_raw_response_list_rules(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.frameworks.with_raw_response.list_rules(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        framework = await response.parse()
        assert_matches_type(FrameworkListRulesResponse, framework, path=["response"])

    @parametrize
    async def test_streaming_response_list_rules(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.frameworks.with_streaming_response.list_rules(
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            framework = await response.parse()
            assert_matches_type(FrameworkListRulesResponse, framework, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_list_rules(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `framework_id` but received ''"):
            await async_client.governance.frameworks.with_raw_response.list_rules(
                framework_id="",
            )
