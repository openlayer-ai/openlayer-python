# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types.governance.frameworks import SectionListRulesResponse

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestSections:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_list_rules(self, client: Openlayer) -> None:
        section = client.governance.frameworks.sections.list_rules(
            section_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(SectionListRulesResponse, section, path=["response"])

    @parametrize
    def test_method_list_rules_with_all_params(self, client: Openlayer) -> None:
        section = client.governance.frameworks.sections.list_rules(
            section_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            include_results=True,
            include_subsection_rules=True,
            page=1,
            per_page=1,
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            status="passing",
        )
        assert_matches_type(SectionListRulesResponse, section, path=["response"])

    @parametrize
    def test_raw_response_list_rules(self, client: Openlayer) -> None:
        response = client.governance.frameworks.sections.with_raw_response.list_rules(
            section_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        section = response.parse()
        assert_matches_type(SectionListRulesResponse, section, path=["response"])

    @parametrize
    def test_streaming_response_list_rules(self, client: Openlayer) -> None:
        with client.governance.frameworks.sections.with_streaming_response.list_rules(
            section_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            section = response.parse()
            assert_matches_type(SectionListRulesResponse, section, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_list_rules(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `framework_id` but received ''"):
            client.governance.frameworks.sections.with_raw_response.list_rules(
                section_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                framework_id="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `section_id` but received ''"):
            client.governance.frameworks.sections.with_raw_response.list_rules(
                section_id="",
                framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            )


class TestAsyncSections:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @parametrize
    async def test_method_list_rules(self, async_client: AsyncOpenlayer) -> None:
        section = await async_client.governance.frameworks.sections.list_rules(
            section_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(SectionListRulesResponse, section, path=["response"])

    @parametrize
    async def test_method_list_rules_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        section = await async_client.governance.frameworks.sections.list_rules(
            section_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            include_results=True,
            include_subsection_rules=True,
            page=1,
            per_page=1,
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            status="passing",
        )
        assert_matches_type(SectionListRulesResponse, section, path=["response"])

    @parametrize
    async def test_raw_response_list_rules(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.governance.frameworks.sections.with_raw_response.list_rules(
            section_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        section = await response.parse()
        assert_matches_type(SectionListRulesResponse, section, path=["response"])

    @parametrize
    async def test_streaming_response_list_rules(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.governance.frameworks.sections.with_streaming_response.list_rules(
            section_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            section = await response.parse()
            assert_matches_type(SectionListRulesResponse, section, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_list_rules(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `framework_id` but received ''"):
            await async_client.governance.frameworks.sections.with_raw_response.list_rules(
                section_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                framework_id="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `section_id` but received ''"):
            await async_client.governance.frameworks.sections.with_raw_response.list_rules(
                section_id="",
                framework_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            )
