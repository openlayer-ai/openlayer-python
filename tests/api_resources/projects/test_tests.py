# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from openlayer import Openlayer, AsyncOpenlayer
from tests.utils import assert_matches_type
from openlayer.types.projects import (
    TestListResponse,
    TestCreateResponse,
    TestUpdateResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestTests:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @parametrize
    def test_method_create(self, client: Openlayer) -> None:
        test = client.projects.tests.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This test checks for duplicate rows in the dataset.",
            name="No duplicate rows",
            subtype="duplicateRowCount",
            thresholds=[{}],
            type="integrity",
        )
        assert_matches_type(TestCreateResponse, test, path=["response"])

    @parametrize
    def test_method_create_with_all_params(self, client: Openlayer) -> None:
        test = client.projects.tests.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This test checks for duplicate rows in the dataset.",
            name="No duplicate rows",
            subtype="duplicateRowCount",
            thresholds=[
                {
                    "insight_name": "duplicateRowCount",
                    "insight_parameters": [
                        {
                            "name": "column_name",
                            "value": "Age",
                        }
                    ],
                    "measurement": "duplicateRowCount",
                    "operator": "<=",
                    "threshold_mode": "automatic",
                    "value": 0,
                }
            ],
            type="integrity",
            archived=False,
            delay_window=0,
            evaluation_window=3600,
            uses_ml_model=False,
            uses_production_data=False,
            uses_reference_dataset=False,
            uses_training_dataset=False,
            uses_validation_dataset=True,
        )
        assert_matches_type(TestCreateResponse, test, path=["response"])

    @parametrize
    def test_raw_response_create(self, client: Openlayer) -> None:
        response = client.projects.tests.with_raw_response.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This test checks for duplicate rows in the dataset.",
            name="No duplicate rows",
            subtype="duplicateRowCount",
            thresholds=[{}],
            type="integrity",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        test = response.parse()
        assert_matches_type(TestCreateResponse, test, path=["response"])

    @parametrize
    def test_streaming_response_create(self, client: Openlayer) -> None:
        with client.projects.tests.with_streaming_response.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This test checks for duplicate rows in the dataset.",
            name="No duplicate rows",
            subtype="duplicateRowCount",
            thresholds=[{}],
            type="integrity",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            test = response.parse()
            assert_matches_type(TestCreateResponse, test, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_create(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `project_id` but received ''"):
            client.projects.tests.with_raw_response.create(
                project_id="",
                description="This test checks for duplicate rows in the dataset.",
                name="No duplicate rows",
                subtype="duplicateRowCount",
                thresholds=[{}],
                type="integrity",
            )

    @parametrize
    def test_method_update(self, client: Openlayer) -> None:
        test = client.projects.tests.update(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            payloads=[{"id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e"}],
        )
        assert_matches_type(TestUpdateResponse, test, path=["response"])

    @parametrize
    def test_raw_response_update(self, client: Openlayer) -> None:
        response = client.projects.tests.with_raw_response.update(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            payloads=[{"id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e"}],
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        test = response.parse()
        assert_matches_type(TestUpdateResponse, test, path=["response"])

    @parametrize
    def test_streaming_response_update(self, client: Openlayer) -> None:
        with client.projects.tests.with_streaming_response.update(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            payloads=[{"id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e"}],
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            test = response.parse()
            assert_matches_type(TestUpdateResponse, test, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_update(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `project_id` but received ''"):
            client.projects.tests.with_raw_response.update(
                project_id="",
                payloads=[{"id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e"}],
            )

    @parametrize
    def test_method_list(self, client: Openlayer) -> None:
        test = client.projects.tests.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(TestListResponse, test, path=["response"])

    @parametrize
    def test_method_list_with_all_params(self, client: Openlayer) -> None:
        test = client.projects.tests.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            include_archived=True,
            origin_version_id="3fa85f64-5717-4562-b3fc-2c963f66afa6",
            page=1,
            per_page=1,
            suggested=True,
            type="integrity",
            uses_production_data=True,
        )
        assert_matches_type(TestListResponse, test, path=["response"])

    @parametrize
    def test_raw_response_list(self, client: Openlayer) -> None:
        response = client.projects.tests.with_raw_response.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        test = response.parse()
        assert_matches_type(TestListResponse, test, path=["response"])

    @parametrize
    def test_streaming_response_list(self, client: Openlayer) -> None:
        with client.projects.tests.with_streaming_response.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            test = response.parse()
            assert_matches_type(TestListResponse, test, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    def test_path_params_list(self, client: Openlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `project_id` but received ''"):
            client.projects.tests.with_raw_response.list(
                project_id="",
            )


class TestAsyncTests:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @parametrize
    async def test_method_create(self, async_client: AsyncOpenlayer) -> None:
        test = await async_client.projects.tests.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This test checks for duplicate rows in the dataset.",
            name="No duplicate rows",
            subtype="duplicateRowCount",
            thresholds=[{}],
            type="integrity",
        )
        assert_matches_type(TestCreateResponse, test, path=["response"])

    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        test = await async_client.projects.tests.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This test checks for duplicate rows in the dataset.",
            name="No duplicate rows",
            subtype="duplicateRowCount",
            thresholds=[
                {
                    "insight_name": "duplicateRowCount",
                    "insight_parameters": [
                        {
                            "name": "column_name",
                            "value": "Age",
                        }
                    ],
                    "measurement": "duplicateRowCount",
                    "operator": "<=",
                    "threshold_mode": "automatic",
                    "value": 0,
                }
            ],
            type="integrity",
            archived=False,
            delay_window=0,
            evaluation_window=3600,
            uses_ml_model=False,
            uses_production_data=False,
            uses_reference_dataset=False,
            uses_training_dataset=False,
            uses_validation_dataset=True,
        )
        assert_matches_type(TestCreateResponse, test, path=["response"])

    @parametrize
    async def test_raw_response_create(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.projects.tests.with_raw_response.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This test checks for duplicate rows in the dataset.",
            name="No duplicate rows",
            subtype="duplicateRowCount",
            thresholds=[{}],
            type="integrity",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        test = await response.parse()
        assert_matches_type(TestCreateResponse, test, path=["response"])

    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.projects.tests.with_streaming_response.create(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            description="This test checks for duplicate rows in the dataset.",
            name="No duplicate rows",
            subtype="duplicateRowCount",
            thresholds=[{}],
            type="integrity",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            test = await response.parse()
            assert_matches_type(TestCreateResponse, test, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_create(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `project_id` but received ''"):
            await async_client.projects.tests.with_raw_response.create(
                project_id="",
                description="This test checks for duplicate rows in the dataset.",
                name="No duplicate rows",
                subtype="duplicateRowCount",
                thresholds=[{}],
                type="integrity",
            )

    @parametrize
    async def test_method_update(self, async_client: AsyncOpenlayer) -> None:
        test = await async_client.projects.tests.update(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            payloads=[{"id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e"}],
        )
        assert_matches_type(TestUpdateResponse, test, path=["response"])

    @parametrize
    async def test_raw_response_update(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.projects.tests.with_raw_response.update(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            payloads=[{"id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e"}],
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        test = await response.parse()
        assert_matches_type(TestUpdateResponse, test, path=["response"])

    @parametrize
    async def test_streaming_response_update(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.projects.tests.with_streaming_response.update(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            payloads=[{"id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e"}],
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            test = await response.parse()
            assert_matches_type(TestUpdateResponse, test, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_update(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `project_id` but received ''"):
            await async_client.projects.tests.with_raw_response.update(
                project_id="",
                payloads=[{"id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e"}],
            )

    @parametrize
    async def test_method_list(self, async_client: AsyncOpenlayer) -> None:
        test = await async_client.projects.tests.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(TestListResponse, test, path=["response"])

    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncOpenlayer) -> None:
        test = await async_client.projects.tests.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
            include_archived=True,
            origin_version_id="3fa85f64-5717-4562-b3fc-2c963f66afa6",
            page=1,
            per_page=1,
            suggested=True,
            type="integrity",
            uses_production_data=True,
        )
        assert_matches_type(TestListResponse, test, path=["response"])

    @parametrize
    async def test_raw_response_list(self, async_client: AsyncOpenlayer) -> None:
        response = await async_client.projects.tests.with_raw_response.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        test = await response.parse()
        assert_matches_type(TestListResponse, test, path=["response"])

    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncOpenlayer) -> None:
        async with async_client.projects.tests.with_streaming_response.list(
            project_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            test = await response.parse()
            assert_matches_type(TestListResponse, test, path=["response"])

        assert cast(Any, response.is_closed) is True

    @parametrize
    async def test_path_params_list(self, async_client: AsyncOpenlayer) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `project_id` but received ''"):
            await async_client.projects.tests.with_raw_response.list(
                project_id="",
            )
