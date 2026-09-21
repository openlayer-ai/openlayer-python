# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal

import httpx

from ...._types import Body, Omit, Query, Headers, NotGiven, omit, not_given
from ...._utils import path_template, maybe_transform, async_maybe_transform
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from ...._base_client import make_request_options
from ....types.governance.frameworks import section_list_rules_params
from ....types.governance.frameworks.section_list_rules_response import SectionListRulesResponse

__all__ = ["SectionsResource", "AsyncSectionsResource"]


class SectionsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> SectionsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return SectionsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> SectionsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return SectionsResourceWithStreamingResponse(self)

    def list_rules(
        self,
        section_id: str,
        *,
        framework_id: str,
        include_results: bool | Omit = omit,
        include_subsection_rules: bool | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        project_id: str | Omit = omit,
        status: Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SectionListRulesResponse:
        """
        List the rules mapped to a section of a framework document.

        Pass `includeSubsectionRules=true` to also return the rules mapped to the
        section's subsections, which is how you get every rule covering a requirement
        and everything under it.

        Args:
          include_results: Whether to include each rule's results inline, in a `results` array.

          include_subsection_rules: Whether to also include the rules mapped to the section's subsections.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          project_id: Only include items that apply to this project.

          status: Only include items whose rule result has this compliance status.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        if not section_id:
            raise ValueError(f"Expected a non-empty value for `section_id` but received {section_id!r}")
        return self._get(
            path_template(
                "/frameworks/{framework_id}/sections/{section_id}/rules",
                framework_id=framework_id,
                section_id=section_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "include_results": include_results,
                        "include_subsection_rules": include_subsection_rules,
                        "page": page,
                        "per_page": per_page,
                        "project_id": project_id,
                        "status": status,
                    },
                    section_list_rules_params.SectionListRulesParams,
                ),
            ),
            cast_to=SectionListRulesResponse,
        )


class AsyncSectionsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncSectionsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#accessing-raw-response-data-eg-headers
        """
        return AsyncSectionsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncSectionsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openlayer-ai/openlayer-python#with_streaming_response
        """
        return AsyncSectionsResourceWithStreamingResponse(self)

    async def list_rules(
        self,
        section_id: str,
        *,
        framework_id: str,
        include_results: bool | Omit = omit,
        include_subsection_rules: bool | Omit = omit,
        page: int | Omit = omit,
        per_page: int | Omit = omit,
        project_id: str | Omit = omit,
        status: Literal["running", "passing", "failing", "skipped", "error", "pending", "due_soon"] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> SectionListRulesResponse:
        """
        List the rules mapped to a section of a framework document.

        Pass `includeSubsectionRules=true` to also return the rules mapped to the
        section's subsections, which is how you get every rule covering a requirement
        and everything under it.

        Args:
          include_results: Whether to include each rule's results inline, in a `results` array.

          include_subsection_rules: Whether to also include the rules mapped to the section's subsections.

          page: The page to return in a paginated query.

          per_page: Maximum number of items to return per page.

          project_id: Only include items that apply to this project.

          status: Only include items whose rule result has this compliance status.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not framework_id:
            raise ValueError(f"Expected a non-empty value for `framework_id` but received {framework_id!r}")
        if not section_id:
            raise ValueError(f"Expected a non-empty value for `section_id` but received {section_id!r}")
        return await self._get(
            path_template(
                "/frameworks/{framework_id}/sections/{section_id}/rules",
                framework_id=framework_id,
                section_id=section_id,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "include_results": include_results,
                        "include_subsection_rules": include_subsection_rules,
                        "page": page,
                        "per_page": per_page,
                        "project_id": project_id,
                        "status": status,
                    },
                    section_list_rules_params.SectionListRulesParams,
                ),
            ),
            cast_to=SectionListRulesResponse,
        )


class SectionsResourceWithRawResponse:
    def __init__(self, sections: SectionsResource) -> None:
        self._sections = sections

        self.list_rules = to_raw_response_wrapper(
            sections.list_rules,
        )


class AsyncSectionsResourceWithRawResponse:
    def __init__(self, sections: AsyncSectionsResource) -> None:
        self._sections = sections

        self.list_rules = async_to_raw_response_wrapper(
            sections.list_rules,
        )


class SectionsResourceWithStreamingResponse:
    def __init__(self, sections: SectionsResource) -> None:
        self._sections = sections

        self.list_rules = to_streamed_response_wrapper(
            sections.list_rules,
        )


class AsyncSectionsResourceWithStreamingResponse:
    def __init__(self, sections: AsyncSectionsResource) -> None:
        self._sections = sections

        self.list_rules = async_to_streamed_response_wrapper(
            sections.list_rules,
        )
