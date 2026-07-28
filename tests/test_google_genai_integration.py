"""Tests for the unified Google Gen AI (``google-genai``) tracer.

Covers OPEN-11891: ``trace_gemini`` previously supported only the legacy
``google-generativeai`` SDK, so users of ``genai.Client()`` (including Vertex
mode) got no auto-instrumentation and a hard error on explicit tracing.

No network calls — a real ``genai.Client`` is constructed with a fake key and its
``generate_content`` methods are replaced with stubs *before* tracing, so the
tracer wraps the stub. Spans are asserted by patching the tracer module's
``add_to_trace``, matching tests/test_bedrock_integration.py.
"""

# google-genai isn't installed in the lint env, and pytest fixtures hide autouse
# functions from static analysis. The last two matter in BOTH directions: without
# the package, `from google import genai` is an unresolved attribute on a
# namespace package; with it, `inference_id` is an extra kwarg the SDK's own
# signature doesn't declare (that's the whole point -- the tracer pops it).
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedFunction=false
# pyright: reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false

import asyncio
import contextvars
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

pytest.importorskip("google.genai")

from openlayer.lib.integrations import google_genai_tracer as gg


# ------------------------------- fixtures ------------------------------- #
@pytest.fixture(autouse=True)
def _disable_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every tracer publish path off."""
    monkeypatch.setenv("OPENLAYER_DISABLE_PUBLISH", "true")
    monkeypatch.setenv("OPENLAYER_API_KEY", "fake")

    from openlayer.lib.tracing import tracer as _tracer

    monkeypatch.setattr(_tracer, "_publish", False, raising=False)


@pytest.fixture(autouse=True)
def _reset_client_class_patch():
    """Undo any class-level ``Client.__init__`` patch so the module-level
    idempotency marker doesn't leak between tests."""
    yield
    gg._unpatch_google_genai()


# ------------------------------- helpers ------------------------------- #
def _usage(
    prompt: Optional[int] = None,
    candidates: Optional[int] = None,
    thoughts: Optional[int] = None,
    total: Optional[int] = None,
    tool_use: Optional[int] = None,
    cached: Optional[int] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
        total_token_count=total,
        tool_use_prompt_token_count=tool_use,
        cached_content_token_count=cached,
    )


class FakeResponse:
    """Minimal stand-in for ``types.GenerateContentResponse``."""

    def __init__(self, text: str = "hello", usage: Any = None, model_version: Optional[str] = None) -> None:
        self.text = text
        self.usage_metadata = usage
        self.model_version = model_version
        self.candidates = []
        # Read by the real Chat.send_message when recording history.
        self.automatic_function_calling_history = None

    def model_dump(self, **_kwargs: Any) -> Dict[str, Any]:
        return {"text": self.text}


def _make_client(vertexai: bool = False):
    """A real ``genai.Client`` that will never reach the network."""
    from google import genai

    if vertexai:
        return genai.Client(vertexai=True, project="test-project", location="us-central1")
    return genai.Client(api_key="fake-key")


def _stub_sync(client, response: Any) -> List[Dict[str, Any]]:
    """Replace sync generate_content with a stub; return the recorded calls."""
    calls: List[Dict[str, Any]] = []

    def _impl(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return response

    client.models.generate_content = _impl
    return calls


def _stub_sync_stream(client, chunks: List[Any]) -> None:
    def _impl(**_kwargs: Any):
        return iter(chunks)

    client.models.generate_content_stream = _impl


def _stub_async(client, response: Any) -> None:
    async def _impl(**_kwargs: Any) -> Any:
        return response

    client.aio.models.generate_content = _impl


def _stub_async_stream(client, chunks: List[Any]) -> None:
    async def _impl(**_kwargs: Any):
        async def _gen():
            for chunk in chunks:
                yield chunk

        return _gen()

    client.aio.models.generate_content_stream = _impl


# ------------------------------- model names ------------------------------- #
class TestNormalizeModelName:
    """Prefixed model strings price at $0, so this is load-bearing."""

    @pytest.mark.parametrize(
        "raw",
        [
            "gemini-2.5-flash",
            "models/gemini-2.5-flash",
            "publishers/google/models/gemini-2.5-flash",
            "projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash",
        ],
    )
    def test_all_four_shapes_reduce_to_bare_slug(self, raw: str) -> None:
        assert gg._normalize_model_name(raw) == "gemini-2.5-flash"

    def test_version_suffix_is_preserved(self) -> None:
        """Rewriting suffixes would misreport which model actually ran."""
        assert gg._normalize_model_name("gemini-2.0-flash-001") == "gemini-2.0-flash-001"

    @pytest.mark.parametrize("raw", [None, "", 123])
    def test_missing_or_non_string_is_unknown(self, raw: Any) -> None:
        assert gg._normalize_model_name(raw) == "unknown"


# ------------------------------- token accounting ------------------------------- #
class TestExtractUsage:
    """OPEN-11891: prompt+candidates undercounts 54x on thinking models."""

    def test_thinking_tokens_measured_case(self) -> None:
        """The real 2.5-flash measurement: 8 / 6 / 743 -> 757, not 14."""
        prompt, completion, total, extras = gg._extract_usage(_usage(prompt=8, candidates=6, thoughts=743, total=757))
        assert prompt == 8
        assert completion == 749, "thinking tokens must be billed as output"
        assert total == 757, "must use total_token_count, not prompt + candidates"
        assert prompt + completion == total
        assert extras["thoughtsTokens"] == 743
        assert extras["candidatesTokens"] == 6

    def test_total_is_derived_when_absent(self) -> None:
        prompt, completion, total, _ = gg._extract_usage(_usage(prompt=10, candidates=5, thoughts=2, tool_use=3))
        assert (prompt, completion, total) == (10, 7, 20)

    def test_no_thinking_tokens_behaves_conventionally(self) -> None:
        prompt, completion, total, extras = gg._extract_usage(_usage(prompt=8, candidates=4, total=12))
        assert (prompt, completion, total) == (8, 4, 12)
        assert "thoughtsTokens" not in extras

    def test_none_usage_is_zeroed(self) -> None:
        assert gg._extract_usage(None) == (0, 0, 0, {})

    def test_cached_and_tool_use_land_in_extras(self) -> None:
        _, _, _, extras = gg._extract_usage(_usage(prompt=5, candidates=5, total=20, tool_use=4, cached=6))
        assert extras["toolUsePromptTokens"] == 4
        assert extras["cachedContentTokens"] == 6


# ------------------------------- sync paths ------------------------------- #
class TestSyncGeneration:
    def test_non_streaming_emits_span_with_slug_and_bare_model(self) -> None:
        client = _make_client()
        calls = _stub_sync(client, FakeResponse("hi there", _usage(prompt=8, candidates=6, thoughts=743, total=757)))
        gg.trace_google_genai(client)

        with patch.object(gg, "add_to_trace") as mock_add:
            response = client.models.generate_content(model="models/gemini-2.5-flash", contents="hi")

        assert response.text == "hi there"
        mock_add.assert_called_once()
        kwargs = mock_add.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-flash", "must be normalized for cost lookup"
        assert kwargs["tokens"] == 757
        assert kwargs["completion_tokens"] == 749
        assert kwargs["output"] == "hi there"
        assert kwargs["inputs"] == {"prompt": [{"role": "user", "content": "hi"}]}
        assert kwargs["metadata"]["requestedModel"] == "models/gemini-2.5-flash"
        # inference_id must never reach the SDK
        assert "inference_id" not in calls[0]

    def test_provider_is_the_cost_slug(self) -> None:
        """provider must be a real slug; 'Google' silently prices at $0."""
        assert gg.PROVIDER == "gemini"

        client = _make_client()
        _stub_sync(client, FakeResponse())
        gg.trace_google_genai(client)

        from openlayer.lib.tracing import tracer as _tracer

        with patch.object(_tracer, "add_chat_completion_step_to_trace") as mock_step:
            client.models.generate_content(model="gemini-2.5-flash", contents="hi")

        assert mock_step.call_args.kwargs["provider"] == "gemini"

    def test_inference_id_is_forwarded_as_step_id(self) -> None:
        client = _make_client()
        _stub_sync(client, FakeResponse())
        gg.trace_google_genai(client)

        with patch.object(gg, "add_to_trace") as mock_add:
            client.models.generate_content(model="gemini-2.5-flash", contents="hi", inference_id="abc-123")

        assert mock_add.call_args.kwargs["id"] == "abc-123"

    def test_streaming_passes_chunks_through_and_emits_once(self) -> None:
        chunks = [
            FakeResponse("Hel", _usage(prompt=8, candidates=2, total=10)),
            FakeResponse("lo!", _usage(prompt=8, candidates=5, total=13)),
        ]
        client = _make_client()
        _stub_sync_stream(client, chunks)
        gg.trace_google_genai(client)

        with patch.object(gg, "add_to_trace") as mock_add:
            received = list(client.models.generate_content_stream(model="gemini-2.5-flash", contents="hi"))

        assert [c.text for c in received] == ["Hel", "lo!"]
        mock_add.assert_called_once()
        kwargs = mock_add.call_args.kwargs
        assert kwargs["output"] == "Hello!"
        # usage_metadata is cumulative per chunk: last wins, never summed
        assert kwargs["tokens"] == 13, "summing chunks would give 23"
        assert kwargs["metadata"]["timeToFirstToken"] is not None

    def test_streaming_span_emitted_when_consumer_abandons_iterator(self) -> None:
        chunks = [FakeResponse("a", _usage(prompt=1, candidates=1, total=2)) for _ in range(5)]
        client = _make_client()
        _stub_sync_stream(client, chunks)
        gg.trace_google_genai(client)

        with patch.object(gg, "add_to_trace") as mock_add:
            stream = client.models.generate_content_stream(model="gemini-2.5-flash", contents="hi")
            next(iter(stream))
            del stream

        mock_add.assert_called_once()


# ------------------------------- async paths ------------------------------- #
class TestAsyncGeneration:
    def test_non_streaming_async(self) -> None:
        client = _make_client()
        _stub_async(client, FakeResponse("async hi", _usage(prompt=3, candidates=4, total=7)))
        gg.trace_google_genai(client)

        with patch.object(gg, "add_to_trace") as mock_add:
            response = asyncio.run(client.aio.models.generate_content(model="gemini-2.5-flash", contents="hi"))

        assert response.text == "async hi"
        assert mock_add.call_args.kwargs["tokens"] == 7

    def test_streaming_async_is_await_then_async_for(self) -> None:
        """AsyncModels.generate_content_stream is a coroutine returning an
        AsyncIterator — the wrapper must preserve that two-step shape."""
        chunks = [
            FakeResponse("x", _usage(prompt=2, candidates=1, total=3)),
            FakeResponse("y", _usage(prompt=2, candidates=3, total=5)),
        ]
        client = _make_client()
        _stub_async_stream(client, chunks)
        gg.trace_google_genai(client)

        async def _drive() -> List[str]:
            stream = await client.aio.models.generate_content_stream(model="gemini-2.5-flash", contents="hi")
            return [chunk.text async for chunk in stream if chunk.text]

        with patch.object(gg, "add_to_trace") as mock_add:
            texts = asyncio.run(_drive())

        assert texts == ["x", "y"]
        mock_add.assert_called_once()
        assert mock_add.call_args.kwargs["output"] == "xy"
        assert mock_add.call_args.kwargs["tokens"] == 5


# ------------------------------- chats ------------------------------- #
class TestChatsCoverage:
    def test_chat_send_message_is_traced_via_patched_models(self) -> None:
        """client.chats wires to the same Models instance, so it traces for free.
        Locked in so a refactor can't silently drop chat coverage."""
        client = _make_client()

        def _impl(**_kwargs: Any) -> Any:
            return FakeResponse("chat reply", _usage(prompt=4, candidates=6, total=10))

        client.models.generate_content = _impl
        gg.trace_google_genai(client)

        with patch.object(gg, "add_to_trace") as mock_add:
            chat = client.chats.create(model="gemini-2.5-flash")
            chat.send_message("hello")

        mock_add.assert_called_once()
        assert mock_add.call_args.kwargs["model"] == "gemini-2.5-flash"


# ------------------------------- Vertex vs AI Studio ------------------------------- #
class TestModeMetadata:
    def test_ai_studio_mode(self) -> None:
        client = _make_client(vertexai=False)
        _stub_sync(client, FakeResponse())
        gg.trace_google_genai(client)

        with patch.object(gg, "add_to_trace") as mock_add:
            client.models.generate_content(model="gemini-2.5-flash", contents="hi")

        assert mock_add.call_args.kwargs["metadata"]["llm_system"] == "google_ai_studio"

    def test_vertex_mode_records_system_and_project(self) -> None:
        client = _make_client(vertexai=True)
        _stub_sync(client, FakeResponse())
        gg.trace_google_genai(client)

        with patch.object(gg, "add_to_trace") as mock_add:
            client.models.generate_content(
                model="projects/test-project/locations/us-central1/publishers/google/models/gemini-2.5-flash",
                contents="hi",
            )

        metadata = mock_add.call_args.kwargs["metadata"]
        assert metadata["llm_system"] == "google_vertex"
        assert metadata["gcp_project"] == "test-project"
        assert metadata["gcp_location"] == "us-central1"
        # The full Vertex resource path must still reduce to the bare slug
        assert mock_add.call_args.kwargs["model"] == "gemini-2.5-flash"


# ------------------------------- ADK dedup ------------------------------- #
class TestAdkSuppression:
    def test_no_span_when_adk_llm_step_is_active(self) -> None:
        """ADK's api_client IS a genai.Client, so both integrations activate for
        ADK users. Without suppression every ADK call double-counts."""
        client = _make_client()
        _stub_sync(client, FakeResponse("adk"))
        gg.trace_google_genai(client)

        fake_adk = SimpleNamespace(_current_llm_step=contextvars.ContextVar("test_llm_step", default=None))
        fake_adk._current_llm_step.set(object())

        with patch.dict("sys.modules", {gg._ADK_TRACER_MODULE: fake_adk}):
            with patch.object(gg, "add_to_trace") as mock_add:
                response = client.models.generate_content(model="gemini-2.5-flash", contents="hi")

        assert response.text == "adk", "the underlying call must still run"
        mock_add.assert_not_called()

    def test_span_emitted_when_adk_step_is_not_active(self) -> None:
        client = _make_client()
        _stub_sync(client, FakeResponse("no adk"))
        gg.trace_google_genai(client)

        fake_adk = SimpleNamespace(_current_llm_step=contextvars.ContextVar("test_llm_step_idle", default=None))

        with patch.dict("sys.modules", {gg._ADK_TRACER_MODULE: fake_adk}):
            with patch.object(gg, "add_to_trace") as mock_add:
                client.models.generate_content(model="gemini-2.5-flash", contents="hi")

        mock_add.assert_called_once()

    def test_absent_adk_module_is_not_treated_as_active(self) -> None:
        import sys

        assert gg._ADK_TRACER_MODULE not in sys.modules or gg._adk_span_active() is False

    def test_inference_id_stripped_even_on_suppressed_path(self) -> None:
        """The SDK rejects unknown kwargs, so the pop must precede the ADK check."""
        client = _make_client()
        calls = _stub_sync(client, FakeResponse())
        gg.trace_google_genai(client)

        fake_adk = SimpleNamespace(_current_llm_step=contextvars.ContextVar("test_llm_step_pop", default=None))
        fake_adk._current_llm_step.set(object())

        with patch.dict("sys.modules", {gg._ADK_TRACER_MODULE: fake_adk}):
            client.models.generate_content(model="gemini-2.5-flash", contents="hi", inference_id="x")

        assert "inference_id" not in calls[0]


# ------------------------------- idempotency & dispatch ------------------------------- #
class TestIdempotency:
    def test_double_patch_wraps_once(self) -> None:
        client = _make_client()
        _stub_sync(client, FakeResponse())

        gg.trace_google_genai(client)
        first = client.models.generate_content
        gg.trace_google_genai(client)

        assert client.models.generate_content is first

        with patch.object(gg, "add_to_trace") as mock_add:
            client.models.generate_content(model="gemini-2.5-flash", contents="hi")

        assert mock_add.call_count == 1, "a double wrap would emit two spans"

    def test_auto_instrument_traces_newly_constructed_clients(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The stub is installed on the CLASS before the class-init patch, so the
        per-instance wrapper wraps the stub rather than being clobbered by it."""
        from google.genai import models as genai_models

        def _impl(_self: Any, **_kwargs: Any) -> Any:
            return FakeResponse("auto-traced", _usage(prompt=2, candidates=3, total=5))

        monkeypatch.setattr(genai_models.Models, "generate_content", _impl)

        gg._patch_google_genai()
        client = _make_client()

        with patch.object(gg, "add_to_trace") as mock_add:
            response = client.models.generate_content(model="gemini-2.5-flash", contents="hi")

        assert response.text == "auto-traced"
        mock_add.assert_called_once()
        assert mock_add.call_args.kwargs["tokens"] == 5

    def test_auto_instrument_then_explicit_call_does_not_rewrap(self) -> None:
        gg._patch_google_genai()
        client = _make_client()

        # Construction already traced this instance via the patched __init__.
        assert getattr(client.models, "_openlayer_patched", False) is True
        wrapped = client.models.generate_content

        assert gg.trace_google_genai(client) is client
        assert client.models.generate_content is wrapped, "explicit call must not re-wrap"


class TestTraceGeminiDispatch:
    def test_client_routes_to_the_unified_tracer(self) -> None:
        from openlayer.lib import trace_gemini

        client = _make_client()
        _stub_sync(client, FakeResponse())
        assert trace_gemini(client) is client
        assert getattr(client, "_openlayer_patched", False) is True

    def test_trace_google_genai_public_alias(self) -> None:
        from openlayer.lib import trace_google_genai

        client = _make_client()
        _stub_sync(client, FakeResponse())
        assert trace_google_genai(client) is client

    def test_bad_type_names_both_accepted_clients(self) -> None:
        from openlayer.lib import trace_gemini

        with pytest.raises(ValueError) as excinfo:
            trace_gemini(object())

        message = str(excinfo.value)
        assert "google.genai.Client" in message
        assert "GenerativeModel" in message


class TestAutoRegistry:
    def test_google_genai_entry_registered_and_probe_resolves(self) -> None:
        from openlayer.lib.integrations._auto import REGISTRY, _is_installed

        spec = next((s for s in REGISTRY if s.name == "google_genai"), None)
        assert spec is not None, "auto_instrument() must cover the unified SDK"
        assert spec.probe == "google.genai"
        assert spec.unpatch is not None, "unpatch_all() must be symmetric"
        assert _is_installed(spec.probe) is True

    def test_legacy_gemini_entry_is_kept(self) -> None:
        """Both packages can be installed side by side."""
        from openlayer.lib.integrations._auto import REGISTRY

        assert any(s.name == "gemini" and s.probe == "google.generativeai" for s in REGISTRY)


# ------------------------------- input formatting ------------------------------- #
class TestInputFormatting:
    def test_plain_string(self) -> None:
        assert gg._format_input_messages("hi") == [{"role": "user", "content": "hi"}]

    def test_none_contents(self) -> None:
        assert gg._format_input_messages(None) == []

    def test_content_objects_with_parts(self) -> None:
        contents = [
            SimpleNamespace(role="user", parts=[SimpleNamespace(text="first")]),
            SimpleNamespace(role="model", parts=[SimpleNamespace(text="second")]),
        ]
        assert gg._format_input_messages(contents) == [
            {"role": "user", "content": "first"},
            {"role": "model", "content": "second"},
        ]

    def test_dict_message_form(self) -> None:
        contents = [{"role": "user", "parts": [{"text": "from dict"}]}]
        assert gg._format_input_messages(contents) == [{"role": "user", "content": "from dict"}]

    def test_system_instruction_becomes_leading_system_message(self) -> None:
        messages = gg._format_input_messages("hi", {"system_instruction": "be terse"})
        assert messages[0] == {"role": "system", "content": "be terse"}
        assert messages[1] == {"role": "user", "content": "hi"}


class TestModelParameters:
    def test_reads_pydantic_config(self) -> None:
        from google.genai import types

        config = types.GenerateContentConfig(temperature=0.25, max_output_tokens=64, top_p=0.9)
        params = gg.get_model_parameters(config)
        assert params["temperature"] == 0.25
        assert params["max_output_tokens"] == 64
        assert params["top_p"] == 0.9

    def test_reads_dict_config(self) -> None:
        params = gg.get_model_parameters({"temperature": 0.5, "top_k": 3})
        assert params["temperature"] == 0.5
        assert params["top_k"] == 3

    def test_thinking_budget_is_surfaced(self) -> None:
        from google.genai import types

        config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0))
        assert gg.get_model_parameters(config)["thinking_budget"] == 0

    def test_none_config(self) -> None:
        assert gg.get_model_parameters(None)["temperature"] is None
