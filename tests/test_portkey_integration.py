"""Test Portkey tracer integration."""

import json
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest  # type: ignore


class TestPortkeyIntegration:
    """Test Portkey integration functionality."""

    def test_import_without_portkey(self) -> None:
        """Module should import even when Portkey is unavailable."""
        from openlayer.lib.integrations import portkey_tracer  # noqa: F401

        assert hasattr(portkey_tracer, "HAVE_PORTKEY")

    def test_trace_portkey_raises_import_error_without_dependency(self) -> None:
        """trace_portkey should raise ImportError when Portkey is missing."""
        with patch("openlayer.lib.integrations.portkey_tracer.HAVE_PORTKEY", False):
            from openlayer.lib.integrations.portkey_tracer import trace_portkey

            with pytest.raises(ImportError) as exc_info:  # type: ignore
                trace_portkey()

            message = str(exc_info.value)  # type: ignore[attr-defined]
            assert "Portkey library is not installed" in message
            assert "pip install portkey-ai" in message

    def test_trace_portkey_patches_portkey_client(self) -> None:
        """trace_portkey should wrap Portkey chat completions for tracing."""

        class DummyPortkey:  # pylint: disable=too-few-public-methods
            """Lightweight Portkey stand-in used for patching behavior."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
                completions = SimpleNamespace(create=Mock(name="original_create"))
                self.chat = SimpleNamespace(completions=completions)
                self._init_args = (args, kwargs)
                self.original_create = completions.create

        with patch("openlayer.lib.integrations.portkey_tracer.HAVE_PORTKEY", True), patch(
            "openlayer.lib.integrations.portkey_tracer.Portkey", DummyPortkey, create=True
        ), patch(
            "openlayer.lib.integrations.portkey_tracer.handle_non_streaming_create",
            autospec=True,
        ) as mock_non_streaming, patch(
            "openlayer.lib.integrations.portkey_tracer.handle_streaming_create",
            autospec=True,
        ) as mock_streaming:
            mock_non_streaming.return_value = "non-stream-result"
            mock_streaming.return_value = "stream-result"

            from openlayer.lib.integrations.portkey_tracer import trace_portkey

            trace_portkey()

            client = DummyPortkey()
            # Non-streaming
            result_non_stream = client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
            assert result_non_stream == "non-stream-result"
            assert mock_non_streaming.call_count == 1
            non_stream_kwargs = mock_non_streaming.call_args.kwargs
            assert non_stream_kwargs["create_func"] is client.original_create
            assert non_stream_kwargs["inference_id"] is None

            # Streaming path
            result_stream = client.chat.completions.create(
                messages=[{"role": "user", "content": "hi"}], stream=True, inference_id="inference-123"
            )
            assert result_stream == "stream-result"
            assert mock_streaming.call_count == 1
            stream_kwargs = mock_streaming.call_args.kwargs
            assert stream_kwargs["create_func"] is client.original_create
            assert stream_kwargs["inference_id"] == "inference-123"

    def test_detect_provider_from_model_name(self) -> None:
        """Provider detection should match model naming heuristics."""
        from openlayer.lib.integrations.portkey_tracer import detect_provider_from_model_name

        test_cases = [
            ("gpt-4", "OpenAI"),
            ("Gpt-3.5-turbo", "OpenAI"),
            ("claude-3-opus", "Anthropic"),
            ("gemini-pro", "Google"),
            ("meta-llama-3-70b", "Meta"),
            ("mixtral-8x7b", "Mistral"),
            ("command-r", "Cohere"),
            ("unknown-model", "Portkey"),
        ]

        for model_name, expected in test_cases:
            assert detect_provider_from_model_name(model_name) == expected

    def test_get_model_parameters(self) -> None:
        """Ensure OpenAI-compatible kwargs are extracted."""
        from openlayer.lib.integrations.portkey_tracer import get_model_parameters

        kwargs = {
            "temperature": 0.5,
            "top_p": 0.7,
            "max_tokens": 256,
            "n": 3,
            "stream": True,
            "stop": ["END"],
            "presence_penalty": 0.1,
            "frequency_penalty": -0.1,
            "logit_bias": {"1": -1},
            "logprobs": True,
            "top_logprobs": 5,
            "parallel_tool_calls": False,
            "seed": 123,
            "response_format": {"type": "json_object"},
            "timeout": 42,
            "api_base": "https://api.example.com",
            "api_version": "2024-05-01",
        }

        params = get_model_parameters(kwargs)

        expected = kwargs.copy()
        assert params == expected

    def test_extract_portkey_metadata(self) -> None:
        """Portkey metadata should redact sensitive headers and include base URL."""
        from openlayer.lib.integrations.portkey_tracer import extract_portkey_metadata

        client = SimpleNamespace(
            base_url="https://gateway.portkey.ai",
            headers={
                "X-Portkey-Api-Key": "secret",
                "X-Portkey-Provider": "openai",
                "Authorization": "Bearer ignored",
            },
        )

        metadata = extract_portkey_metadata(client)

        assert metadata["isPortkey"] is True
        assert metadata["portkeyBaseUrl"] == "https://gateway.portkey.ai"
        assert metadata["portkeyHeaders"]["X-Portkey-Api-Key"] == "***"
        assert metadata["portkeyHeaders"]["X-Portkey-Provider"] == "openai"
        assert "Authorization" not in metadata["portkeyHeaders"]

    def test_extract_portkey_unit_metadata(self) -> None:
        """Unit metadata should capture headers, cost, and provider hints."""
        from openlayer.lib.integrations.portkey_tracer import extract_portkey_unit_metadata

        unit = SimpleNamespace(
            system_fingerprint="fingerprint-123",
            _response_headers={
                "x-portkey-trace-id": "trace-1",
                "x-portkey-provider": "anthropic",
                "x-portkey-cache-status": "HIT",
                "x-portkey-cost": "0.45",
                "content-type": "application/json",
            },
        )

        metadata = extract_portkey_unit_metadata(unit, "claude-3-opus")

        assert metadata["system_fingerprint"] == "fingerprint-123"
        assert metadata["portkey_trace_id"] == "trace-1"
        assert metadata["provider"] == "anthropic"
        assert metadata["portkey_cache_status"] == "HIT"
        assert metadata["cost"] == pytest.approx(0.45)
        assert metadata["portkey_model"] == "claude-3-opus"
        assert metadata["response_headers"]["content-type"] == "application/json"

    def test_extract_portkey_unit_metadata_with_dict_like_headers(self) -> None:
        """Unit metadata should work with dict-like objects (not just dicts)."""
        from openlayer.lib.integrations.portkey_tracer import extract_portkey_unit_metadata

        # Create a dict-like object (has .items() but not isinstance(dict))
        class DictLikeHeaders:
            def __init__(self):
                self._data = {
                    "x-portkey-trace-id": "trace-2",
                    "x-portkey-provider": "openai",
                    "x-portkey-cache-status": "MISS",
                }

            def items(self):
                return self._data.items()

        unit = SimpleNamespace(
            _response_headers=DictLikeHeaders(),
        )

        metadata = extract_portkey_unit_metadata(unit, "gpt-4")

        assert metadata["portkey_trace_id"] == "trace-2"
        assert metadata["provider"] == "openai"
        assert metadata["portkey_cache_status"] == "MISS"

    def test_extract_usage_from_response(self) -> None:
        """Usage extraction should read OpenAI-style usage objects."""
        from openlayer.lib.integrations.portkey_tracer import extract_usage

        usage = SimpleNamespace(total_tokens=50, prompt_tokens=20, completion_tokens=30)
        response = SimpleNamespace(usage=usage)

        assert extract_usage(response) == {
            "total_tokens": 50,
            "prompt_tokens": 20,
            "completion_tokens": 30,
        }

        response_no_usage = SimpleNamespace()
        assert extract_usage(response_no_usage) == {
            "total_tokens": None,
            "prompt_tokens": None,
            "completion_tokens": None,
        }

    def test_extract_usage_from_chunk(self) -> None:
        """Usage data should be derived from multiple potential chunk attributes."""
        from openlayer.lib.integrations.portkey_tracer import extract_usage

        chunk_direct = SimpleNamespace(
            usage=SimpleNamespace(total_tokens=120, prompt_tokens=40, completion_tokens=80)
        )
        assert extract_usage(chunk_direct) == {
            "total_tokens": 120,
            "prompt_tokens": 40,
            "completion_tokens": 80,
        }

        chunk_hidden = SimpleNamespace(
            _hidden_params={"usage": {"total_tokens": 30, "prompt_tokens": 10, "completion_tokens": 20}}
        )
        assert extract_usage(chunk_hidden) == {
            "total_tokens": 30,
            "prompt_tokens": 10,
            "completion_tokens": 20,
        }

        class ChunkWithModelDump:  # pylint: disable=too-few-public-methods
            def model_dump(self) -> Dict[str, Any]:
                return {"usage": {"total_tokens": 12, "prompt_tokens": 5, "completion_tokens": 7}}

        assert extract_usage(ChunkWithModelDump()) == {
            "total_tokens": 12,
            "prompt_tokens": 5,
            "completion_tokens": 7,
        }

    def test_calculate_streaming_usage_and_cost_with_actual_usage(self) -> None:
        """Actual usage data should be returned when available."""
        from openlayer.lib.integrations.portkey_tracer import calculate_streaming_usage_and_cost

        latest_usage = {"total_tokens": 100, "prompt_tokens": 40, "completion_tokens": 60}
        latest_metadata = {"cost": 0.99}

        result = calculate_streaming_usage_and_cost(
            chunks=[],
            messages=[],
            output_content="",
            model_name="gpt-4",
            latest_usage_data=latest_usage,
            latest_chunk_metadata=latest_metadata,
        )

        assert result == (60, 40, 100, 0.99)

    def test_calculate_streaming_usage_and_cost_fallback_estimation(self) -> None:
        """Fallback estimation should approximate tokens and cost when usage is missing."""
        from openlayer.lib.integrations.portkey_tracer import calculate_streaming_usage_and_cost

        output_content = "Generated answer text."
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Tell me something interesting."},
        ]

        completion_tokens, prompt_tokens, total_tokens, cost = calculate_streaming_usage_and_cost(
            chunks=[{"usage": None}],
            messages=messages,
            output_content=output_content,
            model_name="gpt-3.5-turbo",
            latest_usage_data={"total_tokens": None, "prompt_tokens": None, "completion_tokens": None},
            latest_chunk_metadata={},
        )

        assert completion_tokens >= 1
        assert prompt_tokens >= 1
        assert total_tokens == (completion_tokens or 0) + (prompt_tokens or 0)
        assert cost is not None
        assert cost >= 0

    def test_detect_provider_from_response_prefers_headers(self) -> None:
        """Provider detection should prioritize Portkey headers."""
        from openlayer.lib.integrations.portkey_tracer import detect_provider

        client = SimpleNamespace()
        response = SimpleNamespace()

        with patch(
            "openlayer.lib.integrations.portkey_tracer._provider_from_portkey_headers", return_value="header-provider"
        ):
            assert detect_provider(response, client, "gpt-4") == "header-provider"

    def test_detect_provider_from_chunk_prefers_headers(self) -> None:
        """Provider detection from chunk should prioritize header-derived values."""
        from openlayer.lib.integrations.portkey_tracer import detect_provider

        client = SimpleNamespace()
        chunk = SimpleNamespace()

        with patch(
            "openlayer.lib.integrations.portkey_tracer._provider_from_portkey_headers", return_value="header-provider"
        ):
            assert detect_provider(chunk, client, "gpt-4") == "header-provider"

    def test_detect_provider_from_response_fallback(self) -> None:
        """Provider detection should fall back to response metadata or model name."""
        from openlayer.lib.integrations.portkey_tracer import detect_provider

        client = SimpleNamespace(headers={"x-portkey-provider": "openai"})
        response = SimpleNamespace(
            _response_headers={"X-Portkey-Provider": "anthropic"},
            response_metadata={"provider": "anthropic"},
        )

        with patch(
            "openlayer.lib.integrations.portkey_tracer._provider_from_portkey_headers", return_value=None
        ):
            assert detect_provider(response, client, "mistral-7b") == "anthropic"

    def test_detect_provider_from_chunk_fallback(self) -> None:
        """Chunk provider detection should fall back gracefully."""
        from openlayer.lib.integrations.portkey_tracer import detect_provider

        chunk = SimpleNamespace(
            response_metadata={"provider": "cohere"},
            _response_headers={"X-Portkey-Provider": "cohere"},
        )
        client = SimpleNamespace()

        with patch(
            "openlayer.lib.integrations.portkey_tracer._provider_from_portkey_headers", return_value=None
        ):
            assert detect_provider(chunk, client, "command-r") == "cohere"

    def test_provider_from_portkey_headers(self) -> None:
        """Header helper should identify provider values on the client."""
        from openlayer.lib.integrations.portkey_tracer import _provider_from_portkey_headers

        client = SimpleNamespace(
            default_headers={"X-Portkey-Provider": "openai"},
            headers={"X-Portkey-Provider": "anthropic"},
        )

        assert _provider_from_portkey_headers(client) == "openai"

    def test_parse_non_streaming_output_data(self) -> None:
        """Output parsing should support content, function calls, and tool calls."""
        from openlayer.lib.integrations.portkey_tracer import parse_non_streaming_output_data

        # Content message
        response_content = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello!", function_call=None, tool_calls=None))]
        )
        assert parse_non_streaming_output_data(response_content) == "Hello!"

        # Function call
        response_function = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        function_call=SimpleNamespace(name="do_something", arguments=json.dumps({"value": 1})),
                        tool_calls=None,
                    )
                )
            ]
        )
        assert parse_non_streaming_output_data(response_function) == {"name": "do_something", "arguments": {"value": 1}}

        # Tool call
        response_tool = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        function_call=None,
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(name="call_tool", arguments=json.dumps({"value": 2}))
                            )
                        ],
                    )
                )
            ]
        )
        assert parse_non_streaming_output_data(response_tool) == {"name": "call_tool", "arguments": {"value": 2}}

    def test_create_trace_args(self) -> None:
        """Trace argument helper should include optional id and cost."""
        from openlayer.lib.integrations.portkey_tracer import create_trace_args

        args = create_trace_args(
            end_time=1.0,
            inputs={"prompt": []},
            output="response",
            latency=123.0,
            tokens=10,
            prompt_tokens=4,
            completion_tokens=6,
            model="gpt-4",
            id="trace-id",
            cost=0.42,
        )

        assert args["id"] == "trace-id"
        assert args["cost"] == 0.42
        assert args["metadata"] == {}

    def test_add_to_trace_uses_provider_metadata(self) -> None:
        """add_to_trace should pass provider metadata through to tracer."""
        from openlayer.lib.integrations.portkey_tracer import add_to_trace

        with patch(
            "openlayer.lib.integrations.portkey_tracer.tracer.add_chat_completion_step_to_trace"
        ) as mock_add:
            add_to_trace(
                end_time=1.0,
                inputs={},
                output=None,
                latency=10.0,
                tokens=None,
                prompt_tokens=None,
                completion_tokens=None,
                model="model",
                metadata={},
            )

            _, kwargs = mock_add.call_args
            assert kwargs["provider"] == "Portkey"
            assert kwargs["name"] == "Portkey Chat Completion"

            add_to_trace(
                end_time=2.0,
                inputs={},
                output=None,
                latency=5.0,
                tokens=None,
                prompt_tokens=None,
                completion_tokens=None,
                model="model",
                metadata={"provider": "OpenAI"},
            )

            assert mock_add.call_count == 2
            assert mock_add.call_args.kwargs["provider"] == "OpenAI"

    def test_handle_streaming_create_delegates_to_stream_chunks(self) -> None:
        """handle_streaming_create should call the original create and stream_chunks."""
        from openlayer.lib.integrations.portkey_tracer import handle_streaming_create

        client = SimpleNamespace()
        create_func = Mock(return_value=iter(["chunk"]))

        with patch(
            "openlayer.lib.integrations.portkey_tracer.stream_chunks", return_value=iter(["chunk"])
        ) as mock_stream_chunks:
            result_iterator = handle_streaming_create(
                client,
                "arg-1",
                create_func=create_func,
                inference_id="stream-id",
                foo="bar",
            )

            assert list(result_iterator) == ["chunk"]
            create_func.assert_called_once_with("arg-1", foo="bar")
            mock_stream_chunks.assert_called_once()
            stream_kwargs = mock_stream_chunks.call_args.kwargs
            assert stream_kwargs["client"] is client
            assert stream_kwargs["inference_id"] == "stream-id"
            assert stream_kwargs["kwargs"] == {"foo": "bar"}
            assert stream_kwargs["chunks"] is create_func.return_value

    def test_stream_chunks_traces_completion(self) -> None:
        """stream_chunks should yield all chunks and record a traced step."""
        from openlayer.lib.integrations.portkey_tracer import stream_chunks

        chunk_a = object()
        chunk_b = object()
        chunks = [chunk_a, chunk_b]
        kwargs = {"messages": [{"role": "user", "content": "hello"}]}
        client = SimpleNamespace()

        with patch(
            "openlayer.lib.integrations.portkey_tracer.add_to_trace", autospec=True
        ) as mock_add_to_trace, patch(
            "openlayer.lib.integrations.portkey_tracer.extract_usage", autospec=True
        ) as mock_usage, patch(
            "openlayer.lib.integrations.portkey_tracer.extract_portkey_unit_metadata", autospec=True
        ) as mock_unit_metadata, patch(
            "openlayer.lib.integrations.portkey_tracer.detect_provider", autospec=True
        ) as mock_detect_provider, patch(
            "openlayer.lib.integrations.portkey_tracer.get_delta_from_chunk", autospec=True
        ) as mock_delta, patch(
            "openlayer.lib.integrations.portkey_tracer.calculate_streaming_usage_and_cost", autospec=True
        ) as mock_calc, patch(
            "openlayer.lib.integrations.portkey_tracer.extract_portkey_metadata", autospec=True
        ) as mock_client_metadata, patch(
            "openlayer.lib.integrations.portkey_tracer.time.time", side_effect=[100.0, 100.05, 100.2]
        ):
            mock_usage.side_effect = [
                {"total_tokens": None, "prompt_tokens": None, "completion_tokens": None},
                {"total_tokens": 10, "prompt_tokens": 4, "completion_tokens": 6},
            ]
            mock_unit_metadata.side_effect = [{}, {"cost": 0.1}]
            mock_detect_provider.side_effect = ["OpenAI", "OpenAI"]
            mock_delta.side_effect = [
                SimpleNamespace(content="Hello ", function_call=None, tool_calls=None),
                SimpleNamespace(content="World", function_call=None, tool_calls=None),
            ]
            mock_calc.return_value = (6, 4, 10, 0.1)
            mock_client_metadata.return_value = {"portkeyBaseUrl": "https://gateway"}

            yielded = list(
                stream_chunks(
                    chunks=iter(chunks),
                    kwargs=kwargs,
                    client=client,
                    inference_id="trace-123",
                )
            )

            assert yielded == chunks
            mock_add_to_trace.assert_called_once()
            trace_kwargs = mock_add_to_trace.call_args.kwargs
            assert trace_kwargs["metadata"]["provider"] == "OpenAI"
            assert trace_kwargs["metadata"]["portkeyBaseUrl"] == "https://gateway"
            assert trace_kwargs["id"] == "trace-123"
            assert trace_kwargs["tokens"] == 10
            assert trace_kwargs["latency"] == pytest.approx(200.0)

    def test_handle_non_streaming_create_traces_completion(self) -> None:
        """handle_non_streaming_create should record a traced step for completions."""
        from openlayer.lib.integrations.portkey_tracer import handle_non_streaming_create

        response = SimpleNamespace(model="gpt-4", system_fingerprint="fp-1")
        client = SimpleNamespace()
        create_func = Mock(return_value=response)

        with patch(
            "openlayer.lib.integrations.portkey_tracer.parse_non_streaming_output_data", return_value="output"
        ), patch(
            "openlayer.lib.integrations.portkey_tracer.extract_usage",
            return_value={"total_tokens": 10, "prompt_tokens": 4, "completion_tokens": 6},
        ), patch(
            "openlayer.lib.integrations.portkey_tracer.detect_provider", return_value="OpenAI"
        ), patch(
            "openlayer.lib.integrations.portkey_tracer.extract_portkey_unit_metadata",
            return_value={"cost": 0.25},
        ), patch(
            "openlayer.lib.integrations.portkey_tracer.extract_portkey_metadata",
            return_value={"portkeyHeaders": {"X-Portkey-Provider": "openai"}},
        ), patch(
            "openlayer.lib.integrations.portkey_tracer.add_to_trace"
        ) as mock_add_to_trace, patch(
            "openlayer.lib.integrations.portkey_tracer.time.time", side_effect=[10.0, 10.2]
        ):
            result = handle_non_streaming_create(
                client,
                create_func=create_func,
                inference_id="trace-xyz",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert result is response
            mock_add_to_trace.assert_called_once()
            trace_kwargs = mock_add_to_trace.call_args.kwargs
            assert trace_kwargs["id"] == "trace-xyz"
            assert trace_kwargs["metadata"]["provider"] == "OpenAI"
            assert trace_kwargs["metadata"]["cost"] == 0.25
            assert trace_kwargs["metadata"]["portkeyHeaders"]["X-Portkey-Provider"] == "openai"


