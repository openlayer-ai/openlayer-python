"""Test AWS Bedrock integration."""

import io
import json
from unittest.mock import MagicMock, patch


class TestBedrockChatRegression:
    """Lock in existing chat-completion behaviour before refactoring."""

    def _make_anthropic_response(self, body_dict):
        """Build a response dict mimicking what bedrock-runtime returns."""
        from botocore.response import StreamingBody

        body_bytes = json.dumps(body_dict).encode("utf-8")
        return {"body": StreamingBody(io.BytesIO(body_bytes), len(body_bytes))}

    def test_anthropic_chat_invoke_routes_through_existing_handler(self) -> None:
        """invoke_model with a Claude model must hit the existing chat handler."""
        from openlayer.lib.integrations.bedrock_tracer import trace_bedrock

        mock_client = MagicMock()
        mock_client.invoke_model.return_value = self._make_anthropic_response(
            {
                "id": "msg_01",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "hello back"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )

        traced = trace_bedrock(mock_client)

        with patch("openlayer.lib.integrations.bedrock_tracer.add_to_trace") as mock_add:
            response = traced.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                ),
                contentType="application/json",
                accept="application/json",
            )

        # The chat path called add_to_trace (which wraps add_chat_completion_step_to_trace).
        mock_add.assert_called_once()
        kwargs = mock_add.call_args.kwargs
        assert kwargs["model"] == "anthropic.claude-3-haiku-20240307-v1:0"
        assert kwargs["output"] == "hello back"
        assert kwargs["prompt_tokens"] == 10
        assert kwargs["completion_tokens"] == 5

        # Caller can still consume the response body — critical regression guard.
        replayed = response["body"].read()
        assert b"hello back" in replayed


class TestBedrockEmbeddingDetection:
    """Detection routes embedding models away from the chat handler."""

    def test_is_embedding_model_titan(self) -> None:
        from openlayer.lib.integrations.bedrock_tracer import _is_embedding_model

        assert _is_embedding_model("amazon.titan-embed-text-v1") is True
        assert _is_embedding_model("amazon.titan-embed-text-v2:0") is True

    def test_is_embedding_model_cohere(self) -> None:
        from openlayer.lib.integrations.bedrock_tracer import _is_embedding_model

        assert _is_embedding_model("cohere.embed-english-v3") is True
        assert _is_embedding_model("cohere.embed-multilingual-v3") is True

    def test_is_embedding_model_chat_returns_false(self) -> None:
        from openlayer.lib.integrations.bedrock_tracer import _is_embedding_model

        assert _is_embedding_model("anthropic.claude-3-haiku-20240307-v1:0") is False
        assert _is_embedding_model("meta.llama3-70b-instruct-v1:0") is False

    def test_is_embedding_model_handles_empty_string(self) -> None:
        from openlayer.lib.integrations.bedrock_tracer import _is_embedding_model

        assert _is_embedding_model("") is False

    def test_traced_invoke_routes_embedding_to_new_handler(self) -> None:
        """An embedding modelId must call handle_embedding_invoke, not the chat handler."""
        from botocore.response import StreamingBody

        from openlayer.lib.integrations.bedrock_tracer import trace_bedrock

        body = json.dumps(
            {"embedding": [0.1, 0.2], "inputTextTokenCount": 4}
        ).encode("utf-8")
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {
            "body": StreamingBody(io.BytesIO(body), len(body))
        }
        traced = trace_bedrock(mock_client)

        with patch(
            "openlayer.lib.integrations.bedrock_tracer.handle_embedding_invoke"
        ) as mock_embed, patch(
            "openlayer.lib.integrations.bedrock_tracer.handle_non_streaming_invoke"
        ) as mock_chat:
            mock_embed.return_value = {"body": "ok"}
            traced.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({"inputText": "hi"}),
            )

        mock_embed.assert_called_once()
        mock_chat.assert_not_called()


class TestBedrockTitanEmbedding:
    """Titan v1 and v2 embedding requests produce well-formed embedding steps."""

    def _titan_response(self, embedding, token_count):
        from botocore.response import StreamingBody

        body = json.dumps(
            {"embedding": embedding, "inputTextTokenCount": token_count}
        ).encode("utf-8")
        return {"body": StreamingBody(io.BytesIO(body), len(body))}

    def test_titan_v2_single_embedding(self) -> None:
        from openlayer.lib.integrations.bedrock_tracer import trace_bedrock

        vec = [0.1, 0.2, 0.3, 0.4]
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = self._titan_response(vec, 7)
        traced = trace_bedrock(mock_client)

        with patch(
            "openlayer.lib.tracing.tracer.add_embedding_step_to_trace"
        ) as mock_add:
            response = traced.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps(
                    {"inputText": "hello world", "dimensions": 4, "normalize": True}
                ),
                contentType="application/json",
                accept="application/json",
            )

        mock_add.assert_called_once()
        kwargs = mock_add.call_args.kwargs
        assert kwargs["name"] == "AWS Bedrock Embedding"
        assert kwargs["model"] == "amazon.titan-embed-text-v2:0"
        assert kwargs["provider"] == "Bedrock"
        assert kwargs["inputs"] == {"input": "hello world"}
        assert kwargs["output"] == vec
        assert kwargs["embedding_dimensions"] == 4
        assert kwargs["embedding_count"] == 1
        assert kwargs["prompt_tokens"] == 7
        assert kwargs["tokens"] == 7
        assert kwargs["model_parameters"] == {
            "dimensions": 4,
            "normalize": True,
        }
        assert response["body"].read() == json.dumps(
            {"embedding": vec, "inputTextTokenCount": 7}
        ).encode("utf-8")

    def test_titan_v1_single_embedding(self) -> None:
        from openlayer.lib.integrations.bedrock_tracer import trace_bedrock

        vec = [0.5] * 1536
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = self._titan_response(vec, 12)
        traced = trace_bedrock(mock_client)

        with patch(
            "openlayer.lib.tracing.tracer.add_embedding_step_to_trace"
        ) as mock_add:
            traced.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                body=json.dumps({"inputText": "another"}),
            )

        kwargs = mock_add.call_args.kwargs
        assert kwargs["model"] == "amazon.titan-embed-text-v1"
        assert kwargs["embedding_dimensions"] == 1536
        assert kwargs["embedding_count"] == 1
        assert kwargs["prompt_tokens"] == 12
        # v1 has no `dimensions`/`normalize` params in its request body
        assert kwargs["model_parameters"] == {
            "dimensions": None,
            "normalize": None,
        }


class TestBedrockCohereEmbedding:
    """Cohere v3 embedding produces a multi-vector batch step."""

    def _cohere_response(self, embeddings):
        from botocore.response import StreamingBody

        body = json.dumps(
            {
                "embeddings": embeddings,
                "id": "abc-123",
                "response_type": "embeddings_floats",
            }
        ).encode("utf-8")
        return {"body": StreamingBody(io.BytesIO(body), len(body))}

    def test_cohere_embed_batch(self) -> None:
        from openlayer.lib.integrations.bedrock_tracer import trace_bedrock

        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = self._cohere_response(embeddings)
        traced = trace_bedrock(mock_client)

        with patch(
            "openlayer.lib.tracing.tracer.add_embedding_step_to_trace"
        ) as mock_add:
            traced.invoke_model(
                modelId="cohere.embed-english-v3",
                body=json.dumps(
                    {
                        "texts": ["one", "two", "three"],
                        "input_type": "search_document",
                    }
                ),
            )

        kwargs = mock_add.call_args.kwargs
        assert kwargs["model"] == "cohere.embed-english-v3"
        assert kwargs["inputs"] == {"input": ["one", "two", "three"]}
        assert kwargs["output"] == embeddings
        assert kwargs["embedding_dimensions"] == 3
        assert kwargs["embedding_count"] == 3
        # Cohere v3 does not return tokens in its response body.
        assert kwargs["prompt_tokens"] == 0
        assert kwargs["model_parameters"] == {
            "input_type": "search_document",
            "truncate": None,
            "embedding_types": None,
        }


class TestBedrockEmbeddingResilience:
    """Tracing failures must never break the caller; response body must remain usable."""

    def test_embedding_failure_does_not_break_client(self) -> None:
        from botocore.response import StreamingBody

        from openlayer.lib.integrations.bedrock_tracer import trace_bedrock

        body_bytes = json.dumps(
            {"embedding": [0.1, 0.2], "inputTextTokenCount": 4}
        ).encode("utf-8")
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {
            "body": StreamingBody(io.BytesIO(body_bytes), len(body_bytes))
        }
        traced = trace_bedrock(mock_client)

        with patch(
            "openlayer.lib.tracing.tracer.add_embedding_step_to_trace",
            side_effect=RuntimeError("backend down"),
        ):
            response = traced.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({"inputText": "hi"}),
            )

        # Caller still gets the real response.
        assert response["body"].read() == body_bytes

    def test_embedding_response_body_is_replayable(self) -> None:
        from botocore.response import StreamingBody

        from openlayer.lib.integrations.bedrock_tracer import trace_bedrock

        body_bytes = json.dumps(
            {"embedding": [0.9, 0.8, 0.7], "inputTextTokenCount": 3}
        ).encode("utf-8")
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {
            "body": StreamingBody(io.BytesIO(body_bytes), len(body_bytes))
        }
        traced = trace_bedrock(mock_client)

        with patch("openlayer.lib.tracing.tracer.add_embedding_step_to_trace"):
            response = traced.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({"inputText": "x"}),
            )

        # The body must be readable by the caller after tracing has consumed it.
        assert response["body"].read() == body_bytes
