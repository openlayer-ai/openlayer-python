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
