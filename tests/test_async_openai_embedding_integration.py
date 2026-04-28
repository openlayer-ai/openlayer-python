"""Test OpenAI embedding integration (async)."""

import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch


class TestOpenAIAsyncEmbedding:
    def _fake_response(self, embeddings, prompt_tokens=4, model="text-embedding-3-small"):
        response = Mock()
        response.model = model
        response.data = [Mock(embedding=v) for v in embeddings]
        response.usage = Mock(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens)
        response.model_dump = Mock(return_value={"model": model})
        return response

    def test_handle_embedding_async_single_input(self) -> None:
        from openlayer.lib.integrations.async_openai_tracer import (
            handle_embedding_async,
        )

        fake = self._fake_response([[0.1, 0.2, 0.3]])
        original = AsyncMock(return_value=fake)

        with patch(
            "openlayer.lib.tracing.tracer.add_embedding_step_to_trace"
        ) as mock_add:
            result = asyncio.run(
                handle_embedding_async(
                    original_func=original,
                    model="text-embedding-3-small",
                    input="hello",
                )
            )

        assert result is fake
        kwargs = mock_add.call_args.kwargs
        assert kwargs["name"] == "OpenAI Embedding"
        assert kwargs["provider"] == "OpenAI"
        assert kwargs["output"] == [0.1, 0.2, 0.3]
        assert kwargs["embedding_dimensions"] == 3
        assert kwargs["embedding_count"] == 1

    def test_handle_embedding_async_batch_input(self) -> None:
        from openlayer.lib.integrations.async_openai_tracer import (
            handle_embedding_async,
        )

        fake = self._fake_response([[0.1, 0.2], [0.3, 0.4]], prompt_tokens=6)
        original = AsyncMock(return_value=fake)

        with patch(
            "openlayer.lib.tracing.tracer.add_embedding_step_to_trace"
        ) as mock_add:
            asyncio.run(
                handle_embedding_async(
                    original_func=original,
                    model="text-embedding-3-small",
                    input=["a", "b"],
                )
            )

        kwargs = mock_add.call_args.kwargs
        assert kwargs["inputs"] == {"input": ["a", "b"]}
        assert kwargs["output"] == [[0.1, 0.2], [0.3, 0.4]]
        assert kwargs["embedding_count"] == 2
        assert kwargs["prompt_tokens"] == 6

    def test_handle_embedding_async_failure_does_not_break_client(self) -> None:
        from openlayer.lib.integrations.async_openai_tracer import (
            handle_embedding_async,
        )

        fake = self._fake_response([[0.0]])
        original = AsyncMock(return_value=fake)

        with patch(
            "openlayer.lib.tracing.tracer.add_embedding_step_to_trace",
            side_effect=RuntimeError("backend down"),
        ):
            result = asyncio.run(
                handle_embedding_async(
                    original_func=original,
                    model="text-embedding-3-small",
                    input="x",
                )
            )

        assert result is fake

    def test_trace_async_openai_patches_embeddings_create(self) -> None:
        import openai

        from openlayer.lib.integrations.async_openai_tracer import trace_async_openai

        client = MagicMock(spec=openai.AsyncOpenAI)
        original_create = client.embeddings.create

        traced_client = trace_async_openai(client)

        assert traced_client.embeddings.create is not original_create
