"""Test OpenAI embedding integration (sync)."""

# openlayer.lib.integrations is in pyright's ignore list, so imports get
# unknown/partially unknown types; disable these diagnostics for this test file only.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false

from unittest.mock import Mock, MagicMock, patch


class TestOpenAISyncEmbedding:
    """Sync OpenAI client.embeddings.create must be traced."""

    def _fake_response(self, embeddings, prompt_tokens=4, model="text-embedding-3-small"):
        response = Mock()
        response.model = model
        response.data = [Mock(embedding=v) for v in embeddings]
        response.usage = Mock(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens)
        response.model_dump = Mock(return_value={"model": model})
        return response

    def test_handle_embedding_single_input(self) -> None:
        from openlayer.lib.integrations.openai_tracer import handle_embedding

        fake = self._fake_response([[0.1, 0.2, 0.3]])
        original = Mock(return_value=fake)

        with patch(
            "openlayer.lib.tracing.tracer.add_embedding_step_to_trace"
        ) as mock_add:
            result = handle_embedding(
                original_func=original,
                model="text-embedding-3-small",
                input="hello",
                inference_id="abc",
            )

        assert result is fake
        kwargs = mock_add.call_args.kwargs
        assert kwargs["name"] == "OpenAI Embedding"
        assert kwargs["provider"] == "OpenAI"
        assert kwargs["model"] == "text-embedding-3-small"
        assert kwargs["inputs"] == {"input": "hello"}
        assert kwargs["output"] == [0.1, 0.2, 0.3]
        assert kwargs["embedding_dimensions"] == 3
        assert kwargs["embedding_count"] == 1
        assert kwargs["prompt_tokens"] == 4
        assert kwargs["tokens"] == 4
        assert kwargs["id"] == "abc"

    def test_handle_embedding_batch_input(self) -> None:
        from openlayer.lib.integrations.openai_tracer import handle_embedding

        fake = self._fake_response(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], prompt_tokens=9
        )
        original = Mock(return_value=fake)

        with patch(
            "openlayer.lib.tracing.tracer.add_embedding_step_to_trace"
        ) as mock_add:
            handle_embedding(
                original_func=original,
                model="text-embedding-3-small",
                input=["one", "two", "three"],
            )

        kwargs = mock_add.call_args.kwargs
        assert kwargs["inputs"] == {"input": ["one", "two", "three"]}
        assert kwargs["output"] == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        assert kwargs["embedding_dimensions"] == 2
        assert kwargs["embedding_count"] == 3
        assert kwargs["prompt_tokens"] == 9

    def test_handle_embedding_failure_does_not_break_client(self) -> None:
        from openlayer.lib.integrations.openai_tracer import handle_embedding

        fake = self._fake_response([[0.0]])
        original = Mock(return_value=fake)

        with patch(
            "openlayer.lib.tracing.tracer.add_embedding_step_to_trace",
            side_effect=RuntimeError("backend down"),
        ):
            result = handle_embedding(
                original_func=original,
                model="text-embedding-3-small",
                input="x",
            )

        assert result is fake

    def test_trace_openai_patches_embeddings_create(self) -> None:
        """After trace_openai, client.embeddings.create is replaced."""
        import openai  # pyright: ignore[reportMissingImports]

        from openlayer.lib.integrations.openai_tracer import trace_openai

        # Make client appear like a real OpenAI (not Azure) client.
        client = MagicMock(spec=openai.OpenAI)
        original_create = client.embeddings.create

        traced_client = trace_openai(client)

        assert traced_client.embeddings.create is not original_create
