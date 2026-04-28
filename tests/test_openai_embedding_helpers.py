"""Test shared OpenAI embedding parsers."""

from unittest.mock import Mock


class TestParseEmbeddingResponse:
    def test_single_vector(self) -> None:
        from openlayer.lib.integrations._openai_embedding_common import (
            parse_embedding_response,
        )

        response = Mock()
        response.data = [Mock(embedding=[0.1, 0.2, 0.3, 0.4])]

        embeddings, dim, count = parse_embedding_response(response)
        assert embeddings == [0.1, 0.2, 0.3, 0.4]
        assert dim == 4
        assert count == 1

    def test_batch_vectors(self) -> None:
        from openlayer.lib.integrations._openai_embedding_common import (
            parse_embedding_response,
        )

        response = Mock()
        response.data = [
            Mock(embedding=[0.1, 0.2]),
            Mock(embedding=[0.3, 0.4]),
        ]

        embeddings, dim, count = parse_embedding_response(response)
        assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
        assert dim == 2
        assert count == 2

    def test_empty_data(self) -> None:
        from openlayer.lib.integrations._openai_embedding_common import (
            parse_embedding_response,
        )

        response = Mock()
        response.data = []

        embeddings, dim, count = parse_embedding_response(response)
        assert embeddings == []
        assert dim == 0
        assert count == 0

    def test_dict_data_items(self) -> None:
        """Some response shapes carry dict items instead of model objects."""
        from openlayer.lib.integrations._openai_embedding_common import (
            parse_embedding_response,
        )

        response = Mock()
        response.data = [{"embedding": [0.5, 0.6]}]

        embeddings, dim, count = parse_embedding_response(response)
        assert embeddings == [0.5, 0.6]
        assert dim == 2
        assert count == 1


class TestGetEmbeddingModelParameters:
    def test_extracts_relevant_params(self) -> None:
        from openlayer.lib.integrations._openai_embedding_common import (
            get_embedding_model_parameters,
        )

        params = get_embedding_model_parameters(
            {
                "dimensions": 1536,
                "encoding_format": "float",
                "user": "u1",
                "irrelevant": "ignored",
            }
        )
        assert params == {
            "dimensions": 1536,
            "encoding_format": "float",
            "user": "u1",
        }

    def test_missing_params_default_to_none(self) -> None:
        from openlayer.lib.integrations._openai_embedding_common import (
            get_embedding_model_parameters,
        )

        params = get_embedding_model_parameters({})
        assert params == {
            "dimensions": None,
            "encoding_format": None,
            "user": None,
        }
