import numpy as np
import pytest
from unittest.mock import patch
from collections.abc import Generator

from semantic_search.indexing.embedding_model import EmbeddingModel
from semantic_search.models.chunk import TextChunk

CHUNK = TextChunk(
    text="Our refund policy allows returns within 30 days.", source="test/doc.txt"
)


@pytest.fixture
def mock_model() -> Generator[EmbeddingModel]:
    with patch("semantic_search.indexing.embedding_model.SentenceTransformer") as mock:
        mock.return_value.encode.return_value = np.ones((384,))
        yield EmbeddingModel()


def test_encode_returns_numpy_array(mock_model: EmbeddingModel) -> None:
    result = mock_model.encode(CHUNK)
    assert isinstance(result, np.ndarray)


def test_encode_returns_correct_shape(mock_model: EmbeddingModel) -> None:
    result = mock_model.encode(CHUNK)
    assert result.shape == (384,)


def test_encode_calls_model_with_chunk_text(mock_model: EmbeddingModel) -> None:
    mock_model.encode(CHUNK)
    mock_model._model.encode.assert_called_once_with(CHUNK.text, convert_to_numpy=True)
