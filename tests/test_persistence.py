from pathlib import Path

import numpy as np
import pytest
from unittest.mock import MagicMock

from semantic_search.chunker import TextChunker
from semantic_search.document import Document
from semantic_search.embedding_model import EmbeddingModel
from semantic_search.exceptions import IndexCorruptedError, IndexNotFoundError
from semantic_search.search_index import SearchIndex

DOCUMENTS = [
    Document(
        text="Refund policy allows returns within 30 days.", source="data/refund.txt"
    ),
    Document(
        text="Free shipping on orders over 50 dollars.", source="data/shipping.txt"
    ),
]


def make_index() -> SearchIndex:
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    mock_model = MagicMock(spec=EmbeddingModel)
    mock_model.encode.return_value = np.ones((384,))
    index = SearchIndex(chunker, mock_model)
    index.add_documents(DOCUMENTS)
    return index


def test_save_creates_chunks_json(tmp_path: Path) -> None:
    index = make_index()
    index.save(tmp_path / "index")
    assert (tmp_path / "index" / "chunks.json").exists()


def test_save_creates_embeddings_npy(tmp_path: Path) -> None:
    index = make_index()
    index.save(tmp_path / "index")
    assert (tmp_path / "index" / "embeddings.npy").exists()


def test_load_restores_chunks(tmp_path: Path) -> None:
    index = make_index()
    index.save(tmp_path / "index")

    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    mock_model = MagicMock(spec=EmbeddingModel)
    mock_model.encode.return_value = np.ones((384,))

    loaded = SearchIndex.load(tmp_path / "index", chunker, mock_model)
    assert len(loaded._chunks) == len(index._chunks)


def test_load_missing_directory_raises_error(tmp_path: Path) -> None:
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    mock_model = MagicMock(spec=EmbeddingModel)

    with pytest.raises(IndexNotFoundError):
        SearchIndex.load(tmp_path / "nonexistent", chunker, mock_model)


def test_load_incomplete_index_raises_error(tmp_path: Path) -> None:
    (tmp_path / "index").mkdir()
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    mock_model = MagicMock(spec=EmbeddingModel)

    with pytest.raises(IndexCorruptedError):
        SearchIndex.load(tmp_path / "index", chunker, mock_model)


def test_roundtrip_search_returns_results(tmp_path: Path) -> None:
    index = make_index()
    index.save(tmp_path / "index")

    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    mock_model = MagicMock(spec=EmbeddingModel)
    mock_model.encode.return_value = np.ones((384,))

    loaded = SearchIndex.load(tmp_path / "index", chunker, mock_model)
    results = loaded.search("refund policy")
    assert len(results) > 0
