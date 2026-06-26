from unittest.mock import MagicMock
from collections.abc import Generator

import numpy as np
import pytest

from semantic_search.indexing.chunker import TextChunker
from semantic_search.indexing.embedding_model import EmbeddingModel
from semantic_search.models.document import Document
from semantic_search.models.search_result import SearchResult
from semantic_search.search.search_index import SearchIndex

DOCUMENTS = [
    Document(
        text="Refund policy allows returns within 30 days.", source="data/refund.txt"
    ),
    Document(
        text="Free shipping on orders over 50 dollars.", source="data/shipping.txt"
    ),
]


@pytest.fixture
def index() -> Generator[SearchIndex, None, None]:
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)

    mock_model = MagicMock(spec=EmbeddingModel)
    mock_model.encode.return_value = np.ones((384,))

    yield SearchIndex(chunker, mock_model)


def test_add_documents_indexes_chunks(index: SearchIndex) -> None:
    index.add_documents(DOCUMENTS)
    assert len(index._chunks) > 0


def test_search_returns_search_results(index: SearchIndex) -> None:
    index.add_documents(DOCUMENTS)
    results = index.search("How long can I return an item?", top_k=2)
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_empty_index_returns_empty_list(index: SearchIndex) -> None:
    results = index.search("anything")
    assert results == []


def test_search_top_k_limits_results(index: SearchIndex) -> None:
    index.add_documents(DOCUMENTS)
    results = index.search("refund", top_k=1)
    assert len(results) == 1


def test_result_contains_source(index: SearchIndex) -> None:
    index.add_documents(DOCUMENTS)
    results = index.search("refund policy")
    assert all(len(r.source) > 0 for r in results)
