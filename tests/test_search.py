import numpy as np

from semantic_search.models.chunk import TextChunk
from semantic_search.models.search_result import SearchResult
from semantic_search.search.search import search

CHUNKS = [
    TextChunk(
        text="Refund policy allows returns within 30 days.", source="data/refund.txt"
    ),
    TextChunk(
        text="Free shipping on orders over 50 dollars.", source="data/shipping.txt"
    ),
    TextChunk(text="Contact us at support@example.com.", source="data/contact.txt"),
]

# первый чанк похож на запрос, остальные нет
QUERY_EMBEDDING = np.array([1.0, 0.0, 0.0])
CHUNK_EMBEDDINGS = [
    np.array([1.0, 0.0, 0.0]),  # score: 1.0 — идентичен
    np.array([0.0, 1.0, 0.0]),  # score: 0.0 — не похож
    np.array([0.0, 0.0, 1.0]),  # score: 0.0 — не похож
]


def test_most_similar_chunk_is_first() -> None:
    results = search(QUERY_EMBEDDING, CHUNK_EMBEDDINGS, CHUNKS)
    assert results[0].text == CHUNKS[0].text


def test_top_k_limits_results() -> None:
    results = search(QUERY_EMBEDDING, CHUNK_EMBEDDINGS, CHUNKS, top_k=2)
    assert len(results) == 2


def test_empty_chunks_returns_empty_list() -> None:
    results = search(QUERY_EMBEDDING, [], [], top_k=3)
    assert results == []


def test_returns_search_result_instances() -> None:
    results = search(QUERY_EMBEDDING, CHUNK_EMBEDDINGS, CHUNKS)
    assert all(isinstance(r, SearchResult) for r in results)


def test_score_is_between_minus_one_and_one() -> None:
    results = search(QUERY_EMBEDDING, CHUNK_EMBEDDINGS, CHUNKS)
    assert all(-1.0 <= r.score <= 1.0 for r in results)


def test_result_contains_source() -> None:
    results = search(QUERY_EMBEDDING, CHUNK_EMBEDDINGS, CHUNKS)
    assert all(len(r.source) > 0 for r in results)
