import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from semantic_search.models.chunk import TextChunk
from semantic_search.models.search_result import SearchResult


def search(
    query_embedding: np.ndarray,
    chunk_embeddings: list[np.ndarray],
    chunks: list[TextChunk],
    top_k: int = 3,
) -> list[SearchResult]:
    if not chunks:
        return []

    query = query_embedding.reshape(1, -1)
    matrix = np.stack(chunk_embeddings)
    scores = cosine_similarity(query, matrix)[0]

    indices = np.argsort(scores)[::-1][:top_k]

    return [
        SearchResult(
            text=chunks[i].text,
            source=chunks[i].source,
            score=float(scores[i]),
        )
        for i in indices
    ]
