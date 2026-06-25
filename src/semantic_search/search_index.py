import numpy as np

from semantic_search.chunk import TextChunk
from semantic_search.chunker import TextChunker
from semantic_search.document import Document
from semantic_search.embedding_model import EmbeddingModel
from semantic_search.search import search
from semantic_search.search_result import SearchResult


class SearchIndex:
    def __init__(self, chunker: TextChunker, model: EmbeddingModel) -> None:
        self._chunker = chunker
        self._model = model
        self._chunks: list[TextChunk] = []
        self._embeddings: list[np.ndarray] = []

    def add_documents(self, documents: list[Document]) -> None:
        for document in documents:
            chunks = self._chunker.split(document)
            embeddings = [self._model.encode(chunk) for chunk in chunks]
            self._chunks.extend(chunks)
            self._embeddings.extend(embeddings)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not self._chunks:
            return []

        query_chunk = TextChunk(text=query, source="query")
        query_embedding = self._model.encode(query_chunk)

        return search(query_embedding, self._embeddings, self._chunks, top_k=top_k)
