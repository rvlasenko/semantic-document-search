from pathlib import Path
import json

import numpy as np

from semantic_search.chunk import TextChunk
from semantic_search.chunker import TextChunker
from semantic_search.document import Document
from semantic_search.embedding_model import EmbeddingModel
from semantic_search.exceptions import IndexCorruptedError, IndexNotFoundError
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

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        chunks_data = [{"text": c.text, "source": c.source} for c in self._chunks]

        (path / "chunks.json").write_text(
            json.dumps(chunks_data, indent=2), encoding="utf-8"
        )

        np.save(path / "embeddings.npy", np.stack(self._embeddings))

    @classmethod
    def load(
        cls, path: Path, chunker: TextChunker, model: EmbeddingModel
    ) -> "SearchIndex":
        if not path.exists():
            raise IndexNotFoundError(f"Index directory not found {path}")

        chunks_path = path / "chunks.json"
        embeddings_path = path / "embeddings.npy"

        if not chunks_path.exists() or not embeddings_path.exists():
            raise IndexCorruptedError(f"Index is incomplete {path}")

        try:
            chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
            embeddings_matrix = np.load(embeddings_path)
        except Exception as e:
            raise IndexCorruptedError(f"Failed to load index: {e}") from e

        instance = cls(chunker, model)
        instance._chunks = [
            TextChunk(text=d["text"], source=d["source"]) for d in chunks_data
        ]
        instance._embeddings = [
            embeddings_matrix[i] for i in range(len(embeddings_matrix))
        ]

        return instance
