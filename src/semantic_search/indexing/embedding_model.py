import numpy as np
from sentence_transformers import SentenceTransformer

from semantic_search.models.chunk import TextChunk


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def encode(self, chunk: TextChunk) -> np.ndarray:
        result = self._model.encode(chunk.text, convert_to_numpy=True)
        assert isinstance(result, np.ndarray)
        return result
