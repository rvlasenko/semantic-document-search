from semantic_search.models.chunk import TextChunk
from semantic_search.models.document import Document


class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[TextChunk]:
        words = document.text.split()

        if not words:
            return []

        chunks = []
        start = 0

        while start < len(words):
            end = start + self._chunk_size
            chunk_words = words[start:end]
            chunks.append(
                TextChunk(
                    text=" ".join(chunk_words),
                    source=document.source,
                )
            )
            start += self._chunk_size - self._chunk_overlap

        return chunks
