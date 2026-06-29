import nltk

from semantic_search.models.chunk import TextChunk
from semantic_search.models.document import Document

nltk.download("punkt_tab", quiet=True)


class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[TextChunk]:
        sentences = nltk.sent_tokenize(document.text)

        if not sentences:
            return []

        chunks = []
        current: list[str] = []
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if current_words + sentence_words > self._chunk_size and current:
                chunks.append(self._make_chunk(current, document.source))
                overlap = self._take_overlap(current)
                current = overlap
                current_words = sum(len(s.split()) for s in current)

            current.append(sentence)
            current_words += sentence_words

        if current:
            chunks.append(self._make_chunk(current, document.source))

        return chunks

    def _make_chunk(self, sentences: list[str], source: str) -> TextChunk:
        return TextChunk(" ".join(sentences), source)

    def _take_overlap(self, sentences: list[str]) -> list[str]:
        overlap: list[str] = []
        word_count = 0

        for sentence in reversed(sentences):
            words = len(sentence.split())
            if word_count + words > self._chunk_overlap:
                break
            overlap.insert(0, sentence)
            word_count += words

        return overlap
