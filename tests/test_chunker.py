from semantic_search.indexing.chunker import TextChunker
from semantic_search.models.chunk import TextChunk
from semantic_search.models.document import Document


def make_document(text: str) -> Document:
    return Document(text=text, source="test/doc.txt")


def test_empty_text_returns_empty_list() -> None:
    chunker = TextChunker()
    assert chunker.split(make_document("")) == []


def test_short_text_returns_single_chunk() -> None:
    chunker = TextChunker(chunk_size=500)
    chunks = chunker.split(make_document("Hello world."))
    assert len(chunks) == 1


def test_long_text_returns_multiple_chunks() -> None:
    words = " ".join(["word"] * 20)  # word word word ...
    chunker = TextChunker(chunk_size=5, chunk_overlap=1)
    chunks = chunker.split(make_document(words))
    assert len(chunks) > 1


def test_chunk_doesnt_exceed_size() -> None:
    words = " ".join(["word"] * 100)
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.split(make_document(words))
    assert all(len(chunk.text.split()) <= 10 for chunk in chunks)


def test_chunk_inherits_source() -> None:
    chunker = TextChunker(chunk_size=5, chunk_overlap=1)
    words = " ".join(["word"] * 20)
    chunks = chunker.split(make_document(words))
    assert all(chunk.source == "test/doc.txt" for chunk in chunks)


def test_returns_text_chunk_instances() -> None:
    chunker = TextChunker()
    chunks = chunker.split(make_document("Hello world"))
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)
