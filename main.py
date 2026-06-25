from pathlib import Path

from semantic_search.document import Document
from semantic_search.loader import DocumentLoader
from semantic_search.chunker import TextChunker
from semantic_search.embedding_model import EmbeddingModel

loader = DocumentLoader(Path("data"))
chunker = TextChunker(chunk_size=50, chunk_overlap=20)
model = EmbeddingModel()

documents = loader.load()

chunks = []

for document in documents:
    chunks.extend(chunker.split(document))
print(f"Created {len(chunks)} chunks")

embedding = model.encode(chunks[0])
print(f"Embedding shape: {embedding.shape}")
print(f"First 5 values: {embedding[:5]}")
